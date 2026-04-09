#!/usr/bin/env python3
"""
Data-efficient Concept Bottleneck Model (DCBM) for role-based concepts with bounding boxes.

We implemented:
- Concept vocabulary from role names
- Concept instance extraction from role bounding boxes
- Shared image encoder for crops and full images
- Concept embeddings from mean crop embeddings per role
- Concept activations based on projection onto concept vectors
- Sparse linear CBM classifier with L1 regularization
- Inference with top concept contribution explanations
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from sklearn.metrics import precision_recall_fscore_support


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class CropAnnotation:
    image_key: str
    image_path: Path
    role_label: str
    bbox: Tuple[int, int, int, int]


@dataclass
class RoleEvalSample:
    image_key: str
    gold_roles: Set[str]


class VisionEncoder:
    """Pretrained vision encoder shared for crops and full images."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=self.weights)
        model.fc = nn.Identity()
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        self.model = model.to(self.device)
        self.preprocess = self.weights.transforms()

    @torch.no_grad()
    def encode_tensor_batch(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device, non_blocking=True)
        z = self.model(x)
        return z.detach().cpu()


class FullImageDataset(Dataset):
    def __init__(self, image_paths: Sequence[Path], preprocess) -> None:
        self.image_paths = list(image_paths)
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.image_paths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            return self.preprocess(img)


class CropDataset(Dataset):
    def __init__(self, annotations: Sequence[CropAnnotation], preprocess) -> None:
        self.annotations = list(annotations)
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> torch.Tensor:
        ann = self.annotations[idx]
        with Image.open(ann.image_path) as img:
            img = img.convert("RGB")
            w, h = img.size
            x1, y1, x2, y2 = ann.bbox
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            if x2 <= x1:
                x2 = min(w, x1 + 1)
            if y2 <= y1:
                y2 = min(h, y1 + 1)
            crop = img.crop((x1, y1, x2, y2))
            return self.preprocess(crop)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset(json_path: Path) -> Dict[str, dict]:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def index_images(image_dir: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for p in image_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            index.setdefault(p.stem, p)
    return index


def build_concept_vocab(dataset: Dict[str, dict]) -> List[str]:
    roles = set()
    for sample in dataset.values():
        role_map = sample.get("role", {})
        roles.update(role_map.keys())
    return sorted(roles)


def build_event_vocab(dataset: Dict[str, dict]) -> List[str]:
    events = {sample.get("event_type", "UNK") for sample in dataset.values()}
    return sorted(events)


def extract_crops(dataset: Dict[str, dict], image_index: Dict[str, Path]) -> List[CropAnnotation]:
    crops: List[CropAnnotation] = []
    for image_key, sample in dataset.items():
        img_path = image_index.get(image_key)
        if img_path is None:
            continue

        role_map = sample.get("role", {})
        for role_name, boxes in role_map.items():
            for box in boxes:
                if not isinstance(box, list) or len(box) < 5:
                    continue
                try:
                    x1, y1, x2, y2 = map(int, box[1:5])
                except (TypeError, ValueError):
                    continue
                crops.append(
                    CropAnnotation(
                        image_key=image_key,
                        image_path=img_path,
                        role_label=role_name,
                        bbox=(x1, y1, x2, y2),
                    )
                )
    return crops


def encode_images(
    encoder: VisionEncoder,
    image_paths: Sequence[Path],
    batch_size: int = 64,
    num_workers: int = 4,
) -> torch.Tensor:
    ds = FullImageDataset(image_paths=image_paths, preprocess=encoder.preprocess)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    chunks: List[torch.Tensor] = []
    for batch in dl:
        chunks.append(encoder.encode_tensor_batch(batch))
    return torch.cat(chunks, dim=0) if chunks else torch.empty(0)


def encode_crops(
    encoder: VisionEncoder,
    crops: Sequence[CropAnnotation],
    batch_size: int = 64,
    num_workers: int = 4,
) -> torch.Tensor:
    ds = CropDataset(annotations=crops, preprocess=encoder.preprocess)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    chunks: List[torch.Tensor] = []
    for batch in dl:
        chunks.append(encoder.encode_tensor_batch(batch))
    return torch.cat(chunks, dim=0) if chunks else torch.empty(0)


def build_concept_embeddings(
    crop_embeddings: torch.Tensor,
    crop_annotations: Sequence[CropAnnotation],
    concept_list: Sequence[str],
) -> torch.Tensor:
    concept_to_embs: Dict[str, List[torch.Tensor]] = defaultdict(list)
    for emb, ann in zip(crop_embeddings, crop_annotations):
        concept_to_embs[ann.role_label].append(emb)

    emb_dim = crop_embeddings.shape[1]
    concept_matrix = []
    for c in concept_list:
        if concept_to_embs[c]:
            concept_matrix.append(torch.stack(concept_to_embs[c], dim=0).mean(dim=0))
        else:
            concept_matrix.append(torch.zeros(emb_dim))
    return torch.stack(concept_matrix, dim=0)


def compute_activations(image_embeddings: torch.Tensor, concept_embeddings: torch.Tensor) -> torch.Tensor:
    concept_norm2 = (concept_embeddings * concept_embeddings).sum(dim=1).clamp_min(1e-12)
    return image_embeddings @ concept_embeddings.T / concept_norm2.unsqueeze(0)


def train_cbm(
    A: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    role_samples: Sequence[RoleEvalSample],
    concept_list: Sequence[str],
    eval_topk: int,
    epochs: int = 30,
    lr: float = 1e-2,
    l1_lambda: float = 1e-4,
    val_indices: torch.Tensor | None = None,
    train_indices: torch.Tensor | None = None,
    verbose: bool = False,
) -> Tuple[nn.Linear, List[dict], Dict[str, object], Dict[str, object]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    A = A.to(device)
    labels = labels.to(device)

    n_concepts = A.shape[1]
    model = nn.Linear(n_concepts, num_classes, bias=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    if train_indices is None:
        train_indices = torch.arange(A.shape[0], device=device)
    else:
        train_indices = train_indices.to(device)

    if val_indices is not None:
        val_indices = val_indices.to(device)

    history: List[dict] = []
    best_f1 = -1.0
    best_metrics_event: Dict[str, object] = {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "best_epoch": 0,
    }
    best_metrics_role: Dict[str, object] = {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "strict_match_rate": 0.0,
        "counts": {"tp": 0, "fp": 0, "fn": 0, "strict_correct": 0, "num_samples": 0},
        "best_epoch": 0,
    }
    best_state_dict = None

    for epoch in range(1, epochs + 1):
        model.train()
        logits = model(A[train_indices])
        ce_loss = criterion(logits, labels[train_indices])
        l1_penalty = model.weight.abs().sum()
        loss = ce_loss + l1_lambda * l1_penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            val_event_precision = None
            val_event_recall = None
            val_event_f1 = None
            val_role_precision = None
            val_role_recall = None
            val_role_f1 = None
            val_role_strict = None
            val_role_counts = None

            if val_indices is not None and len(val_indices) > 0:
                model.eval()
                A_val = A[val_indices]
                labels_val = labels[val_indices]
                role_val_samples = [role_samples[int(i)] for i in val_indices.tolist()]
                
                # Evaluate event predictions
                val_event_precision, val_event_recall, val_event_f1 = evaluate_event_prf(
                    activations=A_val,
                    model=model,
                    gold_event_labels=labels_val,
                )
                
                # Evaluate role predictions
                val_role_precision, val_role_recall, val_role_f1, val_role_strict, val_role_counts = evaluate_roles_prf(
                    activations=A_val,
                    model=model,
                    role_samples=role_val_samples,
                    concept_list=concept_list,
                    topk=eval_topk,
                )

                # Best model selected by role F1 (explanation quality)
                if val_role_f1 > best_f1:
                    best_f1 = val_role_f1
                    best_state_dict = copy.deepcopy(model.state_dict())
                    best_metrics_event = {
                        "precision": val_event_precision,
                        "recall": val_event_recall,
                        "f1": val_event_f1,
                        "best_epoch": epoch,
                    }
                    best_metrics_role = {
                        "precision": val_role_precision,
                        "recall": val_role_recall,
                        "f1": val_role_f1,
                        "strict_match_rate": val_role_strict,
                        "counts": val_role_counts,
                        "best_epoch": epoch,
                    }

        row = {
            "epoch": epoch,
            "loss": float(loss.item()),
            "val_event_precision": None if val_event_precision is None else float(val_event_precision),
            "val_event_recall": None if val_event_recall is None else float(val_event_recall),
            "val_event_f1": None if val_event_f1 is None else float(val_event_f1),
            "val_role_precision": None if val_role_precision is None else float(val_role_precision),
            "val_role_recall": None if val_role_recall is None else float(val_role_recall),
            "val_role_f1": None if val_role_f1 is None else float(val_role_f1),
            "val_role_strict_match_rate": None if val_role_strict is None else float(val_role_strict),
        }
        history.append(row)

        if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == epochs):
            if val_event_f1 is None:
                print(f"[epoch {epoch:03d}] loss={loss.item():.4f}")
            else:
                print(
                    f"[epoch {epoch:03d}] loss={loss.item():.4f} "
                    f"event_p={val_event_precision:.4f} event_r={val_event_recall:.4f} event_f1={val_event_f1:.4f} | "
                    f"role_p={val_role_precision:.4f} role_r={val_role_recall:.4f} role_f1={val_role_f1:.4f} role_strict={val_role_strict:.4f}"
                )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model.cpu(), history, best_metrics_event, best_metrics_role


def explain_prediction(
    activation_vector: torch.Tensor,
    weight_matrix: torch.Tensor,
    bias_vector: torch.Tensor,
    concept_list: Sequence[str],
    event_list: Sequence[str],
    topk: int = 5,
) -> Tuple[str, List[Tuple[str, float, float]]]:
    logits = activation_vector @ weight_matrix.T + bias_vector
    pred_idx = int(torch.argmax(logits).item())
    pred_label = event_list[pred_idx]

    impact = activation_vector * weight_matrix[pred_idx]
    topk = min(topk, len(concept_list))
    idx = torch.topk(impact, k=topk).indices.tolist()

    rows = []
    for j in idx:
        rows.append((concept_list[j], float(activation_vector[j].item()), float(impact[j].item())))
    return pred_label, rows


def prepare_training_samples(
    dataset: Dict[str, dict],
    image_index: Dict[str, Path],
    event_to_idx: Dict[str, int],
) -> Tuple[List[Path], torch.Tensor, List[str], List[RoleEvalSample]]:
    image_paths: List[Path] = []
    labels: List[int] = []
    image_keys: List[str] = []
    role_samples: List[RoleEvalSample] = []

    for image_key, sample in dataset.items():
        img_path = image_index.get(image_key)
        if img_path is None:
            continue
        event_type = sample.get("event_type", "UNK")
        if event_type not in event_to_idx:
            continue

        image_paths.append(img_path)
        labels.append(event_to_idx[event_type])
        image_keys.append(image_key)
        role_samples.append(
            RoleEvalSample(
                image_key=image_key,
                gold_roles=set(sample.get("role", {}).keys()),
            )
        )

    return image_paths, torch.tensor(labels, dtype=torch.long), image_keys, role_samples


def _topk_role_predictions(
    activations: torch.Tensor,
    model: nn.Linear,
    concept_list: Sequence[str],
    topk: int,
) -> List[Set[str]]:
    with torch.no_grad():
        logits = model(activations)
        pred_event_idx = torch.argmax(logits, dim=1)
        impact = activations * model.weight[pred_event_idx]

    topk = min(topk, len(concept_list))
    topk_idx = torch.topk(impact, k=topk, dim=1).indices
    pred_roles: List[Set[str]] = []
    for row in topk_idx:
        pred_roles.append({concept_list[j] for j in row.tolist()})
    return pred_roles


def evaluate_event_prf(
    activations: torch.Tensor,
    model: nn.Linear,
    gold_event_labels: torch.Tensor,
) -> Tuple[float, float, float]:
    """Evaluate event prediction metrics using macro-averaged P/R/F1."""
    with torch.no_grad():
        logits = model(activations)
        pred_event_idx = torch.argmax(logits, dim=1).cpu().numpy()

    precision, recall, f1, _ = precision_recall_fscore_support(
        gold_event_labels.cpu().numpy(),
        pred_event_idx,
        average="macro",
        zero_division=0,
    )
    return float(precision), float(recall), float(f1)


def evaluate_roles_prf(
    activations: torch.Tensor,
    model: nn.Linear,
    role_samples: Sequence[RoleEvalSample],
    concept_list: Sequence[str],
    topk: int,
) -> Tuple[float, float, float, float, Dict[str, int]]:
    pred_roles = _topk_role_predictions(activations, model, concept_list, topk=topk)

    tp = 0
    fp = 0
    fn = 0
    strict_correct = 0
    n = len(role_samples)

    for pred, sample in zip(pred_roles, role_samples):
        gold = sample.gold_roles
        if gold.issubset(pred):
            strict_correct += 1

        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    strict_match_rate = float(strict_correct / n) if n > 0 else 0.0

    counts = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "strict_correct": strict_correct,
        "num_samples": n,
    }
    return precision, recall, f1, strict_match_rate, counts


def max_roles_per_image(dataset: Dict[str, dict]) -> int:
    max_roles = 0
    for sample in dataset.values():
        role_map = sample.get("role", {})
        max_roles = max(max_roles, len(role_map.keys()))
    return max_roles


def kfold_split(n: int, k: int, seed: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Generate k-fold indices. Returns list of (train_idx, val_idx) tuples."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    fold_size = n // k
    folds = []
    
    for fold_i in range(k):
        start = fold_i * fold_size
        end = start + fold_size if fold_i < k - 1 else n
        val_idx = perm[start:end]
        train_idx = torch.cat([perm[:start], perm[end:]], dim=0) if (start > 0 or end < n) else perm
        folds.append((train_idx, val_idx))
    
    return folds


def run_train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    data_json = Path(args.dataset_json)
    image_dir = Path(args.image_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    dataset = load_dataset(data_json)
    print(f"Loaded {len(dataset)} samples from {data_json}")

    print("Indexing images...")
    image_index = index_images(image_dir)
    print(f"Indexed {len(image_index)} image files in {image_dir}")

    concept_list = build_concept_vocab(dataset)
    event_list = build_event_vocab(dataset)
    event_to_idx = {e: i for i, e in enumerate(event_list)}

    print(f"Discovered {len(concept_list)} concepts (roles)")
    print(f"Discovered {len(event_list)} event types")

    print("Extracting crop annotations from bounding boxes...")
    crop_annotations = extract_crops(dataset, image_index)
    print(f"Extracted {len(crop_annotations)} role-labeled crops")

    if len(crop_annotations) == 0:
        raise RuntimeError("No crop annotations found. Verify paths and JSON format.")

    image_paths, y, image_keys, role_samples = prepare_training_samples(dataset, image_index, event_to_idx)
    if len(image_paths) == 0:
        raise RuntimeError("No training images found with valid labels.")

    print(f"Using {len(image_paths)} full images for CBM training")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Encoder device: {device}")
    encoder = VisionEncoder(device=device)

    print("Encoding crop instances...")
    crop_embeddings = encode_crops(
        encoder=encoder,
        crops=crop_annotations,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print("Building concept embeddings (mean per role)...")
    concept_embeddings = build_concept_embeddings(
        crop_embeddings=crop_embeddings,
        crop_annotations=crop_annotations,
        concept_list=concept_list,
    )

    print("Encoding full images...")
    image_embeddings = encode_images(
        encoder=encoder,
        image_paths=image_paths,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print("Computing concept activations...")
    A = compute_activations(image_embeddings=image_embeddings, concept_embeddings=concept_embeddings)

    max_roles = max_roles_per_image(dataset)
    eval_topk = args.eval_topk if args.topk_strategy == "fixed" else max(args.eval_topk, max_roles)
    print(
        f"Role-eval top-k strategy: {args.topk_strategy} | "
        f"requested={args.eval_topk} | max_roles={max_roles} | using k={eval_topk}"
    )

    kfold = kfold_split(len(image_paths), k=3, seed=args.seed)
    print(f"Using 3-fold cross-validation on {len(image_paths)} samples")

    print("Training sparse linear CBM with 3-fold CV...")
    fold_event_metrics = []
    fold_role_metrics = []
    all_histories = []
    final_model = None
    best_role_f1 = -1.0
    best_fold_role_counts: Dict[str, int] | None = None

    for fold_i, (train_idx, val_idx) in enumerate(kfold):
        print(f"\n--- Fold {fold_i + 1}/3 ---")
        print(f"Train size: {len(train_idx)} | Val size: {len(val_idx)}")
        
        model, history, best_metrics_event, best_metrics_role = train_cbm(
            A=A,
            labels=y,
            num_classes=len(event_list),
            role_samples=role_samples,
            concept_list=concept_list,
            eval_topk=eval_topk,
            epochs=args.epochs,
            lr=args.lr,
            l1_lambda=args.l1_lambda,
            train_indices=train_idx,
            val_indices=val_idx,
            verbose=False,
        )

        # Extract event metrics
        event_p = float(best_metrics_event["precision"])
        event_r = float(best_metrics_event["recall"])
        event_f1 = float(best_metrics_event["f1"])
        event_best_epoch = int(best_metrics_event["best_epoch"])
        
        # Extract role metrics
        role_p = float(best_metrics_role["precision"])
        role_r = float(best_metrics_role["recall"])
        role_f1 = float(best_metrics_role["f1"])
        role_strict_match_rate = float(best_metrics_role["strict_match_rate"])
        role_counts = best_metrics_role["counts"]
        role_best_epoch = int(best_metrics_role["best_epoch"])

        fold_event_metrics.append(
            {
                "precision": event_p,
                "recall": event_r,
                "f1": event_f1,
            }
        )
        fold_role_metrics.append(
            {
                "precision": role_p,
                "recall": role_r,
                "f1": role_f1,
                "strict_match_rate": role_strict_match_rate,
                "counts": role_counts,
            }
        )
        all_histories.append(
            {
                "fold": fold_i + 1,
                "history": history,
                "event_precision": event_p,
                "event_recall": event_r,
                "event_f1": event_f1,
                "event_best_epoch": event_best_epoch,
                "role_precision": role_p,
                "role_recall": role_r,
                "role_f1": role_f1,
                "role_strict_match_rate": role_strict_match_rate,
                "role_counts": role_counts,
                "role_best_epoch": role_best_epoch,
            }
        )

        # Best model selected by role F1
        if role_f1 > best_role_f1:
            best_role_f1 = role_f1
            final_model = model
            best_fold_role_counts = role_counts

        print(
            f"Fold {fold_i + 1} Event Metrics -> "
            f"Precision: {event_p:.4f}, Recall: {event_r:.4f}, F1: {event_f1:.4f}"
        )
        print(
            f"Fold {fold_i + 1} Role Metrics -> "
            f"Precision: {role_p:.4f}, Recall: {role_r:.4f}, F1: {role_f1:.4f}, "
            f"StrictMatch: {role_strict_match_rate:.4f}"
        )

    mean_event_p = sum(m["precision"] for m in fold_event_metrics) / len(fold_event_metrics)
    mean_event_r = sum(m["recall"] for m in fold_event_metrics) / len(fold_event_metrics)
    mean_event_f1 = sum(m["f1"] for m in fold_event_metrics) / len(fold_event_metrics)
    
    mean_role_p = sum(m["precision"] for m in fold_role_metrics) / len(fold_role_metrics)
    mean_role_r = sum(m["recall"] for m in fold_role_metrics) / len(fold_role_metrics)
    mean_role_f1 = sum(m["f1"] for m in fold_role_metrics) / len(fold_role_metrics)
    mean_role_strict = sum(m["strict_match_rate"] for m in fold_role_metrics) / len(fold_role_metrics)

    print(f"\n=== FINAL RESULTS ===")
    print("\n--- Event Prediction Metrics (Main Task) ---")
    for i, m in enumerate(fold_event_metrics, start=1):
        print(f"Fold {i}: Precision={m['precision']:.4f}, Recall={m['recall']:.4f}, F1={m['f1']:.4f}")
    print(f"Mean Event Precision: {mean_event_p:.4f}")
    print(f"Mean Event Recall: {mean_event_r:.4f}")
    print(f"Mean Event F1-Score: {mean_event_f1:.4f}")
    
    print("\n--- Role Prediction Metrics (Explanation/Why) ---")
    for i, m in enumerate(fold_role_metrics, start=1):
        print(
            f"Fold {i}: Precision={m['precision']:.4f}, Recall={m['recall']:.4f}, "
            f"F1={m['f1']:.4f}, StrictMatch={m['strict_match_rate']:.4f}"
        )
    print(f"Mean Role Precision: {mean_role_p:.4f}")
    print(f"Mean Role Recall: {mean_role_r:.4f}")
    print(f"Mean Role F1-Score: {mean_role_f1:.4f}")
    print(f"Mean Role Strict-Match: {mean_role_strict:.4f}")

    ckpt_path = out_dir / "dcbm_checkpoint_kfold.pt"
    payload = {
        "concept_list": concept_list,
        "event_list": event_list,
        "concept_embeddings": concept_embeddings,
        "cbm_weight": final_model.weight.detach().cpu(),
        "cbm_bias": final_model.bias.detach().cpu(),
        "kfold_histories": all_histories,
        "mean_event_precision": float(mean_event_p),
        "mean_event_recall": float(mean_event_r),
        "mean_event_f1": float(mean_event_f1),
        "mean_role_precision": float(mean_role_p),
        "mean_role_recall": float(mean_role_r),
        "mean_role_f1": float(mean_role_f1),
        "mean_role_strict_match": float(mean_role_strict),
        "fold_event_metrics": fold_event_metrics,
        "fold_role_metrics": fold_role_metrics,
        "best_fold_role_counts": best_fold_role_counts,
        "eval_topk": eval_topk,
        "args": vars(args),
        "image_keys": image_keys,
    }
    torch.save(payload, ckpt_path)

    with (out_dir / "kfold_results.json").open("w", encoding="utf-8") as f:
        cv_results = {
            "k": 3,
            "fold_event_metrics": fold_event_metrics,
            "fold_role_metrics": fold_role_metrics,
            "mean_event_precision": float(mean_event_p),
            "mean_event_recall": float(mean_event_r),
            "mean_event_f1": float(mean_event_f1),
            "mean_role_precision": float(mean_role_p),
            "mean_role_recall": float(mean_role_r),
            "mean_role_f1": float(mean_role_f1),
            "mean_role_strict_match": float(mean_role_strict),
            "eval_topk": eval_topk,
            "num_samples": len(image_paths),
        }
        json.dump(cv_results, f, indent=2)

    print(f"Saved checkpoint to: {ckpt_path}")
    print(f"Saved CV results to: {out_dir / 'kfold_results.json'}")


def _resolve_infer_image(image_arg: str, image_dir: Path) -> Path:
    p = Path(image_arg)
    if p.exists() and p.is_file():
        return p

    image_index = index_images(image_dir)
    if image_arg in image_index:
        return image_index[image_arg]

    stem = Path(image_arg).stem
    if stem in image_index:
        return image_index[stem]

    raise FileNotFoundError(
        f"Could not find image '{image_arg}' as a path or key in directory: {image_dir}"
    )


def run_infer(args: argparse.Namespace) -> None:
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    concept_list: List[str] = ckpt["concept_list"]
    event_list: List[str] = ckpt["event_list"]
    concept_embeddings: torch.Tensor = ckpt["concept_embeddings"]
    weight: torch.Tensor = ckpt["cbm_weight"]
    bias: torch.Tensor = ckpt["cbm_bias"]

    image_path = _resolve_infer_image(args.image, Path(args.image_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = VisionEncoder(device=device)

    img_emb = encode_images(
        encoder=encoder,
        image_paths=[image_path],
        batch_size=1,
        num_workers=0,
    )

    a = compute_activations(image_embeddings=img_emb, concept_embeddings=concept_embeddings)[0]

    pred_label, explanation = explain_prediction(
        activation_vector=a,
        weight_matrix=weight,
        bias_vector=bias,
        concept_list=concept_list,
        event_list=event_list,
        topk=args.topk,
    )

    print(f"Explaining Image: {image_path.name}")
    print(f"Prediction: {pred_label}")
    print("------------------------------")
    print("Why? Because I saw:")
    for role_name, activation, impact in explanation:
        print(f" - ('{role_name.lower()}'): Activation {activation:.2f} (Impact: {impact:.2f})")


















def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DCBM for role-based concepts with bbox crops")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train DCBM from img_train.json")
    p_train.add_argument("--dataset-json", type=str, default="m2e2(2)/img_train.json")
    p_train.add_argument("--image-dir", type=str, default="m2e2(2)/image/image")
    p_train.add_argument("--output-dir", type=str, default="dcbm_artifacts")
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--batch-size", type=int, default=64)
    p_train.add_argument("--num-workers", type=int, default=4)
    p_train.add_argument("--lr", type=float, default=1e-2)
    p_train.add_argument("--l1-lambda", type=float, default=1e-4)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument(
        "--eval-topk",
        type=int,
        default=5,
        help="Top-k roles used for role-level PR/F1 evaluation.",
    )
    p_train.add_argument(
        "--topk-strategy",
        type=str,
        default="max-roles",
        choices=["fixed", "max-roles"],
        help="fixed: use eval-topk; max-roles: use max(eval-topk, max roles in dataset).",
    )

    p_infer = sub.add_parser("infer", help="Run inference + explanation on one image")
    p_infer.add_argument("--checkpoint", type=str, default="dcbm_artifacts/dcbm_checkpoint_kfold.pt")
    p_infer.add_argument("--image", type=str, required=True, help="Image path or image key")
    p_infer.add_argument("--image-dir", type=str, default="m2e2(2)/image/image")
    p_infer.add_argument("--topk", type=int, default=5)

    return parser













def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command == "train":
        run_train(args)
    elif args.command == "infer":
        run_infer(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
