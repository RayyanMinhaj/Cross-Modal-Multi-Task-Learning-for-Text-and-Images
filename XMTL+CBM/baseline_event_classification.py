#!/usr/bin/env python3
"""
Event-only image classifier.

This baseline predicts event_type from full-image embeddings only.
We have intentionally removeed any role explanation pipeline.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from torchvision import models


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class VisionEncoder:
    """Frozen pretrained image encoder ResNet50 for full images."""

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
        with Image.open(self.image_paths[idx]) as img:
            img = img.convert("RGB")
            return self.preprocess(img)






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


def build_event_vocab(dataset: Dict[str, dict]) -> List[str]:
    events = {sample.get("event_type", "UNK") for sample in dataset.values()}
    return sorted(events)


def prepare_samples(
    dataset: Dict[str, dict],
    image_index: Dict[str, Path],
    event_to_idx: Dict[str, int],
) -> Tuple[List[Path], torch.Tensor, List[str]]:
    image_paths: List[Path] = []
    labels: List[int] = []
    image_keys: List[str] = []

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

    return image_paths, torch.tensor(labels, dtype=torch.long), image_keys





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




def evaluate_event_prf(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float, float]:
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    gold = labels.cpu().numpy()
    p, r, f1, _ = precision_recall_fscore_support(gold, preds, average="macro", zero_division=0)
    return float(p), float(r), float(f1)






def train_event_classifier(
    X: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    train_indices: torch.Tensor,
    val_indices: torch.Tensor,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> Tuple[nn.Linear, List[dict], Dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    y = y.to(device)
    train_indices = train_indices.to(device)
    val_indices = val_indices.to(device)

    model = nn.Linear(X.shape[1], num_classes, bias=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    history: List[dict] = []
    best_f1 = -1.0
    best_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "best_epoch": 0}
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_logits = model(X[train_indices])
        loss = criterion(train_logits, y[train_indices])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            val_logits = model(X[val_indices])
            p, r, f1 = evaluate_event_prf(val_logits, y[val_indices])

        history.append(
            {
                "epoch": epoch,
                "loss": float(loss.item()),
                "val_precision": p,
                "val_recall": r,
                "val_f1": f1,
            }
        )

        if f1 > best_f1:
            best_f1 = f1
            best_metrics = {"precision": p, "recall": r, "f1": f1, "best_epoch": epoch}
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    return model.cpu(), history, best_metrics







def kfold_split(n: int, k: int, seed: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
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

    raise FileNotFoundError(f"Could not find image '{image_arg}' as a path or key in directory: {image_dir}")











def run_train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    dataset = load_dataset(Path(args.dataset_json))
    image_index = index_images(Path(args.image_dir))

    event_list = build_event_vocab(dataset)
    event_to_idx = {e: i for i, e in enumerate(event_list)}
    image_paths, y, image_keys = prepare_samples(dataset, image_index, event_to_idx)

    if len(image_paths) == 0:
        raise RuntimeError("No training images found with valid labels.")

    print(f"Loaded {len(image_paths)} usable images across {len(event_list)} event classes")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = VisionEncoder(device=device)

    print("Encoding full images...")
    X = encode_images(encoder, image_paths, batch_size=args.batch_size, num_workers=args.num_workers)

    folds = kfold_split(len(image_paths), k=3, seed=args.seed)
    fold_metrics = []
    all_histories = []

    best_global_f1 = -1.0
    best_model = None

    for fold_i, (train_idx, val_idx) in enumerate(folds, start=1):
        print(f"\n===== Fold {fold_i}/3 =====")
        print(f"Train size: {len(train_idx)} | Val size: {len(val_idx)}")

        model, history, metrics = train_event_classifier(
            X,
            y,
            num_classes=len(event_list),
            train_indices=train_idx,
            val_indices=val_idx,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        fold_metrics.append(metrics)
        all_histories.append({"fold": fold_i, "history": history, "best_metrics": metrics})

        print(
            f"Fold {fold_i} Best -> "
            f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}"
        )

        if metrics["f1"] > best_global_f1:
            best_global_f1 = metrics["f1"]
            best_model = model

    mean_p = sum(m["precision"] for m in fold_metrics) / len(fold_metrics)
    mean_r = sum(m["recall"] for m in fold_metrics) / len(fold_metrics)
    mean_f1 = sum(m["f1"] for m in fold_metrics) / len(fold_metrics)

    print("\n===== FINAL 3-FOLD RESULTS =====")
    for i, m in enumerate(fold_metrics, start=1):
        print(f"Fold {i}: Precision={m['precision']:.4f}, Recall={m['recall']:.4f}, F1={m['f1']:.4f}")
    print(f"Mean Precision: {mean_p:.4f}")
    print(f"Mean Recall: {mean_r:.4f}")
    print(f"Mean F1-Score: {mean_f1:.4f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "baseline_event_checkpoint_kfold.pt"

    payload = {
        "event_list": event_list,
        "classifier_weight": best_model.weight.detach().cpu(),
        "classifier_bias": best_model.bias.detach().cpu(),
        "fold_metrics": fold_metrics,
        "kfold_histories": all_histories,
        "mean_precision": mean_p,
        "mean_recall": mean_r,
        "mean_f1": mean_f1,
        "args": vars(args),
        "image_keys": image_keys,
    }
    torch.save(payload, ckpt_path)
    print(f"Saved checkpoint to: {ckpt_path}")


def run_infer(args: argparse.Namespace) -> None:
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    event_list: List[str] = ckpt["event_list"]
    weight: torch.Tensor = ckpt["classifier_weight"]
    bias: torch.Tensor = ckpt["classifier_bias"]

    image_path = _resolve_infer_image(args.image, Path(args.image_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = VisionEncoder(device=device)

    X = encode_images(encoder, [image_path], batch_size=1, num_workers=0)
    logits = X @ weight.T + bias
    pred_idx = int(torch.argmax(logits[0]).item())

    print(f"Image: {image_path.name}")
    print(f"Predicted event: {event_list[pred_idx]}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Baseline event-only classifier (full image -> event_type)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train event-only baseline with 3-fold CV")
    p_train.add_argument("--dataset-json", type=str, default="m2e2(2)/img_train.json")
    p_train.add_argument("--image-dir", type=str, default="m2e2(2)/image/image")
    p_train.add_argument("--output-dir", type=str, default="models")
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--batch-size", type=int, default=64)
    p_train.add_argument("--num-workers", type=int, default=4)
    p_train.add_argument("--lr", type=float, default=1e-2)
    p_train.add_argument("--weight-decay", type=float, default=1e-4)
    p_train.add_argument("--seed", type=int, default=42)

    p_infer = sub.add_parser("infer", help="Run inference on one image")
    p_infer.add_argument("--checkpoint", type=str, default="models/baseline_event_checkpoint_kfold.pt")
    p_infer.add_argument("--image", type=str, required=True, help="Image path or image key")
    p_infer.add_argument("--image-dir", type=str, default="m2e2(2)/image/image")

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
