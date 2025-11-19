import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support


class M2E2ImageArgumentDataset(Dataset):
    def __init__(self, json_path, image_root, preprocess, role2id):
        self.data = json.load(open(json_path))
        self.image_root = image_root
        self.preprocess = preprocess
        self.role2id = role2id

        self.samples = []

        for filename, info in self.data.items():

            #   "role": { "Vehicle": [ [id, x1, y1, x2, y2], ... ] }
            role_dict = info.get("role", {})

            for role, regions in role_dict.items():
                label = self.role2id[role]

                for region in regions:
                    # region = ["1", x1, y1, x2, y2]
                    _, x1, y1, x2, y2 = region

                    self.samples.append({
                        "filename": filename,
                        "bbox": (x1, y1, x2, y2),
                        "label": label
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        img_path = f"{self.image_root}/{item['filename']}.jpg"
        image = Image.open(img_path).convert("RGB")

        x1, y1, x2, y2 = item["bbox"]
        crop = image.crop((x1, y1, x2, y2))

        pixel_values = self.preprocess(
            crop,
            return_tensors="pt"
        )["pixel_values"][0]

        return {
            "image": pixel_values,
            "label": torch.tensor(item["label"], dtype=torch.long)
        }


# Model: CLIP Vision + Linear classifier
class ImageArgumentClassifier(nn.Module):
    def __init__(self, num_labels, model_name="openai/clip-vit-base-patch32"):
        super().__init__()

        self.clip = CLIPModel.from_pretrained(model_name, use_safetensors=True)
        self.vision = self.clip.vision_model

        hidden = self.clip.config.vision_config.hidden_size
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, images, labels=None):
        vision_outputs = self.vision(images)
        pooled = vision_outputs.pooler_output

        logits = self.classifier(pooled)

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss
        else:
            return logits


# --------------------------------------------------------
# Training / Evaluation
# --------------------------------------------------------

def train(model, train_loader, val_loader, optimizer, device, epochs=10):
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            loss = model(images, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Train Loss: {avg:.4f}")

        precision, recall, f1 = evaluate(model, val_loader, device)
        print(f"Val Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    return precision, recall, f1



########

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # all possible argument roles from your dataset
    argument_roles = [
        "Agent",
        "Victim",
        "Instrument",
        "Place",
        "Artifact",
        "Person",
        "Giver",
        "Recipient",
        "Money",
        "Attacker",
        "Target",
        "Demonstrator",
        "Police",
        "Vehicle",
        "Entity"
    ]

    role2id = {r: i for i, r in enumerate(argument_roles)}

    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name, use_safetensors=True)

    train_set = M2E2ImageArgumentDataset(
        "data/m2e2/img_train.json",
        "data/m2e2/image/image",
        processor.image_processor,
        role2id
    )

    val_set = M2E2ImageArgumentDataset(
        "data/m2e2/img_val.json",
        "data/m2e2/image/image",
        processor.image_processor,
        role2id
    )

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False)

    model = ImageArgumentClassifier(
        num_labels=len(argument_roles),
        model_name=model_name
    ).to(device)

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=5e-6)

    train(model, train_loader, val_loader, optimizer, device, epochs=30)

    torch.save(model.state_dict(), "models/t4_argument_extraction_model.pt")
    print("Saved to models/t4_argument_extraction_model.pt")


if __name__ == "__main__":
    main()
