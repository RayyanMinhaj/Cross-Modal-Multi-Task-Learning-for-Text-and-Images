import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm # Import tqdm
from sklearn.metrics import precision_recall_fscore_support # Import for P, R, F1

class M2E2ImageEventDataset(Dataset):
    def __init__(self, json_path, image_root, preprocess, label2id):
        self.data = json.load(open(json_path))
        self.image_root = image_root
        self.preprocess = preprocess
        self.label2id = label2id

        self.samples = []

        for filename, info in self.data.items():
            evt_type = info["event_type"]
            label = self.label2id[evt_type]

            self.samples.append({
                "filename": filename,
                "label": label
            })
    

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # Note: I noticed you had ".jpg" hardcoded in your previous file.
        # Your initial file did not. I've kept it here as it was in the last version you shared.
        img_path = f"{self.image_root}/{item['filename']}.jpg"
        image = Image.open(img_path).convert("RGB")

        pixel_values = self.preprocess(
            image,
            return_tensors="pt"
        )["pixel_values"][0]

        return {
            "image": pixel_values,
            "label": torch.tensor(item["label"], dtype=torch.long)
        }



# clip vision encoding + the classfication head
class ImageEventClassifier(nn.Module):
    def __init__(self, num_labels=8, model_name="openai/clip-vit-base-patch32"):

        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name, use_safetensors=True)
        
        self.vision = self.clip.vision_model
        hidden = self.clip.config.vision_config.hidden_size
        

        # we are going to freeze the parameters in the vision model - we dont need to retrain the clip model, just the small classifier layer at the end.
        #for param in self.vision.parameters():
        #    param.requires_grad = False
        
        
        self.classifier = nn.Linear(hidden, num_labels)



    def forward(self, images, labels=None):
        vision_outputs = self.vision(images)
        pooled = vision_outputs.pooler_output  # (batch, hidden)

        logits = self.classifier(pooled)

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss
        else:
            return logits


def train(model, train_loader, val_loader, optimizer, device, epochs=10):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Use tqdm for a progress bar
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            loss = model(images, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Train Loss: {avg_loss:.4f}")

        # Get P, R, and F1 from the evaluate function
        precision, recall, f1 = evaluate(model, val_loader, device)
        print(f"Val Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

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

    # Calculate precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    return precision, recall, f1


#########

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"


    event_types = [
        "Life:Die",
        "Movement:Transport",
        "Transaction:Transfer-Money",
        "Conflict:Attack",
        "Conflict:Demonstrate",
        "Contact:Meet",
        "Contact:Phone-Write",
        "Justice:Arrest-Jail"
    ]

    label2id = {e:i for i,e in enumerate(event_types)}

    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name, use_safetensors=True)

    train_set = M2E2ImageEventDataset("data/m2e2/img_train.json", "data/m2e2/image/image", processor.image_processor, label2id)
    val_set = M2E2ImageEventDataset("data/m2e2/img_val.json", "data/m2e2/image/image", processor.image_processor, label2id)

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False)

    model = ImageEventClassifier(num_labels=len(event_types)).to(device)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=5e-6)

    train(model, train_loader, val_loader, optimizer, device, epochs=20)

    torch.save(model.state_dict(), "models/t3_verb_classification_model.pt")
    print("Model saved to models/t3_verb_classification_model.pt")


if __name__ == "__main__":
    main()
