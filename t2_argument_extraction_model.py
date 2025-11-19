import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertModel
from sklearn.metrics import precision_recall_fscore_support
from torch.optim import AdamW
from tqdm import tqdm


def insert_argument_prompt(tokens, start, end):
    # this is where we will insert $argument$ around the span
    new_tokens = tokens.copy()
    new_tokens.insert(end + 1, "$/ARG$")
    new_tokens.insert(start, "$ARG$")

    return new_tokens


class M2E2ArgumentDataset(Dataset):
    def __init__(self, json_path, tokenizer, role2id, max_len=128):
        data = json.load(open(json_path))
        self.tokenizer = tokenizer
        self.role2id = role2id
        self.max_len = max_len
        self.samples = []


        for item in data:
            if not item["golden-event-mentions"]:
                continue


            words = item["words"]
            for event in item["golden-event-mentions"]:
                for arg in event["arguments"]:
                    arg_start = arg["start"]
                    arg_end = arg["end"]
                    role = arg["role"]

                    # insert prompt tokens
                    prompted_tokens = insert_argument_prompt(words, arg_start, arg_end)

                    enc = self.tokenizer(
                        prompted_tokens,
                        is_split_into_words=True,
                        return_tensors="pt",
                        padding='max_length',
                        truncation=True,
                        max_length=self.max_len
                    )


                    self.samples.append({
                        "input_ids": enc["input_ids"][0],
                        "attention_mask": enc["attention_mask"][0],
                        "label": self.role2id[role]
                    })



    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



# Argument extraction model
class ArgumentExtractionModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_labels=10):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        hidden = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden, num_labels)

    
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        hidden = outputs.last_hidden_state  # [CLS] token representation
        
        pooled = (hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)

        logits = self.classifier(self.dropout(pooled))
        
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss
        else:
            return torch.argmax(logits, dim=1)



def train_model(train_loader, val_loader, model, optimizer, device, num_epochs=10):

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            loss = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Training Loss: {avg_loss:.4f}")

        precision, recall, f1 = eval_model(val_loader, model, device)
        print(f"Val Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")




def eval_model(loader, model, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            preds = model(input_ids, attention_mask)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    return precision, recall, f1


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # We need to build a role2id mapping by scanning train/val/test first
    all_roles = set()
    for path in ["data/m2e2/train.json", "data/m2e2/val.json", "data/m2e2/test.json"]:
        data = json.load(open(path))
        for item in data:
            for event in item["golden-event-mentions"]:
                for arg in event["arguments"]:
                    all_roles.add(arg["role"])
    

    role2id = {role: idx for idx, role in enumerate(sorted(all_roles))}
    print("Role to ID mapping:", role2id)


    #load datasets
    train = M2E2ArgumentDataset("data/m2e2/train.json", tokenizer, role2id)
    val = M2E2ArgumentDataset("data/m2e2/val.json", tokenizer, role2id)

    train_loader = DataLoader(train, batch_size=8, shuffle=True)
    val_loader = DataLoader(val, batch_size=8, shuffle=False)

    model = ArgumentExtractionModel(bert_model_name="bert-base-uncased", num_labels=len(role2id)).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4)

    train_model(train_loader, val_loader, model, optimizer, device, num_epochs=10)

    torch.save(model.state_dict(), "models/t2_argument_extraction_model.pt")

    print("Model saved to models/t2_argument_extraction_model.pt")


if __name__ == "__main__":
    main()
