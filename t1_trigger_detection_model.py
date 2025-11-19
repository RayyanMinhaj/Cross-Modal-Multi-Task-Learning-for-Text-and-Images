import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertModel
from torch.optim import AdamW
from torchcrf import CRF
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm


class M2E2TriggerDataset(Dataset):
    # just to load the sentences and trigger spans from m2e2 json
    # converst to tokenized inputs and BIO labels

    def __init__(self, json_path, tokenizer, label2id = None, max_length=128):
        self.data = [d for d in json.load(open(json_path)) if d["golden-event-mentions"]]
        self.tokenizer = tokenizer
        self.label2id = label2id or {"O":0, "B-TRIGGER":1, "I-TRIGGER":2}
        self.max_length = max_length

        self.samples = self._prepare_samples()
    

    def _prepare_samples(self):
        samples = []

        for item in self.data:
            words = item["words"]
            sentence = " ".join(words)
            labels = ["O"] * len(words)

            for event in item["golden-event-mentions"]:
                t_start, t_end = event["trigger"]["start"], event["trigger"]["end"]
                labels[t_start] = "B-TRIGGER"

                for i in range(t_start + 1, t_end+1):
                    labels[i] = "I-TRIGGER"
            
            enc = self.tokenizer(
                words,
                is_split_into_words=True,
                return_tensors="pt",
                padding='max_length',
                truncation=True,
                max_length=self.max_length
            )

            label_ids = []
            word_ids = enc.word_ids(batch_index=0)
            previous_word_idx = None

            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(0)  # Special token

                elif word_idx != previous_word_idx:
                    label_ids.append(self.label2id[labels[word_idx]])
                
                else:
                    label_ids.append(self.label2id[labels[word_idx]])
                
                previous_word_idx = word_idx


            samples.append({
                "input_ids": enc["input_ids"][0],
                "attention_mask": enc["attention_mask"][0],
                "labels": torch.tensor(label_ids)
            })

        return samples            

    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    




# this is our actual Trigger detection model

class TriggerDetectionModel(nn.Module):

    # this init does the BERT -> Linear -> CRF
    def __init__(self, bert_model_name='bert-base-uncased', num_labels=10):
        super().__init__()
        
        self.bert = BertModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size #we already know this is 768
        
        self. dropout = nn.Dropout(0.1)
        
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        self.crf = CRF(num_labels, batch_first=True)


    def forward(self, input_ids, attention_mask, labels=None):

        outputs = self.bert(input_ids = input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        emissions = self.classifier(sequence_output)

        if labels is not None:
            # Compute the negative log-likelihood loss
            loss = -self.crf(emissions, labels, mask=attention_mask.bool(), reduction='mean')
            return loss

        else:
            pred_tags = self.crf.decode(emissions, mask=attention_mask.bool())
            return pred_tags



def train_trigger_model(train_loader, val_loader, model, optimizer, id2label, device, epochs=10):
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            loss = model(input_ids, attention_mask, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Training Loss: {total_loss/len(train_loader):.4f}") 

        #evaluate after each epoch
        precision, recall, f1 = eval_model(val_loader, model, id2label, device)
        print(f"Epoch {epoch+1} → Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")   


def eval_model(data_loader, model, id2label, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            preds = model(input_ids, attention_mask)

            # Flatten valid tokens only
            for i, seq in enumerate(preds):
                true_labels = [l.item() for l, m in zip(labels[i], batch["attention_mask"][i]) if m == 1 and l != -100]
                pred_labels = seq[:len(true_labels)]
                all_labels.extend(true_labels)
                all_preds.extend(pred_labels)

    # Compute precision, recall, f1 (macro-averaged)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
    return precision, recall, f1       






##################

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    label2id = {"O":0, "B-TRIGGER":1, "I-TRIGGER":2}
    id2label = {v:k for k,v in label2id.items()}

    print("Loading datasets...")

    train_dataset = M2E2TriggerDataset("data/m2e2/train.json", tokenizer, label2id)
    val_dataset = M2E2TriggerDataset("data/m2e2/val.json", tokenizer, label2id)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model = TriggerDetectionModel(bert_model_name=model_name, num_labels=len(label2id))
    optimizer = AdamW(model.parameters(), lr=1e-4)

    train_trigger_model(train_loader, val_loader, model, optimizer, id2label, device, epochs=10)

    torch.save(model.state_dict(), "models/trigger_detection_model.pt")
    print("Model saved to models/trigger_detection_model.pt")



if __name__ == '__main__': 
    main()
