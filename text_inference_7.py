import os

import polars as pl
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel

from sklearn.metrics import f1_score


def read_split_dataset(filepath):
   raw_df = pl.read_csv(filepath, has_header=False,
                        separator="\n", new_columns=["raw_line"])
   raw_df = raw_df.select(
                       pl.col("raw_line").str.extract_groups(r"^([^,]+),([0,1]),(.*)$")
                       ).unnest("raw_line")
   raw_df.columns = ['file', 'label', 'text']
   return raw_df.to_pandas()


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # Токенизируем одну строку
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }


def model_inference(text_list, tokenizer, model,
                    device, max_length, batch_size=32):

    dataset = TextDataset(text_list, tokenizer, max_length=max_length)
    data_loader = DataLoader(dataset, batch_size=batch_size)

    model.to(device)
    all_preds = []
    
    sigmoid = torch.nn.Sigmoid()

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            logits = model(input_ids, attention_mask)
            preds = sigmoid(logits).cpu().numpy().flatten()
           
            all_preds.extend(preds)
    
    return np.array(all_preds)


# Define a simple classification layer on top of BERT
class ClassificationModel(torch.nn.Module):
    def __init__(self, bert_model):
        super(ClassificationModel, self).__init__()
        self.bert = bert_model
        self.fc = torch.nn.Linear(768, 128)
        self.dropout = torch.nn.Dropout(0.2)
        self.fc2 = torch.nn.Linear(128, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.fc(pooled_output)
        logits = self.dropout(logits)
        logits = self.fc2(logits)

        return logits

    def get_params(self):
        return {}


model_name_1 = "michellejieli/emotion_text_classifier"
tokenizer_1 = AutoTokenizer.from_pretrained(model_name_1)
model_base_1 = AutoModel.from_pretrained(model_name_1)
         
model_3 = ClassificationModel(model_base_1)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# load pre-train model
filepath = os.path.join("models/", "7__exp_5_model_4.pt")
# filepath = "./data/experiments/exp_5/4__model.pt"
model_3.load_state_dict(torch.load(filepath, weights_only=True))
model_3.eval()

# Inference
# for list with strings
text_list = ["hello world", "test", "test", "hello world"]
prediction = model_inference(text_list, tokenizer_1, model_3, device, max_length=256)
print(prediction)
# >>> [0.42449608 0.29468128 0.29468128 0.42449608]

# for datasets
for name, filepath in [
    ("train", "data/input/split/train.txt"),
    ("valid", "data/input/split/val.txt"),
    ("test", "data/input/split/test.txt"),
]:
    df = read_split_dataset(filepath)

    text_list = df['text'].values.tolist()
    true_labels = df['label'].astype(int).values

    pred_proba = model_inference(text_list, tokenizer_1, model_3, device, max_length=256)
    pred_labels = (pred_proba > 0.5).astype(int)
    score = f1_score(y_true=true_labels, y_pred=pred_labels, average='macro')
    score = round(score, 4)

    print(f"{name:5s} f1 score: {score}")


# 7 & Text (full) & Transformer(Emotion-english-distilroberta-base) & Fune-tuning & 68.54 & \textbf{71.49} & \textbf{70.02} & \\
# train f1 score: 0.7337
# valid f1 score: 0.6854
# test  f1 score: 0.7149
