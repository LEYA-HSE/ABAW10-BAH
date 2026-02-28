import polars as pl
import pandas as pd
import numpy as np

import argparse
from pathlib import Path
import pickle

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel

from sklearn.metrics import f1_score


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inp", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("-with_score", action='store_true', required=False)
    return p.parse_args()


def read_split_dataset(filepath):
   raw_df = pl.read_csv(filepath, has_header=False,
                        separator="\n", new_columns=["raw_line"])
   raw_df = raw_df.select(
                       pl.col("raw_line").str.extract_groups(r"^([^,]+),([0,1]),(.*)$")
                       ).unnest("raw_line")
   raw_df.columns = ['file', 'label', 'text']
   return raw_df.to_pandas()


class TextDataset(Dataset):
    def __init__(self, videos, texts, tokenizer, max_length):
        self.videos = videos
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
            'video_name': self.videos[idx],
        }


def model_inference(video_list, text_list, tokenizer, model, device, max_length):

    dataset = TextDataset(video_list, text_list, tokenizer, max_length=max_length)
    data_loader = DataLoader(dataset, batch_size=1)

    model.to(device)
    
    sigmoid = torch.nn.Sigmoid()

    with torch.no_grad():
        out = {}
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            rel_key = batch['video_name'][0]
            
            logits, embeddings = model(input_ids, attention_mask)
            logits = logits.squeeze(0)
            embeddings = embeddings.squeeze(0)
            prob = sigmoid(logits)

            out[rel_key] = {
                "prob": prob.detach().cpu().numpy().astype("float32"),
                "logits": logits.detach().cpu().numpy().astype("float32"),
                "embeddings": embeddings.detach().cpu().numpy().astype("float32"),
            }
           
    return out


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
        emb = self.dropout(logits)
        logits = self.fc2(emb)

        return logits, emb

    def get_params(self):
        return {}


def main():
    args = parse_args()

    model_name_2 = "j-hartmann/emotion-english-distilroberta-base"
    tokenizer_2 = AutoTokenizer.from_pretrained(model_name_2)
    model_base_2 = AutoModel.from_pretrained(model_name_2)

    model_4 = ClassificationModel(model_base_2)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load pre-train model
    filepath = str(Path("./models/7__exp_5_model_4.pt")) 
    model_4.load_state_dict(torch.load(filepath, weights_only=True))
    model_4.eval()

    # Inference
    df = read_split_dataset(args.inp)

    video_list = df['file'].values.tolist()
    text_list = df['text'].values.tolist()

    out_dict = model_inference(video_list, text_list,
                               tokenizer_2, model_4,
                               device=device, max_length=256)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"{args.out} file was saved")
    with args.out.open("wb") as f:
        pickle.dump(out_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    if args.with_score:
        pred_prob = np.array([out_dict[x]['prob'][0] for x in video_list])
        pred_labels = (pred_prob > 0.5).astype(int)
        score = f1_score(df['label'].astype(int).values.tolist(), pred_labels, average='macro')
        print(f"f1 score: {score:.4f}")


if __name__ == "__main__":
    main()


# 7 & Text (full) & Transformer(Emotion-english-distilroberta-base) & Fune-tuning & 68.54 & \textbf{71.49} & \textbf{70.02} & \\
# train f1 score: 0.7337
# valid f1 score: 0.6854
# test  f1 score: 0.7149
