import os

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

from sklearn.metrics import f1_score
from src.text_core import save_experiment_predictions, calc_scores


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float32).view(1)
        return item

    def __len__(self):
        return len(self.labels)


def predict_proba(model, data_loader, device):
    model.eval()  # Перевод в режим оценки (выключает Dropout)
    all_preds = []
    all_labels = []
    
    sigmoid = torch.nn.Sigmoid()
    with torch.no_grad():
        for batch in data_loader:
            # Перенос данных на устройство
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            
            preds = sigmoid(logits).cpu().numpy().flatten()
            preds = np.column_stack((1 - preds, preds))
           
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy().flatten())
    
    return np.array(all_preds), np.array(all_labels)


def create_dataloader(tokenizer, df, max_length=256, batch_size=16, shuffle=False):
    encodings = tokenizer(df['text'].values.tolist(),
                            truncation=True, padding=True, max_length=max_length,
                            return_tensors='pt')
    data_loader = DataLoader(
        TextDataset(encodings, df['label'].astype(int).values.tolist()),
        batch_size=batch_size, shuffle=shuffle)

    return data_loader


def save_nn_experiment_predictions(root_folder, name, model, datasets, device):

    def is_serializable(obj):
        try:
            json.dumps(obj)
            return True
        except (TypeError, OverflowError):
            return False

    # Проверка словаря
    # params = {}
    # for key, value in pipe.get_params().items():
    #     if is_serializable(value):
    #         params[key] = value

    # filepath = os.path.join(root_folder, f"{name}__params.json")
    # print(f"file {filepath} was saved")
    # with open(filepath, 'w') as f:
        # json.dump(params, f, indent=4)

    for dtype, data_loader, y_true in datasets:
        # X = df.drop('label', axis=1) if column is None else df[column] 
        pred, _ = predict_proba(model, data_loader, device)
        # print(pred[:, 1].shape, df['label'].values.shape)
        data = np.column_stack([pred[:, 1], y_true.values])
        # data = np.concatenate([pred[:, 1], df['label'].values], )
        filepath = os.path.join(root_folder, f"{name}__{dtype}.npy")
        print(f"file {filepath} was saved")
        np.save(filepath, data)


def train(model, data_train_loader, data_valid_loader,
          criterion, optimizer, device, num_epoch):

    model.to(device)

    model.train()

    for epoch in range(num_epoch):
        epoch_loss = 0.0
        for batch in data_train_loader:
            optimizer.zero_grad()
            
            # Перенос батча на GPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        pred_train, labels_train = predict_proba(model_1, data_train_loader, device) 
        pred_valid, labels_valid = predict_proba(model_1, data_valid_loader, device) 
        f1_train = f1_score(labels_train, (pred_train[:, 1] > 0.5).astype(int))
        f1_valid = f1_score(labels_valid, (pred_valid[:, 1] > 0.5).astype(int))

        mean_loss = epoch_loss / len(train_loader)
        print(f"epoch: {epoch} train: {f1_train:.4f}, valid: {f1_valid:.4f} loss: {mean_loss:.4f}")

    return model

train_df = pd.read_parquet("data/output/split_train.parquet")
val_df = pd.read_parquet("data/output/split_val.parquet")
test_df = pd.read_parquet("data/output/split_test.parquet")

print(train_df.shape, val_df.shape, test_df.shape)

# %%

# Define a simple classification layer on top of BERT
class ClassificationModel(torch.nn.Module):
    def __init__(self, bert_model):
        super(ClassificationModel, self).__init__()
        self.bert = bert_model
        self.fc = torch.nn.Linear(768, 1)  # 768 is the hidden size of BERT, 1

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.fc(pooled_output)
        return logits

    def get_params(self):
        return {}

model_name_1 = "michellejieli/emotion_text_classifier"
tokenizer_1 = AutoTokenizer.from_pretrained(model_name_1)
model_base_1 = AutoModel.from_pretrained(model_name_1)
# bert_model = BertModel.from_pretrained('bert-base-uncased')

# Freeze all layers
for param in model_base_1.parameters():
    param.requires_grad = False

model_1 = ClassificationModel(model_base_1)
optimizer_1 = torch.optim.SGD(model_1.parameters(), lr=0.005)
# optimizer = AdamW(model.parameters(), lr=2e-5)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

train_loader = create_dataloader(tokenizer_1, train_df)
val_loader = create_dataloader(tokenizer_1, val_df)
test_loader = create_dataloader(tokenizer_1, test_df)

list_datasets_1 = [
    ('train', train_loader, train_df['label']),
    ('val', val_loader, val_df['label']),
    ('test', test_loader, test_df['label']),
]

model_1 = train(model_1, train_loader, val_loader,
                criterion=torch.nn.BCEWithLogitsLoss(), 
                optimizer=optimizer_1, 
                device=device, 
                num_epoch=5)

save_nn_experiment_predictions(root_folder="data/experiments/exp_5/",
                               name="1", model=model_1, datasets=list_datasets_1, 
                               device=device)


results, results_mean = calc_scores(root_folder="data/experiments/exp_5/",
                                    plot_names=[])

top_names = results_mean.head(12).index.tolist()
print(results_mean.to_markdown(tablefmt="github"))
print()
print(results[results['name'].isin(top_names)].to_markdown(tablefmt="github"))

