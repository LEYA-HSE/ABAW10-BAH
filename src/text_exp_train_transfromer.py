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
                          truncation=True, padding=True,
                          max_length=max_length,
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

    # save model
    filepath = os.path.join(root_folder, f"{name}__model.pt")
    torch.save(model.state_dict(), filepath)

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
        pred, y_tmp = predict_proba(model, data_loader, device)
        # print(pred, y_tmp, y_true.astype(int).values)
        data = np.column_stack([pred[:, 1], y_true.astype(int).values])
        # data = np.concatenate([pred[:, 1], df['label'].values], )
        filepath = os.path.join(root_folder, f"{name}__{dtype}.npy")
        print(f"file {filepath} was saved")
        np.save(filepath, data)


def train(model, data_train_loader, data_valid_loader,
          criterion, optimizer, device, num_epoch):

    model.to(device)

    for epoch in range(1, num_epoch + 1):
        model.train()
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

        pred_train, labels_train = predict_proba(model, data_train_loader, device) 
        pred_valid, labels_valid = predict_proba(model, data_valid_loader, device) 
        f1_train = f1_score(labels_train.astype(int), (pred_train[:, 1] > 0.5).astype(int), average='macro')
        f1_valid = f1_score(labels_valid.astype(int), (pred_valid[:, 1] > 0.5).astype(int), average='macro')

        mean_loss = epoch_loss / len(train_loader)
        print(f"epoch: {epoch} train: {f1_train:.4f}, valid: {f1_valid:.4f} loss: {mean_loss:.4f}")

    return model

train_df = pd.read_parquet("data/output/split_train.parquet")
val_df = pd.read_parquet("data/output/split_val.parquet")
test_df = pd.read_parquet("data/output/split_test.parquet")


def word_stats(df, words_list):
    all_names = list(words_list)

    df['len'] = df['text'].map(len)
    all_names.append('len')

    for x in words_list:
        name = f"first_{x}"
        df[name] = df['text'].map(lambda y: y.find(x))
        all_names.append(name)

        name = f"last_{x}"
        df[name] = df['text'].map(lambda y: y.rfind(x))
        all_names.append(name)

        df[x] = df['text'].map(lambda y: x in y).astype(int)

    return df, all_names

# words_list = ['but', 'been', 'stop', 'usually', 'love',
              # 'my', 'want', 'much', 'up', 'off']
words_list = ['but']
pattern = '|'.join(words_list)

train_wo_df = train_df.copy(True)
train_wo_df['text'] = train_wo_df['text'].str.replace(pattern, "", regex=True)

val_wo_df = val_df.copy(True)
val_wo_df['text'] = val_wo_df['text'].str.replace(pattern, "", regex=True)

test_wo_df = test_df.copy(True)
test_wo_df['text'] = test_wo_df['text'].str.replace(pattern, "", regex=True)

train_wo_df, all_names = word_stats(train_wo_df, words_list)
print(train_wo_df[all_names].describe().round(2).T)
print(train_wo_df[words_list + ['label']].corr().round(2))

print(train_wo_df.shape, val_wo_df.shape, test_wo_df.shape)

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
                num_epoch=20)

save_nn_experiment_predictions(root_folder="data/experiments/exp_5/",
                               name="1", model=model_1, datasets=list_datasets_1, 
                               device=device)

# %%

# Define a simple classification layer on top of BERT
class ClassificationModel(torch.nn.Module):
    def __init__(self, bert_model):
        super(ClassificationModel, self).__init__()
        self.bert = bert_model
        self.fc = torch.nn.Linear(768, 128)
        self.fc2 = torch.nn.Linear(128, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.fc(pooled_output)
        logits = self.fc2(logits)

        return logits

    def get_params(self):
        return {}

model_name_1 = "michellejieli/emotion_text_classifier"
tokenizer_1 = AutoTokenizer.from_pretrained(model_name_1)
model_base_1 = AutoModel.from_pretrained(model_name_1)

# Freeze all layers
for param in model_base_1.parameters():
    param.requires_grad = False

model_2 = ClassificationModel(model_base_1)
# optimizer_2 = torch.optim.SGD(model_2.parameters(), lr=0.01)
optimizer_2 = torch.optim.AdamW(model_2.parameters(), lr=0.01)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

train_loader = create_dataloader(tokenizer_1, train_df)
val_loader = create_dataloader(tokenizer_1, val_df)
test_loader = create_dataloader(tokenizer_1, test_df)

list_datasets_1 = [
    ('train', train_loader, train_df['label']),
    ('val', val_loader, val_df['label']),
    ('test', test_loader, test_df['label']),
]

model_2 = train(model_2, train_loader, val_loader,
                criterion=torch.nn.BCEWithLogitsLoss(), 
                optimizer=optimizer_2, 
                device=device, 
                num_epoch=3)

save_nn_experiment_predictions(root_folder="data/experiments/exp_5/",
                               name="2", model=model_2,
                               datasets=list_datasets_1, device=device)


# %%

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

# Freeze all layers
for param in model_base_1.parameters():
    param.requires_grad = False

for layer in list(model_base_1.children())[-2:]:
    for param in layer.parameters():
        param.requires_grad = True
        
model_3 = ClassificationModel(model_base_1)
optimizer_3 = torch.optim.SGD(model_3.parameters(), lr=0.001)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

train_loader = create_dataloader(tokenizer_1, train_df)
val_loader = create_dataloader(tokenizer_1, val_df)
test_loader = create_dataloader(tokenizer_1, test_df)

list_datasets_1 = [
    ('train', train_loader, train_df['label']),
    ('val', val_loader, val_df['label']),
    ('test', test_loader, test_df['label']),
]

model_3 = train(model_3, train_loader, val_loader,
                criterion=torch.nn.BCEWithLogitsLoss(), 
                optimizer=optimizer_3, 
                device=device, 
                num_epoch=9)

save_nn_experiment_predictions(root_folder="data/experiments/exp_5/",
                               name="3", model=model_3,
                               datasets=list_datasets_1, device=device)


# %%

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

model_name_2 = "j-hartmann/emotion-english-distilroberta-base"
tokenizer_2 = AutoTokenizer.from_pretrained(model_name_2)
model_base_2 = AutoModel.from_pretrained(model_name_2)

# Freeze all layers
for param in model_base_2.parameters():
    param.requires_grad = False

for layer in list(model_base_2.children())[-2:]:
    for param in layer.parameters():
        param.requires_grad = True
        
model_4 = ClassificationModel(model_base_2)
optimizer_4 = torch.optim.SGD(model_4.parameters(), lr=0.001)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

train_loader = create_dataloader(tokenizer_2, train_df)
val_loader = create_dataloader(tokenizer_2, val_df)
test_loader = create_dataloader(tokenizer_2, test_df)

list_datasets_2 = [
    ('train', train_loader, train_df['label']),
    ('val', val_loader, val_df['label']),
    ('test', test_loader, test_df['label']),
]

model_4 = train(model_4, train_loader, val_loader,
                criterion=torch.nn.BCEWithLogitsLoss(), 
                optimizer=optimizer_4, 
                device=device, 
                num_epoch=10)

save_nn_experiment_predictions(root_folder="data/experiments/exp_5/",
                               name="4", model=model_4,
                               datasets=list_datasets_2, device=device)

# %%

# Define a simple classification layer on top of BERT
class ClassificationModel(torch.nn.Module):
    def __init__(self, bert_model):
        super(ClassificationModel, self).__init__()
        self.bert = bert_model
        self.fc = torch.nn.Linear(768, 128)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc2 = torch.nn.Linear(128, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.last_hidden_state[:, 0, :]
        logits = self.dropout(logits)
        logits = self.fc(logits)
        logits = self.fc2(logits)

        return logits

    def get_params(self):
        return {}

model_name_1 = "michellejieli/emotion_text_classifier"
tokenizer_1 = AutoTokenizer.from_pretrained(model_name_1)
model_base_1 = AutoModel.from_pretrained(model_name_1)

# Freeze all layers
for param in model_base_1.parameters():
    param.requires_grad = False

for layer in list(model_base_1.children())[-2:]:
    for param in layer.parameters():
        param.requires_grad = True
        
model_5 = ClassificationModel(model_base_1)
optimizer_5 = torch.optim.SGD(model_5.parameters(), lr=0.001)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

train_loader = create_dataloader(tokenizer_1, train_df)
val_loader = create_dataloader(tokenizer_1, val_df)
test_loader = create_dataloader(tokenizer_1, test_df)

list_datasets_1 = [
    ('train', train_loader, train_df['label']),
    ('val', val_loader, val_df['label']),
    ('test', test_loader, test_df['label']),
]

model_5 = train(model_5, train_loader, val_loader,
                criterion=torch.nn.BCEWithLogitsLoss(), 
                optimizer=optimizer_5, 
                device=device, 
                num_epoch=20)

save_nn_experiment_predictions(root_folder="data/experiments/exp_5/",
                               name="5", model=model_5,
                               datasets=list_datasets_1, device=device)

# %%

# Define a simple classification layer on top of BERT
class ClassificationModel(torch.nn.Module):
    def __init__(self, bert_model):
        super(ClassificationModel, self).__init__()
        self.bert = bert_model
        self.fc = torch.nn.Linear(768, 128)
        self.dropout = torch.nn.Dropout(0.1)
        self.fc2 = torch.nn.Linear(128, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.last_hidden_state[:, 0, :]
        logits = self.dropout(logits)
        logits = self.fc(logits)
        logits = self.fc2(logits)

        return logits

    def get_params(self):
        return {}

model_name_1 = "michellejieli/emotion_text_classifier"
tokenizer_1 = AutoTokenizer.from_pretrained(model_name_1)
model_base_1 = AutoModel.from_pretrained(model_name_1)

# Freeze all layers
for param in model_base_1.parameters():
    param.requires_grad = False

for layer in list(model_base_1.children())[-3:]:
    for param in layer.parameters():
        param.requires_grad = True
        
model_6 = ClassificationModel(model_base_1)
optimizer_6 = torch.optim.SGD(model_6.parameters(), lr=0.0005)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

train_loader = create_dataloader(tokenizer_1, train_df)
val_loader = create_dataloader(tokenizer_1, val_df)
test_loader = create_dataloader(tokenizer_1, test_df)

list_datasets_1 = [
    ('train', train_loader, train_df['label']),
    ('val', val_loader, val_df['label']),
    ('test', test_loader, test_df['label']),
]

model_6 = train(model_6, train_loader, val_loader,
                criterion=torch.nn.BCEWithLogitsLoss(), 
                optimizer=optimizer_6, 
                device=device, 
                num_epoch=20)

save_nn_experiment_predictions(root_folder="data/experiments/exp_5/",
                               name="6", model=model_6,
                               datasets=list_datasets_1, device=device)

# %%

# Define a simple classification layer on top of BERT
class ClassificationModel(torch.nn.Module):
    def __init__(self, bert_model):
        super(ClassificationModel, self).__init__()
        self.bert = bert_model
        # self.fc = torch.nn.Linear(768, 128)
        # self.dropout = torch.nn.Dropout(0.1)
        self.fc2 = torch.nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.last_hidden_state[:, 0, :]
        # logits = self.dropout(logits)
        # logits = self.fc(logits)
        logits = self.fc2(logits)

        return logits

    def get_params(self):
        return {}

model_name_1 = "michellejieli/emotion_text_classifier"
tokenizer_1 = AutoTokenizer.from_pretrained(model_name_1)
model_base_1 = AutoModel.from_pretrained(model_name_1)

# Freeze all layers
for param in model_base_1.parameters():
    param.requires_grad = False

for layer in list(model_base_1.children())[-3:]:
    for param in layer.parameters():
        param.requires_grad = True
        
model_7 = ClassificationModel(model_base_1)
optimizer_7 = torch.optim.SGD(model_7.parameters(), lr=0.0005)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

train_loader = create_dataloader(tokenizer_1, train_df)
val_loader = create_dataloader(tokenizer_1, val_df)
test_loader = create_dataloader(tokenizer_1, test_df)

list_datasets_1 = [
    ('train', train_loader, train_df['label']),
    ('val', val_loader, val_df['label']),
    ('test', test_loader, test_df['label']),
]

model_7 = train(model_7, train_loader, val_loader,
                criterion=torch.nn.BCEWithLogitsLoss(), 
                optimizer=optimizer_7, 
                device=device, 
                num_epoch=15)

save_nn_experiment_predictions(root_folder="data/experiments/exp_5/",
                               name="7", model=model_7,
                               datasets=list_datasets_1, device=device)

# %%

# Define a simple classification layer on top of BERT
class ClassificationModel(torch.nn.Module):
    def __init__(self, bert_model):
        super(ClassificationModel, self).__init__()
        self.bert = bert_model
        self.fc = torch.nn.Linear(768, 64)
        self.dropout = torch.nn.Dropout(0.2)
        self.fc2 = torch.nn.Linear(64, 1)

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

# Freeze all layers
for param in model_base_1.parameters():
    param.requires_grad = False

for layer in list(model_base_1.children())[-2:]:
    for param in layer.parameters():
        param.requires_grad = True
        
model_8 = ClassificationModel(model_base_1)
# optimizer_8 = torch.optim.SGD(model_8.parameters(), lr=0.001)
optimizer_8 = torch.optim.AdamW(model_8.parameters(), lr=1e-5)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

train_loader = create_dataloader(tokenizer_1, train_df, max_length=1024)
val_loader = create_dataloader(tokenizer_1, val_df, max_length=1024)
test_loader = create_dataloader(tokenizer_1, test_df, max_length=1024)

list_datasets_1 = [
    ('train', train_loader, train_df['label']),
    ('val', val_loader, val_df['label']),
    ('test', test_loader, test_df['label']),
]

model_8 = train(model_8, train_loader, val_loader,
                criterion=torch.nn.BCEWithLogitsLoss(), 
                optimizer=optimizer_8, 
                device=device, 
                num_epoch=4)

save_nn_experiment_predictions(root_folder="data/experiments/exp_5/",
                               name="8", model=model_8,
                               datasets=list_datasets_1, device=device)

# %%

# Define a simple classification layer on top of BERT
class ClassificationModel(torch.nn.Module):
    def __init__(self, bert_model):
        super(ClassificationModel, self).__init__()
        self.bert = bert_model
        self.fc = torch.nn.Linear(768, 64)
        self.dropout = torch.nn.Dropout(0.1)
        self.fc2 = torch.nn.Linear(64, 1)

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

# Freeze all layers
for param in model_base_1.parameters():
    param.requires_grad = False

for layer in list(model_base_1.children())[-2:]:
    for param in layer.parameters():
        param.requires_grad = True
        
model_9 = ClassificationModel(model_base_1)
optimizer_9 = torch.optim.AdamW(model_9.parameters(), lr=1e-5)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

train_loader = create_dataloader(tokenizer_1, train_df, max_length=1024)
val_loader = create_dataloader(tokenizer_1, val_df, max_length=1024)
test_loader = create_dataloader(tokenizer_1, test_df, max_length=1024)

list_datasets_1 = [
    ('train', train_loader, train_df['label']),
    ('val', val_loader, val_df['label']),
    ('test', test_loader, test_df['label']),
]

model_9 = train(model_9, train_loader, val_loader,
                criterion=torch.nn.BCEWithLogitsLoss(), 
                optimizer=optimizer_9, 
                device=device, 
                num_epoch=5)

save_nn_experiment_predictions(root_folder="data/experiments/exp_5/",
                               name="9", model=model_9,
                               datasets=list_datasets_1, device=device)

# %%

# Define a simple classification layer on top of BERT
class ClassificationModel(torch.nn.Module):
    def __init__(self, bert_model):
        super(ClassificationModel, self).__init__()
        self.bert = bert_model
        self.fc = torch.nn.Linear(768, 64)
        self.dropout = torch.nn.Dropout(0.2)
        self.fc2 = torch.nn.Linear(64, 1)

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

# Freeze all layers
for param in model_base_1.parameters():
    param.requires_grad = False

for layer in list(model_base_1.children())[-2:]:
    for param in layer.parameters():
        param.requires_grad = True
        
model_10 = ClassificationModel(model_base_1)
optimizer_10 = torch.optim.AdamW(model_10.parameters(), lr=1e-5)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

train_loader = create_dataloader(tokenizer_1, train_df, max_length=1400)
val_loader = create_dataloader(tokenizer_1, val_df, max_length=1400)
test_loader = create_dataloader(tokenizer_1, test_df, max_length=1400)

list_datasets_1 = [
    ('train', train_loader, train_df['label']),
    ('val', val_loader, val_df['label']),
    ('test', test_loader, test_df['label']),
]

model_10 = train(model_10, train_loader, val_loader,
                criterion=torch.nn.BCEWithLogitsLoss(), 
                optimizer=optimizer_10, 
                device=device, 
                num_epoch=10)

save_nn_experiment_predictions(root_folder="data/experiments/exp_5/",
                               name="10", model=model_10,
                               datasets=list_datasets_1, device=device)
# %%

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

# Freeze all layers
for param in model_base_1.parameters():
    param.requires_grad = False

for layer in list(model_base_1.children())[-2:]:
    for param in layer.parameters():
        param.requires_grad = True
        
model_11 = ClassificationModel(model_base_1)
optimizer_11 = torch.optim.SGD(model_11.parameters(), lr=0.001)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# one word `but` was removed
train_loader = create_dataloader(tokenizer_1, train_wo_df)
val_loader = create_dataloader(tokenizer_1, val_wo_df)
test_loader = create_dataloader(tokenizer_1, test_wo_df)

list_datasets_11 = [
    ('train', train_loader, train_wo_df['label']),
    ('val', val_loader, val_wo_df['label']),
    ('test', test_loader, test_wo_df['label']),
]

model_11 = train(model_11, train_loader, val_loader,
                criterion=torch.nn.BCEWithLogitsLoss(), 
                optimizer=optimizer_11, 
                device=device, 
                num_epoch=10)

save_nn_experiment_predictions(root_folder="data/experiments/exp_5/",
                               name="11", model=model_11,
                               datasets=list_datasets_11, device=device)

# %%

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

# Freeze all layers
for param in model_base_1.parameters():
    param.requires_grad = False

for layer in list(model_base_1.children())[-2:]:
    for param in layer.parameters():
        param.requires_grad = True
        
model_12 = ClassificationModel(model_base_1)
optimizer_12 = torch.optim.SGD(model_12.parameters(), lr=0.001)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

train_loader = create_dataloader(tokenizer_1, train_wo_df)
val_loader = create_dataloader(tokenizer_1, val_wo_df)
test_loader = create_dataloader(tokenizer_1, test_wo_df)

list_datasets_12 = [
    ('train', train_loader, train_wo_df['label']),
    ('val', val_loader, val_wo_df['label']),
    ('test', test_loader, test_wo_df['label']),
]

model_12 = train(model_12, train_loader, val_loader,
                criterion=torch.nn.BCEWithLogitsLoss(), 
                optimizer=optimizer_12, 
                device=device, 
                num_epoch=20)

save_nn_experiment_predictions(root_folder="data/experiments/exp_5/",
                               name="12", model=model_12,
                               datasets=list_datasets_12, device=device)


results, results_mean = calc_scores(root_folder="data/experiments/exp_5/",
                                    plot_names=['3', '4'])

top_names = results_mean.head(12).index.tolist()
print(results_mean.to_markdown(tablefmt="github"))
print()
print(results[results['name'].isin(top_names)].to_markdown(tablefmt="github"))

