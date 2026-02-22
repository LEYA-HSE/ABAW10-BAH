import os
import pandas as pd
import numpy as np
import tqdm

import pickle

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier

from catboost import CatBoostClassifier

from pulearn.bagging import BaggingPuClassifier
from src.text_core import save_experiment_predictions, calc_scores


train_df = pd.read_parquet("data/output/split_train.parquet")
val_df = pd.read_parquet("data/output/split_val.parquet")
test_df = pd.read_parquet("data/output/split_test.parquet")
list_datasets = [
    ('train', train_df),
    ('val', val_df),
    ('test', test_df),
]
print(train_df.shape, val_df.shape, test_df.shape)


def apply_func_to_datasets(list_datasets, func):
    list_output = []
    for name, df in tqdm.tqdm(list_datasets):
        df_emb = df['text'].apply(func)
        df_emb = pd.DataFrame(df_emb.tolist(), index=df.index)
        df_emb['label'] = df['label'].values
        list_output.append((name, df_emb))

    return list_output

# %%

def get_embedding_1(text):
    inputs = tokenizer_1(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model_1(**inputs)
    return outputs.last_hidden_state[0, 0, :].numpy()


model_name_1 = "michellejieli/emotion_text_classifier"
tokenizer_1 = AutoTokenizer.from_pretrained(model_name_1)
model_1 = AutoModel.from_pretrained(model_name_1)
list_emb_1 = apply_func_to_datasets(list_datasets, get_embedding_1)

# %%

def get_embedding_2(text):
    inputs = tokenizer_2(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model_2(**inputs)
    return outputs.last_hidden_state[0, 0, :].numpy()


model_name_2 = "j-hartmann/emotion-english-distilroberta-base"
tokenizer_2 = AutoTokenizer.from_pretrained(model_name_2)
model_2 = AutoModel.from_pretrained(model_name_2)
list_emb_2 = apply_func_to_datasets(list_datasets, get_embedding_2)

# %%

def get_embedding_3(text):
    inputs = tokenizer_3(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model_3(**inputs)
    return outputs.last_hidden_state[0, 0, :].numpy()


model_name_3 = "microsoft/deberta-base"
tokenizer_3 = AutoTokenizer.from_pretrained(model_name_3)
model_3 = AutoModel.from_pretrained(model_name_3)
list_emb_3 = apply_func_to_datasets(list_datasets, get_embedding_3)

# %%

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_embedding_4(text):
    # inputs = tokenizer_4(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    encoded_input = tokenizer_4(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model_4(**encoded_input)
    return mean_pooling(outputs, encoded_input['attention_mask']).squeeze()


# Load model from HuggingFace Hub
model_name_4 = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer_4 = AutoTokenizer.from_pretrained(model_name_4)
model_4 = AutoModel.from_pretrained(model_name_4)
list_emb_4 = apply_func_to_datasets(list_datasets, get_embedding_4)

# %%

def get_embedding_5(text):
    inputs = tokenizer_5(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model_5(**inputs)
    return outputs.last_hidden_state[0, 0, :].numpy()

model_name_5 = "state-spaces/mamba-370m-hf"

filepath = "data/output/list_emb_5.pkl"
if os.path.exists(filepath):
    with open(filepath, 'rb') as f:
        list_emb_5 = pickle.load(f)
else:
    tokenizer_5 = AutoTokenizer.from_pretrained(model_name_5)
    model_5 = AutoModel.from_pretrained(model_name_5)

    list_emb_5 = apply_func_to_datasets(list_datasets, get_embedding_5)
    with open(filepath, "wb") as f:
        pickle.dump(list_emb_5, f)

# %%

# def get_embedding_6(text):
#     inputs = tokenizer_6(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
#     with torch.no_grad():
#         outputs = model_6(**inputs)
#     return outputs.last_hidden_state[0, 0, :].numpy()
#
#
# model_name_6 = "state-spaces/mamba2-370m"
# tokenizer_6 = AutoTokenizer.from_pretrained(model_name_6)
# # model_6 = AutoModel.from_pretrained(model_name_6, ignore_mismatched_sizes=True)
# model_6 = AutoModelForCausalLM.from_pretrained(model_name_6, ignore_mismatched_sizes=True)
# list_emb_6 = apply_func_to_datasets(list_datasets, get_embedding_6)
#
# with open("data/output/list_emb_6.pkl", "wb") as f:
#     pickle.dump(list_emb_6, f)

# %%

def get_embedding_6(text):
    inputs = tokenizer_6(text, return_tensors="pt", padding=True,
                         truncation=True, max_length=1024)
    with torch.no_grad():
        outputs = model_6(**inputs)
    return outputs.last_hidden_state[0, 0, :].numpy()


model_name_6 = "michellejieli/emotion_text_classifier"
tokenizer_6 = AutoTokenizer.from_pretrained(model_name_6)
model_6 = AutoModel.from_pretrained(model_name_6)
list_emb_6 = apply_func_to_datasets(list_datasets, get_embedding_6)

# %%

X_train = list_emb_1[0][1]
y_train = X_train['label']
X_train = X_train.drop('label', axis=1)

X_val = list_emb_1[1][1]
y_val = X_val['label']
X_val = X_val.drop('label', axis=1)

model_1 = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.01,
    max_depth=6,
    eval_metric='F1',
    random_seed=42,
    verbose=100,
)

model_1.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    early_stopping_rounds=200,
    use_best_model=True
)

save_experiment_predictions(root_folder="data/experiments/exp_3/",
                            name='1',
                            pipe=model_1,
                            datasets=list_emb_1,
                            column=None,
                            )

# %%

X_train = list_emb_1[0][1]
y_train = X_train['label']
X_train = X_train.drop('label', axis=1)

X_val = list_emb_1[1][1]
y_val = X_val['label']
X_val = X_val.drop('label', axis=1)

model_2 = CatBoostClassifier(
    iterations=1000,
    loss_function='Logloss',
    auto_class_weights='Balanced',
    learning_rate=0.01,
    max_depth=5,
    eval_metric='F1',
    random_seed=42,
    verbose=100,
)

model_2.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    early_stopping_rounds=400,
    use_best_model=True
)

save_experiment_predictions(root_folder="data/experiments/exp_3/",
                            name='2',
                            pipe=model_2,
                            datasets=list_emb_1,
                            column=None,
                            )

# %%

X_train = list_emb_1[0][1]
y_train = X_train['label']
X_train = X_train.drop('label', axis=1)

X_val = list_emb_1[1][1]
y_val = X_val['label']
X_val = X_val.drop('label', axis=1)

model_3 = CatBoostClassifier(
    iterations=2000,
    loss_function='Logloss',
    auto_class_weights='Balanced',
    learning_rate=0.005,
    max_depth=5,
    eval_metric='F1',
    random_seed=42,
    verbose=100,
)

model_3.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    early_stopping_rounds=1000,
    use_best_model=True
)

save_experiment_predictions(root_folder="data/experiments/exp_3/",
                            name='3',
                            pipe=model_3,
                            datasets=list_emb_1,
                            column=None,
                            )

# %%

X_train = list_emb_2[0][1]
y_train = X_train['label']
X_train = X_train.drop('label', axis=1)

X_val = list_emb_2[1][1]
y_val = X_val['label']
X_val = X_val.drop('label', axis=1)

model_4 = CatBoostClassifier(
    iterations=2000,
    loss_function='Logloss',
    # auto_class_weights='Balanced',
    learning_rate=0.05,
    max_depth=4,
    eval_metric='F1',
    random_seed=42,
    verbose=100,
)

model_4.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    early_stopping_rounds=1000,
    use_best_model=True
)

save_experiment_predictions(root_folder="data/experiments/exp_3/",
                            name='4',
                            pipe=model_4,
                            datasets=list_emb_2,
                            column=None,
                            )

# %%

X_train = list_emb_2[0][1]
y_train = torch.FloatTensor(X_train['label'].values.astype(float)).view(-1, 1)
X_train = torch.FloatTensor(X_train.drop('label', axis=1).values)

X_val = list_emb_2[1][1]
y_val = torch.FloatTensor(X_val['label'].values.astype(float)).view(-1, 1)
X_val = torch.FloatTensor(X_val.drop('label', axis=1).values)


train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

class FNN(nn.Module):
    def __init__(self, input_dim):
        super(FNN, self).__init__()
        self.net = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)
    
    def predict_proba(self, X):

        self.eval()
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X.values)
            
        with torch.no_grad():
            prob_pos = self.forward(X).numpy().flatten()
            prob_neg = 1 - prob_pos
            
        return np.column_stack((prob_neg, prob_pos))

    def get_params(self):
        return {}


model_5 = FNN(X_train.shape[1])
criterion = nn.BCELoss() # Binary Cross Entropy
optimizer = optim.Adam(model_5.parameters(), lr=0.005)

for epoch in range(1000):
    # model_5.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model_5(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


save_experiment_predictions(root_folder="data/experiments/exp_3/",
                            name='5',
                            pipe=model_5,
                            datasets=list_emb_2,
                            column=None,
                            )

# %%

X_train = list_emb_1[0][1]
y_train = torch.FloatTensor(X_train['label'].values.astype(float)).view(-1, 1)
X_train = torch.FloatTensor(X_train.drop('label', axis=1).values)

X_val = list_emb_1[1][1]
y_val = torch.FloatTensor(X_val['label'].values.astype(float)).view(-1, 1)
X_val = torch.FloatTensor(X_val.drop('label', axis=1).values)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)

class FNN(nn.Module):
    def __init__(self, input_dim):
        super(FNN, self).__init__()
        self.net = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)
    
    def predict_proba(self, X):

        self.eval()
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X.values)
            
        with torch.no_grad():
            prob_pos = self.forward(X).numpy().flatten()
            prob_neg = 1 - prob_pos
            
        return np.column_stack((prob_neg, prob_pos))

    def get_params(self):
        return {}


model_6 = FNN(X_train.shape[1])
criterion = nn.BCELoss() # Binary Cross Entropy
optimizer = optim.Adam(model_6.parameters(), lr=0.005)

for epoch in range(500):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model_6(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


save_experiment_predictions(root_folder="data/experiments/exp_3/",
                            name='6',
                            pipe=model_6,
                            datasets=list_emb_1,
                            column=None,
                            )


# %%

X_train = list_emb_3[0][1]
y_train = X_train['label']
X_train = X_train.drop('label', axis=1)

X_val = list_emb_3[1][1]
y_val = X_val['label']
X_val = X_val.drop('label', axis=1)

model_7 = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.02,
    max_depth=6,
    eval_metric='F1',
    random_seed=42,
    verbose=100,
)

model_7.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    early_stopping_rounds=200,
    use_best_model=True
)

save_experiment_predictions(root_folder="data/experiments/exp_3/",
                            name='7',
                            pipe=model_7,
                            datasets=list_emb_3,
                            column=None,
                            )

# %%

X_train = list_emb_4[0][1]
y_train = X_train['label']
X_train = X_train.drop('label', axis=1)

X_val = list_emb_4[1][1]
y_val = X_val['label']
X_val = X_val.drop('label', axis=1)

model_8 = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.005,
    max_depth=8,
    eval_metric='F1',
    random_seed=42,
    verbose=100,
)

model_8.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    early_stopping_rounds=200,
    use_best_model=True
)

save_experiment_predictions(root_folder="data/experiments/exp_3/",
                            name='8',
                            pipe=model_8,
                            datasets=list_emb_4,
                            column=None,
                            )

# %%

X_train = list_emb_1[0][1]
y_train = X_train['label']
X_train = X_train.drop('label', axis=1)

X_val = list_emb_1[1][1]
y_val = X_val['label']
X_val = X_val.drop('label', axis=1)

model_9 = CatBoostClassifier(
    iterations=1000,
    loss_function='Logloss',
    auto_class_weights='Balanced',
    learning_rate=0.01,
    max_depth=5,
    eval_metric='F1',
    random_seed=42,
    verbose=100,
)

model_9.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    early_stopping_rounds=400,
    use_best_model=True
)

save_experiment_predictions(root_folder="data/experiments/exp_3/",
                            name='9',
                            pipe=model_9,
                            datasets=list_emb_1,
                            column=None,
                            )
# %%

X_train = list_emb_1[0][1]
y_train = X_train['label']
X_train = X_train.drop('label', axis=1)

X_val = list_emb_1[1][1]
y_val = X_val['label']
X_val = X_val.drop('label', axis=1)

model_10 = BaggingPuClassifier(
    estimator=CatBoostClassifier(
        iterations=200,
        loss_function='Logloss',
        auto_class_weights='Balanced',
        learning_rate=0.01,
        depth=6,
        verbose=100,
        eval_metric='F1',
        random_seed=42,
    ),
    n_estimators=10,
    oob_score=False,
)
 
model_10.fit(X_train, y_train)

save_experiment_predictions(root_folder="data/experiments/exp_3/",
                            name='10',
                            pipe=model_10,
                            datasets=list_emb_1,
                            column=None,
                            )
# %%

X_train = list_emb_1[0][1]
y_train = X_train['label']
X_train = X_train.drop('label', axis=1)

X_val = list_emb_1[1][1]
y_val = X_val['label']
X_val = X_val.drop('label', axis=1)

model_11 = BaggingPuClassifier(
    estimator=HistGradientBoostingClassifier(
        loss='log_loss',
        class_weight='balanced',
        max_iter=500,
        learning_rate=0.005,
        max_depth=7,
        verbose=0,
        random_state=42,
        early_stopping='auto',
        scoring='macro_f1',
    ),
    n_estimators=10,
    oob_score=False,
)
 
model_11.fit(X_train, y_train)

save_experiment_predictions(root_folder="data/experiments/exp_3/",
                            name='11',
                            pipe=model_11,
                            datasets=list_emb_1,
                            column=None,
                            )

# %%

X_train = list_emb_1[0][1]
y_train = X_train['label']
X_train = X_train.drop('label', axis=1)

X_val = list_emb_1[1][1]
y_val = X_val['label']
X_val = X_val.drop('label', axis=1)

model_12 = CatBoostClassifier(
    iterations=1000,
    loss_function='Logloss',
    # auto_class_weights='Balanced',
    learning_rate=0.05,
    max_depth=4,
    eval_metric='F1',
    random_seed=42,
    verbose=100,
)

model_12.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    early_stopping_rounds=200,
    use_best_model=True
)

save_experiment_predictions(root_folder="data/experiments/exp_3/",
                            name='12',
                            pipe=model_12,
                            datasets=list_emb_1,
                            column=None,
                            )

# %%

X_train = list_emb_2[0][1]
y_train = X_train['label']
X_train = X_train.drop('label', axis=1)

X_val = list_emb_2[1][1]
y_val = X_val['label']
X_val = X_val.drop('label', axis=1)

model_13 = BaggingPuClassifier(
    estimator=CatBoostClassifier(
        iterations=100,
        loss_function='Logloss',
        # auto_class_weights='Balanced',
        learning_rate=0.001,
        depth=4,
        verbose=100,
        eval_metric='F1',
        random_seed=42,
    ),
    n_estimators=10,
    oob_score=False,
)
 
model_13.fit(X_train, y_train)


save_experiment_predictions(root_folder="data/experiments/exp_3/",
                            name='13',
                            pipe=model_13,
                            datasets=list_emb_2,
                            column=None,
                            )

# %%

X_train = list_emb_2[0][1]
y_train = X_train['label']
X_train = X_train.drop('label', axis=1)

X_val = list_emb_2[1][1]
y_val = X_val['label']
X_val = X_val.drop('label', axis=1)

model_14 = BaggingClassifier(
    estimator=CatBoostClassifier(
        iterations=100,
        loss_function='Logloss',
        # auto_class_weights='Balanced',
        learning_rate=0.001,
        depth=4,
        verbose=100,
        eval_metric='F1',
        random_seed=42,
    ),
    n_estimators=20,
    oob_score=False,
    max_features=0.5,
    max_samples=0.3,
)
 
model_14.fit(X_train, y_train)


save_experiment_predictions(root_folder="data/experiments/exp_3/",
                            name='14',
                            pipe=model_14,
                            datasets=list_emb_2,
                            column=None,
                            )

# %%

X_train = list_emb_5[0][1]
y_train = X_train['label']
X_train = X_train.drop('label', axis=1)

X_val = list_emb_5[1][1]
y_val = X_val['label']
X_val = X_val.drop('label', axis=1)

model_15 = CatBoostClassifier(
    iterations=10,
    loss_function='Logloss',
    # auto_class_weights='Balanced',
    learning_rate=0.005,
    max_depth=8,
    eval_metric='F1',
    random_seed=42,
    verbose=100,
)

model_15.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    early_stopping_rounds=200,
    use_best_model=True
)

save_experiment_predictions(root_folder="data/experiments/exp_3/",
                            name='15',
                            pipe=model_15,
                            datasets=list_emb_5,
                            column=None,
                            )
# %%

X_train = list_emb_6[0][1]
y_train = X_train['label']
X_train = X_train.drop('label', axis=1)

X_val = list_emb_6[1][1]
y_val = X_val['label']
X_val = X_val.drop('label', axis=1)

model_16 = CatBoostClassifier(
    iterations=150,
    loss_function='Logloss',
    # auto_class_weights='Balanced',
    learning_rate=0.05,
    max_depth=3,
    eval_metric='F1',
    random_seed=42,
    verbose=50,
)

model_16.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    # early_stopping_rounds=500,
    # use_best_model=True
)

save_experiment_predictions(root_folder="data/experiments/exp_3/",
                            name='16',
                            pipe=model_16,
                            datasets=list_emb_6,
                            column=None,
                            )


results, results_mean = calc_scores(root_folder="data/experiments/exp_3/",
                                    plot_names=['14', '3'])

top_names = results_mean.head(6).index.tolist()
print(results_mean.to_markdown(tablefmt="github"))
print()
print(results[results['name'].isin(top_names)].to_markdown(tablefmt="github"))

