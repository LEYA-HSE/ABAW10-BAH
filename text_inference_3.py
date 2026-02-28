import os
import pickle

import polars as pl
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier


def read_split_dataset(filepath):
   raw_df = pl.read_csv(filepath, has_header=False,
                        separator="\n", new_columns=["raw_line"])
   raw_df = raw_df.select(
                       pl.col("raw_line").str.extract_groups(r"^([^,]+),([0,1]),(.*)$")
                       ).unnest("raw_line")
   raw_df.columns = ['file', 'label', 'text']
   return raw_df.to_pandas()


pipeline_1 = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=1000,
    )),
    ('catboost', CatBoostClassifier(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        verbose=100,
        random_seed=42,
        eval_metric='Logloss'
    ))
])

# load pre-train model
filepath = os.path.join("models/", "3__exp_1_model_1.pkl")
with open(filepath, 'rb') as f:
    pipeline_1 = pickle.load(f)

# Inference
# for list with strings
text_list = ["hello world", "test", "test", "hello world"]
prediction = pipeline_1.predict_proba(text_list)[:, 1]
print(prediction)
# >>> [0.38256382 0.37388048 0.37388048 0.38256382]

# for datasets
for name, filepath in [
    ("train", "data/input/split/train.txt"),
    ("valid", "data/input/split/val.txt"),
    ("test", "data/input/split/test.txt"),
]:
    df = read_split_dataset(filepath)

    text_list = df['text'].values.tolist()
    true_labels = df['label'].astype(int).values

    pred_proba = pipeline_1.predict_proba(text_list)[:, 1]
    pred_labels = (pred_proba > 0.5).astype(int)
    score = f1_score(y_true=true_labels, y_pred=pred_labels, average='macro')
    score = round(score, 4)

    print(f"{name:5s} f1 score: {score}")


# 3 & Text (full) & TF-IDF & CatBoost & 65.56 & 72.02 & 68.79 &  \\
# train f1 score: 0.9987
# valid f1 score: 0.6556
# test  f1 score: 0.7205
