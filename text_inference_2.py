import os
import pickle

import polars as pl
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression


def read_split_dataset(filepath):
   raw_df = pl.read_csv(filepath, has_header=False,
                        separator="\n", new_columns=["raw_line"])
   raw_df = raw_df.select(
                       pl.col("raw_line").str.extract_groups(r"^([^,]+),([0,1]),(.*)$")
                       ).unnest("raw_line")
   raw_df.columns = ['file', 'label', 'text']
   return raw_df.to_pandas()


pipeline_13 = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 1),
        max_features=300,
        # min_df=0.005,
        stop_words=None,
    )),
    ('logreg', LogisticRegression(C=0.3, random_state=42, max_iter=1000)),
])


# load pre-train model
filepath = os.path.join("models/", "2__exp_1_model_13.pkl")
with open(filepath, 'rb') as f:
    pipeline_13 = pickle.load(f)

# Inference
# for list with strings
text_list = ["hello world", "test", "test", "hello world"]
prediction = pipeline_13.predict_proba(text_list)[:, 1]
print(prediction)
# >>> [0.4149425 0.4149425 0.4149425 0.4149425]

# for datasets
for name, filepath in [
    ("train", "data/input/split/train.txt"),
    ("valid", "data/input/split/val.txt"),
    ("test", "data/input/split/test.txt"),
]:
    df = read_split_dataset(filepath)

    text_list = df['text'].values.tolist()
    true_labels = df['label'].astype(int).values

    pred_proba = pipeline_13.predict_proba(text_list)[:, 1]
    pred_labels = (pred_proba > 0.5).astype(int)
    score = f1_score(y_true=true_labels, y_pred=pred_labels, average='macro')
    score = round(score, 4)

    print(f"{name:5s} f1 score: {score}")


# 2 & Text (full) & TF-IDF & Logistic Regression & 68.30 & 67.75 & 68.03 &  \\
# train f1 score: 0.7583
# valid f1 score: 0.683
# test  f1 score: 0.6775
