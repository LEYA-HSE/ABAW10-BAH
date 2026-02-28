import pickle
import argparse
from pathlib import Path

import polars as pl
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier


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


def model_inference(video_list, text_list, pipeline):
    out = {}
    for rel_key, text in zip(video_list, text_list):
        pred_proba = pipeline.predict_proba([text])[:, 1]
        embeddings = pipeline.steps[0][1].transform([text]).todense()
        out[rel_key] = {
            "prob": pred_proba,
            "logits": pred_proba,
            "embeddings": embeddings,
        }
    return out


def main():
    args = parse_args()

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
    filepath = str(Path("./models/3__exp_1_model_1.pkl")) 
    with open(filepath, 'rb') as f:
        pipeline_1 = pickle.load(f)
   
    # Inference
    df = read_split_dataset(args.inp)

    video_list = df['file'].values.tolist()
    text_list = df['text'].values.tolist()

    out_dict = model_inference(video_list, text_list, pipeline_1)

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


# 3 & Text (full) & TF-IDF & CatBoost & 65.56 & 72.02 & 68.79 &  \\
# train f1 score: 0.9987
# valid f1 score: 0.6556
# test  f1 score: 0.7205
