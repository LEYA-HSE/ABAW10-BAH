import os
import yaml
import json
import glob

import pandas as pd
import numpy as np
import polars as pl

import matplotlib.pyplot as plt

from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, PrecisionRecallDisplay
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression


def read_frames(filepath):

    df = pd.read_csv(filepath, names=["path", "label"])
    df['frame'] = (
        df['path']
        .str.replace(".jpg", "")
        .str.split("/").str[-1:].str.join("")
        .str.split("-").str[-1:].str.join("")
    )
    df['user'] = (
        df['path'].str.split("/").str[1:3].str.join("/")
    )

    df['question'] = (
        df['path'].str.split("/").str[3:4].str.join("/")
    )
    return df


# raw_df = read_frames("data/input/split-frames/train.txt")
#
# df_uq = (
#     raw_df.groupby(['user', 'question'])['label'].agg(['mean', 'count'])
#     .reset_index()
# )
# df_uq.to_csv("data/experiments/exp_2/user_question_stats.csv")
#
# df_u = (
#     raw_df.groupby(['user']).agg({'label': ['mean', 'count'], 'question': 'nunique'})
#     .reset_index()
# )
# df_u.columns = [f"{x}_{y}" for x, y in df_u.columns.ravel()]
# df_u.to_csv("data/experiments/exp_2/user_stats.csv")


def read_yaml_transcription(directory):
    def construct_python_tuple(loader, node):
        return tuple(loader.construct_sequence(node))

    yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/tuple', construct_python_tuple)

    data = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            _, ext = os.path.splitext(filename)
            main_path = root.replace(directory, "").replace("Videos/", "")
            user = "/".join(main_path.split("/")[:2])
            question = main_path.split("/")[-1]
            filepath = os.path.join(root, filename)
            # print(f"{user} {question}")
            with open(filepath, "r") as f:
                data_yaml = yaml.safe_load(f)

            for indx, chunk in enumerate(data_yaml['chunks']):
                data.append({
                    "user": user,
                    "question": question,
                    "chunk_id": indx,
                    "start_sec": chunk['timestamp'][0],
                    "end_sec": chunk['timestamp'][1],
                    "duration": chunk['timestamp'][1] - chunk['timestamp'][0],
                    "text": chunk['text'],
                })

    data = pd.DataFrame(data)
    return data


def read_frame_dataset(filepath, df_script):
    raw_df = pl.read_csv(filepath,
                         has_header=False,
                         separator="\n",
                         new_columns=["raw_line"])
    raw_df = raw_df.select(
        pl.col("raw_line").str.extract_groups(r"^([^,]+),([0,1]),(.*)$")
    ).unnest("raw_line")
    raw_df.columns = ['file', 'label', 'text']
    raw_df = raw_df.to_pandas()
    raw_df['user'] = (
        raw_df['file'].str.split("/").str[1:3].str.join("/")
    )

    raw_df['question'] = (
        raw_df['file'].str.split("/").str[3:4].str.join("/")
    )

    raw_df = raw_df.drop(['text', 'file'], axis=1)
    df = raw_df.merge(df_script, on=['user', 'question'], how='left')

    return df


def save_experiment_predictions(root_folder, name, pipe, datasets):

    def is_serializable(obj):
        try:
            json.dumps(obj)
            return True
        except (TypeError, OverflowError):
            return False

    # Проверка словаря
    params = {}
    for key, value in pipe.get_params().items():
        if is_serializable(value):
            params[key] = value

    filepath = os.path.join(root_folder, f"{name}__params.json")
    print(f"file {filepath} was saved")
    with open(filepath, 'w') as f:
        json.dump(params, f, indent=4)

    for dtype, df in datasets:
        pred = pipe.predict_proba(df['text'])
        # print(pred[:, 1].shape, df['label'].values.shape)
        data = np.column_stack([pred[:, 1], df['label'].values])
        # data = np.concatenate([pred[:, 1], df['label'].values], )
        filepath = os.path.join(root_folder, f"{name}__{dtype}.npy")
        print(f"file {filepath} was saved")
        np.save(filepath, data)


def calc_scores(root_folder, thr=0.5, plot_names=None):
    data_dict = {}
    for filepath in glob.glob(f"{root_folder}/*.npy"):
        basename = os.path.basename(filepath)
        name, dtype = basename.replace(".npy", "").split("__")
        if name not in data_dict:
            data_dict[name] = []
        data_dict[name].append((dtype, filepath))

    if plot_names is not None:
        fig, ax = plt.subplots(figsize=(8, 6))

    scores = []
    for name, dtype_list in data_dict.items():
        for dtype, filepath in sorted(set(dtype_list)):
            # print(filepath)
            pred_and_true = np.load(filepath, allow_pickle=True)
            # print(pred_and_true)
            y_true = pred_and_true[:, 1].astype(int)
            y_score = pred_and_true[:, 0]
            y_pred = (y_score >= thr).astype(int)
            # print(y_true[:10])
            # print(y_pred[:10])
            f1_macro = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
            roc_auc = roc_auc_score(y_true=y_true, y_score=y_score)
            pr_auc = average_precision_score(y_true=y_true, y_score=y_score)

            scores.append({
                "name": name,
                "dtype": dtype,
                "thr": thr,
                "f1": f1_macro,
                "roc": roc_auc,
                "pr": pr_auc,
            })

            if plot_names is not None and name in plot_names:
                PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax, name=f"{name} {dtype}")

    if plot_names is not None:
        filepath = os.path.join(root_folder, "chart.png")
        ax.set_title("PR Curve")
        ax.grid(True)
        print(f"chart {filepath} was created")
        fig.savefig(filepath)

    scores = pd.DataFrame(scores)
    scores = scores.round(4)
    scores = scores.sort_values(["dtype", "f1"], ascending=[True, False])
    scores = scores.reset_index(drop=True)

    scores.to_csv(f"{root_folder}/results.csv")
    with open(f"{root_folder}/results.txt", "w") as f:
        f.write(scores.to_markdown(tablefmt="mixed_outline"))

    scores_mean = (
        scores
        [scores['dtype'] != 'train']
        .groupby(['name'])
        [['f1', 'roc', 'pr']]
        .agg(['mean', 'std'])
        .round(4)
        .sort_values([("f1", "std")])
    )

    scores_mean.columns = [f"{x}_{y}" for x, y in scores_mean.columns.ravel()]

    return scores, scores_mean


# %%


df_script = read_yaml_transcription("data/input/transcription/")
train_df = read_frame_dataset("data/input/split/train.txt", df_script)
val_df = read_frame_dataset("data/input/split/val.txt", df_script)
test_df = read_frame_dataset("data/input/split/test.txt", df_script)

list_datasets = [
    ('train', train_df),
    ('val', val_df),
    ('test', test_df),
]
print(df_script.shape, train_df.shape, val_df.shape, test_df.shape)

# %%

pipeline_1 = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=500,
    )),
    ('catboost', CatBoostClassifier(
        iterations=10,
        learning_rate=0.1,
        depth=6,
        verbose=100,
        random_seed=42,
        eval_metric='Logloss'
    ))
])

pipeline_1.fit(train_df['text'], train_df['label'])

save_experiment_predictions(root_folder="data/experiments/exp_2/",
                            name='1',
                            pipe=pipeline_1,
                            datasets=list_datasets)
# %%

pipeline_2 = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),
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

pipeline_2.fit(train_df['text'], train_df['label'])

save_experiment_predictions(root_folder="data/experiments/exp_2/",
                            name='2',
                            pipe=pipeline_2,
                            datasets=list_datasets)

# %%

pipeline_3 = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=3000,
    )),
    ('catboost', CatBoostClassifier(
        iterations=5000,
        learning_rate=0.02,
        depth=8,
        verbose=100,
        random_seed=42,
        eval_metric='Logloss'
    ))
])

pipeline_3.fit(train_df['text'], train_df['label'])

save_experiment_predictions(root_folder="data/experiments/exp_2/",
                            name='3',
                            pipe=pipeline_3,
                            datasets=list_datasets)

# %%

pipeline_4 = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),
        # max_features=1000,
        min_df=0.005,
        stop_words=None,
    )),
    ('logreg', LogisticRegression(C=1, random_state=42, max_iter=1000)),
])

pipeline_4.fit(train_df['text'], train_df['label'])

save_experiment_predictions(root_folder="data/experiments/exp_2/",
                            name='4',
                            pipe=pipeline_4,
                            datasets=list_datasets)

# %%

pipeline_5 = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=3000,
    )),
    ('catboost', CatBoostClassifier(
        iterations=500,
        learning_rate=0.02,
        depth=5,
        verbose=100,
        random_seed=42,
        eval_metric='Logloss'
    ))
])

pipeline_5.fit(train_df['text'], train_df['label'])

save_experiment_predictions(root_folder="data/experiments/exp_2/",
                            name='5',
                            pipe=pipeline_5,
                            datasets=list_datasets)

results, results_mean = calc_scores(root_folder="data/experiments/exp_2/",
                                    plot_names=['4', '2'])

top_names = results_mean.head(5).index.tolist()
print(results_mean.to_markdown(tablefmt="github"))
print(results[results['name'].isin(top_names)].to_markdown(tablefmt="github"))

