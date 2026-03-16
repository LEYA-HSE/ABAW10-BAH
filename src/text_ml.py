import os
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


def read_split_dataset(filepath):
   raw_df = pl.read_csv(filepath, has_header=False, 
                        separator="\n", new_columns=["raw_line"])
   raw_df = raw_df.select(
                       pl.col("raw_line").str.extract_groups(r"^([^,]+),([0,1]),(.*)$")
                       ).unnest("raw_line")
   raw_df.columns = ['file', 'label', 'text']
   return raw_df.to_pandas()


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


train_df = read_split_dataset("data/input/split/train.txt")
val_df = read_split_dataset("data/input/split/val.txt")
test_df = read_split_dataset("data/input/split/test.txt")
list_datasets = [
    ('train', train_df),
    ('val', val_df),
    ('test', test_df),
]
print(train_df.shape, val_df.shape, test_df.shape)

# %%

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

pipeline_1.fit(train_df['text'], train_df['label'])

save_experiment_predictions(root_folder="data/experiments/exp_1/",
                            name='1',
                            pipe=pipeline_1,
                            datasets=list_datasets,
                            )
# %%

pipeline_2 = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=10000,
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

save_experiment_predictions(root_folder="data/experiments/exp_1/",
                            name='2',
                            pipe=pipeline_2,
                            datasets=list_datasets,
                            )

# %%

pipeline_3 = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=100,
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

pipeline_3.fit(train_df['text'], train_df['label'])

save_experiment_predictions(root_folder="data/experiments/exp_1/",
                            name='3',
                            pipe=pipeline_3,
                            datasets=list_datasets,
                            )
# %%

pipeline_4 = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=500,
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

pipeline_4.fit(train_df['text'], train_df['label'])

save_experiment_predictions(root_folder="data/experiments/exp_1/",
                            name='4',
                            pipe=pipeline_4,
                            datasets=list_datasets)

# %%

pipeline_5 = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=500,
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

pipeline_5.fit(train_df['text'], train_df['label'])

save_experiment_predictions(root_folder="data/experiments/exp_1/",
                            name='5',
                            pipe=pipeline_5,
                            datasets=list_datasets)
# %%

pipeline_6 = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=500,
    )),
    ('catboost', CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        depth=6,
        verbose=100,
        random_seed=42,
        eval_metric='Logloss'
    ))
])

pipeline_6.fit(train_df['text'], train_df['label'])

save_experiment_predictions(root_folder="data/experiments/exp_1/",
                            name='6',
                            pipe=pipeline_6,
                            datasets=list_datasets)
# %%

pipeline_7 = Pipeline([
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

pipeline_7.fit(train_df['text'], train_df['label'])

coef = pipeline_7.steps[-1][1].get_feature_importance()
columns = pipeline_7.steps[0][1].get_feature_names_out()
imp = pd.Series(coef, index=columns)
imp = imp.sort_values(ascending=False)
imp.to_csv("data/experiments/exp_1/7_imp.csv")

save_experiment_predictions(root_folder="data/experiments/exp_1/",
                            name='7',
                            pipe=pipeline_7,
                            datasets=list_datasets)
# %%

pipeline_8 = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=500,
    )),
    ('catboost', CatBoostClassifier(
        iterations=50,
        learning_rate=0.05,
        depth=3,
        verbose=100,
        random_seed=42,
        eval_metric='Logloss'
    ))
])

pipeline_8.fit(train_df['text'], train_df['label'])

save_experiment_predictions(root_folder="data/experiments/exp_1/",
                            name='8',
                            pipe=pipeline_8,
                            datasets=list_datasets)
# %%

pipeline_9 = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=500,
    )),
    ('catboost', CatBoostClassifier(
        iterations=500,
        learning_rate=0.002,
        depth=4,
        verbose=100,
        random_seed=42,
        eval_metric='Logloss'
    ))
])

pipeline_9.fit(train_df['text'], train_df['label'])

save_experiment_predictions(root_folder="data/experiments/exp_1/",
                            name='9',
                            pipe=pipeline_9,
                            datasets=list_datasets)
# %%

pipeline_10 = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 1),
        max_features=100,
        stop_words=None,
    )),
    ('catboost', CatBoostClassifier(
        iterations=500,
        learning_rate=0.002,
        depth=4,
        # verbose=100,
        random_seed=42,
        eval_metric='Logloss'
    ))
])

pipeline_10.fit(train_df['text'], train_df['label'])

save_experiment_predictions(root_folder="data/experiments/exp_1/",
                            name='10',
                            pipe=pipeline_10,
                            datasets=list_datasets)
# %%

pipeline_11 = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 1),
        # max_features=100,
        min_df=0.005,
        stop_words=None,
    )),
    ('catboost', CatBoostClassifier(
        iterations=500,
        learning_rate=0.002,
        depth=4,
        verbose=100,
        random_seed=42,
        eval_metric='Logloss'
    ))
])

pipeline_11.fit(train_df['text'], train_df['label'])

save_experiment_predictions(root_folder="data/experiments/exp_1/",
                            name='11',
                            pipe=pipeline_11,
                            datasets=list_datasets)
# %%

pipeline_12 = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 1),
        # max_features=1000,
        min_df=0.005,
        stop_words=None,
    )),
    ('svd', TruncatedSVD(n_components=200, n_iter=100, random_state=42)),
    ('catboost', CatBoostClassifier(
        iterations=500,
        learning_rate=0.002,
        depth=4,
        verbose=100,
        random_seed=42,
        eval_metric='Logloss'
    ))
])

pipeline_12.fit(train_df['text'], train_df['label'])

save_experiment_predictions(root_folder="data/experiments/exp_1/",
                            name='12',
                            pipe=pipeline_12,
                            datasets=list_datasets)

# %%

pipeline_13 = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 1),
        max_features=300,
        # min_df=0.005,
        stop_words=None,
    )),
    ('logreg', LogisticRegression(C=0.3, random_state=42, max_iter=1000)),
])

pipeline_13.fit(train_df['text'], train_df['label'])

save_experiment_predictions(root_folder="data/experiments/exp_1/",
                            name='13',
                            pipe=pipeline_13,
                            datasets=list_datasets)

coef = pipeline_13.steps[-1][1].coef_
columns = pipeline_13.steps[0][1].get_feature_names_out()
imp = pd.Series(coef[0], index=columns)
imp = imp.sort_values()
imp.to_csv("data/experiments/exp_1/13_imp.csv")

# %%

pipeline_14 = Pipeline([
    ('count', CountVectorizer(
        ngram_range=(1, 2),
        max_features=500,
        # min_df=0.005,
        stop_words=None,
    )),
    ('logreg', LogisticRegression(C=0.3, random_state=42, max_iter=1000)),
])

pipeline_14.fit(train_df['text'], train_df['label'])

save_experiment_predictions(root_folder="data/experiments/exp_1/",
                            name='14',
                            pipe=pipeline_14,
                            datasets=list_datasets)


results, results_mean = calc_scores(root_folder="data/experiments/exp_1/",
                                    plot_names=['13', '7'])

top_names = results_mean.head(5).index.tolist()
print(results_mean.to_markdown(tablefmt="github"))
print(results[results['name'].isin(top_names)].to_markdown(tablefmt="github"))
