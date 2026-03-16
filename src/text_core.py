import os
import json
import glob
import pickle

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, PrecisionRecallDisplay


def save_experiment_predictions(root_folder, name, pipe, datasets, column='text'):

    def is_serializable(obj):
        try:
            json.dumps(obj)
            return True
        except (TypeError, OverflowError):
            return False

    params = {}
    for key, value in pipe.get_params().items():
        if is_serializable(value):
            params[key] = value

    filepath = os.path.join(root_folder, f"{name}__params.json")
    print(f"file {filepath} was saved")
    with open(filepath, 'w') as f:
        json.dump(params, f, indent=4)

    filepath = os.path.join(root_folder, f"{name}__model.pkl")
    print(f"file {filepath} was saved")
    with open(filepath, 'wb') as f:
        pickle.dump(pipe, f)

    for dtype, df in datasets:
        X = df.drop('label', axis=1) if column is None else df[column] 
        pred = pipe.predict_proba(X)
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
        .sort_values([("f1", "mean")], ascending=False)
    )

    scores_mean.columns = [f"{x}_{y}" for x, y in scores_mean.columns.ravel()]

    return scores, scores_mean

