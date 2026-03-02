from __future__ import annotations
import numpy as np
from sklearn.metrics import f1_score


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        t = int(t); p = int(p)
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def compute_mf1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    if y_true.size == 0:
        return 0.0

    labels = list(range(int(num_classes)))
    return float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))


