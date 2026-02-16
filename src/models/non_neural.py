# coding: utf-8
from __future__ import annotations
from typing import Optional

import numpy as np


def _one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    y = y.astype(int)
    out = np.zeros((y.size, num_classes), dtype=np.float32)
    out[np.arange(y.size), y] = 1.0
    return out


def _rbf_kernel(X1: np.ndarray, X2: np.ndarray, gamma: float) -> np.ndarray:
    # ||x-y||^2 = x^2 + y^2 - 2xy
    X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
    X2_sq = np.sum(X2 ** 2, axis=1, keepdims=True).T
    d2 = X1_sq + X2_sq - 2.0 * np.dot(X1, X2.T)
    return np.exp(-gamma * d2)


class KernelELMClassifier:
    """
    Kernel Extreme Learning Machine (RBF kernel).
    """
    def __init__(self, C: float = 1.0, gamma: float = 1.0):
        self.C = float(C)
        self.gamma = float(gamma)
        self.X_train: Optional[np.ndarray] = None
        self.beta: Optional[np.ndarray] = None
        self.num_classes: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray, num_classes: int):
        self.X_train = X.astype(np.float32)
        self.num_classes = int(num_classes)
        T = _one_hot(y, self.num_classes)
        K = _rbf_kernel(self.X_train, self.X_train, self.gamma)
        n = K.shape[0]
        reg = np.eye(n, dtype=np.float32) / self.C
        self.beta = np.linalg.solve(K + reg, T)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.X_train is None or self.beta is None:
            raise RuntimeError("KernelELMClassifier is not fitted.")
        K = _rbf_kernel(X.astype(np.float32), self.X_train, self.gamma)
        return K @ self.beta

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        return scores.argmax(axis=1)


class ELMClassifier:
    """
    Linear ELM with random hidden layer + ridge regression.
    """
    def __init__(
        self,
        hidden_dim: int = 512,
        activation: str = "relu",
        C: float = 1.0,
        seed: int = 42,
    ):
        self.hidden_dim = int(hidden_dim)
        self.activation = activation
        self.C = float(C)
        self.rng = np.random.RandomState(seed)
        self.W: Optional[np.ndarray] = None
        self.b: Optional[np.ndarray] = None
        self.beta: Optional[np.ndarray] = None
        self.num_classes: int = 0

    def _act(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "tanh":
            return np.tanh(x)
        if self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-x))
        if self.activation == "relu":
            return np.maximum(0.0, x)
        raise ValueError(f"Unknown activation: {self.activation}")

    def fit(self, X: np.ndarray, y: np.ndarray, num_classes: int):
        X = X.astype(np.float32)
        n, d = X.shape
        self.num_classes = int(num_classes)
        self.W = self.rng.normal(0, 1, size=(d, self.hidden_dim)).astype(np.float32)
        self.b = self.rng.normal(0, 1, size=(self.hidden_dim,)).astype(np.float32)
        H = self._act(X @ self.W + self.b)  # [N, H]
        T = _one_hot(y, self.num_classes)
        # ridge regression: beta = (H^T H + I/C)^-1 H^T T
        reg = np.eye(self.hidden_dim, dtype=np.float32) / self.C
        self.beta = np.linalg.solve(H.T @ H + reg, H.T @ T)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.W is None or self.b is None or self.beta is None:
            raise RuntimeError("ELMClassifier is not fitted.")
        X = X.astype(np.float32)
        H = self._act(X @ self.W + self.b)
        return H @ self.beta

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        return scores.argmax(axis=1)
