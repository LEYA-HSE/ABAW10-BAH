from __future__ import annotations
from typing import Any, Dict, Optional
import os
import numpy as np


class EmbeddingStore:
    def __init__(self, npz_path: str):
        self.npz_path = npz_path

        d = np.load(npz_path, allow_pickle=False)
        if "keys" not in d:
            raise KeyError(f"NPZ '{npz_path}' must contain 'keys'. Found: {list(d.keys())}")

        keys = np.asarray(d["keys"])
        self.keys = np.array([str(k) for k in keys], dtype=object)
        self.id2idx: Dict[str, int] = {k: i for i, k in enumerate(self.keys)}

        if "v_0" not in d:
            raise KeyError(f"NPZ '{npz_path}' has 'keys' but missing 'v_0'")
        d.close()

        self._pid = None
        self._npz = None

    def _ensure_open(self):
        pid = os.getpid()
        if self._npz is None or self._pid != pid:
            self._npz = np.load(self.npz_path, allow_pickle=False)
            self._pid = pid

    def get_dim(self) -> int:
        self._ensure_open()
        v0 = np.array(self._npz["v_0"])
        if v0.ndim == 3 and v0.shape[0] == 1:
            v0 = v0[0]
        if v0.ndim != 2:
            raise ValueError(f"Expected v_0 (T,H) or (1,T,H), got {v0.shape}")
        return int(v0.shape[-1])

    def get(self, key: Any, allow_index_alignment: bool = False, idx: Optional[int] = None) -> np.ndarray:
        self._ensure_open()
        k = str(key)
        if k not in self.id2idx:
            if k.startswith("Videos/"):
                k2 = "Audio/" + k[len("Videos/"):]
                if k2 in self.id2idx:
                    k = k2
            elif k.startswith("Audio/"):
                k2 = "Videos/" + k[len("Audio/"):]
                if k2 in self.id2idx:
                    k = k2

        if k not in self.id2idx:
            raise KeyError(f"Embedding id '{key}' not found in NPZ (example keys: {self.keys[:3]})")

        i = self.id2idx[k]
        emb = np.array(self._npz[f"v_{i}"])

        if emb.ndim == 3 and emb.shape[0] == 1:
            emb = emb[0]
        if emb.ndim != 2:
            raise ValueError(f"Expected (T,H) or (1,T,H), got {emb.shape} for key={k}")
        return emb
