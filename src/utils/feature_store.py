from __future__ import annotations
import os
import pickle
import time
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Any, Tuple

import torch


# ---------------------------
# 1) Cache key
# ---------------------------

@dataclass(frozen=True)
class CacheKey:
    mod: str
    extractor_fp: str
    avg: str
    frames: int
    pre_v: str = "v1"

    def short_id(self) -> str:
        # Windows-safe sanitization: \ / : * ? " < > | and spaces ? '-'
        def _sanitize(s: str) -> str:
            bad = '\\/:*?"<>|'
            t = []
            for ch in s:
                t.append('-' if ch in bad or ch.isspace() else ch)
            out = ''.join(t)
            while '--' in out:
                out = out.replace('--', '-')
            return out.strip('-')

        parts = [
            self.mod,
            self.extractor_fp,
            f"frames{self.frames}",
            f"avg-{self.avg}",
            f"pv-{self.pre_v}",
        ]
        human = '__'.join(_sanitize(str(p)) for p in parts if p)
        return human[:144]


def build_cache_key(mod: str, extractor: Any, cfg: Any, avg_override: Optional[str] = None) -> CacheKey:
    """
    Build CacheKey for 'face'.
    """
    if mod != "face":
        raise ValueError(f"Unsupported modality '{mod}', expected 'face'.")

    fp_fn = getattr(extractor, "fingerprint", None)
    extractor_fp = fp_fn() if callable(fp_fn) else type(extractor).__name__

    if avg_override is None:
        avg = str(getattr(cfg, "average_features", "raw")).lower()
    else:
        avg = str(avg_override).lower()
    frames = int(getattr(cfg, "segment_length", 30))
    pre_v = str(getattr(cfg, "preprocess_version", "v1"))
    mode = str(getattr(cfg, "video_mode", "stable"))

    return CacheKey(
        mod="face",
        extractor_fp=extractor_fp,
        avg=avg,
        frames=frames,
        pre_v=f"{pre_v}-{mode}",
    )


# ---------------------------
# 2) Save/load helpers
# ---------------------------

def _safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)


def _atomic_save_pt(obj: Any, path: str):
    tmp = f"{path}.tmp_{os.getpid()}_{int(time.time()*1000)}"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_save_pickle(obj: Any, path: str):
    tmp = f"{path}.tmp_{os.getpid()}_{int(time.time()*1000)}"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


# ---------------------------
# 3) Feature store
# ---------------------------

class FeatureStore:
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self._stores_mem: Dict[Tuple[str, str, str, int, int, str], Dict[str, Optional[dict]]] = {}

    def _base_dir(self, dataset: str, split: str) -> str:
        return os.path.join(self.root, dataset, split)

    def meta_path(self, dataset: str, split: str, seed: int, subset: int) -> str:
        base = self._base_dir(dataset, split)
        _safe_makedirs(base)
        return os.path.join(base, f"meta_seed{seed}_subset{subset}.pickle")

    def feats_path(self, dataset: str, split: str, key: CacheKey, seed: int, subset: int) -> str:
        base = self._base_dir(dataset, split)
        mod_dir = os.path.join(base, key.mod, key.short_id())
        _safe_makedirs(mod_dir)
        fname = f"feats_seed{seed}_subset{subset}_avg-{key.avg}.pt"
        return os.path.join(mod_dir, fname)

    # --- meta
    def load_meta(self, dataset: str, split: str, seed: int, subset: int) -> list[dict]:
        p = self.meta_path(dataset, split, seed, subset)
        if not os.path.exists(p):
            return []
        with open(p, "rb") as f:
            return pickle.load(f)

    def save_meta(self, dataset: str, split: str, seed: int, subset: int, meta: list[dict]):
        p = self.meta_path(dataset, split, seed, subset)
        _atomic_save_pickle(meta, p)

    # --- features
    def load_modality_store(self, dataset: str, split: str, key: CacheKey, seed: int, subset: int) -> Tuple[Dict[str, Optional[dict]], Optional[CacheKey]]:
        p = self.feats_path(dataset, split, key, seed, subset)
        if not os.path.exists(p):
            return {}, None
        obj = torch.load(p, map_location="cpu")
        data = obj.get("data", {}) if isinstance(obj, dict) else obj
        header = obj.get("header", None)
        if isinstance(header, dict):
            header = CacheKey(**header)
        return data, header

    def save_modality_store(self, dataset: str, split: str, key: CacheKey, seed: int, subset: int, store: Dict[str, Optional[dict]]):
        p = self.feats_path(dataset, split, key, seed, subset)
        payload = {"header": asdict(key), "data": store}
        _atomic_save_pt(payload, p)

    def get_store(self, dataset: str, split: str, key: CacheKey, seed: int, subset: int) -> Dict[str, Optional[dict]]:
        mem_key = (dataset, split, key.mod, seed, subset, key.avg)
        if mem_key in self._stores_mem:
            return self._stores_mem[mem_key]
        store, _ = self.load_modality_store(dataset, split, key, seed, subset)
        self._stores_mem[mem_key] = store
        return store


# ---------------------------
# 4) Helpers
# ---------------------------

def need_full_reextract(cfg: Any, mod: str, old_header: Optional[CacheKey], new_key: CacheKey) -> bool:
    if getattr(cfg, "overwrite_modality_cache", False):
        return True
    force_list = set(getattr(cfg, "force_reextract", []) or [])
    if mod in force_list:
        return True
    return (old_header is None) or (old_header != new_key)


def merge_missing(store: Dict[str, Optional[dict]], sample_names: list[str]) -> list[str]:
    return [s for s in sample_names if s not in store]
