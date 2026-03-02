from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .embedding_store import EmbeddingStore


_CHUNK_RE = re.compile(r"^(?P<prefix>.*?)(?P<stem>[^/]+)_chunk(?P<idx>\d{4})\.mp4$")


def parse_chunk_key(chunk_key: str) -> Tuple[str, int]:
    m = _CHUNK_RE.match(chunk_key)
    if not m:
        raise ValueError(f"Bad chunk key format: {chunk_key}")
    stem = m.group("stem")
    idx = int(m.group("idx"))
    base = stem + ".mp4"
    return base, idx


def _safe_speaker_from_base(base_filename: str) -> Optional[str]:
    first = base_filename.split("_", 1)[0]
    return first if first.isdigit() else None


def _yaml_path(chunks_yaml_root: str, speaker: str, visit: str, base_filename: str) -> str:
    base_no_ext = os.path.splitext(base_filename)[0]
    return os.path.join(chunks_yaml_root, "Videos", speaker, visit, base_filename, base_no_ext + ".yml")


def _load_yaml(path: str) -> dict:
    try:
        import yaml  
    except Exception as e:
        raise ImportError("pyyaml is required for chunk training. Install: pip install pyyaml") from e

    class _Loader(yaml.SafeLoader):
        pass

    def _construct_tuple(loader, node):
        return tuple(loader.construct_sequence(node))

    _Loader.add_constructor("tag:yaml.org,2002:python/tuple", _construct_tuple)

    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=_Loader)  


@dataclass
class ChunkMeta:
    speaker: str
    visit: str


class ChunkLabelIndex:
 

    def __init__(self, chunks_yaml_root: str, base2meta: Dict[str, ChunkMeta]):
        self.root = chunks_yaml_root
        self.base2meta = base2meta
        self._cache: Dict[str, List[int]] = {}

    def _get_labels_for_base(self, base_filename: str) -> List[int]:
        if base_filename in self._cache:
            return self._cache[base_filename]

        meta = self.base2meta.get(base_filename)
        if meta is None:
            sp = _safe_speaker_from_base(base_filename)
            if sp is None:
                raise FileNotFoundError(f"Cannot infer speaker for base file '{base_filename}'")
            meta = ChunkMeta(speaker=sp, visit="Visite_1")

        yml = _yaml_path(self.root, meta.speaker, meta.visit, base_filename)
        if not os.path.exists(yml):
            raise FileNotFoundError(yml)

        data = _load_yaml(yml)
        chunks = data.get("chunks", []) or []
        labels: List[int] = []
        for ch in chunks:
            if isinstance(ch, dict) and "label" in ch:
                labels.append(int(ch["label"]))
            else:
                labels.append(-1)
        self._cache[base_filename] = labels
        return labels

    def get_label(self, base_filename: str, chunk_idx: int) -> int:
        labels = self._get_labels_for_base(base_filename)
        if chunk_idx < 0 or chunk_idx >= len(labels):
            raise IndexError(f"chunk_idx {chunk_idx} out of range for {base_filename} (num={len(labels)})")
        lab = int(labels[chunk_idx])
        if lab < 0:
            raise KeyError(f"Missing label for {base_filename} chunk {chunk_idx}")
        return lab


def build_train_chunk_pairs(
    *,
    store_chunks: EmbeddingStore,
    train_base_filenames: List[str],
    base2meta: Dict[str, ChunkMeta],
    chunks_yaml_root: str,
) -> Tuple[List[Tuple[str, int]], Dict[str, int]]:
    
    train_set = set(train_base_filenames)
    index = ChunkLabelIndex(chunks_yaml_root, base2meta)

    pairs: List[Tuple[str, int]] = []
    stats = {"total_keys": 0, "kept": 0, "skipped_not_train": 0, "skipped_no_yaml": 0, "skipped_bad": 0}

    for ck in store_chunks.keys:
        stats["total_keys"] += 1
        try:
            base, cidx = parse_chunk_key(ck)
        except Exception:
            stats["skipped_bad"] += 1
            continue

        if base not in train_set:
            stats["skipped_not_train"] += 1
            continue

        try:
            y = index.get_label(base, cidx)
        except FileNotFoundError:
            stats["skipped_no_yaml"] += 1
            continue
        except Exception:
            stats["skipped_bad"] += 1
            continue

        pairs.append((ck, int(y)))
        stats["kept"] += 1

    return pairs, stats


class BAHChunkDataset:

    def __init__(self, pairs: List[Tuple[str, int]], store: EmbeddingStore):
        self.pairs = pairs
        self.store = store

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        sid, y = self.pairs[idx]
        emb = self.store.get(sid, allow_index_alignment=False, idx=None)
        return {"id": sid, "x": emb.astype(np.float32), "y": int(y)}
