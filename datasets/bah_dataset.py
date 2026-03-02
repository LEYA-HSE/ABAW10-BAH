from __future__ import annotations
from typing import Dict, List, Tuple, Optional

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .bah_chunks_dataset import BAHChunkDataset, build_train_chunk_pairs, ChunkMeta
from training.config import DataConfig, TrainConfig
from .embedding_store import EmbeddingStore


def _parse_txt_lines(txt_path: str, sep: str, id_field: int, label_field: int) -> List[Tuple[str, int]]:
    pairs: List[Tuple[str, int]] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(sep)
            if len(parts) < max(id_field, label_field) + 1:
                raise ValueError(f"Bad line in {txt_path}: '{line}'")
            sid = parts[id_field].strip()
            y = int(parts[label_field].strip())
            pairs.append((sid, y))
    return pairs


class BAHDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, int]], store: EmbeddingStore, allow_index_alignment: bool = False):
        self.pairs = pairs
        self.store = store
        self.allow_index_alignment = allow_index_alignment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        sid, y = self.pairs[idx]
        emb = self.store.get(sid, allow_index_alignment=self.allow_index_alignment, idx=idx)
        return {"id": sid, "x": emb.astype(np.float32), "y": int(y)}


def bah_collate(batch: List[dict]):
    xs = [b["x"] for b in batch]
    ys = torch.tensor([b["y"] for b in batch], dtype=torch.long)
    ids = [b["id"] for b in batch]

    if xs[0].ndim == 1:
        x = torch.tensor(np.stack(xs, axis=0), dtype=torch.float32)
        return {"id": ids, "x": x, "mask": None, "y": ys}

    lens = [x.shape[0] for x in xs]
    max_len = max(lens)
    d = xs[0].shape[1]
    x_pad = np.zeros((len(xs), max_len, d), dtype=np.float32)
    mask = np.zeros((len(xs), max_len), dtype=np.bool_)
    for i, x_i in enumerate(xs):
        T = x_i.shape[0]
        x_pad[i, :T, :] = x_i
        mask[i, :T] = True

    return {
        "id": ids,
        "x": torch.tensor(x_pad, dtype=torch.float32),
        "mask": torch.tensor(mask, dtype=torch.bool),
        "y": ys,
    }


def build_bah_loaders(data_cfg: DataConfig, train_cfg: TrainConfig):
    store_full = EmbeddingStore(data_cfg.embeddings_npz)

    train_pairs = _parse_txt_lines(data_cfg.train_txt, data_cfg.sep, data_cfg.id_field, data_cfg.label_field)
    val_pairs   = _parse_txt_lines(data_cfg.val_txt,   data_cfg.sep, data_cfg.id_field, data_cfg.label_field)
    test_pairs  = _parse_txt_lines(data_cfg.test_txt,  data_cfg.sep, data_cfg.id_field, data_cfg.label_field)

    eval_all_pairs = val_pairs + test_pairs

    ds_val      = BAHDataset(val_pairs, store_full, allow_index_alignment=data_cfg.allow_index_alignment)
    ds_test     = BAHDataset(test_pairs, store_full, allow_index_alignment=data_cfg.allow_index_alignment)
    ds_eval_all = BAHDataset(eval_all_pairs, store_full, allow_index_alignment=data_cfg.allow_index_alignment)

    train_mode = (data_cfg.train_mode or "full").lower()
    sizes: Dict[str, int] = {}

    chunk_pairs: List[Tuple[str, int]] = []  
    if train_mode == "full":
        ds_train = BAHDataset(train_pairs, store_full, allow_index_alignment=data_cfg.allow_index_alignment)
        sizes["train_full"] = len(ds_train)
        sizes["train_chunks"] = 0
    else:
        
        train_base = [os.path.basename(sid) for sid, _ in train_pairs]

        base2meta: Dict[str, ChunkMeta] = {}
        for sid, _ in train_pairs:
            parts = sid.split("/")
            if len(parts) >= 3:
                base2meta[os.path.basename(sid)] = ChunkMeta(speaker=parts[1], visit=parts[2])

        if not data_cfg.embeddings_npz_chunks:
            raise ValueError("data.embeddings_npz_chunks is required for train_mode='chunks' or 'mixed'")
        if not data_cfg.chunks_yaml_root:
            raise ValueError("data.chunks_yaml_root is required for train_mode='chunks' or 'mixed'")

        store_chunks = EmbeddingStore(data_cfg.embeddings_npz_chunks)
        chunk_pairs, chunk_stats = build_train_chunk_pairs(
            store_chunks=store_chunks,
            train_base_filenames=train_base,
            base2meta=base2meta,
            chunks_yaml_root=data_cfg.chunks_yaml_root,
        )

        ds_train_chunks = BAHChunkDataset(chunk_pairs, store_chunks)
        sizes["train_chunks"] = len(ds_train_chunks)
        sizes["chunks_total_keys"] = int(chunk_stats.get("total_keys", 0))
        sizes["chunks_skipped_not_train"] = int(chunk_stats.get("skipped_not_train", 0))
        sizes["chunks_skipped_no_yaml"] = int(chunk_stats.get("skipped_no_yaml", 0))
        sizes["chunks_skipped_bad"] = int(chunk_stats.get("skipped_bad", 0))

        if train_mode == "chunks":
            ds_train = ds_train_chunks
            sizes["train_full"] = 0
        elif train_mode == "mixed":
            from torch.utils.data import ConcatDataset
            ds_train_full = BAHDataset(train_pairs, store_full, allow_index_alignment=data_cfg.allow_index_alignment)
            ds_train = ConcatDataset([ds_train_full, ds_train_chunks])
            sizes["train_full"] = len(ds_train_full)
        else:
            raise ValueError(f"Unknown data.train_mode: {data_cfg.train_mode}")

    if train_mode == "full":
        src_pairs = train_pairs
    elif train_mode == "chunks":
        src_pairs = chunk_pairs
    elif train_mode == "mixed":
        src_pairs = train_pairs + chunk_pairs
    else:
        src_pairs = train_pairs

    train_label_counts: Dict[int, int] = {}
    for _, y in src_pairs:
        yy = int(y)
        train_label_counts[yy] = train_label_counts.get(yy, 0) + 1

    for k, v in sorted(train_label_counts.items()):
        sizes[f"train_label_{k}"] = int(v)

    class_weights: Optional[List[float]] = None
    if train_label_counts:
        C = max(train_label_counts.keys()) + 1
        total = sum(train_label_counts.values())
        w: List[float] = []
        for c in range(C):
            cnt = train_label_counts.get(c, 0)
            if cnt <= 0:
                w.append(0.0)
            else:
                w.append(float(total) / float(C * cnt))
        class_weights = w

    def mk_loader(ds, shuffle: bool, batch_size: int):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=train_cfg.num_workers,
            pin_memory=train_cfg.pin_memory,
            collate_fn=bah_collate,
            drop_last=False,
        )

    eval_bs = getattr(train_cfg, "eval_batch_size", train_cfg.batch_size)
    if eval_bs is None or eval_bs <= 0:
        eval_bs = train_cfg.batch_size

    return {
        "train": mk_loader(ds_train, shuffle=True, batch_size=train_cfg.batch_size),
        "val": mk_loader(ds_val, shuffle=False, batch_size=eval_bs),
        "test": mk_loader(ds_test, shuffle=False, batch_size=eval_bs),
        "eval_all": mk_loader(ds_eval_all, shuffle=False, batch_size=eval_bs),
        "input_dim": store_full.get_dim(),
        "class_weights": class_weights, 
        "sizes": {
            "train": len(ds_train),
            "val": len(ds_val),
            "test": len(ds_test),
            "eval_all": len(ds_eval_all),
            **sizes,
        },
    }
