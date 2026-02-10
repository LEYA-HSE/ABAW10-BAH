from __future__ import annotations
from typing import Dict, Any, List, Tuple
import os
import torch
from torch.utils.data import DataLoader, ConcatDataset

from .dataset_bah import BAHFaceDataset


def bah_collate_fn(batch: List[Dict[str, Any]]):
    """Simple collate: names, video paths, labels. Features stay as-is."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    names = [b["sample_name"] for b in batch]
    vpaths = [b["video_path"] for b in batch]

    labels = torch.stack(
        [
            torch.as_tensor(b["label"], dtype=torch.long)
            if not isinstance(b["label"], torch.Tensor) else b["label"].to(torch.long)
            for b in batch
        ],
        dim=0,
    )

    features = [b.get("features") for b in batch]

    return {
        "video_paths": vpaths,
        "labels": labels,
        "names": names,
        "features": features,
    }


def make_bah_dataset_and_loader(config, split: str) -> Tuple[ConcatDataset, DataLoader]:
    """
    Expect dataset sections in config.toml that start with "bah_".
    Example:
      [datasets.bah_ambivalence]
    Each section should define:
      base_dir, csv_path, video_dir (with {base_dir} and {split} placeholders)
    """
    ds_list = []
    for ds_name, ds_cfg in getattr(config, "datasets", {}).items():
        if not ds_name.lower().startswith("bah_"):
            continue
        csv_path = ds_cfg["csv_path"].format(base_dir=ds_cfg["base_dir"], split=split)
        video_dir = ds_cfg["video_dir"].format(base_dir=ds_cfg["base_dir"], split=split)
        if not os.path.exists(csv_path):
            print(f"[BAH] skip {ds_name} for split={split}: CSV not found -> {csv_path}")
            continue

        ds = BAHFaceDataset(
            csv_path=csv_path,
            video_dir=video_dir,
            config=config,
            split=split,
            modality_processors=getattr(config, "modality_processors"),
            modality_feature_extractors=getattr(config, "modality_extractors"),
            dataset_name=ds_name,
            device=getattr(config, "device", "cuda"),
        )
        ds_list.append(ds)

    if not ds_list:
        raise ValueError(f"For split='{split}' no BAH datasets were found.")

    dataset = ds_list[0] if len(ds_list) == 1 else ConcatDataset(ds_list)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(split == "train"),
        num_workers=config.num_workers,
        collate_fn=bah_collate_fn,
    )
    return dataset, loader
