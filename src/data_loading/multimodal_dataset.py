from __future__ import annotations

import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


PROB_KEYS = ("prob", "probs", "final_prob", "cls_prob", "proto_prob")
LOGIT_KEYS = ("logits", "logit")
EMB_KEYS = ("embeddings", "embedding", "emb")


def _sample_id_from_value(raw_value: Any) -> str:
    text = str(raw_value).strip()
    return Path(text).stem


def _to_tensor(raw_value: Any) -> torch.Tensor | None:
    if raw_value is None:
        return None
    try:
        tensor = torch.as_tensor(raw_value, dtype=torch.float32)
    except Exception:
        return None
    if tensor.numel() == 0:
        return None
    return tensor.detach().cpu().reshape(-1)


def _first_tensor(payload: Dict[str, Any], keys: Iterable[str]) -> torch.Tensor | None:
    for key in keys:
        if key in payload:
            tensor = _to_tensor(payload[key])
            if tensor is not None:
                return tensor
    return None


def _load_pickle(path: str) -> Dict[str, Any]:
    _ensure_numpy_pickle_compat()
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict in artifact '{path}', got {type(data)}")
    return data


def _ensure_numpy_pickle_compat() -> None:
    """
    Compatibility bridge for pickles produced with different NumPy internals
    (e.g. numpy._core.* vs numpy.core.* module paths).
    """
    try:
        import numpy as np  # noqa: F401
        import numpy.core as np_core  # type: ignore
    except Exception:
        return

    sys.modules.setdefault("numpy._core", np_core)
    if hasattr(np_core, "numeric"):
        sys.modules.setdefault("numpy._core.numeric", np_core.numeric)  # type: ignore[attr-defined]
    if hasattr(np_core, "multiarray"):
        sys.modules.setdefault("numpy._core.multiarray", np_core.multiarray)  # type: ignore[attr-defined]


def _resolve_artifact_path(source_cfg: Any, split: str) -> str:
    if isinstance(source_cfg, str):
        return source_cfg.format(split=split)
    if not isinstance(source_cfg, dict):
        raise TypeError(f"Unsupported multimodal source config: {type(source_cfg)}")
    candidate = source_cfg.get(split, source_cfg.get("path"))
    if not candidate:
        raise KeyError(f"Missing artifact path for split='{split}' in multimodal source config")
    return str(candidate).format(split=split)


def _default_artifact_path(config, modality: str, split: str) -> str:
    artifacts_dir = getattr(config, "multimodal_artifacts_dir", "./features")
    if modality == "face" and getattr(config, "face_artifact_tag", ""):
        return str(Path(artifacts_dir) / modality / config.face_artifact_tag / f"{split}.pkl")
    if modality == "audio" and getattr(config, "audio_artifact_tag", ""):
        return str(Path(artifacts_dir) / modality / config.audio_artifact_tag / f"{split}.pkl")
    if modality == "text" and getattr(config, "text_artifact_tag", ""):
        return str(Path(artifacts_dir) / modality / config.text_artifact_tag / f"{split}.pkl")
    if modality == "scene" and getattr(config, "scene_artifact_tag", ""):
        return str(Path(artifacts_dir) / modality / config.scene_artifact_tag / f"{split}.pkl")
    return str(Path(artifacts_dir) / modality / f"{split}.pkl")


def _source_kind(source_cfg: Any) -> str:
    if isinstance(source_cfg, dict):
        return str(source_cfg.get("kind", "artifact")).lower()
    return "artifact"


def _pick_video_column(df: pd.DataFrame) -> str | None:
    for column in ("video_path", "video_name"):
        if column in df.columns:
            return column
    return None


def _build_label_map(config, split: str) -> Dict[str, int]:
    label_map: Dict[str, int] = {}

    for ds_name, ds_cfg in getattr(config, "datasets", {}).items():
        csv_tpl = ds_cfg.get("csv_path")
        base_dir = ds_cfg.get("base_dir")
        if not csv_tpl or not base_dir:
            continue

        csv_path = csv_tpl.format(base_dir=base_dir, split=split)
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        video_column = _pick_video_column(df)
        if video_column is None or "label" not in df.columns:
            logging.warning(
                "[MultimodalDataset] skip labels from %s: expected columns 'video_path' or 'video_name', and 'label'",
                csv_path,
            )
            continue

        for _, row in df.iterrows():
            sample_id = _sample_id_from_value(row[video_column])
            label = int(row["label"])
            prev = label_map.get(sample_id)
            if prev is not None and prev != label:
                logging.warning(
                    "[MultimodalDataset] conflicting labels for sample '%s': %s vs %s",
                    sample_id,
                    prev,
                    label,
                )
            label_map[sample_id] = label

    return label_map


def _canonicalize_entry(raw_key: str, payload: Any) -> Dict[str, Any]:
    payload = payload if isinstance(payload, dict) else {}
    sample_name = payload.get("name", raw_key)
    sample_id = _sample_id_from_value(sample_name if sample_name else raw_key)

    prob = _first_tensor(payload, PROB_KEYS)
    logits = _first_tensor(payload, LOGIT_KEYS)
    emb = _first_tensor(payload, EMB_KEYS)
    present = any(value is not None for value in (prob, logits, emb))

    return {
        "sample_id": sample_id,
        "prob": prob,
        "logits": logits,
        "emb": emb,
        "present": present,
    }


class MultimodalArtifactsDataset(Dataset):
    def __init__(self, config, split: str) -> None:
        super().__init__()
        self.config = config
        self.split = split
        self.modalities: List[str] = list(getattr(config, "multimodal_modalities", []))
        if not self.modalities:
            raise ValueError("multimodal.modalities must contain at least one modality")

        self.min_modalities = 1
        self.sources = getattr(config, "multimodal_sources", {})
        if not isinstance(self.sources, dict):
            raise ValueError("multimodal.sources must be a dict when provided")

        self.label_map = _build_label_map(config, split)
        if not self.label_map:
            logging.warning("[MultimodalDataset] no labels found for split='%s'", split)

        self.entries_by_modality: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.modality_dims = {
            "prob": {},
            "logits": {},
            "emb": {},
        }

        for modality in self.modalities:
            source_cfg = self.sources.get(modality, {})
            source_kind = _source_kind(source_cfg)
            if source_kind not in ("artifact", ""):
                raise ValueError(
                    f"Unsupported source kind '{source_kind}' for modality '{modality}'. "
                    "Use 'artifact'."
                )

            artifact_path = (
                _resolve_artifact_path(source_cfg, split)
                if source_cfg
                else _default_artifact_path(self.config, modality, split)
            )
            if not os.path.exists(artifact_path):
                raise FileNotFoundError(
                    f"Artifact for modality '{modality}' and split '{split}' not found: {artifact_path}. "
                    f"Generate it first with the corresponding exporter "
                    f"(for face: from src.exporters import run_face_export; run_face_export())."
                )

            raw_entries = _load_pickle(artifact_path)
            canonical: Dict[str, Dict[str, Any]] = {}
            for raw_key, payload in raw_entries.items():
                entry = _canonicalize_entry(str(raw_key), payload)
                if not entry["present"]:
                    continue
                canonical[entry["sample_id"]] = entry

            if not canonical:
                raise ValueError(f"Artifact '{artifact_path}' does not contain usable entries")

            self.entries_by_modality[modality] = canonical
            self.modality_dims["prob"][modality] = self._infer_dim(canonical, "prob")
            self.modality_dims["logits"][modality] = self._infer_dim(canonical, "logits")
            self.modality_dims["emb"][modality] = self._infer_dim(canonical, "emb")

        self.samples = self._build_samples()
        if not self.samples:
            raise ValueError(f"No multimodal samples available for split='{split}'")

        logging.info(
            "[MultimodalDataset] split=%s samples=%d min_modalities=%d modalities=%s",
            self.split,
            len(self.samples),
            self.min_modalities,
            ",".join(self.modalities),
        )
        for modality in self.modalities:
            covered = sum(1 for sample in self.samples if sample["mask"][modality])
            logging.info(
                "[MultimodalDataset] split=%s modality=%s coverage=%d/%d prob_dim=%d logits_dim=%d emb_dim=%d",
                self.split,
                modality,
                covered,
                len(self.samples),
                self.modality_dims["prob"][modality],
                self.modality_dims["logits"][modality],
                self.modality_dims["emb"][modality],
            )

    @staticmethod
    def _infer_dim(entries: Dict[str, Dict[str, Any]], field: str) -> int:
        for payload in entries.values():
            tensor = payload.get(field)
            if tensor is not None:
                return int(tensor.numel())
        return 0

    def _zeros(self, field: str, modality: str) -> torch.Tensor:
        return torch.zeros(self.modality_dims[field][modality], dtype=torch.float32)

    def _build_samples(self) -> List[Dict[str, Any]]:
        sample_ids = set(self.label_map.keys())
        if not sample_ids:
            for entries in self.entries_by_modality.values():
                sample_ids.update(entries.keys())

        samples: List[Dict[str, Any]] = []
        for sample_id in sorted(sample_ids):
            active = {
                modality: sample_id in self.entries_by_modality.get(modality, {})
                for modality in self.modalities
            }
            if sum(active.values()) < self.min_modalities:
                continue

            if sample_id not in self.label_map:
                logging.warning(
                    "[MultimodalDataset] sample '%s' has modality artifacts but no label in split '%s'",
                    sample_id,
                    self.split,
                )
                continue

            samples.append(
                {
                    "sample_id": sample_id,
                    "label": int(self.label_map[sample_id]),
                    "mask": active,
                }
            )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        meta = self.samples[idx]
        item = {
            "sample_id": meta["sample_id"],
            "label": torch.tensor(meta["label"], dtype=torch.long),
            "emb": {},
            "prob": {},
            "logits": {},
            "mask": {},
        }

        for modality in self.modalities:
            entry = self.entries_by_modality[modality].get(meta["sample_id"])
            exists = entry is not None
            item["mask"][modality] = torch.tensor(exists, dtype=torch.bool)

            if self.modality_dims["emb"][modality] > 0:
                item["emb"][modality] = (
                    entry["emb"].to(torch.float32) if exists and entry.get("emb") is not None else self._zeros("emb", modality)
                )
            if self.modality_dims["prob"][modality] > 0:
                item["prob"][modality] = (
                    entry["prob"].to(torch.float32) if exists and entry.get("prob") is not None else self._zeros("prob", modality)
                )
            if self.modality_dims["logits"][modality] > 0:
                item["logits"][modality] = (
                    entry["logits"].to(torch.float32)
                    if exists and entry.get("logits") is not None
                    else self._zeros("logits", modality)
                )

        return item


def multimodal_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    out = {
        "sample_ids": [item["sample_id"] for item in batch],
        "labels": torch.stack([item["label"] for item in batch], dim=0),
        "emb": {},
        "prob": {},
        "logits": {},
        "mask": {},
    }

    for field in ("emb", "prob", "logits", "mask"):
        modalities = sorted({mod for item in batch for mod in item[field].keys()})
        for modality in modalities:
            out[field][modality] = torch.stack([item[field][modality] for item in batch], dim=0)

    return out


def make_multimodal_dataset_and_loader(config, split: str) -> Tuple[MultimodalArtifactsDataset, DataLoader]:
    dataset = MultimodalArtifactsDataset(config=config, split=split)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(split == "train" and bool(getattr(config, "shuffle", True))),
        num_workers=config.num_workers,
        collate_fn=multimodal_collate_fn,
    )
    return dataset, loader
