from __future__ import annotations
import os
import logging
from typing import Dict, Any, List, Optional

import torch
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm

from .video_preprocessor import get_face_pixel_values
from src.utils.feature_store import FeatureStore, build_cache_key, need_full_reextract, merge_missing


class BAHFaceDataset(Dataset):
    """
    Independent video samples.
    CSV columns: clip_name, absence_full, presence_full.
    Label is derived from absence_full / presence_full.

    Video path:
      - if clip_name is an absolute path, use it as-is
      - otherwise: <video_dir>/<clip_name>
    """

    def __init__(
        self,
        csv_path: str,
        video_dir: str,
        config,
        split: str,
        modality_processors: Dict[str, Any],
        modality_feature_extractors: Dict[str, Any],
        dataset_name: str = "bah",
        device: str = "cuda",
    ) -> None:
        super().__init__()

        self.csv_path = csv_path
        self.video_dir = video_dir
        self.config = config
        self.split = split
        self.dataset_name = dataset_name
        self.device = device

        # extraction params
        self.segment_length = config.segment_length
        self.subset_size = config.subset_size
        self.average_features = config.average_features  # 'raw'|'mean'|'mean_std'
        self.yolo_weights = config.yolo_weights
        self.video_mode = config.video_mode

        # processors/extractors (face only)
        self.proc = modality_processors.get("face", None)
        self.extr = modality_feature_extractors.get("face", None)
        if self.proc is None:
            raise ValueError("Missing image processor for 'face'.")
        if self.extr is None:
            raise ValueError("Missing feature extractor for 'face'.")

        # cache
        self.save_prepared_data = config.save_prepared_data
        self.save_feature_path = config.save_feature_path
        self.store = FeatureStore(self.save_feature_path)

        # CSV
        df = pd.read_csv(self.csv_path)
        required = {"video_path", "label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV must contain columns {sorted(required)}. Missing: {sorted(missing)}"
            )

        if self.subset_size > 0:
            df = df.head(self.subset_size)
        logging.info(
            f"[BAHFaceDataset] {self.dataset_name}/{self.split}: "
            f"subset_size={self.subset_size} -> rows={len(df)}"
        )
        self.df = df

        # meta (paths + labels)
        self.meta: List[Dict[str, Any]] = []
        self._meta_rebuilt = False
        if self.save_prepared_data:
            self.meta = self.store.load_meta(
                self.dataset_name, self.split, getattr(self.config, "random_seed", 0), self.subset_size
            )
            if self.meta and len(self.meta) != len(self.df):
                logging.warning(
                    f"[BAHFaceDataset] cached meta size {len(self.meta)} does not match "
                    f"CSV rows {len(self.df)} for {self.dataset_name}/{self.split}. "
                    "Rebuilding meta and refreshing cache."
                )
                self.meta = []
                self._meta_rebuilt = True
        if not self.meta:
            self._build_meta_only()
            if self.save_prepared_data:
                self.store.save_meta(
                    self.dataset_name, self.split, getattr(self.config, "random_seed", 0),
                    self.subset_size, self.meta
                )

        # prepare / fill feature cache
        self._prepare_face_cache()

    @staticmethod
    def _video_path(base_dir: str, video_name: str) -> str:
        if os.path.isabs(video_name) and os.path.exists(video_name):
            return video_name

        name = str(video_name)
        p = os.path.join(base_dir, name)
        if os.path.exists(p):
            return p

        root, ext = os.path.splitext(name)
        if not ext:
            for e in (".mp4", ".avi", ".mov", ".mkv"):
                cand = os.path.join(base_dir, root + e)
                if os.path.exists(cand):
                    return cand

        raise FileNotFoundError(f"Expected video at: {p}")

    @staticmethod
    def _to_float(raw: Any) -> float:
        try:
            return float(raw)
        except Exception:
            s = str(raw).strip().lower()
            if s in {"1", "true", "yes", "present", "presence", "ambivalent", "ambivalence"}:
                return 1.0
            if s in {"0", "false", "no", "absent", "absence"}:
                return 0.0
        return float("nan")

    def _label_from_row(self, row: Any) -> int:
        a = self._to_float(row.get("absence_full"))
        p = self._to_float(row.get("presence_full"))
        if pd.isna(a):
            a = 0.0
        if pd.isna(p):
            p = 0.0
        if p > a:
            return 1
        if a > p:
            return 0
        return 1 if p > 0 else 0

    def _build_meta_only(self) -> None:
        self.meta = []
        for _, row in tqdm(self.df.iterrows(), total=len(self.df),
                           desc=f"Indexing BAH videos [{self.dataset_name}/{self.split}]"):
            vid = str(row["video_path"])
            vpath = self._video_path(self.video_dir, vid)

            # class_id = self._label_from_row(row)
            class_id = row.get("label")
            sample_name = os.path.splitext(os.path.basename(vid))[0]

            self.meta.append({
                "sample_name": sample_name,
                "video_path": vpath,
                "label": int(class_id),
            })

        logging.info(
            f"[BAHFaceDataset] {self.dataset_name}/{self.split}: "
            f"indexed segments={len(self.meta)} / rows={len(self.df)}"
        )

    # feature caching
    def _prepare_face_cache(self) -> None:
        if not self.meta:
            return
        sample_names = [m["sample_name"] for m in self.meta]

        mod = "face"
        ex = self.extr

        key = build_cache_key(mod, ex, self.config)
        store, header = self.store.load_modality_store(
            self.dataset_name, self.split, key, getattr(self.config, "random_seed", 0), self.subset_size
        )
        if self._meta_rebuilt:
            # Meta changed -> ignore old cache to avoid mixing mismatched entries.
            store = {}
            header = None
        if need_full_reextract(self.config, mod, header, key):
            store = {}

        missing = merge_missing(store, sample_names)
        if missing and self.average_features != "raw":
            raw_key = build_cache_key(mod, ex, self.config, avg_override="raw")
            raw_store, raw_header = self.store.load_modality_store(
                self.dataset_name, self.split, raw_key, getattr(self.config, "random_seed", 0), self.subset_size
            )
            if not need_full_reextract(self.config, mod, raw_header, raw_key):
                filled = 0
                for name in list(missing):
                    raw_feats = raw_store.get(name, None)
                    if raw_feats is None:
                        continue
                    agg = self._aggregate_cached_raw(raw_feats, self.average_features)
                    if agg is not None:
                        store[name] = agg
                        filled += 1
                if filled > 0:
                    logging.info(
                        f"[BAHFaceDataset] reused raw cache -> {self.average_features}: {filled} samples"
                    )
                missing = merge_missing(store, sample_names)

        if not missing:
            # Save aggregated cache derived from raw so __getitem__ can load it.
            self.store.save_modality_store(
                self.dataset_name, self.split, key, getattr(self.config, "random_seed", 0), self.subset_size, store
            )
            return

        path_by_name = {m["sample_name"]: m["video_path"] for m in self.meta}

        for name in tqdm(
            missing,
            desc=f"Extracting {mod} [{self.dataset_name}/{self.split}]",
            leave=True
        ):
            try:
                vpath = path_by_name.get(name)
                if not vpath:
                    store[name] = None
                    continue

                _, face_pv = get_face_pixel_values(
                    video_path=vpath,
                    segment_length=self.segment_length,
                    image_processor=self.proc,
                    device=self.device,
                    yolo_weights=self.yolo_weights,
                    mode=self.video_mode,
                )

                feats = ex.extract(pixel_values=face_pv) if face_pv is not None else None
                feats = self._aggregate(feats, self.average_features) if feats is not None else None
                store[name] = feats

            except Exception as e:
                logging.warning(f"{mod} extract error {name}: {e}")
                store[name] = None

        self.store.save_modality_store(
            self.dataset_name, self.split, key, getattr(self.config, "random_seed", 0), self.subset_size, store
        )
        torch.cuda.empty_cache()

    def _aggregate_cached_raw(self, feats: Any, average: str) -> Optional[dict]:
        """
        Aggregate cached raw features without re-extracting.
        Cached raw format: {'seq': Tensor [T,D]}.
        """
        if feats is None:
            return None
        if average == "raw":
            return feats if isinstance(feats, dict) else None

        if isinstance(feats, dict):
            emb = feats.get("seq", None)
            if emb is None:
                emb = feats.get("embedding", None)
        else:
            emb = None

        if emb is None or not isinstance(emb, torch.Tensor):
            return None
        if emb.ndim == 1:
            emb = emb.unsqueeze(0)

        if average == "mean_std":
            return {"mean": emb.mean(dim=0), "std": emb.std(dim=0, unbiased=False)}
        if average == "mean":
            return {"mean": emb.mean(dim=0)}
        return {"seq": emb}

    def _aggregate(self, feats: Any, average: str) -> Optional[dict]:
        """
        feats: {'embedding': Tensor [T,D] or [D]}
        'raw'      -> {'seq': [T,D]}
        'mean'     -> {'mean': [D]}
        'mean_std' -> {'mean':[D],'std':[D]}
        """
        if not isinstance(feats, dict):
            raise TypeError(f"Expected dict with key 'embedding', got {type(feats)}")
        emb = feats.get("embedding", None)
        if emb is None or not isinstance(emb, torch.Tensor):
            raise TypeError(f"Features dict must contain 'embedding' Tensor, got keys {list(feats.keys())}")

        if emb.ndim == 1:
            emb = emb.unsqueeze(0)  # [1,D]

        if average == "mean_std":
            return {"mean": emb.mean(dim=0), "std": emb.std(dim=0, unbiased=False)}
        if average == "mean":
            return {"mean": emb.mean(dim=0)}
        return {"seq": emb}

    # dataset API
    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base = self.meta[idx]
        name = base["sample_name"]

        features = {}
        key = build_cache_key("face", self.extr, self.config)
        cache = self.store.get_store(
            self.dataset_name, self.split, key, getattr(self.config, "random_seed", 0), self.subset_size
        )
        features["face"] = cache.get(name, None)
        label_idx = int(base["label"])
        return {
            "sample_name": name,
            "video_path": base["video_path"],
            "label": torch.tensor(label_idx, dtype=torch.long),
            "features": features,
        }
