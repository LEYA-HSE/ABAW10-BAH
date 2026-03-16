# coding: utf-8
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

from src.utils.config_loader import ConfigLoader
from src.utils.logger_setup import setup_logger


def _import_decord():
    try:
        from decord import VideoReader, cpu  # type: ignore
        return VideoReader, cpu
    except Exception as exc:
        raise ImportError(
            "Scene exporter requires 'decord'. Install the dependency before enabling scene artifacts."
        ) from exc


class VideoPredictor:
    def __init__(self, model_path: str, config_dir: str, model_name: str, num_frames: int = 16):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_processor = VideoMAEImageProcessor.from_pretrained(config_dir)
        self.model = VideoMAEForVideoClassification.from_pretrained(
            model_name,
            num_labels=2,
            ignore_mismatched_sizes=True,
        )

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.num_frames = int(num_frames)

    def _get_video_frames(self, video_path: str):
        VideoReader, cpu = _import_decord()
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
        frames = vr.get_batch(indices).asnumpy()
        return list(frames)

    @torch.inference_mode()
    def predict(self, video_path: str):
        frames = self._get_video_frames(video_path)
        inputs = self.image_processor(frames, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs, output_hidden_states=True)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        last_hidden_state = outputs.hidden_states[-1]
        embeddings = last_hidden_state.mean(dim=1)

        return {
            "prob": probs.squeeze(0).detach().cpu().numpy().astype("float32"),
            "logits": logits.squeeze(0).detach().cpu().numpy().astype("float32"),
            "embeddings": embeddings.squeeze(0).detach().cpu().numpy().astype("float32"),
        }


def _pick_video_column(df: pd.DataFrame, csv_path: Path) -> str:
    for column in ("video_path", "video_name"):
        if column in df.columns:
            return column
    raise KeyError(f"CSV '{csv_path}' must contain 'video_path' or 'video_name'")


def _resolve_dataset_path(template_or_path: str, base_dir: str, split: str | None = None) -> Path:
    return Path(str(template_or_path).format(base_dir=base_dir, split=split or ""))


def _resolve_video_path(video_dir: str, raw_video_path: str, video_exts: tuple[str, ...]) -> Path:
    raw = str(raw_video_path).strip()
    candidate = Path(raw)
    if candidate.is_file():
        return candidate

    joined = Path(video_dir) / raw
    if joined.is_file():
        return joined

    stem = Path(raw).stem
    for ext in video_exts:
        ext_candidate = Path(video_dir) / f"{stem}{ext}"
        if ext_candidate.is_file():
            return ext_candidate

    raise FileNotFoundError(f"Video not found for '{raw_video_path}' under '{video_dir}'")


def _iter_split_videos(cfg, split: str) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    video_exts = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm")

    for _, ds_cfg in getattr(cfg, "datasets", {}).items():
        csv_tpl = ds_cfg.get("csv_path")
        base_dir = ds_cfg.get("base_dir")
        video_dir_tpl = ds_cfg.get("video_dir")
        if not csv_tpl or not base_dir or not video_dir_tpl:
            continue

        csv_path = Path(str(csv_tpl).format(base_dir=base_dir, split=split))
        video_dir = _resolve_dataset_path(str(video_dir_tpl), base_dir, split=split)
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        video_column = _pick_video_column(df, csv_path)
        for _, row in df.iterrows():
            raw_video_path = str(row[video_column])
            sample_id = Path(raw_video_path).stem
            resolved = _resolve_video_path(str(video_dir), raw_video_path, video_exts)
            items.append((sample_id, resolved))

    if not items:
        raise ValueError(f"No scene videos found for split='{split}' from config datasets")

    dedup: dict[str, Path] = {}
    for sample_id, path in items:
        dedup[sample_id] = path
    return sorted(dedup.items(), key=lambda item: item[0])


def _load_existing_artifacts(out_path: Path, overwrite: bool) -> dict[str, dict]:
    if overwrite or not out_path.exists():
        return {}
    with out_path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict in artifact '{out_path}', got {type(data)}")
    return data


def _export_split(cfg, split: str) -> None:
    out_path = Path(cfg.scene_export_output_dir) / f"{split}.pkl"
    out = _load_existing_artifacts(out_path, overwrite=cfg.scene_export_overwrite_cache)

    items = _iter_split_videos(cfg, split)
    pending = [(sample_id, path) for sample_id, path in items if sample_id not in out]
    if not pending:
        logging.info("Scene export split=%s is already up to date: %s", split, out_path)
        return

    if not str(cfg.scene_checkpoint_path).strip():
        raise ValueError("scene_export.checkpoint_path is empty")

    predictor = VideoPredictor(
        model_path=cfg.scene_checkpoint_path,
        config_dir=cfg.scene_preprocessor_dir,
        model_name=cfg.scene_model_name,
        num_frames=cfg.scene_num_frames,
    )
    for sample_id, video_path in tqdm(pending, desc=f"Scene export -> {split}.pkl"):
        out[sample_id] = predictor.predict(str(video_path))
        out[sample_id]["name"] = sample_id

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info("Saved %d scene artifacts to %s", len(out), out_path)


def run_scene_export(
    config_path: str = "config.toml",
    *,
    configure_logging: bool = True,
    splits: list[str] | None = None,
) -> None:
    if configure_logging:
        setup_logger(logging.INFO)
    cfg = ConfigLoader(config_path)
    cfg.show_config()

    export_splits = list(splits) if splits is not None else list(cfg.scene_export_splits)
    for split in export_splits:
        _export_split(cfg, split)
