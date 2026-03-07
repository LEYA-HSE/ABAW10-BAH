# coding: utf-8
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.utils.config_loader import ConfigLoader
from src.utils.logger_setup import setup_logger
from src.data_loading.video_preprocessor import get_face_pixel_values
from src.data_loading.pretrained_extractors import (
    build_extractors_from_config,
    AffectNetImageProcessor,
)
from src.models.models import VectorMLP


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


def _pick_video_column(df: pd.DataFrame, csv_path: Path) -> str:
    for column in ("video_path", "video_name"):
        if column in df.columns:
            return column
    raise KeyError(f"CSV '{csv_path}' must contain 'video_path' or 'video_name'")


def _resolve_dataset_path(template_or_path: str, base_dir: str, split: str | None = None) -> Path:
    return Path(str(template_or_path).format(base_dir=base_dir, split=split or ""))


def aggregate_for_vector(embedding: torch.Tensor, average_mode: str) -> torch.Tensor:
    if embedding.ndim == 1:
        embedding = embedding.unsqueeze(0)
    mode = average_mode.lower()
    if mode == "mean_std":
        mean = embedding.mean(dim=0)
        std = embedding.std(dim=0, unbiased=False)
        return torch.cat([mean, std], dim=0)
    return embedding.mean(dim=0)


def _resolve_video_path(base_dir: str, raw_video_path: str) -> Path:
    raw = str(raw_video_path).strip()
    candidate = Path(raw)
    if candidate.is_file():
        return candidate

    joined = Path(base_dir) / raw
    if joined.is_file():
        return joined

    stem = Path(raw).stem
    for ext in VIDEO_EXTS:
        ext_candidate = Path(base_dir) / f"{stem}{ext}"
        if ext_candidate.is_file():
            return ext_candidate

    raise FileNotFoundError(f"Video not found for '{raw_video_path}' under '{base_dir}'")


def _iter_split_videos(cfg, split: str) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []

    for _, ds_cfg in getattr(cfg, "datasets", {}).items():
        csv_tpl = ds_cfg.get("csv_path")
        base_dir = ds_cfg.get("base_dir")
        video_dir_tpl = ds_cfg.get("video_dir", base_dir)
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
            resolved = _resolve_video_path(str(video_dir), raw_video_path)
            items.append((sample_id, resolved))

    if not items:
        raise ValueError(f"No videos found for split='{split}' from config datasets")

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


def _load_face_pipeline(cfg):
    device = torch.device(cfg.device if str(cfg.device).startswith("cuda") and torch.cuda.is_available() else "cpu")

    weights_path = Path(cfg.face_classifier_weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Face classifier weights not found: {weights_path}")

    face_processor = AffectNetImageProcessor(image_size=cfg.affectnet_image_size)
    face_extractor = build_extractors_from_config(cfg)["face"]

    state = torch.load(str(weights_path), map_location=device)
    w1 = state["feature_extractor.0.weight"]
    w2 = state["feature_extractor.4.weight"]
    wc = state["classifier.weight"]

    model = VectorMLP(
        input_dim=int(w1.shape[1]),
        hidden_dim=int(w1.shape[0]),
        out_features=int(w2.shape[0]),
        num_classes=int(wc.shape[0]),
        dropout=0.0,
    ).to(device).eval()
    model.load_state_dict(state, strict=True)

    return device, face_processor, face_extractor, model


def _compute_artifact(cfg, sample_key: str, vp: Path, device, face_processor, face_extractor, model):
    _, face_pv = get_face_pixel_values(
        video_path=str(vp),
        segment_length=int(cfg.segment_length),
        image_processor=face_processor,
        device=str(device),
        yolo_weights=str(cfg.yolo_weights),
        mode=str(cfg.video_mode),
    )

    extr_out = face_extractor.extract(pixel_values=face_pv)
    embedding_seq = extr_out["embedding"]
    x = aggregate_for_vector(embedding_seq, cfg.average_features).unsqueeze(0).to(device)

    model_out = model(x, features=True)
    logits = model_out["prob"].squeeze(0)
    embeddings = model_out["features"].squeeze(0)
    prob = F.softmax(logits, dim=-1)

    return {
        "name": sample_key,
        "prob": prob.detach().cpu().numpy().astype("float32"),
        "logits": logits.detach().cpu().numpy().astype("float32"),
        "embeddings": embeddings.detach().cpu().numpy().astype("float32"),
    }


def _export_items(cfg, items: list[tuple[str, Path]], out_path: Path) -> None:
    device, face_processor, face_extractor, model = _load_face_pipeline(cfg)

    out = _load_existing_artifacts(out_path, overwrite=cfg.face_export_overwrite_cache)
    cache_hits = 0
    cache_misses = 0

    for sample_key, vp in tqdm(items, desc=f"Face export -> {out_path.name}"):
        if sample_key in out:
            cache_hits += 1
            continue

        artifact = _compute_artifact(cfg, sample_key, vp, device, face_processor, face_extractor, model)
        out[sample_key] = artifact
        cache_misses += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info("Saved %d face artifacts to %s", len(out), out_path)
    logging.info("Face export reuse: hits=%d misses=%d file=%s", cache_hits, cache_misses, out_path)


def run_face_export(
    config_path: str = "config.toml",
    *,
    configure_logging: bool = True,
    splits: list[str] | None = None,
) -> None:
    if configure_logging:
        setup_logger(logging.INFO)
    cfg = ConfigLoader(config_path)
    cfg.show_config()

    out_dir = Path(cfg.face_export_output_dir)
    export_splits = list(splits) if splits is not None else list(cfg.face_export_splits)
    for split in export_splits:
        items = _iter_split_videos(cfg, split)
        _export_items(cfg, items, out_dir / f"{split}.pkl")
