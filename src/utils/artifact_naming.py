# coding: utf-8
from __future__ import annotations

import re
from pathlib import Path

from src.utils.text_impl import (
    default_text_checkpoint_for_impl,
    default_text_hf_model_for_impl,
    normalize_text_impl,
)


def _path_stem(value: str) -> str:
    text = str(value).strip()
    if not text:
        return "none"
    return Path(text).stem or "none"


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    text = re.sub(r"-{2,}", "-", text).strip("-._")
    return text or "none"


def build_face_artifact_tag(cfg) -> str:
    parts = [
        getattr(cfg, "video_extractor", "off"),
        _path_stem(getattr(cfg, "affectnet_ckpt_path", "")),
        _path_stem(getattr(cfg, "face_classifier_weights", "")),
        getattr(cfg, "average_features", "mean"),
        getattr(cfg, "video_mode", "stable"),
        f"seg{getattr(cfg, 'segment_length', 0)}",
        _path_stem(getattr(cfg, "yolo_weights", "")),
    ]
    return "__".join(_slug(part) for part in parts)


def build_audio_artifact_tag(cfg) -> str:
    parts = [
        getattr(cfg, "audio_wav2vec_model", "off"),
        _path_stem(getattr(cfg, "audio_checkpoint_path", "")),
        f"layer{getattr(cfg, 'audio_hidden_state_index', 10)}",
        f"sr{getattr(cfg, 'audio_sample_rate', 16000)}",
    ]
    return "__".join(_slug(part) for part in parts)


def build_scene_artifact_tag(cfg) -> str:
    parts = [
        getattr(cfg, "scene_model_name", "off"),
        _path_stem(getattr(cfg, "scene_checkpoint_path", "")),
        _path_stem(getattr(cfg, "scene_preprocessor_dir", "")),
        f"frames{getattr(cfg, 'scene_num_frames', 16)}",
    ]
    return "__".join(_slug(part) for part in parts)


def build_text_artifact_tag(cfg) -> str:
    impl = normalize_text_impl(str(getattr(cfg, "text_export_impl", "7")))
    ckpt = str(getattr(cfg, "text_checkpoint_path_resolved", "")).strip() or default_text_checkpoint_for_impl(impl)
    hf_model = str(getattr(cfg, "text_hf_model_name_resolved", "")).strip() or default_text_hf_model_for_impl(impl)

    parts = [
        f"impl{impl}",
        _path_stem(ckpt),
        hf_model or "none",
        getattr(cfg, "text_input_column", "auto"),
        f"len{getattr(cfg, 'text_max_length', 256)}",
    ]
    return "__".join(_slug(part) for part in parts)
