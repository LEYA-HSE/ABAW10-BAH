# coding: utf-8
from __future__ import annotations

import logging
import pickle
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prepare_challenge_csv import prepare_challenge_csv
from src.exporters import run_face_export, run_scene_export, run_text_export
from src.utils.config_loader import ConfigLoader
from src.utils.logger_setup import setup_logger


CONFIG_PATH = "config.challenge.toml"
SPLIT = "test"
RUN_FACE = True
RUN_TEXT = True
RUN_SCENE = True
CHECK_AUDIO_PRECOMPUTED = True
MIRROR_AUDIO_PRECOMPUTED = True
AUDIO_PRECOMPUTED_SOURCE = "./features/audio/best_1/{split}.pkl"


def _audio_artifact_path(cfg: ConfigLoader, split: str) -> Path:
    source_cfg = getattr(cfg, "multimodal_sources", {}).get("audio")
    if isinstance(source_cfg, dict):
        raw = source_cfg.get(split, source_cfg.get("path"))
        if raw:
            return Path(str(raw).format(split=split))
    if isinstance(source_cfg, str) and source_cfg.strip():
        return Path(str(source_cfg).format(split=split))
    return Path(str(getattr(cfg, "audio_precomputed_path", "")).format(split=split))


def _load_pickle(path: Path):
    try:
        import numpy.core as np_core  # type: ignore

        sys.modules.setdefault("numpy._core", np_core)
        if hasattr(np_core, "numeric"):
            sys.modules.setdefault("numpy._core.numeric", np_core.numeric)  # type: ignore[attr-defined]
        if hasattr(np_core, "multiarray"):
            sys.modules.setdefault("numpy._core.multiarray", np_core.multiarray)  # type: ignore[attr-defined]
    except Exception:
        pass

    with path.open("rb") as handle:
        return pickle.load(handle)


def _ensure_audio_precomputed_available(cfg: ConfigLoader, split: str) -> Path:
    target_path = _audio_artifact_path(cfg, split)
    if target_path.exists():
        return target_path

    source_path = Path(str(AUDIO_PRECOMPUTED_SOURCE).format(split=split))
    if not source_path.exists():
        raise FileNotFoundError(
            f"Missing audio precomputed source: {source_path}. "
            f"Expected to mirror into challenge path: {target_path}"
        )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    logging.info("Copied audio precomputed: %s -> %s", source_path, target_path)
    return target_path


def _validate_audio_precomputed(cfg: ConfigLoader, split: str, csv_path: Path) -> None:
    audio_path = _audio_artifact_path(cfg, split)
    if not audio_path.exists():
        raise FileNotFoundError(f"Missing precomputed audio artifact: {audio_path}")

    data = _load_pickle(audio_path)
    if not isinstance(data, dict):
        raise TypeError(f"Audio artifact must be dict: {audio_path}")
    audio_ids = {str(k) for k in data.keys()}

    df = pd.read_csv(csv_path)
    if "video_path" not in df.columns:
        raise KeyError(f"{csv_path} must contain 'video_path'")
    csv_ids = {Path(str(v)).stem for v in df["video_path"].astype(str).tolist()}

    missing = sorted(csv_ids - audio_ids)
    extra = sorted(audio_ids - csv_ids)
    if missing or extra:
        raise ValueError(
            "Precomputed audio does not match challenge CSV: "
            f"missing={len(missing)} extra={len(extra)}"
        )
    logging.info("Audio precomputed OK: %s (samples=%d)", audio_path, len(audio_ids))


def run_challenge_feature_prepare(
    config_path: str = CONFIG_PATH,
    split: str = SPLIT,
) -> None:
    setup_logger(logging.INFO)

    csv_path = prepare_challenge_csv()
    logging.info("Challenge CSV ready: %s", csv_path)

    cfg = ConfigLoader(config_path)
    cfg.show_config()

    if MIRROR_AUDIO_PRECOMPUTED:
        _ensure_audio_precomputed_available(cfg, split=split)

    if CHECK_AUDIO_PRECOMPUTED:
        _validate_audio_precomputed(cfg, split=split, csv_path=csv_path)

    if RUN_FACE:
        run_face_export(config_path=config_path, configure_logging=False, splits=[split])
    if RUN_TEXT:
        run_text_export(config_path=config_path, configure_logging=False, splits=[split])
    if RUN_SCENE:
        run_scene_export(config_path=config_path, configure_logging=False, splits=[split])

    logging.info("Challenge feature preparation finished for split=%s", split)


if __name__ == "__main__":
    run_challenge_feature_prepare()
