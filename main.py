# main.py
# coding: utf-8
import logging
import os
import shutil
import datetime
from pathlib import Path

import toml
from src.utils.config_loader import ConfigLoader
from src.utils.logger_setup import setup_logger
from src.utils.search_utils import greedy_search, exhaustive_search, optuna_search, write_single_run_overrides
from src.utils.telegram_utils import notify_telegram
from src.data_loading.multimodal_dataset import make_multimodal_dataset_and_loader
from src.data_loading.multimodal_runtime import any_split_exists, log_multimodal_batch_stats
from src.exporters import run_audio_export, run_face_export, run_scene_export, run_text_export
from src.train import train

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def _resolve_face_artifact_path(base_config, split: str) -> Path:
    face_source = getattr(base_config, "multimodal_sources", {}).get("face")
    if face_source is None:
        return (
            Path(base_config.multimodal_artifacts_dir)
            / "face"
            / base_config.face_artifact_tag
            / f"{split}.pkl"
        )

    if isinstance(face_source, str):
        return Path(str(face_source).format(split=split))
    if isinstance(face_source, dict):
        raw_path = face_source.get(split, face_source.get("path"))
        if not raw_path:
            raise KeyError(f"Face artifact path is not configured for split='{split}'")
        return Path(str(raw_path).format(split=split))
    raise TypeError(f"Unsupported face source config type: {type(face_source)}")


def _resolve_audio_artifact_path(base_config, split: str) -> Path:
    audio_source = getattr(base_config, "multimodal_sources", {}).get("audio")
    if audio_source is None:
        return (
            Path(base_config.multimodal_artifacts_dir)
            / "audio"
            / base_config.audio_artifact_tag
            / f"{split}.pkl"
        )

    if isinstance(audio_source, str):
        return Path(str(audio_source).format(split=split))
    if isinstance(audio_source, dict):
        raw_path = audio_source.get(split, audio_source.get("path"))
        if not raw_path:
            raise KeyError(f"Audio artifact path is not configured for split='{split}'")
        return Path(str(raw_path).format(split=split))
    raise TypeError(f"Unsupported audio source config type: {type(audio_source)}")


def _resolve_scene_artifact_path(base_config, split: str) -> Path:
    scene_source = getattr(base_config, "multimodal_sources", {}).get("scene")
    if scene_source is None:
        return (
            Path(base_config.multimodal_artifacts_dir)
            / "scene"
            / base_config.scene_artifact_tag
            / f"{split}.pkl"
        )

    if isinstance(scene_source, str):
        return Path(str(scene_source).format(split=split))
    if isinstance(scene_source, dict):
        raw_path = scene_source.get(split, scene_source.get("path"))
        if not raw_path:
            raise KeyError(f"Scene artifact path is not configured for split='{split}'")
        return Path(str(raw_path).format(split=split))
    raise TypeError(f"Unsupported scene source config type: {type(scene_source)}")


def _resolve_text_artifact_path(base_config, split: str) -> Path:
    text_source = getattr(base_config, "multimodal_sources", {}).get("text")
    if text_source is None:
        return (
            Path(base_config.multimodal_artifacts_dir)
            / "text"
            / base_config.text_artifact_tag
            / f"{split}.pkl"
        )

    if isinstance(text_source, str):
        return Path(str(text_source).format(split=split))
    if isinstance(text_source, dict):
        raw_path = text_source.get(split, text_source.get("path"))
        if not raw_path:
            raise KeyError(f"Text artifact path is not configured for split='{split}'")
        return Path(str(raw_path).format(split=split))
    raise TypeError(f"Unsupported text source config type: {type(text_source)}")


def _required_pipeline_splits(base_config) -> list[str]:
    configured_splits = list(getattr(base_config, "export_splits", []))
    if not configured_splits:
        configured_splits = list(getattr(base_config, "face_export_splits", []))
    splits = ["train"]

    if "dev" in configured_splits and any_split_exists(base_config, "dev"):
        eval_split = "dev"
    elif "val" in configured_splits and any_split_exists(base_config, "val"):
        eval_split = "val"
    else:
        eval_split = "dev" if any_split_exists(base_config, "dev") else "val"

    splits.append(eval_split)
    splits.append("test" if any_split_exists(base_config, "test") else splits[-1])
    return list(dict.fromkeys(splits))


def _splits_to_process(required_splits: list[str], resolve_path_fn, force_reexport: bool) -> list[str]:
    if force_reexport:
        return list(required_splits)
    return [split for split in required_splits if not resolve_path_fn(split).exists()]


def _ensure_face_artifacts(base_config) -> None:
    if "face" not in getattr(base_config, "multimodal_modalities", []):
        return

    required_splits = _required_pipeline_splits(base_config)
    force_reexport = bool(getattr(base_config, "face_export_overwrite_cache", False))
    splits_to_process = _splits_to_process(
        required_splits,
        lambda split: _resolve_face_artifact_path(base_config, split),
        force_reexport,
    )
    if not splits_to_process:
        return

    if force_reexport:
        logging.info("Force face re-export enabled. Processing splits=%s", splits_to_process)
    else:
        logging.info("Missing face artifacts for splits=%s. Running face exporter automatically.", splits_to_process)
    run_face_export(
        config_path="config.toml",
        configure_logging=False,
        splits=splits_to_process,
    )

    still_missing = [
        split for split in required_splits
        if not _resolve_face_artifact_path(base_config, split).exists()
    ]
    if still_missing:
        raise FileNotFoundError(
            f"Face exporter finished, but artifacts are still missing for splits={still_missing}"
        )


def _ensure_audio_artifacts(base_config) -> None:
    if "audio" not in getattr(base_config, "multimodal_modalities", []):
        return

    required_splits = _required_pipeline_splits(base_config)
    force_reexport = bool(getattr(base_config, "audio_export_overwrite_cache", False))
    audio_source = str(getattr(base_config, "audio_export_source", "export")).lower()

    if audio_source == "precomputed":
        missing_splits = [
            split for split in required_splits if not _resolve_audio_artifact_path(base_config, split).exists()
        ]
        if force_reexport:
            logging.info("audio_export.overwrite_cache=true ignored for source='precomputed'")
        if not missing_splits:
            return
        raise FileNotFoundError(
            "Missing precomputed audio artifacts for splits="
            f"{missing_splits}. Expected path template: "
            f"{getattr(base_config, 'audio_precomputed_path', '<unset>')}"
        )

    splits_to_process = _splits_to_process(
        required_splits,
        lambda split: _resolve_audio_artifact_path(base_config, split),
        force_reexport,
    )
    if not splits_to_process:
        return

    if force_reexport:
        logging.info("Force audio re-export enabled. Processing splits=%s", splits_to_process)
    else:
        logging.info("Missing audio artifacts for splits=%s. Running audio exporter automatically.", splits_to_process)
    run_audio_export(
        config_path="config.toml",
        configure_logging=False,
        splits=splits_to_process,
    )

    still_missing = [
        split for split in required_splits
        if not _resolve_audio_artifact_path(base_config, split).exists()
    ]
    if still_missing:
        raise FileNotFoundError(
            f"Audio exporter finished, but artifacts are still missing for splits={still_missing}"
        )


def _ensure_scene_artifacts(base_config) -> None:
    if "scene" not in getattr(base_config, "multimodal_modalities", []):
        return

    required_splits = _required_pipeline_splits(base_config)
    force_reexport = bool(getattr(base_config, "scene_export_overwrite_cache", False))
    splits_to_process = _splits_to_process(
        required_splits,
        lambda split: _resolve_scene_artifact_path(base_config, split),
        force_reexport,
    )
    if not splits_to_process:
        return

    if force_reexport:
        logging.info("Force scene re-export enabled. Processing splits=%s", splits_to_process)
    else:
        logging.info("Missing scene artifacts for splits=%s. Running scene exporter automatically.", splits_to_process)
    run_scene_export(
        config_path="config.toml",
        configure_logging=False,
        splits=splits_to_process,
    )

    still_missing = [
        split for split in required_splits
        if not _resolve_scene_artifact_path(base_config, split).exists()
    ]
    if still_missing:
        raise FileNotFoundError(
            f"Scene exporter finished, but artifacts are still missing for splits={still_missing}"
        )


def _ensure_text_artifacts(base_config) -> None:
    if "text" not in getattr(base_config, "multimodal_modalities", []):
        return

    required_splits = _required_pipeline_splits(base_config)
    force_reexport = bool(getattr(base_config, "text_export_overwrite_cache", False))
    splits_to_process = _splits_to_process(
        required_splits,
        lambda split: _resolve_text_artifact_path(base_config, split),
        force_reexport,
    )
    if not splits_to_process:
        return

    if force_reexport:
        logging.info("Force text re-export enabled. Processing splits=%s", splits_to_process)
    else:
        logging.info("Missing text artifacts for splits=%s. Running text exporter automatically.", splits_to_process)
    run_text_export(
        config_path="config.toml",
        configure_logging=False,
        splits=splits_to_process,
    )

    still_missing = [
        split for split in required_splits
        if not _resolve_text_artifact_path(base_config, split).exists()
    ]
    if still_missing:
        raise FileNotFoundError(
            f"Text exporter finished, but artifacts are still missing for splits={still_missing}"
        )


def _run_multimodal_pipeline(base_config, results_dir: str, use_tg: bool) -> None:
    required_splits = _required_pipeline_splits(base_config)
    dev_split = next(split for split in required_splits if split in ("dev", "val"))
    _ensure_face_artifacts(base_config)
    _ensure_audio_artifacts(base_config)
    _ensure_text_artifacts(base_config)
    _ensure_scene_artifacts(base_config)

    logging.info("Loading multimodal artifacts (train/dev/test)...")
    _, train_loader = make_multimodal_dataset_and_loader(base_config, "train")
    _, dev_loader = make_multimodal_dataset_and_loader(base_config, dev_split)

    if any_split_exists(base_config, "test"):
        _, test_loader = make_multimodal_dataset_and_loader(base_config, "test")
    else:
        test_loader = dev_loader

    log_multimodal_batch_stats(train_loader)
    log_multimodal_batch_stats(dev_loader)
    if test_loader is not dev_loader:
        log_multimodal_batch_stats(test_loader)

    if base_config.prepare_only:
        logging.info("== prepare_only mode: multimodal artifacts validated, no training ==")
        notify_telegram(
            f"<b>multimodal_artifacts</b>: prepare_only completed\nresults: {results_dir}",
            enabled=use_tg,
        )
        return

    search_type = str(getattr(base_config, "search_type", "none")).lower()
    overrides_file = os.path.join(results_dir, "overrides.txt")
    if search_type == "none":
        logging.info("== Single training run (no hyperparameter search) ==")
        summary = train(
            base_config,
            train_loader=train_loader,
            dev_loader=dev_loader,
            test_loader=test_loader,
            results_dir=results_dir,
        )
        write_single_run_overrides(
            cfg=base_config,
            summary=summary,
            overrides_file=overrides_file,
        )
        best_score = summary.get("best_score", float("nan"))
        best_ckpt = summary.get("best_checkpoint", "")
        logging.info("Fusion training finished: best_score=%.4f checkpoint=%s", float(best_score), best_ckpt)
        notify_telegram(
            f"<b>fusion_train</b>: done\nMF1_AVG={float(best_score):.4f}\ncheckpoint: {best_ckpt}",
            enabled=use_tg,
        )
        return

    search_params_path = str(getattr(base_config, "search_params_path", "search_params.toml"))
    if not os.path.exists(search_params_path):
        raise FileNotFoundError(f"Search params file not found: {search_params_path}")
    search_cfg = toml.load(search_params_path)
    param_grid = dict(search_cfg.get("grid", {}))
    default_values = dict(search_cfg.get("defaults", {}))
    optuna_options = dict(search_cfg.get("optuna", {}))
    if search_type in {"greedy", "exhaustive"} and not param_grid:
        raise ValueError(f"No [grid] params found in {search_params_path}")

    runs_root = os.path.join(results_dir, "search_runs")
    os.makedirs(runs_root, exist_ok=True)
    logging.info(
        "Starting %s search: selection_metric=%s split=%s params=%s",
        search_type,
        getattr(base_config, "search_selection_metric", "MF1"),
        getattr(base_config, "search_early_stop_on", "avg"),
        list(param_grid.keys()),
    )

    if search_type == "greedy":
        search_result = greedy_search(
            base_config=base_config,
            train_loader=train_loader,
            dev_loader=dev_loader,
            test_loader=test_loader,
            train_fn=train,
            overrides_file=overrides_file,
            param_grid=param_grid,
            default_values=default_values,
            runs_root=runs_root,
        )
    elif search_type == "exhaustive":
        search_result = exhaustive_search(
            base_config=base_config,
            train_loader=train_loader,
            dev_loader=dev_loader,
            test_loader=test_loader,
            train_fn=train,
            overrides_file=overrides_file,
            param_grid=param_grid,
            default_values=default_values,
            runs_root=runs_root,
        )
    elif search_type == "optuna":
        search_result = optuna_search(
            base_config=base_config,
            train_loader=train_loader,
            dev_loader=dev_loader,
            test_loader=test_loader,
            train_fn=train,
            overrides_file=overrides_file,
            param_grid=param_grid,
            default_values=default_values,
            runs_root=runs_root,
            optuna_cfg=optuna_options,
        )
    else:
        raise ValueError(f"Unknown search.type='{search_type}'")

    best_score = float(search_result.get("best_score", float("nan")))
    best_params = search_result.get("best_params", {})
    logging.info("Search finished: best_score=%.4f best_params=%s", best_score, best_params)
    notify_telegram(
        f"<b>{search_type}_search</b>: done\nbest_score={best_score:.4f}\nresults: {results_dir}",
        enabled=use_tg,
    )


def main():
    base_config = ConfigLoader("config.toml")

    run_name = "multimodal_pipeline"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = f"results/results_{run_name}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    log_file = os.path.join(results_dir, "session_log.txt")
    setup_logger(logging.INFO, log_file=log_file)
    base_config.show_config()

    use_tg = base_config.use_telegram
    logging.info(
        f"use_telegram = {use_tg}  (env token={bool(os.getenv('TELEGRAM_BOT_TOKEN'))}, chat={bool(os.getenv('TELEGRAM_CHAT_ID'))})"
    )

    notify_telegram(f"Start: <b>{run_name}</b>\nresults: {results_dir}", enabled=use_tg)

    # Save config copy
    shutil.copy("config.toml", os.path.join(results_dir, "config_copy.toml"))

    _run_multimodal_pipeline(base_config, results_dir, use_tg)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        notify_telegram(
            f"Crash: <code>{type(e).__name__}</code>\n{e}",
            enabled=True
        )
        raise
