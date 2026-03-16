# coding: utf-8
from __future__ import annotations

import json
import logging
import sys
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading.multimodal_dataset import make_multimodal_dataset_and_loader
from src.models.fusion_model import MultiModalModel
from src.utils.config_loader import ConfigLoader
from src.utils.logger_setup import setup_logger
from src.utils.measures import mf1_ah, uar_ah


CONFIG_PATH = "config.challenge.toml"
SPLIT = "test"
CHECKPOINT_PATH = "./results/proto_single/checkpoints/fusion_best_ep040_mf1avg_0.8325.pt"
CHECKPOINT_PATHS = [
    "./results/proto_1/checkpoints/fusion_best_ep015_mf1avg_0.8135.pt",
    "./results/proto_2/checkpoints/fusion_best_ep020_mf1avg_0.8148.pt",
    "./results/proto_3/checkpoints/fusion_best_ep016_mf1avg_0.8084.pt",
    "./results/proto_4/checkpoints/fusion_best_ep017_mf1avg_0.8224.pt",
    "./results/proto_5/checkpoints/fusion_best_ep012_mf1avg_0.8175.pt",
]

# CHECKPOINT_PATHS = [
#     "./results/model_1/checkpoints/fusion_best_ep010_mf1avg_0.8106.pt",
#     "./results/model_2/checkpoints/fusion_best_ep009_mf1avg_0.8199.pt",
#     "./results/model_3/checkpoints/fusion_best_ep014_mf1avg_0.8173.pt",
#     "./results/model_4/checkpoints/fusion_best_ep012_mf1avg_0.8051.pt",
#     "./results/model_5/checkpoints/fusion_best_ep011_mf1avg_0.8054.pt",
# ]
OUTPUT_ROOT = "./results/challenge_submissions"
RUN_TAG = "best_run"
RUN_MODE = "eval_metrics"  # challenge_submit | eval_metrics

EVAL_CONFIG_PATH = "config.toml"
EVAL_SPLITS = ["dev", "test"]
EVAL_OUTPUT_ROOT = "./results/eval_inference"
EVAL_TAG = "best_run"


def _resolve_device(device_str: str) -> torch.device:
    s = str(device_str).lower()
    if s.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if s == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {
        "labels": batch["labels"].to(device, non_blocking=True),
        "emb": {},
        "prob": {},
        "logits": {},
        "mask": {},
    }
    for field in ("emb", "prob", "logits", "mask"):
        for modality, tensor in batch.get(field, {}).items():
            out[field][modality] = tensor.to(device, non_blocking=True)
    return out


def _resolve_split_csv_path(cfg: ConfigLoader, split: str) -> Path:
    for _, ds_cfg in getattr(cfg, "datasets", {}).items():
        csv_tpl = ds_cfg.get("csv_path")
        base_dir = ds_cfg.get("base_dir")
        if not csv_tpl or not base_dir:
            continue
        csv_path = Path(str(csv_tpl).format(base_dir=base_dir, split=split))
        if csv_path.exists():
            return csv_path
    raise FileNotFoundError(f"Cannot resolve csv_path for split='{split}' from config datasets")


def _load_reference_video_order(cfg: ConfigLoader, split: str) -> List[str]:
    for _, ds_cfg in getattr(cfg, "datasets", {}).items():
        base_dir = ds_cfg.get("base_dir")
        if not base_dir:
            continue
        split_txt = Path(str(base_dir)) / "split" / f"{split}.txt"
        if split_txt.exists():
            order: List[str] = []
            with split_txt.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    video_path = line.rstrip("\n").split(",", 2)[0].strip()
                    order.append(video_path)
            if order:
                return order

    csv_path = _resolve_split_csv_path(cfg, split)
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "video_path" not in df.columns:
        raise KeyError(f"CSV '{csv_path}' must contain 'video_path' for challenge submission ordering")
    return [str(v) for v in df["video_path"].astype(str).tolist()]


def _load_model_and_cfg(checkpoint_path: Path, emb_dims: Dict[str, int], device: torch.device) -> Tuple[MultiModalModel, Dict]:
    state = torch.load(str(checkpoint_path), map_location=device)
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint must be dict: {checkpoint_path}")
    if "model_cfg" not in state or "model_state" not in state:
        raise KeyError(f"Checkpoint missing model_cfg/model_state: {checkpoint_path}")

    model_cfg = dict(state["model_cfg"])
    model = MultiModalModel(model_cfg, emb_dims=emb_dims).to(device)
    model.load_state_dict(state["model_state"], strict=True)
    model.eval()
    return model, model_cfg


def _resolve_checkpoint_paths(
    checkpoint_path: str | None = None,
    checkpoint_paths: List[str] | None = None,
) -> List[Path]:
    raw = list(checkpoint_paths or [])
    if not raw:
        raw = list(CHECKPOINT_PATHS)
    if not raw and checkpoint_path:
        raw = [checkpoint_path]
    out = [Path(str(p)) for p in raw]
    if not out:
        raise ValueError("No checkpoint paths provided")
    for ckpt in out:
        if not ckpt.exists():
            raise FileNotFoundError(f"Fusion checkpoint not found: {ckpt}")
    return out


def _load_models_and_cfgs(
    checkpoint_paths: List[Path],
    emb_dims: Dict[str, int],
    device: torch.device,
) -> Tuple[List[MultiModalModel], Dict]:
    models: List[MultiModalModel] = []
    cfg_ref: Dict | None = None
    for ckpt in checkpoint_paths:
        model, model_cfg = _load_model_and_cfg(ckpt, emb_dims=emb_dims, device=device)
        if cfg_ref is None:
            cfg_ref = dict(model_cfg)
        elif dict(model_cfg) != cfg_ref:
            logging.warning("Model cfg differs for checkpoint: %s", ckpt)
        models.append(model)
    return models, (cfg_ref or {})


def _predict(models: List[MultiModalModel], loader, device: torch.device) -> Tuple[Dict[str, Dict[str, float | int]], Dict[str, int]]:
    if not models:
        raise ValueError("No models loaded for prediction")
    pred_map: Dict[str, Dict[str, float | int]] = {}
    label_map: Dict[str, int] = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc="Challenge inference"):
            if batch is None:
                continue
            b = _move_batch_to_device(batch, device)
            probs_sum = None
            for model in models:
                logits = model(b)["logits"]
                probs = torch.softmax(logits, dim=1)
                probs_sum = probs if probs_sum is None else (probs_sum + probs)
            assert probs_sum is not None
            probs = probs_sum / float(len(models))
            preds = probs.argmax(dim=1)

            for sample_id, prob_vec, pred, label in zip(
                batch["sample_ids"],
                probs.detach().cpu(),
                preds.detach().cpu(),
                batch["labels"].detach().cpu(),
            ):
                p0 = float(prob_vec[0].item())
                p1 = float(prob_vec[1].item())
                pred_map[str(sample_id)] = {
                    "p0": p0,
                    "p1": p1,
                    "pred": int(pred.item()),
                }
                label_map[str(sample_id)] = int(label.item())
    return pred_map, label_map


def _validate_records(records: List[Tuple[str, float, float, int]]) -> None:
    for video_path, p0, p1, pred in records:
        if pred not in (0, 1):
            raise ValueError(f"Invalid pred={pred} for {video_path}")
        if not (0.0 <= p0 <= 1.0 and 0.0 <= p1 <= 1.0):
            raise ValueError(f"Invalid probs for {video_path}: p0={p0} p1={p1}")


def _write_submission_files(out_dir: Path, records: List[Tuple[str, float, float, int]]) -> Dict[str, str]:
    no_prob_path = out_dir / "no_probabilities" / "trial-0.txt"
    prob_path = out_dir / "with_probabilities" / "trial-0.txt"
    hard_prob_path = out_dir / "with_probabilities_hard" / "trial-0.txt"
    no_prob_csv_path = out_dir / "no_probabilities" / "trial-0.csv"
    prob_csv_path = out_dir / "with_probabilities" / "trial-0.csv"
    hard_prob_csv_path = out_dir / "with_probabilities_hard" / "trial-0.csv"

    no_prob_path.parent.mkdir(parents=True, exist_ok=True)
    prob_path.parent.mkdir(parents=True, exist_ok=True)
    hard_prob_path.parent.mkdir(parents=True, exist_ok=True)

    with no_prob_path.open("w", encoding="utf-8", newline="\n") as handle:
        for video_path, _, _, pred in records:
            handle.write(f"{video_path},{pred}\n")

    with no_prob_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_path", "pred"])
        for video_path, _, _, pred in records:
            writer.writerow([video_path, pred])

    with prob_path.open("w", encoding="utf-8", newline="\n") as handle:
        for video_path, p0, _, pred in records:
            # Keep validator-safe exact complement at 4 decimals.
            p0_4 = float(f"{p0:.4f}")
            p1_4 = float(f"{(1.0 - p0_4):.4f}")
            handle.write(f"{video_path},{p0_4:.4f},{p1_4:.4f},{pred}\n")

    with prob_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_path", "p0", "p1", "pred"])
        for video_path, p0, _, pred in records:
            p0_4 = float(f"{p0:.4f}")
            p1_4 = float(f"{(1.0 - p0_4):.4f}")
            writer.writerow([video_path, f"{p0_4:.4f}", f"{p1_4:.4f}", pred])

    with hard_prob_path.open("w", encoding="utf-8", newline="\n") as handle:
        for video_path, _, _, pred in records:
            if pred == 1:
                p0, p1 = 0.0, 1.0
            else:
                p0, p1 = 1.0, 0.0
            handle.write(f"{video_path},{p0:.1f},{p1:.1f},{pred}\n")

    with hard_prob_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_path", "p0", "p1", "pred"])
        for video_path, _, _, pred in records:
            if pred == 1:
                p0, p1 = 0.0, 1.0
            else:
                p0, p1 = 1.0, 0.0
            writer.writerow([video_path, f"{p0:.1f}", f"{p1:.1f}", pred])

    return {
        "no_probabilities": str(no_prob_path),
        "no_probabilities_csv": str(no_prob_csv_path),
        "with_probabilities": str(prob_path),
        "with_probabilities_csv": str(prob_csv_path),
        "with_probabilities_hard": str(hard_prob_path),
        "with_probabilities_hard_csv": str(hard_prob_csv_path),
    }


def run_challenge_inference(
    config_path: str = CONFIG_PATH,
    checkpoint_path: str = CHECKPOINT_PATH,
    checkpoint_paths: List[str] | None = None,
    split: str = SPLIT,
) -> Dict[str, str]:
    setup_logger(logging.INFO)
    cfg = ConfigLoader(config_path)
    cfg.show_config()

    ckpts = _resolve_checkpoint_paths(checkpoint_path=checkpoint_path, checkpoint_paths=checkpoint_paths)
    logging.info("Using %d checkpoint(s) for ensemble", len(ckpts))
    for i, ckpt in enumerate(ckpts, start=1):
        logging.info("  [%d] %s", i, ckpt)

    device = _resolve_device(str(getattr(cfg, "device", "cpu")))
    dataset, loader = make_multimodal_dataset_and_loader(cfg, split)
    emb_dims = {m: int(dataset.modality_dims["emb"].get(m, 0)) for m in cfg.multimodal_modalities}

    models, model_cfg = _load_models_and_cfgs(ckpts, emb_dims=emb_dims, device=device)
    pred_map, _ = _predict(models, loader, device=device)

    video_order = _load_reference_video_order(cfg, split)
    records: List[Tuple[str, float, float, int]] = []
    missing = []
    for video_path in video_order:
        sample_id = Path(video_path).stem
        payload = pred_map.get(sample_id)
        if payload is None:
            missing.append(sample_id)
            continue
        records.append(
            (
                video_path,
                float(payload["p0"]),
                float(payload["p1"]),
                int(payload["pred"]),
            )
        )
    if missing:
        raise ValueError(f"Missing predictions for {len(missing)} sample_ids. Example: {missing[:5]}")

    _validate_records(records)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if len(ckpts) == 1:
        suffix = ckpts[0].stem
    else:
        suffix = f"ensemble_{len(ckpts)}"
    out_dir = Path(OUTPUT_ROOT) / f"{RUN_TAG}_{suffix}_{timestamp}"
    paths = _write_submission_files(out_dir, records)

    meta = {
        "config_path": str(config_path),
        "checkpoint_paths": [str(p) for p in ckpts],
        "ensemble_size": len(ckpts),
        "split": split,
        "num_samples": len(records),
        "device": str(device),
        "model_cfg": model_cfg,
        "outputs": paths,
    }
    meta_path = out_dir / "submission_meta.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)

    logging.info("Challenge inference done. Samples=%d", len(records))
    logging.info("No-prob submission: %s", paths["no_probabilities"])
    logging.info("With-prob submission: %s", paths["with_probabilities"])
    logging.info("With-prob submission (hard, optional): %s", paths["with_probabilities_hard"])
    logging.info("Meta: %s", meta_path)
    return paths


def _compute_eval_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)
    if yt.size == 0:
        return {}
    out = {
        "ACC": float((yt == yp).mean()),
        "MF1": float(mf1_ah(yt, yp)),
        "UAR": float(uar_ah(yt, yp)),
    }
    for cls in sorted(set(yt.tolist() + yp.tolist())):
        mask = (yt == cls)
        denom = int(mask.sum())
        recall = 0.0 if denom == 0 else float(((yp == cls) & mask).sum() / denom)
        out[f"recall_c{int(cls)}"] = recall
    return out


def _average_metrics_across_splits(metrics_by_split: Dict[str, Dict[str, float]], splits: List[str]) -> Dict[str, float]:
    selected = [metrics_by_split.get(s, {}) for s in splits if isinstance(metrics_by_split.get(s), dict)]
    if not selected:
        return {}
    keys = sorted(set().union(*(m.keys() for m in selected)))
    out: Dict[str, float] = {}
    for key in keys:
        vals = [m.get(key) for m in selected if isinstance(m.get(key), (int, float))]
        if vals:
            out[key] = float(sum(float(v) for v in vals) / len(vals))
    return out


def _run_eval_for_split(
    cfg: ConfigLoader,
    models: List[MultiModalModel],
    device: torch.device,
    split: str,
    out_dir: Path,
) -> Dict[str, float]:
    import pandas as pd

    dataset, loader = make_multimodal_dataset_and_loader(cfg, split)
    pred_map, label_map = _predict(models, loader, device=device)

    csv_path = _resolve_split_csv_path(cfg, split)
    df = pd.read_csv(csv_path)
    id_col = "video_path" if "video_path" in df.columns else ("video_name" if "video_name" in df.columns else None)

    rows = []
    y_true: List[int] = []
    y_pred: List[int] = []

    if id_col is not None:
        for _, row in df.iterrows():
            raw = str(row[id_col])
            sample_id = Path(raw).stem
            payload = pred_map.get(sample_id)
            if payload is None:
                continue
            label = label_map.get(sample_id)
            pred = int(payload["pred"])
            p0 = float(payload["p0"])
            p1 = float(payload["p1"])
            rows.append(
                {
                    "split": split,
                    "sample_id": sample_id,
                    "video_id": raw,
                    "label": label,
                    "pred": pred,
                    "p0": p0,
                    "p1": p1,
                    "correct": (int(label) == pred) if label is not None else "",
                }
            )
            if label is not None:
                y_true.append(int(label))
                y_pred.append(pred)
    else:
        for sample_id in sorted(pred_map.keys()):
            payload = pred_map[sample_id]
            label = label_map.get(sample_id)
            pred = int(payload["pred"])
            p0 = float(payload["p0"])
            p1 = float(payload["p1"])
            rows.append(
                {
                    "split": split,
                    "sample_id": sample_id,
                    "video_id": sample_id,
                    "label": label,
                    "pred": pred,
                    "p0": p0,
                    "p1": p1,
                    "correct": (int(label) == pred) if label is not None else "",
                }
            )
            if label is not None:
                y_true.append(int(label))
                y_pred.append(pred)

    pred_csv = out_dir / f"{split}_predictions.csv"
    pred_csv.parent.mkdir(parents=True, exist_ok=True)
    with pred_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "sample_id", "video_id", "label", "pred", "p0", "p1", "correct"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    metrics = _compute_eval_metrics(y_true, y_pred)
    metrics["num_samples"] = float(len(rows))
    metrics["num_labeled"] = float(len(y_true))
    logging.info(
        "[Eval %s] samples=%d labeled=%d ACC=%.4f MF1=%.4f UAR=%.4f",
        split,
        len(rows),
        len(y_true),
        float(metrics.get("ACC", float("nan"))),
        float(metrics.get("MF1", float("nan"))),
        float(metrics.get("UAR", float("nan"))),
    )
    logging.info("[Eval %s] predictions csv: %s", split, pred_csv)
    return metrics


def run_eval_metrics(
    config_path: str = EVAL_CONFIG_PATH,
    checkpoint_path: str = CHECKPOINT_PATH,
    checkpoint_paths: List[str] | None = None,
    splits: List[str] | None = None,
) -> Dict[str, Dict[str, float]]:
    setup_logger(logging.INFO)
    cfg = ConfigLoader(config_path)
    cfg.show_config()

    ckpts = _resolve_checkpoint_paths(checkpoint_path=checkpoint_path, checkpoint_paths=checkpoint_paths)
    logging.info("Using %d checkpoint(s) for ensemble", len(ckpts))
    for i, ckpt in enumerate(ckpts, start=1):
        logging.info("  [%d] %s", i, ckpt)

    use_splits = list(splits) if splits is not None else list(EVAL_SPLITS)
    if not use_splits:
        raise ValueError("No splits configured for eval_metrics")

    device = _resolve_device(str(getattr(cfg, "device", "cpu")))
    # build emb dims from first split
    first_dataset, _ = make_multimodal_dataset_and_loader(cfg, use_splits[0])
    emb_dims = {m: int(first_dataset.modality_dims["emb"].get(m, 0)) for m in cfg.multimodal_modalities}
    models, model_cfg = _load_models_and_cfgs(ckpts, emb_dims=emb_dims, device=device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if len(ckpts) == 1:
        suffix = ckpts[0].stem
    else:
        suffix = f"ensemble_{len(ckpts)}"
    out_dir = Path(EVAL_OUTPUT_ROOT) / f"{EVAL_TAG}_{suffix}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_split: Dict[str, Dict[str, float]] = {}
    for split in use_splits:
        per_split[split] = _run_eval_for_split(cfg, models=models, device=device, split=split, out_dir=out_dir)
    avg_metrics = _average_metrics_across_splits(per_split, use_splits)
    if avg_metrics:
        logging.info(
            "[Eval AVG %s] ACC=%.4f MF1=%.4f UAR=%.4f",
            "+".join(use_splits),
            float(avg_metrics.get("ACC", float("nan"))),
            float(avg_metrics.get("MF1", float("nan"))),
            float(avg_metrics.get("UAR", float("nan"))),
        )

    summary = {
        "config_path": str(config_path),
        "checkpoint_paths": [str(p) for p in ckpts],
        "ensemble_size": len(ckpts),
        "splits": use_splits,
        "device": str(device),
        "model_cfg": model_cfg,
        "metrics": per_split,
        "metrics_avg": avg_metrics,
    }
    summary_path = out_dir / "eval_metrics.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    logging.info("Eval summary saved: %s", summary_path)
    return per_split


if __name__ == "__main__":
    if RUN_MODE == "challenge_submit":
        run_challenge_inference()
    elif RUN_MODE == "eval_metrics":
        run_eval_metrics()
    else:
        raise ValueError("RUN_MODE must be one of: challenge_submit, eval_metrics")
