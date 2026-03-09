# coding: utf-8
from __future__ import annotations

import json
import logging
import math
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.models.fusion_model import MultiModalModel
from src.utils.logger_setup import color_metric, color_split
from src.utils.losses import build_classification_loss
from src.utils.measures import mf1_ah, uar_ah
from src.utils.schedulers import SmartScheduler


CLASS_LABELS = {
    0: "no_ambivalence",
    1: "ambivalence",
}


def _exporters_snapshot(cfg) -> Dict[str, Any]:
    return {
        "face": {
            "artifact_tag": str(getattr(cfg, "face_artifact_tag", "")),
            "video_extractor": str(getattr(cfg, "video_extractor", "")),
            "average_features": str(getattr(cfg, "average_features", "")),
            "classifier_weights": str(getattr(cfg, "face_classifier_weights", "")),
            "feature_ckpt": str(getattr(cfg, "affectnet_ckpt_path", "")),
        },
        "audio": {
            "artifact_tag": str(getattr(cfg, "audio_artifact_tag", "")),
            "source": str(getattr(cfg, "audio_export_source", "")),
            "impl": str(getattr(cfg, "audio_export_impl", "")),
            "impl_resolved": str(getattr(cfg, "audio_export_impl_resolved", "")),
            "checkpoint_path": str(getattr(cfg, "audio_checkpoint_path", "")),
            "precomputed_path": str(getattr(cfg, "audio_precomputed_path", "")),
        },
        "text": {
            "artifact_tag": str(getattr(cfg, "text_artifact_tag", "")),
            "impl": str(getattr(cfg, "text_export_impl", "")),
            "impl_resolved": str(getattr(cfg, "text_export_impl_resolved", "")),
            "checkpoint_path": str(getattr(cfg, "text_checkpoint_path_resolved", "")),
            "text_column": str(getattr(cfg, "text_input_column", "")),
        },
        "scene": {
            "artifact_tag": str(getattr(cfg, "scene_artifact_tag", "")),
            "model_name": str(getattr(cfg, "scene_model_name", "")),
            "checkpoint_path": str(getattr(cfg, "scene_checkpoint_path", "")),
            "num_frames": int(getattr(cfg, "scene_num_frames", 0)),
        },
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve_device(device_str: str) -> torch.device:
    s = str(device_str).lower()
    if s.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if s == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {
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


def _collect_labels(loader) -> torch.Tensor:
    ys = []
    for batch in loader:
        if batch is None:
            continue
        ys.append(batch["labels"].detach().cpu())
    if not ys:
        raise RuntimeError("Loader has no labels.")
    return torch.cat(ys, dim=0)


def _compute_class_weights(
    labels: torch.Tensor,
    num_classes: int,
    mode: str,
    manual_weights: list[float] | None,
    device: torch.device,
) -> torch.Tensor | None:
    mode = str(mode).lower()
    if mode == "none":
        return None

    if mode == "manual":
        if not isinstance(manual_weights, list) or len(manual_weights) != num_classes:
            raise ValueError(
                f"class_weighting='manual' requires fusion_train.class_weights of length {num_classes}"
            )
        return torch.tensor([float(x) for x in manual_weights], dtype=torch.float32, device=device)

    if mode == "balanced":
        counts = torch.bincount(labels.to(torch.long), minlength=num_classes).to(torch.float32)
        n = float(labels.numel())
        weights = n / (float(num_classes) * counts.clamp_min(1.0))
        return weights.to(device)

    raise ValueError(f"Unknown class_weighting='{mode}'. Use: none|balanced|manual")


def _macro_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, float]:
    acc = float((y_true == y_pred).mean()) if y_true.size > 0 else 0.0
    out = {
        "ACC": acc,
        "MF1": float(mf1_ah(y_true, y_pred)),
        "UAR": float(uar_ah(y_true, y_pred)),
    }
    for c in range(num_classes):
        cls_true = (y_true == c)
        denom = float(cls_true.sum())
        if denom == 0:
            recall = 0.0
        else:
            recall = float(((y_pred == c) & cls_true).sum() / denom)
        out[f"recall_c{c}_{CLASS_LABELS.get(c, f'class{c}')}"] = recall
    return out


@torch.no_grad()
def _eval_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Dict[str, float]:
    model.eval()
    all_y = []
    all_p = []
    total_loss = 0.0
    total_n = 0

    for batch in tqdm(loader, desc="Eval", leave=False):
        if batch is None:
            continue
        b = _move_batch_to_device(batch, device)
        y = b["labels"]
        out = model(b)
        logits = out["logits"]
        loss = criterion(logits, y)

        pred = logits.argmax(dim=1)
        all_y.append(y.detach().cpu())
        all_p.append(pred.detach().cpu())
        total_loss += float(loss.item()) * y.size(0)
        total_n += y.size(0)

    if not all_y:
        return {}

    y_true = torch.cat(all_y).numpy()
    y_pred = torch.cat(all_p).numpy()
    metrics = _macro_metrics(y_true, y_pred, num_classes=num_classes)
    metrics["LOSS"] = total_loss / max(1, total_n)
    return metrics


def _train_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: SmartScheduler,
    device: torch.device,
    num_classes: int,
    grad_clip: float,
    lambda_proto: float,
    lambda_proto_div: float,
) -> Dict[str, float]:
    model.train()
    all_y = []
    all_p = []
    total_loss = 0.0
    total_n = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        if batch is None:
            continue
        b = _move_batch_to_device(batch, device)
        y = b["labels"]

        optimizer.zero_grad(set_to_none=True)
        out = model(b)
        logits = out["logits"]

        loss = criterion(logits, y)
        if getattr(model, "use_prototypes", False) and out.get("proto_logits") is not None:
            loss_proto = criterion(out["proto_logits"], y)
            loss = loss + float(lambda_proto) * loss_proto
            loss = loss + float(lambda_proto_div) * model.proto.diversity_loss()

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
        optimizer.step()
        scheduler.step(batch_level=True)

        pred = logits.argmax(dim=1)
        all_y.append(y.detach().cpu())
        all_p.append(pred.detach().cpu())
        total_loss += float(loss.item()) * y.size(0)
        total_n += y.size(0)

    if not all_y:
        return {}

    y_true = torch.cat(all_y).numpy()
    y_pred = torch.cat(all_p).numpy()
    metrics = _macro_metrics(y_true, y_pred, num_classes=num_classes)
    metrics["LOSS"] = total_loss / max(1, total_n)
    return metrics


def _mean_or_nan(values: list[float]) -> float:
    valid = [v for v in values if not math.isnan(v)]
    if not valid:
        return float("nan")
    return float(np.mean(valid))


def train(cfg, train_loader, dev_loader, test_loader, *, results_dir: str) -> Dict[str, Any]:
    """
    Fusion training loop adapted to multimodal dataloader.
    Keeps early stopping / scheduler / metric selection logic close to face baseline.
    """
    seed_everything(int(getattr(cfg, "fusion_random_seed", 42)))
    device = _resolve_device(str(getattr(cfg, "device", "cpu")))

    modalities = list(getattr(cfg, "multimodal_modalities", []))
    if not modalities:
        raise ValueError("multimodal.modalities must not be empty")

    emb_dims = {m: int(train_loader.dataset.modality_dims["emb"].get(m, 0)) for m in modalities}
    input_type = str(getattr(cfg, "fusion_input_type", "emb+prob"))
    if input_type in {"emb", "emb+prob"}:
        bad = [m for m in modalities if emb_dims.get(m, 0) <= 0]
        if bad:
            raise ValueError(
                f"fusion.input_type='{input_type}' requires non-zero embedding dim for all modalities. Missing: {bad}"
            )

    model_cfg = {
        "modalities": modalities,
        "input_type": input_type,
        "fusion": str(getattr(cfg, "fusion_type", "exchange_transformer")),
        "d_model": int(getattr(cfg, "fusion_d_model", 256)),
        "drop": float(getattr(cfg, "fusion_drop", 0.1)),
        "use_prototypes": bool(getattr(cfg, "fusion_use_prototypes", False)),
        "num_prototypes": int(getattr(cfg, "fusion_num_prototypes", 4)),
        "proto_tau": float(getattr(cfg, "fusion_proto_tau", 0.07)),
        "x_layers": int(getattr(cfg, "fusion_x_layers", 2)),
        "x_heads": int(getattr(cfg, "fusion_x_heads", 4)),
        "x_ff_mult": int(getattr(cfg, "fusion_x_ff_mult", 4)),
        "x_use_cls": bool(getattr(cfg, "fusion_x_use_cls", True)),
        "x_layer_impl": str(getattr(cfg, "fusion_x_layer_impl", "torch")),
        "x_positional_encoding": bool(getattr(cfg, "fusion_x_positional_encoding", False)),
        "videoformer_positional_encoding": bool(getattr(cfg, "fusion_videoformer_positional_encoding", False)),
        "videoformer_gate_mode": str(getattr(cfg, "fusion_videoformer_gate_mode", "none")),
    }
    model = MultiModalModel(model_cfg, emb_dims=emb_dims).to(device)

    labels_all = _collect_labels(train_loader).to(torch.long)
    num_classes = max(2, int(labels_all.max().item()) + 1)
    class_weights = _compute_class_weights(
        labels_all,
        num_classes=num_classes,
        mode=str(getattr(cfg, "fusion_class_weighting", "balanced")),
        manual_weights=getattr(cfg, "fusion_class_weights", None),
        device=device,
    )
    criterion = build_classification_loss(
        str(getattr(cfg, "fusion_loss_name", "cross_entropy")),
        class_weights=class_weights,
        label_smoothing=float(getattr(cfg, "fusion_label_smoothing", 0.0)),
        focal_gamma=float(getattr(cfg, "fusion_focal_gamma", 2.0)),
    )

    optim_name = str(getattr(cfg, "fusion_optimizer", "adamw")).lower()
    lr = float(getattr(cfg, "fusion_lr", 2e-4))
    wd = float(getattr(cfg, "fusion_weight_decay", 1e-4))
    momentum = float(getattr(cfg, "fusion_momentum", 0.9))
    if optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif optim_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    elif optim_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
    else:
        raise ValueError(f"Unknown fusion_train.optimizer='{optim_name}'")

    scheduler_cfg = SimpleNamespace(
        num_epochs=int(getattr(cfg, "fusion_num_epochs", 30)),
        lr=lr,
        warmup_ratio=float(getattr(cfg, "fusion_warmup_ratio", 0.1)),
    )
    scheduler = SmartScheduler(
        scheduler_type=str(getattr(cfg, "fusion_scheduler_type", "plateau")),
        optimizer=optimizer,
        config=scheduler_cfg,
        steps_per_epoch=max(1, len(train_loader)),
    )

    num_epochs = int(getattr(cfg, "fusion_num_epochs", 30))
    max_patience = int(getattr(cfg, "fusion_max_patience", 8))
    grad_clip = float(getattr(cfg, "fusion_grad_clip", 1.0))
    lambda_proto = float(getattr(cfg, "fusion_lambda_proto", 0.3))
    lambda_proto_div = float(getattr(cfg, "fusion_lambda_proto_div", 0.02))

    ckpt_dir = Path(results_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path: Optional[Path] = None
    best_score = -1.0
    patience = 0
    history: list[dict[str, Any]] = []
    best_snapshot: dict[str, Any] = {}

    for epoch in range(1, num_epochs + 1):
        tr = _train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_classes=num_classes,
            grad_clip=grad_clip,
            lambda_proto=lambda_proto,
            lambda_proto_div=lambda_proto_div,
        )
        dv = _eval_epoch(model=model, loader=dev_loader, criterion=criterion, device=device, num_classes=num_classes)
        ts = _eval_epoch(model=model, loader=test_loader, criterion=criterion, device=device, num_classes=num_classes)

        mf1_dev = float(dv.get("MF1", float("nan")))
        mf1_test = float(ts.get("MF1", float("nan")))
        uar_dev = float(dv.get("UAR", float("nan")))
        uar_test = float(ts.get("UAR", float("nan")))
        score = _mean_or_nan([mf1_dev, mf1_test])
        uar_avg = _mean_or_nan([uar_dev, uar_test])
        scheduler.step(metric=score if not math.isnan(score) else mf1_dev, batch_level=False)

        lr_cur = float(optimizer.param_groups[0]["lr"])
        def _metric_line(m: Dict[str, float]) -> str:
            parts = [
                color_metric("LOSS", m.get("LOSS", float("nan"))),
                color_metric("UAR", m.get("UAR", float("nan"))),
                color_metric("MF1", m.get("MF1", float("nan"))),
            ]
            for c in range(num_classes):
                key = f"recall_c{c}_{CLASS_LABELS.get(c, f'class{c}')}"
                if key in m:
                    parts.append(color_metric(key, m[key]))
            return " | ".join(parts)

        log_train = _metric_line(tr)
        log_dev = _metric_line(dv)
        log_test = _metric_line(ts)
        logging.info("=== EPOCH %d/%d ===", epoch, num_epochs)
        logging.info("[%s] %s", color_split("TRAIN"), log_train)
        logging.info("[%s] %s", color_split("DEV"), log_dev)
        logging.info("[%s] %s", color_split("TEST"), log_test)
        logging.info(
            "[AVG] %s | %s | lr=%.6f",
            color_metric("MF1_AVG", score),
            color_metric("UAR_AVG", uar_avg),
            lr_cur,
        )

        history.append(
            {
                "epoch": epoch,
                "lr": lr_cur,
                "train": tr,
                "dev": dv,
                "test": ts,
                "score": score,
                "MF1_AVG": score,
                "UAR_AVG": uar_avg,
            }
        )

        if not math.isnan(score) and score > best_score:
            best_score = score
            patience = 0
            best_ckpt_path = ckpt_dir / f"fusion_best_ep{epoch:03d}_mf1avg_{best_score:.4f}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "score": best_score,
                    "model_cfg": model_cfg,
                    "model_state": model.state_dict(),
                },
                best_ckpt_path,
            )
            best_snapshot = {"epoch": epoch, "score": best_score, "train": tr, "dev": dv, "test": ts}
            logging.info("Saved best fusion checkpoint: %s", best_ckpt_path)
        else:
            patience += 1
            if patience >= max_patience:
                logging.info("Early stopping.")
                break

    if best_ckpt_path is not None and best_ckpt_path.exists():
        state = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(state["model_state"], strict=True)
        final_dev = _eval_epoch(model=model, loader=dev_loader, criterion=criterion, device=device, num_classes=num_classes)
        final_test = _eval_epoch(model=model, loader=test_loader, criterion=criterion, device=device, num_classes=num_classes)
    else:
        final_dev = {}
        final_test = {}

    summary = {
        "best_score": best_score,
        "best_checkpoint": str(best_ckpt_path) if best_ckpt_path is not None else "",
        "exporters": _exporters_snapshot(cfg),
        "best": best_snapshot,
        "final_dev": final_dev,
        "final_test": final_test,
        "history": history,
    }

    metrics_path = Path(results_dir) / "fusion_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    logging.info("Fusion metrics saved: %s", metrics_path)

    return summary
