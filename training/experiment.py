from __future__ import annotations

import time
from typing import Dict, Any, Optional, List

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from training.config import ExperimentConfig
from training.utils import set_seed, ensure_dir, save_json, pick_device, AverageMeter
from datasets.bah_dataset import build_bah_loaders
from models.bah_model import BAHClassifier
from metrics.classification import compute_mf1
from training.optim import build_optimizer

def _build_criteria(cfg: ExperimentConfig, loaders: Dict[str, Any], device: torch.device):
   
    use_cw = bool(getattr(cfg.train, "use_class_weights", False))
    mode = getattr(cfg.train, "class_weights_mode", "balanced")
    manual = getattr(cfg.train, "class_weights_manual", None)

    class_weights_tensor: Optional[torch.Tensor] = None

    if use_cw:
        if mode == "manual":
            if manual is None:
                raise ValueError("train.class_weights_mode='manual' but train.class_weights_manual is None")
            if not isinstance(manual, list) or len(manual) == 0:
                raise ValueError("train.class_weights_manual must be a non-empty list, e.g. [1.0, 5.89]")
            class_weights_tensor = torch.tensor([float(x) for x in manual], dtype=torch.float32, device=device)
        else:
            cw = loaders.get("class_weights", None)
            if cw is None:
                print("[WARN] use_class_weights=True but loaders['class_weights'] is None -> using no weights")
            else:
                class_weights_tensor = torch.tensor([float(x) for x in cw], dtype=torch.float32, device=device)

        if class_weights_tensor is not None and torch.any(class_weights_tensor == 0):
            print(f"[WARN] Some class weights are 0 (missing classes in train?): {class_weights_tensor.tolist()}")

    criterion_train = nn.CrossEntropyLoss(
        weight=class_weights_tensor,
        label_smoothing=float(getattr(cfg.model, "label_smoothing", 0.0)),
    )

    criterion_eval = nn.CrossEntropyLoss(
        weight=class_weights_tensor,
        label_smoothing=0.0,
    )

    return criterion_train, criterion_eval, class_weights_tensor


def _evaluate_mf1(
    model: nn.Module,
    loader,
    device: torch.device,
    num_classes: int,
    criterion_eval: nn.Module,
) -> Dict[str, float]:
    model.eval()
    ys, ps = [], []
    loss_meter = AverageMeter()

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            mask = batch["mask"]
            if mask is not None:
                mask = mask.to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            logits = model(x, mask=mask)
            loss = criterion_eval(logits, y)

            pred = torch.argmax(logits, dim=-1)
            ys.append(y.cpu().numpy())
            ps.append(pred.cpu().numpy())
            loss_meter.update(loss.item(), n=int(y.size(0)))

    y_true = np.concatenate(ys) if ys else np.array([], dtype=np.int64)
    y_pred = np.concatenate(ps) if ps else np.array([], dtype=np.int64)
    mf1 = compute_mf1(y_true, y_pred, num_classes=num_classes)
    return {"loss": float(loss_meter.avg), "mf1": float(mf1)}


def _fmt_epoch_table(
    epoch: int,
    num_epochs: int,
    train_loss: float,
    metrics: Dict[str, Dict[str, float]],
    best: float,
    best_epoch: int,
    bad_epochs: int,
) -> str:
    def line(name: str):
        m = metrics[name]
        return f"{name:<9} | loss={m['loss']:.4f} | mf1={m['mf1']:.4f}"

    lines = [
        f"===== EPOCH {epoch}/{num_epochs} =====",
        f"Train step loss: {train_loss:.4f}",
        line("train"),
        line("val"),
        line("test"),
        line("eval_all"),
        f"Best mf1 (eval_all): {best:.4f} @ epoch {best_epoch} | bad_epochs={bad_epochs}",
    ]
    return "\n".join(lines)


def run_single_experiment(cfg: ExperimentConfig) -> Dict[str, Any]:
    cfg.train.monitor_metric = "mf1"
    cfg.train.monitor_mode = "max"

    set_seed(cfg.train.seed)
    device = pick_device(cfg.train.device)

    workdir = ensure_dir(cfg.train.workdir)
    ensure_dir(workdir / "checkpoints")
    ensure_dir(workdir / "logs")
    save_json(workdir / "config.json", cfg)

    loaders = build_bah_loaders(cfg.data, cfg.train)
    input_dim = loaders["input_dim"] if cfg.model.input_dim is None else cfg.model.input_dim

    print("DATA SIZES:", loaders.get("sizes", {}))

    model = BAHClassifier(cfg.model, input_dim=input_dim).to(device)

    criterion_train, criterion_eval, cw = _build_criteria(cfg, loaders, device)
    if cw is not None:
        print(f"[INFO] Using class weights: {cw.tolist()}")
    else:
        print("[INFO] Class weights: OFF")

    optimizer = build_optimizer(model, cfg.train)
    scaler = GradScaler(enabled=bool(cfg.train.amp and device.type == "cuda"))

    best_score = None
    best_epoch = -1
    bad_epochs = 0
    history = []

    for epoch in range(1, cfg.train.num_epochs + 1):
        t0 = time.time()
        model.train()
        loss_meter = AverageMeter()

        for batch in loaders["train"]:
            x = batch["x"].to(device, non_blocking=True)
            mask = batch["mask"]
            if mask is not None:
                mask = mask.to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=scaler.is_enabled()):
                logits = model(x, mask=mask)
                loss = criterion_train(logits, y)

            scaler.scale(loss).backward()
            if cfg.train.grad_clip_norm and cfg.train.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), n=int(y.size(0)))

        train_step_loss = float(loss_meter.avg)

        m_train = _evaluate_mf1(model, loaders["train"], device=device, num_classes=cfg.model.num_classes, criterion_eval=criterion_eval)
        m_val   = _evaluate_mf1(model, loaders["val"], device=device, num_classes=cfg.model.num_classes, criterion_eval=criterion_eval)
        m_test  = _evaluate_mf1(model, loaders["test"], device=device, num_classes=cfg.model.num_classes, criterion_eval=criterion_eval)
        m_all   = _evaluate_mf1(model, loaders["eval_all"], device=device, num_classes=cfg.model.num_classes, criterion_eval=criterion_eval)

        score = float(m_all["mf1"]) 

        improved = best_score is None or score > best_score

        if improved:
            best_score = score
            best_epoch = epoch
            bad_epochs = 0
            if cfg.train.save_best:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "best_score": best_score,
                        "cfg": cfg.to_dict(),
                    },
                    workdir / "checkpoints" / "best.pt",
                )
        else:
            bad_epochs += 1

        if cfg.train.save_last:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_score": best_score,
                    "cfg": cfg.to_dict(),
                },
                workdir / "checkpoints" / "last.pt",
            )

        row = {
            "epoch": epoch,
            "time_sec": float(time.time() - t0),
            "train_step_loss": train_step_loss,
            "train": m_train,
            "val": m_val,
            "test": m_test,
            "eval_all": m_all,
            "best_score": float(best_score) if best_score is not None else None,
            "best_epoch": int(best_epoch),
        }
        history.append(row)
        save_json(workdir / "logs" / "history.json", history)

        print(
            _fmt_epoch_table(
                epoch,
                cfg.train.num_epochs,
                train_step_loss,
                {"train": m_train, "val": m_val, "test": m_test, "eval_all": m_all},
                best=float(best_score) if best_score is not None else float("nan"),
                best_epoch=best_epoch,
                bad_epochs=bad_epochs,
            )
        )

        if bad_epochs >= cfg.train.early_stop_patience:
            print(f"Early stopping: no improvement for {cfg.train.early_stop_patience} epochs.")
            break

    best_path = workdir / "checkpoints" / "best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=True)

    final_train = _evaluate_mf1(model, loaders["train"], device=device, num_classes=cfg.model.num_classes, criterion_eval=criterion_eval)
    final_val   = _evaluate_mf1(model, loaders["val"], device=device, num_classes=cfg.model.num_classes, criterion_eval=criterion_eval)
    final_test  = _evaluate_mf1(model, loaders["test"], device=device, num_classes=cfg.model.num_classes, criterion_eval=criterion_eval)
    final_all   = _evaluate_mf1(model, loaders["eval_all"], device=device, num_classes=cfg.model.num_classes, criterion_eval=criterion_eval)

    summary = {
        "best_epoch": best_epoch,
        "best_score": best_score,  
        "final": {"train": final_train, "val": final_val, "test": final_test, "eval_all": final_all},
        "workdir": str(workdir),
    }
    save_json(workdir / "logs" / "summary.json", summary)

    print("===== FINAL (best checkpoint) =====")
    print(f"train   mf1={final_train['mf1']:.4f} loss={final_train['loss']:.4f}")
    print(f"val     mf1={final_val['mf1']:.4f} loss={final_val['loss']:.4f}")
    print(f"test    mf1={final_test['mf1']:.4f} loss={final_test['loss']:.4f}")
    print(f"eval_all mf1={final_all['mf1']:.4f} loss={final_all['loss']:.4f}")
    return summary
