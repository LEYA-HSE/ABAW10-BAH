# src/train.py
# coding: utf-8
from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, recall_score

from src.models.models import VideoFormer, VideoMamba, VectorMLP
from src.models.non_neural import KernelELMClassifier, ELMClassifier
from src.utils.logger_setup import color_metric, color_split, dbg_dump_logits
from src.utils.schedulers import SmartScheduler
import pickle

CLASS_LABELS = {
    0: "no_ambivalence",
    1: "ambivalence",
}


def seed_everything(seed: int):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _stack_face_features(
    features_list: List[Optional[dict]],
    average_mode: str = "mean",
    segment_length: Optional[int] = None,
):
    if average_mode not in {"mean", "mean_std", "raw"}:
        raise ValueError(f"unknown average_mode={average_mode!r} (expected 'mean'|'mean_std'|'raw')")

    rows: List[torch.Tensor] = []
    keep_idx: List[int] = []
    lengths: List[int] = []

    for i, feats in enumerate(features_list):
        if not feats or "face" not in feats or feats["face"] is None:
            continue
        face = feats["face"]

        if average_mode == "mean_std" and "mean" in face and "std" in face:
            x = torch.cat([face["mean"].view(-1), face["std"].view(-1)], dim=0).to(torch.float32)
            rows.append(x)
            keep_idx.append(i)

        elif average_mode == "mean" and "mean" in face:
            x = face["mean"].view(-1).to(torch.float32)
            rows.append(x)
            keep_idx.append(i)

        elif average_mode == "raw" and "seq" in face:
            s = face["seq"].to(torch.float32)  # [T, D]
            rows.append(s)
            lengths.append(s.size(0))
            keep_idx.append(i)

        else:
            continue

    if not rows:
        raise RuntimeError("No valid face features in batch. Check cache and average_features.")

    if average_mode == "raw":
        X = pad_sequence(rows, batch_first=True, padding_value=0.0)  # [B, T_max, D]
        T = X.size(1)
        mask = torch.zeros(X.size(0), T, dtype=torch.bool, device=X.device)
        if lengths:
            for bi, L in enumerate(lengths):
                mask[bi, :min(L, T)] = True
        else:
            mask[:] = True
    else:
        X = torch.stack(rows, dim=0)
        mask = None

    return X, keep_idx, mask


def _filter_labels(labels: torch.Tensor, keep_idx: List[int]) -> torch.Tensor:
    return labels[keep_idx]


def _gather_all_labels(loader: DataLoader, average_mode: str, segment_length: Optional[int] = None) -> np.ndarray:
    ys = []
    for batch in loader:
        if batch is None:
            continue
        _, keep, _ = _stack_face_features(batch["features"], average_mode, segment_length=segment_length)
        y = _filter_labels(batch["labels"], keep)
        ys.append(y.cpu().numpy())
    if not ys:
        raise RuntimeError("Failed to collect labels from train loader.")
    return np.concatenate(ys, axis=0)


def _num_classes_from_loader(loader: DataLoader, average_mode: str, segment_length: Optional[int] = None) -> int:
    y = _gather_all_labels(loader, average_mode, segment_length=segment_length)
    return int(np.max(y) + 1)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, float]:
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    uar = recall_score(y_true, y_pred, average="macro", zero_division=0)
    per_cls = recall_score(y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0)
    out: Dict[str, float] = {"MF1": float(mf1), "UAR": float(uar)}
    for c, r in enumerate(per_cls):
        name = CLASS_LABELS.get(c, f"class{c}")
        out[f"recall_c{c}_{name}"] = float(r)
    return out


def _log_loader_feature_shape(
    loader: DataLoader,
    avg_mode: str,
    segment_length: Optional[int],
    name: str,
) -> None:
    first = None
    for b in loader:
        if b is not None:
            first = b
            break
    if first is None:
        logging.info(f"[FEATURES:{name}] loader empty")
        return
    try:
        X, keep, mask = _stack_face_features(first["features"], avg_mode, segment_length=segment_length)
    except Exception as e:
        logging.warning(f"[FEATURES:{name}] failed to stack features: {e}")
        return

    total = len(first["features"]) if isinstance(first.get("features"), list) else "?"
    if X.ndim == 3:
        seq_len = int(X.shape[1])
        feat_dim = int(X.shape[2])
    else:
        seq_len = 1
        feat_dim = int(X.shape[1])
    mask_info = f", mask_shape={tuple(mask.shape)}" if mask is not None else ""
    logging.info(
        f"[FEATURES:{name}] avg={avg_mode} X={tuple(X.shape)} seq_len={seq_len} "
        f"feat_dim={feat_dim} kept={len(keep)}/{total}{mask_info}"
    )


def _build_model(cfg, input_dim: int, seq_len: int, num_classes: int, device: torch.device) -> nn.Module:
    model_name = cfg.model_name.lower()

    if model_name in ("mamba", "vmamba", "video_mamba"):
        model = VideoMamba(
            input_dim=input_dim,
            hidden_dim=cfg.hidden_dim,
            mamba_d_state=cfg.mamba_d_state,
            mamba_ker_size=cfg.mamba_ker_size,
            mamba_layer_number=cfg.mamba_layers,
            d_discr=getattr(cfg, "mamba_d_discr", None),
            dropout=cfg.dropout,
            seg_len=seq_len,
            out_features=cfg.out_features,
            num_classes=num_classes,
            device=str(device),
        )
    elif model_name in ("transformer", "former", "videoformer", "tr"):
        model = VideoFormer(
            input_dim=input_dim,
            hidden_dim=cfg.hidden_dim,
            num_transformer_heads=cfg.num_transformer_heads,
            positional_encoding=cfg.positional_encoding,
            dropout=cfg.dropout,
            tr_layer_number=cfg.tr_layers,
            seg_len=seq_len,
            out_features=cfg.out_features,
            num_classes=num_classes,
        )
    elif model_name in ("vector", "mlp", "vector_mlp", "mlp_vector"):
        model = VectorMLP(
            input_dim=input_dim,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
            out_features=cfg.out_features,
            num_classes=num_classes,
        )
    else:
        raise ValueError(
            f"Unknown model='{cfg.model_name}'. Use 'mamba', 'transformer', or 'vector'."
        )
    return model.to(device)


def _collect_vector_features(
    loader: DataLoader,
    avg_mode: str,
    segment_length: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    X_list, y_list = [], []
    for batch in loader:
        if batch is None:
            continue
        X, keep, mask = _stack_face_features(batch["features"], avg_mode, segment_length=segment_length)
        y = _filter_labels(batch["labels"], keep)
        if X.ndim == 3:
            if mask is None:
                Xv = X.mean(dim=1)
            else:
                denom = mask.sum(dim=1).clamp(min=1).unsqueeze(-1).to(X.dtype)
                Xv = X.masked_fill(~mask.unsqueeze(-1), 0.0).sum(dim=1) / denom
        else:
            Xv = X
        X_list.append(Xv.cpu().numpy())
        y_list.append(y.cpu().numpy())
    if not X_list:
        raise RuntimeError("No samples collected from loader.")
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


@torch.no_grad()
def _eval_epoch(
    cfg,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    avg_mode: str,
    metrics_num_classes: int,
    criterion: nn.Module,
    segment_length: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    all_y, all_p = [], []
    total_loss = 0.0
    total_n = 0

    for bidx, batch in enumerate(tqdm(loader, desc="Eval", leave=False)):
        if batch is None:
            continue

        X, keep, mask = _stack_face_features(batch["features"], avg_mode, segment_length=segment_length)
        y = _filter_labels(batch["labels"], keep).to(device)
        if X.ndim == 2:
            X = X.unsqueeze(1)

        logits = model(
            X.to(device, non_blocking=True),
            mask=mask.to(device, non_blocking=True) if mask is not None else None,
        )

        if bidx == 0:
            dbg_dump_logits(logits, cfg.print_logits, prefix="[DBG:VAL:final]", max_rows=5, max_cols=logits.size(1))

        loss = criterion(logits, y)
        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs

        pred = logits.argmax(dim=1)
        all_y.append(y.cpu())
        all_p.append(pred.cpu())

    if not all_y:
        return {}
    y_true = torch.cat(all_y).numpy()
    y_pred = torch.cat(all_p).numpy()
    out = _metrics(y_true, y_pred, metrics_num_classes)
    if total_n > 0:
        out["LOSS"] = total_loss / total_n
    return out


# helpers for early stopping

def _mean_metric(metrics_map: Dict[str, float], metric_name: str) -> float:
    if not metrics_map:
        return float("nan")
    pref = f"{metric_name}_"
    vals = [
        v for k, v in metrics_map.items()
        if isinstance(v, (int, float)) and (k == metric_name or k.startswith(pref))
    ]
    return float(np.mean(vals)) if vals else float("nan")


def _probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)


@torch.no_grad()
def export_logits_to_pkl(
    cfg,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    avg_mode: str,
    out_path: str,
    segment_length: int | None,
):
    model.eval()
    out = {}

    for batch in tqdm(loader, desc=f"Export logits -> {out_path}", leave=False):
        if batch is None:
            continue

        X, keep, mask = _stack_face_features(batch["features"], avg_mode, segment_length=segment_length)
        if len(keep) == 0:
            continue

        if "names" not in batch or "video_paths" not in batch:
            raise KeyError("export_logits_to_pkl expects batch['names'] and batch['video_paths']")

        if X.ndim == 2:
            X = X.unsqueeze(1)

        names = [str(batch["names"][i]) for i in keep]
        keys = [str(batch["video_paths"][i]) for i in keep]

        Xd = X.to(device, non_blocking=True)
        md = mask.to(device, non_blocking=True) if mask is not None else None

        try:
            logits, embeddings = model(Xd, mask=md, return_embeddings=True)
        except TypeError:
            logits = model(Xd, mask=md)
            embeddings = torch.full((logits.size(0), 1), float("nan"), device=logits.device)

        final_v = _probs_from_logits(logits).detach().cpu()
        embeddings = embeddings.detach().cpu()

        C = final_v.size(1)
        cls_v = torch.full((final_v.size(0), C), float("nan"))
        proto_v = torch.full((final_v.size(0), C), float("nan"))

        for i in range(len(keys)):
            k = keys[i]
            if k in out:
                k = f"{k}__dup{i}"

            out[k] = {
                "name": names[i],
                "final_prob": final_v[i].numpy().astype(np.float32),
                "cls_prob": cls_v[i].numpy().astype(np.float32),
                "proto_prob": proto_v[i].numpy().astype(np.float32),
                "embeddings": embeddings[i].numpy().astype(np.float32),
            }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info(f"[EXPORT] saved {len(out)} files -> {out_path}")


# train loop

def train(
    cfg,
    mm_loader: DataLoader,
    dev_loaders: Dict[str, DataLoader] | None = None,
    test_loaders: Dict[str, DataLoader] | None = None,
):
    """
    Single-label classification with CE + class weights + UAR/MF1.
    Early stopping uses average MF1 on validation and test.
    """
    seed_everything(cfg.random_seed)
    device = torch.device(cfg.device)
    avg_mode = cfg.average_features.lower()

    first = None
    for b in mm_loader:
        if b is not None:
            first = b
            break
    if first is None:
        raise RuntimeError("train loader is empty (or collate filtered everything).")
    X0, keep0, mask0 = _stack_face_features(first["features"], avg_mode, segment_length=cfg.segment_length)

    if X0.ndim == 3:
        in_dim = int(X0.shape[2])
        seq_len = int(X0.shape[1])
    else:
        in_dim = int(X0.shape[1])
        seq_len = 1

    total0 = len(first["features"]) if isinstance(first.get("features"), list) else "?"
    mask_info = f", mask_shape={tuple(mask0.shape)}" if mask0 is not None else ""
    logging.info(
        f"[FEATURES:train] avg={avg_mode} X={tuple(X0.shape)} seq_len={seq_len} "
        f"feat_dim={in_dim} kept={len(keep0)}/{total0}{mask_info}"
    )

    model_num_classes = getattr(cfg, "num_classes", None)
    if model_num_classes is None:
        model_num_classes = _num_classes_from_loader(mm_loader, avg_mode, segment_length=cfg.segment_length)
    metrics_num_classes = int(model_num_classes)

    if dev_loaders:
        for name, ldr in dev_loaders.items():
            _log_loader_feature_shape(ldr, avg_mode, cfg.segment_length, f"dev:{name}")
    if test_loaders:
        for name, ldr in test_loaders.items():
            _log_loader_feature_shape(ldr, avg_mode, cfg.segment_length, f"test:{name}")

    # ---------------------
    # Non-neural models
    # ---------------------
    model_name = str(cfg.model_name).lower()
    if model_name in {"kelm", "kernel_elm", "elm", "catboost"}:
        if avg_mode == "raw":
            raise ValueError("Non-neural models require vector features. Set average_features='mean' or 'mean_std'.")
        X_train, y_train = _collect_vector_features(mm_loader, avg_mode, segment_length=cfg.segment_length)

        # optional class weights as sample weights (for catboost)
        sample_weights = None
        if cfg.class_weighting in ("balanced", "manual"):
            classes = np.arange(model_num_classes)
            if cfg.class_weighting == "balanced":
                cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
            else:
                w = getattr(cfg, "class_weights", None)
                if not isinstance(w, (list, tuple)) or len(w) != model_num_classes:
                    raise ValueError(
                        f"class_weighting='manual' requires class_weights list of length {model_num_classes}."
                    )
                cw = np.array([float(x) for x in w], dtype=np.float32)
            sample_weights = cw[y_train]

        if model_name in {"kelm", "kernel_elm"}:
            clf = KernelELMClassifier(C=cfg.kelm_C, gamma=cfg.kelm_gamma)
            clf.fit(X_train, y_train, model_num_classes)
            predict_fn = clf.predict
        elif model_name == "elm":
            clf = ELMClassifier(hidden_dim=cfg.elm_hidden, activation=cfg.elm_activation, C=cfg.elm_C, seed=cfg.random_seed)
            clf.fit(X_train, y_train, model_num_classes)
            predict_fn = clf.predict
        else:
            try:
                from catboost import CatBoostClassifier
            except Exception as e:
                raise ImportError("catboost is not installed. Add it to requirements.txt to use model_name='catboost'.") from e
            clf = CatBoostClassifier(
                iterations=cfg.catboost_iters,
                depth=cfg.catboost_depth,
                learning_rate=cfg.catboost_lr,
                loss_function="Logloss",
                verbose=False,
            )
            clf.fit(X_train, y_train, sample_weight=sample_weights)
            predict_fn = lambda X: clf.predict(X).astype(int).reshape(-1)

        # eval dev/test
        best_dev, best_test = {}, {}
        if dev_loaders:
            for name, ldr in dev_loaders.items():
                Xd, yd = _collect_vector_features(ldr, avg_mode, segment_length=cfg.segment_length)
                pred = predict_fn(Xd)
                md = _metrics(yd, pred, metrics_num_classes)
                best_dev.update({f"{k}_{name}": v for k, v in md.items()})
                msg = " | ".join(color_metric(k, v) for k, v in md.items())
                logging.info(f"[{color_split('DEV')}:{name}] {msg}")
        if test_loaders:
            for name, ldr in test_loaders.items():
                Xt, yt = _collect_vector_features(ldr, avg_mode, segment_length=cfg.segment_length)
                pred = predict_fn(Xt)
                mt = _metrics(yt, pred, metrics_num_classes)
                best_test.update({f"{k}_{name}": v for k, v in mt.items()})
                msg = " | ".join(color_metric(k, v) for k, v in mt.items())
                logging.info(f"[{color_split('TEST')}:{name}] {msg}")

        return best_dev, best_test

    if cfg.class_weighting == "none":
        ce_weights = None
        logging.info("Class weighting: none")
    elif cfg.class_weighting == "balanced":
        y_all = _gather_all_labels(mm_loader, avg_mode, segment_length=cfg.segment_length)
        classes = np.arange(model_num_classes)
        class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_all)
        ce_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
        logging.info(f"Class weighting: balanced -> {class_weights.tolist()}")
    elif cfg.class_weighting == "manual":
        w = getattr(cfg, "class_weights", None)
        if not isinstance(w, (list, tuple)) or len(w) != model_num_classes:
            raise ValueError(
                f"class_weighting='manual' requires class_weights list of length {model_num_classes}."
            )
        class_weights = [float(x) for x in w]
        ce_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
        logging.info(f"Class weighting: manual -> {class_weights}")
    else:
        raise ValueError(f"Unknown class_weighting: {cfg.class_weighting}")

    model = _build_model(cfg, in_dim, seq_len, model_num_classes, device)

    if cfg.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    elif cfg.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg.lr)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")
    logging.info(f"Optimizer: {cfg.optimizer}, learning rate: {cfg.lr}")

    steps_per_epoch = sum(1 for b in mm_loader if b is not None)
    scheduler = SmartScheduler(
        scheduler_type=cfg.scheduler_type,
        optimizer=optimizer,
        config=cfg,
        steps_per_epoch=steps_per_epoch,
    )

    criterion = nn.CrossEntropyLoss(weight=ce_weights)

    best_score = -1.0
    best_dev, best_test = {}, {}
    patience = 0
    best_ckpt_path = None

    for epoch in range(cfg.num_epochs):
        logging.info(f"=== EPOCH {epoch + 1}/{cfg.num_epochs} ===")
        model.train()
        tot_loss, tot_n = 0.0, 0
        tr_y, tr_p = [], []

        for batch_idx, batch in enumerate(tqdm(mm_loader, desc="Train")):
            if batch is None:
                continue
            X, keep, mask = _stack_face_features(batch["features"], avg_mode, segment_length=cfg.segment_length)

            y = _filter_labels(batch["labels"], keep).to(device)

            if X.ndim == 2:
                X = X.unsqueeze(1)

            logits = model(
                X.to(device, non_blocking=True),
                mask=mask.to(device, non_blocking=True) if mask is not None else None,
            )
            if batch_idx == 0:
                dbg_dump_logits(logits, cfg.print_logits, prefix="[DBG:TRAIN:final]", max_rows=5, max_cols=logits.size(1))

            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step(batch_level=True)

            bs = y.size(0)
            tot_loss += loss.item() * bs
            tot_n += bs

            tr_y.append(_filter_labels(batch["labels"], keep).cpu())
            tr_p.append(logits.argmax(dim=1).detach().cpu())

        train_loss = tot_loss / max(1, tot_n)
        tr_y_np = torch.cat(tr_y).numpy() if tr_y else np.array([])
        tr_p_np = torch.cat(tr_p).numpy() if tr_p else np.array([])
        if tr_y_np.size > 0:
            m_tr = _metrics(tr_y_np, tr_p_np, metrics_num_classes)
            parts = [
                f"Loss={train_loss:.4f}",
                color_metric("UAR", m_tr["UAR"]),
                color_metric("MF1", m_tr["MF1"]),
            ]
            for c in range(metrics_num_classes):
                key = f"recall_c{c}_{CLASS_LABELS.get(c, f'class{c}')}"
                if key in m_tr:
                    parts.append(color_metric(key, m_tr[key]))
            logging.info(f"[{color_split('TRAIN')}] " + " | ".join(parts))
        else:
            logging.info(f"[{color_split('TRAIN')}] Loss={train_loss:.4f} | (empty metrics)")

        cur_dev = {}
        if dev_loaders:
            for name, ldr in dev_loaders.items():
                md = _eval_epoch(
                    cfg,
                    model,
                    ldr,
                    device,
                    avg_mode,
                    metrics_num_classes,
                    criterion,
                    segment_length=cfg.segment_length,
                )
                if md:
                    cur_dev.update({f"{k}_{name}": v for k, v in md.items()})
                    msg = " | ".join(color_metric(k, v) for k, v in md.items())
                    logging.info(f"[{color_split('DEV')}:{name}] {msg}")

        cur_test = {}
        if test_loaders:
            for name, ldr in test_loaders.items():
                mt = _eval_epoch(
                    cfg,
                    model,
                    ldr,
                    device,
                    avg_mode,
                    metrics_num_classes,
                    criterion,
                    segment_length=cfg.segment_length,
                )
                if mt:
                    cur_test.update({f"{k}_{name}": v for k, v in mt.items()})
                    msg = " | ".join(color_metric(k, v) for k, v in mt.items())
                    logging.info(f"[{color_split('TEST')}:{name}] {msg}")

        mf1_dev = _mean_metric(cur_dev, "MF1")
        mf1_test = _mean_metric(cur_test, "MF1")
        score = float(np.nanmean([mf1_dev, mf1_test]))
        if not np.isnan(mf1_dev) and not np.isnan(mf1_test):
            logging.info(f"[AVG] {color_metric('MF1_AVG', score)}")

        scheduler.step(score)

        if score > best_score:
            best_score = score
            best_dev, best_test = cur_dev, cur_test
            patience = 0
            os.makedirs(cfg.checkpoint_dir, exist_ok=True)
            ckpt = Path(cfg.checkpoint_dir) / f"best_ep{epoch+1}_mf1avg_{best_score:.4f}.pt"
            torch.save(model.state_dict(), ckpt)
            best_ckpt_path = str(ckpt)
            logging.info(f"Saved best model (mf1avg={best_score:.4f}): {ckpt.name}")
        else:
            patience += 1
            if patience >= cfg.max_patience:
                logging.info("Early stopping.")
                break

    # ONE-TIME EXPORT AFTER TRAIN
    export_dir = "pkl_logits"
    os.makedirs(export_dir, exist_ok=True)

    if best_ckpt_path is not None:
        state = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(state)
        model.eval()

        if test_loaders:
            for split_name, ldr in test_loaders.items():
                out_path = os.path.join(export_dir, f"{cfg.model_name.lower()}_{split_name}_best.pkl")
                export_logits_to_pkl(
                    cfg=cfg,
                    model=model,
                    loader=ldr,
                    device=device,
                    avg_mode=avg_mode,
                    out_path=out_path,
                    segment_length=cfg.segment_length,
                )

    return best_dev, best_test
