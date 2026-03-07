# coding: utf-8
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.utils.config_loader import ConfigLoader
from src.utils.logger_setup import setup_logger
from src.utils.text_impl import (
    default_text_checkpoint_for_impl,
    default_text_hf_model_for_impl,
    normalize_text_impl,
)


def _device_from_str(s: str) -> torch.device:
    s = (s or "cpu").lower()
    if s.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if s == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _pick_video_column(df: pd.DataFrame, csv_path: Path) -> str:
    for column in ("video_path", "video_name"):
        if column in df.columns:
            return column
    raise KeyError(f"CSV '{csv_path}' must contain 'video_path' or 'video_name'")


def _pick_text_column(df: pd.DataFrame, csv_path: Path, preferred: str | None) -> str:
    if preferred and preferred in df.columns:
        return preferred

    for column in ("transcript", "text", "scene_text", "description"):
        if column in df.columns:
            return column
    raise KeyError(
        f"CSV '{csv_path}' must contain text column. Checked preferred='{preferred}' "
        "and fallbacks: transcript/text/scene_text/description"
    )


def _iter_split_texts(cfg, split: str) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []

    for _, ds_cfg in getattr(cfg, "datasets", {}).items():
        csv_tpl = ds_cfg.get("csv_path")
        base_dir = ds_cfg.get("base_dir")
        if not csv_tpl or not base_dir:
            continue

        csv_path = Path(str(csv_tpl).format(base_dir=base_dir, split=split))
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        video_column = _pick_video_column(df, csv_path)
        text_column = _pick_text_column(df, csv_path, getattr(cfg, "text_input_column", ""))

        for _, row in df.iterrows():
            raw_video_path = str(row[video_column])
            sample_id = Path(raw_video_path).stem
            raw_text = row[text_column]
            text = "" if pd.isna(raw_text) else str(raw_text)
            items.append((sample_id, text))

    if not items:
        raise ValueError(f"No text rows found for split='{split}' from config datasets")

    dedup: dict[str, str] = {}
    for sample_id, text in items:
        dedup[sample_id] = text
    return sorted(dedup.items(), key=lambda item: item[0])


def _load_existing_artifacts(out_path: Path, overwrite: bool) -> dict[str, dict]:
    if overwrite or not out_path.exists():
        return {}
    with out_path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict in artifact '{out_path}', got {type(data)}")
    return data


def _load_pickle_pipeline(path: str, *, impl: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Text checkpoint not found: {p}")
    try:
        with p.open("rb") as handle:
            return pickle.load(handle)
    except ModuleNotFoundError as exc:
        missing_mod = str(getattr(exc, "name", "")) or "unknown module"
        raise ImportError(
            f"Text impl={impl} checkpoint '{p}' requires missing dependency '{missing_mod}'. "
            "Install the dependency (for impl=3 usually: catboost) or switch text_export.impl "
            "to another variant (e.g. 2/6/7)."
        ) from exc


def _binary_probs_and_logits_from_p1(p1: float, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    p1 = float(np.clip(p1, eps, 1.0 - eps))
    p0 = 1.0 - p1
    logit_p1 = float(np.log(p1 / p0))
    probs = np.asarray([p0, p1], dtype=np.float32)
    logits = np.asarray([0.0, logit_p1], dtype=np.float32)
    return probs, logits


def _is_binary_2d_text_artifacts(data: dict[str, dict]) -> bool:
    for payload in data.values():
        if not isinstance(payload, dict):
            continue
        prob = payload.get("prob")
        logits = payload.get("logits")
        if prob is not None and int(np.asarray(prob).reshape(-1).shape[0]) != 2:
            return False
        if logits is not None and int(np.asarray(logits).reshape(-1).shape[0]) != 2:
            return False
    return True


def _predict_text_ml(text: str, pipeline) -> dict[str, np.ndarray]:
    pred_proba = pipeline.predict_proba([text])[0]
    classes = list(getattr(pipeline, "classes_", [0, 1]))
    if 1 in classes:
        p1 = float(pred_proba[classes.index(1)])
    else:
        p1 = float(pred_proba[-1])
    probs, logits = _binary_probs_and_logits_from_p1(p1)
    vectorizer = pipeline.steps[0][1]
    embeddings = vectorizer.transform([text]).todense()
    return {
        "prob": probs,
        "logits": logits,
        "embeddings": np.asarray(embeddings, dtype=np.float32).reshape(-1),
    }


class _TextDataset(Dataset):
    def __init__(self, samples: list[tuple[str, str]], tokenizer, max_length: int):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample_id, text = self.samples[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "sample_id": sample_id,
        }


class _ClassificationModel(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, hidden_size: int):
        super().__init__()
        # Keep original naming from source inference scripts so checkpoint keys match ("bert.*").
        self.bert = backbone
        self.fc = torch.nn.Linear(int(hidden_size), 128)
        self.dropout = torch.nn.Dropout(0.2)
        self.fc2 = torch.nn.Linear(128, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        logits = self.fc(pooled)
        emb = self.dropout(logits)
        logits = self.fc2(emb)
        return logits, emb


def _load_transformer_model(impl: str, ckpt_path: str, model_name: str, device: torch.device):
    try:
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:
        raise ImportError(
            "Text exporter for impl=6/7 requires 'transformers'. Install dependency first."
        ) from exc

    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Text checkpoint not found: {ckpt_path}")

    if not model_name:
        raise ValueError(f"text_export.hf_model_name is empty for impl={impl}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    backbone = AutoModel.from_pretrained(model_name)
    hidden_size = int(getattr(backbone.config, "hidden_size", 768))
    model = _ClassificationModel(backbone, hidden_size=hidden_size)

    if impl == "7":
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return tokenizer, model


@torch.inference_mode()
def _predict_text_transformer(
    pending: list[tuple[str, str]],
    tokenizer,
    model: torch.nn.Module,
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    ds = _TextDataset(pending, tokenizer=tokenizer, max_length=max_length)
    loader = DataLoader(ds, batch_size=max(1, int(batch_size)), shuffle=False)
    sigmoid = torch.nn.Sigmoid()

    for batch in tqdm(loader, desc="Text transformer inference"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        sample_ids = batch["sample_id"]

        logits, embeddings = model(input_ids, attention_mask)
        probs = sigmoid(logits)

        for idx, sample_id in enumerate(sample_ids):
            p1 = float(probs[idx].item())
            probs_2d, logits_2d = _binary_probs_and_logits_from_p1(p1)
            out[sample_id] = {
                "prob": probs_2d,
                "logits": logits_2d,
                "embeddings": embeddings[idx].detach().cpu().numpy().astype("float32").reshape(-1),
            }
    return out


def _export_split(cfg, split: str) -> None:
    impl = normalize_text_impl(str(getattr(cfg, "text_export_impl", "7")))
    ckpt_path = str(getattr(cfg, "text_checkpoint_path_resolved", "")).strip() or default_text_checkpoint_for_impl(impl)
    hf_model_name = (
        str(getattr(cfg, "text_hf_model_name_resolved", "")).strip() or default_text_hf_model_for_impl(impl)
    )

    out_path = Path(cfg.text_export_output_dir) / f"{split}.pkl"
    out = _load_existing_artifacts(out_path, overwrite=cfg.text_export_overwrite_cache)
    if out and not _is_binary_2d_text_artifacts(out):
        logging.info(
            "Text artifact format changed to 2D prob/logits. Rebuilding split=%s from scratch: %s",
            split,
            out_path,
        )
        out = {}

    items = _iter_split_texts(cfg, split)
    pending = [(sample_id, text) for sample_id, text in items if sample_id not in out]
    if not pending:
        logging.info("Text export split=%s is already up to date: %s", split, out_path)
        return

    if impl in {"2", "3"}:
        pipeline = _load_pickle_pipeline(ckpt_path, impl=impl)
        for sample_id, text in tqdm(pending, desc=f"Text export[{impl}] -> {split}.pkl"):
            out[sample_id] = _predict_text_ml(text, pipeline)
            out[sample_id]["name"] = sample_id
    else:
        device = _device_from_str(cfg.device)
        tokenizer, model = _load_transformer_model(
            impl=impl,
            ckpt_path=ckpt_path,
            model_name=hf_model_name,
            device=device,
        )
        predicted = _predict_text_transformer(
            pending,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_length=int(getattr(cfg, "text_max_length", 256)),
            batch_size=int(getattr(cfg, "text_batch_size", 1)),
        )
        for sample_id, payload in predicted.items():
            out[sample_id] = payload
            out[sample_id]["name"] = sample_id

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info("Saved %d text artifacts to %s", len(out), out_path)


def run_text_export(
    config_path: str = "config.toml",
    *,
    configure_logging: bool = True,
    splits: list[str] | None = None,
) -> None:
    if configure_logging:
        setup_logger(logging.INFO)
    cfg = ConfigLoader(config_path)
    cfg.show_config()

    export_splits = list(splits) if splits is not None else list(cfg.text_export_splits)
    for split in export_splits:
        _export_split(cfg, split)
