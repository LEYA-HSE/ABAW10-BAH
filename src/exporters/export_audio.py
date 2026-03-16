# coding: utf-8
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
import torch.nn as nn
import librosa
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from src.utils.config_loader import ConfigLoader
from src.utils.logger_setup import setup_logger


def _device_from_str(s: str) -> torch.device:
    s = (s or "cpu").lower()
    if s.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if s == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _import_mamba_v1():
    try:
        from mamba_ssm.modules.mamba_simple import Mamba  # type: ignore
        return Mamba
    except Exception:
        from mamba_ssm import Mamba  # type: ignore
        return Mamba


class _MambaStack(nn.Module):
    def __init__(self, d_model: int, num_layers: int, d_state: int, d_conv: int, expand: int, dropout: float):
        super().__init__()
        Mamba = _import_mamba_v1()
        self.layers = nn.ModuleList(
            [Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(int(num_layers))]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(int(num_layers))])
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer, ln in zip(self.layers, self.norms):
            h = layer(x)
            x = ln(x + self.drop(h))
            if mask is not None:
                x = x * mask.unsqueeze(-1).to(x.dtype)
        return x


class MambaSequenceEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_layers: int,
        dropout: float,
        pooling: str = "mean",
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(int(input_dim), int(d_model)) if int(input_dim) != int(d_model) else nn.Identity()
        self.dropout = nn.Dropout(float(dropout))
        self.pooling = str(pooling).lower()
        if self.pooling != "mean":
            raise ValueError(f"Unsupported pooling in audio exporter: {self.pooling}")
        self.stack = _MambaStack(
            d_model=int(d_model),
            num_layers=int(num_layers),
            d_state=int(mamba_d_state),
            d_conv=int(mamba_d_conv),
            expand=int(mamba_expand),
            dropout=float(dropout),
        )
        self.out_dim = int(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
            if mask is None:
                mask = torch.ones(x.size(0), 1, dtype=torch.bool, device=x.device)

        x = self.input_proj(x)
        x = self.dropout(x)

        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)

        h = self.stack(x, mask=mask)
        if mask is None:
            return h.mean(dim=1)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(h.dtype)
        return (h * mask.unsqueeze(-1)).sum(dim=1) / denom


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, dropout: float):
        super().__init__()
        hidden = max(64, int(in_dim) // 2)
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), hidden),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BAHClassifier(nn.Module):
    def __init__(self, mcfg: dict, input_dim: int):
        super().__init__()
        self.encoder = MambaSequenceEncoder(
            input_dim=input_dim,
            d_model=int(mcfg["d_model"]),
            num_layers=int(mcfg["num_layers"]),
            dropout=float(mcfg["dropout"]),
            pooling=str(mcfg.get("pooling", "mean")),
            mamba_d_state=int(mcfg.get("mamba_d_state", 16)),
            mamba_d_conv=int(mcfg.get("mamba_d_conv", 4)),
            mamba_expand=int(mcfg.get("mamba_expand", 2)),
        )
        self.head = MLPHead(
            self.encoder.out_dim,
            num_classes=int(mcfg.get("num_classes", 2)),
            dropout=float(mcfg["dropout"]),
        )


def _pick_audio_column(df: pd.DataFrame, csv_path: Path) -> str:
    for column in ("audio_path", "audio_name", "video_name", "video_path"):
        if column in df.columns:
            return column
    raise KeyError(
        f"CSV '{csv_path}' must contain one of: 'audio_path', 'audio_name', 'video_name', 'video_path'"
    )


def _resolve_dataset_path(template_or_path: str, base_dir: str, split: str | None = None) -> Path:
    return Path(str(template_or_path).format(base_dir=base_dir, split=split or ""))


def _resolve_audio_path(audio_dir: str, raw_value: str, audio_exts: list[str]) -> Path:
    raw = str(raw_value).strip()
    candidate = Path(raw)
    if candidate.is_file():
        return candidate

    joined = Path(audio_dir) / raw
    if joined.is_file():
        return joined

    stem = Path(raw).stem
    for ext in audio_exts:
        ext_candidate = Path(audio_dir) / f"{stem}{ext}"
        if ext_candidate.is_file():
            return ext_candidate

    raise FileNotFoundError(f"Audio not found for '{raw_value}' under '{audio_dir}'")


def _iter_split_audios(cfg, split: str) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []

    for _, ds_cfg in getattr(cfg, "datasets", {}).items():
        csv_tpl = ds_cfg.get("csv_path")
        base_dir = ds_cfg.get("base_dir")
        audio_dir_tpl = ds_cfg.get("audio_dir")
        if not csv_tpl or not base_dir or not audio_dir_tpl:
            continue

        csv_path = Path(str(csv_tpl).format(base_dir=base_dir, split=split))
        audio_dir = _resolve_dataset_path(str(audio_dir_tpl), base_dir, split=split)
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        audio_column = _pick_audio_column(df, csv_path)
        for _, row in df.iterrows():
            raw_audio_path = str(row[audio_column])
            sample_id = Path(raw_audio_path).stem
            resolved = _resolve_audio_path(str(audio_dir), raw_audio_path, cfg.audio_exts)
            items.append((sample_id, resolved))

    if not items:
        raise ValueError(f"No audios found for split='{split}'. Configure datasets.*.audio_dir first.")

    dedup: dict[str, Path] = {}
    for sample_id, path in items:
        dedup[sample_id] = path
    return sorted(dedup.items(), key=lambda item: item[0])


@torch.inference_mode()
def extract_layer10(audio_path: str, processor: Wav2Vec2Processor, w2v: Wav2Vec2Model, device: torch.device):
    signal, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(signal, sampling_rate=sr, return_tensors="pt", padding=True)
    input_values = inputs["input_values"].to(device)
    out = w2v(input_values, output_hidden_states=True)
    layer10 = out.hidden_states[10]
    mask = torch.ones(layer10.shape[0], layer10.shape[1], dtype=torch.bool, device=device)
    return layer10.to(device), mask


def _load_audio_models(cfg, device: torch.device):
    processor = Wav2Vec2Processor.from_pretrained(cfg.audio_wav2vec_model)
    w2v = Wav2Vec2Model.from_pretrained(cfg.audio_wav2vec_model).to(device).eval()

    ckpt_path = Path(cfg.audio_checkpoint_path)
    if not str(cfg.audio_checkpoint_path).strip():
        raise ValueError("audio_export.checkpoint_path is empty")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Audio checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    mcfg = ckpt["cfg"]["model"]
    clf = BAHClassifier(mcfg, input_dim=1024).to(device).eval()
    clf.load_state_dict(ckpt["model_state"], strict=True)

    return processor, w2v, clf


@torch.inference_mode()
def predict_audio_full(
    audio_path: str,
    processor: Wav2Vec2Processor,
    w2v: Wav2Vec2Model,
    clf: BAHClassifier,
    device: torch.device,
):
    x, mask = extract_layer10(audio_path, processor, w2v, device)
    z = clf.encoder(x, mask=mask)
    h = clf.head.net[0](z)
    h = clf.head.net[1](h)
    h = clf.head.net[2](h)
    logits = clf.head.net[3](h)
    prob = torch.softmax(logits, dim=-1)

    return {
        "prob": prob.squeeze(0).detach().cpu().numpy().astype("float32"),
        "logits": logits.squeeze(0).detach().cpu().numpy().astype("float32"),
        "embeddings": h.squeeze(0).detach().cpu().numpy().astype("float32"),
    }


def _load_existing_artifacts(out_path: Path, overwrite: bool) -> dict[str, dict]:
    if overwrite or not out_path.exists():
        return {}
    with out_path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict in artifact '{out_path}', got {type(data)}")
    return data


def _export_split(cfg, split: str) -> None:
    device = _device_from_str(cfg.device)
    out_path = Path(cfg.audio_export_output_dir) / f"{split}.pkl"
    out = _load_existing_artifacts(out_path, overwrite=cfg.audio_export_overwrite_cache)

    items = _iter_split_audios(cfg, split)
    pending = [(sample_id, path) for sample_id, path in items if sample_id not in out]
    if not pending:
        logging.info("Audio export split=%s is already up to date: %s", split, out_path)
        return

    processor, w2v, clf = _load_audio_models(cfg, device)
    for sample_id, audio_path in tqdm(pending, desc=f"Audio export -> {split}.pkl"):
        out[sample_id] = predict_audio_full(str(audio_path), processor, w2v, clf, device)
        out[sample_id]["name"] = sample_id

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info("Saved %d audio artifacts to %s", len(out), out_path)


def run_audio_export(
    config_path: str = "config.toml",
    *,
    configure_logging: bool = True,
    splits: list[str] | None = None,
) -> None:
    if configure_logging:
        setup_logger(logging.INFO)
    cfg = ConfigLoader(config_path)
    cfg.show_config()

    export_splits = list(splits) if splits is not None else list(cfg.audio_export_splits)
    for split in export_splits:
        _export_split(cfg, split)
