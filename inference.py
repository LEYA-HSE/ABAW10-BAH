from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model


# ===================== EDIT THESE =====================
AUDIO_ROOT = Path("")          # папка с аудио
OUT_PKL    = Path("")      # куда сохранить pickle
CKPT_PATH  = Path("")            # pt файл
W2V_MODEL  = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
DEVICE_STR = "cuda"  # "cuda" | "cpu"
AUDIO_EXTS = {".wav"}
# ======================================================


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
        **_unused,
    ):
        super().__init__()
        self.input_proj = nn.Linear(int(input_dim), int(d_model)) if int(input_dim) != int(d_model) else nn.Identity()
        self.dropout = nn.Dropout(float(dropout))
        self.pooling = str(pooling).lower()
        if self.pooling != "mean":
            raise ValueError(f"Unsupported pooling in this script: {self.pooling}")
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

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = self.encoder(x, mask=mask)
        return self.head(z)


def iter_audio_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            files.append(p)
    files.sort()
    return files


@torch.inference_mode()
def extract_layer10(audio_path: str, processor: Wav2Vec2Processor, w2v: Wav2Vec2Model, device: torch.device):
    signal, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(signal, sampling_rate=sr, return_tensors="pt", padding=True)
    input_values = inputs["input_values"].to(device)

    out = w2v(input_values, output_hidden_states=True)
    layer10 = out.hidden_states[10]  # (1, T, 1024)

    mask = torch.ones(layer10.shape[0], layer10.shape[1], dtype=torch.bool, device=device)
    return layer10.to(device), mask


@torch.inference_mode()
def predict_audio_full(audio_path: str, processor: Wav2Vec2Processor, w2v: Wav2Vec2Model, clf: BAHClassifier, device: torch.device):
    x, mask = extract_layer10(audio_path, processor, w2v, device)

    z = clf.encoder(x, mask=mask)   

    h = clf.head.net[0](z)
    h = clf.head.net[1](h)
    h = clf.head.net[2](h)        
    embeddings = h

    logits = clf.head.net[3](embeddings)
    prob = torch.softmax(logits, dim=-1)

    return {
        "prob": prob.squeeze(0).detach().cpu().numpy().astype("float32"),
        "logits": logits.squeeze(0).detach().cpu().numpy().astype("float32"),
        "embeddings": embeddings.squeeze(0).detach().cpu().numpy().astype("float32"),
    }


def main():
    device = _device_from_str(DEVICE_STR)

    ckpt = torch.load(str(CKPT_PATH), map_location="cpu")
    cfg = ckpt["cfg"]
    mcfg = cfg["model"]

    processor = Wav2Vec2Processor.from_pretrained(W2V_MODEL)
    w2v = Wav2Vec2Model.from_pretrained(W2V_MODEL).to(device).eval()

    clf = BAHClassifier(mcfg, input_dim=1024).to(device).eval()
    clf.load_state_dict(ckpt["model_state"], strict=True)

    audios = iter_audio_files(AUDIO_ROOT)

    out: Dict[str, Any] = {}
    for ap in audios:
        rel_key = str(ap.relative_to(AUDIO_ROOT)).replace("\\", "/")
        out[rel_key] = predict_audio_full(str(ap), processor, w2v, clf, device)

    OUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PKL.open("wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
