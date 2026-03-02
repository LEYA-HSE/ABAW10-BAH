from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn

from .registry import register_encoder

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        self.d_model = d_model
        self.register_buffer("pe", self._build_pe(max_len, d_model), persistent=False)

    @staticmethod
    def _build_pe(max_len: int, d_model: int) -> torch.Tensor:
        import math
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        if L > self.pe.size(1):
            self.pe = self._build_pe(L, self.d_model).to(device=x.device, dtype=x.dtype)
        return x + self.pe[:, :L, :]


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.score(x).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        w = torch.softmax(logits, dim=-1).unsqueeze(-1)
        return (x * w).sum(dim=1)

class TransformerSequenceEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        pooling: str = "mean",
        max_len: int = 4096,
        norm_first: bool = True,
        **_unused,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=int(max_len))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=bool(norm_first),
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

        pooling = pooling.lower()
        self.pooling = pooling
        if pooling == "mean":
            self.pool = None
        elif pooling == "attn":
            self.pool = AttentionPooling(d_model, dropout)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        self.out_dim = d_model

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
            if mask is None:
                mask = torch.ones(x.size(0), 1, dtype=torch.bool, device=x.device)

        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.dropout(x)

        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask = ~mask

        h = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        if self.pooling == "mean":
            if mask is None:
                return h.mean(dim=1)
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(h.dtype)
            return (h * mask.unsqueeze(-1)).sum(dim=1) / denom
        else:
            return self.pool(h, mask=mask)

@register_encoder("transformer")
def build_transformer_encoder(**kwargs):
    return TransformerSequenceEncoder(**kwargs)
