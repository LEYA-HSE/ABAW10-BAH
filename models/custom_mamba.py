from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .registry import register_encoder
from .transformer import AttentionPooling


class CustomMambaBlock(nn.Module):

    def __init__(self, d_input: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(d_input, d_model)
        self.s_B = nn.Linear(d_model, d_model)
        self.s_C = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_input)
        self.norm = nn.LayerNorm(d_input)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x
        x = self.in_proj(x)
        B = self.s_B(x)
        C = self.s_C(x)
        x = x + B + C
        x = self.activation(x)
        x = self.out_proj(x)
        x = self.dropout(x)
        x = self.norm(x + x_in)
        return x


class CustomMambaSequenceEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_layers: int,
        dropout: float,
        pooling: str = "mean",
        **_unused,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()
        self.blocks = nn.ModuleList(
            [CustomMambaBlock(d_model, d_model, dropout=dropout) for _ in range(int(num_layers))]
        )

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
        for blk in self.blocks:
            x = blk(x)

        if self.pooling == "mean":
            if mask is None:
                return x.mean(dim=1)
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(x.dtype)
            return (x * mask.unsqueeze(-1)).sum(dim=1) / denom
        else:
            return self.pool(x, mask=mask)


@register_encoder("custom_mamba")
def build_custom_mamba_encoder(**kwargs):
    return CustomMambaSequenceEncoder(**kwargs)
