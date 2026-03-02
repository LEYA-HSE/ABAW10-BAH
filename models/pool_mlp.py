from __future__ import annotations
from typing import Optional, Literal

import torch
import torch.nn as nn

from .registry import register_encoder


PoolMode = Literal["mean", "mean_std", "mean_std_max"]


def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=1)
    m = mask.to(x.dtype).unsqueeze(-1)  
    denom = m.sum(dim=1).clamp_min(1.0)
    return (x * m).sum(dim=1) / denom


def _masked_var(x: torch.Tensor, mask: Optional[torch.Tensor], mean: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return x.var(dim=1, unbiased=False)
    m = mask.to(x.dtype).unsqueeze(-1)
    denom = m.sum(dim=1).clamp_min(1.0)
    diff2 = (x - mean.unsqueeze(1)) ** 2
    return (diff2 * m).sum(dim=1) / denom


def _masked_max(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return x.max(dim=1).values
    neg_inf = torch.finfo(x.dtype).min
    x2 = x.masked_fill(~mask.unsqueeze(-1), neg_inf)
    return x2.max(dim=1).values


class MeanStatsMLPEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        num_layers: int = 2,       
        dropout: float = 0.1,
        pooling_mlp: PoolMode = "mean_std",
        mlp_hidden: Optional[int] = None,
        activation: str = "gelu",
    ):
        super().__init__()
        self.pooling_mlp: PoolMode = pooling_mlp

        if pooling_mlp == "mean":
            pooled_dim = input_dim
        elif pooling_mlp == "mean_std":
            pooled_dim = input_dim * 2
        elif pooling_mlp == "mean_std_max":
            pooled_dim = input_dim * 3
        else:
            raise ValueError(f"Unknown pooling: {pooling_mlp}")

        hidden = int(mlp_hidden) if mlp_hidden is not None else int(d_model)

        if activation.lower() == "relu":
            act = nn.ReLU()
        elif activation.lower() == "gelu":
            act = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        layers = []
        in_dim = pooled_dim
        L = int(num_layers)
        if L <= 0:
            raise ValueError("num_layers must be >= 1 for pool_mlp")

        for i in range(L):
            out_dim = hidden if i < L - 1 else int(d_model)
            layers.append(nn.Linear(in_dim, out_dim))
            if i < L - 1:
                layers.append(act)
                if dropout and dropout > 0:
                    layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)
        self.out_dim = int(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        mean = _masked_mean(x, mask)
        if self.pooling_mlp == "mean":
            pooled = mean
        else:
            var = _masked_var(x, mask, mean)
            std = torch.sqrt(var.clamp_min(1e-8))
            if self.pooling_mlp == "mean_std":
                pooled = torch.cat([mean, std], dim=-1)
            else:
                mx = _masked_max(x, mask)
                pooled = torch.cat([mean, std, mx], dim=-1)
        return self.mlp(pooled)


@register_encoder("pool_mlp")
def build_pool_mlp_encoder(**kwargs) -> MeanStatsMLPEncoder:
    return MeanStatsMLPEncoder(**kwargs)