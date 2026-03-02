from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_encoder
from .zeros_attention import ZeroSAttention


@dataclass
class ZeroSEncoderConfig:
 
    input_dim: int
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int

    block_size: int
    is_causal: bool
    use_associative: bool
    use_norm: bool
    bias: bool

    dropout: float
    pooling: str  

    force_block_size: bool
    crop_mode: str     


class ZeroSEncoderLayer(nn.Module):
    def __init__(self, cfg: ZeroSEncoderConfig, is_first_layer: bool):
        super().__init__()

        attn_cfg = SimpleNamespace(
            n_embd=cfg.d_model,
            n_head=cfg.nhead,
            bias=cfg.bias,
            dropout=cfg.dropout,
            block_size=cfg.block_size,
            is_first_layer=is_first_layer,
            is_causal=cfg.is_causal,
            use_norm=cfg.use_norm,
            use_associative=cfg.use_associative,
        )
        self.attn = ZeroSAttention(attn_cfg)

        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)

        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.dim_feedforward),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.dim_feedforward, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class ZeroSSequenceEncoder(nn.Module):
   
    out_dim: int

    def __init__(self, cfg: ZeroSEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.out_dim = cfg.d_model

        self.input_proj = nn.Linear(cfg.input_dim, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

        if cfg.pooling == "cls":
            self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        else:
            self.cls = None

        self.layers = nn.ModuleList(
            [ZeroSEncoderLayer(cfg, is_first_layer=(i == 0)) for i in range(cfg.num_layers)]
        )
        self.out_norm = nn.LayerNorm(cfg.d_model)

    def _pad_or_crop_to_block(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        if mask is None:
            mask = torch.ones((B, L), dtype=torch.bool, device=x.device)

        block = int(self.cfg.block_size)
        if L == block:
            return x, mask

        if L > block:
            if self.cfg.crop_mode == "random" and self.training:
                max_off = L - block
                off = int(torch.randint(low=0, high=max_off + 1, size=(1,), device=x.device).item())
                x = x[:, off:off + block, :]
                mask = mask[:, off:off + block]
            else:
                x = x[:, :block, :]
                mask = mask[:, :block]
            return x, mask

        pad = block - L
        x = F.pad(x, (0, 0, 0, pad, 0, 0))
        mask = F.pad(mask, (0, pad), value=False)
        return x, mask

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
            if mask is None:
                mask = torch.ones((x.size(0), 1), dtype=torch.bool, device=x.device)

        if x.ndim != 3:
            raise ValueError(f"Expected x to be (B,L,D) or (B,D), got {tuple(x.shape)}")

        if self.cfg.force_block_size:
            x, mask = self._pad_or_crop_to_block(x, mask)
        else:
            if mask is None:
                mask = torch.ones((x.size(0), x.size(1)), dtype=torch.bool, device=x.device)

        x = x * mask.unsqueeze(-1).to(x.dtype)
        x = self.input_proj(x)
        x = self.dropout(x)
        x = x * mask.unsqueeze(-1).to(x.dtype)

        if self.cls is not None:
            cls = self.cls.expand(x.size(0), -1, -1)
            x = torch.cat([cls, x], dim=1)
            mask = torch.cat([torch.ones((mask.size(0), 1), dtype=torch.bool, device=mask.device), mask], dim=1)

        for layer in self.layers:
            x = layer(x)
            x = x * mask.unsqueeze(-1).to(x.dtype)

        x = self.out_norm(x)

        if self.cls is not None:
            return x[:, 0, :]

        denom = mask.sum(dim=1).clamp(min=1).to(x.dtype)
        pooled = (x * mask.unsqueeze(-1).to(x.dtype)).sum(dim=1) / denom.unsqueeze(-1)
        return pooled


@register_encoder("zeros")
def build_zeros_encoder(
    *,
    input_dim: int,
    d_model: int,
    num_layers: int,
    dropout: float,
    pooling: str = "mean",
    nhead: int = 8,
    dim_feedforward: int = 2048,
    zeros_block_size: int = 2048,
    zeros_is_causal: bool = False,
    zeros_use_associative: bool = True,
    zeros_use_norm: bool = True,
    zeros_bias: bool = True,
    zeros_force_block_size: bool = True,
    zeros_crop_mode: str = "truncate",
    **_unused: Any,
) -> ZeroSSequenceEncoder:
    if d_model % nhead != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead}) for ZeroSAttention.")

    cfg = ZeroSEncoderConfig(
        input_dim=int(input_dim),
        d_model=int(d_model),
        nhead=int(nhead),
        num_layers=int(num_layers),
        dim_feedforward=int(dim_feedforward),
        block_size=int(zeros_block_size),
        is_causal=bool(zeros_is_causal),
        use_associative=bool(zeros_use_associative),
        use_norm=bool(zeros_use_norm),
        bias=bool(zeros_bias),
        dropout=float(dropout),
        pooling=str(pooling),
        force_block_size=bool(zeros_force_block_size),
        crop_mode=str(zeros_crop_mode),
    )
    return ZeroSSequenceEncoder(cfg)
