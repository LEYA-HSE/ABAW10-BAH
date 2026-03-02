from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .registry import register_encoder
from .transformer import SinusoidalPositionalEncoding


def _norm_mamba_version(v: str) -> str:
    s = (v or "").lower().strip()
    if s in ("v2", "mamba2", "m2"):
        return "v2"
    return "v1"


class _MambaResidualBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float,
        version: str,
        d_state: int,
        d_conv: int,
        expand: int,
        headdim: Optional[int] = None,
        ngroups: Optional[int] = None,
        chunk_size: Optional[int] = None,
    ):
        super().__init__()
        self.version = _norm_mamba_version(version)

        if self.version == "v2":
            from .mamba import _import_mamba_v2  

            Mamba2 = _import_mamba_v2()
            kwargs = dict(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            if headdim is not None:
                kwargs["headdim"] = int(headdim)
            if ngroups is not None:
                kwargs["ngroups"] = int(ngroups)
            if chunk_size is not None:
                kwargs["chunk_size"] = int(chunk_size)
            self.mamba = Mamba2(**kwargs)
        else:
            from .mamba import _import_mamba_v1  

            Mamba = _import_mamba_v1()
            self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        y = self.mamba(x)
        y = self.drop(y)
        x = self.norm(x + y)
        if mask is not None:
            x = x * mask.to(x.dtype).unsqueeze(-1)
        return x


class HybridAttnMambaEncoder(nn.Module):
   

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        pooling: str = "mean",
    
        hybrid_attn_every: int = 3,
        hybrid_mamba_version: str = "v1",
     
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba2_headdim: Optional[int] = None,
        mamba2_ngroups: Optional[int] = None,
        mamba2_chunk_size: Optional[int] = None,
       
        max_len: int = 4096,
        norm_first: bool = True,
        **_unused,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=int(max_len))
        self.in_drop = nn.Dropout(dropout)

        attn_every = int(hybrid_attn_every) if hybrid_attn_every is not None else 3
        attn_every = max(1, attn_every)
        mamba_version = _norm_mamba_version(hybrid_mamba_version)

        blocks: list[nn.Module] = []
        for i in range(int(num_layers)):
            use_attn = (i % attn_every) == 0
            if use_attn:
                blocks.append(
                    nn.TransformerEncoderLayer(
                        d_model=d_model,
                        nhead=int(nhead),
                        dim_feedforward=int(dim_feedforward),
                        dropout=float(dropout),
                        batch_first=True,
                        norm_first=bool(norm_first),
                        activation="gelu",
                    )
                )
            else:
                blocks.append(
                    _MambaResidualBlock(
                        d_model=d_model,
                        dropout=float(dropout),
                        version=mamba_version,
                        d_state=int(mamba_d_state),
                        d_conv=int(mamba_d_conv),
                        expand=int(mamba_expand),
                        headdim=mamba2_headdim,
                        ngroups=mamba2_ngroups,
                        chunk_size=mamba2_chunk_size,
                    )
                )
        self.blocks = nn.ModuleList(blocks)

        pooling = pooling.lower()
        self.pooling = pooling
        if pooling == "mean":
            self.pool = None
        elif pooling == "attn":
            from .transformer import AttentionPooling

            self.pool = AttentionPooling(d_model, dropout)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        self.out_dim = int(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
            if mask is None:
                mask = torch.ones(x.size(0), 1, dtype=torch.bool, device=x.device)

        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.in_drop(x)

        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask = ~mask

        for blk in self.blocks:
            if isinstance(blk, nn.TransformerEncoderLayer):
                x = blk(x, src_key_padding_mask=src_key_padding_mask)
            else:
                x = blk(x, mask=mask)

        if self.pooling == "mean":
            if mask is None:
                return x.mean(dim=1)
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(x.dtype)
            return (x * mask.unsqueeze(-1)).sum(dim=1) / denom
        else:
            return self.pool(x, mask=mask)


@register_encoder("hybrid_attn_mamba")
def build_hybrid_attn_mamba_encoder(**kwargs) -> HybridAttnMambaEncoder:
    return HybridAttnMambaEncoder(**kwargs)
