from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .registry import register_encoder
from .transformer import AttentionPooling  


def _import_mamba_v1():
    try:
        from mamba_ssm.modules.mamba_simple import Mamba  
        return Mamba
    except Exception:
        try:
            from mamba_ssm import Mamba  
            return Mamba
        except Exception as e:
            raise ImportError(
                "Mamba v1 not available. Install dependency: `pip install mamba-ssm` "
                "(and possibly `pip install causal-conv1d` if your platform requires it)."
            ) from e


def _import_mamba_v2():
    candidates = [
        ("mamba_ssm.modules.mamba2", "Mamba2"),
        ("mamba_ssm.modules.mamba2_simple", "Mamba2"),
        ("mamba_ssm", "Mamba2"),
    ]
    last_err = None
    for mod, name in candidates:
        try:
            m = __import__(mod, fromlist=[name])
            return getattr(m, name)
        except Exception as e:
            last_err = e
            continue
    raise ImportError(
        "Mamba v2 (Mamba2) not available. Install/upgrade dependency: `pip install -U mamba-ssm`."
    ) from last_err


class _MambaStack(nn.Module):

    def __init__(
        self,
        block_ctor,
        d_model: int,
        num_layers: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
        headdim: Optional[int] = None,
        ngroups: Optional[int] = None,
        chunk_size: Optional[int] = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.drop = nn.Dropout(dropout)

        for _ in range(int(num_layers)):
            kwargs = dict(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            
            if headdim is not None:
                kwargs["headdim"] = headdim
            if ngroups is not None:
                kwargs["ngroups"] = ngroups
            if chunk_size is not None:
                kwargs["chunk_size"] = chunk_size
            try:
                layer = block_ctor(**kwargs)
            except TypeError:
                kwargs.pop("headdim", None)
                kwargs.pop("ngroups", None)
                kwargs.pop("chunk_size", None)
                layer = block_ctor(**kwargs)

            self.layers.append(layer)
            self.norms.append(nn.LayerNorm(d_model))

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
        
        mamba2_headdim: Optional[int] = None,
        mamba2_ngroups: Optional[int] = None,
        mamba2_chunk_size: Optional[int] = None,
     
        _version: str = "v1",
        **_unused,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()
        self.dropout = nn.Dropout(dropout)

        pooling = pooling.lower()
        self.pooling = pooling
        if pooling == "mean":
            self.pool = None
        elif pooling == "attn":
            self.pool = AttentionPooling(d_model, dropout)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        if _version == "v1":
            Mamba = _import_mamba_v1()
            block_ctor = lambda **kw: Mamba(**kw)
        elif _version == "v2":
            Mamba2 = _import_mamba_v2()
            block_ctor = lambda **kw: Mamba2(**kw)
        else:
            raise ValueError(f"Unknown Mamba version: {_version}")

        self.stack = _MambaStack(
            block_ctor=block_ctor,
            d_model=d_model,
            num_layers=num_layers,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            dropout=dropout,
            headdim=mamba2_headdim if _version == "v2" else None,
            ngroups=mamba2_ngroups if _version == "v2" else None,
            chunk_size=mamba2_chunk_size if _version == "v2" else None,
        )

        self.out_dim = d_model

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

        if self.pooling == "mean":
            if mask is None:
                return h.mean(dim=1)
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(h.dtype)
            return (h * mask.unsqueeze(-1)).sum(dim=1) / denom
        else:
            return self.pool(h, mask=mask)


@register_encoder("mamba1")
def build_mamba1_encoder(**kwargs):
    kwargs["_version"] = "v1"
    kwargs.pop("nhead", None)
    kwargs.pop("dim_feedforward", None)
    return MambaSequenceEncoder(**kwargs)


@register_encoder("mamba2")
def build_mamba2_encoder(**kwargs):
    kwargs["_version"] = "v2"
    kwargs.pop("nhead", None)
    kwargs.pop("dim_feedforward", None)
    return MambaSequenceEncoder(**kwargs)
