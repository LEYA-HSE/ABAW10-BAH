from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn

from training.config import ModelConfig
from .registry import build_encoder
from .heads import MLPHead

from . import transformer  
from . import mamba  
from . import custom_mamba  
from . import zeros_encoder  
from . import pool_mlp 
from . import hybrid_attn_mamba 


class BAHClassifier(nn.Module):
    def __init__(self, cfg: ModelConfig, input_dim: int):
        super().__init__()
        self.cfg = cfg

        name = cfg.encoder_name

        encoder_kwargs = dict(
            input_dim=input_dim,
            d_model=cfg.d_model,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        )

        if name == "transformer":
            encoder_kwargs.update(
                dict(
                    nhead=cfg.nhead,
                    pooling=getattr(cfg, "pooling", "mean"),
                    dim_feedforward=cfg.dim_feedforward,
                    norm_first=getattr(cfg, "norm_first", True),
                )
            )
            if hasattr(cfg, "max_len"):
                encoder_kwargs["max_len"] = getattr(cfg, "max_len")

        elif name in ("zeros", "zeros_encoder", "zeros_transformer"):
            encoder_kwargs.update(
                dict(
                    nhead=cfg.nhead,
                    dim_feedforward=cfg.dim_feedforward,
                    norm_first=getattr(cfg, "norm_first", True),
                    zeros_block_size=getattr(cfg, "zeros_block_size", 2048),
                    zeros_is_causal=getattr(cfg, "zeros_is_causal", False),
                    zeros_use_associative=getattr(cfg, "zeros_use_associative", True),
                    zeros_use_norm=getattr(cfg, "zeros_use_norm", True),
                    zeros_bias=getattr(cfg, "zeros_bias", True),
                    zeros_force_block_size=getattr(cfg, "zeros_force_block_size", True),
                    zeros_crop_mode=getattr(cfg, "zeros_crop_mode", "truncate"),
                    pooling=getattr(cfg, "pooling", "mean"),
                )
            )

        elif name in ("mamba1", "mamba2"):
            encoder_kwargs.update(
                dict(
                    mamba_d_state=cfg.mamba_d_state,
                    mamba_d_conv=cfg.mamba_d_conv,
                    mamba_expand=cfg.mamba_expand,
                    pooling=getattr(cfg, "pooling", "mean"),
                )
            )
            if name == "mamba2":
                encoder_kwargs.update(
                    dict(
                        mamba2_headdim=getattr(cfg, "mamba2_headdim", None),
                        mamba2_ngroups=getattr(cfg, "mamba2_ngroups", None),
                        mamba2_chunk_size=getattr(cfg, "mamba2_chunk_size", None),
                        pooling=getattr(cfg, "pooling", "mean"),
                    )
                )

        elif name == "custom_mamba":
            encoder_kwargs.update(
                dict(
                    pooling=getattr(cfg, "pooling", "mean"),
                )
         
            )

        elif name == "pool_mlp":
            encoder_kwargs.update(
                dict(
                    pooling_mlp=getattr(cfg, "pooling_mlp", "mean_std"),
                    mlp_hidden=getattr(cfg, "mlp_hidden", None),
                    activation=getattr(cfg, "activation", "gelu"),
                )
            )

        elif name == "hybrid_attn_mamba":
            encoder_kwargs.update(
                dict(
                    nhead=cfg.nhead,
                    dim_feedforward=cfg.dim_feedforward,
                    norm_first=getattr(cfg, "norm_first", True),
                    max_len=getattr(cfg, "max_len", 4096),
                    pooling=getattr(cfg, "pooling", "mean"),
                    hybrid_attn_every=getattr(cfg, "hybrid_attn_every", 3),
                    hybrid_mamba_version=getattr(cfg, "hybrid_mamba_version", "v1"),
                    mamba_d_state=cfg.mamba_d_state,
                    mamba_d_conv=cfg.mamba_d_conv,
                    mamba_expand=cfg.mamba_expand,
                    mamba2_headdim=getattr(cfg, "mamba2_headdim", None),
                    mamba2_ngroups=getattr(cfg, "mamba2_ngroups", None),
                    mamba2_chunk_size=getattr(cfg, "mamba2_chunk_size", None),
                )
            )

        else:
            pass

        self.encoder = build_encoder(name, **encoder_kwargs)
        self.head = MLPHead(self.encoder.out_dim, num_classes=cfg.num_classes, dropout=cfg.dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = self.encoder(x, mask=mask)
        return self.head(z)