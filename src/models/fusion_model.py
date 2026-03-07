# coding: utf-8
"""
Simple multimodal fusion model.

Input tensors (already prepared):
  emb[mod]    -> (N, Dm)
  prob[mod]   -> (N, 2)
  logits[mod] -> (N, 2)
  labels      -> (N,)
Optional:
  mask[mod]   -> (N,) bool

No Dataset/DataLoader here.
Batches are created with plain index slicing.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ModalityProjector(nn.Module):
    def __init__(self, in_dim: int, d_model: int, drop: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(drop),
        )

    def forward(self, x):
        return self.proj(x)


class AttnFusion(nn.Module):
    def __init__(self, token_dim: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.Tanh(),
            nn.Linear(token_dim, 1),
        )

    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # tokens: (B, M, D)
        scores = self.scorer(tokens).squeeze(-1)  # (B, M)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e4)
        w = torch.softmax(scores, dim=-1)  # (B, M)
        fused = torch.sum(tokens * w.unsqueeze(-1), dim=1)  # (B, D)
        return fused, w


class ClassWeightedFusion(nn.Module):
    """
    Learn class-specific modality weights.
    x: (B, M, 2), W: (M, 2), out: (B, 2)
    """

    def __init__(self, num_modalities: int):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(num_modalities, 2))
        nn.init.normal_(self.W, std=0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        return torch.sum(x * self.W.unsqueeze(0), dim=1)


class PrototypeHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int = 2, k_per_class: int = 4, tau: float = 0.07):
        super().__init__()
        self.num_classes = num_classes
        self.k_per_class = k_per_class
        self.tau = tau
        self.P = nn.Parameter(torch.randn(num_classes, k_per_class, d_model))
        nn.init.normal_(self.P, std=0.02)

    def forward(self, h: torch.Tensor):
        # h: (B, D)
        h = F.normalize(h, dim=-1)
        p = F.normalize(self.P, dim=-1)  # (C, K, D)
        sim = torch.einsum("bd,ckd->bck", h, p)  # (B, C, K)
        logits = torch.logsumexp(sim / self.tau, dim=-1)  # (B, C)
        return logits

    def diversity_loss(self, margin_intra: float = 0.3, margin_inter: float = 0.3):
        p = F.normalize(self.P, dim=-1)
        loss = 0.0
        c = self.num_classes
        k = self.k_per_class

        for ci in range(c):
            s = p[ci] @ p[ci].t()
            eye = torch.eye(k, device=s.device, dtype=s.dtype)
            s = s * (1 - eye)
            loss = loss + F.relu(s - margin_intra).mean()

        for c1 in range(c):
            for c2 in range(c1 + 1, c):
                s = p[c1] @ p[c2].t()
                loss = loss + F.relu(margin_inter - s).mean()
        return loss


class ExchangeFusionTransformer(nn.Module):
    """
    Modality token exchange via TransformerEncoder.
    """

    def __init__(
        self,
        token_dim: int,
        d_model: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        ff_mult: int = 4,
        drop: float = 0.1,
        use_cls: bool = True,
        max_modalities: int = 4,
    ):
        super().__init__()
        self.use_cls = use_cls
        self.in_proj = nn.Identity() if token_dim == d_model else nn.Linear(token_dim, d_model)

        self.mod_emb = nn.Parameter(torch.zeros(1, max_modalities + (1 if use_cls else 0), d_model))
        nn.init.normal_(self.mod_emb, std=0.02)

        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_mult * d_model,
            dropout=drop,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):
        # tokens: (B, M, Dtok), mask: (B, M) True=exists
        b, m, _ = tokens.shape
        x = self.in_proj(tokens)
        key_padding_mask = ~mask

        if self.use_cls:
            cls = self.cls_token.expand(b, 1, -1)
            x = torch.cat([cls, x], dim=1)
            cls_pad = torch.zeros(b, 1, dtype=torch.bool, device=x.device)
            key_padding_mask = torch.cat([cls_pad, key_padding_mask], dim=1)

        x = x + self.mod_emb[:, : x.size(1), :]
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.out_norm(x)

        if self.use_cls:
            return x[:, 0, :]

        w = mask.float().unsqueeze(-1)
        return (x * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)


class MultiModalModel(nn.Module):
    """
    Config:
      modalities: ["face","text","scene","audio"] subset
      input_type: "logits" | "prob" | "emb" | "emb+prob"
      fusion: "class_weighted" | "attn" | "concat_mlp" | "exchange_transformer"
      d_model: int
      drop: float
      use_prototypes: bool
    """

    def __init__(self, cfg: Dict, emb_dims: Dict[str, int]):
        super().__init__()
        self.cfg = cfg
        self.modalities: List[str] = cfg["modalities"]
        self.input_type: str = cfg["input_type"]
        self.fusion: str = cfg["fusion"]
        self.d_model: int = int(cfg.get("d_model", 256))
        drop = float(cfg.get("drop", 0.1))

        self.proj = nn.ModuleDict()
        if "emb" in self.input_type:
            for m in self.modalities:
                self.proj[m] = ModalityProjector(int(emb_dims[m]), self.d_model, drop=drop)

        if self.input_type in ["logits", "prob"]:
            token_dim = 2
        elif self.input_type == "emb":
            token_dim = self.d_model
        elif self.input_type == "emb+prob":
            token_dim = self.d_model + 2
        else:
            raise ValueError("input_type must be one of: logits, prob, emb, emb+prob")

        if self.fusion == "class_weighted":
            if self.input_type not in ["logits", "prob"]:
                raise ValueError("class_weighted fusion supports only logits/prob input")
            self.class_fuser = ClassWeightedFusion(len(self.modalities))
            self.classifier = None
            self.to_d_model = None
        elif self.fusion == "attn":
            self.attn = AttnFusion(token_dim=token_dim)
            self.to_d_model = None if token_dim == self.d_model else MLP(token_dim, self.d_model, hidden=max(128, self.d_model), drop=drop)
            self.classifier = nn.Linear(self.d_model, 2)
        elif self.fusion == "concat_mlp":
            self.concat_mlp = MLP(len(self.modalities) * token_dim, self.d_model, hidden=max(256, self.d_model), drop=drop)
            self.classifier = nn.Linear(self.d_model, 2)
        elif self.fusion == "exchange_transformer":
            self.exchange = ExchangeFusionTransformer(
                token_dim=token_dim,
                d_model=self.d_model,
                n_layers=int(cfg.get("x_layers", 2)),
                n_heads=int(cfg.get("x_heads", 4)),
                ff_mult=int(cfg.get("x_ff_mult", 4)),
                drop=drop,
                use_cls=bool(cfg.get("x_use_cls", True)),
                max_modalities=len(self.modalities),
            )
            self.classifier = nn.Linear(self.d_model, 2)
        else:
            raise ValueError("fusion must be one of: class_weighted, attn, concat_mlp, exchange_transformer")

        self.use_prototypes = bool(cfg.get("use_prototypes", False))
        if self.use_prototypes:
            self.proto = PrototypeHead(
                d_model=self.d_model,
                num_classes=2,
                k_per_class=int(cfg.get("num_prototypes", 4)),
                tau=float(cfg.get("proto_tau", 0.07)),
            )

    def forward(self, batch: Dict):
        tokens = []
        masks = []
        fused = None
        attn_w = None

        for m in self.modalities:
            m_mask = batch["mask"][m].bool()
            masks.append(m_mask)

            if self.input_type in ["logits", "prob"]:
                x = batch[self.input_type][m]  # (B,2)
                tokens.append(x.unsqueeze(1))
            elif self.input_type == "emb":
                e = batch["emb"][m]            # (B,Dm)
                h = self.proj[m](e)            # (B,d)
                tokens.append(h.unsqueeze(1))
            else:  # emb+prob
                e = batch["emb"][m]
                p = batch["prob"][m]
                h = self.proj[m](e)
                u = torch.cat([h, p], dim=-1)
                tokens.append(u.unsqueeze(1))

        tokens = torch.cat(tokens, dim=1)      # (B,M,Dt)
        mask = torch.stack(masks, dim=1)       # (B,M)

        if self.fusion == "class_weighted":
            out_logits = self.class_fuser(tokens, mask=mask)
        elif self.fusion == "attn":
            fused, attn_w = self.attn(tokens, mask=mask)
            if self.to_d_model is not None:
                fused = self.to_d_model(fused)
            out_logits = self.classifier(fused)
        elif self.fusion == "concat_mlp":
            tokens = tokens.masked_fill(~mask.unsqueeze(-1), 0.0)
            fused = self.concat_mlp(tokens.reshape(tokens.size(0), -1))
            out_logits = self.classifier(fused)
        else:
            fused = self.exchange(tokens, mask=mask)
            out_logits = self.classifier(fused)

        proto_logits = None
        if self.use_prototypes:
            if fused is None:
                fused = out_logits
                if fused.size(-1) != self.d_model:
                    fused = MLP(2, self.d_model, hidden=max(128, self.d_model)).to(fused.device)(fused)
            proto_logits = self.proto(fused)

        return {
            "logits": out_logits,
            "fused": fused,
            "attn_w": attn_w,
            "proto_logits": proto_logits,
        }
