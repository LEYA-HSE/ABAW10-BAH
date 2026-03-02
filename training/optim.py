from __future__ import annotations

import torch
from torch import nn

from training.config import TrainConfig


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    name = (cfg.optimizer or "adamw").lower()

    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=float(getattr(cfg, "momentum", 0.0)),
            nesterov=bool(getattr(cfg, "nesterov", False)) if float(getattr(cfg, "momentum", 0.0)) > 0 else False,
            weight_decay=cfg.weight_decay,
        )

    raise ValueError(f"Unknown optimizer: {cfg.optimizer}. Use 'adamw' or 'sgd'.")