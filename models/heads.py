from __future__ import annotations
import torch
import torch.nn as nn

class MLPHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, dropout: float):
        super().__init__()
        hidden = max(64, in_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
