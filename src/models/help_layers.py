# coding: utf-8
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionWiseFeedForward(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return self.layer_2(x)


class AddAndNorm(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.dropout(residual))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(1)].detach()
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """
    Custom encoder layer.
    """

    def __init__(self, input_dim: int, num_heads: int, dropout: float = 0.1, positional_encoding: bool = False):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout, batch_first=True)
        self.feed_forward = PositionWiseFeedForward(input_dim, input_dim, dropout=dropout)
        self.add_norm_after_attention = AddAndNorm(input_dim, dropout=dropout)
        self.add_norm_after_ff = AddAndNorm(input_dim, dropout=dropout)
        self.positional_encoding = PositionalEncoding(input_dim, dropout=dropout) if positional_encoding else None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.positional_encoding is not None:
            key = self.positional_encoding(key)
            value = self.positional_encoding(value)
            query = self.positional_encoding(query)

        attn_output, _ = self.self_attention(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
        x = self.add_norm_after_attention(attn_output, query)
        ff_output = self.feed_forward(x)
        return self.add_norm_after_ff(ff_output, x)
