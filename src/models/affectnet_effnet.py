# coding: utf-8
from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torchvision


class _BackboneWrapper(nn.Module):
    def __init__(self, backbone: nn.Module, pool: nn.Module | None = None):
        super().__init__()
        self.backbone = backbone
        self.pool = pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if self.pool is not None:
            feats = self.pool(feats)
        return feats


def _build_torchvision_backbone(
    name: str,
    imagenet_pretrained: bool = False,
) -> Tuple[nn.Module, int]:
    name = name.lower()
    if name == "efficientnet_b0":
        weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1 if imagenet_pretrained else None
        model = torchvision.models.efficientnet_b0(weights=weights)
        backbone = _BackboneWrapper(model.features, model.avgpool)
        feat_dim = 1280
    elif name == "efficientnet_b1":
        weights = torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V1 if imagenet_pretrained else None
        model = torchvision.models.efficientnet_b1(weights=weights)
        backbone = _BackboneWrapper(model.features, model.avgpool)
        feat_dim = 1280
    elif name == "resnet18":
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if imagenet_pretrained else None
        model = torchvision.models.resnet18(weights=weights)
        backbone = nn.Sequential(*list(model.children())[:-1])
        feat_dim = 512
    else:
        raise ValueError("Unknown backbone. Choose efficientnet_b0/efficientnet_b1/resnet18.")
    return backbone, feat_dim


class AffectNetEfficientNet(nn.Module):
    """
    EfficientNet backbone + optional adapter + emotion/VA heads.
    Kept compatible with the AffectNet checkpoint structure.
    """
    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        image_size: int = 224,  # kept for compatibility
        imagenet_pretrained: bool = False,
        proj_dim: int = 0,
        dropout: float = 0.2,
        emotion_classes: int = 8,
    ):
        super().__init__()
        self.image_size = int(image_size)
        self.backbone, feat_dim = _build_torchvision_backbone(backbone, imagenet_pretrained)
        self.feat_dim = feat_dim
        self.proj_dim = int(proj_dim) if proj_dim else 0

        self.adapter = None
        head_dim = feat_dim
        if self.proj_dim and self.proj_dim > 0:
            self.adapter = nn.Sequential(
                nn.Linear(feat_dim, self.proj_dim),
                nn.ReLU(inplace=True),
            )
            head_dim = self.proj_dim

        self.dropout = nn.Dropout(dropout)
        self.emotion_head = nn.Linear(head_dim, emotion_classes)
        self.va_head = nn.Linear(head_dim, 2)

    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)
        if type(feats).__name__ == "tTensor":
            feats = feats.as_subclass(torch.Tensor)
        if feats.ndim == 4:
            feats = torch.flatten(feats, 1)
        if self.adapter is not None:
            feats = self.adapter(feats)
        feats = self.dropout(feats)
        emo_logits = self.emotion_head(feats)
        va = self.va_head(feats)
        return emo_logits, va
