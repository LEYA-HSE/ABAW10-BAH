# coding: utf-8
from __future__ import annotations

from typing import Dict, Any, Optional, Union
import logging
import numpy as np
import torch
import cv2
import os
import torchvision.transforms as transforms
from transformers import (
    CLIPModel, CLIPProcessor,
    ViTModel, AutoImageProcessor,
)


# -------------------------
# Utils
# -------------------------

def _ensure_device(device: Union[str, torch.device]) -> torch.device:
    if isinstance(device, torch.device):
        return device
    d = (device or "cpu").lower()
    if d.startswith("cuda") and torch.cuda.is_available():
        try:
            return torch.device(d)
        except Exception:
            return torch.device("cuda")
    return torch.device("cpu")


def _pool_framewise(seq: torch.Tensor, mode: str) -> torch.Tensor:
    """
    seq: [T, L, D] (L = 1 + num_patches; index 0 is CLS)
    mode: "frame-cls" | "frame-mean" | "tokens"
    returns:
      - "frame-cls":  [T, D] (CLS per frame)
      - "frame-mean": [T, D] (mean over patch tokens per frame, excludes CLS)
      - "tokens":     [T*(L-1), D] (all patch tokens, flattened over time)
    """
    if mode == "frame-cls":
        return seq[:, 0, :]
    elif mode == "frame-mean":
        if seq.size(1) <= 1:
            return seq[:, 0, :]
        return seq[:, 1:, :].mean(dim=1)
    elif mode == "tokens":
        if seq.size(1) > 1:
            seq = seq[:, 1:, :]
        return seq.flatten(0, 1).contiguous()
    else:
        raise ValueError(f"Unsupported framewise pooling mode: {mode}")


# -------------------------
# Extractors (IDENTICAL LOGIC)
# -------------------------

class AffectNetImageProcessor:
    """
    Simple image processor for AffectNet EfficientNet models.
    Applies resize -> to tensor -> ImageNet normalize.
    """
    def __init__(
        self,
        image_size: int = 224,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.image_size = int(image_size)
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=list(mean), std=list(std)),
            ]
        )

    def to_pixel_values(self, image_rgb: np.ndarray) -> Optional[torch.Tensor]:
        if image_rgb is None or image_rgb.size == 0 or image_rgb.ndim != 3:
            return None
        img = image_rgb
        if img.shape[2] != 3:
            return None
        pv = self.transform(img).unsqueeze(0)  # [1,3,H,W]
        return pv

    def __call__(self, images, return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        if images is None:
            return {"pixel_values": torch.empty((0, 3, self.image_size, self.image_size))}
        if isinstance(images, np.ndarray):
            images = [images]
        pvs = [self.to_pixel_values(img) for img in images]
        pvs = [pv for pv in pvs if pv is not None]
        if not pvs:
            return {"pixel_values": torch.empty((0, 3, self.image_size, self.image_size))}
        return {"pixel_values": torch.cat(pvs, dim=0)}


from src.models.affectnet_effnet import AffectNetEfficientNet


class AffectNetEfficientNetExtractor:
    """
    AffectNet EfficientNet checkpoint -> per-frame embeddings.
    Uses the backbone (and optional adapter) to return frame-level features.
    """
    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        *,
        backbone: str = "efficientnet_b0",
        image_size: int = 224,
        proj_dim: int = 0,
    ):
        self.device = _ensure_device(device)
        self.ckpt_path = ckpt_path
        self.backbone = backbone
        self.image_size = int(image_size)
        self.proj_dim = int(proj_dim) if proj_dim else 0

        self.model = AffectNetEfficientNet(
            backbone=self.backbone,
            image_size=self.image_size,
            proj_dim=self.proj_dim,
        ).to(self.device).eval()

        if self.ckpt_path:
            if not os.path.isfile(self.ckpt_path):
                raise FileNotFoundError(f"AffectNet checkpoint not found: {self.ckpt_path}")
            try:
                ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
            except TypeError:
                ckpt = torch.load(self.ckpt_path, map_location="cpu")
            state = ckpt.get("model") if isinstance(ckpt, dict) else ckpt
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if missing:
                logging.warning(f"[AffectNetEfficientNetExtractor] Missing keys: {len(missing)}")
            if unexpected:
                logging.warning(f"[AffectNetEfficientNetExtractor] Unexpected keys: {len(unexpected)}")

        self.out_dim = int(self.model.proj_dim) if getattr(self.model, "proj_dim", 0) else int(self.model.feat_dim)

    def fingerprint(self) -> str:
        ckpt_name = os.path.basename(self.ckpt_path) if self.ckpt_path else "no-ckpt"
        return f"affectnet:{self.backbone}:img{self.image_size}:proj{self.proj_dim}:{ckpt_name}"

    @torch.no_grad()
    def extract(self, *, pixel_values: Optional[torch.Tensor] = None, **_) -> Dict[str, torch.Tensor]:
        if pixel_values is None:
            return {"embedding": torch.empty((0, self.out_dim), device=self.device), "frames": 0, "tokens_per_frame": 1}

        pv = pixel_values.to(self.device)
        feats = self.model.backbone(pv)
        if type(feats).__name__ == "tTensor":
            feats = feats.as_subclass(torch.Tensor)
        if feats.ndim == 4:
            feats = torch.flatten(feats, 1)
        if getattr(self.model, "adapter", None) is not None:
            feats = self.model.adapter(feats)
        return {"embedding": feats, "frames": pv.size(0), "tokens_per_frame": 1}


class ClipVideoExtractor:
    """
    CLIP vision encoder → per-frame features.
    Identical logic to ViT: we operate on vision_model hidden states (D = hidden_size, e.g., 768)
    and support the same output_mode values.

    output_mode:
      - "frame-cls"  (default): CLS per frame → [T, D]
      - "frame-mean": mean over patch tokens per frame → [T, D]
      - "tokens":     all patch tokens flattened → [T*(L-1), D]
      - "pooled":     CLIP projection via get_image_features → [T, 512] (special case)
    """
    def __init__(self,
                 model_name: str = "openai/clip-vit-base-patch32",
                 device: str = "cuda",
                 output_mode: str = "frame-cls"):
        self.model_name = model_name
        self.device = _ensure_device(device)
        self.output_mode = output_mode  # "frame-cls" | "frame-mean" | "tokens" | "pooled"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.proc  = CLIPProcessor.from_pretrained(model_name)

    def fingerprint(self) -> str:
        return f"clipv:{self.model_name}:{self.output_mode}"

    @torch.no_grad()
    def extract(self,
                *,
                pixel_values: Optional[torch.Tensor] = None,
                face_tensor: Optional[torch.Tensor] = None,
                images: Optional[Union[np.ndarray, list]] = None,
                **_) -> Dict[str, torch.Tensor]:

        # Normalize input → pixel_values [T,3,H,W]
        if pixel_values is None:
            if images is not None:
                if isinstance(images, np.ndarray):
                    images = [images]
                batch = self.proc(images=list(images), return_tensors="pt")
                pixel_values = batch["pixel_values"]
            elif face_tensor is not None:
                if face_tensor.ndim == 3:
                    face_tensor = face_tensor.unsqueeze(0)
                imgs_cpu = [img.cpu() for img in face_tensor]
                pixel_values = self.proc(images=imgs_cpu, return_tensors="pt")["pixel_values"]
            else:
                # Empty input → empty tensor with correct width
                if self.output_mode == "pooled":
                    D = self.model.visual_projection.out_features  # 512
                else:
                    D = self.model.vision_model.config.hidden_size  # typically 768
                return {"embedding": torch.empty((0, D), device=self.device),
                        "frames": 0,
                        "tokens_per_frame": 1}

        pv = pixel_values.to(self.device)  # [T,3,H,W]
        if pv.ndim == 4 and pv.shape[1] != 3:
            logging.warning(
                f"[ClipVideoExtractor] pixel_values has shape {tuple(pv.shape)}, "
                f"expected [T,3,H,W]. Preprocessing might be wrong."
            )

        # Special-case: CLIP projection (512d)
        if self.output_mode == "pooled":
            emb = self.model.get_image_features(pixel_values=pv)  # [T, 512]
            return {"embedding": emb, "frames": pv.size(0), "tokens_per_frame": 1}

        # Vision encoder hidden states (identical path to ViT)
        vout = self.model.vision_model(pixel_values=pv, return_dict=True)
        seq = vout.last_hidden_state  # [T, L, D]
        emb = _pool_framewise(seq, mode=self.output_mode)

        # Meta helps downstream aggregation keep time vs. tokens straight
        if self.output_mode == "tokens":
            tpf = (seq.size(1) - 1) if seq.size(1) > 0 else 1
            return {"embedding": emb, "frames": pv.size(0), "tokens_per_frame": tpf}
        else:
            return {"embedding": emb.contiguous(), "frames": pv.size(0), "tokens_per_frame": 1}


class VitVideoExtractor:
    """
    ViT → per-frame features.
    Identical output_mode semantics to CLIP vision above.

    output_mode:
      - "frame-cls"  (default): CLS per frame → [T, D]
      - "frame-mean": mean over patch tokens per frame → [T, D]
      - "tokens":     all patch tokens flattened → [T*(L-1), D]
    """
    def __init__(self,
                 model_name: str = "google/vit-base-patch16-224",
                 device: str = "cuda",
                 output_mode: str = "frame-cls"):
        self.model_name = model_name
        self.device = _ensure_device(device)
        self.output_mode = output_mode  # "frame-cls" | "frame-mean" | "tokens"
        self.model = ViTModel.from_pretrained(model_name).to(self.device).eval()
        self.proc  = AutoImageProcessor.from_pretrained(model_name)

    def fingerprint(self) -> str:
        return f"vitv:{self.model_name}:{self.output_mode}"

    @torch.no_grad()
    def extract(self,
                *,
                pixel_values: Optional[torch.Tensor] = None,
                images: Optional[Union[np.ndarray, list]] = None,
                **_) -> Dict[str, torch.Tensor]:
        # Normalize input → pixel_values [T,3,H,W]
        if pixel_values is None:
            if images is not None:
                if isinstance(images, np.ndarray):
                    images = [images]
                batch = self.proc(images=list(images), return_tensors="pt")
                pixel_values = batch["pixel_values"]
            else:
                D = self.model.config.hidden_size
                return {"embedding": torch.empty((0, D), device=self.device),
                        "frames": 0,
                        "tokens_per_frame": 1}

        pv = pixel_values.to(self.device)  # [T,3,H,W]
        out = self.model(pixel_values=pv, return_dict=True)
        seq = out.last_hidden_state  # [T, L, D]
        emb = _pool_framewise(seq, mode=self.output_mode)

        if self.output_mode == "tokens":
            tpf = (seq.size(1) - 1) if seq.size(1) > 0 else 1
            return {"embedding": emb, "frames": pv.size(0), "tokens_per_frame": tpf}
        else:
            return {"embedding": emb.contiguous(), "frames": pv.size(0), "tokens_per_frame": 1}


# -------------------------
# Factory
# -------------------------

def build_extractors_from_config(cfg) -> Dict[str, Any]:
    device = cfg.device
    # Keep existing config surface: cfg.video_output_mode (optional)
    output_mode = cfg.video_output_mode

    ex: Dict[str, Any] = {}

    vid_model: str = cfg.video_extractor
    if isinstance(vid_model, str) and vid_model.lower() != "off":
        v = vid_model.lower()
        if v.startswith("affectnet_"):
            ex["face"] = AffectNetEfficientNetExtractor(
                ckpt_path=cfg.affectnet_ckpt_path,
                device=device,
                backbone=cfg.affectnet_backbone,
                image_size=cfg.affectnet_image_size,
                proj_dim=cfg.affectnet_proj_dim,
            )
        elif "clip" in v:
            ex["face"] = ClipVideoExtractor(model_name=vid_model,
                                            device=device,
                                            output_mode=output_mode)
        elif "vit" in v:
            ex["face"] = VitVideoExtractor(model_name=vid_model,
                                           device=device,
                                           output_mode=output_mode)
        else:
            raise ValueError(
                f"Video extractor '{vid_model}' is not supported "
                f"(expected CLIP/VIT/AffectNet EfficientNet)."
            )

    return ex
