# coding: utf-8
from __future__ import annotations

from typing import Dict, Any, Optional, Union
import logging
import numpy as np
import torch
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
        if "clip" in v:
            ex["face"] = ClipVideoExtractor(model_name=vid_model,
                                            device=device,
                                            output_mode=output_mode)
        elif "vit" in v:
            ex["face"] = VitVideoExtractor(model_name=vid_model,
                                           device=device,
                                           output_mode=output_mode)
        else:
            raise ValueError(f"Video extractor '{vid_model}' is not supported (expected CLIP/VIT).")

    return ex
