# coding: utf-8
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.utils.config_loader import ConfigLoader
from src.data_loading.video_preprocessor import get_face_pixel_values
from src.data_loading.pretrained_extractors import (
    build_extractors_from_config,
    AffectNetImageProcessor,
)
from src.models.models import VectorMLP


# fixed model (as requested)
CHECKPOINT_PATH = Path(
    r"best_model_weights.pt"
)
CONFIG_PATH = Path(
    r"best_config_copy.toml"
)
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--videos-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def aggregate_for_vector(embedding: torch.Tensor, average_mode: str) -> torch.Tensor:
    if embedding.ndim == 1:
        embedding = embedding.unsqueeze(0)
    mode = average_mode.lower()
    if mode == "mean_std":
        mean = embedding.mean(dim=0)
        std = embedding.std(dim=0, unbiased=False)
        return torch.cat([mean, std], dim=0)
    return embedding.mean(dim=0)


def main():
    args = parse_args()

    cfg = ConfigLoader(str(CONFIG_PATH))
    device = torch.device(cfg.device if str(cfg.device).startswith("cuda") and torch.cuda.is_available() else "cpu")

    # preprocessing + feature extractor from train config
    face_processor = AffectNetImageProcessor(image_size=cfg.affectnet_image_size)
    face_extractor = build_extractors_from_config(cfg)["face"]

    # build VectorMLP from checkpoint shapes
    state = torch.load(str(CHECKPOINT_PATH), map_location=device)
    w1 = state["feature_extractor.0.weight"]   # [hidden_dim, input_dim]
    w2 = state["feature_extractor.4.weight"]   # [out_features, hidden_dim]
    wc = state["classifier.weight"]            # [num_classes, out_features]

    model = VectorMLP(
        input_dim=int(w1.shape[1]),
        hidden_dim=int(w1.shape[0]),
        out_features=int(w2.shape[0]),
        num_classes=int(wc.shape[0]),
        dropout=float(cfg.dropout),
    ).to(device).eval()
    model.load_state_dict(state, strict=True)

    videos_root = args.videos_root
    videos = sorted([p for p in videos_root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS])

    out = {}
    for vp in tqdm(videos, desc="Inference"):
        rel_key = str(vp.relative_to(videos_root)).replace("\\", "/")

        _, face_pv = get_face_pixel_values(
            video_path=str(vp),
            segment_length=int(cfg.segment_length),
            image_processor=face_processor,
            device=str(device),
            yolo_weights=str(cfg.yolo_weights),
            mode=str(cfg.video_mode),
        )

        extr_out = face_extractor.extract(pixel_values=face_pv)
        embedding_seq = extr_out["embedding"]
        x = aggregate_for_vector(embedding_seq, cfg.average_features).unsqueeze(0).to(device)

        model_out = model(x, features=True)
        logits = model_out["prob"].squeeze(0)
        embeddings = model_out["features"].squeeze(0)
        prob = F.softmax(logits, dim=-1)

        out[rel_key] = {
            "prob": prob.detach().cpu().numpy().astype("float32"),
            "logits": logits.detach().cpu().numpy().astype("float32"),
            "embeddings": embeddings.detach().cpu().numpy().astype("float32"),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
