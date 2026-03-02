from __future__ import annotations

import torch

from training.experiment import run_single_experiment
from training.hyper_search import run_grid_search, run_greedy_search
from training.config import ExperimentConfig, DataConfig, ModelConfig, TrainConfig, SearchConfig


def _filter_space_for_runtime(space: dict) -> dict:
    if "model.encoder_name" not in space:
        return space

    encs = list(space["model.encoder_name"])
    if not torch.cuda.is_available():
        encs = [e for e in encs if e not in ("mamba1", "mamba2")]
    space["model.encoder_name"] = encs
    return space


def main():
    data = DataConfig(
        embeddings_npz="path",
        embeddings_npz_chunks="path",
        chunks_yaml_root="path",
        # train_mode: "full" | "chunks" | "mixed"
        train_mode="full",
        train_txt="Data/split/train.txt",
        val_txt="Data/split/val.txt",
        test_txt="Data/split/test.txt",
        allow_index_alignment=False,
        sep=",",
        id_field=0,
        label_field=1,
    )

    model = ModelConfig(
        encoder_name="mamba1",  # "transformer" | "custom_mamba" | "mamba1" | "zeros | "mlp"
        input_dim=None, 
        d_model=256,
        nhead=4,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        pooling="mean",  # "mean" | "attn"
        pooling_mlp="mean_std",  # mean, mean_std, mean_std_max
        mlp_hidden=512, 
        activation="gelu",
        hybrid_mamba_version="v1",
        hybrid_attn_every = 2,
        zeros_block_size=1024,
        zeros_force_block_size=True,
        zeros_crop_mode="random",

        mamba_d_state=8,
        mamba_d_conv=4,
        mamba_expand=2,

        num_classes=2,
        label_smoothing=0.0,
    )

    train = TrainConfig(
        seed=42,
        device="cuda" if torch.cuda.is_available() else "cpu",
        amp=True,
        num_epochs=30,
        batch_size=8,
        lr=1e-4,
        weight_decay=0.0,
        grad_clip_norm=0.0,
        num_workers=4,
        pin_memory=True,

        optimizer="adamw",
        momentum=0.9,
        nesterov=True,
        monitor_metric="mf1",
        monitor_mode="max",
        early_stop_patience=3,

        workdir="../Runs/baseline",
        save_best=True,
        save_last=True,
    )

    mode = "grid"  # "single" | "grid" | "greedy"

    grid = SearchConfig(
        metric="mf1",
        mode="max",
        max_runs=None,
        space=_filter_space_for_runtime(
            {
                "model.num_layers": [2, 4],
                "model.label_smoothing": [0.0, 0.2, 0.3],
                "train.weight_decay": [0.0, 1e-4],
                "train.lr": [1e-4],
                "model.dropout": [0.05, 0.1, 0.15],
            }
        ),
    )

    greedy = SearchConfig(
        metric="mf1",
        mode="max",
        max_runs=None,
        space=_filter_space_for_runtime(
            {
                "model.mamba_d_conv": [2, 3, 4],
                "model.mamba_expand": [1, 2, 4],
                "model.num_layers": [2, 4],
                "model.label_smoothing": [0.0, 0.1],
                "train.weight_decay": [0.0, 1e-2],
                "train.grad_clip_norm": [0.0, 2.0],
                "model.d_model": [64, 256, 384],
                "train.lr": [1e-3, 1e-4, 1e-5],
                "model.dim_feedforward": [256, 512, 1024],

            }
        ),
    )

    cfg = ExperimentConfig(data=data, model=model, train=train)

    if mode == "single":
        run_single_experiment(cfg)
    elif mode == "grid":
        run_grid_search(cfg, grid)
    elif mode == "greedy":
        run_greedy_search(cfg, greedy)
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
