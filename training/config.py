from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Any


@dataclass
class DataConfig:
    embeddings_npz: str

    train_txt: str
    val_txt: str
    test_txt: str

    embeddings_npz_chunks: Optional[str] = None

    chunks_yaml_root: Optional[str] = None

    train_mode: str = "full"

    allow_index_alignment: bool = False

    sep: str = ","
    id_field: int = 0
    label_field: int = 1


@dataclass
class ModelConfig:
    encoder_name: str = "transformer"  # "transformer" | "mamba1" | "mamba2" | "hybrid_attn_mamba" | ...

    input_dim: Optional[int] = None
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    pooling: str = "mean"  # "mean" | "attn"

    pooling_mlp: str = "mean_std"  
    mlp_hidden: int | None = None 
    activation: str = "gelu"  # gelu / relu

    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2

    mamba2_headdim: Optional[int] = None
    mamba2_ngroups: Optional[int] = None
    mamba2_chunk_size: Optional[int] = None

    hybrid_attn_every: int = 3
    hybrid_mamba_version: str = "v1"

    num_classes: int = 2
    label_smoothing: float = 0.0


    zeros_block_size: int = 2048
    zeros_is_causal: bool = False
    zeros_use_associative: bool = True
    zeros_use_norm: bool = True
    zeros_bias: bool = True
    zeros_force_block_size: bool = True
    zeros_crop_mode: str = "truncate"   # "truncate" | "random"
@dataclass
class TrainConfig:

    seed: int = 42
    device: str = "cuda"
    amp: bool = True

    # --- optimizer ---
    optimizer: str = "adamw"  # "adamw" | "sgd"
    momentum: float = 0.9  
    nesterov: bool = True 

    num_epochs: int = 30
    batch_size: int = 32

    lr: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip_norm: float = 1.0

    num_workers: int = 4
    pin_memory: bool = True

    monitor_metric: str = "mf1"   
    monitor_mode: str = "max"
    early_stop_patience: int = 8

    workdir: str = "../Runs/exp"
    save_best: bool = True
    save_last: bool = True


@dataclass
class SearchConfig:
    metric: str = "mf1"
    mode: str = "max"  
    max_runs: Optional[int] = None
    space: Dict[str, list] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    train: TrainConfig

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def clone(self) -> "ExperimentConfig":
        d = self.to_dict()
        return ExperimentConfig(
            data=DataConfig(**d["data"]),
            model=ModelConfig(**d["model"]),
            train=TrainConfig(**d["train"]),
        )
