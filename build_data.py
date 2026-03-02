from datasets.bah_dataset import build_bah_loaders
from training.config import ExperimentConfig


def build_all_loaders(cfg: ExperimentConfig):
    return build_bah_loaders(cfg.data, cfg.train)
