# utils/config_loader.py

import os
import toml
import logging

class ConfigLoader:
    """
    Loader and handler for configuration from `config.toml`.
    """

    def __init__(self, config_path="config.toml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file `{config_path}` not found!")

        self.config = toml.load(config_path)

        # ---------------------------
        # General
        # ---------------------------
        general_cfg = self.config.get("general", {})
        self.use_telegram = general_cfg.get("use_telegram", False)

        # ---------------------------
        # Data paths
        # ---------------------------
        self.datasets = self.config.get("datasets", {})

        # ---------------------------
        # DataLoader
        # ---------------------------
        dataloader_cfg = self.config.get("dataloader", {})
        self.num_workers = dataloader_cfg.get("num_workers", 0)
        self.shuffle = dataloader_cfg.get("shuffle", True)
        self.prepare_only = dataloader_cfg.get("prepare_only", False)

        # ---------------------------
        # Training: general
        # ---------------------------
        train_general = self.config.get("train", {}).get("general", {})
        self.random_seed = train_general.get("random_seed", 42)
        self.subset_size = train_general.get("subset_size", 0)
        self.batch_size = train_general.get("batch_size", 8)
        self.num_epochs = train_general.get("num_epochs", 100)
        self.max_patience = train_general.get("max_patience", 10)
        self.save_best_model = train_general.get("save_best_model", False)
        self.save_prepared_data = train_general.get("save_prepared_data", True)
        self.save_feature_path = train_general.get("save_feature_path", "./features/")
        self.search_type = train_general.get("search_type", "none")
        self.early_stop_on = train_general.get("early_stop_on", "dev")
        self.checkpoint_dir = train_general.get("checkpoint_dir", "checkpoints")
        self.device = train_general.get("device", "cuda")
        self.selection_metric = train_general.get("selection_metric", "mean_combo")
        self.add_similarity = train_general.get("add_similarity", True)
        self.class_weighting = train_general.get("class_weighting", "balanced")
        self.num_archetypes = train_general.get("num_archetypes", 1)
        self.print_logits = train_general.get("print_logits", False)

        # ---------------------------
        # Training: model params
        # ---------------------------
        train_model = self.config.get("train", {}).get("model", {})

        # generic
        self.model_name = train_model.get("model_name", "mamba")  # "mamba" or "transformer"

        self.hidden_dim = train_model.get("hidden_dim", 256)
        self.dropout = train_model.get("dropout", 0.15)
        self.out_features = train_model.get("out_features", 128)
        self.gate_mode = train_model.get("gate_mode", "None")

        # transformer-specific
        self.num_transformer_heads = train_model.get("num_transformer_heads", 8)
        self.positional_encoding   = train_model.get("positional_encoding", True)
        self.tr_layers             = train_model.get("tr_layers", 2)

        # mamba-specific
        self.mamba_d_state   = train_model.get("mamba_d_state", 8)
        self.mamba_ker_size  = train_model.get("mamba_ker_size", 3)
        self.mamba_layers    = train_model.get("mamba_layers", 2)
        self.mamba_d_discr   = train_model.get("mamba_d_discr", None)

        # ---------------------------
        # Training: optimizer
        # ---------------------------
        train_optimizer = self.config.get("train", {}).get("optimizer", {})
        self.optimizer = train_optimizer.get("optimizer", "adam")
        self.lr = train_optimizer.get("lr", 1e-4)
        self.weight_decay = train_optimizer.get("weight_decay", 0.0)
        self.momentum = train_optimizer.get("momentum", 0.9)

        # ---------------------------
        # Training: scheduler
        # ---------------------------
        train_scheduler = self.config.get("train", {}).get("scheduler", {})
        self.scheduler_type = train_scheduler.get("scheduler_type", "plateau")
        self.warmup_ratio = train_scheduler.get("warmup_ratio", 0.1)

        # ---------------------------
        # Embeddings
        # ---------------------------
        emb_cfg = self.config.get("embeddings", {})
        self.average_features = emb_cfg.get("average_features", "mean_std")
        self.video_output_mode = emb_cfg.get("video_output_mode", "frame-cls")
        self.video_extractor = emb_cfg.get("video_extractor", "off")
        self.yolo_weights = emb_cfg.get("yolo_weights", "src/data_loading/yolov8n-face.pt")
        self.video_mode = emb_cfg.get("video_mode", "stable")
        self.segment_length = emb_cfg.get("segment_length", 20)
        self.emb_normalize = emb_cfg.get("emb_normalize", True)

        # ---------------------------
        # Cache
        # ---------------------------
        cache_cfg = self.config.get("cache", {})
        self.per_modality_cache = cache_cfg.get("per_modality_cache", True)
        self.overwrite_modality_cache = cache_cfg.get("overwrite_modality_cache", False)
        self.force_reextract = cache_cfg.get("force_reextract", [])
        self.preprocess_version = cache_cfg.get("preprocess_version", "v1")

        if __name__ == "__main__":
            self.log_config()

    def log_config(self):
        logging.info("=== CONFIGURATION ===")
        logging.info(f"Datasets loaded: {list(self.datasets.keys())}")
        for name, ds in self.datasets.items():
            logging.info(f"[Dataset: {name}]")
            logging.info(f"  Base Dir: {ds.get('base_dir', 'N/A')}")
            logging.info(f"  CSV Path: {ds.get('csv_path', '')}")
            logging.info(f"  Video Dir: {ds.get('video_dir', '')}")

        logging.info("--- Training Config ---")
        logging.info(f"DataLoader: batch_size={self.batch_size}, num_workers={self.num_workers}, shuffle={self.shuffle}")
        logging.info(f"Model Name: {self.model_name}")
        logging.info(f"Random Seed: {self.random_seed}")
        logging.info(f"Optimizer: {self.optimizer}")
        logging.info(f"Scheduler Type: {self.scheduler_type}")
        logging.info(f"Warmup Ratio: {self.warmup_ratio}")
        logging.info(f"Weight Decay for Adam: {self.weight_decay}")
        logging.info(f"Momentum (SGD): {self.momentum}")
        logging.info(f"Positional Encoding: {self.positional_encoding}")
        logging.info(f"Dropout: {self.dropout}")
        logging.info(f"Out Features: {self.out_features}")
        logging.info(f"LR: {self.lr}")
        logging.info(f"Num Epochs: {self.num_epochs}")
        logging.info(f"Max Patience={self.max_patience}")
        logging.info(f"Save Prepared Data={self.save_prepared_data}")
        logging.info(f"Path to Save Features={self.save_feature_path}")
        logging.info(f"Search Type={self.search_type}")

    def show_config(self):
        self.log_config()
