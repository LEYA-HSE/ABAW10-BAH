# utils/config_loader.py
from __future__ import annotations

import logging
import os
from typing import Any, Dict

import toml
from src.utils.artifact_naming import (
    build_audio_artifact_tag,
    build_face_artifact_tag,
    build_scene_artifact_tag,
    build_text_artifact_tag,
)
from src.utils.audio_impl import (
    default_audio_checkpoint_for_impl,
    default_audio_precomputed_path_for_impl,
    normalize_audio_impl,
)
from src.utils.text_impl import (
    default_text_checkpoint_for_impl,
    default_text_hf_model_for_impl,
    normalize_text_impl,
)


def _section(data: Dict[str, Any], *path: str) -> Dict[str, Any]:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict):
            return {}
        cur = cur.get(key, {})
    return cur if isinstance(cur, dict) else {}


def _pick(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


class ConfigLoader:
    """
    Minimal config loader for:
    - multimodal artifact validation/loading
    - modality artifact export (face/audio/text/scene)

    It keeps a few legacy fallbacks so existing auxiliary files such as
    `assets/configs/face_legacy_best.toml` can still be read by the face exporter.
    """

    def __init__(self, config_path: str = "config.toml") -> None:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file `{config_path}` not found!")

        self.config = toml.load(config_path)

        general_cfg = _section(self.config, "general")
        dataloader_cfg = _section(self.config, "dataloader")
        dataloader_mm_cfg = _section(self.config, "dataloader", "multimodal")
        runtime_cfg = _section(self.config, "runtime")
        search_cfg = _section(self.config, "search")
        multimodal_cfg = _section(self.config, "multimodal")
        exports_cfg = _section(self.config, "exports")
        model_cfg = _section(self.config, "model")
        training_cfg = _section(self.config, "training")
        if not model_cfg:
            model_cfg = _section(self.config, "fusion")
        if not training_cfg:
            training_cfg = _section(self.config, "fusion_train")
        face_export_cfg = _section(self.config, "face_export")
        audio_export_cfg = _section(self.config, "audio_export")
        scene_export_cfg = _section(self.config, "scene_export")
        text_export_cfg = _section(self.config, "text_export")

        legacy_train_general = _section(self.config, "train", "general")
        legacy_embeddings = _section(self.config, "embeddings")

        self.use_telegram = bool(general_cfg.get("use_telegram", False))

        self.datasets = _section(self.config, "datasets")
        self.multimodal_modalities = list(multimodal_cfg.get("modalities", []))
        self.multimodal_artifacts_dir = str(multimodal_cfg.get("artifacts_dir", "./features"))
        self.multimodal_placeholder_modalities = list(_pick(
            dataloader_mm_cfg.get("placeholder_modalities"),
            multimodal_cfg.get("placeholder_modalities"),  # legacy fallback
            default=[],
        ))
        self.multimodal_placeholder_prob_dim = int(_pick(
            dataloader_mm_cfg.get("placeholder_prob_dim"),
            multimodal_cfg.get("placeholder_prob_dim"),  # legacy fallback
            default=0,
        ))
        self.multimodal_placeholder_logits_dim = int(_pick(
            dataloader_mm_cfg.get("placeholder_logits_dim"),
            multimodal_cfg.get("placeholder_logits_dim"),  # legacy fallback
            default=0,
        ))
        self.multimodal_placeholder_emb_dim = int(_pick(
            dataloader_mm_cfg.get("placeholder_emb_dim"),
            multimodal_cfg.get("placeholder_emb_dim"),  # legacy fallback
            default=0,
        ))
        self.multimodal_sources = dict(multimodal_cfg.get("sources", {}))
        self.export_splits = list(exports_cfg.get("splits", ["train", "dev", "test"]))

        self.batch_size = int(_pick(
            dataloader_cfg.get("batch_size"),
            legacy_train_general.get("batch_size"),
            default=8,
        ))
        self.num_workers = int(dataloader_cfg.get("num_workers", 0))
        self.shuffle = bool(dataloader_cfg.get("shuffle", True))
        self.prepare_only = bool(dataloader_cfg.get("prepare_only", False))

        self.device = str(_pick(
            runtime_cfg.get("device"),
            legacy_train_general.get("device"),
            default="cuda",
        ))
        self.search_type = str(search_cfg.get("type", "none")).lower()
        self.search_selection_metric = str(search_cfg.get("selection_metric", "MF1"))
        self.search_early_stop_on = str(search_cfg.get("early_stop_on", "avg")).lower()
        self.search_params_path = str(search_cfg.get("params_path", "search_params.toml"))
        if self.search_type not in {"none", "greedy", "exhaustive", "optuna"}:
            raise ValueError("search.type must be one of: none, greedy, exhaustive, optuna")
        if self.search_early_stop_on not in {"avg", "dev", "test"}:
            raise ValueError("search.early_stop_on must be one of: avg, dev, test")

        self.fusion_input_type = str(model_cfg.get("input_type", "emb+prob"))
        self.fusion_type = str(model_cfg.get("type", "exchange_transformer"))
        self.fusion_d_model = int(model_cfg.get("d_model", 256))
        self.fusion_drop = float(model_cfg.get("drop", 0.1))
        self.fusion_use_prototypes = bool(model_cfg.get("use_prototypes", False))
        self.fusion_num_prototypes = int(model_cfg.get("num_prototypes", 4))
        self.fusion_proto_tau = float(model_cfg.get("proto_tau", 0.07))
        self.fusion_x_layers = int(model_cfg.get("x_layers", 2))
        self.fusion_x_heads = int(model_cfg.get("x_heads", 4))
        self.fusion_x_ff_mult = int(model_cfg.get("x_ff_mult", 4))
        self.fusion_x_use_cls = bool(model_cfg.get("x_use_cls", True))
        self.fusion_x_layer_impl = str(model_cfg.get("x_layer_impl", "torch")).lower()
        self.fusion_x_positional_encoding = bool(model_cfg.get("x_positional_encoding", False))
        if self.fusion_x_layer_impl not in {"torch", "custom"}:
            raise ValueError("model.x_layer_impl must be one of: torch, custom")
        self.fusion_videoformer_positional_encoding = bool(model_cfg.get("videoformer_positional_encoding", False))
        self.fusion_videoformer_gate_mode = str(model_cfg.get("videoformer_gate_mode", "none"))

        self.fusion_random_seed = int(training_cfg.get("random_seed", 42))
        self.fusion_num_epochs = int(training_cfg.get("num_epochs", 30))
        self.fusion_max_patience = int(training_cfg.get("max_patience", 8))
        self.fusion_optimizer = str(training_cfg.get("optimizer", "adamw"))
        self.fusion_lr = float(training_cfg.get("lr", 2e-4))
        self.fusion_weight_decay = float(training_cfg.get("weight_decay", 1e-4))
        self.fusion_momentum = float(training_cfg.get("momentum", 0.9))
        self.fusion_scheduler_type = str(training_cfg.get("scheduler_type", "plateau"))
        self.fusion_warmup_ratio = float(training_cfg.get("warmup_ratio", 0.1))
        self.fusion_loss_name = str(training_cfg.get("loss_name", "cross_entropy"))
        self.fusion_label_smoothing = float(training_cfg.get("label_smoothing", 0.0))
        self.fusion_focal_gamma = float(training_cfg.get("focal_gamma", 2.0))
        self.fusion_class_weighting = str(training_cfg.get("class_weighting", "balanced"))
        raw_fusion_class_weights = training_cfg.get("class_weights", [1.0, 1.0])
        self.fusion_class_weights = [float(x) for x in raw_fusion_class_weights] if isinstance(raw_fusion_class_weights, list) else None
        self.fusion_grad_clip = float(training_cfg.get("grad_clip", 1.0))
        self.fusion_lambda_proto = float(training_cfg.get("lambda_proto", 0.3))
        self.fusion_lambda_proto_div = float(training_cfg.get("lambda_proto_div", 0.02))
        self.fusion_save_checkpoints = bool(training_cfg.get("save_checkpoints", True))

        self.face_export_splits = list(face_export_cfg.get("splits", self.export_splits))
        self.face_export_overwrite_cache = bool(face_export_cfg.get("overwrite_cache", False))
        self.face_classifier_weights = str(_pick(
            face_export_cfg.get("classifier_weights"),
            default="assets/checkpoints/face/best_model_weights.pt",
        ))
        self.average_features = str(_pick(
            face_export_cfg.get("average_features"),
            legacy_embeddings.get("average_features"),
            default="mean_std",
        ))
        self.video_output_mode = "frame-cls"
        self.video_extractor = str(_pick(
            face_export_cfg.get("video_extractor"),
            legacy_embeddings.get("video_extractor"),
            default="off",
        ))
        self.yolo_weights = str(_pick(
            face_export_cfg.get("yolo_weights"),
            legacy_embeddings.get("yolo_weights"),
            default="src/data_loading/yolov8n-face.pt",
        ))
        self.video_mode = str(_pick(
            face_export_cfg.get("video_mode"),
            legacy_embeddings.get("video_mode"),
            default="stable",
        ))
        self.segment_length = int(_pick(
            face_export_cfg.get("segment_length"),
            legacy_embeddings.get("segment_length"),
            default=30,
        ))
        self.affectnet_ckpt_path = str(_pick(
            face_export_cfg.get("affectnet_ckpt_path"),
            legacy_embeddings.get("affectnet_ckpt_path"),
            default="",
        ))
        self.affectnet_backbone = str(_pick(
            face_export_cfg.get("affectnet_backbone"),
            legacy_embeddings.get("affectnet_backbone"),
            default="efficientnet_b0",
        ))
        self.affectnet_image_size = int(_pick(
            face_export_cfg.get("affectnet_image_size"),
            legacy_embeddings.get("affectnet_image_size"),
            default=224,
        ))
        self.affectnet_proj_dim = int(_pick(
            face_export_cfg.get("affectnet_proj_dim"),
            legacy_embeddings.get("affectnet_proj_dim"),
            default=0,
        ))
        self.face_artifact_tag = build_face_artifact_tag(self)
        self.face_export_output_dir = str(
            face_export_cfg.get(
                "output_dir",
                os.path.join(self.multimodal_artifacts_dir, "face", self.face_artifact_tag),
            )
        )

        self.audio_export_splits = list(audio_export_cfg.get("splits", self.export_splits))
        self.audio_export_overwrite_cache = bool(audio_export_cfg.get("overwrite_cache", False))
        self.audio_export_impl = str(audio_export_cfg.get("impl", "best_1"))
        self.audio_export_source = str(audio_export_cfg.get("source", "precomputed")).lower()
        if self.audio_export_source not in {"precomputed", "export"}:
            raise ValueError("audio_export.source must be one of: precomputed, export")
        self.audio_export_impl_resolved = normalize_audio_impl(self.audio_export_impl)
        self.audio_checkpoint_path = default_audio_checkpoint_for_impl(self.audio_export_impl_resolved)
        self.audio_precomputed_path = default_audio_precomputed_path_for_impl(self.audio_export_impl_resolved)
        self.audio_wav2vec_model = str(
            audio_export_cfg.get("wav2vec_model", "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim")
        )
        self.audio_sample_rate = int(audio_export_cfg.get("sample_rate", 16000))
        self.audio_hidden_state_index = int(audio_export_cfg.get("hidden_state_index", 10))
        self.audio_exts = [str(ext).lower() for ext in audio_export_cfg.get("audio_exts", [".wav"])]
        if self.audio_export_source == "precomputed":
            self.multimodal_sources.setdefault("audio", {"path": self.audio_precomputed_path})
        self.audio_artifact_tag = build_audio_artifact_tag(self)
        self.audio_export_output_dir = str(
            audio_export_cfg.get(
                "output_dir",
                os.path.join(self.multimodal_artifacts_dir, "audio", self.audio_artifact_tag),
            )
        )

        self.scene_export_splits = list(scene_export_cfg.get("splits", self.export_splits))
        self.scene_export_overwrite_cache = bool(scene_export_cfg.get("overwrite_cache", False))
        self.scene_checkpoint_path = str(scene_export_cfg.get("checkpoint_path", ""))
        self.scene_model_name = str(
            scene_export_cfg.get("model_name", "MCG-NJU/videomae-base-finetuned-kinetics")
        )
        self.scene_preprocessor_dir = str(
            scene_export_cfg.get("preprocessor_dir", "./assets/checkpoints/scene")
        )
        self.scene_num_frames = int(scene_export_cfg.get("num_frames", 16))
        self.scene_artifact_tag = build_scene_artifact_tag(self)
        self.scene_export_output_dir = str(
            scene_export_cfg.get(
                "output_dir",
                os.path.join(self.multimodal_artifacts_dir, "scene", self.scene_artifact_tag),
            )
        )

        self.text_export_splits = list(text_export_cfg.get("splits", self.export_splits))
        self.text_export_overwrite_cache = bool(text_export_cfg.get("overwrite_cache", False))
        self.text_export_impl = str(text_export_cfg.get("impl", "7"))
        self.text_input_column = str(text_export_cfg.get("text_column", ""))
        self.text_max_length = int(text_export_cfg.get("max_length", 256))
        self.text_batch_size = int(text_export_cfg.get("batch_size", 1))
        self.text_export_impl_resolved = normalize_text_impl(self.text_export_impl)
        self.text_checkpoint_path_resolved = default_text_checkpoint_for_impl(self.text_export_impl_resolved)
        self.text_hf_model_name_resolved = default_text_hf_model_for_impl(self.text_export_impl_resolved)
        self.text_artifact_tag = build_text_artifact_tag(self)
        self.text_export_output_dir = str(
            text_export_cfg.get(
                "output_dir",
                os.path.join(self.multimodal_artifacts_dir, "text", self.text_artifact_tag),
            )
        )

    def log_config(self) -> None:
        logging.info("=== CONFIGURATION ===")
        logging.info("Datasets loaded: %s", list(self.datasets.keys()))
        for name, ds in self.datasets.items():
            logging.info("[Dataset: %s]", name)
            logging.info("  Base Dir: %s", ds.get("base_dir", "N/A"))
            logging.info("  Video Dir: %s", ds.get("video_dir", ds.get("base_dir", "N/A")))
            logging.info("  CSV Path: %s", ds.get("csv_path", ""))

        logging.info("--- Runtime ---")
        logging.info(
            "DataLoader: batch_size=%s, num_workers=%s, shuffle=%s, prepare_only=%s",
            self.batch_size,
            self.num_workers,
            self.shuffle,
            self.prepare_only,
        )
        logging.info("Device: %s", self.device)
        logging.info(
            "Search: type=%s selection_metric=%s early_stop_on=%s params_path=%s",
            self.search_type,
            self.search_selection_metric,
            self.search_early_stop_on,
            self.search_params_path,
        )
        logging.info("Modalities: %s", self.multimodal_modalities)
        logging.info("Artifacts Dir: %s", self.multimodal_artifacts_dir)
        logging.info("Export Splits (default): %s", self.export_splits)
        logging.info("Placeholder Modalities: %s", self.multimodal_placeholder_modalities)

        logging.info("--- Face Export ---")
        logging.info("artifact_tag=%s", self.face_artifact_tag)
        logging.info("splits=%s", self.face_export_splits)
        logging.info("output_dir=%s", self.face_export_output_dir)
        logging.info("overwrite_cache=%s", self.face_export_overwrite_cache)
        logging.info("classifier_weights=%s", self.face_classifier_weights)
        logging.info("video_extractor=%s", self.video_extractor)
        logging.info("average_features=%s", self.average_features)
        logging.info("segment_length=%s", self.segment_length)

        logging.info("--- Audio Export ---")
        logging.info("artifact_tag=%s", self.audio_artifact_tag)
        logging.info("splits=%s", self.audio_export_splits)
        logging.info("output_dir=%s", self.audio_export_output_dir)
        logging.info("overwrite_cache=%s", self.audio_export_overwrite_cache)
        logging.info("impl=%s (resolved=%s)", self.audio_export_impl, self.audio_export_impl_resolved)
        logging.info("source=%s", self.audio_export_source)
        logging.info("precomputed_path=%s", self.audio_precomputed_path)
        logging.info("checkpoint_path=%s", self.audio_checkpoint_path)
        logging.info("wav2vec_model=%s", self.audio_wav2vec_model)
        logging.info("sample_rate=%s", self.audio_sample_rate)
        logging.info("hidden_state_index=%s", self.audio_hidden_state_index)

        logging.info("--- Fusion Model ---")
        logging.info("input_type=%s", self.fusion_input_type)
        logging.info("type=%s", self.fusion_type)
        logging.info("d_model=%s", self.fusion_d_model)
        logging.info("drop=%s", self.fusion_drop)
        logging.info("use_prototypes=%s", self.fusion_use_prototypes)
        logging.info("num_prototypes=%s", self.fusion_num_prototypes)
        logging.info("proto_tau=%s", self.fusion_proto_tau)
        logging.info("x_layers=%s", self.fusion_x_layers)
        logging.info("x_heads=%s", self.fusion_x_heads)
        logging.info("x_ff_mult=%s", self.fusion_x_ff_mult)
        logging.info("x_use_cls=%s", self.fusion_x_use_cls)
        logging.info("x_layer_impl=%s", self.fusion_x_layer_impl)
        logging.info("x_positional_encoding=%s", self.fusion_x_positional_encoding)
        logging.info("videoformer_positional_encoding=%s", self.fusion_videoformer_positional_encoding)
        logging.info("videoformer_gate_mode=%s", self.fusion_videoformer_gate_mode)

        logging.info("--- Fusion Train ---")
        logging.info("random_seed=%s", self.fusion_random_seed)
        logging.info("num_epochs=%s", self.fusion_num_epochs)
        logging.info("max_patience=%s", self.fusion_max_patience)
        logging.info("optimizer=%s", self.fusion_optimizer)
        logging.info("lr=%s", self.fusion_lr)
        logging.info("weight_decay=%s", self.fusion_weight_decay)
        logging.info("momentum=%s", self.fusion_momentum)
        logging.info("scheduler_type=%s", self.fusion_scheduler_type)
        logging.info("warmup_ratio=%s", self.fusion_warmup_ratio)
        logging.info("loss_name=%s", self.fusion_loss_name)
        logging.info("label_smoothing=%s", self.fusion_label_smoothing)
        logging.info("focal_gamma=%s", self.fusion_focal_gamma)
        logging.info("class_weighting=%s", self.fusion_class_weighting)
        logging.info("class_weights=%s", self.fusion_class_weights)
        logging.info("grad_clip=%s", self.fusion_grad_clip)
        logging.info("lambda_proto=%s", self.fusion_lambda_proto)
        logging.info("lambda_proto_div=%s", self.fusion_lambda_proto_div)
        logging.info("save_checkpoints=%s", self.fusion_save_checkpoints)

        logging.info("--- Scene Export ---")
        logging.info("artifact_tag=%s", self.scene_artifact_tag)
        logging.info("splits=%s", self.scene_export_splits)
        logging.info("output_dir=%s", self.scene_export_output_dir)
        logging.info("overwrite_cache=%s", self.scene_export_overwrite_cache)
        logging.info("checkpoint_path=%s", self.scene_checkpoint_path)
        logging.info("model_name=%s", self.scene_model_name)
        logging.info("preprocessor_dir=%s", self.scene_preprocessor_dir)
        logging.info("num_frames=%s", self.scene_num_frames)

        logging.info("--- Text Export ---")
        logging.info("artifact_tag=%s", self.text_artifact_tag)
        logging.info("splits=%s", self.text_export_splits)
        logging.info("output_dir=%s", self.text_export_output_dir)
        logging.info("overwrite_cache=%s", self.text_export_overwrite_cache)
        logging.info("impl=%s (resolved=%s)", self.text_export_impl, self.text_export_impl_resolved)
        logging.info("checkpoint_path=%s", self.text_checkpoint_path_resolved)
        logging.info("hf_model_name=%s", self.text_hf_model_name_resolved)
        logging.info("text_column=%s", self.text_input_column)
        logging.info("max_length=%s", self.text_max_length)
        logging.info("batch_size=%s", self.text_batch_size)

    def show_config(self) -> None:
        self.log_config()
