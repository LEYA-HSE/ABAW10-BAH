# main.py
# coding: utf-8
import logging
import os
import shutil
import datetime
import toml
import requests

from tqdm import tqdm
from src.utils.config_loader import ConfigLoader
from src.utils.logger_setup import setup_logger
from src.utils.search_utils import greedy_search, exhaustive_search

from src.data_loading.dataset_builder import make_bah_dataset_and_loader
from src.data_loading.pretrained_extractors import build_extractors_from_config, AffectNetImageProcessor

from transformers import CLIPProcessor, AutoImageProcessor

# If you have a trainer, wire it here. Otherwise you can comment this out temporarily.
from src.train import train

# ???????????????????? optionally load .env ??????????????????????????
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ???????????????????? Telegram helper ???????????????????????????????

def _notify_telegram(text: str, enabled: bool = True) -> bool:
    """Send a message to TG if enabled and TELEGRAM_BOT_TOKEN/CHAT_ID are set."""
    if not enabled:
        logging.info("TG notify: disabled by config")
        return False
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logging.info("TG notify: skipped (no TELEGRAM_BOT_TOKEN/CHAT_ID)")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=8,
        )
        try:
            payload = r.json()
        except Exception:
            payload = {"raw": r.text}
        if r.ok and isinstance(payload, dict) and payload.get("ok"):
            logging.info("TG notify: sent")
            return True
        logging.warning(f"TG notify: API error {r.status_code} -> {payload}")
        return False
    except Exception as e:
        logging.warning(f"TG notify failed: {e}")
        return False


def _any_split_exists(cfg, split_name: str) -> bool:
    """
    Check if any CSV exists for a split among datasets.bah_* sections.
    """
    for ds_name, ds_cfg in getattr(cfg, "datasets", {}).items():
        if not ds_name.lower().startswith("bah_"):
            continue
        csv_path = ds_cfg["csv_path"].format(base_dir=ds_cfg["base_dir"], split=split_name)
        if os.path.exists(csv_path):
            return True
    return False


def main():
    # ???????????????????? 1. Config and directories ????????????????????
    base_config = ConfigLoader("config.toml")

    model_name = base_config.model_name.replace("/", "_").replace(" ", "_").lower()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = f"results/results_{model_name}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    base_config.checkpoint_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(base_config.checkpoint_dir, exist_ok=True)

    epochlog_dir = os.path.join(results_dir, "metrics_by_epoch")
    os.makedirs(epochlog_dir, exist_ok=True)

    # ???????????????????? 2. Logging ?????????????????????????????????
    log_file = os.path.join(results_dir, "session_log.txt")
    setup_logger(logging.INFO, log_file=log_file)
    base_config.show_config()

    use_tg = base_config.use_telegram
    logging.info(
        f"use_telegram = {use_tg}  (env token={bool(os.getenv('TELEGRAM_BOT_TOKEN'))}, chat={bool(os.getenv('TELEGRAM_CHAT_ID'))})"
    )

    # startup ping
    _notify_telegram(f"?? Start: <b>{model_name}</b>\n?? {results_dir}", enabled=use_tg)

    # Save config copy and overrides log
    shutil.copy("config.toml", os.path.join(results_dir, "config_copy.toml"))
    overrides_file = os.path.join(results_dir, "overrides.txt")
    csv_prefix = os.path.join(epochlog_dir, "metrics_epochlog")

    # ???????????????????? 3. Extractors/processors ???????????????????
    logging.info("?? Initializing extractors from config (face only)...")

    # build_extractors_from_config should return key 'face'
    modality_extractors = build_extractors_from_config(base_config)

    # Video processor: AutoImageProcessor for ViT, CLIPProcessor for CLIP
    if getattr(base_config, "video_extractor", "").lower() == "off":
        raise ValueError("video_extractor='off' is not supported ? processor for 'face' is required.")

    model_name = base_config.video_extractor
    try:
        vname = model_name.lower()
        if vname in {
            "affectnet_efficientnet_b0",
            "affectnet_efficientnet",
            "affectnet_enet_b0",
            "affectnet_effnet",
        }:
            face_processor = AffectNetImageProcessor(image_size=base_config.affectnet_image_size)
        elif "vit" in vname:
            face_processor = AutoImageProcessor.from_pretrained(model_name)
        else:
            face_processor = CLIPProcessor.from_pretrained(model_name)
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize image processor from '{model_name}'. "
            f"Check config.video_extractor. Original error: {e}"
        )

    modality_processors = {"face": face_processor}

    # store in config for dataset builder
    base_config.modality_extractors = modality_extractors
    base_config.modality_processors = modality_processors

    enabled = ", ".join(sorted(modality_extractors.keys())) or "?"
    logging.info(f"? Active modalities: {enabled}")

    # ???????????????????? 4. Data loaders (BAH) ??????????????????????
    # dev/val: if any dev CSV exists, use 'dev', otherwise 'val'
    dev_split = "dev" if _any_split_exists(base_config, "dev") else "val"

    logging.info("?? Loading BAH (train/dev/test)...")
    _, train_loader = make_bah_dataset_and_loader(base_config, "train")
    _, dev_loader = make_bah_dataset_and_loader(base_config, dev_split)

    # test: if no test split, reuse dev
    if _any_split_exists(base_config, "test"):
        _, test_loader = make_bah_dataset_and_loader(base_config, "test")
    else:
        test_loader = dev_loader

    # ???????????????????? 5. prepare_only mode ???????????????????????
    if base_config.prepare_only:
        logging.info("== prepare_only mode: only data preparation, no training ==")
        _notify_telegram(
            f"? <b>{model_name}</b>: prepare_only completed\n?? {results_dir}",
            enabled=use_tg
        )
        return

    # ???????????????????? 6. Search/training ?????????????????????????
    search_type = base_config.search_type

    dev_loaders = {"bah": dev_loader}
    test_loaders = {"bah": test_loader}

    if search_type == "greedy":
        search_config = toml.load("search_params.toml")
        param_grid = dict(search_config.get("grid", {}))
        default_values = dict(search_config.get("defaults", {}))

        greedy_search(
            base_config=base_config,
            train_loader=train_loader,
            dev_loader=dev_loaders,
            test_loader=test_loaders,
            train_fn=train,
            overrides_file=overrides_file,
            param_grid=param_grid,
            default_values=default_values,
        )
        _notify_telegram(
            f"? <b>{model_name}</b>: greedy search finished\n?? {results_dir}",
            enabled=use_tg
        )

    elif search_type == "exhaustive":
        search_config = toml.load("search_params.toml")
        param_grid = dict(search_config.get("grid", {}))

        exhaustive_search(
            base_config=base_config,
            train_loader=train_loader,
            dev_loader=dev_loaders,
            test_loader=test_loaders,
            train_fn=train,
            overrides_file=overrides_file,
            param_grid=param_grid,
        )
        _notify_telegram(
            f"? <b>{model_name}</b>: exhaustive search finished\n?? {results_dir}",
            enabled=use_tg
        )

    elif search_type == "none":
        logging.info("== Single training run (no hyperparameter search) ==")
        train(
            cfg=base_config,
            mm_loader=train_loader,
            dev_loaders=dev_loaders,
            test_loaders=test_loaders,
        )
        _notify_telegram(
            f"? <b>{model_name}</b>: training (no search) completed\n?? {results_dir}",
            enabled=use_tg
        )

    else:
        raise ValueError(
            f"?? Invalid search_type: '{base_config.search_type}'. "
            f"Use 'greedy', 'exhaustive' or 'none'."
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        _notify_telegram(
            f"? Crash: <code>{type(e).__name__}</code>\n{e}",
            enabled=True
        )
        raise
