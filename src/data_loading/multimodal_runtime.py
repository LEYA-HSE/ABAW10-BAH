from __future__ import annotations

import logging
import os


def any_split_exists(cfg, split_name: str) -> bool:
    for _, ds_cfg in getattr(cfg, "datasets", {}).items():
        csv_path_tpl = ds_cfg.get("csv_path")
        base_dir = ds_cfg.get("base_dir")
        if not csv_path_tpl or not base_dir:
            continue
        csv_path = str(csv_path_tpl).format(base_dir=base_dir, split=split_name)
        if os.path.exists(csv_path):
            return True
    return False


def log_multimodal_batch_stats(loader) -> None:
    first = None
    for batch in loader:
        if batch is not None:
            first = batch
            break
    if first is None:
        logging.info("[Multimodal] loader is empty")
        return

    batch_size = int(first["labels"].size(0))
    logging.info("[Multimodal] first batch size=%d", batch_size)
    for modality, mask in first.get("mask", {}).items():
        present = int(mask.sum().item())
        emb_shape = tuple(first["emb"][modality].shape) if modality in first.get("emb", {}) else None
        prob_shape = tuple(first["prob"][modality].shape) if modality in first.get("prob", {}) else None
        logits_shape = tuple(first["logits"][modality].shape) if modality in first.get("logits", {}) else None
        logging.info(
            "[Multimodal] modality=%s present=%d/%d emb=%s prob=%s logits=%s",
            modality,
            present,
            batch_size,
            emb_shape,
            prob_shape,
            logits_shape,
        )
