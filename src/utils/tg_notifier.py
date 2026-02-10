# coding: utf-8
"""
Pretty Telegram notifications for experiments.
One job: start (params) and done (duration + metrics).

Usage:
  from src.utils.telegram_notifier import tg_start, tg_done
  tg_start(cfg, results_dir)
  tg_done(results_dir, start_time, metrics_dev=dev_m, metrics_test=te_m,
          selection_metric=cfg.selection_metric, early_stop_on=cfg.early_stop_on)

Env:
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
"""
from __future__ import annotations
import os, datetime, requests
from typing import Dict, Optional

# ───────────────── helpers ─────────────────

def _htime(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _send(text: str, enabled: bool = True) -> bool:
    if not enabled:
        return False
    token   = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True},
            timeout=10,
        )
        try:
            ok = r.ok and r.json().get("ok", False)
        except Exception:
            ok = False
        return ok
    except Exception:
        return False


def _kv_block(title: str, kv: Dict[str, str]) -> str:
    lines = [f"<b>{title}</b>"]
    for k, v in kv.items():
        lines.append(f"• <b>{k}:</b> {v}")
    return "\n".join(lines)


def _fmt_params(cfg, results_dir: str) -> Dict[str, str]:
    g = lambda k, d=None: str(getattr(cfg, k, d))
    return {
        "Results": results_dir,
        "Model": g("model_name"),
        "Search": g("search_type"),
        "Extractor": g("video_extractor"),
        "video_output_mode": g("video_output_mode", "frame-cls"),
        "average_features": g("average_features", "raw"),
        "segment_length": g("segment_length", 30),
        "batch_size": g("batch_size", 32),
        "epochs": g("num_epochs", 100),
        "optimizer": g("optimizer", "adam"),
        "lr": g("lr", 1e-5),
    }


# ───────────────── public API ─────────────────

def tg_start(cfg, results_dir: str, enabled: bool = True) -> bool:
    title = f"🚀 <b>Start</b>: {getattr(cfg, 'model_name', 'model')}"
    params = _fmt_params(cfg, results_dir)
    msg = title + "\n" + _kv_block("Params", params)
    return _send(msg, enabled)


def _pick_selection(metrics: Dict[str, float] | None, selection_metric: str, split_name: str) -> Optional[str]:
    """Pull split-specific selection metric like 'UAR_dev' or fallback to plain 'UAR'."""
    if not isinstance(metrics, dict):
        return None
    key_split = f"{selection_metric}_{split_name}"
    if key_split in metrics and isinstance(metrics[key_split], (int, float)):
        return f"{metrics[key_split]:.4f}"
    if selection_metric in metrics and isinstance(metrics[selection_metric], (int, float)):
        return f"{metrics[selection_metric]:.4f}"
    return None


def tg_done(
    results_dir: str,
    start_time: datetime.datetime,
    *,
    enabled: bool = True,
    metrics_dev: Optional[Dict[str, float]] = None,
    metrics_test: Optional[Dict[str, float]] = None,
    selection_metric: str = "UAR",
    early_stop_on: str = "dev",
    best_combo: Optional[dict] = None,
) -> bool:
    dt = _htime((datetime.datetime.now() - start_time).total_seconds())
    title = "🏁 <b>Run complete</b>"

    # try to surface a single headline metric
    head_val = None
    if early_stop_on == "dev":
        head_val = _pick_selection(metrics_dev, selection_metric, "bah") or _pick_selection(metrics_dev, selection_metric, "dev")
    else:
        head_val = _pick_selection(metrics_test, selection_metric, "bah") or _pick_selection(metrics_test, selection_metric, "test")

    kv = {"Total": dt, "Results": results_dir}
    if head_val:
        kv[f"{selection_metric} [{early_stop_on}]"] = head_val
    if best_combo:
        kv["Best params"] = ", ".join(f"{k}={v}" for k, v in best_combo.items())

    msg = title + "\n" + _kv_block("Summary", kv)
    return _send(msg, enabled)
