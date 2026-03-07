# coding: utf-8
from __future__ import annotations

import logging
import os

import requests


def notify_telegram(text: str, enabled: bool = True) -> bool:
    if not enabled:
        logging.info("TG notify: disabled by config")
        return False

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logging.info("TG notify: skipped (no TELEGRAM_BOT_TOKEN/CHAT_ID)")
        return False

    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=8,
        )
        try:
            payload = response.json()
        except Exception:
            payload = {"raw": response.text}

        if response.ok and isinstance(payload, dict) and payload.get("ok"):
            logging.info("TG notify: sent")
            return True

        logging.warning("TG notify: API error %s -> %s", response.status_code, payload)
        return False
    except Exception as exc:
        logging.warning("TG notify failed: %s", exc)
        return False
