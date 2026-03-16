# utils/logger_setup.py
import torch
import logging
from colorlog import ColoredFormatter
import numpy as np


def setup_logger(level=logging.INFO, log_file=None):
    """
    Configure root logger for colored console output and (optionally) file logging.

    :param level: Logging level (e.g., logging.DEBUG)
    :param log_file: Path to log file (if not None, logs are written to this file)
    """
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler with colorlog
    console_handler = logging.StreamHandler()
    log_format = (
        "%(log_color)s%(asctime)s [%(levelname)s]%(reset)s %(blue)s%(message)s"
    )
    console_formatter = ColoredFormatter(
        log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red"
        }
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_format = "%(asctime)s [%(levelname)s] %(message)s"
        file_formatter = logging.Formatter(file_format, datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    logger.setLevel(level)
    return logger


def color_metric(metric_name, value):
    """
    Colorize metric key/value. Supports UAR/MF1 and recall_*.
    """
    # base ANSI
    END = "\033[0m"
    GRAY = "\033[90m"
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    BLUE = "\033[34m"
    GREEN = "\033[32m"

    COLORS = {
        "mF1": "\033[96m",
        "mUAR": "\033[91m",
        "ACC": "\033[32m",
        "CCC": "\033[33m",
        "UAR": "\033[1;34m",
        "MF1": "\033[1;35m",
    }

    # direct match
    color = COLORS.get(metric_name, "")

    # per-class recall_* color by class index
    if not color and metric_name.lower().startswith("recall_"):
        import re
        m = re.search(r"recall_c(\d+)", metric_name.lower())
        c_idx = int(m.group(1)) if m else None
        if c_idx == 0:
            color = CYAN
        elif c_idx == 1:
            color = YELLOW
        elif c_idx == 2:
            color = MAGENTA
        else:
            color = GRAY

    try:
        return f"{color}{metric_name}:{float(value):.4f}{END}" if color else f"{metric_name}:{float(value):.4f}"
    except Exception:
        return f"{color}{metric_name}={value}{END}" if color else f"{metric_name}={value}"


def color_split(name: str) -> str:
    """
    Return a colored split tag. Supports: TRAIN / DEV / TEST.
    """
    SPLIT_COLORS = {
        "TRAIN": "\033[1;33m",  # bright yellow
        "DEV":   "\033[1;34m",  # bright blue
        "TEST":  "\033[1;35m",  # bright magenta
    }
    END = "\033[0m"
    key = name.upper()
    return f"{SPLIT_COLORS.get(key, '')}{name}{END}"


# ===== DEBUG LOGITS CHECK (root logger) =====

def dbg_check_logits(final_logits=None, cls_logits=None, proto_logits=None, print_logits=False, prefix="[DBG]"):
    if not print_logits:
        return
    x = final_logits if final_logits is not None else (cls_logits if cls_logits is not None else proto_logits)
    if x is None:
        return
    x = x.detach().float().cpu()
    logging.info(f"{prefix} mean={x.mean():.3f} std={x.std():.3f} min={x.min():.3f} max={x.max():.3f}")
    top2 = torch.topk(x, k=2, dim=1).values
    logging.info(f"{prefix} margin(top1-top2)={(top2[:,0]-top2[:,1]).mean().item():.3f}")
    logging.info(f"{prefix} per-class mean: " + " ".join(f"{v:.2f}" for v in x.mean(0).tolist()))
    if cls_logits is not None and proto_logits is not None:
        c = cls_logits.detach().float().cpu().argmax(1)
        p = proto_logits.detach().float().cpu().argmax(1)
        agree = (c == p).float().mean().item()
        logging.info(f"{prefix} heads agree={agree:.3f}")

# ===== END DEBUG LOGITS CHECK =====

# ===== DEBUG RAW LOGITS SLICE (root logger) =====

def dbg_dump_logits(x, printed=False, prefix="[DBG]", max_rows=5, max_cols=8):
    if not printed or x is None:
        return
    x = x.detach().float().cpu()
    r = min(max_rows, x.size(0))
    c = min(max_cols, x.size(1))
    np.set_printoptions(precision=2, suppress=True, linewidth=200)
    logging.info(f"{prefix} logits[:{r}, :{c}] =\n{ x[:r, :c].numpy() }")

# ===== END DEBUG RAW LOGITS SLICE =====
