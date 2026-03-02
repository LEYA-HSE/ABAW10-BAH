from __future__ import annotations
import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_json(path: str | Path, obj: Any):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if is_dataclass(o):
            return asdict(o)
        raise TypeError(f"Not JSON serializable: {type(o)}")

    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=default)

def pick_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.cnt = 0

    def update(self, val: float, n: int = 1):
        self.sum += float(val) * n
        self.cnt += int(n)

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.cnt)
