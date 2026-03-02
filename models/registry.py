from __future__ import annotations
from typing import Callable, Dict

ENCODERS: Dict[str, Callable] = {}

def register_encoder(name: str):
    def deco(fn: Callable):
        ENCODERS[name] = fn
        return fn
    return deco

def build_encoder(name: str, **kwargs):
    if name not in ENCODERS:
        raise ValueError(f"Unknown encoder '{name}'. Available: {list(ENCODERS.keys())}")
    return ENCODERS[name](**kwargs)
