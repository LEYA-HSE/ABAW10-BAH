# coding: utf-8
from __future__ import annotations


AUDIO_IMPL_SPECS: dict[str, dict[str, str | tuple[str, ...]]] = {
    "best_1": {
        "aliases": ("1", "best_1", "best_1.pt", "best_1_pkl"),
        "checkpoint": "./assets/checkpoints/audio/best_1.pt",
        "precomputed_path": "./features/audio/best_1_pkl/{split}.pkl",
    },
    "best_3": {
        # keep old "best_2*" aliases for backward compatibility
        "aliases": ("3", "best_3", "best_3.pt", "best_3_pkl", "2", "best_2", "best_2.pt", "best_2_pkl"),
        "checkpoint": "./assets/checkpoints/audio/best_3.pt",
        "precomputed_path": "./features/audio/best_3_pkl/{split}.pkl",
    },
}


def supported_audio_impl_values() -> str:
    values: list[str] = []
    for impl_id, spec in AUDIO_IMPL_SPECS.items():
        values.append(impl_id)
        values.extend(str(alias) for alias in spec["aliases"])  # type: ignore[index]
    uniq = sorted(set(values), key=str.lower)
    return ", ".join(uniq)


def normalize_audio_impl(raw_value: str, *, default_impl: str = "best_1") -> str:
    raw = str(raw_value).strip().lower()
    if not raw:
        return default_impl

    for impl_id, spec in AUDIO_IMPL_SPECS.items():
        aliases = [str(x).lower() for x in spec["aliases"]]  # type: ignore[index]
        if raw == impl_id or raw in aliases:
            return impl_id

    raise ValueError(
        f"Unsupported audio_export.impl='{raw_value}'. Supported values: {supported_audio_impl_values()}"
    )


def default_audio_checkpoint_for_impl(impl: str) -> str:
    spec = AUDIO_IMPL_SPECS.get(str(impl))
    if spec is None:
        return ""
    return str(spec["checkpoint"])


def default_audio_precomputed_path_for_impl(impl: str) -> str:
    spec = AUDIO_IMPL_SPECS.get(str(impl))
    if spec is None:
        return ""
    return str(spec["precomputed_path"])
