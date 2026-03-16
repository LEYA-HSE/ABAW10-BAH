# coding: utf-8
from __future__ import annotations


TEXT_IMPL_SPECS: dict[str, dict[str, str | tuple[str, ...]]] = {
    "2": {
        "aliases": ("2", "2__exp_1_model_13", "2__exp_1_model_13.pkl", "tfidf_logreg"),
        "checkpoint": "./assets/checkpoints/text/2__exp_1_model_13.pkl",
        "hf_model_name": "",
    },
    "3": {
        "aliases": ("3", "3__exp_1_model_1", "3__exp_1_model_1.pkl", "tfidf_catboost"),
        "checkpoint": "./assets/checkpoints/text/3__exp_1_model_1.pkl",
        "hf_model_name": "",
    },
    "6": {
        "aliases": (
            "6",
            "6__exp_5_model_3",
            "6__exp_5_model_3.pt",
            "emotion_text_classifier",
            "emotion_text_classifier_ft",
        ),
        "checkpoint": "./assets/checkpoints/text/6__exp_5_model_3.pt",
        "hf_model_name": "michellejieli/emotion_text_classifier",
    },
    "7": {
        "aliases": (
            "7",
            "7__exp_5_model_4",
            "7__exp_5_model_4.pt",
            "emotion_english_distilroberta",
            "emotion_english_distilroberta_ft",
        ),
        "checkpoint": "./assets/checkpoints/text/7__exp_5_model_4.pt",
        "hf_model_name": "j-hartmann/emotion-english-distilroberta-base",
    },
}


def supported_text_impl_values() -> str:
    values: list[str] = []
    for impl_id, spec in TEXT_IMPL_SPECS.items():
        values.append(impl_id)
        values.extend(str(alias) for alias in spec["aliases"])  # type: ignore[index]
    uniq = sorted(set(values), key=str.lower)
    return ", ".join(uniq)


def normalize_text_impl(raw_value: str, *, default_impl: str = "7") -> str:
    raw = str(raw_value).strip().lower()
    if not raw:
        return default_impl

    for impl_id, spec in TEXT_IMPL_SPECS.items():
        aliases = [str(x).lower() for x in spec["aliases"]]  # type: ignore[index]
        if raw == impl_id or raw in aliases:
            return impl_id

    raise ValueError(
        f"Unsupported text_export.impl='{raw_value}'. Supported values: {supported_text_impl_values()}"
    )


def default_text_checkpoint_for_impl(impl: str) -> str:
    spec = TEXT_IMPL_SPECS.get(str(impl))
    if spec is None:
        return ""
    return str(spec["checkpoint"])


def default_text_hf_model_for_impl(impl: str) -> str:
    spec = TEXT_IMPL_SPECS.get(str(impl))
    if spec is None:
        return ""
    return str(spec["hf_model_name"])
