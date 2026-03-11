# coding: utf-8
from __future__ import annotations

import copy
import logging
import math
import os
import statistics
from itertools import product
from typing import Any


# Display order: base metrics, then recall_*, then the rest alphabetically.
METRIC_ORDER = [
    "LOSS",
    "ACC",
    "UAR",
    "MF1",
    "UAR_AVG",
    "MF1_AVG",
    "mUAR",
    "mF1",
]


PARAM_ALIASES = {
    # model section aliases
    "d_model": "fusion_d_model",
    "drop": "fusion_drop",
    "input_type": "fusion_input_type",
    "fusion_type": "fusion_type",
    "use_prototypes": "fusion_use_prototypes",
    "num_prototypes": "fusion_num_prototypes",
    "proto_tau": "fusion_proto_tau",
    "x_layers": "fusion_x_layers",
    "x_heads": "fusion_x_heads",
    "x_ff_mult": "fusion_x_ff_mult",
    "x_use_cls": "fusion_x_use_cls",
    "x_layer_impl": "fusion_x_layer_impl",
    "x_positional_encoding": "fusion_x_positional_encoding",
    "videoformer_positional_encoding": "fusion_videoformer_positional_encoding",
    "videoformer_gate_mode": "fusion_videoformer_gate_mode",
    # training section aliases
    "random_seed": "fusion_random_seed",
    "num_epochs": "fusion_num_epochs",
    "max_patience": "fusion_max_patience",
    "optimizer": "fusion_optimizer",
    "lr": "fusion_lr",
    "weight_decay": "fusion_weight_decay",
    "momentum": "fusion_momentum",
    "scheduler_type": "fusion_scheduler_type",
    "warmup_ratio": "fusion_warmup_ratio",
    "loss_name": "fusion_loss_name",
    "label_smoothing": "fusion_label_smoothing",
    "focal_gamma": "fusion_focal_gamma",
    "class_weighting": "fusion_class_weighting",
    "class_weights": "fusion_class_weights",
    "grad_clip": "fusion_grad_clip",
    "lambda_proto": "fusion_lambda_proto",
    "lambda_proto_div": "fusion_lambda_proto_div",
    "save_checkpoints": "fusion_save_checkpoints",
}


def _normalize_key(key: str) -> str:
    return str(key).strip().lower()


def _resolve_param_key(key: str) -> str:
    """
    Resolve human-friendly search keys to runtime ConfigLoader attrs.
    Keeps backward compatibility for existing fusion_* keys.
    """
    k = str(key).strip()
    if not k:
        return k
    if k.startswith("fusion_"):
        return k
    return PARAM_ALIASES.get(k, k)


def _pick_score(metrics: dict[str, Any], metric_name: str) -> float:
    """Select strictly by selection_metric (case-insensitive)."""
    if not isinstance(metrics, dict):
        return 0.0
    target = _normalize_key(metric_name)
    for k, v in metrics.items():
        if _normalize_key(k) == target:
            try:
                return float(v)
            except Exception:
                return 0.0
    return 0.0


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float))


def _avg_metrics(dev_metrics: dict[str, Any], test_metrics: dict[str, Any]) -> dict[str, Any]:
    """Average only numeric metrics present in both dev and test."""
    if not isinstance(dev_metrics, dict):
        dev_metrics = {}
    if not isinstance(test_metrics, dict):
        test_metrics = {}

    out: dict[str, Any] = {}
    shared = sorted(set(dev_metrics.keys()).intersection(test_metrics.keys()))
    for key in shared:
        dv = dev_metrics.get(key)
        tv = test_metrics.get(key)
        if _is_num(dv) and _is_num(tv):
            out[key] = (float(dv) + float(tv)) / 2.0

    if "MF1" in out:
        out["MF1_AVG"] = float(out["MF1"])
    if "UAR" in out:
        out["UAR_AVG"] = float(out["UAR"])
    return out


def _extract_run_metrics(summary: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Pull dev/test metrics from train summary and derive avg metrics."""
    best = summary.get("best", {}) if isinstance(summary, dict) else {}
    dev_metrics = summary.get("final_dev", {}) if isinstance(summary, dict) else {}
    test_metrics = summary.get("final_test", {}) if isinstance(summary, dict) else {}

    if not isinstance(dev_metrics, dict) or not dev_metrics:
        dev_metrics = best.get("dev", {}) if isinstance(best, dict) else {}
    if not isinstance(test_metrics, dict) or not test_metrics:
        test_metrics = best.get("test", {}) if isinstance(best, dict) else {}

    if not isinstance(dev_metrics, dict):
        dev_metrics = {}
    if not isinstance(test_metrics, dict):
        test_metrics = {}

    avg_metrics = _avg_metrics(dev_metrics, test_metrics)
    return dev_metrics, test_metrics, avg_metrics


def _ordered_keys(metrics: dict[str, Any]) -> list[str]:
    base = [k for k in METRIC_ORDER if k in metrics]
    recalls = sorted(k for k in metrics.keys() if k.startswith("recall_"))
    rest = sorted(k for k in metrics.keys() if k not in METRIC_ORDER and not k.startswith("recall_") and k != "by_dataset")
    return base + recalls + rest


def _ordered_keys_ds(ds: dict[str, Any]) -> list[str]:
    base = [k for k in METRIC_ORDER if k in ds and k != "name"]
    recalls = sorted(k for k in ds.keys() if k.startswith("recall_") and k != "name")
    rest = sorted(k for k in ds.keys() if k not in METRIC_ORDER and not k.startswith("recall_") and k != "name")
    return base + recalls + rest


def _format_metrics_block(metrics: dict[str, Any], label: str, *, is_best: bool, selection_metric: str, selection_split: str) -> list[str]:
    lines = [f"  Results ({label.upper()}):"]
    for k in _ordered_keys(metrics):
        v = metrics[k]
        line = f"    {k.upper():16} = {v:.4f}" if _is_num(v) else f"    {k.upper():16} = {v}"
        if is_best and _normalize_key(label) == _normalize_key(selection_split) and _normalize_key(k) == _normalize_key(selection_metric):
            line += " *"
        lines.append(line)
    return lines


def format_result_box_triple(
    step_num: int,
    param_name: str,
    candidate: Any,
    fixed_params: dict[str, Any],
    dev_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    avg_metrics: dict[str, Any],
    *,
    is_best: bool = False,
    selection_metric: str = "MF1",
    selection_split: str = "avg",
) -> str:
    """Pretty ASCII box with DEV/TEST/AVG blocks."""
    title = f"Step {step_num}: {param_name} = {candidate}"
    fixed_lines = [f"{k} = {v}" for k, v in fixed_params.items()]
    content_lines = [
        title,
        f"  Selection: {selection_metric} [{selection_split}]",
        "  Fixed:",
    ]
    content_lines += [f"    {line}" for line in fixed_lines]
    content_lines += _format_metrics_block(dev_metrics or {}, "dev", is_best=is_best, selection_metric=selection_metric, selection_split=selection_split)
    content_lines.append("")
    content_lines += _format_metrics_block(test_metrics or {}, "test", is_best=is_best, selection_metric=selection_metric, selection_split=selection_split)
    content_lines.append("")
    content_lines += _format_metrics_block(avg_metrics or {}, "avg", is_best=is_best, selection_metric=selection_metric, selection_split=selection_split)

    max_width = max(len(line) for line in content_lines) if content_lines else 0
    border = "+" + "-" * (max_width + 2) + "+"
    box = [border]
    for line in content_lines:
        box.append(f"| {line.ljust(max_width)} |")
    box.append(border)
    return "\n".join(box)


def _log_dataset_metrics(metrics: dict[str, Any], file_path: str, label: str) -> None:
    by_ds = metrics.get("by_dataset") if isinstance(metrics, dict) else None
    if not isinstance(by_ds, list):
        return
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"\n>>> Detailed metrics per dataset ({label})\n")
        for ds in by_ds:
            name = ds.get("name", "unknown")
            f.write(f"  - {name}:\n")
            for k in _ordered_keys_ds(ds):
                v = ds[k]
                if _is_num(v):
                    f.write(f"      {k.upper():14} = {float(v):.4f}\n")
                else:
                    f.write(f"      {k.upper():14} = {v}\n")
        f.write(f"<<< End of detailed metrics ({label})\n")


def _choose_split_metrics(
    dev_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    avg_metrics: dict[str, Any],
    selection_split: str,
) -> dict[str, Any]:
    split = str(selection_split).lower()
    if split == "dev":
        return dev_metrics
    if split == "test":
        return test_metrics
    return avg_metrics


def _avg_numeric_metrics(metrics_list: list[dict[str, Any]]) -> dict[str, Any]:
    if not metrics_list:
        return {}
    keys = sorted(set().union(*(m.keys() for m in metrics_list if isinstance(m, dict))))
    out: dict[str, Any] = {}
    for key in keys:
        vals = [m.get(key) for m in metrics_list if isinstance(m, dict) and _is_num(m.get(key))]
        if vals:
            out[key] = float(sum(float(v) for v in vals) / len(vals))
    return out


def _apply_param_overrides(cfg: Any, overrides: dict[str, Any]) -> None:
    for raw_key, value in overrides.items():
        key = _resolve_param_key(raw_key)
        if not hasattr(cfg, key):
            logging.warning(
                "[search] config has no attribute '%s' (from '%s', value=%s). It will be attached dynamically.",
                key,
                raw_key,
                value,
            )
        setattr(cfg, key, value)


def _exporter_header_lines(cfg) -> list[str]:
    return [
        "Exporters:",
        f"  face:  tag={getattr(cfg, 'face_artifact_tag', '')} | extractor={getattr(cfg, 'video_extractor', '')}",
        (
            "  audio: "
            f"tag={getattr(cfg, 'audio_artifact_tag', '')} | "
            f"source={getattr(cfg, 'audio_export_source', '')} | "
            f"impl={getattr(cfg, 'audio_export_impl', '')} ({getattr(cfg, 'audio_export_impl_resolved', '')})"
        ),
        (
            "  text:  "
            f"tag={getattr(cfg, 'text_artifact_tag', '')} | "
            f"impl={getattr(cfg, 'text_export_impl', '')} ({getattr(cfg, 'text_export_impl_resolved', '')}) | "
            f"column={getattr(cfg, 'text_input_column', '')}"
        ),
        f"  scene: tag={getattr(cfg, 'scene_artifact_tag', '')} | model={getattr(cfg, 'scene_model_name', '')}",
    ]


def _run_one(
    *,
    base_config,
    params: dict[str, Any],
    train_loader,
    dev_loader,
    test_loader,
    train_fn,
    run_dir: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    cfg = copy.copy(base_config)
    _apply_param_overrides(cfg, params)
    os.makedirs(run_dir, exist_ok=True)
    summary = train_fn(
        cfg,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        results_dir=run_dir,
    )
    if not isinstance(summary, dict):
        raise TypeError(f"train_fn must return dict summary, got {type(summary)}")
    dev_metrics, test_metrics, avg_metrics = _extract_run_metrics(summary)
    return summary, dev_metrics, test_metrics, avg_metrics


def greedy_search(
    base_config,
    train_loader,
    dev_loader,
    test_loader,
    train_fn,
    overrides_file: str,
    param_grid: dict[str, list],
    default_values: dict[str, Any],
    *,
    runs_root: str,
) -> dict[str, Any]:
    """
    Stepwise search.
    Selection strictly by cfg.search_selection_metric on cfg.search_early_stop_on (avg/dev/test).
    """
    current_best_params = copy.deepcopy(default_values)
    all_param_names = list(param_grid.keys())
    for name in all_param_names:
        if name not in current_best_params and param_grid.get(name):
            current_best_params[name] = param_grid[name][0]
    selection_metric = getattr(base_config, "search_selection_metric", "MF1")
    selection_split = getattr(base_config, "search_early_stop_on", "avg")

    with open(overrides_file, "a", encoding="utf-8") as f:
        f.write("=== Greedy hyperparameter search (Multimodal) ===\n")
        f.write(f"Selection: {selection_metric} [{selection_split}]\n")
        for line in _exporter_header_lines(base_config):
            f.write(line + "\n")

    overall_best_score = float("-inf")
    overall_best_cfg: dict[str, Any] = copy.deepcopy(current_best_params)

    for i, param_name in enumerate(all_param_names):
        candidates = list(param_grid[param_name])
        tried_value = current_best_params.get(param_name)
        candidates_now = [v for v in candidates if v != tried_value]

        best_val_for_param = tried_value
        best_metric_for_param = float("-inf")

        params_def = copy.deepcopy(current_best_params)
        run_dir = os.path.join(runs_root, f"greedy_step{i+1}_{param_name}_{tried_value}")
        _, dev_def, test_def, avg_def = _run_one(
            base_config=base_config,
            params=params_def,
            train_loader=train_loader,
            dev_loader=dev_loader,
            test_loader=test_loader,
            train_fn=train_fn,
            run_dir=run_dir,
        )
        eval_metrics = _choose_split_metrics(dev_def, test_def, avg_def, selection_split)
        cur_score = _pick_score(eval_metrics, selection_metric)
        best_metric_for_param = cur_score

        box = format_result_box_triple(
            i + 1,
            param_name,
            tried_value,
            {k: v for k, v in current_best_params.items() if k != param_name},
            dev_def,
            test_def,
            avg_def,
            is_best=True,
            selection_metric=selection_metric,
            selection_split=selection_split,
        )
        with open(overrides_file, "a", encoding="utf-8") as f:
            f.write("\n" + box + "\n")
        _log_dataset_metrics(dev_def, overrides_file, "dev")
        _log_dataset_metrics(test_def, overrides_file, "test")
        _log_dataset_metrics(avg_def, overrides_file, "avg")

        for cand in candidates_now:
            params_cand = copy.deepcopy(current_best_params)
            params_cand[param_name] = cand
            logging.info("[GREEDY step %d] try %s=%s with fixed=%s", i + 1, param_name, cand, current_best_params)

            run_dir = os.path.join(runs_root, f"greedy_step{i+1}_{param_name}_{cand}")
            _, dev_met, test_met, avg_met = _run_one(
                base_config=base_config,
                params=params_cand,
                train_loader=train_loader,
                dev_loader=dev_loader,
                test_loader=test_loader,
                train_fn=train_fn,
                run_dir=run_dir,
            )
            eval_metrics = _choose_split_metrics(dev_met, test_met, avg_met, selection_split)
            cand_score = _pick_score(eval_metrics, selection_metric)
            is_better = cand_score > best_metric_for_param

            box = format_result_box_triple(
                i + 1,
                param_name,
                cand,
                {k: v for k, v in current_best_params.items() if k != param_name},
                dev_met,
                test_met,
                avg_met,
                is_best=is_better,
                selection_metric=selection_metric,
                selection_split=selection_split,
            )
            with open(overrides_file, "a", encoding="utf-8") as f:
                f.write("\n" + box + "\n")
            _log_dataset_metrics(dev_met, overrides_file, "dev")
            _log_dataset_metrics(test_met, overrides_file, "test")
            _log_dataset_metrics(avg_met, overrides_file, "avg")

            if is_better:
                best_val_for_param = cand
                best_metric_for_param = cand_score

        current_best_params[param_name] = best_val_for_param
        with open(overrides_file, "a", encoding="utf-8") as f:
            f.write(
                f"\n>> [Step {i+1} result] best {param_name}={best_val_for_param}, "
                f"{selection_split}_{selection_metric}={best_metric_for_param:.4f}\n"
            )

        if best_metric_for_param > overall_best_score:
            overall_best_score = best_metric_for_param
            overall_best_cfg = copy.deepcopy(current_best_params)

    with open(overrides_file, "a", encoding="utf-8") as f:
        f.write("\n=== Final combination ===\n")
        for k, v in current_best_params.items():
            f.write(f"{k} = {v}\n")
    logging.info("Greedy search finished. Best score=%.4f", overall_best_score)
    return {"best_score": overall_best_score, "best_params": overall_best_cfg}


def exhaustive_search(
    base_config,
    train_loader,
    dev_loader,
    test_loader,
    train_fn,
    overrides_file: str,
    param_grid: dict[str, list],
    default_values: dict[str, Any],
    *,
    runs_root: str,
) -> dict[str, Any]:
    """
    Full grid search.
    Selection strictly by cfg.search_selection_metric on cfg.search_early_stop_on (avg/dev/test).
    """
    all_param_names = list(param_grid.keys())
    selection_metric = getattr(base_config, "search_selection_metric", "MF1")
    selection_split = getattr(base_config, "search_early_stop_on", "avg")

    with open(overrides_file, "a", encoding="utf-8") as f:
        f.write("=== Exhaustive hyperparameter search (Multimodal) ===\n")
        f.write(f"Selection: {selection_metric} [{selection_split}]\n")
        for line in _exporter_header_lines(base_config):
            f.write(line + "\n")

    best_config = None
    best_score = float("-inf")
    combo_id = 0

    base_params = copy.deepcopy(default_values or {})

    for combo in product(*(param_grid[p] for p in all_param_names)):
        combo_id += 1
        param_combo = copy.deepcopy(base_params)
        param_combo.update(dict(zip(all_param_names, combo)))
        run_dir = os.path.join(runs_root, f"combo_{combo_id:04d}")
        logging.info("[EXHAUSTIVE combo %d] %s", combo_id, param_combo)

        _, dev_met, test_met, avg_met = _run_one(
            base_config=base_config,
            params=param_combo,
            train_loader=train_loader,
            dev_loader=dev_loader,
            test_loader=test_loader,
            train_fn=train_fn,
            run_dir=run_dir,
        )
        eval_metrics = _choose_split_metrics(dev_met, test_met, avg_met, selection_split)
        cand_score = _pick_score(eval_metrics, selection_metric)
        is_better = cand_score > best_score

        box = format_result_box_triple(
            combo_id,
            " + ".join(all_param_names),
            str(combo),
            {},
            dev_met,
            test_met,
            avg_met,
            is_best=is_better,
            selection_metric=selection_metric,
            selection_split=selection_split,
        )
        with open(overrides_file, "a", encoding="utf-8") as f:
            f.write("\n" + box + "\n")
        _log_dataset_metrics(dev_met, overrides_file, "dev")
        _log_dataset_metrics(test_met, overrides_file, "test")
        _log_dataset_metrics(avg_met, overrides_file, "avg")

        if is_better:
            best_score = cand_score
            best_config = param_combo

    with open(overrides_file, "a", encoding="utf-8") as f:
        f.write("\n=== Best combination ===\n")
        for k, v in (best_config or {}).items():
            f.write(f"{k} = {v}\n")
    logging.info("Exhaustive search finished. Best score=%.4f", best_score)
    return {"best_score": best_score, "best_params": best_config or {}}


def optuna_search(
    base_config,
    train_loader,
    dev_loader,
    test_loader,
    train_fn,
    overrides_file: str,
    param_grid: dict[str, list],
    default_values: dict[str, Any],
    *,
    runs_root: str,
    optuna_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Optuna-based search over discrete space from [grid].
    Selection strictly by cfg.search_selection_metric on cfg.search_early_stop_on (avg/dev/test).
    """
    try:
        import optuna  # type: ignore
    except Exception as exc:
        raise ImportError(
            "search.type='optuna' requires dependency 'optuna'. Install it first (e.g. pip install optuna)."
        ) from exc

    options = dict(optuna_cfg or {})
    selection_metric = getattr(base_config, "search_selection_metric", "MF1")
    selection_split = getattr(base_config, "search_early_stop_on", "avg")

    n_trials = int(options.get("n_trials", 100))
    timeout_sec_raw = options.get("timeout_sec", None)
    timeout_sec = int(timeout_sec_raw) if timeout_sec_raw not in (None, "", 0, "0") else None
    n_jobs = int(options.get("n_jobs", 1))
    seed = int(options.get("seed", getattr(base_config, "fusion_random_seed", 42)))
    fail_value = float(options.get("fail_value", -1.0))
    direction = str(options.get("direction", "maximize")).lower()
    if direction not in {"maximize", "minimize"}:
        raise ValueError("optuna.direction must be one of: maximize, minimize")
    optuna_save_checkpoints = bool(options.get("save_checkpoints", False))
    eval_seeds_raw = options.get("eval_seeds", [])
    if isinstance(eval_seeds_raw, (list, tuple)):
        optuna_eval_seeds = [int(s) for s in eval_seeds_raw]
    elif eval_seeds_raw in ("", None):
        optuna_eval_seeds = []
    else:
        optuna_eval_seeds = [int(eval_seeds_raw)]
    seed_aggregate = str(options.get("seed_aggregate", "mean")).lower()
    seed_std_penalty = float(options.get("seed_std_penalty", 0.0))
    if seed_aggregate not in {"mean", "mean_std_penalty"}:
        raise ValueError("optuna.seed_aggregate must be one of: mean, mean_std_penalty")

    sampler_name = str(options.get("sampler", "tpe")).lower()
    n_startup_trials = int(options.get("n_startup_trials", 20))
    if sampler_name == "random":
        sampler = optuna.samplers.RandomSampler(seed=seed)
    elif sampler_name == "tpe":
        sampler = optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=n_startup_trials,
            multivariate=bool(options.get("multivariate", True)),
            constant_liar=bool(options.get("constant_liar", False)),
        )
    else:
        raise ValueError("optuna.sampler must be one of: tpe, random")

    pruner_name = str(options.get("pruner", "none")).lower()
    if pruner_name == "none":
        pruner = optuna.pruners.NopPruner()
    elif pruner_name == "median":
        pruner = optuna.pruners.MedianPruner(n_startup_trials=int(options.get("pruner_startup_trials", 5)))
    else:
        raise ValueError("optuna.pruner must be one of: none, median")

    storage = str(options.get("storage", "")).strip() or None
    study_name_raw = str(options.get("study_name", "")).strip()
    study_name = study_name_raw or None
    load_if_exists = bool(options.get("load_if_exists", False))

    def _collect_space_from_grid(grid: dict[str, list] | None) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, values in (grid or {}).items():
            vals = list(values) if isinstance(values, (list, tuple)) else []
            if not vals:
                continue
            out[key] = vals
        return out

    raw_space = options.get("space", {})
    if isinstance(raw_space, dict) and raw_space:
        space: dict[str, Any] = dict(raw_space)
        space_source = "optuna.space"
    else:
        space = _collect_space_from_grid(param_grid)
        space_source = "grid"
    if not space:
        raise ValueError(
            "Optuna search space is empty. Provide [optuna.space] in search_params.toml "
            "(or keep non-empty [grid] as fallback)."
        )

    base_params = copy.deepcopy(default_values or {})
    os.makedirs(runs_root, exist_ok=True)

    with open(overrides_file, "a", encoding="utf-8") as f:
        f.write("=== Optuna hyperparameter search (Multimodal) ===\n")
        f.write(f"Selection: {selection_metric} [{selection_split}]\n")
        f.write(f"Space source: {space_source}\n")
        f.write(
            "Optuna: "
            f"n_trials={n_trials}, timeout_sec={timeout_sec}, n_jobs={n_jobs}, "
            f"sampler={sampler_name}, pruner={pruner_name}, seed={seed}, "
            f"save_checkpoints={optuna_save_checkpoints}\n"
        )
        f.write(
            "Eval seeds: "
            f"{optuna_eval_seeds if optuna_eval_seeds else '[single-seed from config/trial]'} | "
            f"aggregate={seed_aggregate} | std_penalty={seed_std_penalty}\n"
        )
        f.write(f"Search params: {sorted(space.keys())}\n")
        for line in _exporter_header_lines(base_config):
            f.write(line + "\n")

    best_tracker = {"score": float("-inf"), "params": copy.deepcopy(base_params)}

    def _suggest_from_spec(trial, param_name: str, spec: Any) -> Any:
        # Shorthand: list/tuple => categorical
        if isinstance(spec, (list, tuple)):
            values = list(spec)
            if not values:
                raise ValueError(f"Empty categorical choices for '{param_name}'")
            return trial.suggest_categorical(param_name, values)

        # Shorthand: scalar => fixed constant
        if not isinstance(spec, dict):
            return spec

        kind = str(spec.get("type", "categorical")).strip().lower()
        if kind in {"categorical", "choice", "choices"}:
            values = spec.get("choices", spec.get("values", spec.get("options", [])))
            if not isinstance(values, (list, tuple)) or len(values) == 0:
                raise ValueError(
                    f"optuna.space.{param_name}: categorical requires non-empty choices/values/options"
                )
            return trial.suggest_categorical(param_name, list(values))

        if kind in {"float", "uniform", "loguniform"}:
            if "low" not in spec or "high" not in spec:
                raise ValueError(f"optuna.space.{param_name}: float requires low/high")
            low = float(spec["low"])
            high = float(spec["high"])
            step_raw = spec.get("step", None)
            step = None
            if step_raw not in (None, "", 0, 0.0):
                step = float(step_raw)
            log = bool(spec.get("log", False))
            return trial.suggest_float(param_name, low, high, step=step, log=log)

        if kind in {"int", "integer"}:
            if "low" not in spec or "high" not in spec:
                raise ValueError(f"optuna.space.{param_name}: int requires low/high")
            low = int(spec["low"])
            high = int(spec["high"])
            step = int(spec.get("step", 1))
            log = bool(spec.get("log", False))
            return trial.suggest_int(param_name, low, high, step=step, log=log)

        if kind in {"bool", "boolean"}:
            return trial.suggest_categorical(param_name, [False, True])

        if kind in {"fixed", "const", "constant"}:
            return spec.get("value")

        raise ValueError(
            f"Unsupported optuna.space.{param_name}.type='{kind}'. "
            "Use: categorical|float|int|bool|fixed."
        )

    def _objective(trial) -> float:
        sampled: dict[str, Any] = {}
        if "scheduler_type" in space:
            sampled["scheduler_type"] = _suggest_from_spec(trial, "scheduler_type", space["scheduler_type"])

        for param_name, spec in space.items():
            if param_name == "scheduler_type":
                continue
            if param_name == "warmup_ratio":
                sched_name = str(
                    sampled.get(
                        "scheduler_type",
                        base_params.get("scheduler_type", getattr(base_config, "fusion_scheduler_type", "")),
                    )
                ).lower()
                if not sched_name.startswith("huggingface_"):
                    continue
            sampled[param_name] = _suggest_from_spec(trial, param_name, spec)

        trial_params = copy.deepcopy(base_params)
        trial_params.update(sampled)
        if "save_checkpoints" not in sampled and "fusion_save_checkpoints" not in sampled:
            trial_params["save_checkpoints"] = optuna_save_checkpoints
        trial_seed_list = list(optuna_eval_seeds)
        if not trial_seed_list:
            if "random_seed" in trial_params:
                trial_seed_list = [int(trial_params["random_seed"])]
            elif "fusion_random_seed" in trial_params:
                trial_seed_list = [int(trial_params["fusion_random_seed"])]
            else:
                trial_seed_list = [int(getattr(base_config, "fusion_random_seed", 42))]
        run_dir = os.path.join(runs_root, f"trial_{trial.number + 1:04d}")
        logging.info(
            "[OPTUNA trial %d] params=%s | eval_seeds=%s",
            trial.number + 1,
            trial_params,
            trial_seed_list,
        )

        try:
            per_seed_scores: list[float] = []
            per_seed_dev: list[dict[str, Any]] = []
            per_seed_test: list[dict[str, Any]] = []
            per_seed_avg: list[dict[str, Any]] = []

            for eval_seed in trial_seed_list:
                seed_params = copy.deepcopy(trial_params)
                seed_params["random_seed"] = int(eval_seed)
                seed_run_dir = os.path.join(run_dir, f"seed_{int(eval_seed)}")
                _, dev_met, test_met, avg_met = _run_one(
                    base_config=base_config,
                    params=seed_params,
                    train_loader=train_loader,
                    dev_loader=dev_loader,
                    test_loader=test_loader,
                    train_fn=train_fn,
                    run_dir=seed_run_dir,
                )
                eval_metrics = _choose_split_metrics(dev_met, test_met, avg_met, selection_split)
                seed_score = _pick_score(eval_metrics, selection_metric)
                if not isinstance(seed_score, (int, float)) or math.isnan(float(seed_score)):
                    seed_score = fail_value
                per_seed_scores.append(float(seed_score))
                per_seed_dev.append(dev_met)
                per_seed_test.append(test_met)
                per_seed_avg.append(avg_met)

            score_mean = float(statistics.fmean(per_seed_scores)) if per_seed_scores else fail_value
            score_std = float(statistics.pstdev(per_seed_scores)) if len(per_seed_scores) > 1 else 0.0
            if seed_aggregate == "mean_std_penalty":
                score = score_mean - float(seed_std_penalty) * score_std
            else:
                score = score_mean

            dev_met = _avg_numeric_metrics(per_seed_dev)
            test_met = _avg_numeric_metrics(per_seed_test)
            avg_met = _avg_numeric_metrics(per_seed_avg)
            avg_met["SEED_SCORE_MEAN"] = score_mean
            avg_met["SEED_SCORE_STD"] = score_std
            avg_met["SEED_SCORE_AGG"] = float(score)
            avg_met["SEED_COUNT"] = len(per_seed_scores)

            is_best = float(score) > float(best_tracker["score"])
            if is_best:
                best_tracker["score"] = float(score)
                best_tracker["params"] = copy.deepcopy(trial_params)

            box = format_result_box_triple(
                trial.number + 1,
                "optuna_params",
                sampled,
                base_params,
                dev_met,
                test_met,
                avg_met,
                is_best=is_best,
                selection_metric=selection_metric,
                selection_split=selection_split,
            )
            with open(overrides_file, "a", encoding="utf-8") as f:
                f.write("\n" + box + "\n")
                f.write(f"  Per-seed scores ({selection_metric}[{selection_split}]): {per_seed_scores}\n")
            _log_dataset_metrics(dev_met, overrides_file, "dev")
            _log_dataset_metrics(test_met, overrides_file, "test")
            _log_dataset_metrics(avg_met, overrides_file, "avg")

            trial.set_user_attr("params_full", trial_params)
            trial.set_user_attr("score", float(score))
            trial.set_user_attr("eval_seeds", trial_seed_list)
            trial.set_user_attr("per_seed_scores", per_seed_scores)
            trial.set_user_attr("score_mean", score_mean)
            trial.set_user_attr("score_std", score_std)
            return float(score)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            logging.exception("[OPTUNA trial %d] failed: %s", trial.number + 1, msg)
            with open(overrides_file, "a", encoding="utf-8") as f:
                f.write(
                    f"\n[OPTUNA trial {trial.number + 1}] FAILED\n"
                    f"params={trial_params}\nerror={msg}\n"
                )
            trial.set_user_attr("error", msg)
            trial.set_user_attr("params_full", trial_params)
            return fail_value

    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=storage,
        load_if_exists=load_if_exists,
    )
    study.optimize(_objective, n_trials=n_trials, timeout=timeout_sec, n_jobs=n_jobs)

    if len(study.trials) > 0:
        best_trial = study.best_trial
        best_score = float(study.best_value)
        best_params = copy.deepcopy(base_params)
        best_params.update(dict(best_trial.params))
        if "save_checkpoints" not in best_params and "fusion_save_checkpoints" not in best_params:
            best_params["save_checkpoints"] = optuna_save_checkpoints
    else:
        best_trial = None
        best_score = float(best_tracker["score"])
        best_params = copy.deepcopy(best_tracker["params"])

    with open(overrides_file, "a", encoding="utf-8") as f:
        f.write("\n=== Best combination (Optuna) ===\n")
        f.write(f"best_score = {best_score:.4f}\n")
        if best_trial is not None:
            f.write(f"best_trial = {best_trial.number + 1}\n")
        for k, v in best_params.items():
            f.write(f"{k} = {v}\n")
    logging.info("Optuna search finished. Best score=%.4f", best_score)
    return {"best_score": best_score, "best_params": best_params}


def _single_run_params_snapshot(cfg) -> dict[str, Any]:
    return {
        "input_type": getattr(cfg, "fusion_input_type", None),
        "fusion_type": getattr(cfg, "fusion_type", None),
        "d_model": getattr(cfg, "fusion_d_model", None),
        "drop": getattr(cfg, "fusion_drop", None),
        "x_layers": getattr(cfg, "fusion_x_layers", None),
        "x_heads": getattr(cfg, "fusion_x_heads", None),
        "x_ff_mult": getattr(cfg, "fusion_x_ff_mult", None),
        "x_use_cls": getattr(cfg, "fusion_x_use_cls", None),
        "x_layer_impl": getattr(cfg, "fusion_x_layer_impl", None),
        "x_positional_encoding": getattr(cfg, "fusion_x_positional_encoding", None),
        "videoformer_positional_encoding": getattr(cfg, "fusion_videoformer_positional_encoding", None),
        "videoformer_gate_mode": getattr(cfg, "fusion_videoformer_gate_mode", None),
        "num_epochs": getattr(cfg, "fusion_num_epochs", None),
        "max_patience": getattr(cfg, "fusion_max_patience", None),
        "optimizer": getattr(cfg, "fusion_optimizer", None),
        "lr": getattr(cfg, "fusion_lr", None),
        "weight_decay": getattr(cfg, "fusion_weight_decay", None),
        "momentum": getattr(cfg, "fusion_momentum", None),
        "scheduler_type": getattr(cfg, "fusion_scheduler_type", None),
        "loss_name": getattr(cfg, "fusion_loss_name", None),
        "label_smoothing": getattr(cfg, "fusion_label_smoothing", None),
        "focal_gamma": getattr(cfg, "fusion_focal_gamma", None),
        "class_weighting": getattr(cfg, "fusion_class_weighting", None),
        "grad_clip": getattr(cfg, "fusion_grad_clip", None),
        "save_checkpoints": getattr(cfg, "fusion_save_checkpoints", None),
        "lambda_proto": getattr(cfg, "fusion_lambda_proto", None),
        "lambda_proto_div": getattr(cfg, "fusion_lambda_proto_div", None),
    }


def write_single_run_overrides(
    *,
    cfg,
    summary: dict[str, Any],
    overrides_file: str,
) -> None:
    """
    Write overrides-style report for search.type='none'.
    """
    selection_metric = getattr(cfg, "search_selection_metric", "MF1")
    selection_split = getattr(cfg, "search_early_stop_on", "avg")
    dev_metrics, test_metrics, avg_metrics = _extract_run_metrics(summary if isinstance(summary, dict) else {})
    selected_metrics = _choose_split_metrics(dev_metrics, test_metrics, avg_metrics, selection_split)
    selected_score = _pick_score(selected_metrics, selection_metric)
    best_ckpt = summary.get("best_checkpoint", "") if isinstance(summary, dict) else ""
    best_score = summary.get("best_score", float("nan")) if isinstance(summary, dict) else float("nan")

    params_snapshot = _single_run_params_snapshot(cfg)
    box = format_result_box_triple(
        1,
        "single_run",
        "config.toml",
        params_snapshot,
        dev_metrics,
        test_metrics,
        avg_metrics,
        is_best=True,
        selection_metric=selection_metric,
        selection_split=selection_split,
    )

    with open(overrides_file, "w", encoding="utf-8") as f:
        f.write("=== Single run report (Multimodal) ===\n")
        f.write(f"Selection: {selection_metric} [{selection_split}]\n")
        f.write(f"selected_score={selected_score:.4f}\n")
        if isinstance(best_score, (int, float)):
            f.write(f"best_score={float(best_score):.4f}\n")
        f.write(f"best_checkpoint={best_ckpt}\n")
        for line in _exporter_header_lines(cfg):
            f.write(line + "\n")
        f.write("\n" + box + "\n")

    _log_dataset_metrics(dev_metrics, overrides_file, "dev")
    _log_dataset_metrics(test_metrics, overrides_file, "test")
    _log_dataset_metrics(avg_metrics, overrides_file, "avg")
