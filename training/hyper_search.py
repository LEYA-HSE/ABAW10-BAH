from __future__ import annotations
import itertools
import hashlib
from typing import Dict, Any

from training.config import ExperimentConfig, SearchConfig
from training.utils import ensure_dir, save_json
from training.experiment import run_single_experiment


def _set_by_path(cfg: ExperimentConfig, path: str, value: Any):
    parts = path.split(".")
    if len(parts) != 2:
        raise ValueError(f"Search key must be like 'model.xxx' or 'train.xxx' or 'data.xxx'. Got: {path}")
    group, key = parts
    obj = getattr(cfg, group)
    if not hasattr(obj, key):
        raise AttributeError(f"Config has no field '{path}'")
    setattr(obj, key, value)


def _hash_cfg(cfg: ExperimentConfig) -> str:
    s = str(cfg.to_dict()).encode("utf-8")
    return hashlib.md5(s).hexdigest()[:10]


def run_grid_search(base_cfg: ExperimentConfig, search: SearchConfig) -> Dict[str, Any]:
  
    space = search.space
    keys = list(space.keys())
    values = [space[k] for k in keys]

    base_cfg.train.monitor_metric = "mf1"
    base_cfg.train.monitor_mode = "max"
    search.metric = "mf1"
    search.mode = "max"

    base_workdir = ensure_dir(base_cfg.train.workdir)
    grid_root = ensure_dir(base_workdir / "grid")

    results = []
    best = None

    combos = list(itertools.product(*values))
    if search.max_runs is not None:
        combos = combos[: int(search.max_runs)]

    for i, combo in enumerate(combos, start=1):
        cfg = base_cfg.clone()
        cfg.train.monitor_metric = "mf1"
        cfg.train.monitor_mode = "max"

        for k, v in zip(keys, combo):
            _set_by_path(cfg, k, v)

        run_id = _hash_cfg(cfg)
        cfg.train.workdir = str(grid_root / run_id)

        print(f"[GRID] Run {i}/{len(combos)} | id={run_id} | params={dict(zip(keys, combo))}")
        summary = run_single_experiment(cfg)

        score = float(summary.get("best_score", float("nan")))
        row = {"run_id": run_id, "params": dict(zip(keys, combo)), "summary": summary, "score": score}
        results.append(row)
        save_json(grid_root / "grid_results.json", results)

        if best is None or row["score"] > best["score"]:
            best = row

    save_json(grid_root / "grid_best.json", best)
    return {"best": best, "results_path": str(grid_root / "grid_results.json")}


def run_greedy_search(base_cfg: ExperimentConfig, search: SearchConfig) -> Dict[str, Any]:
   
    keys = list(search.space.keys())

    base_cfg.train.monitor_metric = "mf1"
    base_cfg.train.monitor_mode = "max"
    search.metric = "mf1"
    search.mode = "max"

    base_workdir = ensure_dir(base_cfg.train.workdir)
    greedy_root = ensure_dir(base_workdir / "greedy")

    fixed_cfg = base_cfg.clone()
    fixed_cfg.train.monitor_metric = "mf1"
    fixed_cfg.train.monitor_mode = "max"

    fixed_params = {}
    steps = []
    run_count = 0

    for step_idx, key in enumerate(keys, start=1):
        candidates = search.space[key]
        best_row = None

        for v in candidates:
            cfg = fixed_cfg.clone()
            cfg.train.monitor_metric = "mf1"
            cfg.train.monitor_mode = "max"

            _set_by_path(cfg, key, v)

            run_id = _hash_cfg(cfg)
            cfg.train.workdir = str(greedy_root / f"step{step_idx}_{key.replace('.', '_')}" / run_id)

            run_count += 1
            if search.max_runs is not None and run_count > int(search.max_runs):
                break

            print(f"[GREEDY] Step {step_idx}/{len(keys)} | try {key}={v} | fixed={fixed_params}")
            summary = run_single_experiment(cfg)

            score = float(summary.get("best_score", float("nan")))
            row = {"run_id": run_id, "param": {key: v}, "fixed": dict(fixed_params), "summary": summary, "score": score}

            if best_row is None or row["score"] > best_row["score"]:
                best_row = row

        if best_row is None:
            break

        best_val = list(best_row["param"].values())[0]
        _set_by_path(fixed_cfg, key, best_val)
        fixed_params[key] = best_val
        steps.append(best_row)
        save_json(greedy_root / "greedy_steps.json", steps)

    final = {"fixed_params": fixed_params, "steps": steps, "final_cfg": fixed_cfg.to_dict()}
    save_json(greedy_root / "greedy_final.json", final)
    return final
