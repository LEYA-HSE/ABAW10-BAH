#!/usr/bin/env python
# coding: utf-8
from __future__ import annotations

import ast
import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence


STEP_RE = re.compile(r"^Step\s+(?P<step>\d+):\s*(?P<lhs>.*?)\s*=\s*(?P<rhs>.+?)\s*$")
SEL_RE = re.compile(r"^Selection:\s*(?P<metric>[A-Za-z0-9_]+)\s*\[(?P<split>[A-Za-z0-9_]+)\]\s*$")
RES_RE = re.compile(r"^Results\s*\((?P<split>[A-Za-z0-9_]+)\):\s*$")
KV_RE = re.compile(r"^(?P<key>[A-Za-z0-9_]+)\s*=\s*(?P<val>.+?)\s*$")
NUM_RE = re.compile(r"^[+\-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+\-]?\d+)?$")

DEFAULT_COLUMNS = [
    "step",
    "selected_score",
    "selection_split",
    "selection_metric",
    "param:optimizer",
    "param:lr",
    "param:weight_decay",
    "param:d_model",
    "param:x_layers",
    "param:x_heads",
    "avg.MF1",
    "avg.UAR",
    "dev.MF1",
    "test.MF1",
]


def _clean_box_line(line: str) -> str:
    s = line.rstrip("\n")
    if s.startswith("|") and s.endswith("|"):
        s = s[1:-1]
    return s.strip()


def _parse_value(text: str) -> Any:
    t = text.strip()
    if t.endswith("*"):
        t = t[:-1].strip()
    try:
        return ast.literal_eval(t)
    except Exception:
        pass
    if NUM_RE.match(t):
        try:
            return float(t)
        except Exception:
            return t
    return t


def _to_number(v: Any) -> float | None:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str) and NUM_RE.match(v.strip()):
        try:
            return float(v)
        except Exception:
            return None
    return None


def _parse_step_params(lhs: str, rhs: str) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    names = [x.strip() for x in lhs.split("+")] if "+" in lhs else [lhs.strip()]
    rhs_val = _parse_value(rhs)

    if len(names) == 1:
        params[names[0]] = rhs_val
        return params

    if isinstance(rhs_val, tuple) and len(rhs_val) == len(names):
        for k, v in zip(names, rhs_val):
            params[k] = v
        return params

    params[lhs.strip()] = rhs_val
    return params


def parse_overrides(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    records: List[Dict[str, Any]] = []

    global_metric = "MF1"
    global_split = "avg"
    for line in lines:
        m = SEL_RE.match(line.strip())
        if m:
            global_metric = m.group("metric").upper()
            global_split = m.group("split").lower()
            break

    cur: Dict[str, Any] | None = None
    mode: str | None = None
    cur_split: str | None = None

    def flush() -> None:
        nonlocal cur
        if cur is not None:
            records.append(cur)
            cur = None

    for raw in lines:
        line = _clean_box_line(raw)
        if not line:
            continue

        m_step = STEP_RE.match(line)
        if m_step:
            flush()
            step = int(m_step.group("step"))
            lhs = m_step.group("lhs")
            rhs = m_step.group("rhs")
            cur = {
                "step": step,
                "label": lhs,
                "raw_value": rhs,
                "params": _parse_step_params(lhs, rhs),
                "fixed": {},
                "metrics": {"dev": {}, "test": {}, "avg": {}},
                "selection_metric": global_metric,
                "selection_split": global_split,
            }
            mode = None
            cur_split = None
            continue

        if cur is None:
            continue

        m_sel = SEL_RE.match(line)
        if m_sel:
            cur["selection_metric"] = m_sel.group("metric").upper()
            cur["selection_split"] = m_sel.group("split").lower()
            continue

        if line == "Fixed:":
            mode = "fixed"
            cur_split = None
            continue

        m_res = RES_RE.match(line)
        if m_res:
            mode = "results"
            cur_split = m_res.group("split").lower()
            cur["metrics"].setdefault(cur_split, {})
            continue

        m_kv = KV_RE.match(line)
        if not m_kv:
            continue
        k = m_kv.group("key")
        v = _parse_value(m_kv.group("val"))

        if mode == "fixed":
            cur["fixed"][k] = v
        elif mode == "results" and cur_split is not None:
            cur["metrics"][cur_split][k.upper()] = v

    flush()

    for r in records:
        merged = dict(r.get("fixed", {}))
        merged.update(r.get("params", {}))
        r["params_merged"] = merged

        sel_split = str(r.get("selection_split", "avg")).lower()
        sel_metric = str(r.get("selection_metric", "MF1")).upper()
        selected = r.get("metrics", {}).get(sel_split, {}).get(sel_metric)
        r["selected_score"] = _to_number(selected)

    return records


def col_value(rec: Dict[str, Any], col: str) -> Any:
    if col in {"step", "label", "selection_split", "selection_metric", "selected_score"}:
        return rec.get(col)
    if col.startswith("param:"):
        return rec.get("params_merged", {}).get(col[6:])
    if col.startswith("fixed:"):
        return rec.get("fixed", {}).get(col[6:])
    if "." in col:
        split, metric = col.split(".", 1)
        return rec.get("metrics", {}).get(split.lower(), {}).get(metric.upper())
    return rec.get("params_merged", {}).get(col)


def _match_filters(rec: Dict[str, Any], where: Sequence[str]) -> bool:
    for expr in where:
        if "=" not in expr:
            return False
        key, raw_val = expr.split("=", 1)
        key = key.strip()
        want = _parse_value(raw_val.strip())
        got = col_value(rec, key)
        if str(got) != str(want):
            return False
    return True


def analyze(
    file: str | Path,
    *,
    metric: str | None = None,
    split: str | None = None,
    sort_by: str | None = None,
    top: int = 20,
    where: Sequence[str] | None = None,
    min_score: float | None = None,
    max_score: float | None = None,
) -> List[Dict[str, Any]]:
    records = parse_overrides(file)
    if not records:
        return []

    use_split = (split or records[0].get("selection_split") or "avg").lower()
    use_metric = (metric or records[0].get("selection_metric") or "MF1").upper()
    sort_col = sort_by or f"{use_split}.{use_metric}"
    filters = list(where or [])

    filtered: List[Dict[str, Any]] = []
    for rec in records:
        if filters and not _match_filters(rec, filters):
            continue
        score = _to_number(col_value(rec, sort_col))
        if min_score is not None and (score is None or score < min_score):
            continue
        if max_score is not None and (score is None or score > max_score):
            continue
        filtered.append(rec)

    def score_key(rec: Dict[str, Any]) -> float:
        v = _to_number(col_value(rec, sort_col))
        return float("-inf") if v is None else float(v)

    filtered.sort(key=score_key, reverse=True)
    return filtered[: max(0, top)]


def to_rows(records: Sequence[Dict[str, Any]], columns: Sequence[str] | None = None) -> List[Dict[str, Any]]:
    cols = list(columns or DEFAULT_COLUMNS)
    return [{c: col_value(rec, c) for c in cols} for rec in records]


def auto_columns(
    records: Sequence[Dict[str, Any]],
    *,
    include_metrics: bool = True,
    include_fixed: bool = False,
    prefer_order: Sequence[str] | None = None,
) -> List[str]:
    base = ["step", "selected_score", "selection_split", "selection_metric"]
    if prefer_order:
        base = [c for c in prefer_order]

    param_keys = sorted(
        {
            f"param:{k}"
            for r in records
            for k in (r.get("params_merged", {}) or {}).keys()
        }
    )

    fixed_keys = sorted(
        {
            f"fixed:{k}"
            for r in records
            for k in (r.get("fixed", {}) or {}).keys()
        }
    )

    metric_cols: List[str] = []
    if include_metrics:
        pairs = sorted(
            {
                f"{split}.{metric}"
                for r in records
                for split, m in (r.get("metrics", {}) or {}).items()
                for metric in (m or {}).keys()
            }
        )
        metric_cols = pairs

    cols = list(base)
    cols.extend(param_keys)
    if include_fixed:
        cols.extend([c for c in fixed_keys if c.replace("fixed:", "param:") not in param_keys])
    cols.extend([c for c in metric_cols if c not in cols])
    return cols


def to_dataframe(records: Sequence[Dict[str, Any]], columns: Sequence[str] | None = None):
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pandas is required for to_dataframe()") from exc
    return pd.DataFrame(to_rows(records, columns=columns))


def save_csv(records: Sequence[Dict[str, Any]], out_csv: str | Path, columns: Sequence[str] | None = None) -> Path:
    cols = list(columns or DEFAULT_COLUMNS)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for rec in records:
            writer.writerow([col_value(rec, c) for c in cols])
    return out_path


def _as_text(v: Any, max_width: int) -> str:
    s = "" if v is None else str(v)
    if max_width > 0 and len(s) > max_width:
        return s[: max_width - 1] + "..."
    return s


def print_table(
    records: Sequence[Dict[str, Any]],
    *,
    columns: Sequence[str] | None = None,
    max_col_width: int = 28,
) -> None:
    cols = list(columns or DEFAULT_COLUMNS)
    matrix: List[List[str]] = [[_as_text(col_value(r, c), max_width=max_col_width) for c in cols] for r in records]
    widths = [len(h) for h in cols]
    for row in matrix:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def sep() -> str:
        return "+-" + "-+-".join("-" * w for w in widths) + "-+"

    print(sep())
    print("| " + " | ".join(cols[i].ljust(widths[i]) for i in range(len(cols))) + " |")
    print(sep())
    for row in matrix:
        print("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(cols))) + " |")
    print(sep())


def analyze_and_print(
    file: str | Path,
    *,
    metric: str | None = None,
    split: str | None = None,
    sort_by: str | None = None,
    top: int = 20,
    where: Sequence[str] | None = None,
    min_score: float | None = None,
    max_score: float | None = None,
    columns: Sequence[str] | None = None,
    max_col_width: int = 28,
) -> List[Dict[str, Any]]:
    records = analyze(
        file,
        metric=metric,
        split=split,
        sort_by=sort_by,
        top=top,
        where=where,
        min_score=min_score,
        max_score=max_score,
    )
    print_table(records, columns=columns, max_col_width=max_col_width)
    return records


def show_auto_table(
    file: str | Path,
    *,
    metric: str | None = None,
    split: str | None = None,
    sort_by: str | None = None,
    top: int = 20,
    where: Sequence[str] | None = None,
    min_score: float | None = None,
    max_score: float | None = None,
    include_metrics: bool = True,
    include_fixed: bool = False,
    max_col_width: int = 28,
) -> List[Dict[str, Any]]:
    records = analyze(
        file,
        metric=metric,
        split=split,
        sort_by=sort_by,
        top=top,
        where=where,
        min_score=min_score,
        max_score=max_score,
    )
    cols = auto_columns(records, include_metrics=include_metrics, include_fixed=include_fixed)
    print_table(records, columns=cols, max_col_width=max_col_width)
    return records
