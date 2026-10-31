#!/usr/bin/env python3
"""Round55 full fixed-physics baseline for frozen Huayi R52-C-75 rows."""

from __future__ import annotations

import argparse
import importlib.util
import math
import multiprocessing as mp
import queue
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
ROUND52_DIR = Path("/Users/liuzhen/AI-projects/gasifier-data-prep/outputs/round52")
PRIMARY_DATASET_ID = "R52-C-75"
FORBIDDEN_CLAIMS = (
    "no WGS scalar fitting; no low-capacity alpha_wgs(x_t); no direct residual "
    "fitting; no learned multi-alpha; no z_d; no Arrhenius tuning; no window, "
    "split, target, or metric changes; no predictive-improvement claim"
)


def _load_round54_module():
    path = ROOT / "scripts" / "generate_huayi_round54_tractable_evaluator.py"
    spec = importlib.util.spec_from_file_location("round54_evaluator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_frozen_rows(round54: Any) -> tuple[bool, list[dict[str, Any]], pd.DataFrame]:
    split_manifest = _read_yaml(ROUND52_DIR / "huayi_round52_split_freeze.yaml")
    inventory = pd.read_csv(ROUND52_DIR / "huayi_round52_window_inventory.csv")
    primary = inventory[inventory["dataset_id"] == PRIMARY_DATASET_ID].copy()
    expected = {role: set(split_manifest["splits"]["row_ids"][role]) for role in ("train", "validation", "test")}
    frozen_consumed = all(set(primary.loc[primary["split_role"] == role, "window_id"]) == ids for role, ids in expected.items())
    return frozen_consumed, round54._model_case_rows(primary), primary


def _status_class(status: str) -> str:
    if status == "ok":
        return "completed"
    if status == "timeout":
        return "timeout"
    if str(status).startswith("error:"):
        return "error"
    return "other"


def _row_runtime_table(predictions: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "dataset_id",
        "window_id",
        "split_role",
        "phase",
        "method",
        "N_cells",
        "row_timeout_s",
        "status",
        "wall_time_s",
        "fallback_count",
        "poor_convergence_count",
        "poor_convergence_cell_index",
        "cell0_cost",
    ]
    rows = predictions[cols].drop_duplicates("window_id").copy()
    rows["timeout_flag"] = rows["status"].eq("timeout")
    rows["error_message"] = rows["status"].where(rows["status"].astype(str).str.startswith("error:"), "")
    return rows


def _metric_matrix(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (split_role, target), group in predictions.groupby(["split_role", "target"], dropna=False):
        attempted = int(group["window_id"].nunique())
        completed_group = group[group["status"] == "ok"].copy()
        completed = int(completed_group["window_id"].nunique())
        timeout = int(group[group["status"] == "timeout"]["window_id"].nunique())
        error = int(group[group["status"].astype(str).str.startswith("error:")]["window_id"].nunique())
        if completed:
            residual = completed_group["residual"].astype(float).to_numpy()
            rmse = float(np.sqrt(np.mean(residual * residual)))
            mae = float(np.mean(np.abs(residual)))
            bias = float(np.mean(residual))
        else:
            rmse = mae = bias = math.nan
        rows.append(
            {
                "dataset_id": PRIMARY_DATASET_ID,
                "split_role": split_role,
                "method": "fixed_physics",
                "target": target,
                "metric_scope": "full_frozen_split_completed_rows",
                "rmse": rmse,
                "mae": mae,
                "bias": bias,
                "n_attempted": attempted,
                "n_completed": completed,
                "n_timeout": timeout,
                "n_error": error,
                "status": "ok" if completed == attempted else "partial",
            }
        )
    return pd.DataFrame(rows).sort_values(["split_role", "target"]).reset_index(drop=True)


def _worker_solve(row: Mapping[str, Any], n_cells: int, result_queue: Any) -> None:
    started = time.perf_counter()
    try:
        src_dir = str(ROOT / "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from model.bounded_hybrid_correction_smoke import solve_with_wgs_correction

        result = solve_with_wgs_correction(row, 1.0, n_cells=n_cells)
        result_queue.put(
            {
                "status": "ok",
                "wall_time_s": time.perf_counter() - started,
                "pred_CO_vol_pct": float(result["CO_volpct_dry"]),
                "pred_H2_vol_pct": float(result["H2_volpct_dry"]),
                "pred_CO2_vol_pct": float(result["CO2_volpct_dry"]),
                "pred_H2_CO_ratio": float(result["H2_CO_ratio"]),
                "pred_dry_flow_nm3_h": float(row["syngas_dry_volumetric_flow_value"]),
                "fallback_count": int(result.get("fallback_count", 0) or 0),
                "poor_convergence_count": int(result.get("poor_convergence_count", 0) or 0),
                "poor_convergence_cell_index": result.get("poor_convergence_cell_index", ""),
                "cell0_cost": result.get("cell0_cost", math.nan),
            }
        )
    except Exception as exc:  # pragma: no cover - runtime diagnostic path
        result_queue.put(
            {
                "status": f"error:{type(exc).__name__}:{exc}",
                "wall_time_s": time.perf_counter() - started,
            }
        )


def _run_row_with_timeout(row: Mapping[str, Any], *, n_cells: int, timeout_s: float) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(target=_worker_solve, args=(dict(row), int(n_cells), result_queue))
    started = time.perf_counter()
    proc.start()
    proc.join(float(timeout_s))
    if proc.is_alive():
        proc.terminate()
        proc.join(2.0)
        return {"status": "timeout", "wall_time_s": time.perf_counter() - started}
    try:
        return dict(result_queue.get_nowait())
    except queue.Empty:
        return {"status": f"error:empty_result_exitcode_{proc.exitcode}", "wall_time_s": time.perf_counter() - started}


def _prediction_rows(round54: Any, model_rows: Sequence[Mapping[str, Any]], timeout_s: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in model_rows:
        outcome = _run_row_with_timeout(row, n_cells=round54.N_CELLS, timeout_s=timeout_s)
        base = {
            "dataset_id": row["dataset_id"],
            "window_id": row["window_id"],
            "split_role": row["split_role"],
            "phase": row["phase_number"],
            "method": "fixed_physics",
            "N_cells": round54.N_CELLS,
            "row_timeout_s": timeout_s,
            "status": outcome.get("status", "unknown"),
            "wall_time_s": outcome.get("wall_time_s", math.nan),
            "fallback_count": outcome.get("fallback_count", math.nan),
            "poor_convergence_count": outcome.get("poor_convergence_count", math.nan),
            "poor_convergence_cell_index": outcome.get("poor_convergence_cell_index", ""),
            "cell0_cost": outcome.get("cell0_cost", math.nan),
        }
        for target, true_col, pred_col in round54.TARGETS:
            y_true = float(row[true_col])
            y_pred = float(outcome[pred_col]) if outcome.get("status") == "ok" and pred_col in outcome else math.nan
            item = dict(base)
            item.update(
                {
                    "target": target,
                    "target_value": y_true,
                    "predicted_value": y_pred,
                    "residual": y_pred - y_true if math.isfinite(y_pred) else math.nan,
                }
            )
            rows.append(item)
    return pd.DataFrame(rows)


def _run_full_predictions(
    round54: Any,
    model_rows: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    timeout_s: float,
) -> pd.DataFrame:
    cache_path = output_dir / "_huayi_round55_row_solve_cache.csv"
    cached = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame()
    if not cached.empty and cached["status"].astype(str).str.contains("empty_result_exitcode_1").any():
        cached = pd.DataFrame()
    done_ids = set(cached["window_id"]) if not cached.empty and "window_id" in cached.columns else set()
    chunks: list[pd.DataFrame] = [cached] if not cached.empty else []
    started = time.perf_counter()
    total = len(model_rows)
    for index, row in enumerate(model_rows, start=1):
        if row["window_id"] in done_ids:
            continue
        predictions = _prediction_rows(round54, [row], timeout_s)
        chunks.append(predictions)
        done_ids.add(row["window_id"])
        pd.concat(chunks, ignore_index=True).to_csv(cache_path, index=False)
        elapsed = time.perf_counter() - started
        print(f"round55 row {index}/{total} {row['split_role']} {row['window_id']} status={predictions['status'].iloc[0]} elapsed={elapsed:.1f}s", flush=True)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def _write(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def run(output_dir: Path, timeout_s: float) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    round54 = _load_round54_module()
    frozen_consumed, model_rows, _primary = _load_frozen_rows(round54)

    predictions = _run_full_predictions(round54, model_rows, output_dir=output_dir, timeout_s=timeout_s)
    predictions.to_csv(output_dir / "huayi_round55_fixed_physics_predictions.csv", index=False)
    metrics = _metric_matrix(predictions)
    metrics.to_csv(output_dir / "huayi_round55_fixed_physics_metric_matrix.csv", index=False)
    runtime = _row_runtime_table(predictions)
    runtime.to_csv(output_dir / "huayi_round55_runtime_and_convergence.csv", index=False)

    attempted = int(runtime["window_id"].nunique())
    completed = int(runtime["status"].eq("ok").sum())
    timeouts = int(runtime["status"].eq("timeout").sum())
    errors = int(runtime["status"].astype(str).str.startswith("error:").sum())
    split_completed = runtime[runtime["status"] == "ok"].groupby("split_role")["window_id"].nunique().to_dict()
    fallback_sum = pd.to_numeric(runtime["fallback_count"], errors="coerce").sum()
    poor_sum = pd.to_numeric(runtime["poor_convergence_count"], errors="coerce").sum()
    wall_sum = float(pd.to_numeric(runtime["wall_time_s"], errors="coerce").sum())
    wall_median = float(pd.to_numeric(runtime["wall_time_s"], errors="coerce").median())

    if completed == 142 and timeouts == 0 and errors == 0 and frozen_consumed:
        decision = "FIXED_BASELINE_READY"
        readiness = "ready: full fixed-physics baseline completed; WGS comparators can be dispatched using the same timeout/cache harness"
        allowed_next = "Dispatch Round56 global bounded WGS scalar and low-capacity WGS comparator using the frozen split and unchanged metrics."
    elif completed >= 120 and frozen_consumed:
        decision = "FIXED_BASELINE_CONDITIONAL"
        readiness = "conditional: most fixed-physics rows completed, but failures/timeouts require bounded WGS subset or engineering pass"
        allowed_next = "Inspect failed rows and either repair evaluator runtime or dispatch a predeclared bounded WGS subset."
    else:
        decision = "FIXED_BASELINE_NO_GO"
        readiness = "not_ready: full fixed-physics baseline did not complete sufficiently under current harness"
        allowed_next = "Fix fixed-physics evaluator before any WGS comparator."

    failed = runtime[runtime["status"] != "ok"].copy()
    failure_lines = [
        "# Huayi Round55 Failure Audit",
        "",
        f"- Decision: {decision}",
        f"- Attempted/completed: {attempted}/{completed}",
        f"- Timeouts/errors: {timeouts}/{errors}",
        "",
    ]
    if failed.empty:
        failure_lines.append("No timeout or error rows.")
    else:
        failure_lines.extend(["| window_id | split | status | wall_time_s | error |", "|---|---|---|---:|---|"])
        for row in failed.itertuples(index=False):
            failure_lines.append(
                f"| {row.window_id} | {row.split_role} | {row.status} | {float(row.wall_time_s):.3f} | {getattr(row, 'error_message', '')} |"
            )
    _write(output_dir / "huayi_round55_failure_audit.md", "\n".join(failure_lines))

    _write(
        output_dir / "huayi_round55_wgs_comparator_readiness.md",
        "\n".join(
            [
                "# Huayi Round55 WGS Comparator Readiness",
                "",
                f"- Readiness: {readiness}",
                f"- Fixed rows completed: {completed}/142",
                f"- Runtime total/median: {wall_sum:.3f}s / {wall_median:.3f}s",
                "- No WGS scalar, alpha_wgs(x_t), direct residual fitting, multi-alpha, z_d, or Arrhenius tuning was run.",
            ]
        ),
    )
    _write(
        output_dir / "huayi_round55_claim_boundary.md",
        "\n".join(
            [
                "# Huayi Round55 Claim Boundary",
                "",
                f"Decision: **{decision}**",
                "",
                "Allowed claim:",
                "- Fixed-physics baseline metrics and runtime/convergence audit are available under the frozen R52-C-75 split.",
                "",
                "Forbidden claims:",
                f"- {FORBIDDEN_CLAIMS}",
            ]
        ),
    )

    metric_summary = "; ".join(
        f"{row.split_role}:{row.target}:rmse={row.rmse:.6g},n={int(row.n_completed)}/{int(row.n_attempted)}"
        for row in metrics.itertuples(index=False)
    )
    report = "\n".join(
        [
            "ROUND55_FIXED_BASELINE_DONE",
            f"Decision: {decision}",
            f"Frozen split consumed: {frozen_consumed}; R52-C-75 row IDs and roles preserved",
            f"Rows attempted/completed: {attempted}/{completed}",
            f"Train/validation/test completed: {int(split_completed.get('train', 0))}/{int(split_completed.get('validation', 0))}/{int(split_completed.get('test', 0))}",
            f"Runtime summary: total_wall_time_s={wall_sum:.3f}; median_row_time_s={wall_median:.3f}; row_timeout_s={timeout_s:.1f}",
            f"Metric summary: {metric_summary}",
            f"Failures/timeouts: timeout={timeouts}; error={errors}",
            f"Cell-0/fallback audit: fallback_count_sum={int(fallback_sum)}; poor_convergence_count_sum={int(poor_sum)}; cell0_cost_min={runtime['cell0_cost'].min()}; cell0_cost_max={runtime['cell0_cost'].max()}",
            f"WGS comparator readiness: {readiness}",
            "Created files: huayi_round55_fixed_physics_predictions.csv, huayi_round55_fixed_physics_metric_matrix.csv, huayi_round55_runtime_and_convergence.csv, huayi_round55_failure_audit.md, huayi_round55_wgs_comparator_readiness.md, huayi_round55_claim_boundary.md, huayi_round55_result.md",
            "Tests/checks run: frozen split row-ID intake check; full 142-row fixed-physics timeout/cache run; row-level cache write/read; split-target metric matrix generated; CSV schema/readback checks; git diff --check scoped to Round55 files",
            "Known limitations: fixed-physics uses Round54 N_cells=1 policy and converter CO2 carrier engineering ratio; no WGS/direct residual/adaptive comparators were run.",
            f"Allowed next task: {allowed_next}",
            f"Forbidden claims: {FORBIDDEN_CLAIMS}",
        ]
    )
    _write(output_dir / "huayi_round55_result.md", report)
    return {"decision": decision, "report": report, "output_dir": output_dir}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Round55 full fixed-physics frozen split baseline")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "diagnostics" / "round55")
    parser.add_argument("--timeout-s", type=float, default=15.0)
    args = parser.parse_args()
    result = run(args.output_dir, args.timeout_s)
    print(result["report"])


if __name__ == "__main__":
    main()
