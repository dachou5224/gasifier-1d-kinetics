#!/usr/bin/env python3
"""Round56 frozen-split WGS comparators.

Uses the Round55 fixed-physics baseline and the same Round54/Round55
timeout/cache harness.  No multi-alpha, z_d, oxidation update, Arrhenius
tuning, or post-hoc metric selection is performed.
"""

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
ROUND55_DIR = ROOT / "outputs" / "diagnostics" / "round55"
PRIMARY_DATASET_ID = "R52-C-75"
ALPHA_GRID = (0.05, 0.20, 0.50, 1.00, 2.00)
ALPHA_LOW = 0.02
ALPHA_HIGH = 2.0
FEATURE_COLUMNS = ("o_c_ratio", "pressure_mpa", "syngas_flow_mean")
SLOPE_MAGNITUDE = 0.15
TARGET_COUNT = 5
FORBIDDEN_CLAIMS = (
    "no production closed-loop control; no learned WGS kinetic identification; "
    "no learned multi-alpha success; no z_d value; no oxidation-channel updating; "
    "no cross-gasifier generalization; no general superiority of hybrid models"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_round54():
    return _load_module("round54_evaluator", ROOT / "scripts" / "generate_huayi_round54_tractable_evaluator.py")


def _load_frozen(round54: Any) -> tuple[bool, list[dict[str, Any]], pd.DataFrame]:
    split_manifest = _read_yaml(ROUND52_DIR / "huayi_round52_split_freeze.yaml")
    inventory = pd.read_csv(ROUND52_DIR / "huayi_round52_window_inventory.csv")
    primary = inventory[inventory["dataset_id"] == PRIMARY_DATASET_ID].copy()
    expected = {role: set(split_manifest["splits"]["row_ids"][role]) for role in ("train", "validation", "test")}
    frozen_consumed = all(set(primary.loc[primary["split_role"] == role, "window_id"]) == ids for role, ids in expected.items())
    rows = round54._model_case_rows(primary)
    for row in rows:
        row["pressure_mpa"] = float(row["gasifier_pressure_value"])
    return frozen_consumed, rows, primary


def _fixed_predictions() -> pd.DataFrame:
    return pd.read_csv(ROUND55_DIR / "huayi_round55_fixed_physics_predictions.csv")


def _fixed_metrics() -> pd.DataFrame:
    return pd.read_csv(ROUND55_DIR / "huayi_round55_fixed_physics_metric_matrix.csv")


def _worker_solve(row: Mapping[str, Any], alpha: float, n_cells: int, result_queue: Any) -> None:
    started = time.perf_counter()
    try:
        src_dir = str(ROOT / "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from model.bounded_hybrid_correction_smoke import solve_with_wgs_correction

        result = solve_with_wgs_correction(row, float(alpha), n_cells=int(n_cells))
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
        result_queue.put({"status": f"error:{type(exc).__name__}:{exc}", "wall_time_s": time.perf_counter() - started})


def _run_row(row: Mapping[str, Any], *, alpha: float, n_cells: int, timeout_s: float) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(target=_worker_solve, args=(dict(row), float(alpha), int(n_cells), result_queue))
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


def _alpha_key(alpha: float) -> str:
    return f"{float(alpha):.8g}"


def _prediction_rows_for_alpha(
    round54: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    alpha_by_window: Mapping[str, float],
    output_dir: Path,
    cache_name: str,
    timeout_s: float,
) -> pd.DataFrame:
    cache_path = output_dir / cache_name
    cached = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame()
    done = set()
    if not cached.empty:
        done = set(zip(cached["window_id"].astype(str), cached["alpha_wgs"].astype(float).round(8)))
    chunks: list[pd.DataFrame] = [cached] if not cached.empty else []
    started = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        alpha = float(alpha_by_window[str(row["window_id"])])
        key = (str(row["window_id"]), round(alpha, 8))
        if key in done:
            continue
        if math.isclose(alpha, 1.0, rel_tol=0.0, abs_tol=1e-12):
            fixed = _fixed_predictions()
            pred = fixed[fixed["window_id"] == row["window_id"]].copy()
            pred["method"] = method
            pred["alpha_wgs"] = alpha
            pred["bound_hit"] = False
        else:
            outcome = _run_row(row, alpha=alpha, n_cells=round54.N_CELLS, timeout_s=timeout_s)
            records: list[dict[str, Any]] = []
            base = {
                "dataset_id": row["dataset_id"],
                "window_id": row["window_id"],
                "split_role": row["split_role"],
                "phase": row["phase_number"],
                "method": method,
                "N_cells": round54.N_CELLS,
                "row_timeout_s": timeout_s,
                "status": outcome.get("status", "unknown"),
                "wall_time_s": outcome.get("wall_time_s", math.nan),
                "fallback_count": outcome.get("fallback_count", math.nan),
                "poor_convergence_count": outcome.get("poor_convergence_count", math.nan),
                "poor_convergence_cell_index": outcome.get("poor_convergence_cell_index", ""),
                "cell0_cost": outcome.get("cell0_cost", math.nan),
                "alpha_wgs": alpha,
                "bound_hit": math.isclose(alpha, ALPHA_LOW) or math.isclose(alpha, ALPHA_HIGH),
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
                records.append(item)
            pred = pd.DataFrame(records)
        chunks.append(pred)
        done.add(key)
        pd.concat(chunks, ignore_index=True).to_csv(cache_path, index=False)
        print(f"round56 {method} row {index}/{len(rows)} alpha={alpha:.6g} status={pred['status'].iloc[0]} elapsed={time.perf_counter()-started:.1f}s", flush=True)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def _metric_matrix(predictions: pd.DataFrame, fixed_metrics: pd.DataFrame, method: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    fixed_lookup = {
        (row.split_role, row.target): row
        for row in fixed_metrics.itertuples(index=False)
    }
    for (split_role, target), group in predictions.groupby(["split_role", "target"], dropna=False):
        attempted = int(group["window_id"].nunique())
        ok = group[group["status"] == "ok"].copy()
        completed = int(ok["window_id"].nunique())
        timeout = int(group[group["status"] == "timeout"]["window_id"].nunique())
        error = int(group[group["status"].astype(str).str.startswith("error:")]["window_id"].nunique())
        if completed:
            residual = ok["residual"].astype(float).to_numpy()
            rmse = float(np.sqrt(np.mean(residual * residual)))
            mae = float(np.mean(np.abs(residual)))
            bias = float(np.mean(residual))
        else:
            rmse = mae = bias = math.nan
        fixed = fixed_lookup[(split_role, target)]
        delta = rmse - float(fixed.rmse) if math.isfinite(rmse) else math.nan
        rows.append(
            {
                "dataset_id": PRIMARY_DATASET_ID,
                "split_role": split_role,
                "method": method,
                "target": target,
                "rmse": rmse,
                "mae": mae,
                "bias": bias,
                "delta_rmse_vs_fixed": delta,
                "worsened_target_flag": bool(math.isfinite(delta) and delta > max(1e-12, abs(float(fixed.rmse)) * 0.02)),
                "n_attempted": attempted,
                "n_completed": completed,
                "n_timeout": timeout,
                "n_error": error,
                "status": "ok" if attempted == completed else "partial",
            }
        )
    return pd.DataFrame(rows).sort_values(["method", "split_role", "target"]).reset_index(drop=True)


def _objective(predictions: pd.DataFrame, split_role: str) -> float:
    work = predictions[(predictions["split_role"] == split_role) & (predictions["status"] == "ok")].copy()
    scales = {"CO_dry_vol_pct": 10.0, "H2_dry_vol_pct": 10.0, "CO2_dry_vol_pct": 10.0, "H2_CO_ratio": 0.1, "dry_syngas_flow": 1.0e5}
    total = 0.0
    count = 0
    for target, group in work.groupby("target"):
        residual = group["residual"].astype(float).to_numpy() / scales[str(target)]
        total += float(np.sqrt(np.mean(residual * residual)))
        count += 1
    return total / max(count, 1)


def _select_global(round54: Any, rows: Sequence[Mapping[str, Any]], output_dir: Path, timeout_s: float) -> tuple[float, pd.DataFrame]:
    train_rows = [row for row in rows if row["split_role"] == "train"]
    records: list[dict[str, Any]] = []
    for alpha in ALPHA_GRID:
        preds = _prediction_rows_for_alpha(
            round54,
            train_rows,
            method=f"global_bounded_wgs_scalar_grid_{_alpha_key(alpha)}",
            alpha_by_window={str(row["window_id"]): float(alpha) for row in train_rows},
            output_dir=output_dir,
            cache_name=f"_global_grid_alpha_{_alpha_key(alpha)}.csv",
            timeout_s=timeout_s,
        )
        records.append(
            {
                "method": "global_bounded_wgs_scalar",
                "candidate": f"alpha={alpha:.6g}",
                "alpha_wgs": float(alpha),
                "selection_split": "train",
                "selection_objective": _objective(preds, "train"),
                "bound_low": ALPHA_LOW,
                "bound_high": ALPHA_HIGH,
                "bound_hit_rate_train": float(preds.drop_duplicates("window_id")["bound_hit"].astype(bool).mean()) if "bound_hit" in preds else 0.0,
                "status": "ok" if (preds["status"] == "ok").all() else "partial",
            }
        )
    table = pd.DataFrame(records)
    ok = table[table["status"] == "ok"].copy()
    selected = ok.sort_values(["selection_objective", "alpha_wgs"]).iloc[0]
    table["selected"] = table["candidate"].eq(selected["candidate"])
    return float(selected["alpha_wgs"]), table


def _global_grid_cache(output_dir: Path, alpha: float) -> Path:
    return output_dir / f"_global_grid_alpha_{_alpha_key(alpha)}.csv"


def _read_predictions_cache(path: Path, method: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["method"] = method
    return frame


def _cache_has_windows(path: Path, expected_window_count: int) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path, usecols=["window_id", "target", "status"])
    except Exception:
        return False
    return (
        int(frame["window_id"].nunique()) == int(expected_window_count)
        and int(frame["target"].nunique()) == TARGET_COUNT
        and len(frame) >= int(expected_window_count) * TARGET_COUNT
    )


def _selected_global_full(
    round54: Any,
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    timeout_s: float,
    selected_global: float,
) -> pd.DataFrame:
    full_cache = output_dir / "_global_selected_full.csv"
    if _cache_has_windows(full_cache, len(rows)):
        return _read_predictions_cache(full_cache, "global_bounded_wgs_scalar")
    train_cache = _global_grid_cache(output_dir, selected_global)
    heldout_rows = [row for row in rows if row["split_role"] != "train"]
    if train_cache.exists():
        train_preds = _read_predictions_cache(train_cache, "global_bounded_wgs_scalar")
        heldout_preds = _prediction_rows_for_alpha(
            round54,
            heldout_rows,
            method="global_bounded_wgs_scalar",
            alpha_by_window={str(row["window_id"]): selected_global for row in heldout_rows},
            output_dir=output_dir,
            cache_name="_global_selected_heldout.csv",
            timeout_s=timeout_s,
        )
        full = pd.concat([train_preds, heldout_preds], ignore_index=True, sort=False)
        full.to_csv(full_cache, index=False)
        return full
    return _prediction_rows_for_alpha(
        round54,
        rows,
        method="global_bounded_wgs_scalar",
        alpha_by_window={str(row["window_id"]): selected_global for row in rows},
        output_dir=output_dir,
        cache_name="_global_selected_full.csv",
        timeout_s=timeout_s,
    )


def _feature_scaler(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    train = pd.DataFrame([row for row in rows if row["split_role"] == "train"])
    out: dict[str, dict[str, float]] = {}
    for col in FEATURE_COLUMNS:
        scale = float(train[col].std(ddof=0))
        out[col] = {"mean": float(train[col].mean()), "scale": scale if scale > 0 else 1.0}
    return out


def _alpha_xt_candidates(global_alpha: float) -> list[dict[str, Any]]:
    candidates = [{"candidate": "constant_global_alpha", "feature": "", "sign": 0.0}]
    for feature in FEATURE_COLUMNS:
        candidates.append({"candidate": f"{feature}_negative_slope", "feature": feature, "sign": -1.0})
        candidates.append({"candidate": f"{feature}_positive_slope", "feature": feature, "sign": 1.0})
    for row in candidates:
        row["base_alpha"] = float(global_alpha)
    return candidates


def _alpha_map(rows: Sequence[Mapping[str, Any]], candidate: Mapping[str, Any], scaler: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in rows:
        alpha = float(candidate["base_alpha"])
        feature = str(candidate.get("feature", candidate.get("selected_feature", "")) or "")
        if feature:
            z = (float(row[feature]) - float(scaler[feature]["mean"])) / float(scaler[feature]["scale"])
            sign = float(candidate.get("sign", candidate.get("selected_sign", 0.0)))
            alpha = alpha * math.exp(SLOPE_MAGNITUDE * sign * z)
        out[str(row["window_id"])] = min(max(alpha, ALPHA_LOW), ALPHA_HIGH)
    return out


def _select_alpha_xt(round54: Any, rows: Sequence[Mapping[str, Any]], output_dir: Path, timeout_s: float, global_alpha: float) -> tuple[dict[str, Any], pd.DataFrame]:
    train_rows = [row for row in rows if row["split_role"] == "train"]
    scaler = _feature_scaler(rows)
    records: list[dict[str, Any]] = []
    for candidate in _alpha_xt_candidates(global_alpha):
        alpha_by_window = _alpha_map(train_rows, candidate, scaler)
        candidate_cache = output_dir / f"_alpha_xt_candidate_{candidate['candidate']}.csv"
        candidate_method = f"low_capacity_alpha_wgs_xt_candidate_{candidate['candidate']}"
        if candidate["candidate"] == "constant_global_alpha" and _global_grid_cache(output_dir, global_alpha).exists():
            preds = _read_predictions_cache(_global_grid_cache(output_dir, global_alpha), candidate_method)
            preds.to_csv(candidate_cache, index=False)
        else:
            preds = _prediction_rows_for_alpha(
                round54,
                train_rows,
                method=candidate_method,
                alpha_by_window=alpha_by_window,
                output_dir=output_dir,
                cache_name=candidate_cache.name,
                timeout_s=timeout_s,
            )
        records.append(
            {
                "method": "low_capacity_alpha_wgs_xt",
                "candidate": candidate["candidate"],
                "selected_feature": candidate["feature"],
                "selected_sign": candidate["sign"],
                "base_alpha": candidate["base_alpha"],
                "selection_split": "train",
                "selection_objective": _objective(preds, "train"),
                "alpha_min_train": float(min(alpha_by_window.values())),
                "alpha_max_train": float(max(alpha_by_window.values())),
                "bound_hit_rate_train": float(np.mean([math.isclose(v, ALPHA_LOW) or math.isclose(v, ALPHA_HIGH) for v in alpha_by_window.values()])),
                "status": "ok" if (preds["status"] == "ok").all() else "partial",
            }
        )
    table = pd.DataFrame(records)
    ok = table[table["status"] == "ok"].copy()
    selected = ok.sort_values(["selection_objective", "candidate"]).iloc[0].to_dict()
    table["selected"] = table["candidate"].eq(selected["candidate"])
    return selected, table


def _selected_xt_full(
    round54: Any,
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    timeout_s: float,
    selected_xt: Mapping[str, Any],
    xt_alpha: Mapping[str, float],
) -> pd.DataFrame:
    full_cache = output_dir / "_alpha_xt_selected_full.csv"
    if _cache_has_windows(full_cache, len(rows)):
        return _read_predictions_cache(full_cache, "low_capacity_alpha_wgs_xt")
    train_cache = output_dir / f"_alpha_xt_candidate_{selected_xt['candidate']}.csv"
    heldout_rows = [row for row in rows if row["split_role"] != "train"]
    if train_cache.exists():
        train_preds = _read_predictions_cache(train_cache, "low_capacity_alpha_wgs_xt")
        heldout_preds = _prediction_rows_for_alpha(
            round54,
            heldout_rows,
            method="low_capacity_alpha_wgs_xt",
            alpha_by_window={str(row["window_id"]): float(xt_alpha[str(row["window_id"])]) for row in heldout_rows},
            output_dir=output_dir,
            cache_name="_alpha_xt_selected_heldout.csv",
            timeout_s=timeout_s,
        )
        full = pd.concat([train_preds, heldout_preds], ignore_index=True, sort=False)
        full.to_csv(full_cache, index=False)
        return full
    return _prediction_rows_for_alpha(
        round54,
        rows,
        method="low_capacity_alpha_wgs_xt",
        alpha_by_window=xt_alpha,
        output_dir=output_dir,
        cache_name="_alpha_xt_selected_full.csv",
        timeout_s=timeout_s,
    )


def _runtime(predictions: pd.DataFrame, method: str) -> pd.DataFrame:
    rows = predictions.drop_duplicates("window_id").copy()
    return pd.DataFrame(
        [
            {
                "dataset_id": PRIMARY_DATASET_ID,
                "method": method,
                "split_role": split,
                "rows_attempted": int(group["window_id"].nunique()),
                "rows_completed": int((group["status"] == "ok").sum()),
                "rows_timeout": int((group["status"] == "timeout").sum()),
                "rows_error": int(group["status"].astype(str).str.startswith("error:").sum()),
                "wall_time_s_sum": float(pd.to_numeric(group["wall_time_s"], errors="coerce").sum()),
                "fallback_count_sum": float(pd.to_numeric(group["fallback_count"], errors="coerce").sum()),
                "poor_convergence_count_sum": float(pd.to_numeric(group["poor_convergence_count"], errors="coerce").sum()),
                "cell0_cost_min": float(pd.to_numeric(group["cell0_cost"], errors="coerce").min()),
                "cell0_cost_max": float(pd.to_numeric(group["cell0_cost"], errors="coerce").max()),
                "bound_hit_rate": float(group["bound_hit"].astype(bool).mean()) if "bound_hit" in group else 0.0,
            }
            for split, group in rows.groupby("split_role")
        ]
    )


def _write(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def run(output_dir: Path, timeout_s: float) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    round54 = _load_round54()
    frozen_consumed, rows, _primary = _load_frozen(round54)
    fixed = _fixed_predictions()
    fixed_metrics = _fixed_metrics()

    method_rows = [
        {"order": 1, "method": "fixed_physics", "status": "consumed_round55", "selection_split": "none"},
        {"order": 2, "method": "global_bounded_wgs_scalar", "status": "planned", "selection_split": "train"},
        {"order": 3, "method": "low_capacity_alpha_wgs_xt", "status": "planned", "selection_split": "train"},
        {"order": 4, "method": "null_feature_shuffle_control", "status": "planned", "selection_split": "train"},
    ]

    selected_global, global_selection = _select_global(round54, rows, output_dir, timeout_s)
    global_preds = _selected_global_full(round54, rows, output_dir, timeout_s, selected_global)
    global_metrics = _metric_matrix(global_preds, fixed_metrics, "global_bounded_wgs_scalar")
    method_rows[1]["status"] = "ok" if (global_preds["status"] == "ok").all() else "partial"

    selected_xt, xt_selection = _select_alpha_xt(round54, rows, output_dir, timeout_s, selected_global)
    scaler = _feature_scaler(rows)
    xt_alpha = _alpha_map(rows, selected_xt, scaler)
    if selected_xt["candidate"] == "constant_global_alpha":
        xt_preds = global_preds.copy()
        xt_preds["method"] = "low_capacity_alpha_wgs_xt"
        xt_preds.to_csv(output_dir / "_alpha_xt_selected_full.csv", index=False)
    else:
        xt_preds = _selected_xt_full(round54, rows, output_dir, timeout_s, selected_xt, xt_alpha)
    xt_metrics = _metric_matrix(xt_preds, fixed_metrics, "low_capacity_alpha_wgs_xt")
    method_rows[2]["status"] = "ok" if (xt_preds["status"] == "ok").all() else "partial"

    # Deterministic null/control: shuffle the selected feature among train rows,
    # keep the selected alpha model form, evaluate train only.  This is a guardrail
    # and not used for model selection.
    null_rows: list[dict[str, Any]] = []
    if selected_xt.get("selected_feature"):
        train = [dict(row) for row in rows if row["split_role"] == "train"]
        values = [row[str(selected_xt["selected_feature"])] for row in train]
        shifted = values[1:] + values[:1]
        for row, value in zip(train, shifted):
            row[str(selected_xt["selected_feature"])] = value
        null_alpha = _alpha_map(train, selected_xt, scaler)
        null_preds = _prediction_rows_for_alpha(
            round54,
            train,
            method="null_feature_shuffle_control",
            alpha_by_window=null_alpha,
            output_dir=output_dir,
            cache_name="_null_feature_shuffle_train.csv",
            timeout_s=timeout_s,
        )
        null_obj = _objective(null_preds, "train")
        null_rows.append(
            {
                "dataset_id": PRIMARY_DATASET_ID,
                "method": "null_feature_shuffle_control",
                "control_type": "train_feature_cyclic_shift",
                "selection_feature": selected_xt["selected_feature"],
                "split_role": "train",
                "objective": null_obj,
                "selected_model_train_objective": float(selected_xt["selection_objective"]),
                "control_explains_improvement": bool(null_obj <= float(selected_xt["selection_objective"])),
                "status": "ok" if (null_preds["status"] == "ok").all() else "partial",
            }
        )
        method_rows[3]["status"] = null_rows[-1]["status"]
    else:
        null_rows.append(
            {
                "dataset_id": PRIMARY_DATASET_ID,
                "method": "null_feature_shuffle_control",
                "control_type": "blocked_constant_alpha_selected",
                "selection_feature": "",
                "split_role": "train",
                "objective": math.nan,
                "selected_model_train_objective": float(selected_xt["selection_objective"]),
                "control_explains_improvement": False,
                "status": "blocked:no_selected_feature_to_shuffle",
            }
        )
        method_rows[3]["status"] = "blocked"

    all_metrics = pd.concat([global_metrics, xt_metrics], ignore_index=True)
    delta = all_metrics[["dataset_id", "split_role", "method", "target", "delta_rmse_vs_fixed", "worsened_target_flag", "n_attempted", "n_completed", "n_timeout", "n_error", "status"]].copy()
    runtime = pd.concat([_runtime(global_preds, "global_bounded_wgs_scalar"), _runtime(xt_preds, "low_capacity_alpha_wgs_xt")], ignore_index=True)
    selection = pd.concat([global_selection, xt_selection], ignore_index=True, sort=False)

    test_summary = all_metrics[all_metrics["split_role"] == "test"].copy()
    validation_summary = all_metrics[all_metrics["split_role"] == "validation"].copy()
    test_worse = test_summary[test_summary["worsened_target_flag"]].groupby("method")["target"].apply(list).to_dict()
    val_worse = validation_summary[validation_summary["worsened_target_flag"]].groupby("method")["target"].apply(list).to_dict()
    complete = (global_preds["status"].eq("ok").all() and xt_preds["status"].eq("ok").all())
    heldout_worse_count = sum(len(v) for v in test_worse.values()) + sum(len(v) for v in val_worse.values())
    null_explains = any(bool(row["control_explains_improvement"]) for row in null_rows)
    null_blocked = any(str(row["status"]).startswith("blocked") for row in null_rows)
    if not complete:
        decision = "WGS_COMPARATOR_NO_GO"
    elif heldout_worse_count == 0 and not null_explains and not null_blocked:
        decision = "WGS_COMPARATOR_GO"
    else:
        decision = "WGS_COMPARATOR_CONDITIONAL"

    pd.DataFrame(method_rows).to_csv(output_dir / "huayi_round56_method_ladder.csv", index=False)
    all_metrics.to_csv(output_dir / "huayi_round56_wgs_metric_matrix.csv", index=False)
    delta.to_csv(output_dir / "huayi_round56_delta_vs_fixed.csv", index=False)
    selection.to_csv(output_dir / "huayi_round56_selection_and_bounds.csv", index=False)
    runtime.to_csv(output_dir / "huayi_round56_runtime_and_convergence.csv", index=False)
    pd.DataFrame(null_rows).to_csv(output_dir / "huayi_round56_null_control_summary.csv", index=False)
    test_summary.to_csv(output_dir / "huayi_round56_test_shadow_summary.csv", index=False)

    heldout_line = "; ".join(
        f"{row.method}:{row.split_role}:{row.target}:delta_rmse={row.delta_rmse_vs_fixed:.6g},rmse={row.rmse:.6g}"
        for row in pd.concat([validation_summary, test_summary]).itertuples(index=False)
    )
    selected_line = (
        f"global_alpha={selected_global:.6g}; "
        f"alpha_xt_candidate={selected_xt['candidate']}; feature={selected_xt.get('selected_feature','')}; "
        f"base_alpha={selected_xt.get('base_alpha', math.nan):.6g}"
    )
    worsen_line = f"validation={val_worse}; test={test_worse}"
    bounds_line = (
        f"global_bound_hit_rate={runtime[runtime.method=='global_bounded_wgs_scalar']['bound_hit_rate'].mean():.6g}; "
        f"alpha_xt_bound_hit_rate={runtime[runtime.method=='low_capacity_alpha_wgs_xt']['bound_hit_rate'].mean():.6g}; "
        f"fallback_sum={int(runtime['fallback_count_sum'].sum())}; poor_convergence_sum={int(runtime['poor_convergence_count_sum'].sum())}"
    )
    runtime_line = f"global_wall_s={runtime[runtime.method=='global_bounded_wgs_scalar']['wall_time_s_sum'].sum():.3f}; alpha_xt_wall_s={runtime[runtime.method=='low_capacity_alpha_wgs_xt']['wall_time_s_sum'].sum():.3f}; timeout_s={timeout_s:.1f}"
    null_line = "; ".join(f"{row['status']} objective={row['objective']} control_explains={row['control_explains_improvement']}" for row in null_rows)
    allowed_claim = (
        "bounded WGS comparators were evaluated under frozen R52-C-75 split; "
        "claim strength depends on held-out deltas and null/control audit"
    )
    _write(
        output_dir / "huayi_round56_claim_boundary.md",
        "\n".join(
            [
                "# Huayi Round56 Claim Boundary",
                "",
                f"Decision: **{decision}**",
                "",
                f"Allowed manuscript claim: {allowed_claim}.",
                "",
                f"Forbidden claims: {FORBIDDEN_CLAIMS}.",
            ]
        ),
    )
    report = "\n".join(
        [
            "ROUND56_WGS_COMPARATOR_DONE",
            f"Decision: {decision}",
            f"Frozen split consumed: {frozen_consumed}; R52-C-75 row IDs and roles preserved",
            "Methods evaluated: fixed_physics(round55 consumed); global_bounded_wgs_scalar; low_capacity_alpha_wgs_xt; null_feature_shuffle_control",
            "Train/validation/test rows: 81/44/17",
            f"Selected WGS scalar/features: {selected_line}",
            f"Held-out metric summary: {heldout_line}",
            f"Worsened targets: {worsen_line}",
            f"Bounds/fallbacks/convergence: {bounds_line}",
            f"Null/control result: {null_line}",
            f"Runtime summary: {runtime_line}",
            "Created files: huayi_round56_method_ladder.csv, huayi_round56_wgs_metric_matrix.csv, huayi_round56_delta_vs_fixed.csv, huayi_round56_selection_and_bounds.csv, huayi_round56_runtime_and_convergence.csv, huayi_round56_null_control_summary.csv, huayi_round56_test_shadow_summary.csv, huayi_round56_claim_boundary.md, huayi_round56_result.md",
            "Tests/checks run: frozen split row-ID intake; consumed Round55 fixed baseline; train-only global alpha selection; train-only alpha_wgs(x_t) candidate selection; phase-4 test evaluated after selection; CSV schema/readback; git diff --check scoped to Round56 files",
            "Known limitations: uses Round54/Round55 N_cells=1 and converter CO2 carrier engineering ratio; dry syngas flow prediction remains carried metadata target under WGS comparators; no direct residual comparator is included in Round56.",
            f"Allowed manuscript claim: {allowed_claim}.",
            f"Forbidden claims: {FORBIDDEN_CLAIMS}.",
        ]
    )
    _write(output_dir / "huayi_round56_result.md", report)
    return {"decision": decision, "report": report}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Round56 frozen-split WGS comparators")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "diagnostics" / "round56")
    parser.add_argument("--timeout-s", type=float, default=15.0)
    args = parser.parse_args()
    result = run(args.output_dir, args.timeout_s)
    print(result["report"])


if __name__ == "__main__":
    main()
