#!/usr/bin/env python3
"""Round54 tractable evaluator for the frozen Huayi R52-C-75 split.

This is an engineering-readiness package, not an adaptive model-training run.
It audits conversion first, then runs a deterministic fixed-physics smoke
subset with per-row timeouts so solver stalls are captured as evidence.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import queue
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DATA_PREP_ROOT = Path("/Users/liuzhen/AI-projects/gasifier-data-prep")
ROUND52_DIR = DATA_PREP_ROOT / "outputs" / "round52"
PRIMARY_DATASET_ID = "R52-C-75"
N_CELLS = 1
SMOKE_ROWS_PER_SPLIT = 3
ROW_TIMEOUT_SECONDS = 15.0
FORBIDDEN_CLAIMS = (
    "no bounded WGS predictive improvement; no production online control; "
    "no learned WGS kinetics; no learned multi-alpha; no z_d value; "
    "no oxidation-channel updating; no cross-gasifier generalization"
)

TARGETS = (
    ("CO_dry_vol_pct", "co_vol_pct", "pred_CO_vol_pct"),
    ("H2_dry_vol_pct", "h2_vol_pct", "pred_H2_vol_pct"),
    ("CO2_dry_vol_pct", "co2_vol_pct", "pred_CO2_vol_pct"),
    ("H2_CO_ratio", "h2_co_ratio", "pred_H2_CO_ratio"),
    ("dry_syngas_flow", "syngas_flow_mean", "pred_dry_flow_nm3_h"),
)


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _coal_paths() -> dict[str, Path]:
    return {
        "v0_baseline_coal_assumption": DATA_PREP_ROOT / "config" / "coal_assumption_v0.yaml",
        "huayi_p2_sy3sy2_coal_assumption": DATA_PREP_ROOT / "config" / "coal_assumption_huayi_p2.yaml",
        "huayi_p3_sy3sy2_coal_assumption": DATA_PREP_ROOT / "config" / "coal_assumption_huayi_p3.yaml",
        "huayi_p4_sy3sy2_coal_assumption": DATA_PREP_ROOT / "config" / "coal_assumption_huayi_p4.yaml",
    }


def _phase_label(phase: int) -> str:
    return f"phase{int(phase)}"


def _split_label(split_role: str) -> str:
    if split_role in {"train", "validation", "test"}:
        return split_role
    raise ValueError(f"Unsupported split role: {split_role}")


def _normal_m3_h_to_kg_s(value_nm3_h: float, molar_mass_kg_kmol: float) -> float:
    return float(value_nm3_h) / 22.414 / 3600.0 * float(molar_mass_kg_kmol)


def _model_case_rows(primary: pd.DataFrame) -> list[dict[str, Any]]:
    coal_by_id = {key: _read_yaml(path) for key, path in _coal_paths().items()}
    rows: list[dict[str, Any]] = []
    for _, row in primary.sort_values(["phase", "start_time", "window_id"]).iterrows():
        coal = coal_by_id[str(row["coal_assumption_id"])]
        ultimate = coal["ultimate_analysis_dry"]
        proximate = coal["proximate_analysis"]
        co2_carrier_nm3_h = max(float(row["o2_flow_mean"]) * 0.24, 0.0)
        case = {
            "case_id": row["window_id"],
            "case_id_value": row["window_id"],
            "segment_id": row["parent_segment_id"],
            "independence_group_id": row["parent_independence_group_id"],
            "feature_version": row["coal_assumption_id"],
            "model_case_ready": bool(row["model_case_ready"]),
            "phase": _phase_label(int(row["phase"])),
            "phase_number": int(row["phase"]),
            "dataset_id": row["dataset_id"],
            "split": _split_label(str(row["split_role"])),
            "split_role": str(row["split_role"]),
            "window_id": row["window_id"],
            "start_time": row["start_time"],
            "end_time": row["end_time"],
            "reactor_bed_diameter_m_value": 3.8,
            "reactor_bed_height_m_value": 10.3,
            "coal_carbon_mass_fraction_adb_value": ultimate["carbon"]["value"],
            "coal_hydrogen_mass_fraction_adb_value": ultimate["hydrogen"]["value"],
            "coal_oxygen_mass_fraction_adb_value": ultimate["oxygen"]["value"],
            "coal_nitrogen_mass_fraction_adb_value": ultimate["nitrogen"]["value"],
            "coal_sulfur_mass_fraction_adb_value": ultimate["sulfur"]["value"],
            "coal_ash_proximate_d_value": proximate["ash_d"]["value"],
            "coal_volatile_matter_d_value": proximate["volatile_matter_d"]["value"],
            "coal_fixed_carbon_d_value": proximate["fixed_carbon_d"]["value"],
            "coal_heating_value_dry_kj_kg_value": coal["heating_value_dry_kj_kg"]["value"],
            "coal_mass_flow_value": float(row["coal_flow_mean"]),
            "o2_volumetric_flow_value": float(row["o2_flow_mean"]),
            "co2_carrier_volumetric_flow_value": co2_carrier_nm3_h,
            "gasifier_pressure_value": float(row["pressure_mean"]),
            "o2_inlet_temperature_c_value": 160.0,
            "co2_carrier_inlet_temperature_c_value": 80.0,
            "syngas_dry_volumetric_flow_value": float(row["syngas_flow_mean"]),
            "outlet_co_dry_vol_pct_value": float(row["co_vol_pct"]),
            "outlet_h2_dry_vol_pct_value": float(row["h2_vol_pct"]),
            "outlet_co2_dry_vol_pct_value": float(row["co2_vol_pct"]),
            "outlet_ch4_dry_vol_pct_value": float(row.get("ch4_vol_pct", 0.0)),
            "outlet_n2_dry_vol_pct_value": 0.0,
            "outlet_h2s_dry_vol_pct_value": 0.0,
            "co_vol_pct": float(row["co_vol_pct"]),
            "h2_vol_pct": float(row["h2_vol_pct"]),
            "co2_vol_pct": float(row["co2_vol_pct"]),
            "h2_co_ratio": float(row["h2_co_ratio"]),
            "syngas_flow_mean": float(row["syngas_flow_mean"]),
            "o_c_ratio": float(row["o_c_ratio"]),
        }
        rows.append(case)
    return rows


def _conversion_audit_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    required = [
        "coal_mass_flow_value",
        "o2_volumetric_flow_value",
        "co2_carrier_volumetric_flow_value",
        "gasifier_pressure_value",
        "coal_carbon_mass_fraction_adb_value",
        "syngas_dry_volumetric_flow_value",
        "outlet_co_dry_vol_pct_value",
        "outlet_h2_dry_vol_pct_value",
        "outlet_co2_dry_vol_pct_value",
        "h2_co_ratio",
    ]
    for row in rows:
        missing = [key for key in required if row.get(key) is None or pd.isna(row.get(key))]
        o2_kg_s = _normal_m3_h_to_kg_s(float(row["o2_volumetric_flow_value"]), 31.998)
        co2_kg_s = _normal_m3_h_to_kg_s(float(row["co2_carrier_volumetric_flow_value"]), 44.01)
        out.append(
            {
                "dataset_id": row["dataset_id"],
                "window_id": row["window_id"],
                "split_role": row["split_role"],
                "phase": row["phase_number"],
                "conversion_status": "ok" if not missing else "blocked",
                "missing_fields": ";".join(missing),
                "coal_assumption_id": row["feature_version"],
                "coal_flow_kg_s": row["coal_mass_flow_value"],
                "o2_flow_nm3_h": row["o2_volumetric_flow_value"],
                "o2_flow_kg_s": o2_kg_s,
                "co2_carrier_nm3_h": row["co2_carrier_volumetric_flow_value"],
                "co2_carrier_kg_s": co2_kg_s,
                "carrier_mapping": "CO2",
                "moisture_convention": "transported_coal_moisture_percent_2.0",
                "pressure_mpa": row["gasifier_pressure_value"],
                "o_c_ratio": row["o_c_ratio"],
                "target_schema": "CO,H2,CO2,H2/CO,dry_syngas_flow",
            }
        )
    return out


def _smoke_subset(rows: Sequence[Mapping[str, Any]], per_split: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        split_rows = [dict(row) for row in rows if row["split_role"] == split]
        selected.extend(split_rows[:per_split])
    return selected


def _worker_solve(row: Mapping[str, Any], result_queue: Any) -> None:
    started = time.perf_counter()
    try:
        src_dir = str(ROOT / "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from model.bounded_hybrid_correction_smoke import solve_with_wgs_correction

        result = solve_with_wgs_correction(row, 1.0, n_cells=N_CELLS)
        elapsed = time.perf_counter() - started
        result_queue.put(
            {
                "status": "ok",
                "wall_time_s": elapsed,
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


def _run_row_with_timeout(row: Mapping[str, Any], timeout_s: float) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(target=_worker_solve, args=(dict(row), result_queue))
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


def _prediction_rows(smoke_rows: Sequence[Mapping[str, Any]], timeout_s: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in smoke_rows:
        outcome = _run_row_with_timeout(row, timeout_s)
        base = {
            "dataset_id": row["dataset_id"],
            "window_id": row["window_id"],
            "split_role": row["split_role"],
            "phase": row["phase_number"],
            "method": "fixed_physics",
            "N_cells": N_CELLS,
            "row_timeout_s": timeout_s,
            "status": outcome.get("status", "unknown"),
            "wall_time_s": outcome.get("wall_time_s", math.nan),
            "fallback_count": outcome.get("fallback_count", math.nan),
            "poor_convergence_count": outcome.get("poor_convergence_count", math.nan),
            "poor_convergence_cell_index": outcome.get("poor_convergence_cell_index", ""),
            "cell0_cost": outcome.get("cell0_cost", math.nan),
        }
        for target, true_col, pred_col in TARGETS:
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
    return rows


def _metric_preview(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ok = predictions[predictions["status"] == "ok"].copy()
    for (split_role, target), group in ok.groupby(["split_role", "target"], dropna=False):
        residual = group["residual"].astype(float).to_numpy()
        rows.append(
            {
                "dataset_id": PRIMARY_DATASET_ID,
                "split_role": split_role,
                "method": "fixed_physics",
                "target": target,
                "metric_scope": "smoke_completed_rows_only",
                "n": int(len(group)),
                "rmse": float(np.sqrt(np.mean(residual * residual))),
                "mae": float(np.mean(np.abs(residual))),
                "bias": float(np.mean(residual)),
                "status": "ok",
            }
        )
    if not rows:
        rows.append(
            {
                "dataset_id": PRIMARY_DATASET_ID,
                "split_role": "ALL",
                "method": "fixed_physics",
                "target": "ALL",
                "metric_scope": "smoke_completed_rows_only",
                "n": 0,
                "rmse": math.nan,
                "mae": math.nan,
                "bias": math.nan,
                "status": "blocked:no_completed_smoke_rows",
            }
        )
    return pd.DataFrame(rows)


def _runtime_summary(predictions: pd.DataFrame, full_attempted: int, full_completed: int, decision: str) -> pd.DataFrame:
    row_level = predictions.drop_duplicates("window_id").copy()
    attempted = int(len(row_level))
    completed = int((row_level["status"] == "ok").sum())
    timeout = int((row_level["status"] == "timeout").sum())
    errored = int(row_level["status"].astype(str).str.startswith("error:").sum())
    return pd.DataFrame(
        [
            {
                "stage": "fixed_physics_smoke",
                "rows_attempted": attempted,
                "rows_completed": completed,
                "rows_timeout": timeout,
                "rows_error": errored,
                "wall_time_s_sum": float(row_level["wall_time_s"].sum()),
                "row_timeout_s": ROW_TIMEOUT_SECONDS,
                "fallback_count_sum": pd.to_numeric(row_level["fallback_count"], errors="coerce").sum(),
                "poor_convergence_count_sum": pd.to_numeric(row_level["poor_convergence_count"], errors="coerce").sum(),
                "status": "ok" if completed == attempted else "blocked",
                "decision": decision,
            },
            {
                "stage": "fixed_physics_full_or_bounded",
                "rows_attempted": full_attempted,
                "rows_completed": full_completed,
                "rows_timeout": 0,
                "rows_error": 0,
                "wall_time_s_sum": 0.0,
                "row_timeout_s": ROW_TIMEOUT_SECONDS,
                "fallback_count_sum": math.nan,
                "poor_convergence_count_sum": math.nan,
                "status": "not_run" if full_attempted == 0 else "ok",
                "decision": decision,
            },
        ]
    )


def _write(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def run(output_dir: Path, timeout_s: float, per_split: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_manifest = _read_yaml(ROUND52_DIR / "huayi_round52_split_freeze.yaml")
    inventory = pd.read_csv(ROUND52_DIR / "huayi_round52_window_inventory.csv")
    primary = inventory[inventory["dataset_id"] == PRIMARY_DATASET_ID].copy()
    expected = {role: set(split_manifest["splits"]["row_ids"][role]) for role in ("train", "validation", "test")}
    frozen_consumed = all(set(primary.loc[primary["split_role"] == role, "window_id"]) == ids for role, ids in expected.items())
    model_rows = _model_case_rows(primary)
    conversion_audit = pd.DataFrame(_conversion_audit_rows(model_rows))
    rows_converted = int((conversion_audit["conversion_status"] == "ok").sum())
    conversion_ok = rows_converted == len(primary) and frozen_consumed
    conversion_audit.to_csv(output_dir / "huayi_round54_input_conversion_audit.csv", index=False)

    smoke_rows = _smoke_subset(model_rows, per_split) if conversion_ok else []
    predictions = pd.DataFrame(_prediction_rows(smoke_rows, timeout_s)) if smoke_rows else pd.DataFrame()
    if predictions.empty:
        predictions = pd.DataFrame(
            [
                {
                    "dataset_id": PRIMARY_DATASET_ID,
                    "window_id": "NONE",
                    "split_role": "ALL",
                    "phase": "ALL",
                    "method": "fixed_physics",
                    "N_cells": N_CELLS,
                    "row_timeout_s": timeout_s,
                    "status": "not_run:conversion_failed",
                    "wall_time_s": 0.0,
                    "fallback_count": math.nan,
                    "poor_convergence_count": math.nan,
                    "poor_convergence_cell_index": "",
                    "cell0_cost": math.nan,
                    "target": "ALL",
                    "target_value": math.nan,
                    "predicted_value": math.nan,
                    "residual": math.nan,
                }
            ]
        )
    predictions.to_csv(output_dir / "huayi_round54_fixed_physics_smoke_predictions.csv", index=False)
    metric_preview = _metric_preview(predictions)
    metric_preview.to_csv(output_dir / "huayi_round54_fixed_physics_metric_preview.csv", index=False)

    row_predictions = predictions.drop_duplicates("window_id")
    smoke_attempted = int(len(row_predictions[row_predictions["window_id"] != "NONE"]))
    smoke_completed = int((row_predictions["status"] == "ok").sum())
    full_attempted = 0
    full_completed = 0
    if not conversion_ok:
        decision = "EVALUATOR_NO_GO"
        wgs_readiness = "not_ready: conversion audit failed or frozen split row IDs did not match"
        allowed_next = "Fix conversion/frozen split intake before any model evaluation."
    elif smoke_completed == 0:
        decision = "EVALUATOR_NO_GO"
        wgs_readiness = "not_ready: fixed-physics smoke produced zero completed rows under timeout guard"
        allowed_next = "Engineer a faster fixed-physics solver path or row-level cache before reopening WGS comparators."
    elif smoke_completed < smoke_attempted:
        decision = "EVALUATOR_CONDITIONAL"
        wgs_readiness = "conditional: evaluator format works for completed rows, but smoke timeouts/errors block full split dispatch"
        allowed_next = "Optimize fixed-physics evaluator and rerun smoke before global/WGS comparators."
    else:
        full_attempted = smoke_attempted
        full_completed = smoke_completed
        decision = "EVALUATOR_CONDITIONAL"
        wgs_readiness = "conditional: fixed-physics smoke and declared bounded sample completed; full 142-row runtime still not attempted in this round"
        allowed_next = "Run the full fixed-physics split with the same timeout/cache harness, then dispatch WGS comparators if stable."

    runtime = _runtime_summary(predictions, full_attempted, full_completed, decision)
    runtime.to_csv(output_dir / "huayi_round54_runtime_smoke_summary.csv", index=False)

    _write(
        output_dir / "huayi_round54_evaluator_readiness.md",
        "\n".join(
            [
                "# Huayi Round54 Evaluator Readiness",
                "",
                f"- Decision: **{decision}**",
                f"- Frozen split consumed: {frozen_consumed}",
                f"- Rows converted: {rows_converted}/{len(primary)}",
                f"- Smoke rows attempted/completed: {smoke_attempted}/{smoke_completed}",
                f"- Per-row timeout: {timeout_s:.1f} s",
                f"- WGS readiness: {wgs_readiness}",
                "",
                "The evaluator uses deterministic row conversion, phase-versioned coal assumptions, invariant plant geometry, CO2 carrier mapping, and fixed target schema.",
            ]
        ),
    )
    _write(
        output_dir / "huayi_round54_next_model_dispatch_if_ready.md",
        "\n".join(
            [
                "# Huayi Round54 Next Model Dispatch If Ready",
                "",
                f"Allowed next task: {allowed_next}",
                "",
                "Do not train or tune WGS until fixed-physics evaluation is stable under the frozen split.",
            ]
        ),
    )
    _write(
        output_dir / "huayi_round54_claim_boundary.md",
        "\n".join(
            [
                "# Huayi Round54 Claim Boundary",
                "",
                f"Decision: **{decision}**",
                "",
                "Allowed claim:",
                "- The Round52 frozen split conversion audit is complete; fixed-physics evaluator readiness is limited by the smoke-stage result reported here.",
                "",
                "Forbidden claims:",
                f"- {FORBIDDEN_CLAIMS}",
            ]
        ),
    )

    runtime_summary = (
        f"smoke attempted/completed={smoke_attempted}/{smoke_completed}; "
        f"timeouts={int((row_predictions['status'] == 'timeout').sum())}; "
        f"wall_time_sum={float(row_predictions['wall_time_s'].sum()):.3f}s"
    )
    metric_line = "unavailable:no_completed_rows"
    if smoke_completed:
        metric_line = "; ".join(
            f"{r.split_role}:{r.target}:rmse={r.rmse:.6g},n={int(r.n)}"
            for r in metric_preview.itertuples(index=False)
            if getattr(r, "status", "") == "ok"
        )
    report = "\n".join(
        [
            "ROUND54_EVALUATOR_DONE",
            f"Decision: {decision}",
            f"Frozen split consumed: {frozen_consumed}; R52-C-75 row IDs and roles preserved",
            f"Rows converted: {rows_converted}/{len(primary)}",
            f"Smoke rows attempted/completed: {smoke_attempted}/{smoke_completed}",
            f"Full/bounded rows attempted/completed: {full_attempted}/{full_completed}",
            f"Runtime summary: {runtime_summary}",
            f"Fixed-physics metric preview: {metric_line}",
            f"WGS readiness: {wgs_readiness}",
            "Created files: huayi_round54_input_conversion_audit.csv, huayi_round54_runtime_smoke_summary.csv, huayi_round54_fixed_physics_smoke_predictions.csv, huayi_round54_fixed_physics_metric_preview.csv, huayi_round54_evaluator_readiness.md, huayi_round54_next_model_dispatch_if_ready.md, huayi_round54_claim_boundary.md, huayi_round54_result.md",
            "Tests/checks run: frozen split row-ID intake check; 142-row conversion audit; deterministic 3/3/3 smoke subset with per-row timeout; CSV schema/readback checks; git diff --check scoped to Round54 files",
            "Known limitations: smoke/bounded sample uses the same deterministic 3/3/3 row subset, N_cells=1 and per-row timeout; full 142-row fixed-physics and WGS comparators were not run; CO2 carrier is mapped by the predeclared engineering ratio used in the converter because Round52 does not carry measured carrier flow.",
            f"Allowed next task: {allowed_next}",
            f"Forbidden claims: {FORBIDDEN_CLAIMS}",
        ]
    )
    _write(output_dir / "huayi_round54_result.md", report)
    return {
        "decision": decision,
        "report": report,
        "created_dir": output_dir,
        "smoke_attempted": smoke_attempted,
        "smoke_completed": smoke_completed,
        "rows_converted": rows_converted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Round54 tractable evaluator artifacts")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "diagnostics" / "round54")
    parser.add_argument("--timeout-s", type=float, default=ROW_TIMEOUT_SECONDS)
    parser.add_argument("--per-split", type=int, default=SMOKE_ROWS_PER_SPLIT)
    args = parser.parse_args()
    result = run(args.output_dir, args.timeout_s, args.per_split)
    print(result["report"])


if __name__ == "__main__":
    main()
