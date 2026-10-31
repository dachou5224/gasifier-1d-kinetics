"""
Chronological split and global WGS alpha baseline scaffolding.

This module implements the first post-protocol baseline layer:

* Huayi 22-case chronological diagnostic split manifest.
* Split-aware fixed-physics metrics from existing diagnostics.
* A non-learned global bounded WGS scalar selected on train only.

It does not implement alpha_j(x_t), z_d models, MLP training, or production
control logic.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .bounded_hybrid_correction_smoke import (
    BOUNDED_CORRECTION_EVIDENCE_BOUNDARY,
    CORRECTION_BOUND_HIGH,
    CORRECTION_BOUND_LOW,
    solve_with_wgs_correction,
)
from .model_case_adapter import build_gasifier_inputs_from_model_case

SPLIT_EVIDENCE_BOUNDARY = (
    "Huayi 22-case chronological diagnostic split only; not random split, "
    "trained hybrid accuracy, final temporal generalization, production control, "
    "weak-label proof, or Yulin/slurry generalization"
)

GLOBAL_WGS_EVIDENCE_BOUNDARY = (
    "non-learned global bounded WGS scalar comparator selected on train split "
    "only; not alpha_j(x_t), trained hybrid model, learned multiplier "
    "interpretability, production control, final temporal generalization, or "
    "Yulin/slurry generalization"
)

SPLIT_MANIFEST_COLUMNS: tuple[str, ...] = (
    "case_id",
    "phase",
    "dataset_id",
    "split",
    "split_role",
    "chronological_rank",
    "is_future_holdout",
    "independence_group",
    "source_manifest",
    "evidence_boundary",
)

PREDICTION_COLUMNS: tuple[str, ...] = (
    "model_name",
    "case_id",
    "phase",
    "dataset_id",
    "split",
    "split_role",
    "N_cells",
    "target",
    "y_true",
    "y_pred",
    "residual",
    "fallback_count",
    "poor_convergence_count",
    "poor_convergence_cell_index",
    "cell0_cost",
    "cell0_max_abs_residual",
    "status",
    "evidence_boundary",
)

METRIC_COLUMNS: tuple[str, ...] = (
    "model_name",
    "split",
    "target",
    "n",
    "rmse",
    "mae",
    "bias",
    "N_cells",
    "fallback_count_sum",
    "poor_convergence_count_sum",
    "cell0_cost_min",
    "cell0_cost_max",
    "rmse_delta",
    "mae_delta",
    "bias_delta",
    "evidence_boundary",
)

GLOBAL_WGS_SUMMARY_COLUMNS: tuple[str, ...] = (
    "alpha_name",
    "selected_alpha",
    "bound_low",
    "bound_high",
    "hit_lower_bound",
    "hit_upper_bound",
    "training_objective_initial",
    "training_objective_final",
    "training_objective_delta",
    "selection_split",
    "N_cells",
    "steps_or_grid_points",
    "selection_method",
    "status",
    "validation_excluded_from_selection",
    "test_excluded_from_selection",
    "evidence_boundary",
)

GLOBAL_WGS_GRID_COLUMNS: tuple[str, ...] = (
    "alpha_name",
    "alpha_value",
    "selection_split",
    "N_cells",
    "train_case_count",
    "training_objective",
    "status",
    "evidence_boundary",
)

DEFAULT_WGS_ALPHA_GRID: tuple[float, ...] = (
    0.02,
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
    0.40,
    0.50,
    0.75,
    1.00,
    1.25,
    1.50,
    2.00,
)

TARGET_MAP: Mapping[str, tuple[str, str]] = {
    "CO_volpct_dry": ("target_CO_vol_pct", "pred_CO_vol_pct"),
    "H2_volpct_dry": ("target_H2_vol_pct", "pred_H2_vol_pct"),
    "CO2_volpct_dry": ("target_CO2_vol_pct", "pred_CO2_vol_pct"),
    "H2_CO_ratio": ("target_H2_CO_ratio", "pred_H2_CO_ratio"),
}


def split_for_phase(phase: str) -> tuple[str, str, bool]:
    if phase in {"phase1", "phase2"}:
        return "train", "fit_alpha_and_fit_preprocessing_only", False
    if phase == "phase3":
        return "validation", "select_protocol_not_alpha", True
    if phase == "phase4":
        return "test", "final_future_holdout_diagnostic", True
    raise ValueError(f"Unsupported phase for Huayi split: {phase}")


def build_split_manifest(contract_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, row in enumerate(contract_rows, start=1):
        split, role, is_future = split_for_phase(str(row["phase"]))
        rows.append(
            {
                "case_id": str(row["case_id"]),
                "phase": str(row["phase"]),
                "dataset_id": str(row["dataset_id"]),
                "split": split,
                "split_role": role,
                "chronological_rank": rank,
                "is_future_holdout": bool(is_future),
                "independence_group": row.get("independence_group"),
                "source_manifest": "outputs/diagnostics/huayi_expanded_model_case_contract_audit.csv",
                "evidence_boundary": SPLIT_EVIDENCE_BOUNDARY,
            }
        )
    return [{column: row.get(column) for column in SPLIT_MANIFEST_COLUMNS} for row in rows]


def validate_split_manifest(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    case_ids = [str(row["case_id"]) for row in rows]
    split_counts = pd.Series([row["split"] for row in rows]).value_counts().to_dict() if rows else {}
    chronological = [int(row["chronological_rank"]) for row in rows]
    return {
        "row_count": len(rows),
        "split_counts": {str(k): int(v) for k, v in split_counts.items()},
        "has_duplicate_case_id": len(case_ids) != len(set(case_ids)),
        "chronological_rank_is_monotonic": chronological == sorted(chronological),
        "expected_counts_match": split_counts.get("train", 0) == 12
        and split_counts.get("validation", 0) == 6
        and split_counts.get("test", 0) == 4,
        "no_random_shuffle_claim": True,
        "evidence_boundary": SPLIT_EVIDENCE_BOUNDARY,
    }


def _split_index(split_rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row["case_id"]): row for row in split_rows}


def fixed_physics_prediction_rows(
    baseline_rows: Sequence[Mapping[str, Any]],
    split_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    split_by_case = _split_index(split_rows)
    out: list[dict[str, Any]] = []
    for row in baseline_rows:
        split = split_by_case[str(row["case_id"])]
        for target, (true_col, pred_col) in TARGET_MAP.items():
            y_true = float(row[true_col])
            y_pred = float(row[pred_col])
            out.append(
                {
                    "model_name": "fixed_physics",
                    "case_id": row["case_id"],
                    "phase": split["phase"],
                    "dataset_id": split["dataset_id"],
                    "split": split["split"],
                    "split_role": split["split_role"],
                    "N_cells": int(row["N_cells"]),
                    "target": target,
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "residual": y_pred - y_true,
                    "fallback_count": int(row["fallback_count"]),
                    "poor_convergence_count": int(row["poor_convergence_count"]),
                    "poor_convergence_cell_index": row.get("poor_convergence_cell_index", ""),
                    "cell0_cost": float(row["cell0_cost"]),
                    "cell0_max_abs_residual": float(row["cell0_max_abs_residual"]),
                    "status": "ok",
                    "evidence_boundary": SPLIT_EVIDENCE_BOUNDARY,
                }
            )
    return [{column: row.get(column) for column in PREDICTION_COLUMNS} for row in out]


def compute_metric_rows(
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    evidence_boundary: str,
    fixed_metric_rows: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    df = pd.DataFrame(prediction_rows)
    if df.empty:
        return []
    fixed_lookup: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    if fixed_metric_rows:
        fixed_lookup = {
            (str(row["split"]), str(row["target"]), int(row["N_cells"])): row
            for row in fixed_metric_rows
            if row.get("model_name") == "fixed_physics"
        }

    rows: list[dict[str, Any]] = []
    group_cols = ["model_name", "split", "target", "N_cells"]
    for (model_name, split, target, n_cells), group in df.groupby(group_cols, dropna=False):
        residual = group["residual"].astype(float).to_numpy()
        rmse = float(np.sqrt(np.mean(residual * residual)))
        mae = float(np.mean(np.abs(residual)))
        bias = float(np.mean(residual))
        fixed = fixed_lookup.get((str(split), str(target), int(n_cells)))
        rows.append(
            {
                "model_name": model_name,
                "split": split,
                "target": target,
                "n": int(len(group)),
                "rmse": rmse,
                "mae": mae,
                "bias": bias,
                "N_cells": int(n_cells),
                "fallback_count_sum": int(group["fallback_count"].sum()),
                "poor_convergence_count_sum": int(group["poor_convergence_count"].sum()),
                "cell0_cost_min": float(group["cell0_cost"].min()),
                "cell0_cost_max": float(group["cell0_cost"].max()),
                "rmse_delta": rmse - float(fixed["rmse"]) if fixed else math.nan,
                "mae_delta": mae - float(fixed["mae"]) if fixed else math.nan,
                "bias_delta": bias - float(fixed["bias"]) if fixed else math.nan,
                "evidence_boundary": evidence_boundary,
            }
        )
    rows.sort(key=lambda row: (int(row["N_cells"]), str(row["split"]), str(row["target"]), str(row["model_name"])))
    return [{column: row.get(column) for column in METRIC_COLUMNS} for row in rows]


def _model_case_id(row: Mapping[str, Any]) -> str:
    return str(build_gasifier_inputs_from_model_case(row)["case_id"])


def attach_split_to_model_cases(
    model_case_rows: Sequence[Mapping[str, Any]],
    split_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    split_by_case = _split_index(split_rows)
    out: list[dict[str, Any]] = []
    for row in model_case_rows:
        item = dict(row)
        case_id = _model_case_id(item)
        split = split_by_case[case_id]
        item["_model_case_id"] = case_id
        item["_split"] = split["split"]
        item["_split_role"] = split["split_role"]
        out.append(item)
    return out


def wgs_objective_from_result(result: Mapping[str, Any]) -> float:
    target_h2co = float(result["target_H2_volpct_dry"]) / max(float(result["target_CO_volpct_dry"]), 1.0e-12)
    return float(
        ((float(result["CO_volpct_dry"]) - float(result["target_CO_volpct_dry"])) / 10.0) ** 2
        + ((float(result["H2_volpct_dry"]) - float(result["target_H2_volpct_dry"])) / 10.0) ** 2
        + ((float(result["CO2_volpct_dry"]) - float(result["target_CO2_volpct_dry"])) / 10.0) ** 2
        + ((float(result["H2_CO_ratio"]) - target_h2co) / 0.1) ** 2
    )


def select_global_wgs_alpha(
    model_case_rows_with_split: Sequence[Mapping[str, Any]],
    *,
    alpha_grid: Sequence[float] = DEFAULT_WGS_ALPHA_GRID,
    n_cells: int = 3,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train_rows = [row for row in model_case_rows_with_split if row["_split"] == "train"]
    grid_rows: list[dict[str, Any]] = []
    for alpha in alpha_grid:
        objectives: list[float] = []
        status = "ok"
        for row in train_rows:
            try:
                result = solve_with_wgs_correction(row, float(alpha), n_cells=n_cells)
                objectives.append(wgs_objective_from_result(result))
            except Exception as exc:  # pragma: no cover - diagnostic failure path
                status = f"failed:{type(exc).__name__}:{exc}"
                objectives.append(math.nan)
        objective = float(np.nanmean(objectives)) if objectives else math.nan
        grid_rows.append(
            {
                "alpha_name": "alpha_wgs",
                "alpha_value": float(alpha),
                "selection_split": "train",
                "N_cells": int(n_cells),
                "train_case_count": len(train_rows),
                "training_objective": objective,
                "status": status,
                "evidence_boundary": GLOBAL_WGS_EVIDENCE_BOUNDARY,
            }
        )

    ok_grid = [row for row in grid_rows if row["status"] == "ok" and math.isfinite(float(row["training_objective"]))]
    if not ok_grid:
        raise RuntimeError("No valid WGS alpha grid rows were generated")
    selected = min(ok_grid, key=lambda row: (float(row["training_objective"]), abs(float(row["alpha_value"]) - 1.0)))
    return selected, [{column: row.get(column) for column in GLOBAL_WGS_GRID_COLUMNS} for row in grid_rows]


def wgs_prediction_rows(
    model_case_rows_with_split: Sequence[Mapping[str, Any]],
    *,
    selected_alpha: float,
    n_cells: int = 3,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in model_case_rows_with_split:
        result = solve_with_wgs_correction(row, float(selected_alpha), n_cells=n_cells)
        target_h2co = float(result["target_H2_volpct_dry"]) / max(float(result["target_CO_volpct_dry"]), 1.0e-12)
        values = {
            "CO_volpct_dry": (result["target_CO_volpct_dry"], result["CO_volpct_dry"]),
            "H2_volpct_dry": (result["target_H2_volpct_dry"], result["H2_volpct_dry"]),
            "CO2_volpct_dry": (result["target_CO2_volpct_dry"], result["CO2_volpct_dry"]),
            "H2_CO_ratio": (target_h2co, result["H2_CO_ratio"]),
        }
        for target, (y_true, y_pred) in values.items():
            rows.append(
                {
                    "model_name": "global_wgs_alpha",
                    "case_id": result["case_id"],
                    "phase": result["phase"],
                    "dataset_id": result["dataset_id"],
                    "split": row["_split"],
                    "split_role": row["_split_role"],
                    "N_cells": int(n_cells),
                    "target": target,
                    "y_true": float(y_true),
                    "y_pred": float(y_pred),
                    "residual": float(y_pred) - float(y_true),
                    "fallback_count": int(result["fallback_count"]),
                    "poor_convergence_count": int(result["poor_convergence_count"]),
                    "poor_convergence_cell_index": result.get("poor_convergence_cell_index", ""),
                    "cell0_cost": float(result["cell0_cost"]),
                    "cell0_max_abs_residual": math.nan,
                    "status": result.get("solve_status", "ok"),
                    "evidence_boundary": GLOBAL_WGS_EVIDENCE_BOUNDARY,
                }
            )
    return [{column: row.get(column) for column in PREDICTION_COLUMNS} for row in rows]


def build_wgs_summary(
    selected_grid_row: Mapping[str, Any],
    grid_rows: Sequence[Mapping[str, Any]],
    *,
    n_cells: int,
) -> list[dict[str, Any]]:
    initial = next(row for row in grid_rows if math.isclose(float(row["alpha_value"]), 1.0, rel_tol=0.0, abs_tol=1.0e-12))
    selected_alpha = float(selected_grid_row["alpha_value"])
    initial_obj = float(initial["training_objective"])
    final_obj = float(selected_grid_row["training_objective"])
    row = {
        "alpha_name": "alpha_wgs",
        "selected_alpha": selected_alpha,
        "bound_low": CORRECTION_BOUND_LOW,
        "bound_high": CORRECTION_BOUND_HIGH,
        "hit_lower_bound": math.isclose(selected_alpha, CORRECTION_BOUND_LOW, rel_tol=0.0, abs_tol=1.0e-12),
        "hit_upper_bound": math.isclose(selected_alpha, CORRECTION_BOUND_HIGH, rel_tol=0.0, abs_tol=1.0e-12),
        "training_objective_initial": initial_obj,
        "training_objective_final": final_obj,
        "training_objective_delta": final_obj - initial_obj,
        "selection_split": "train",
        "N_cells": int(n_cells),
        "steps_or_grid_points": len(grid_rows),
        "selection_method": "declared_grid_search_train_split_only",
        "status": "ok",
        "validation_excluded_from_selection": True,
        "test_excluded_from_selection": True,
        "evidence_boundary": GLOBAL_WGS_EVIDENCE_BOUNDARY,
    }
    return [{column: row.get(column) for column in GLOBAL_WGS_SUMMARY_COLUMNS}]


def split_manifest_markdown(rows: Sequence[Mapping[str, Any]], validation: Mapping[str, Any]) -> str:
    counts = validation["split_counts"]
    return (
        "# Huayi hybrid chronological split manifest\n\n"
        f"- Rows: {validation['row_count']}\n"
        f"- Split counts: {counts}\n"
        f"- Duplicate case IDs: {validation['has_duplicate_case_id']}\n"
        f"- Chronological rank monotonic: {validation['chronological_rank_is_monotonic']}\n"
        f"- Expected 12/6/4 counts match: {validation['expected_counts_match']}\n"
        "- Split rule: phase1+phase2 train, phase3 validation, phase4 test.\n\n"
        f"Evidence boundary: {SPLIT_EVIDENCE_BOUNDARY}.\n"
    )


def metrics_markdown(title: str, metric_rows: Sequence[Mapping[str, Any]], evidence_boundary: str) -> str:
    lines = [
        f"# {title}",
        "",
        "| model | split | N | target | n | RMSE | MAE | bias | fallback | poor |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(
            "| {model} | {split} | {n_cells} | {target} | {n} | {rmse:.6g} | {mae:.6g} | {bias:.6g} | {fallback} | {poor} |".format(
                model=row["model_name"],
                split=row["split"],
                n_cells=int(row["N_cells"]),
                target=row["target"],
                n=int(row["n"]),
                rmse=float(row["rmse"]),
                mae=float(row["mae"]),
                bias=float(row["bias"]),
                fallback=int(row["fallback_count_sum"]),
                poor=int(row["poor_convergence_count_sum"]),
            )
        )
    lines.extend(["", f"Evidence boundary: {evidence_boundary}."])
    return "\n".join(lines) + "\n"


def wgs_baseline_markdown(
    summary_rows: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
    grid_rows: Sequence[Mapping[str, Any]],
) -> str:
    summary = summary_rows[0]
    lines = [
        "# Huayi global WGS alpha baseline",
        "",
        "This is a non-learned global scalar comparator. Alpha is selected using train split only.",
        "",
        f"- Selected alpha: {float(summary['selected_alpha']):.6g}",
        f"- Bounds: [{summary['bound_low']}, {summary['bound_high']}]",
        f"- Hit lower bound: {summary['hit_lower_bound']}",
        f"- Hit upper bound: {summary['hit_upper_bound']}",
        f"- Training objective initial alpha=1.0: {float(summary['training_objective_initial']):.6g}",
        f"- Training objective selected: {float(summary['training_objective_final']):.6g}",
        f"- Grid points: {summary['steps_or_grid_points']}",
        f"- Validation excluded from selection: {summary['validation_excluded_from_selection']}",
        f"- Test excluded from selection: {summary['test_excluded_from_selection']}",
        "",
        "## Grid",
        "",
        "| alpha | train_objective | status |",
        "|---:|---:|---|",
    ]
    for row in grid_rows:
        lines.append(f"| {float(row['alpha_value']):.6g} | {float(row['training_objective']):.6g} | {row['status']} |")
    lines.extend(["", "## Metrics", ""])
    lines.append(metrics_markdown("Global WGS alpha split metrics", metric_rows, GLOBAL_WGS_EVIDENCE_BOUNDARY))
    lines.append(f"Evidence boundary: {GLOBAL_WGS_EVIDENCE_BOUNDARY}.")
    return "\n".join(lines) + "\n"


__all__ = [
    "DEFAULT_WGS_ALPHA_GRID",
    "GLOBAL_WGS_GRID_COLUMNS",
    "GLOBAL_WGS_SUMMARY_COLUMNS",
    "METRIC_COLUMNS",
    "PREDICTION_COLUMNS",
    "SPLIT_MANIFEST_COLUMNS",
    "attach_split_to_model_cases",
    "build_split_manifest",
    "build_wgs_summary",
    "compute_metric_rows",
    "fixed_physics_prediction_rows",
    "metrics_markdown",
    "select_global_wgs_alpha",
    "split_manifest_markdown",
    "validate_split_manifest",
    "wgs_baseline_markdown",
    "wgs_objective_from_result",
    "wgs_prediction_rows",
]
