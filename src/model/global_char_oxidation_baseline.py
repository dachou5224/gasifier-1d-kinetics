"""
Global char-oxidation scalar baseline under the Huayi chronological split.

This module evaluates a single non-learned scalar through the existing
``op_conds.CharCombustionRateFactor`` path.  It is a comparator only, not an
alpha_j(x_t) model or trained hybrid.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .gasifier_system import GasifierSystem
from .model_case_adapter import build_gasifier_inputs_from_model_case
from .model_case_diagnostic_baseline import POOR_CONVERGENCE_COST_THRESHOLD, cell_residual_quality
from .split_global_alpha_baseline import PREDICTION_COLUMNS

CHAR_ALPHA_NAME = "alpha_char_oxidation"
CHAR_ALPHA_PARAMETER = "op_conds.CharCombustionRateFactor"
CHAR_ALPHA_BOUND_LOW = 0.05
CHAR_ALPHA_BOUND_HIGH = 1.0
DEFAULT_CHAR_ALPHA_GRID: tuple[float, ...] = (0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0)

GLOBAL_CHAR_EVIDENCE_BOUNDARY = (
    "non-learned global char-oxidation scalar comparator selected on train "
    "split only through op_conds.CharCombustionRateFactor; not trained hybrid, "
    "alpha_j(x_t), learned multiplier interpretability, weak-label proof, "
    "production control, final temporal generalization, or Yulin/slurry "
    "generalization"
)

GLOBAL_CHAR_SUMMARY_COLUMNS: tuple[str, ...] = (
    "alpha_name",
    "mapped_parameter",
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
    "parameter_path_clean",
    "jax_parity_caveat",
    "evidence_boundary",
)

GLOBAL_CHAR_GRID_COLUMNS: tuple[str, ...] = (
    "alpha_name",
    "alpha_value",
    "selection_split",
    "N_cells",
    "train_case_count",
    "training_objective",
    "fallback_count_sum",
    "poor_convergence_count_sum",
    "cell0_cost_min",
    "cell0_cost_max",
    "status",
    "evidence_boundary",
)


SolverFn = Callable[[Mapping[str, Any], float, int], Mapping[str, Any]]


def clamp_char_alpha(value: float) -> float:
    return min(max(float(value), CHAR_ALPHA_BOUND_LOW), CHAR_ALPHA_BOUND_HIGH)


def validate_char_alpha_grid(alpha_grid: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in alpha_grid)
    if not values:
        raise ValueError("alpha grid must not be empty")
    out_of_bounds = [value for value in values if value < CHAR_ALPHA_BOUND_LOW or value > CHAR_ALPHA_BOUND_HIGH]
    if out_of_bounds:
        raise ValueError(f"alpha grid values outside char-oxidation bounds: {out_of_bounds}")
    if 1.0 not in values:
        raise ValueError("alpha grid must include 1.0 for initial objective reporting")
    return values


def inject_char_oxidation_factor(op_conds: Mapping[str, Any], alpha: float) -> dict[str, Any]:
    """Return a copied op_conds dict with only CharCombustionRateFactor changed."""
    bounded = clamp_char_alpha(alpha)
    out = dict(op_conds)
    out["CharCombustionRateFactor"] = bounded
    return out


def char_oxidation_parameter_path_audit() -> dict[str, Any]:
    return {
        "mapped_parameter": CHAR_ALPHA_PARAMETER,
        "mainline_cell_path": "Cell._calc_rates -> KineticsService.calc_heterogeneous_rates(char_combustion_factor=...)",
        "system_jax_contract_path": "GasifierSystem passes CharCombustionRateFactor to reactor_solve_v4 hot path",
        "per_run_injection_without_public_api_change": True,
        "bounded_direct_factor": [CHAR_ALPHA_BOUND_LOW, CHAR_ALPHA_BOUND_HIGH],
        "does_not_mutate_adapter_op_conds": True,
        "jax_parity_caveat": "round43 selection/evaluation uses existing NumPy/minimize diagnostic path; JAX parity is parameter-wired but not rerun as a char-alpha training baseline here",
        "status": "clean_for_nonlearned_global_scalar_comparator",
        "evidence_boundary": GLOBAL_CHAR_EVIDENCE_BOUNDARY,
    }


def _dry_syngas_kpis(profile: np.ndarray) -> dict[str, float]:
    last = np.asarray(profile[-1], dtype=float)
    gas = last[:8]
    dry = float(np.sum(gas[:7]))
    denom = dry + 1.0e-12
    co = float(gas[2] / denom * 100.0)
    h2 = float(gas[5] / denom * 100.0)
    return {
        "CO_volpct_dry": co,
        "H2_volpct_dry": h2,
        "CO2_volpct_dry": float(gas[3] / denom * 100.0),
        "H2_CO_ratio": h2 / max(co, 1.0e-12),
    }


def _case_meta(model_case_row: Mapping[str, Any], built: Mapping[str, Any], n_cells: int) -> dict[str, Any]:
    return {
        "case_id": built["case_id"],
        "phase": model_case_row.get("phase"),
        "dataset_id": model_case_row.get("dataset_id"),
        "split": model_case_row.get("_split", model_case_row.get("split")),
        "split_role": model_case_row.get("_split_role", model_case_row.get("split_role")),
        "N_cells": int(n_cells),
    }


def char_objective_from_result(result: Mapping[str, Any]) -> float:
    target_h2co = float(result["target_H2_volpct_dry"]) / max(float(result["target_CO_volpct_dry"]), 1.0e-12)
    return float(
        ((float(result["CO_volpct_dry"]) - float(result["target_CO_volpct_dry"])) / 10.0) ** 2
        + ((float(result["H2_volpct_dry"]) - float(result["target_H2_volpct_dry"])) / 10.0) ** 2
        + ((float(result["CO2_volpct_dry"]) - float(result["target_CO2_volpct_dry"])) / 10.0) ** 2
        + ((float(result["H2_CO_ratio"]) - target_h2co) / 0.1) ** 2
    )


def solve_with_char_oxidation_alpha(
    model_case_row: Mapping[str, Any],
    alpha: float,
    n_cells: int = 3,
) -> dict[str, Any]:
    built = build_gasifier_inputs_from_model_case(model_case_row)
    op_conds = inject_char_oxidation_factor(built["op_conds"], alpha)
    start = time.perf_counter()
    system = GasifierSystem(built["geometry"], built["coal_props"], op_conds)
    profile, _z = system.solve(N_cells=int(n_cells), solver_method="minimize", jacobian_mode="scipy", jax_warmup=False)
    runtime_s = time.perf_counter() - start
    kpis = _dry_syngas_kpis(profile)
    residual_quality = cell_residual_quality(system, profile)
    poor_cells = [item["cell_index"] for item in residual_quality if item["cost"] > POOR_CONVERGENCE_COST_THRESHOLD]
    cell0 = residual_quality[0] if residual_quality else {}
    targets = built["targets"]
    out = {
        **_case_meta(model_case_row, built, n_cells),
        "alpha_char_oxidation": clamp_char_alpha(alpha),
        "runtime_s": runtime_s,
        "fallback_count": int(system.solve_stats.get("fallback_count", 0)),
        "poor_convergence_count": int(system.solve_stats.get("poor_convergence_count", 0)),
        "poor_convergence_cell_index": ";".join(str(i) for i in poor_cells),
        "cell0_cost": float(cell0.get("cost", math.nan)),
        "cell0_max_abs_residual": float(cell0.get("max_abs_residual", math.nan)),
        "solve_status": "ok",
        "target_CO_volpct_dry": float(targets["outlet_co_dry_vol_pct"]),
        "target_H2_volpct_dry": float(targets["outlet_h2_dry_vol_pct"]),
        "target_CO2_volpct_dry": float(targets["outlet_co2_dry_vol_pct"]),
    }
    out["target_H2_CO_ratio"] = out["target_H2_volpct_dry"] / max(out["target_CO_volpct_dry"], 1.0e-12)
    out.update(kpis)
    out["objective"] = char_objective_from_result(out)
    return out


def _safe_solve(solver_fn: SolverFn, row: Mapping[str, Any], alpha: float, n_cells: int) -> dict[str, Any]:
    try:
        return dict(solver_fn(row, float(alpha), int(n_cells)))
    except Exception as exc:  # pragma: no cover - diagnostic failure path
        return {
            "case_id": row.get("_model_case_id", row.get("case_id", "unknown")),
            "phase": row.get("phase"),
            "dataset_id": row.get("dataset_id"),
            "split": row.get("_split", row.get("split")),
            "split_role": row.get("_split_role", row.get("split_role")),
            "N_cells": int(n_cells),
            "objective": math.nan,
            "fallback_count": math.nan,
            "poor_convergence_count": math.nan,
            "cell0_cost": math.nan,
            "solve_status": f"failed:{type(exc).__name__}:{exc}",
        }


def select_global_char_oxidation_alpha(
    model_case_rows_with_split: Sequence[Mapping[str, Any]],
    *,
    alpha_grid: Sequence[float] = DEFAULT_CHAR_ALPHA_GRID,
    n_cells: int = 3,
    solver_fn: SolverFn = solve_with_char_oxidation_alpha,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grid = validate_char_alpha_grid(alpha_grid)
    train_rows = [row for row in model_case_rows_with_split if row.get("_split", row.get("split")) == "train"]
    if not train_rows:
        raise ValueError("No train rows available for char-oxidation alpha selection")
    grid_rows: list[dict[str, Any]] = []
    for alpha in grid:
        results = [_safe_solve(solver_fn, row, alpha, n_cells) for row in train_rows]
        objectives = [float(row["objective"]) for row in results if math.isfinite(float(row.get("objective", math.nan)))]
        status = "ok" if len(objectives) == len(train_rows) else "partial_or_failed"
        cell0_values = [float(row.get("cell0_cost", math.nan)) for row in results if math.isfinite(float(row.get("cell0_cost", math.nan)))]
        grid_rows.append(
            {
                "alpha_name": CHAR_ALPHA_NAME,
                "alpha_value": float(alpha),
                "selection_split": "train",
                "N_cells": int(n_cells),
                "train_case_count": len(train_rows),
                "training_objective": float(np.mean(objectives)) if objectives else math.nan,
                "fallback_count_sum": int(sum(float(row.get("fallback_count", 0.0)) for row in results if math.isfinite(float(row.get("fallback_count", math.nan))))),
                "poor_convergence_count_sum": int(sum(float(row.get("poor_convergence_count", 0.0)) for row in results if math.isfinite(float(row.get("poor_convergence_count", math.nan))))),
                "cell0_cost_min": float(min(cell0_values)) if cell0_values else math.nan,
                "cell0_cost_max": float(max(cell0_values)) if cell0_values else math.nan,
                "status": status,
                "evidence_boundary": GLOBAL_CHAR_EVIDENCE_BOUNDARY,
            }
        )
    ok_grid = [row for row in grid_rows if row["status"] == "ok" and math.isfinite(float(row["training_objective"]))]
    if not ok_grid:
        raise RuntimeError("No valid char-oxidation grid rows were generated")
    selected = min(ok_grid, key=lambda row: (float(row["training_objective"]), abs(float(row["alpha_value"]) - 1.0)))
    return selected, [{column: row.get(column) for column in GLOBAL_CHAR_GRID_COLUMNS} for row in grid_rows]


def char_oxidation_prediction_rows(
    model_case_rows_with_split: Sequence[Mapping[str, Any]],
    *,
    selected_alpha: float,
    n_cells: int = 3,
    solver_fn: SolverFn = solve_with_char_oxidation_alpha,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in model_case_rows_with_split:
        result = _safe_solve(solver_fn, row, selected_alpha, n_cells)
        values = {
            "CO_volpct_dry": (result.get("target_CO_volpct_dry"), result.get("CO_volpct_dry")),
            "H2_volpct_dry": (result.get("target_H2_volpct_dry"), result.get("H2_volpct_dry")),
            "CO2_volpct_dry": (result.get("target_CO2_volpct_dry"), result.get("CO2_volpct_dry")),
            "H2_CO_ratio": (result.get("target_H2_CO_ratio"), result.get("H2_CO_ratio")),
        }
        for target, (y_true, y_pred) in values.items():
            y_true_f = float(y_true) if y_true is not None else math.nan
            y_pred_f = float(y_pred) if y_pred is not None else math.nan
            rows.append(
                {
                    "model_name": "global_char_oxidation_alpha",
                    "case_id": result.get("case_id"),
                    "phase": result.get("phase"),
                    "dataset_id": result.get("dataset_id"),
                    "split": result.get("split"),
                    "split_role": result.get("split_role"),
                    "N_cells": int(n_cells),
                    "target": target,
                    "y_true": y_true_f,
                    "y_pred": y_pred_f,
                    "residual": y_pred_f - y_true_f,
                    "fallback_count": int(result.get("fallback_count", 0) or 0),
                    "poor_convergence_count": int(result.get("poor_convergence_count", 0) or 0),
                    "poor_convergence_cell_index": result.get("poor_convergence_cell_index", ""),
                    "cell0_cost": float(result.get("cell0_cost", math.nan)),
                    "cell0_max_abs_residual": float(result.get("cell0_max_abs_residual", math.nan)),
                    "status": result.get("solve_status", "ok"),
                    "evidence_boundary": GLOBAL_CHAR_EVIDENCE_BOUNDARY,
                }
            )
    rows.sort(key=lambda item: (str(item["split"]), str(item["case_id"]), str(item["target"])))
    return [{column: row.get(column) for column in PREDICTION_COLUMNS} for row in rows]


def build_char_summary(
    selected_grid_row: Mapping[str, Any],
    grid_rows: Sequence[Mapping[str, Any]],
    *,
    n_cells: int,
) -> list[dict[str, Any]]:
    initial = next(row for row in grid_rows if math.isclose(float(row["alpha_value"]), 1.0, rel_tol=0.0, abs_tol=1.0e-12))
    selected_alpha = float(selected_grid_row["alpha_value"])
    audit = char_oxidation_parameter_path_audit()
    row = {
        "alpha_name": CHAR_ALPHA_NAME,
        "mapped_parameter": CHAR_ALPHA_PARAMETER,
        "selected_alpha": selected_alpha,
        "bound_low": CHAR_ALPHA_BOUND_LOW,
        "bound_high": CHAR_ALPHA_BOUND_HIGH,
        "hit_lower_bound": math.isclose(selected_alpha, CHAR_ALPHA_BOUND_LOW, rel_tol=0.0, abs_tol=1.0e-12),
        "hit_upper_bound": math.isclose(selected_alpha, CHAR_ALPHA_BOUND_HIGH, rel_tol=0.0, abs_tol=1.0e-12),
        "training_objective_initial": float(initial["training_objective"]),
        "training_objective_final": float(selected_grid_row["training_objective"]),
        "training_objective_delta": float(selected_grid_row["training_objective"]) - float(initial["training_objective"]),
        "selection_split": "train",
        "N_cells": int(n_cells),
        "steps_or_grid_points": len(grid_rows),
        "selection_method": "declared_grid_search_train_split_only",
        "status": "ok",
        "validation_excluded_from_selection": True,
        "test_excluded_from_selection": True,
        "parameter_path_clean": True,
        "jax_parity_caveat": audit["jax_parity_caveat"],
        "evidence_boundary": GLOBAL_CHAR_EVIDENCE_BOUNDARY,
    }
    return [{column: row.get(column) for column in GLOBAL_CHAR_SUMMARY_COLUMNS}]


def char_baseline_markdown(
    summary_rows: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
    grid_rows: Sequence[Mapping[str, Any]],
) -> str:
    summary = summary_rows[0]
    audit = char_oxidation_parameter_path_audit()
    lines = [
        "# Huayi global char-oxidation alpha baseline",
        "",
        "This is a non-learned global scalar comparator. Alpha is selected using train split only.",
        "",
        "## Parameter-path audit",
        "",
        f"- Mapped parameter: `{audit['mapped_parameter']}`",
        f"- Mainline path: {audit['mainline_cell_path']}",
        f"- JAX contract path: {audit['system_jax_contract_path']}",
        f"- Per-run injection without public API change: {audit['per_run_injection_without_public_api_change']}",
        f"- JAX/parity caveat: {audit['jax_parity_caveat']}",
        "",
        "## Selection",
        "",
        f"- Selected alpha: {float(summary['selected_alpha']):.6g}",
        f"- Bounds: [{summary['bound_low']}, {summary['bound_high']}]",
        f"- Hit lower bound: {summary['hit_lower_bound']}",
        f"- Hit upper bound: {summary['hit_upper_bound']}",
        f"- Training objective initial alpha=1.0: {float(summary['training_objective_initial']):.6g}",
        f"- Training objective selected: {float(summary['training_objective_final']):.6g}",
        f"- Training objective delta: {float(summary['training_objective_delta']):.6g}",
        f"- Validation excluded from selection: {summary['validation_excluded_from_selection']}",
        f"- Test excluded from selection: {summary['test_excluded_from_selection']}",
        "",
        "## Grid",
        "",
        "| alpha | train_objective | fallback_sum | poor_sum | cell0_min | cell0_max | status |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in grid_rows:
        lines.append(
            "| {alpha:.6g} | {objective:.6g} | {fallback} | {poor} | {cell0_min:.6g} | {cell0_max:.6g} | {status} |".format(
                alpha=float(row["alpha_value"]),
                objective=float(row["training_objective"]),
                fallback=int(row["fallback_count_sum"]),
                poor=int(row["poor_convergence_count_sum"]),
                cell0_min=float(row["cell0_cost_min"]),
                cell0_max=float(row["cell0_cost_max"]),
                status=row["status"],
            )
        )
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| model | split | N | target | n | RMSE | MAE | bias | rmse_delta_vs_fixed | fallback | poor |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metric_rows:
        lines.append(
            "| {model} | {split} | {n_cells} | {target} | {n} | {rmse:.6g} | {mae:.6g} | {bias:.6g} | {rmse_delta:.6g} | {fallback} | {poor} |".format(
                model=row["model_name"],
                split=row["split"],
                n_cells=int(row["N_cells"]),
                target=row["target"],
                n=int(row["n"]),
                rmse=float(row["rmse"]),
                mae=float(row["mae"]),
                bias=float(row["bias"]),
                rmse_delta=float(row["rmse_delta"]),
                fallback=int(row["fallback_count_sum"]),
                poor=int(row["poor_convergence_count_sum"]),
            )
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            f"Evidence boundary: {GLOBAL_CHAR_EVIDENCE_BOUNDARY}.",
            "",
            "This comparator can indicate whether the existing char-oxidation severity knob is worth staged learned-alpha exploration. It does not train alpha_j(x_t), identify learned mechanisms, prove temporal generalization, or support production control.",
        ]
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "CHAR_ALPHA_BOUND_HIGH",
    "CHAR_ALPHA_BOUND_LOW",
    "DEFAULT_CHAR_ALPHA_GRID",
    "GLOBAL_CHAR_EVIDENCE_BOUNDARY",
    "GLOBAL_CHAR_GRID_COLUMNS",
    "GLOBAL_CHAR_SUMMARY_COLUMNS",
    "build_char_summary",
    "char_baseline_markdown",
    "char_objective_from_result",
    "char_oxidation_parameter_path_audit",
    "char_oxidation_prediction_rows",
    "clamp_char_alpha",
    "inject_char_oxidation_factor",
    "select_global_char_oxidation_alpha",
    "solve_with_char_oxidation_alpha",
    "validate_char_alpha_grid",
]
