"""
Bounded global multi-alpha scalar diagnostic.

This module selects global scalar multipliers on the frozen train split for
small alpha subsets.  It does not implement alpha_j(x_t), z_d, or neural
training.
"""
from __future__ import annotations

import itertools
import math
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .gasifier_system import GasifierSystem
from .model_case_adapter import build_gasifier_inputs_from_model_case
from .model_case_diagnostic_baseline import POOR_CONVERGENCE_COST_THRESHOLD, cell_residual_quality

GLOBAL_MULTIALPHA_EVIDENCE_BOUNDARY = (
    "bounded global multi-alpha scalar diagnostic selected on train split only; "
    "not learned alpha_j(x_t), weak-label coal-quality modeling, learned "
    "multiplier interpretability proof, production control, or final "
    "temporal-generalization evidence"
)

TARGETS: tuple[str, ...] = ("CO_volpct_dry", "H2_volpct_dry", "CO2_volpct_dry", "H2_CO_ratio")

ALPHA_BOUNDS: dict[str, tuple[float, float]] = {
    "alpha_wgs": (0.02, 2.0),
    "alpha_char_oxidation": (0.05, 1.0),
    "alpha_volatile_oxidation": (0.2, 1.2),
}

ALPHA_DEFAULTS: dict[str, float] = {
    "alpha_wgs": 1.0,
    "alpha_char_oxidation": 1.0,
    "alpha_volatile_oxidation": 1.0,
}

ALPHA_GRIDS: dict[str, tuple[float, ...]] = {
    "alpha_wgs": (0.3, 1.0),
    "alpha_char_oxidation": (0.1, 1.0),
    "alpha_volatile_oxidation": (0.2, 1.0),
}

SUBSETS: tuple[tuple[str, ...], ...] = (
    ("alpha_wgs",),
    ("alpha_char_oxidation",),
    ("alpha_volatile_oxidation",),
    ("alpha_wgs", "alpha_char_oxidation"),
    ("alpha_wgs", "alpha_volatile_oxidation"),
    ("alpha_char_oxidation", "alpha_volatile_oxidation"),
    ("alpha_wgs", "alpha_char_oxidation", "alpha_volatile_oxidation"),
)

SUMMARY_COLUMNS: tuple[str, ...] = (
    "subset_name",
    "included_alphas",
    "selection_split",
    "N_cells",
    "alpha_wgs",
    "alpha_char_oxidation",
    "alpha_volatile_oxidation",
    "alpha_wgs_bound_hit",
    "alpha_char_oxidation_bound_hit",
    "alpha_volatile_oxidation_bound_hit",
    "train_objective_initial",
    "train_objective_final",
    "train_objective_delta",
    "grid_points",
    "selection_method",
    "validation_excluded_from_selection",
    "test_excluded_from_selection",
    "fallback_count_sum_train",
    "poor_convergence_count_sum_train",
    "cell0_cost_min_train",
    "cell0_cost_max_train",
    "compensation_flag",
    "recommendation_hint",
    "status",
    "evidence_boundary",
)

GRID_COLUMNS: tuple[str, ...] = (
    "subset_name",
    "included_alphas",
    "selection_split",
    "N_cells",
    "alpha_wgs",
    "alpha_char_oxidation",
    "alpha_volatile_oxidation",
    "train_objective",
    "fallback_count_sum",
    "poor_convergence_count_sum",
    "cell0_cost_min",
    "cell0_cost_max",
    "status",
    "evidence_boundary",
)

PREDICTION_COLUMNS: tuple[str, ...] = (
    "subset_name",
    "case_id",
    "phase",
    "split",
    "target",
    "target_value",
    "fixed_prediction",
    "multialpha_prediction",
    "fixed_residual",
    "multialpha_residual",
    "global_wgs_prediction",
    "global_wgs_residual",
    "global_char_prediction",
    "global_char_residual",
    "alpha_wgs_xt_prediction",
    "alpha_wgs_xt_residual",
    "N_cells",
    "alpha_wgs",
    "alpha_char_oxidation",
    "alpha_volatile_oxidation",
    "fallback_count",
    "poor_convergence_count",
    "poor_convergence_cell_index",
    "cell0_cost",
    "status",
    "evidence_boundary",
)

METRIC_COLUMNS: tuple[str, ...] = (
    "subset_name",
    "split",
    "target",
    "N_cells",
    "n",
    "fixed_rmse",
    "multialpha_rmse",
    "rmse_delta_vs_fixed",
    "global_wgs_rmse",
    "rmse_delta_vs_global_wgs",
    "global_char_rmse",
    "rmse_delta_vs_global_char",
    "alpha_wgs_xt_rmse",
    "rmse_delta_vs_alpha_wgs_xt",
    "fixed_mae",
    "multialpha_mae",
    "mae_delta_vs_fixed",
    "fixed_bias",
    "multialpha_bias",
    "bias_delta_vs_fixed",
    "fallback_count_sum",
    "poor_convergence_count_sum",
    "cell0_cost_min",
    "cell0_cost_max",
    "evidence_boundary",
)


def subset_name(subset: Sequence[str]) -> str:
    return "+".join(subset)


def explicit_volatile_mapping_note() -> str:
    return "alpha_volatile_oxidation maps to both op_conds.CO_OxidationFactor and op_conds.H2_OxidationFactor"


def apply_alpha_values(op_conds: Mapping[str, Any], alpha_values: Mapping[str, float]) -> dict[str, Any]:
    out = dict(op_conds)
    if "alpha_wgs" in alpha_values:
        out["WGS_CatalyticFactor"] = float(alpha_values["alpha_wgs"])
    if "alpha_char_oxidation" in alpha_values:
        out["CharCombustionRateFactor"] = float(alpha_values["alpha_char_oxidation"])
    if "alpha_volatile_oxidation" in alpha_values:
        value = float(alpha_values["alpha_volatile_oxidation"])
        out["CO_OxidationFactor"] = value
        out["H2_OxidationFactor"] = value
    return out


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


def objective_from_result(result: Mapping[str, Any]) -> float:
    target_h2co = float(result["target_H2_volpct_dry"]) / max(float(result["target_CO_volpct_dry"]), 1.0e-12)
    return float(
        ((float(result["CO_volpct_dry"]) - float(result["target_CO_volpct_dry"])) / 10.0) ** 2
        + ((float(result["H2_volpct_dry"]) - float(result["target_H2_volpct_dry"])) / 10.0) ** 2
        + ((float(result["CO2_volpct_dry"]) - float(result["target_CO2_volpct_dry"])) / 10.0) ** 2
        + ((float(result["H2_CO_ratio"]) - target_h2co) / 0.1) ** 2
    )


def solve_with_alpha_values(
    model_case_row: Mapping[str, Any],
    alpha_values: Mapping[str, float],
    n_cells: int = 3,
) -> dict[str, Any]:
    built = build_gasifier_inputs_from_model_case(model_case_row)
    op_conds = apply_alpha_values(built["op_conds"], alpha_values)
    start = time.perf_counter()
    system = GasifierSystem(built["geometry"], built["coal_props"], op_conds)
    profile, _z = system.solve(N_cells=int(n_cells), solver_method="minimize", jacobian_mode="scipy", jax_warmup=False)
    runtime_s = time.perf_counter() - start
    residual_quality = cell_residual_quality(system, profile)
    poor_cells = [item["cell_index"] for item in residual_quality if item["cost"] > POOR_CONVERGENCE_COST_THRESHOLD]
    cell0 = residual_quality[0] if residual_quality else {}
    targets = built["targets"]
    kpis = _dry_syngas_kpis(profile)
    out = {
        "case_id": built["case_id"],
        "phase": model_case_row.get("phase"),
        "dataset_id": model_case_row.get("dataset_id"),
        "split": model_case_row.get("_split", model_case_row.get("split")),
        "split_role": model_case_row.get("_split_role", model_case_row.get("split_role")),
        "runtime_s": runtime_s,
        "fallback_count": int(system.solve_stats.get("fallback_count", 0)),
        "poor_convergence_count": int(system.solve_stats.get("poor_convergence_count", 0)),
        "poor_convergence_cell_index": ";".join(str(i) for i in poor_cells),
        "cell0_cost": float(cell0.get("cost", math.nan)),
        "solve_status": "ok",
        "target_CO_volpct_dry": float(targets["outlet_co_dry_vol_pct"]),
        "target_H2_volpct_dry": float(targets["outlet_h2_dry_vol_pct"]),
        "target_CO2_volpct_dry": float(targets["outlet_co2_dry_vol_pct"]),
    }
    out["target_H2_CO_ratio"] = out["target_H2_volpct_dry"] / max(out["target_CO_volpct_dry"], 1.0e-12)
    out.update(kpis)
    out["objective"] = objective_from_result(out)
    return out


SolverFn = Callable[[Mapping[str, Any], Mapping[str, float], int], Mapping[str, Any]]


def _safe_solve(solver_fn: SolverFn, row: Mapping[str, Any], alpha_values: Mapping[str, float], n_cells: int) -> dict[str, Any]:
    try:
        return dict(solver_fn(row, alpha_values, int(n_cells)))
    except Exception as exc:  # pragma: no cover - diagnostic failure path
        return {
            "case_id": row.get("_model_case_id", row.get("case_id", "unknown")),
            "phase": row.get("phase"),
            "split": row.get("_split", row.get("split")),
            "objective": math.nan,
            "fallback_count": math.nan,
            "poor_convergence_count": math.nan,
            "cell0_cost": math.nan,
            "solve_status": f"failed:{type(exc).__name__}:{exc}",
        }


def alpha_grid_for_subset(subset: Sequence[str]) -> list[dict[str, float]]:
    names = tuple(subset)
    value_lists = [ALPHA_GRIDS[name] for name in names]
    rows: list[dict[str, float]] = []
    for values in itertools.product(*value_lists):
        item = {name: float(value) for name, value in zip(names, values)}
        rows.append(item)
    return rows


def _full_alpha_values(subset_values: Mapping[str, float]) -> dict[str, float]:
    out = dict(ALPHA_DEFAULTS)
    out.update({key: float(value) for key, value in subset_values.items()})
    return out


def _bound_hit(alpha_name: str, value: float) -> bool:
    low, high = ALPHA_BOUNDS[alpha_name]
    return math.isclose(float(value), low, rel_tol=0.0, abs_tol=1.0e-12) or math.isclose(float(value), high, rel_tol=0.0, abs_tol=1.0e-12)


def _target_values(result: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    return {
        "CO_volpct_dry": (float(result["target_CO_volpct_dry"]), float(result["CO_volpct_dry"])),
        "H2_volpct_dry": (float(result["target_H2_volpct_dry"]), float(result["H2_volpct_dry"])),
        "CO2_volpct_dry": (float(result["target_CO2_volpct_dry"]), float(result["CO2_volpct_dry"])),
        "H2_CO_ratio": (float(result["target_H2_CO_ratio"]), float(result["H2_CO_ratio"])),
    }


def select_global_multialpha_subsets(
    model_case_rows_with_split: Sequence[Mapping[str, Any]],
    *,
    n_cells: int = 3,
    solver_fn: SolverFn = solve_with_alpha_values,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows = [row for row in model_case_rows_with_split if row.get("_split", row.get("split")) == "train"]
    grid_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for subset in SUBSETS:
        name = subset_name(subset)
        subset_grid = alpha_grid_for_subset(subset)
        scored: list[dict[str, Any]] = []
        for values in subset_grid:
            full_values = _full_alpha_values(values)
            results = [_safe_solve(solver_fn, row, full_values, n_cells) for row in train_rows]
            objectives = [float(row["objective"]) for row in results if math.isfinite(float(row.get("objective", math.nan)))]
            cell0_values = [float(row.get("cell0_cost", math.nan)) for row in results if math.isfinite(float(row.get("cell0_cost", math.nan)))]
            status = "ok" if len(objectives) == len(train_rows) else "partial_or_failed"
            train_objective = float(np.mean(objectives)) if objectives else math.nan
            item = {
                "subset_name": name,
                "included_alphas": ",".join(subset),
                "selection_split": "train",
                "N_cells": int(n_cells),
                "alpha_wgs": full_values["alpha_wgs"],
                "alpha_char_oxidation": full_values["alpha_char_oxidation"],
                "alpha_volatile_oxidation": full_values["alpha_volatile_oxidation"],
                "train_objective": train_objective,
                "fallback_count_sum": int(sum(float(row.get("fallback_count", 0.0)) for row in results if math.isfinite(float(row.get("fallback_count", math.nan))))),
                "poor_convergence_count_sum": int(sum(float(row.get("poor_convergence_count", 0.0)) for row in results if math.isfinite(float(row.get("poor_convergence_count", math.nan))))),
                "cell0_cost_min": float(min(cell0_values)) if cell0_values else math.nan,
                "cell0_cost_max": float(max(cell0_values)) if cell0_values else math.nan,
                "status": status,
                "evidence_boundary": GLOBAL_MULTIALPHA_EVIDENCE_BOUNDARY,
            }
            grid_rows.append(item)
            if status == "ok" and math.isfinite(train_objective):
                scored.append(item)
        if not scored:
            raise RuntimeError(f"No valid grid row for subset {name}")
        selected = min(scored, key=lambda row: (float(row["train_objective"]), sum(_bound_hit(alpha, row[alpha]) for alpha in ALPHA_BOUNDS)))
        initial = next(row for row in grid_rows if row["subset_name"] == name and all(math.isclose(float(row[alpha]), ALPHA_DEFAULTS[alpha], rel_tol=0.0, abs_tol=1.0e-12) for alpha in ALPHA_BOUNDS))
        strong_moves = sum(abs(math.log(max(float(selected[alpha]), 1.0e-12) / ALPHA_DEFAULTS[alpha])) > math.log(1.5) for alpha in subset)
        bound_hits = {alpha: (alpha in subset and _bound_hit(alpha, selected[alpha])) for alpha in ALPHA_BOUNDS}
        compensation_flag = "multi_alpha_strong_move" if strong_moves >= 2 else "single_or_mild_move"
        if any(bound_hits.values()):
            compensation_flag += "+bound_hit"
        summary_rows.append(
            {
                "subset_name": name,
                "included_alphas": ",".join(subset),
                "selection_split": "train",
                "N_cells": int(n_cells),
                "alpha_wgs": selected["alpha_wgs"],
                "alpha_char_oxidation": selected["alpha_char_oxidation"],
                "alpha_volatile_oxidation": selected["alpha_volatile_oxidation"],
                "alpha_wgs_bound_hit": bound_hits["alpha_wgs"],
                "alpha_char_oxidation_bound_hit": bound_hits["alpha_char_oxidation"],
                "alpha_volatile_oxidation_bound_hit": bound_hits["alpha_volatile_oxidation"],
                "train_objective_initial": float(initial["train_objective"]),
                "train_objective_final": float(selected["train_objective"]),
                "train_objective_delta": float(selected["train_objective"]) - float(initial["train_objective"]),
                "grid_points": len(subset_grid),
                "selection_method": "coarse_bounded_grid_train_split_only",
                "validation_excluded_from_selection": True,
                "test_excluded_from_selection": True,
                "fallback_count_sum_train": int(selected["fallback_count_sum"]),
                "poor_convergence_count_sum_train": int(selected["poor_convergence_count_sum"]),
                "cell0_cost_min_train": float(selected["cell0_cost_min"]),
                "cell0_cost_max_train": float(selected["cell0_cost_max"]),
                "compensation_flag": compensation_flag,
                "recommendation_hint": "diagnostic_only_review_heldout_before_learning",
                "status": "ok",
                "evidence_boundary": GLOBAL_MULTIALPHA_EVIDENCE_BOUNDARY,
            }
        )
    return (
        [{column: row.get(column) for column in SUMMARY_COLUMNS} for row in summary_rows],
        [{column: row.get(column) for column in GRID_COLUMNS} for row in grid_rows],
    )


def _prediction_lookup(rows: Sequence[Mapping[str, Any]], n_cells: int) -> dict[tuple[str, str], Mapping[str, Any]]:
    out: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        if int(row.get("N_cells", n_cells)) != int(n_cells):
            continue
        out[(str(row["case_id"]), str(row["target"]))] = row
    return out


def global_multialpha_prediction_rows(
    model_case_rows_with_split: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    fixed_prediction_rows: Sequence[Mapping[str, Any]],
    global_wgs_prediction_rows: Sequence[Mapping[str, Any]],
    global_char_prediction_rows: Sequence[Mapping[str, Any]],
    alpha_wgs_xt_prediction_rows: Sequence[Mapping[str, Any]],
    *,
    n_cells: int = 3,
    solver_fn: SolverFn = solve_with_alpha_values,
) -> list[dict[str, Any]]:
    fixed_lookup = _prediction_lookup(fixed_prediction_rows, n_cells)
    wgs_lookup = _prediction_lookup(global_wgs_prediction_rows, n_cells)
    char_lookup = _prediction_lookup(global_char_prediction_rows, n_cells)
    xt_lookup = _prediction_lookup(alpha_wgs_xt_prediction_rows, n_cells)
    rows: list[dict[str, Any]] = []
    for summary in summary_rows:
        subset = summary["subset_name"]
        alpha_values = {alpha: float(summary[alpha]) for alpha in ALPHA_BOUNDS}
        for model_case in model_case_rows_with_split:
            result = _safe_solve(solver_fn, model_case, alpha_values, n_cells)
            values = _target_values(result)
            for target, (target_value, prediction) in values.items():
                case_id = str(result["case_id"])
                fixed = fixed_lookup[(case_id, target)]
                wgs = wgs_lookup.get((case_id, target), {})
                char = char_lookup.get((case_id, target), {})
                xt = xt_lookup.get((case_id, target), {})
                fixed_pred = float(fixed["y_pred"])
                rows.append(
                    {
                        "subset_name": subset,
                        "case_id": case_id,
                        "phase": result.get("phase"),
                        "split": result.get("split"),
                        "target": target,
                        "target_value": target_value,
                        "fixed_prediction": fixed_pred,
                        "multialpha_prediction": prediction,
                        "fixed_residual": fixed_pred - target_value,
                        "multialpha_residual": prediction - target_value,
                        "global_wgs_prediction": float(wgs["y_pred"]) if wgs else math.nan,
                        "global_wgs_residual": float(wgs["residual"]) if wgs else math.nan,
                        "global_char_prediction": float(char["y_pred"]) if char else math.nan,
                        "global_char_residual": float(char["residual"]) if char else math.nan,
                        "alpha_wgs_xt_prediction": float(xt["alpha_prediction"]) if xt else math.nan,
                        "alpha_wgs_xt_residual": float(xt["alpha_residual"]) if xt else math.nan,
                        "N_cells": int(n_cells),
                        "alpha_wgs": alpha_values["alpha_wgs"],
                        "alpha_char_oxidation": alpha_values["alpha_char_oxidation"],
                        "alpha_volatile_oxidation": alpha_values["alpha_volatile_oxidation"],
                        "fallback_count": int(result.get("fallback_count", 0) or 0),
                        "poor_convergence_count": int(result.get("poor_convergence_count", 0) or 0),
                        "poor_convergence_cell_index": result.get("poor_convergence_cell_index", ""),
                        "cell0_cost": float(result.get("cell0_cost", math.nan)),
                        "status": result.get("solve_status", "ok"),
                        "evidence_boundary": GLOBAL_MULTIALPHA_EVIDENCE_BOUNDARY,
                    }
                )
    return [{column: row.get(column) for column in PREDICTION_COLUMNS} for row in rows]


def global_multialpha_metric_rows(prediction_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame(prediction_rows)
    rows: list[dict[str, Any]] = []
    for (subset, split, target, n_cells), group in df.groupby(["subset_name", "split", "target", "N_cells"], dropna=False):
        fixed = group["fixed_residual"].astype(float).to_numpy()
        model = group["multialpha_residual"].astype(float).to_numpy()
        wgs = group["global_wgs_residual"].astype(float).to_numpy()
        char = group["global_char_residual"].astype(float).to_numpy()
        xt = group["alpha_wgs_xt_residual"].astype(float).to_numpy()
        fixed_rmse = float(np.sqrt(np.mean(fixed * fixed)))
        model_rmse = float(np.sqrt(np.mean(model * model)))
        wgs_rmse = float(np.sqrt(np.nanmean(wgs * wgs)))
        char_rmse = float(np.sqrt(np.nanmean(char * char)))
        xt_rmse = float(np.sqrt(np.nanmean(xt * xt)))
        fixed_mae = float(np.mean(np.abs(fixed)))
        model_mae = float(np.mean(np.abs(model)))
        fixed_bias = float(np.mean(fixed))
        model_bias = float(np.mean(model))
        rows.append(
            {
                "subset_name": subset,
                "split": split,
                "target": target,
                "N_cells": int(n_cells),
                "n": int(len(group)),
                "fixed_rmse": fixed_rmse,
                "multialpha_rmse": model_rmse,
                "rmse_delta_vs_fixed": model_rmse - fixed_rmse,
                "global_wgs_rmse": wgs_rmse,
                "rmse_delta_vs_global_wgs": model_rmse - wgs_rmse,
                "global_char_rmse": char_rmse,
                "rmse_delta_vs_global_char": model_rmse - char_rmse,
                "alpha_wgs_xt_rmse": xt_rmse,
                "rmse_delta_vs_alpha_wgs_xt": model_rmse - xt_rmse,
                "fixed_mae": fixed_mae,
                "multialpha_mae": model_mae,
                "mae_delta_vs_fixed": model_mae - fixed_mae,
                "fixed_bias": fixed_bias,
                "multialpha_bias": model_bias,
                "bias_delta_vs_fixed": model_bias - fixed_bias,
                "fallback_count_sum": int(group["fallback_count"].sum()),
                "poor_convergence_count_sum": int(group["poor_convergence_count"].sum()),
                "cell0_cost_min": float(group["cell0_cost"].min()),
                "cell0_cost_max": float(group["cell0_cost"].max()),
                "evidence_boundary": GLOBAL_MULTIALPHA_EVIDENCE_BOUNDARY,
            }
        )
    return [{column: row.get(column) for column in METRIC_COLUMNS} for row in rows]


def add_heldout_compensation_to_summary(
    summary_rows: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    metrics = pd.DataFrame(metric_rows)
    out: list[dict[str, Any]] = []
    for row in summary_rows:
        item = dict(row)
        heldout = metrics[(metrics["subset_name"] == row["subset_name"]) & (metrics["split"].isin(["validation", "test"]))]
        improved = int((heldout["rmse_delta_vs_fixed"] < 0.0).sum())
        worsened = int((heldout["rmse_delta_vs_fixed"] > 0.0).sum())
        if improved > 0 and worsened > 0:
            item["compensation_flag"] = str(item["compensation_flag"]) + "+heldout_target_tradeoff"
        item["recommendation_hint"] = "review_heldout_tradeoff_before_learning"
        out.append({column: item.get(column) for column in SUMMARY_COLUMNS})
    return out


def recommend_next_experiment(summary_rows: Sequence[Mapping[str, Any]], metric_rows: Sequence[Mapping[str, Any]]) -> str:
    metrics = pd.DataFrame(metric_rows)
    heldout = metrics[metrics["split"].isin(["validation", "test"])]
    by_subset = (
        heldout.groupby("subset_name")
        .agg(mean_delta_vs_fixed=("rmse_delta_vs_fixed", "mean"), improved_vs_fixed=("rmse_delta_vs_fixed", lambda x: int((x < 0).sum())))
        .reset_index()
    )
    if by_subset.empty:
        return "stop_and_expand_data_before_learned_multi_alpha"
    best = by_subset.sort_values(["mean_delta_vs_fixed", "improved_vs_fixed"], ascending=[True, False]).iloc[0]
    best_name = str(best["subset_name"])
    best_summary = next(row for row in summary_rows if row["subset_name"] == best_name)
    bound_hits = sum(bool(best_summary[f"{alpha}_bound_hit"]) for alpha in ALPHA_BOUNDS)
    if float(best["mean_delta_vs_fixed"]) >= 0.0:
        return "stop_and_expand_data_before_learned_multi_alpha"
    if "+" not in best_name:
        return f"stay_{best_name}_only"
    if bound_hits:
        return f"{best_name}_with_strong_regularization_and_bound_hit_monitoring"
    return f"{best_name}_with_strong_regularization"


def multialpha_markdown(
    summary_rows: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
) -> str:
    metrics = pd.DataFrame(metric_rows)
    recommendation = recommend_next_experiment(summary_rows, metric_rows)
    lines = [
        "# Huayi bounded global multi-alpha diagnostic",
        "",
        "This diagnostic selects global scalar multipliers on the frozen train split only.",
        "",
        f"- Volatile mapping: {explicit_volatile_mapping_note()}",
        f"- Recommendation: {recommendation}",
        "",
        "## Subset selection",
        "",
        "| subset | alpha_wgs | alpha_char | alpha_vol | train objective | delta | bound hits | compensation |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        bound_hits = sum(bool(row[f"{alpha}_bound_hit"]) for alpha in ALPHA_BOUNDS)
        lines.append(
            "| {subset} | {wgs:.6g} | {char:.6g} | {vol:.6g} | {obj:.6g} | {delta:.6g} | {hits} | {flag} |".format(
                subset=row["subset_name"],
                wgs=float(row["alpha_wgs"]),
                char=float(row["alpha_char_oxidation"]),
                vol=float(row["alpha_volatile_oxidation"]),
                obj=float(row["train_objective_final"]),
                delta=float(row["train_objective_delta"]),
                hits=bound_hits,
                flag=row["compensation_flag"],
            )
        )
    lines.extend(["", "## Held-out metric headline", "", "| subset | improved vs fixed | improved vs global WGS | mean delta vs fixed | mean delta vs WGS |", "|---|---:|---:|---:|---:|"])
    heldout = metrics[metrics["split"].isin(["validation", "test"])]
    for subset, group in heldout.groupby("subset_name"):
        lines.append(
            "| {subset} | {fixed_improved}/{n} | {wgs_improved}/{n} | {fixed_delta:.6g} | {wgs_delta:.6g} |".format(
                subset=subset,
                fixed_improved=int((group["rmse_delta_vs_fixed"] < 0.0).sum()),
                wgs_improved=int((group["rmse_delta_vs_global_wgs"] < 0.0).sum()),
                n=len(group),
                fixed_delta=float(group["rmse_delta_vs_fixed"].mean()),
                wgs_delta=float(group["rmse_delta_vs_global_wgs"].mean()),
            )
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            f"Evidence boundary: {GLOBAL_MULTIALPHA_EVIDENCE_BOUNDARY}.",
            "",
            "This is not learned alpha_j(x_t), weak-label coal-quality modeling, learned multiplier interpretability proof, production control, or final temporal-generalization evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "ALPHA_BOUNDS",
    "GRID_COLUMNS",
    "METRIC_COLUMNS",
    "PREDICTION_COLUMNS",
    "SUBSETS",
    "SUMMARY_COLUMNS",
    "add_heldout_compensation_to_summary",
    "apply_alpha_values",
    "explicit_volatile_mapping_note",
    "global_multialpha_metric_rows",
    "global_multialpha_prediction_rows",
    "multialpha_markdown",
    "recommend_next_experiment",
    "select_global_multialpha_subsets",
]
