"""
Minimal WGS-only bounded alpha(x_t) scaffold for Huayi diagnostics.

The model is intentionally tiny: train-only feature scaling, a linear logit
map to bounded ``WGS_CatalyticFactor``, and finite candidate selection.  It is
not a full hybrid, multi-alpha model, or z_d model.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .bounded_hybrid_correction_smoke import solve_with_wgs_correction
from .direct_residual_baseline import OPERATING_FEATURE_COLUMNS, operating_feature_rows
from .split_global_alpha_baseline import wgs_objective_from_result

ALPHA_WGS_XT_MODEL_NAME = "alpha_wgs_ridge_logit"
ALPHA_WGS_NAME = "alpha_wgs"
ALPHA_WGS_LOW = 0.02
ALPHA_WGS_HIGH = 2.0
DEFAULT_BASE_ALPHA = 0.3
DEFAULT_SLOPE_MAGNITUDE = 0.4
DEFAULT_REGULARIZATION = 1.0e-3

ALPHA_WGS_XT_EVIDENCE_BOUNDARY = (
    "minimal WGS-only bounded alpha_wgs(x_t) scaffold selected on train split "
    "only; not full hybrid, multi-alpha calibration, learned multiplier "
    "interpretability, weak-label proof, production control, final temporal "
    "generalization, or Yulin/slurry generalization"
)

ALPHA_WGS_XT_SUMMARY_COLUMNS: tuple[str, ...] = (
    "model_name",
    "alpha_name",
    "feature_set",
    "bounds",
    "regularization",
    "selection_or_training_method",
    "train_objective_initial",
    "train_objective_final",
    "train_objective_delta",
    "selected_candidate_name",
    "selected_intercept",
    "selected_nonzero_feature",
    "selected_nonzero_coefficient",
    "validation_excluded_from_training",
    "test_excluded_from_training",
    "N_cells",
    "candidate_count",
    "status",
    "evidence_boundary",
)

ALPHA_WGS_XT_PREDICTION_COLUMNS: tuple[str, ...] = (
    "case_id",
    "phase",
    "split",
    "alpha_wgs",
    "bound_hit",
    "target",
    "target_value",
    "fixed_prediction",
    "alpha_prediction",
    "fixed_residual",
    "alpha_residual",
    "N_cells",
    "status",
    "evidence_boundary",
    "dataset_id",
    "split_role",
    "global_wgs_prediction",
    "global_wgs_residual",
    "fallback_count",
    "poor_convergence_count",
    "poor_convergence_cell_index",
    "cell0_cost",
)

ALPHA_WGS_XT_METRIC_COLUMNS: tuple[str, ...] = (
    "model_name",
    "split",
    "target",
    "N_cells",
    "n",
    "fixed_rmse",
    "alpha_rmse",
    "rmse_delta_vs_fixed",
    "global_wgs_rmse",
    "rmse_delta_vs_global_wgs",
    "fixed_mae",
    "alpha_mae",
    "mae_delta_vs_fixed",
    "global_wgs_mae",
    "mae_delta_vs_global_wgs",
    "fixed_bias",
    "alpha_bias",
    "bias_delta_vs_fixed",
    "alpha_min",
    "alpha_p05",
    "alpha_median",
    "alpha_p95",
    "alpha_max",
    "bound_hit_count",
    "bound_hit_fraction",
    "fallback_count_sum",
    "poor_convergence_count_sum",
    "cell0_cost_min",
    "cell0_cost_max",
    "evidence_boundary",
)

ALPHA_WGS_XT_AUDIT_COLUMNS: tuple[str, ...] = (
    "record_type",
    "name",
    "value",
    "feature",
    "coefficient",
    "mean",
    "scale",
    "fit_split",
    "validation_excluded_from_training",
    "test_excluded_from_training",
    "status",
    "evidence_boundary",
)

SolverFn = Callable[[Mapping[str, Any], float, int], Mapping[str, Any]]


def solve_wgs_for_alpha_xt(row: Mapping[str, Any], alpha: float, n_cells: int) -> Mapping[str, Any]:
    """Adapter around the existing WGS correction solver's keyword-only API."""
    return solve_with_wgs_correction(row, float(alpha), n_cells=int(n_cells))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _logit(prob: float) -> float:
    p = min(max(float(prob), 1.0e-12), 1.0 - 1.0e-12)
    return math.log(p / (1.0 - p))


def alpha_to_logit(alpha: float) -> float:
    p = (float(alpha) - ALPHA_WGS_LOW) / (ALPHA_WGS_HIGH - ALPHA_WGS_LOW)
    return _logit(p)


def logit_to_alpha(logit_value: float) -> float:
    return ALPHA_WGS_LOW + (ALPHA_WGS_HIGH - ALPHA_WGS_LOW) * _sigmoid(float(logit_value))


def feature_rows_with_split(model_case_rows_with_split: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    base = operating_feature_rows(model_case_rows_with_split)
    meta = {
        str(row.get("_model_case_id", row.get("case_id"))): {
            "split": row.get("_split", row.get("split")),
            "split_role": row.get("_split_role", row.get("split_role")),
            "phase": row.get("phase"),
            "dataset_id": row.get("dataset_id"),
        }
        for row in model_case_rows_with_split
    }
    rows: list[dict[str, Any]] = []
    for row in base:
        item = dict(row)
        item.update(meta[str(row["case_id"])])
        rows.append(item)
    return rows


def fit_train_feature_scaler(feature_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame(feature_rows)
    train = df[df["split"] == "train"]
    if train.empty:
        raise ValueError("No train feature rows available for alpha_wgs(x_t) scaling")
    values = train[list(OPERATING_FEATURE_COLUMNS)].astype(float).to_numpy()
    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    return {
        "feature_columns": tuple(OPERATING_FEATURE_COLUMNS),
        "mean": tuple(float(v) for v in mean),
        "scale": tuple(float(v) for v in scale),
        "fit_split": "train",
        "validation_excluded_from_training": True,
        "test_excluded_from_training": True,
    }


def transform_feature_row(row: Mapping[str, Any], scaler: Mapping[str, Any]) -> np.ndarray:
    values = np.array([float(row[column]) for column in scaler["feature_columns"]], dtype=float)
    mean = np.array(scaler["mean"], dtype=float)
    scale = np.array(scaler["scale"], dtype=float)
    return (values - mean) / scale


def build_candidate_models(
    scaler: Mapping[str, Any],
    *,
    base_alpha: float = DEFAULT_BASE_ALPHA,
    slope_magnitude: float = DEFAULT_SLOPE_MAGNITUDE,
) -> list[dict[str, Any]]:
    feature_columns = tuple(scaler["feature_columns"])
    zero = tuple(0.0 for _ in feature_columns)
    candidates = [
        {
            "candidate_name": "constant_alpha_1p0_initial",
            "intercept": alpha_to_logit(1.0),
            "coefficients": zero,
            "nonzero_feature": "",
            "nonzero_coefficient": 0.0,
        },
        {
            "candidate_name": f"constant_alpha_{base_alpha:g}",
            "intercept": alpha_to_logit(base_alpha),
            "coefficients": zero,
            "nonzero_feature": "",
            "nonzero_coefficient": 0.0,
        },
    ]
    for idx, feature in enumerate(feature_columns):
        for sign in (-1.0, 1.0):
            coeff = [0.0 for _ in feature_columns]
            coeff[idx] = sign * float(slope_magnitude)
            candidates.append(
                {
                    "candidate_name": f"{feature}_{'pos' if sign > 0 else 'neg'}_slope",
                    "intercept": alpha_to_logit(base_alpha),
                    "coefficients": tuple(coeff),
                    "nonzero_feature": feature,
                    "nonzero_coefficient": coeff[idx],
                }
            )
    return candidates


def predict_alpha_for_feature_row(model: Mapping[str, Any], feature_row: Mapping[str, Any], scaler: Mapping[str, Any]) -> float:
    z = transform_feature_row(feature_row, scaler)
    coeff = np.array(model["coefficients"], dtype=float)
    return logit_to_alpha(float(model["intercept"]) + float(z @ coeff))


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
            "solve_status": f"failed:{type(exc).__name__}:{exc}",
        }


def select_alpha_wgs_xt_model(
    model_case_rows_with_split: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    scaler: Mapping[str, Any],
    *,
    n_cells: int = 3,
    regularization: float = DEFAULT_REGULARIZATION,
    solver_fn: SolverFn = solve_wgs_for_alpha_xt,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows_by_case = {str(row.get("_model_case_id", row.get("case_id"))): row for row in model_case_rows_with_split}
    train_features = [row for row in feature_rows if row["split"] == "train"]
    candidates = build_candidate_models(scaler)
    candidate_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        objectives: list[float] = []
        alphas: list[float] = []
        status = "ok"
        for feature_row in train_features:
            alpha = predict_alpha_for_feature_row(candidate, feature_row, scaler)
            alphas.append(alpha)
            result = _safe_solve(solver_fn, rows_by_case[str(feature_row["case_id"])], alpha, n_cells)
            if not str(result.get("solve_status", "ok")).startswith("ok"):
                status = str(result.get("solve_status"))
                objectives.append(math.nan)
                continue
            objectives.append(wgs_objective_from_result(result))
        finite_objectives = [value for value in objectives if math.isfinite(float(value))]
        mean_objective = float(np.mean(finite_objectives)) if finite_objectives else math.nan
        penalty = float(regularization) * float(np.mean((np.array(alphas) - 1.0) ** 2)) if alphas else math.nan
        candidate_rows.append(
            {
                "candidate_name": candidate["candidate_name"],
                "train_objective": mean_objective,
                "regularization_penalty": penalty,
                "selection_objective": mean_objective + penalty if math.isfinite(mean_objective) else math.nan,
                "train_alpha_min": float(min(alphas)) if alphas else math.nan,
                "train_alpha_max": float(max(alphas)) if alphas else math.nan,
                "status": status,
                **candidate,
            }
        )
    ok_rows = [row for row in candidate_rows if row["status"] == "ok" and math.isfinite(float(row["selection_objective"]))]
    if not ok_rows:
        raise RuntimeError("No valid alpha_wgs(x_t) candidates were generated")
    selected = min(ok_rows, key=lambda row: (float(row["selection_objective"]), abs(float(row["nonzero_coefficient"]))))
    return selected, candidate_rows


def _target_values(result: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    target_h2co = float(result["target_H2_volpct_dry"]) / max(float(result["target_CO_volpct_dry"]), 1.0e-12)
    return {
        "CO_volpct_dry": (float(result["target_CO_volpct_dry"]), float(result["CO_volpct_dry"])),
        "H2_volpct_dry": (float(result["target_H2_volpct_dry"]), float(result["H2_volpct_dry"])),
        "CO2_volpct_dry": (float(result["target_CO2_volpct_dry"]), float(result["CO2_volpct_dry"])),
        "H2_CO_ratio": (target_h2co, float(result["H2_CO_ratio"])),
    }


def alpha_wgs_xt_prediction_rows(
    model_case_rows_with_split: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    scaler: Mapping[str, Any],
    model: Mapping[str, Any],
    fixed_prediction_rows: Sequence[Mapping[str, Any]],
    global_wgs_prediction_rows: Sequence[Mapping[str, Any]],
    *,
    n_cells: int = 3,
    solver_fn: SolverFn = solve_wgs_for_alpha_xt,
) -> list[dict[str, Any]]:
    rows_by_case = {str(row.get("_model_case_id", row.get("case_id"))): row for row in model_case_rows_with_split}
    fixed = pd.DataFrame(fixed_prediction_rows)
    fixed = fixed[fixed["N_cells"].astype(int) == int(n_cells)]
    global_wgs = pd.DataFrame(global_wgs_prediction_rows)
    if not global_wgs.empty:
        global_wgs = global_wgs[global_wgs["N_cells"].astype(int) == int(n_cells)]
    fixed_lookup = {
        (str(row["case_id"]), str(row["target"])): row
        for row in fixed.to_dict("records")
    }
    global_lookup = {
        (str(row["case_id"]), str(row["target"])): row
        for row in global_wgs.to_dict("records")
    }
    rows: list[dict[str, Any]] = []
    for feature_row in feature_rows:
        case_id = str(feature_row["case_id"])
        alpha = predict_alpha_for_feature_row(model, feature_row, scaler)
        bound_hit = math.isclose(alpha, ALPHA_WGS_LOW, rel_tol=0.0, abs_tol=1.0e-10) or math.isclose(alpha, ALPHA_WGS_HIGH, rel_tol=0.0, abs_tol=1.0e-10)
        result = _safe_solve(solver_fn, rows_by_case[case_id], alpha, n_cells)
        values = _target_values(result)
        for target, (target_value, alpha_prediction) in values.items():
            fixed_row = fixed_lookup[(case_id, target)]
            global_row = global_lookup.get((case_id, target), {})
            fixed_prediction = float(fixed_row["y_pred"])
            rows.append(
                {
                    "case_id": case_id,
                    "phase": result.get("phase", feature_row.get("phase")),
                    "split": feature_row["split"],
                    "alpha_wgs": alpha,
                    "bound_hit": bound_hit,
                    "target": target,
                    "target_value": target_value,
                    "fixed_prediction": fixed_prediction,
                    "alpha_prediction": alpha_prediction,
                    "fixed_residual": fixed_prediction - target_value,
                    "alpha_residual": alpha_prediction - target_value,
                    "N_cells": int(n_cells),
                    "status": result.get("solve_status", "ok"),
                    "evidence_boundary": ALPHA_WGS_XT_EVIDENCE_BOUNDARY,
                    "dataset_id": result.get("dataset_id", feature_row.get("dataset_id")),
                    "split_role": feature_row.get("split_role"),
                    "global_wgs_prediction": float(global_row["y_pred"]) if global_row else math.nan,
                    "global_wgs_residual": float(global_row["residual"]) if global_row else math.nan,
                    "fallback_count": int(result.get("fallback_count", 0) or 0),
                    "poor_convergence_count": int(result.get("poor_convergence_count", 0) or 0),
                    "poor_convergence_cell_index": result.get("poor_convergence_cell_index", ""),
                    "cell0_cost": float(result.get("cell0_cost", math.nan)),
                }
            )
    rows.sort(key=lambda item: (str(item["split"]), str(item["case_id"]), str(item["target"])))
    return [{column: row.get(column) for column in ALPHA_WGS_XT_PREDICTION_COLUMNS} for row in rows]


def alpha_wgs_xt_metric_rows(prediction_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame(prediction_rows)
    if df.empty:
        return []
    rows: list[dict[str, Any]] = []
    for (split, target, n_cells), group in df.groupby(["split", "target", "N_cells"], dropna=False):
        fixed = group["fixed_residual"].astype(float).to_numpy()
        alpha = group["alpha_residual"].astype(float).to_numpy()
        global_resid = group["global_wgs_residual"].astype(float).to_numpy()
        alpha_values = group["alpha_wgs"].astype(float).to_numpy()
        fixed_rmse = float(np.sqrt(np.mean(fixed * fixed)))
        alpha_rmse = float(np.sqrt(np.mean(alpha * alpha)))
        global_rmse = float(np.sqrt(np.nanmean(global_resid * global_resid)))
        fixed_mae = float(np.mean(np.abs(fixed)))
        alpha_mae = float(np.mean(np.abs(alpha)))
        global_mae = float(np.nanmean(np.abs(global_resid)))
        fixed_bias = float(np.mean(fixed))
        alpha_bias = float(np.mean(alpha))
        rows.append(
            {
                "model_name": ALPHA_WGS_XT_MODEL_NAME,
                "split": split,
                "target": target,
                "N_cells": int(n_cells),
                "n": int(len(group)),
                "fixed_rmse": fixed_rmse,
                "alpha_rmse": alpha_rmse,
                "rmse_delta_vs_fixed": alpha_rmse - fixed_rmse,
                "global_wgs_rmse": global_rmse,
                "rmse_delta_vs_global_wgs": alpha_rmse - global_rmse,
                "fixed_mae": fixed_mae,
                "alpha_mae": alpha_mae,
                "mae_delta_vs_fixed": alpha_mae - fixed_mae,
                "global_wgs_mae": global_mae,
                "mae_delta_vs_global_wgs": alpha_mae - global_mae,
                "fixed_bias": fixed_bias,
                "alpha_bias": alpha_bias,
                "bias_delta_vs_fixed": alpha_bias - fixed_bias,
                "alpha_min": float(np.min(alpha_values)),
                "alpha_p05": float(np.quantile(alpha_values, 0.05)),
                "alpha_median": float(np.median(alpha_values)),
                "alpha_p95": float(np.quantile(alpha_values, 0.95)),
                "alpha_max": float(np.max(alpha_values)),
                "bound_hit_count": int(group["bound_hit"].astype(bool).sum()),
                "bound_hit_fraction": float(group["bound_hit"].astype(bool).mean()),
                "fallback_count_sum": int(group["fallback_count"].sum()),
                "poor_convergence_count_sum": int(group["poor_convergence_count"].sum()),
                "cell0_cost_min": float(group["cell0_cost"].min()),
                "cell0_cost_max": float(group["cell0_cost"].max()),
                "evidence_boundary": ALPHA_WGS_XT_EVIDENCE_BOUNDARY,
            }
        )
    rows.sort(key=lambda row: (str(row["split"]), str(row["target"]), int(row["N_cells"])))
    return [{column: row.get(column) for column in ALPHA_WGS_XT_METRIC_COLUMNS} for row in rows]


def build_alpha_wgs_xt_summary(
    selected_model: Mapping[str, Any],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    n_cells: int,
    regularization: float = DEFAULT_REGULARIZATION,
) -> list[dict[str, Any]]:
    initial = next(row for row in candidate_rows if row["candidate_name"] == "constant_alpha_1p0_initial")
    row = {
        "model_name": ALPHA_WGS_XT_MODEL_NAME,
        "alpha_name": ALPHA_WGS_NAME,
        "feature_set": ",".join(OPERATING_FEATURE_COLUMNS),
        "bounds": f"[{ALPHA_WGS_LOW},{ALPHA_WGS_HIGH}]",
        "regularization": float(regularization),
        "selection_or_training_method": "finite_candidate_train_only_ridge_logit_scaffold; one nonzero operating-feature slope or constant",
        "train_objective_initial": float(initial["train_objective"]),
        "train_objective_final": float(selected_model["train_objective"]),
        "train_objective_delta": float(selected_model["train_objective"]) - float(initial["train_objective"]),
        "selected_candidate_name": selected_model["candidate_name"],
        "selected_intercept": float(selected_model["intercept"]),
        "selected_nonzero_feature": selected_model["nonzero_feature"],
        "selected_nonzero_coefficient": float(selected_model["nonzero_coefficient"]),
        "validation_excluded_from_training": True,
        "test_excluded_from_training": True,
        "N_cells": int(n_cells),
        "candidate_count": len(candidate_rows),
        "status": "ok",
        "evidence_boundary": ALPHA_WGS_XT_EVIDENCE_BOUNDARY,
    }
    return [{column: row.get(column) for column in ALPHA_WGS_XT_SUMMARY_COLUMNS}]


def alpha_wgs_xt_audit_rows(
    scaler: Mapping[str, Any],
    selected_model: Mapping[str, Any],
    candidate_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature, mean, scale in zip(scaler["feature_columns"], scaler["mean"], scaler["scale"]):
        rows.append(
            {
                "record_type": "train_feature_scaler",
                "name": "scaler",
                "value": "",
                "feature": feature,
                "coefficient": math.nan,
                "mean": float(mean),
                "scale": float(scale),
                "fit_split": "train",
                "validation_excluded_from_training": True,
                "test_excluded_from_training": True,
                "status": "ok",
                "evidence_boundary": ALPHA_WGS_XT_EVIDENCE_BOUNDARY,
            }
        )
    for feature, coefficient in zip(scaler["feature_columns"], selected_model["coefficients"]):
        rows.append(
            {
                "record_type": "selected_model_coefficient",
                "name": selected_model["candidate_name"],
                "value": float(selected_model["intercept"]),
                "feature": feature,
                "coefficient": float(coefficient),
                "mean": math.nan,
                "scale": math.nan,
                "fit_split": "train",
                "validation_excluded_from_training": True,
                "test_excluded_from_training": True,
                "status": "ok",
                "evidence_boundary": ALPHA_WGS_XT_EVIDENCE_BOUNDARY,
            }
        )
    pred = pd.DataFrame(prediction_rows)
    if not pred.empty:
        by_case = pred.drop_duplicates(["case_id", "split"])
        for split, group in by_case.groupby("split"):
            rows.append(
                {
                    "record_type": "alpha_distribution",
                    "name": str(split),
                    "value": float(group["alpha_wgs"].median()),
                    "feature": "alpha_wgs",
                    "coefficient": math.nan,
                    "mean": float(group["alpha_wgs"].mean()),
                    "scale": float(group["alpha_wgs"].std(ddof=0)),
                    "fit_split": "train",
                    "validation_excluded_from_training": True,
                    "test_excluded_from_training": True,
                    "status": f"bound_hits={int(group['bound_hit'].astype(bool).sum())}/{len(group)}",
                    "evidence_boundary": ALPHA_WGS_XT_EVIDENCE_BOUNDARY,
                }
            )
    rows.append(
        {
            "record_type": "candidate_search",
            "name": "candidate_count",
            "value": len(candidate_rows),
            "feature": "",
            "coefficient": math.nan,
            "mean": math.nan,
            "scale": math.nan,
            "fit_split": "train",
            "validation_excluded_from_training": True,
            "test_excluded_from_training": True,
            "status": "finite_candidate_search",
            "evidence_boundary": ALPHA_WGS_XT_EVIDENCE_BOUNDARY,
        }
    )
    return [{column: row.get(column) for column in ALPHA_WGS_XT_AUDIT_COLUMNS} for row in rows]


def alpha_wgs_xt_markdown(
    summary_rows: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
    audit_rows: Sequence[Mapping[str, Any]],
) -> str:
    summary = summary_rows[0]
    lines = [
        "# Huayi alpha_wgs(x_t) minimal scaffold",
        "",
        "This diagnostic fits a tiny WGS-only bounded alpha scaffold on the frozen train split only.",
        "",
        f"- Model form: {summary['model_name']}",
        f"- Feature set: {summary['feature_set']}",
        f"- Bounds: {summary['bounds']}",
        f"- Selection/training method: {summary['selection_or_training_method']}",
        f"- Regularization: {summary['regularization']}",
        f"- Selected candidate: {summary['selected_candidate_name']}",
        f"- Selected nonzero feature: {summary['selected_nonzero_feature'] or 'none'}",
        f"- Train objective: {float(summary['train_objective_initial']):.6g} -> {float(summary['train_objective_final']):.6g}",
        f"- Validation excluded from training: {summary['validation_excluded_from_training']}",
        f"- Test excluded from training: {summary['test_excluded_from_training']}",
        "",
        "## Metrics",
        "",
        "| split | target | n | fixed RMSE | alpha RMSE | delta fixed | global WGS RMSE | delta global | alpha median | bound hits |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(
            "| {split} | {target} | {n} | {fixed:.6g} | {alpha:.6g} | {df:.6g} | {global_rmse:.6g} | {dg:.6g} | {median:.6g} | {hits} |".format(
                split=row["split"],
                target=row["target"],
                n=int(row["n"]),
                fixed=float(row["fixed_rmse"]),
                alpha=float(row["alpha_rmse"]),
                df=float(row["rmse_delta_vs_fixed"]),
                global_rmse=float(row["global_wgs_rmse"]),
                dg=float(row["rmse_delta_vs_global_wgs"]),
                median=float(row["alpha_median"]),
                hits=int(row["bound_hit_count"]),
            )
        )
    alpha_audit = [row for row in audit_rows if row["record_type"] == "alpha_distribution"]
    lines.extend(["", "## Alpha distribution", "", "| split | median | mean | std | status |", "|---|---:|---:|---:|---|"])
    for row in alpha_audit:
        lines.append(f"| {row['name']} | {float(row['value']):.6g} | {float(row['mean']):.6g} | {float(row['scale']):.6g} | {row['status']} |")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            f"Evidence boundary: {ALPHA_WGS_XT_EVIDENCE_BOUNDARY}.",
            "",
            "This scaffold can test whether a low-capacity bounded WGS alpha path is operational. It does not establish full hybrid calibration, learned multiplier interpretability, production control, or temporal generalization.",
        ]
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "ALPHA_WGS_XT_AUDIT_COLUMNS",
    "ALPHA_WGS_XT_METRIC_COLUMNS",
    "ALPHA_WGS_XT_PREDICTION_COLUMNS",
    "ALPHA_WGS_XT_SUMMARY_COLUMNS",
    "alpha_to_logit",
    "alpha_wgs_xt_audit_rows",
    "alpha_wgs_xt_markdown",
    "alpha_wgs_xt_metric_rows",
    "alpha_wgs_xt_prediction_rows",
    "build_alpha_wgs_xt_summary",
    "build_candidate_models",
    "feature_rows_with_split",
    "fit_train_feature_scaler",
    "logit_to_alpha",
    "predict_alpha_for_feature_row",
    "select_alpha_wgs_xt_model",
    "solve_wgs_for_alpha_xt",
]
