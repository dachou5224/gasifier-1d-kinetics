"""
Direct outlet residual correction baselines for Huayi split diagnostics.

These baselines deliberately patch outlet residuals outside the physics model.
They are included as black-box comparators against bounded reaction-family
corrections, not as mechanistic hybrid models.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .model_case_adapter import build_gasifier_inputs_from_model_case

DIRECT_RESIDUAL_EVIDENCE_BOUNDARY = (
    "direct outlet residual correction comparator only; black-box outlet "
    "patching fit on train split only; not trained physics-guided hybrid, "
    "bounded alpha_j result, learned multiplier interpretability, weak-label "
    "proof, production control, final temporal generalization, or "
    "Yulin/slurry generalization"
)

CONSTANT_VARIANT = "residual_constant_train_mean"
LINEAR_VARIANT = "residual_linear_operating"

OPERATING_FEATURE_COLUMNS: tuple[str, ...] = (
    "o_c_ratio",
    "coal_flow_kg_s",
    "o2_flow_kg_s",
    "carrier_gas_flow_kg_s",
    "carrier_to_coal_ratio",
    "pressure_pa",
    "T_in_K",
)

DIRECT_RESIDUAL_PREDICTION_COLUMNS: tuple[str, ...] = (
    "case_id",
    "phase",
    "split",
    "target",
    "target_value",
    "fixed_prediction",
    "fixed_residual",
    "corrected_prediction",
    "corrected_residual",
    "baseline_variant",
    "N_cells",
    "fit_split",
    "status",
    "evidence_boundary",
    "dataset_id",
    "split_role",
    "correction_value",
    "feature_set",
    "preprocessing_fit_split",
    "validation_excluded_from_fit",
    "test_excluded_from_fit",
)

DIRECT_RESIDUAL_METRIC_COLUMNS: tuple[str, ...] = (
    "baseline_variant",
    "split",
    "target",
    "N_cells",
    "n",
    "fixed_rmse",
    "corrected_rmse",
    "rmse_delta",
    "fixed_mae",
    "corrected_mae",
    "mae_delta",
    "fixed_bias",
    "corrected_bias",
    "bias_delta",
    "fit_split",
    "evidence_boundary",
    "fallback_count_sum",
    "poor_convergence_count_sum",
    "cell0_cost_min",
    "cell0_cost_max",
)

DIRECT_RESIDUAL_MODEL_COLUMNS: tuple[str, ...] = (
    "baseline_variant",
    "target",
    "N_cells",
    "fit_split",
    "train_n",
    "feature_set",
    "ridge_lambda",
    "correction_or_intercept",
    "preprocessing_fit_split",
    "validation_excluded_from_fit",
    "test_excluded_from_fit",
    "status",
    "evidence_boundary",
)


def _case_id(row: Mapping[str, Any]) -> str:
    return str(row.get("case_id", build_gasifier_inputs_from_model_case(row)["case_id"]))


def _feature_row(row: Mapping[str, Any]) -> dict[str, Any]:
    built = build_gasifier_inputs_from_model_case(row)
    case_id = str(built["case_id"])
    op_conds = built["op_conds"]
    coal_props = built["coal_props"]
    coal_flow = float(op_conds["coal_flow"])
    o2_flow = float(op_conds["o2_flow"])
    carrier_flow = float(op_conds.get("carrier_gas_flow", 0.0))
    carbon_mass_fraction = float(coal_props["Cd"]) / 100.0
    carbon_kmol_s = coal_flow * carbon_mass_fraction / 12.011
    o2_kmol_s = o2_flow / 31.998
    o_c_ratio = (2.0 * o2_kmol_s) / max(carbon_kmol_s, 1.0e-12)
    return {
        "case_id": case_id,
        "o_c_ratio": o_c_ratio,
        "coal_flow_kg_s": coal_flow,
        "o2_flow_kg_s": o2_flow,
        "carrier_gas_flow_kg_s": carrier_flow,
        "carrier_to_coal_ratio": carrier_flow / max(coal_flow, 1.0e-12),
        "pressure_pa": float(op_conds["P"]),
        "T_in_K": float(op_conds["T_in"]),
    }


def operating_feature_rows(model_case_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [_feature_row(row) for row in model_case_rows]
    seen: set[str] = set()
    duplicates: set[str] = set()
    for row in rows:
        case_id = str(row["case_id"])
        if case_id in seen:
            duplicates.add(case_id)
        seen.add(case_id)
    if duplicates:
        raise ValueError(f"Duplicate model case IDs in feature rows: {sorted(duplicates)}")
    return rows


def validate_split_for_direct_residual(split_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    from .split_global_alpha_baseline import validate_split_manifest

    validation = dict(validate_split_manifest(split_rows))
    validation["validation_excluded_from_fit"] = True
    validation["test_excluded_from_fit"] = True
    validation["fit_split"] = "train"
    validation["evidence_boundary"] = DIRECT_RESIDUAL_EVIDENCE_BOUNDARY
    return validation


def _fixed_prediction_df(fixed_prediction_rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(fixed_prediction_rows).copy()
    if df.empty:
        raise ValueError("fixed prediction rows are empty")
    rename = {"y_true": "target_value", "y_pred": "fixed_prediction", "residual": "fixed_residual"}
    df = df.rename(columns=rename)
    required = {
        "case_id",
        "phase",
        "dataset_id",
        "split",
        "split_role",
        "N_cells",
        "target",
        "target_value",
        "fixed_prediction",
        "fixed_residual",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing fixed prediction columns: {sorted(missing)}")
    for column in ("N_cells", "target_value", "fixed_prediction", "fixed_residual"):
        df[column] = pd.to_numeric(df[column])
    return df


def _feature_df(feature_rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(feature_rows).copy()
    if df.empty:
        raise ValueError("operating feature rows are empty")
    missing = {"case_id", *OPERATING_FEATURE_COLUMNS}.difference(df.columns)
    if missing:
        raise ValueError(f"Missing operating feature columns: {sorted(missing)}")
    for column in OPERATING_FEATURE_COLUMNS:
        df[column] = pd.to_numeric(df[column])
    return df


def fit_direct_residual_models(
    fixed_prediction_rows: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    *,
    ridge_lambda: float = 1.0e-3,
) -> list[dict[str, Any]]:
    fixed_df = _fixed_prediction_df(fixed_prediction_rows)
    features = _feature_df(feature_rows)
    joined = fixed_df.merge(features, on="case_id", how="left", validate="many_to_one")
    if joined[list(OPERATING_FEATURE_COLUMNS)].isna().any().any():
        raise ValueError("Missing operating features after joining fixed predictions")

    models: list[dict[str, Any]] = []
    group_cols = ["N_cells", "target"]
    for (n_cells, target), group in joined.groupby(group_cols, dropna=False):
        train = group[group["split"] == "train"].copy()
        if train.empty:
            raise ValueError(f"No train rows for target={target} N_cells={n_cells}")
        residual = train["fixed_residual"].astype(float).to_numpy()
        constant = float(np.mean(residual))
        models.append(
            {
                "baseline_variant": CONSTANT_VARIANT,
                "target": str(target),
                "N_cells": int(n_cells),
                "fit_split": "train",
                "train_n": int(len(train)),
                "feature_set": "none",
                "ridge_lambda": math.nan,
                "correction": constant,
                "intercept": constant,
                "feature_columns": (),
                "feature_mean": (),
                "feature_scale": (),
                "coefficients": (),
                "preprocessing_fit_split": "train",
                "validation_excluded_from_fit": True,
                "test_excluded_from_fit": True,
                "status": "ok",
                "evidence_boundary": DIRECT_RESIDUAL_EVIDENCE_BOUNDARY,
            }
        )

        x_train = train[list(OPERATING_FEATURE_COLUMNS)].astype(float).to_numpy()
        mean = np.mean(x_train, axis=0)
        scale = np.std(x_train, axis=0)
        scale = np.where(scale > 0.0, scale, 1.0)
        x_scaled = (x_train - mean) / scale
        design = np.column_stack([np.ones(len(x_scaled)), x_scaled])
        penalty = np.eye(design.shape[1]) * float(ridge_lambda)
        penalty[0, 0] = 0.0
        try:
            beta = np.linalg.solve(design.T @ design + penalty, design.T @ residual)
            status = "exploratory_fit_ok"
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(design.T @ design + penalty, design.T @ residual, rcond=None)[0]
            status = "exploratory_fit_lstsq_fallback"
        models.append(
            {
                "baseline_variant": LINEAR_VARIANT,
                "target": str(target),
                "N_cells": int(n_cells),
                "fit_split": "train",
                "train_n": int(len(train)),
                "feature_set": ",".join(OPERATING_FEATURE_COLUMNS),
                "ridge_lambda": float(ridge_lambda),
                "correction": math.nan,
                "intercept": float(beta[0]),
                "feature_columns": tuple(OPERATING_FEATURE_COLUMNS),
                "feature_mean": tuple(float(v) for v in mean),
                "feature_scale": tuple(float(v) for v in scale),
                "coefficients": tuple(float(v) for v in beta[1:]),
                "preprocessing_fit_split": "train",
                "validation_excluded_from_fit": True,
                "test_excluded_from_fit": True,
                "status": status,
                "evidence_boundary": DIRECT_RESIDUAL_EVIDENCE_BOUNDARY,
            }
        )
    return models


def _model_correction(model: Mapping[str, Any], row: Mapping[str, Any]) -> float:
    if model["baseline_variant"] == CONSTANT_VARIANT:
        return float(model["correction"])
    values = np.array([float(row[column]) for column in model["feature_columns"]], dtype=float)
    mean = np.array(model["feature_mean"], dtype=float)
    scale = np.array(model["feature_scale"], dtype=float)
    coeff = np.array(model["coefficients"], dtype=float)
    return float(model["intercept"] + ((values - mean) / scale) @ coeff)


def direct_residual_prediction_rows(
    fixed_prediction_rows: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    models: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    fixed_df = _fixed_prediction_df(fixed_prediction_rows)
    features = _feature_df(feature_rows)
    joined = fixed_df.merge(features, on="case_id", how="left", validate="many_to_one")
    rows: list[dict[str, Any]] = []
    for model in models:
        subset = joined[
            (joined["target"].astype(str) == str(model["target"]))
            & (joined["N_cells"].astype(int) == int(model["N_cells"]))
        ]
        for _, item in subset.iterrows():
            correction = _model_correction(model, item)
            corrected_prediction = float(item["fixed_prediction"]) - correction
            target_value = float(item["target_value"])
            rows.append(
                {
                    "case_id": item["case_id"],
                    "phase": item["phase"],
                    "split": item["split"],
                    "target": item["target"],
                    "target_value": target_value,
                    "fixed_prediction": float(item["fixed_prediction"]),
                    "fixed_residual": float(item["fixed_residual"]),
                    "corrected_prediction": corrected_prediction,
                    "corrected_residual": corrected_prediction - target_value,
                    "baseline_variant": model["baseline_variant"],
                    "N_cells": int(item["N_cells"]),
                    "fit_split": "train",
                    "status": model["status"],
                    "evidence_boundary": DIRECT_RESIDUAL_EVIDENCE_BOUNDARY,
                    "dataset_id": item.get("dataset_id", ""),
                    "split_role": item.get("split_role", ""),
                    "correction_value": correction,
                    "feature_set": model["feature_set"],
                    "preprocessing_fit_split": model["preprocessing_fit_split"],
                    "validation_excluded_from_fit": True,
                    "test_excluded_from_fit": True,
                }
            )
    rows.sort(key=lambda row: (str(row["baseline_variant"]), int(row["N_cells"]), str(row["split"]), str(row["case_id"]), str(row["target"])))
    return [{column: row.get(column) for column in DIRECT_RESIDUAL_PREDICTION_COLUMNS} for row in rows]


def direct_residual_metric_rows(prediction_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame(prediction_rows)
    if df.empty:
        return []
    rows: list[dict[str, Any]] = []
    group_cols = ["baseline_variant", "split", "target", "N_cells"]
    for (variant, split, target, n_cells), group in df.groupby(group_cols, dropna=False):
        fixed = group["fixed_residual"].astype(float).to_numpy()
        corrected = group["corrected_residual"].astype(float).to_numpy()
        fixed_rmse = float(np.sqrt(np.mean(fixed * fixed)))
        corrected_rmse = float(np.sqrt(np.mean(corrected * corrected)))
        fixed_mae = float(np.mean(np.abs(fixed)))
        corrected_mae = float(np.mean(np.abs(corrected)))
        fixed_bias = float(np.mean(fixed))
        corrected_bias = float(np.mean(corrected))
        rows.append(
            {
                "baseline_variant": variant,
                "split": split,
                "target": target,
                "N_cells": int(n_cells),
                "n": int(len(group)),
                "fixed_rmse": fixed_rmse,
                "corrected_rmse": corrected_rmse,
                "rmse_delta": corrected_rmse - fixed_rmse,
                "fixed_mae": fixed_mae,
                "corrected_mae": corrected_mae,
                "mae_delta": corrected_mae - fixed_mae,
                "fixed_bias": fixed_bias,
                "corrected_bias": corrected_bias,
                "bias_delta": corrected_bias - fixed_bias,
                "fit_split": "train",
                "evidence_boundary": DIRECT_RESIDUAL_EVIDENCE_BOUNDARY,
                "fallback_count_sum": math.nan,
                "poor_convergence_count_sum": math.nan,
                "cell0_cost_min": math.nan,
                "cell0_cost_max": math.nan,
            }
        )
    rows.sort(key=lambda row: (str(row["baseline_variant"]), int(row["N_cells"]), str(row["split"]), str(row["target"])))
    return [{column: row.get(column) for column in DIRECT_RESIDUAL_METRIC_COLUMNS} for row in rows]


def direct_residual_model_audit_rows(models: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in models:
        rows.append(
            {
                "baseline_variant": model["baseline_variant"],
                "target": model["target"],
                "N_cells": int(model["N_cells"]),
                "fit_split": model["fit_split"],
                "train_n": int(model["train_n"]),
                "feature_set": model["feature_set"],
                "ridge_lambda": model["ridge_lambda"],
                "correction_or_intercept": model["intercept"],
                "preprocessing_fit_split": model["preprocessing_fit_split"],
                "validation_excluded_from_fit": bool(model["validation_excluded_from_fit"]),
                "test_excluded_from_fit": bool(model["test_excluded_from_fit"]),
                "status": model["status"],
                "evidence_boundary": DIRECT_RESIDUAL_EVIDENCE_BOUNDARY,
            }
        )
    rows.sort(key=lambda row: (str(row["baseline_variant"]), int(row["N_cells"]), str(row["target"])))
    return [{column: row.get(column) for column in DIRECT_RESIDUAL_MODEL_COLUMNS} for row in rows]


def direct_residual_markdown(
    prediction_rows: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
    model_rows: Sequence[Mapping[str, Any]],
    split_validation: Mapping[str, Any],
) -> str:
    lines = [
        "# Huayi direct outlet residual baseline",
        "",
        "This artifact fits black-box outlet residual corrections on the frozen Huayi chronological train split only.",
        "Validation and test rows are transform-only; they are not used for fitting or feature scaling.",
        "",
        f"- Split counts: {split_validation['split_counts']}",
        f"- Duplicate case IDs: {split_validation['has_duplicate_case_id']}",
        f"- Expected 12/6/4 counts match: {split_validation['expected_counts_match']}",
        f"- Validation excluded from fit/preprocessing: {split_validation['validation_excluded_from_fit']}",
        f"- Test excluded from fit/preprocessing: {split_validation['test_excluded_from_fit']}",
        f"- Variants: {CONSTANT_VARIANT}, {LINEAR_VARIANT}",
        f"- Operating features for linear variant: {', '.join(OPERATING_FEATURE_COLUMNS)}",
        "",
        "## Model audit",
        "",
        "| variant | N | target | train_n | fit_split | preprocessing | status |",
        "|---|---:|---|---:|---|---|---|",
    ]
    for row in model_rows:
        lines.append(
            f"| {row['baseline_variant']} | {int(row['N_cells'])} | {row['target']} | {int(row['train_n'])} | {row['fit_split']} | {row['preprocessing_fit_split']} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| variant | split | N | target | n | fixed RMSE | corrected RMSE | delta | fixed MAE | corrected MAE |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metric_rows:
        lines.append(
            "| {variant} | {split} | {n_cells} | {target} | {n} | {fixed_rmse:.6g} | {corrected_rmse:.6g} | {rmse_delta:.6g} | {fixed_mae:.6g} | {corrected_mae:.6g} |".format(
                variant=row["baseline_variant"],
                split=row["split"],
                n_cells=int(row["N_cells"]),
                target=row["target"],
                n=int(row["n"]),
                fixed_rmse=float(row["fixed_rmse"]),
                corrected_rmse=float(row["corrected_rmse"]),
                rmse_delta=float(row["rmse_delta"]),
                fixed_mae=float(row["fixed_mae"]),
                corrected_mae=float(row["corrected_mae"]),
            )
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            f"Evidence boundary: {DIRECT_RESIDUAL_EVIDENCE_BOUNDARY}.",
            "",
            "This comparator can quantify apparent outlet-error reduction from direct residual patching. It does not identify reaction mechanisms, train alpha_j(x_t), prove temporal generalization, or support production control.",
        ]
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "CONSTANT_VARIANT",
    "DIRECT_RESIDUAL_EVIDENCE_BOUNDARY",
    "DIRECT_RESIDUAL_METRIC_COLUMNS",
    "DIRECT_RESIDUAL_MODEL_COLUMNS",
    "DIRECT_RESIDUAL_PREDICTION_COLUMNS",
    "LINEAR_VARIANT",
    "OPERATING_FEATURE_COLUMNS",
    "direct_residual_markdown",
    "direct_residual_metric_rows",
    "direct_residual_model_audit_rows",
    "direct_residual_prediction_rows",
    "fit_direct_residual_models",
    "operating_feature_rows",
    "validate_split_for_direct_residual",
]
