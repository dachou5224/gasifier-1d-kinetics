"""
Robustness and identifiability stress diagnostics for alpha_wgs(x_t).

The diagnostics reuse the round44 WGS-only finite-candidate scaffold.  They do
not add new model capacity, z_d, multi-alpha coupling, or neural networks.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .alpha_wgs_xt_baseline import (
    ALPHA_WGS_HIGH,
    ALPHA_WGS_LOW,
    ALPHA_WGS_XT_EVIDENCE_BOUNDARY,
    DEFAULT_REGULARIZATION,
    alpha_wgs_xt_metric_rows,
    alpha_wgs_xt_prediction_rows,
    build_candidate_models,
    fit_train_feature_scaler,
    predict_alpha_for_feature_row,
    select_alpha_wgs_xt_model,
    solve_wgs_for_alpha_xt,
)
from .direct_residual_baseline import OPERATING_FEATURE_COLUMNS

ROBUSTNESS_EVIDENCE_BOUNDARY = (
    "robustness and identifiability stress audit for minimal WGS-only "
    "alpha_wgs(x_t) scaffold; not full hybrid training, multi-alpha "
    "calibration, weak-label coal-quality modeling, learned multiplier "
    "interpretability proof, production control, or final temporal "
    "generalization evidence"
)

REGULARIZATION_GRID_COLUMNS: tuple[str, ...] = (
    "diagnostic",
    "regularization",
    "selected_candidate_name",
    "selected_nonzero_feature",
    "selected_nonzero_coefficient",
    "train_objective",
    "selection_objective",
    "split",
    "target",
    "N_cells",
    "n",
    "rmse_delta_vs_fixed",
    "rmse_delta_vs_global_wgs",
    "alpha_min",
    "alpha_median",
    "alpha_max",
    "bound_hit_count",
    "status",
    "evidence_boundary",
)

FEATURE_STABILITY_COLUMNS: tuple[str, ...] = (
    "diagnostic",
    "protocol_variant",
    "candidate_name",
    "selected_candidate_name",
    "nonzero_feature",
    "nonzero_coefficient",
    "train_objective",
    "selection_objective",
    "train_alpha_min",
    "train_alpha_max",
    "candidate_rank",
    "status",
    "evidence_boundary",
)

NULL_CONTROL_COLUMNS: tuple[str, ...] = (
    "diagnostic",
    "null_kind",
    "selected_candidate_name",
    "selected_nonzero_feature",
    "train_objective",
    "selection_objective",
    "train_alpha_min",
    "train_alpha_max",
    "validation_test_untouched",
    "status",
    "evidence_boundary",
)

SENSITIVITY_PROXY_COLUMNS: tuple[str, ...] = (
    "diagnostic",
    "case_id",
    "phase",
    "split",
    "target",
    "alpha_minus",
    "alpha_plus",
    "target_value",
    "prediction_minus",
    "prediction_plus",
    "fd_gradient",
    "gradient_sign",
    "N_cells",
    "status_minus",
    "status_plus",
    "evidence_boundary",
)


def _candidate_alpha_mse(candidate: Mapping[str, Any], feature_rows: Sequence[Mapping[str, Any]], scaler: Mapping[str, Any]) -> float:
    train = [row for row in feature_rows if row["split"] == "train"]
    alphas = np.array([predict_alpha_for_feature_row(candidate, row, scaler) for row in train], dtype=float)
    return float(np.mean((alphas - 1.0) ** 2))


def select_candidate_from_cached_rows(
    candidate_rows: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    scaler: Mapping[str, Any],
    *,
    regularization: float,
    allowed_features: set[str] | None = None,
) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    for row in candidate_rows:
        feature = str(row.get("nonzero_feature") or "")
        if allowed_features is not None and feature and feature not in allowed_features:
            continue
        if row.get("status") != "ok" or not math.isfinite(float(row["train_objective"])):
            continue
        item = dict(row)
        alpha_mse = _candidate_alpha_mse(item, feature_rows, scaler)
        item["alpha_mse_to_one"] = alpha_mse
        item["selection_objective"] = float(row["train_objective"]) + float(regularization) * alpha_mse
        item["regularization"] = float(regularization)
        scored.append(item)
    if not scored:
        raise RuntimeError("No candidate rows available for cached robustness selection")
    scored.sort(key=lambda item: (float(item["selection_objective"]), abs(float(item.get("nonzero_coefficient", 0.0)))))
    return scored[0]


def regularization_sweep_rows(
    candidate_rows: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    scaler: Mapping[str, Any],
    prediction_rows_by_candidate: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    regularization_grid: Sequence[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for penalty in regularization_grid:
        selected = select_candidate_from_cached_rows(candidate_rows, feature_rows, scaler, regularization=float(penalty))
        metrics = alpha_wgs_xt_metric_rows(prediction_rows_by_candidate[str(selected["candidate_name"])])
        for metric in metrics:
            rows.append(
                {
                    "diagnostic": "regularization_sweep",
                    "regularization": float(penalty),
                    "selected_candidate_name": selected["candidate_name"],
                    "selected_nonzero_feature": selected.get("nonzero_feature", ""),
                    "selected_nonzero_coefficient": float(selected.get("nonzero_coefficient", 0.0)),
                    "train_objective": float(selected["train_objective"]),
                    "selection_objective": float(selected["selection_objective"]),
                    "split": metric["split"],
                    "target": metric["target"],
                    "N_cells": int(metric["N_cells"]),
                    "n": int(metric["n"]),
                    "rmse_delta_vs_fixed": float(metric["rmse_delta_vs_fixed"]),
                    "rmse_delta_vs_global_wgs": float(metric["rmse_delta_vs_global_wgs"]),
                    "alpha_min": float(metric["alpha_min"]),
                    "alpha_median": float(metric["alpha_median"]),
                    "alpha_max": float(metric["alpha_max"]),
                    "bound_hit_count": int(metric["bound_hit_count"]),
                    "status": "ok",
                    "evidence_boundary": ROBUSTNESS_EVIDENCE_BOUNDARY,
                }
            )
    return [{column: row.get(column) for column in REGULARIZATION_GRID_COLUMNS} for row in rows]


def feature_stability_rows(
    candidate_rows: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    scaler: Mapping[str, Any],
    *,
    regularization: float = DEFAULT_REGULARIZATION,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scored: list[dict[str, Any]] = []
    for row in candidate_rows:
        if row.get("status") != "ok":
            continue
        item = dict(row)
        item["selection_objective"] = float(row["train_objective"]) + float(regularization) * _candidate_alpha_mse(item, feature_rows, scaler)
        scored.append(item)
    scored.sort(key=lambda item: float(item["selection_objective"]))
    selected_all = scored[0]
    for rank, item in enumerate(scored, start=1):
        rows.append(
            {
                "diagnostic": "single_feature_candidate_rank",
                "protocol_variant": "all_features_candidate_family",
                "candidate_name": item["candidate_name"],
                "selected_candidate_name": selected_all["candidate_name"],
                "nonzero_feature": item.get("nonzero_feature", ""),
                "nonzero_coefficient": float(item.get("nonzero_coefficient", 0.0)),
                "train_objective": float(item["train_objective"]),
                "selection_objective": float(item["selection_objective"]),
                "train_alpha_min": float(item["train_alpha_min"]),
                "train_alpha_max": float(item["train_alpha_max"]),
                "candidate_rank": rank,
                "status": "ok",
                "evidence_boundary": ROBUSTNESS_EVIDENCE_BOUNDARY,
            }
        )
    all_features = set(OPERATING_FEATURE_COLUMNS)
    for dropped in OPERATING_FEATURE_COLUMNS:
        selected = select_candidate_from_cached_rows(
            candidate_rows,
            feature_rows,
            scaler,
            regularization=regularization,
            allowed_features=all_features.difference({dropped}),
        )
        rows.append(
            {
                "diagnostic": "leave_one_feature_out_selection",
                "protocol_variant": f"drop_{dropped}",
                "candidate_name": "",
                "selected_candidate_name": selected["candidate_name"],
                "nonzero_feature": selected.get("nonzero_feature", ""),
                "nonzero_coefficient": float(selected.get("nonzero_coefficient", 0.0)),
                "train_objective": float(selected["train_objective"]),
                "selection_objective": float(selected["selection_objective"]),
                "train_alpha_min": float(selected["train_alpha_min"]),
                "train_alpha_max": float(selected["train_alpha_max"]),
                "candidate_rank": 1,
                "status": "ok",
                "evidence_boundary": ROBUSTNESS_EVIDENCE_BOUNDARY,
            }
        )
    return [{column: row.get(column) for column in FEATURE_STABILITY_COLUMNS} for row in rows]


def permute_train_feature_rows(feature_rows: Sequence[Mapping[str, Any]], *, feature: str = "carrier_to_coal_ratio") -> list[dict[str, Any]]:
    rows = [dict(row) for row in feature_rows]
    train_indices = [idx for idx, row in enumerate(rows) if row["split"] == "train"]
    values = [rows[idx][feature] for idx in train_indices]
    shifted = values[1:] + values[:1]
    for idx, value in zip(train_indices, shifted):
        rows[idx][feature] = value
    return rows


def null_control_rows(
    selected_null: Mapping[str, Any],
    *,
    null_kind: str = "deterministic_train_feature_permutation_carrier_to_coal_ratio",
) -> list[dict[str, Any]]:
    row = {
        "diagnostic": "null_control",
        "null_kind": null_kind,
        "selected_candidate_name": selected_null["candidate_name"],
        "selected_nonzero_feature": selected_null.get("nonzero_feature", ""),
        "train_objective": float(selected_null["train_objective"]),
        "selection_objective": float(selected_null["selection_objective"]),
        "train_alpha_min": float(selected_null["train_alpha_min"]),
        "train_alpha_max": float(selected_null["train_alpha_max"]),
        "validation_test_untouched": True,
        "status": selected_null.get("status", "ok"),
        "evidence_boundary": ROBUSTNESS_EVIDENCE_BOUNDARY,
    }
    return [{column: row.get(column) for column in NULL_CONTROL_COLUMNS}]


def _target_values(result: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    target_h2co = float(result["target_H2_volpct_dry"]) / max(float(result["target_CO_volpct_dry"]), 1.0e-12)
    return {
        "CO_volpct_dry": (float(result["target_CO_volpct_dry"]), float(result["CO_volpct_dry"])),
        "H2_volpct_dry": (float(result["target_H2_volpct_dry"]), float(result["H2_volpct_dry"])),
        "CO2_volpct_dry": (float(result["target_CO2_volpct_dry"]), float(result["CO2_volpct_dry"])),
        "H2_CO_ratio": (target_h2co, float(result["H2_CO_ratio"])),
    }


def sensitivity_proxy_rows(
    model_case_rows_with_split: Sequence[Mapping[str, Any]],
    *,
    alpha_center: float = 0.3,
    alpha_step: float = 0.03,
    n_cells: int = 3,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    alpha_minus = max(ALPHA_WGS_LOW, float(alpha_center) - float(alpha_step))
    alpha_plus = min(ALPHA_WGS_HIGH, float(alpha_center) + float(alpha_step))
    denom = alpha_plus - alpha_minus
    for row in model_case_rows_with_split:
        minus = solve_wgs_for_alpha_xt(row, alpha_minus, n_cells)
        plus = solve_wgs_for_alpha_xt(row, alpha_plus, n_cells)
        minus_values = _target_values(minus)
        plus_values = _target_values(plus)
        case_id = str(row.get("_model_case_id", row.get("case_id")))
        for target, (target_value, pred_minus) in minus_values.items():
            pred_plus = plus_values[target][1]
            gradient = (pred_plus - pred_minus) / denom
            rows.append(
                {
                    "diagnostic": "centered_fd_wgs_alpha_sensitivity",
                    "case_id": case_id,
                    "phase": row.get("phase"),
                    "split": row.get("_split", row.get("split")),
                    "target": target,
                    "alpha_minus": alpha_minus,
                    "alpha_plus": alpha_plus,
                    "target_value": target_value,
                    "prediction_minus": pred_minus,
                    "prediction_plus": pred_plus,
                    "fd_gradient": gradient,
                    "gradient_sign": "positive" if gradient > 0 else "negative" if gradient < 0 else "zero",
                    "N_cells": int(n_cells),
                    "status_minus": minus.get("solve_status", "ok"),
                    "status_plus": plus.get("solve_status", "ok"),
                    "evidence_boundary": ROBUSTNESS_EVIDENCE_BOUNDARY,
                }
            )
    return [{column: row.get(column) for column in SENSITIVITY_PROXY_COLUMNS} for row in rows]


def robustness_markdown(
    regularization_rows: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    null_rows: Sequence[Mapping[str, Any]],
    sensitivity_rows: Sequence[Mapping[str, Any]],
) -> str:
    reg = pd.DataFrame(regularization_rows)
    feat = pd.DataFrame(feature_rows)
    sens = pd.DataFrame(sensitivity_rows)
    lines = [
        "# Huayi alpha_wgs(x_t) robustness and identifiability stress audit",
        "",
        "This audit reuses the round44 WGS-only finite-candidate bounded-alpha scaffold.",
        "",
        "## Regularization sweep",
        "",
        "| regularization | selected candidate | held-out improved vs fixed | held-out improved vs global WGS | held-out mean delta vs fixed | held-out mean delta vs global |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for penalty, group in reg.groupby("regularization"):
        heldout = group[group["split"].isin(["validation", "test"])]
        lines.append(
            "| {penalty:.6g} | {candidate} | {fixed_improved}/{n} | {global_improved}/{n} | {fixed_delta:.6g} | {global_delta:.6g} |".format(
                penalty=float(penalty),
                candidate=group.iloc[0]["selected_candidate_name"],
                fixed_improved=int((heldout["rmse_delta_vs_fixed"] < 0.0).sum()),
                global_improved=int((heldout["rmse_delta_vs_global_wgs"] < 0.0).sum()),
                n=len(heldout),
                fixed_delta=float(heldout["rmse_delta_vs_fixed"].mean()),
                global_delta=float(heldout["rmse_delta_vs_global_wgs"].mean()),
            )
        )
    top = feat[feat["diagnostic"] == "single_feature_candidate_rank"].sort_values("candidate_rank").head(5)
    lines.extend(["", "## Feature stability", "", "| rank | candidate | feature | train objective | selection objective |", "|---:|---|---|---:|---:|"])
    for _, row in top.iterrows():
        lines.append(f"| {int(row['candidate_rank'])} | {row['candidate_name']} | {row['nonzero_feature'] or 'none'} | {float(row['train_objective']):.6g} | {float(row['selection_objective']):.6g} |")
    leave = feat[feat["diagnostic"] == "leave_one_feature_out_selection"]
    lines.extend(["", "Leave-one-feature-out selected candidates:"])
    for _, row in leave.iterrows():
        lines.append(f"- {row['protocol_variant']}: {row['selected_candidate_name']}")
    null = null_rows[0] if null_rows else {}
    lines.extend(
        [
            "",
            "## Null control",
            "",
            f"- Null kind: {null.get('null_kind', 'not_available')}",
            f"- Selected candidate under null: {null.get('selected_candidate_name', 'not_available')}",
            f"- Train objective under null: {float(null.get('train_objective', math.nan)):.6g}",
            "",
            "## WGS sensitivity proxy",
            "",
            "| target | positive | negative | zero | median fd gradient |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for target, group in sens.groupby("target"):
        lines.append(
            "| {target} | {pos} | {neg} | {zero} | {median:.6g} |".format(
                target=target,
                pos=int((group["fd_gradient"] > 0).sum()),
                neg=int((group["fd_gradient"] < 0).sum()),
                zero=int((group["fd_gradient"] == 0).sum()),
                median=float(group["fd_gradient"].median()),
            )
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            f"Evidence boundary: {ROBUSTNESS_EVIDENCE_BOUNDARY}.",
            "",
            "This audit is designed to stress stability and identifiability of a minimal scaffold. It does not establish learned multiplier interpretability, final temporal generalization, or production control.",
        ]
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "FEATURE_STABILITY_COLUMNS",
    "NULL_CONTROL_COLUMNS",
    "REGULARIZATION_GRID_COLUMNS",
    "SENSITIVITY_PROXY_COLUMNS",
    "feature_stability_rows",
    "null_control_rows",
    "permute_train_feature_rows",
    "regularization_sweep_rows",
    "robustness_markdown",
    "select_candidate_from_cached_rows",
    "sensitivity_proxy_rows",
]
