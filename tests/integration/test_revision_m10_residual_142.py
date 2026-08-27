from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from model.revision_m10_residual_142 import (
    ALL_TARGETS,
    CONSTANT_VARIANT,
    EFFECT_COLUMNS,
    LINEAR_VARIANT,
    PRIMARY_TARGETS,
    apply_constant_patch,
    effect_matrix,
)


def _fixed_frame(validation_residual: float = 100.0) -> pd.DataFrame:
    rows = []
    specs = [
        ("W1", "train", 10.0, 13.0),
        ("W2", "train", 10.0, 15.0),
        ("W3", "validation", 10.0, 10.0 + validation_residual),
        ("W4", "test", 10.0, 19.0),
    ]
    for window_id, split_role, y_true, y_pred in specs:
        for target in ALL_TARGETS:
            rows.append(
                {
                    "dataset_id": "R52-C-75",
                    "window_id": window_id,
                    "phase": 1,
                    "split_role": split_role,
                    "method": "fixed_physics",
                    "configuration": "reference",
                    "N_cells": 1,
                    "target": target,
                    "target_value": y_true,
                    "predicted_value": y_pred,
                    "residual": y_pred - y_true,
                    "execution_status": "completed",
                    "residual_quality_status": "poor",
                    "cell0_cost": 0.01,
                    "fallback_count": 0,
                    "poor_convergence_count": 1,
                }
            )
    return pd.DataFrame(rows)


def test_constant_fit_uses_train_only():
    a, corr_a = apply_constant_patch(_fixed_frame(100.0))
    b, corr_b = apply_constant_patch(_fixed_frame(-500.0))
    assert corr_a == corr_b
    assert all(abs(corr_a[target] - 4.0) < 1e-12 for target in ALL_TARGETS)
    val = a[(a["split_role"] == "validation") & (a["target"] == "CO_dry_vol_pct")].iloc[0]
    assert abs(val["y_residual"] - 106.0) < 1e-12
    assert bool(val["validation_excluded_from_fit"]) is True
    assert bool(val["test_excluded_from_fit"]) is True


def test_effect_matrix_uses_injected_table4_rmse_fixed():
    patched, _ = apply_constant_patch(_fixed_frame())
    patched["method"] = CONSTANT_VARIANT
    rmse_fixed = []
    for split_role in ("train", "validation", "test"):
        for target in ALL_TARGETS:
            group = patched[(patched["split_role"] == split_role) & (patched["target"] == target)]
            residual = (group["y_fixed"] - group["y_plant"]).to_numpy()
            rmse_fixed.append(
                {
                    "split_role": split_role,
                    "target": target,
                    "rmse_fixed": float((residual * residual).mean() ** 0.5),
                }
            )
    table = pd.DataFrame(rmse_fixed)
    matrix = effect_matrix(patched, table, method=CONSTANT_VARIANT, source_commit_hash="deadbeef")
    assert tuple(matrix.columns) == EFFECT_COLUMNS
    assert set(matrix["method"]) == {CONSTANT_VARIANT}
    test_co = matrix[(matrix["split_role"] == "test") & (matrix["target"] == "CO_dry_vol_pct")].iloc[0]
    assert test_co["n_completed"] == 1
    assert test_co["execution_status"] == "completed"


def test_primary_form_is_predeclared_and_linear_is_exploratory():
    assert CONSTANT_VARIANT == "residual_constant_train_mean"
    assert LINEAR_VARIANT == "residual_linear_operating"
    assert PRIMARY_TARGETS == ("CO_dry_vol_pct", "CO2_dry_vol_pct", "H2_dry_vol_pct", "H2_CO_ratio")
    assert LINEAR_VARIANT != CONSTANT_VARIANT
