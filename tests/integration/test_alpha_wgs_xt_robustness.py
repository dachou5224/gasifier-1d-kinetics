from __future__ import annotations

import math
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))


def _features():
    rows = []
    for case_id, split, oc in [
        ("train_1", "train", 0.7),
        ("train_2", "train", 0.9),
        ("validation_1", "validation", 1.1),
        ("test_1", "test", 1.3),
    ]:
        rows.append(
            {
                "case_id": case_id,
                "split": split,
                "phase": "phase1",
                "dataset_id": "fixture",
                "split_role": "fit" if split == "train" else "holdout",
                "o_c_ratio": oc,
                "coal_flow_kg_s": 10.0 + oc,
                "o2_flow_kg_s": 8.0 + oc,
                "carrier_gas_flow_kg_s": 1.0,
                "carrier_to_coal_ratio": 0.1 + oc * 0.01,
                "pressure_pa": 4.0e6,
                "T_in_K": 300.0,
            }
        )
    return rows


def _candidate_rows():
    return [
        {
            "candidate_name": "constant_alpha_1p0_initial",
            "train_objective": 2.0,
            "selection_objective": 2.0,
            "regularization_penalty": 0.0,
            "train_alpha_min": 1.0,
            "train_alpha_max": 1.0,
            "status": "ok",
            "intercept": 0.0,
            "coefficients": (0.0,) * 7,
            "nonzero_feature": "",
            "nonzero_coefficient": 0.0,
        },
        {
            "candidate_name": "carrier_to_coal_ratio_neg_slope",
            "train_objective": 1.0,
            "selection_objective": 1.0,
            "regularization_penalty": 0.0,
            "train_alpha_min": 0.2,
            "train_alpha_max": 0.4,
            "status": "ok",
            "intercept": -1.8,
            "coefficients": (0.0, 0.0, 0.0, 0.0, -0.4, 0.0, 0.0),
            "nonzero_feature": "carrier_to_coal_ratio",
            "nonzero_coefficient": -0.4,
        },
    ]


def test_cached_regularization_and_feature_stability_schema():
    from model.alpha_wgs_xt_baseline import fit_train_feature_scaler
    from model.alpha_wgs_xt_robustness import (
        FEATURE_STABILITY_COLUMNS,
        REGULARIZATION_GRID_COLUMNS,
        feature_stability_rows,
        regularization_sweep_rows,
        select_candidate_from_cached_rows,
    )

    scaler = fit_train_feature_scaler(_features())
    selected = select_candidate_from_cached_rows(_candidate_rows(), _features(), scaler, regularization=0.001)
    assert selected["candidate_name"] == "carrier_to_coal_ratio_neg_slope"
    fake_predictions = {
        selected["candidate_name"]: [
            {
                "split": "train",
                "target": "CO_volpct_dry",
                "N_cells": 3,
                "fixed_residual": 2.0,
                "alpha_residual": 1.0,
                "global_wgs_residual": 1.1,
                "alpha_wgs": 0.3,
                "bound_hit": False,
                "fallback_count": 0,
                "poor_convergence_count": 0,
                "cell0_cost": 0.0,
            }
        ]
    }
    reg_rows = regularization_sweep_rows(_candidate_rows(), _features(), scaler, fake_predictions, regularization_grid=[0.001])
    feat_rows = feature_stability_rows(_candidate_rows(), _features(), scaler)
    assert tuple(reg_rows[0].keys()) == REGULARIZATION_GRID_COLUMNS
    assert tuple(feat_rows[0].keys()) == FEATURE_STABILITY_COLUMNS


def test_null_control_and_permutation_keep_holdout_rows_untouched():
    from model.alpha_wgs_xt_robustness import NULL_CONTROL_COLUMNS, null_control_rows, permute_train_feature_rows

    permuted = permute_train_feature_rows(_features(), feature="carrier_to_coal_ratio")
    assert [row["carrier_to_coal_ratio"] for row in permuted if row["split"] == "validation"] == [
        row["carrier_to_coal_ratio"] for row in _features() if row["split"] == "validation"
    ]
    null_rows = null_control_rows(_candidate_rows()[1])
    assert tuple(null_rows[0].keys()) == NULL_CONTROL_COLUMNS
    assert null_rows[0]["validation_test_untouched"] is True


def test_sensitivity_proxy_schema_with_fake_solver(monkeypatch):
    import model.alpha_wgs_xt_robustness as robustness
    from model.alpha_wgs_xt_robustness import SENSITIVITY_PROXY_COLUMNS

    def fake_solver(row, alpha, n_cells):
        return {
            "target_CO_volpct_dry": 60.0,
            "target_H2_volpct_dry": 30.0,
            "target_CO2_volpct_dry": 10.0,
            "CO_volpct_dry": 60.0 + alpha,
            "H2_volpct_dry": 30.0 - alpha,
            "CO2_volpct_dry": 10.0 + 0.5 * alpha,
            "H2_CO_ratio": (30.0 - alpha) / (60.0 + alpha),
            "solve_status": "ok",
        }

    monkeypatch.setattr(robustness, "solve_wgs_for_alpha_xt", fake_solver)
    rows = robustness.sensitivity_proxy_rows([
        {"case_id": "case_1", "_model_case_id": "case_1", "phase": "phase1", "_split": "train"}
    ])
    assert tuple(rows[0].keys()) == SENSITIVITY_PROXY_COLUMNS
    assert len(rows) == 4
    co = [row for row in rows if row["target"] == "CO_volpct_dry"][0]
    assert math.isclose(co["fd_gradient"], 1.0)
