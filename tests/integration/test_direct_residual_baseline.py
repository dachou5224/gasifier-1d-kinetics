from __future__ import annotations

import math
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))


def _fixed_rows(validation_residual: float = 100.0):
    rows = []
    split_rows = [
        ("train_1", "train", 11.0, 13.0),
        ("train_2", "train", 11.0, 15.0),
        ("validation_1", "validation", 11.0, 11.0 + validation_residual),
        ("test_1", "test", 11.0, 19.0),
    ]
    for case_id, split, y_true, y_pred in split_rows:
        rows.append(
            {
                "case_id": case_id,
                "phase": "phase1",
                "dataset_id": "fixture",
                "split": split,
                "split_role": "fixture_role",
                "N_cells": 3,
                "target": "CO_volpct_dry",
                "y_true": y_true,
                "y_pred": y_pred,
                "residual": y_pred - y_true,
                "fallback_count": 0,
                "poor_convergence_count": 0,
                "poor_convergence_cell_index": "",
                "cell0_cost": 0.0,
                "cell0_max_abs_residual": 0.0,
                "status": "ok",
                "evidence_boundary": "fixture",
            }
        )
    return rows


def _feature_rows():
    return [
        {
            "case_id": "train_1",
            "o_c_ratio": 0.7,
            "coal_flow_kg_s": 10.0,
            "o2_flow_kg_s": 8.0,
            "carrier_gas_flow_kg_s": 1.0,
            "carrier_to_coal_ratio": 0.1,
            "pressure_pa": 4.0e6,
            "T_in_K": 300.0,
        },
        {
            "case_id": "train_2",
            "o_c_ratio": 0.9,
            "coal_flow_kg_s": 12.0,
            "o2_flow_kg_s": 10.0,
            "carrier_gas_flow_kg_s": 1.2,
            "carrier_to_coal_ratio": 0.1,
            "pressure_pa": 4.0e6,
            "T_in_K": 301.0,
        },
        {
            "case_id": "validation_1",
            "o_c_ratio": 1.1,
            "coal_flow_kg_s": 14.0,
            "o2_flow_kg_s": 12.0,
            "carrier_gas_flow_kg_s": 1.4,
            "carrier_to_coal_ratio": 0.1,
            "pressure_pa": 4.0e6,
            "T_in_K": 302.0,
        },
        {
            "case_id": "test_1",
            "o_c_ratio": 1.3,
            "coal_flow_kg_s": 16.0,
            "o2_flow_kg_s": 14.0,
            "carrier_gas_flow_kg_s": 1.6,
            "carrier_to_coal_ratio": 0.1,
            "pressure_pa": 4.0e6,
            "T_in_K": 303.0,
        },
    ]


def test_split_guard_counts_exclusivity_and_future_holdout_policy():
    from model.direct_residual_baseline import validate_split_for_direct_residual
    from model.split_global_alpha_baseline import build_split_manifest

    contract_rows = []
    phases = ["phase1"] * 6 + ["phase2"] * 6 + ["phase3"] * 6 + ["phase4"] * 4
    for idx, phase in enumerate(phases, start=1):
        contract_rows.append(
            {
                "case_id": f"case_{idx:02d}",
                "phase": phase,
                "dataset_id": f"dataset_{phase}",
                "independence_group": f"IG{idx:03d}",
            }
        )
    split_rows = build_split_manifest(contract_rows)
    validation = validate_split_for_direct_residual(split_rows)
    assert validation["has_duplicate_case_id"] is False
    assert validation["expected_counts_match"] is True
    assert validation["split_counts"] == {"train": 12, "validation": 6, "test": 4}
    assert validation["fit_split"] == "train"
    assert validation["validation_excluded_from_fit"] is True
    assert validation["test_excluded_from_fit"] is True


def test_constant_baseline_uses_train_only_residual_mean():
    from model.direct_residual_baseline import (
        CONSTANT_VARIANT,
        DIRECT_RESIDUAL_PREDICTION_COLUMNS,
        direct_residual_prediction_rows,
        fit_direct_residual_models,
    )

    models = fit_direct_residual_models(_fixed_rows(validation_residual=100.0), _feature_rows())
    constant = [row for row in models if row["baseline_variant"] == CONSTANT_VARIANT][0]
    assert constant["fit_split"] == "train"
    assert math.isclose(constant["correction"], 3.0)
    assert constant["validation_excluded_from_fit"] is True
    assert constant["test_excluded_from_fit"] is True

    predictions = direct_residual_prediction_rows(_fixed_rows(validation_residual=100.0), _feature_rows(), [constant])
    assert tuple(predictions[0].keys()) == DIRECT_RESIDUAL_PREDICTION_COLUMNS
    validation = [row for row in predictions if row["split"] == "validation"][0]
    assert math.isclose(validation["corrected_prediction"], 108.0)
    assert math.isclose(validation["corrected_residual"], 97.0)


def test_validation_residual_does_not_change_train_only_fit():
    from model.direct_residual_baseline import CONSTANT_VARIANT, fit_direct_residual_models

    models_a = fit_direct_residual_models(_fixed_rows(validation_residual=100.0), _feature_rows())
    models_b = fit_direct_residual_models(_fixed_rows(validation_residual=-500.0), _feature_rows())
    constant_a = [row for row in models_a if row["baseline_variant"] == CONSTANT_VARIANT][0]
    constant_b = [row for row in models_b if row["baseline_variant"] == CONSTANT_VARIANT][0]
    assert math.isclose(constant_a["correction"], constant_b["correction"])


def test_linear_baseline_records_train_only_scaling_and_metrics_schema():
    from model.direct_residual_baseline import (
        DIRECT_RESIDUAL_METRIC_COLUMNS,
        DIRECT_RESIDUAL_MODEL_COLUMNS,
        LINEAR_VARIANT,
        direct_residual_metric_rows,
        direct_residual_model_audit_rows,
        direct_residual_prediction_rows,
        fit_direct_residual_models,
    )

    models = fit_direct_residual_models(_fixed_rows(), _feature_rows(), ridge_lambda=1.0e-3)
    linear = [row for row in models if row["baseline_variant"] == LINEAR_VARIANT][0]
    assert linear["preprocessing_fit_split"] == "train"
    assert linear["validation_excluded_from_fit"] is True
    assert linear["test_excluded_from_fit"] is True
    assert len(linear["feature_mean"]) == 7
    assert math.isclose(linear["feature_mean"][0], 0.8)
    assert linear["status"].startswith("exploratory_fit_")

    predictions = direct_residual_prediction_rows(_fixed_rows(), _feature_rows(), models)
    metrics = direct_residual_metric_rows(predictions)
    model_rows = direct_residual_model_audit_rows(models)
    assert tuple(metrics[0].keys()) == DIRECT_RESIDUAL_METRIC_COLUMNS
    assert tuple(model_rows[0].keys()) == DIRECT_RESIDUAL_MODEL_COLUMNS
    assert {row["baseline_variant"] for row in metrics} == {"residual_constant_train_mean", "residual_linear_operating"}
