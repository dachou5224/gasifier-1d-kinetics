from __future__ import annotations

import math
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))


def _features():
    return [
        {
            "case_id": "train_1",
            "split": "train",
            "phase": "phase1",
            "dataset_id": "fixture",
            "split_role": "fit",
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
            "split": "train",
            "phase": "phase1",
            "dataset_id": "fixture",
            "split_role": "fit",
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
            "split": "validation",
            "phase": "phase3",
            "dataset_id": "fixture",
            "split_role": "holdout",
            "o_c_ratio": 99.0,
            "coal_flow_kg_s": 14.0,
            "o2_flow_kg_s": 12.0,
            "carrier_gas_flow_kg_s": 1.4,
            "carrier_to_coal_ratio": 0.1,
            "pressure_pa": 4.0e6,
            "T_in_K": 302.0,
        },
    ]


def _model_case_rows():
    return [
        {"case_id": row["case_id"], "_model_case_id": row["case_id"], "_split": row["split"], "_split_role": row["split_role"], "phase": row["phase"], "dataset_id": row["dataset_id"]}
        for row in _features()
    ]


def test_train_only_scaling_and_alpha_bounds():
    from model.alpha_wgs_xt_baseline import (
        ALPHA_WGS_HIGH,
        ALPHA_WGS_LOW,
        alpha_to_logit,
        build_candidate_models,
        fit_train_feature_scaler,
        logit_to_alpha,
        predict_alpha_for_feature_row,
    )

    scaler = fit_train_feature_scaler(_features())
    assert scaler["fit_split"] == "train"
    assert scaler["validation_excluded_from_training"] is True
    assert math.isclose(scaler["mean"][0], 0.8)
    assert scaler["mean"][0] != 99.0
    assert math.isclose(logit_to_alpha(alpha_to_logit(0.3)), 0.3)
    model = build_candidate_models(scaler)[0]
    alpha = predict_alpha_for_feature_row(model, _features()[2], scaler)
    assert ALPHA_WGS_LOW <= alpha <= ALPHA_WGS_HIGH


def _fake_solver(row, alpha, n_cells):
    if row["_split"] != "train":
        raise AssertionError("selection must not solve validation/test rows")
    return {
        "case_id": row["case_id"],
        "phase": row["phase"],
        "dataset_id": row["dataset_id"],
        "split": row["_split"],
        "split_role": row["_split_role"],
        "N_cells": n_cells,
        "CO_volpct_dry": 60.0,
        "H2_volpct_dry": 30.0,
        "CO2_volpct_dry": 10.0,
        "H2_CO_ratio": 0.5,
        "target_CO_volpct_dry": 60.0,
        "target_H2_volpct_dry": 30.0,
        "target_CO2_volpct_dry": 10.0,
        "fallback_count": 0,
        "poor_convergence_count": 0,
        "cell0_cost": 0.0,
        "solve_status": "ok",
    }


def test_selection_is_train_only_and_summary_has_required_schema():
    from model.alpha_wgs_xt_baseline import (
        ALPHA_WGS_XT_SUMMARY_COLUMNS,
        build_alpha_wgs_xt_summary,
        fit_train_feature_scaler,
        select_alpha_wgs_xt_model,
    )

    scaler = fit_train_feature_scaler(_features())
    selected, candidates = select_alpha_wgs_xt_model(_model_case_rows(), _features(), scaler, solver_fn=_fake_solver)
    summary = build_alpha_wgs_xt_summary(selected, candidates, n_cells=3)
    assert tuple(summary[0].keys()) == ALPHA_WGS_XT_SUMMARY_COLUMNS
    assert summary[0]["validation_excluded_from_training"] is True
    assert summary[0]["test_excluded_from_training"] is True
    assert summary[0]["candidate_count"] == len(candidates)
    assert "z_d" not in summary[0]["feature_set"]


def test_prediction_metric_and_audit_schema_from_fixture():
    from model.alpha_wgs_xt_baseline import (
        ALPHA_WGS_XT_AUDIT_COLUMNS,
        ALPHA_WGS_XT_METRIC_COLUMNS,
        ALPHA_WGS_XT_PREDICTION_COLUMNS,
        alpha_wgs_xt_audit_rows,
        alpha_wgs_xt_metric_rows,
        alpha_wgs_xt_prediction_rows,
        build_candidate_models,
        fit_train_feature_scaler,
    )

    def solver(row, alpha, n_cells):
        return {
            "case_id": row["case_id"],
            "phase": row["phase"],
            "dataset_id": row["dataset_id"],
            "split": row["_split"],
            "split_role": row["_split_role"],
            "N_cells": n_cells,
            "CO_volpct_dry": 61.0,
            "H2_volpct_dry": 29.0,
            "CO2_volpct_dry": 9.0,
            "H2_CO_ratio": 29.0 / 61.0,
            "target_CO_volpct_dry": 60.0,
            "target_H2_volpct_dry": 30.0,
            "target_CO2_volpct_dry": 10.0,
            "fallback_count": 0,
            "poor_convergence_count": 1,
            "poor_convergence_cell_index": "0",
            "cell0_cost": 0.01,
            "solve_status": "ok",
        }

    fixed = []
    global_rows = []
    for row in _features():
        for target, y_true, y_pred in [
            ("CO_volpct_dry", 60.0, 62.0),
            ("H2_volpct_dry", 30.0, 28.0),
            ("CO2_volpct_dry", 10.0, 11.0),
            ("H2_CO_ratio", 0.5, 28.0 / 62.0),
        ]:
            item = {"case_id": row["case_id"], "target": target, "N_cells": 3, "y_true": y_true, "y_pred": y_pred, "residual": y_pred - y_true}
            fixed.append(item)
            global_rows.append({**item, "y_pred": y_pred - 0.5, "residual": y_pred - 0.5 - y_true})
    scaler = fit_train_feature_scaler(_features())
    model = build_candidate_models(scaler)[1]
    predictions = alpha_wgs_xt_prediction_rows(_model_case_rows(), _features(), scaler, model, fixed, global_rows, solver_fn=solver)
    metrics = alpha_wgs_xt_metric_rows(predictions)
    audit = alpha_wgs_xt_audit_rows(scaler, model, [], predictions)
    assert tuple(predictions[0].keys()) == ALPHA_WGS_XT_PREDICTION_COLUMNS
    assert tuple(metrics[0].keys()) == ALPHA_WGS_XT_METRIC_COLUMNS
    assert tuple(audit[0].keys()) == ALPHA_WGS_XT_AUDIT_COLUMNS
    assert len(predictions) == 12
    assert metrics[0]["bound_hit_count"] == 0
