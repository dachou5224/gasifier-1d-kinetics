from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))


def test_subset_names_bounds_and_volatile_mapping_are_explicit():
    from model.global_multialpha_diagnostic import ALPHA_BOUNDS, SUBSETS, explicit_volatile_mapping_note

    names = {"none" if not subset else "+".join(subset) for subset in SUBSETS}
    assert {
        "alpha_wgs",
        "alpha_char_oxidation",
        "alpha_volatile_oxidation",
        "alpha_wgs+alpha_char_oxidation",
        "alpha_wgs+alpha_volatile_oxidation",
        "alpha_char_oxidation+alpha_volatile_oxidation",
        "alpha_wgs+alpha_char_oxidation+alpha_volatile_oxidation",
    } <= names
    assert ALPHA_BOUNDS["alpha_wgs"] == (0.02, 2.0)
    assert "CO_OxidationFactor" in explicit_volatile_mapping_note()
    assert "H2_OxidationFactor" in explicit_volatile_mapping_note()


def test_apply_alpha_values_is_non_mutating_and_maps_shared_volatile():
    from model.global_multialpha_diagnostic import apply_alpha_values

    op_conds = {"coal_flow": 1.0}
    out = apply_alpha_values(
        op_conds,
        {
            "alpha_wgs": 0.3,
            "alpha_char_oxidation": 0.1,
            "alpha_volatile_oxidation": 0.2,
        },
    )
    assert op_conds == {"coal_flow": 1.0}
    assert out["WGS_CatalyticFactor"] == 0.3
    assert out["CharCombustionRateFactor"] == 0.1
    assert out["CO_OxidationFactor"] == 0.2
    assert out["H2_OxidationFactor"] == 0.2


def test_selection_prediction_metric_schemas_with_fake_solver():
    from model.global_multialpha_diagnostic import (
        GRID_COLUMNS,
        METRIC_COLUMNS,
        PREDICTION_COLUMNS,
        SUMMARY_COLUMNS,
        global_multialpha_metric_rows,
        global_multialpha_prediction_rows,
        select_global_multialpha_subsets,
    )

    def solver(row, alpha_values, n_cells):
        alpha_wgs = float(alpha_values["alpha_wgs"])
        return {
            "case_id": row["_model_case_id"],
            "phase": row["phase"],
            "split": row["_split"],
            "target_CO_volpct_dry": 60.0,
            "target_H2_volpct_dry": 30.0,
            "target_CO2_volpct_dry": 10.0,
            "target_H2_CO_ratio": 0.5,
            "CO_volpct_dry": 60.0 + alpha_wgs,
            "H2_volpct_dry": 30.0 - alpha_wgs,
            "CO2_volpct_dry": 10.0,
            "H2_CO_ratio": (30.0 - alpha_wgs) / (60.0 + alpha_wgs),
            "objective": alpha_wgs,
            "fallback_count": 0,
            "poor_convergence_count": 1,
            "poor_convergence_cell_index": "0",
            "cell0_cost": 0.01,
            "solve_status": "ok",
        }

    model_rows = [
        {"_model_case_id": "case_train", "phase": "phase1", "_split": "train"},
        {"_model_case_id": "case_validation", "phase": "phase3", "_split": "validation"},
    ]
    summary, grid = select_global_multialpha_subsets(model_rows, solver_fn=solver)
    assert tuple(summary[0].keys()) == SUMMARY_COLUMNS
    assert tuple(grid[0].keys()) == GRID_COLUMNS
    assert all(row["selection_split"] == "train" for row in summary)
    assert all(row["validation_excluded_from_selection"] is True for row in summary)
    assert all(row["test_excluded_from_selection"] is True for row in summary)

    fixed = []
    for case_id in ["case_train", "case_validation"]:
        for target, y_true in [("CO_volpct_dry", 60.0), ("H2_volpct_dry", 30.0), ("CO2_volpct_dry", 10.0), ("H2_CO_ratio", 0.5)]:
            fixed.append({"case_id": case_id, "target": target, "N_cells": 3, "y_pred": y_true, "residual": 0.0})
    predictions = global_multialpha_prediction_rows(model_rows, summary[:1], fixed, fixed, fixed, [], solver_fn=solver)
    metrics = global_multialpha_metric_rows(predictions)
    assert tuple(predictions[0].keys()) == PREDICTION_COLUMNS
    assert tuple(metrics[0].keys()) == METRIC_COLUMNS
    assert len(predictions) == 8
    assert metrics[0]["poor_convergence_count_sum"] == 1
