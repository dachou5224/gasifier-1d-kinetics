from __future__ import annotations

import math
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))


def test_char_bounds_and_parameter_injection_are_explicit():
    from model.global_char_oxidation_baseline import (
        CHAR_ALPHA_BOUND_HIGH,
        CHAR_ALPHA_BOUND_LOW,
        char_oxidation_parameter_path_audit,
        clamp_char_alpha,
        inject_char_oxidation_factor,
        validate_char_alpha_grid,
    )

    assert clamp_char_alpha(-1.0) == CHAR_ALPHA_BOUND_LOW
    assert clamp_char_alpha(2.0) == CHAR_ALPHA_BOUND_HIGH
    op_conds = {"coal_flow": 1.0, "WGS_CatalyticFactor": 0.3}
    injected = inject_char_oxidation_factor(op_conds, 0.35)
    assert injected["CharCombustionRateFactor"] == 0.35
    assert injected["WGS_CatalyticFactor"] == 0.3
    assert "CharCombustionRateFactor" not in op_conds
    assert validate_char_alpha_grid([0.05, 1.0]) == (0.05, 1.0)
    audit = char_oxidation_parameter_path_audit()
    assert audit["per_run_injection_without_public_api_change"] is True
    assert audit["status"] == "clean_for_nonlearned_global_scalar_comparator"


def _rows():
    rows = []
    for split in ("train", "train", "validation", "test"):
        rows.append(
            {
                "_model_case_id": f"{split}_{len(rows)}",
                "case_id": f"{split}_{len(rows)}",
                "phase": "phase1",
                "dataset_id": "fixture",
                "_split": split,
                "_split_role": "fixture",
            }
        )
    return rows


def _fake_solver(row, alpha, n_cells):
    if row["_split"] != "train":
        raise AssertionError("selection must not solve validation/test rows")
    target = 0.35
    return {
        "case_id": row["case_id"],
        "phase": row["phase"],
        "dataset_id": row["dataset_id"],
        "split": row["_split"],
        "split_role": row["_split_role"],
        "N_cells": n_cells,
        "objective": (float(alpha) - target) ** 2,
        "fallback_count": 0,
        "poor_convergence_count": 1,
        "cell0_cost": 0.01,
        "solve_status": "ok",
    }


def test_train_only_selection_and_summary_schema_from_fake_solver():
    from model.global_char_oxidation_baseline import (
        GLOBAL_CHAR_GRID_COLUMNS,
        GLOBAL_CHAR_SUMMARY_COLUMNS,
        build_char_summary,
        select_global_char_oxidation_alpha,
    )

    selected, grid_rows = select_global_char_oxidation_alpha(
        _rows(),
        alpha_grid=[0.05, 0.35, 1.0],
        n_cells=3,
        solver_fn=_fake_solver,
    )
    assert tuple(grid_rows[0].keys()) == GLOBAL_CHAR_GRID_COLUMNS
    assert selected["alpha_value"] == 0.35
    summary = build_char_summary(selected, grid_rows, n_cells=3)
    assert tuple(summary[0].keys()) == GLOBAL_CHAR_SUMMARY_COLUMNS
    assert summary[0]["selection_split"] == "train"
    assert summary[0]["validation_excluded_from_selection"] is True
    assert summary[0]["test_excluded_from_selection"] is True
    assert summary[0]["training_objective_delta"] < 0.0


def test_prediction_rows_and_objective_schema_from_fake_solver():
    from model.global_char_oxidation_baseline import (
        char_objective_from_result,
        char_oxidation_prediction_rows,
    )
    from model.split_global_alpha_baseline import PREDICTION_COLUMNS

    def solver(row, alpha, n_cells):
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
            "target_CO_volpct_dry": 62.0,
            "target_H2_volpct_dry": 29.0,
            "target_CO2_volpct_dry": 9.0,
            "target_H2_CO_ratio": 29.0 / 62.0,
            "fallback_count": 0,
            "poor_convergence_count": 1,
            "poor_convergence_cell_index": "0",
            "cell0_cost": 0.01,
            "cell0_max_abs_residual": 0.1,
            "solve_status": "ok",
        }

    predictions = char_oxidation_prediction_rows(_rows(), selected_alpha=0.35, solver_fn=solver)
    assert tuple(predictions[0].keys()) == PREDICTION_COLUMNS
    assert len(predictions) == 16
    assert predictions[0]["model_name"] == "global_char_oxidation_alpha"
    objective = char_objective_from_result(solver(_rows()[0], 0.35, 3))
    assert math.isfinite(objective)
