from __future__ import annotations

import math
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))


def _contract_rows():
    rows = []
    for idx, phase in enumerate(["phase1", "phase1", "phase2", "phase3", "phase4"], start=1):
        rows.append(
            {
                "case_id": f"case_{idx}",
                "phase": phase,
                "dataset_id": f"dataset_{phase}",
                "independence_group": f"IG{idx:03d}",
            }
        )
    return rows


def test_split_manifest_exclusivity_and_order_from_fixture():
    from model.split_global_alpha_baseline import build_split_manifest, validate_split_manifest

    rows = build_split_manifest(_contract_rows())
    validation = validate_split_manifest(rows)
    assert len({row["case_id"] for row in rows}) == len(rows)
    assert [row["chronological_rank"] for row in rows] == [1, 2, 3, 4, 5]
    assert [row["split"] for row in rows] == ["train", "train", "train", "validation", "test"]
    assert validation["has_duplicate_case_id"] is False
    assert validation["chronological_rank_is_monotonic"] is True


def test_fixed_metric_schema_and_values_from_fixture():
    from model.split_global_alpha_baseline import (
        METRIC_COLUMNS,
        PREDICTION_COLUMNS,
        build_split_manifest,
        compute_metric_rows,
        fixed_physics_prediction_rows,
    )

    split_rows = build_split_manifest(_contract_rows())
    baseline_rows = [
        {
            "case_id": "case_1",
            "N_cells": 3,
            "target_CO_vol_pct": 60.0,
            "pred_CO_vol_pct": 63.0,
            "target_H2_vol_pct": 30.0,
            "pred_H2_vol_pct": 28.0,
            "target_CO2_vol_pct": 8.0,
            "pred_CO2_vol_pct": 9.0,
            "target_H2_CO_ratio": 0.5,
            "pred_H2_CO_ratio": 28.0 / 63.0,
            "fallback_count": 0,
            "poor_convergence_count": 1,
            "poor_convergence_cell_index": "0",
            "cell0_cost": 0.01,
            "cell0_max_abs_residual": 0.1,
        }
    ]
    predictions = fixed_physics_prediction_rows(baseline_rows, split_rows)
    metrics = compute_metric_rows(predictions, evidence_boundary="boundary")
    assert tuple(predictions[0].keys()) == PREDICTION_COLUMNS
    assert tuple(metrics[0].keys()) == METRIC_COLUMNS
    co_metric = [row for row in metrics if row["target"] == "CO_volpct_dry"][0]
    assert co_metric["n"] == 1
    assert math.isclose(co_metric["rmse"], 3.0)
    assert co_metric["fallback_count_sum"] == 0
    assert co_metric["poor_convergence_count_sum"] == 1


def test_wgs_summary_enforces_train_only_selection_and_bounds():
    from model.split_global_alpha_baseline import GLOBAL_WGS_SUMMARY_COLUMNS, build_wgs_summary

    grid_rows = [
        {
            "alpha_name": "alpha_wgs",
            "alpha_value": 0.5,
            "selection_split": "train",
            "N_cells": 3,
            "train_case_count": 12,
            "training_objective": 1.0,
            "status": "ok",
            "evidence_boundary": "boundary",
        },
        {
            "alpha_name": "alpha_wgs",
            "alpha_value": 1.0,
            "selection_split": "train",
            "N_cells": 3,
            "train_case_count": 12,
            "training_objective": 2.0,
            "status": "ok",
            "evidence_boundary": "boundary",
        },
    ]
    rows = build_wgs_summary(grid_rows[0], grid_rows, n_cells=3)
    assert tuple(rows[0].keys()) == GLOBAL_WGS_SUMMARY_COLUMNS
    assert rows[0]["selection_split"] == "train"
    assert rows[0]["validation_excluded_from_selection"] is True
    assert rows[0]["test_excluded_from_selection"] is True
    assert rows[0]["bound_low"] <= rows[0]["selected_alpha"] <= rows[0]["bound_high"]
    assert rows[0]["training_objective_delta"] < 0.0
