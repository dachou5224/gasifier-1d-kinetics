from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))


def test_influence_tracing_schema_and_boundary_from_fixture_rows():
    from model.influence_tracing_diagnostic import (
        AXIS_SUMMARY_COLUMNS,
        EVIDENCE_BOUNDARY_NOTE,
        INFLUENCE_TRACING_COLUMNS,
        build_axis_summary_rows,
        build_influence_tracing_rows,
        influence_tracing_markdown,
    )

    sensitivity_rows = [
        {
            "case_id": "case_1",
            "source_candidate_id": "seg_1",
            "independence_group": "IG001",
            "N_cells": 5,
            "parameter_name": "oxygen_flow",
            "parameter_key": "op_conds.o2_flow",
            "perturbation_fraction": 0.02,
            "base_parameter_value": 1.0,
            "base_value": 100.0,
            "minus_value": 99.0,
            "plus_value": 101.0,
            "raw_delta_centered": 1.0,
            "raw_sensitivity_per_unit": 2.0,
            "normalized_sensitivity": 0.2,
            "abs_normalized_sensitivity": 0.2,
            "target_name": "T_out_K",
            "status": "ok",
        },
        {
            "case_id": "case_1",
            "source_candidate_id": "seg_1",
            "independence_group": "IG001",
            "N_cells": 5,
            "parameter_name": "HeatLossPercent",
            "parameter_key": "op_conds.HeatLossPercent",
            "perturbation_fraction": 0.02,
            "base_parameter_value": 2.0,
            "base_value": 100.0,
            "minus_value": 100.5,
            "plus_value": 99.5,
            "raw_delta_centered": -0.5,
            "raw_sensitivity_per_unit": -1.0,
            "normalized_sensitivity": -0.02,
            "abs_normalized_sensitivity": 0.02,
            "target_name": "T_out_K",
            "status": "ok",
        },
    ]
    ranking_rows = [
        {
            "rank": 1,
            "parameter_name": "oxygen_flow",
            "parameter_key": "op_conds.o2_flow",
            "mean_abs_normalized_sensitivity": 0.2,
            "median_abs_normalized_sensitivity": 0.2,
            "max_abs_normalized_sensitivity": 0.2,
        },
        {
            "rank": 2,
            "parameter_name": "HeatLossPercent",
            "parameter_key": "op_conds.HeatLossPercent",
            "mean_abs_normalized_sensitivity": 0.02,
            "median_abs_normalized_sensitivity": 0.02,
            "max_abs_normalized_sensitivity": 0.02,
        },
    ]
    parity_rows = [
        {
            "parameter": "oxygen_flow",
            "output": "final_T_K",
            "status": "ok",
            "abs_error": 1.0e-7,
            "rel_error": 1.0e-6,
        },
        {
            "parameter": "HeatLossPercent",
            "output": "final_T_K",
            "status": "ok",
            "abs_error": 2.0e-7,
            "rel_error": 2.0e-6,
        },
    ]
    baseline_rows = [
        {
            "case_id": "case_1",
            "N_cells": 5,
            "cell0_cost": 0.01,
            "cell0_max_abs_residual": 0.1,
            "fallback_count": 0,
            "poor_convergence_count": 1,
            "poor_convergence_cell_index": "0",
        }
    ]
    contract_rows = [{"case_id": "case_1", "phase": "phase1", "dataset_id": "dataset"}]

    rows = build_influence_tracing_rows(
        sensitivity_rows,
        ranking_rows,
        parity_rows,
        baseline_rows,
        contract_rows,
    )
    axes = build_axis_summary_rows(rows, parity_rows)
    text = influence_tracing_markdown(rows, axes)

    assert tuple(rows[0].keys()) == INFLUENCE_TRACING_COLUMNS
    assert tuple(axes[0].keys()) == AXIS_SUMMARY_COLUMNS
    assert rows[0]["case_target_axis_rank"] == 1
    assert rows[0]["gradient_support_status"] == "supported_fixed_branch_not_tested_full_model"
    assert rows[0]["cell0_localization_status"] == "persistent_cell0_residual_quality_flag"
    assert EVIDENCE_BOUNDARY_NOTE in rows[0]["evidence_boundary_note"]
    assert "candidate online-correction" in text
    assert "not a trained hybrid" in text
