from __future__ import annotations

import math
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))


def _built_fixture() -> dict:
    return {
        "case_id": "case_1",
        "segment_id": "seg_1",
        "geometry": {},
        "coal_props": {"Cd": 75.0},
        "op_conds": {
            "coal_flow": 10.0,
            "o2_flow": 8.0,
            "carrier_gas_flow": 1.0,
            "HeatLossPercent": 2.0,
        },
        "targets": {},
        "assumptions": {},
    }


def test_engineering_coordinate_setters_preserve_intended_boundaries():
    from model.engineering_coordinate_influence import ENGINEERING_COORDINATE_PARAMETERS

    params = {param.name: param for param in ENGINEERING_COORDINATE_PARAMETERS}

    built = _built_fixture()
    base_oc = params["O_C_ratio_at_fixed_load"].getter(built)
    params["O_C_ratio_at_fixed_load"].setter(built, base_oc * 1.02)
    assert math.isclose(params["O_C_ratio_at_fixed_load"].getter(built), base_oc * 1.02)
    assert built["op_conds"]["coal_flow"] == 10.0
    assert built["op_conds"]["carrier_gas_flow"] == 1.0

    built = _built_fixture()
    params["load_at_fixed_O_C"].setter(built, 11.0)
    assert built["op_conds"]["coal_flow"] == 11.0
    assert math.isclose(built["op_conds"]["o2_flow"], 8.8)
    assert math.isclose(built["op_conds"]["carrier_gas_flow"], 1.1)

    built = _built_fixture()
    params["carrier_to_coal_ratio_boundary"].setter(built, 0.12)
    assert math.isclose(built["op_conds"]["carrier_gas_flow"], 1.2)
    assert built["op_conds"]["coal_flow"] == 10.0
    assert built["op_conds"]["o2_flow"] == 8.0


def test_engineering_coordinate_schema_and_boundary_from_fixture_rows():
    from model.engineering_coordinate_influence import (
        ENGINEERING_AXIS_COLUMNS,
        ENGINEERING_COORDINATE_COLUMNS,
        ENGINEERING_EVIDENCE_BOUNDARY,
        build_engineering_axis_summary,
        build_engineering_coordinate_rows,
        engineering_coordinate_markdown,
    )

    sensitivity_rows = [
        {
            "case_id": "case_1",
            "source_candidate_id": "seg_1",
            "independence_group": "IG001",
            "N_cells": 5,
            "parameter_name": "O_C_ratio_at_fixed_load",
            "parameter_key": "engineering.O_C_ratio",
            "perturbation_fraction": 0.02,
            "base_parameter_value": 0.82,
            "minus_parameter_value": 0.8036,
            "plus_parameter_value": 0.8364,
            "target_name": "T_out_K",
            "base_value": 1500.0,
            "minus_value": 1492.0,
            "plus_value": 1508.0,
            "raw_delta_centered": 8.0,
            "raw_sensitivity_per_unit": 487.8,
            "normalized_sensitivity": 0.2667,
            "abs_normalized_sensitivity": 0.2667,
            "status": "ok",
        },
        {
            "case_id": "case_1",
            "source_candidate_id": "seg_1",
            "independence_group": "IG001",
            "N_cells": 5,
            "parameter_name": "HeatLossPercent_calibration",
            "parameter_key": "engineering.HeatLossPercent_calibration",
            "perturbation_fraction": 0.02,
            "base_parameter_value": 2.0,
            "minus_parameter_value": 1.96,
            "plus_parameter_value": 2.04,
            "target_name": "T_out_K",
            "base_value": 1500.0,
            "minus_value": 1501.0,
            "plus_value": 1499.0,
            "raw_delta_centered": -1.0,
            "raw_sensitivity_per_unit": -25.0,
            "normalized_sensitivity": -0.0333,
            "abs_normalized_sensitivity": 0.0333,
            "status": "ok",
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

    axes = build_engineering_axis_summary(sensitivity_rows)
    rows = build_engineering_coordinate_rows(sensitivity_rows, axes, baseline_rows, contract_rows)
    text = engineering_coordinate_markdown(rows, axes)

    assert tuple(axes[0].keys()) == ENGINEERING_AXIS_COLUMNS
    assert tuple(rows[0].keys()) == ENGINEERING_COORDINATE_COLUMNS
    assert rows[0]["coordinate_class"] == "plant_operating"
    assert rows[0]["case_target_coordinate_rank"] == 1
    assert rows[0]["gradient_support_status"].startswith("mapped_to_supported")
    assert ENGINEERING_EVIDENCE_BOUNDARY in rows[0]["evidence_boundary_note"]
    assert "not an independent gasifier control priority" in text
    assert "not a plant operating knob" in text
