from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))


def _fixture_response_rows():
    rows = []
    for multiplier, temp, co, h2, co2, flow in [
        (0.96, 1490.0, 65.0, 24.0, 10.5, 1.20),
        (0.98, 1495.0, 66.0, 23.0, 10.2, 1.22),
        (1.00, 1500.0, 67.0, 22.0, 10.0, 1.24),
        (1.02, 1505.0, 68.0, 21.0, 9.8, 1.26),
        (1.04, 1510.0, 69.0, 20.0, 9.6, 1.28),
    ]:
        rows.append(
            {
                "phase": "phase1",
                "dataset_id": "dataset",
                "case_id": "case_1",
                "source_candidate_id": "seg_1",
                "independence_group": "IG001",
                "N_cells": 5,
                "oc_multiplier": multiplier,
                "base_o_c_ratio": 0.9,
                "o_c_ratio": 0.9 * multiplier,
                "o2_flow_kg_s": 8.0 * multiplier,
                "coal_flow_kg_s": 10.0,
                "carrier_gas_flow_kg_s": 1.0,
                "runtime_s": 0.1,
                "status": "ok",
                "failure_reason": "",
                "fallback_count": 0,
                "poor_convergence_count": 1,
                "poor_convergence_cell_index": "0",
                "cell0_cost": 0.01,
                "cell0_max_abs_residual": 0.1,
                "T_out_K": temp,
                "T_out_degC": temp - 273.15,
                "CO_volpct_dry": co,
                "H2_volpct_dry": h2,
                "CO2_volpct_dry": co2,
                "N2_volpct_dry": 0.5,
                "dry_syngas_flow_kmol_s": flow,
                "H2_CO_ratio": h2 / co,
                "evidence_boundary_note": "boundary",
            }
        )
    return rows


def test_oc_response_axis_schema_and_local_slope_from_fixture_rows():
    from model.oc_response_curve_diagnostic import (
        OC_RESPONSE_AXIS_COLUMNS,
        OC_RESPONSE_COLUMNS,
        OC_RESPONSE_EVIDENCE_BOUNDARY,
        build_oc_response_axis_rows,
        oc_response_markdown,
    )

    rows = _fixture_response_rows()
    axes = build_oc_response_axis_rows(rows)
    text = oc_response_markdown(rows, axes)

    assert tuple(rows[0].keys()) == OC_RESPONSE_COLUMNS
    assert tuple(axes[0].keys()) == OC_RESPONSE_AXIS_COLUMNS
    temp_axes = [row for row in axes if row["target_name"] == "T_out_K"]
    assert len(temp_axes) == 5
    assert temp_axes[0]["curve_direction_status"] == "monotonic_increasing"
    assert temp_axes[0]["local_slope_sign"] == "positive"
    assert temp_axes[0]["slope_direction_matches_curve"] is True
    assert "fixed-physics O/C response evidence" in text
    assert OC_RESPONSE_EVIDENCE_BOUNDARY in text
