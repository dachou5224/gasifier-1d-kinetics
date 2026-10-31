from __future__ import annotations

import math
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))


def test_sensitivity_schema_and_ranking_from_fixture_rows():
    from model.model_case_sensitivity import (
        RANKING_COLUMNS,
        SENSITIVITY_COLUMNS,
        build_parameter_ranking,
        sensitivity_markdown,
    )

    rows = []
    for parameter_name, value in [("HeatLossPercent", 0.2), ("oxygen_flow", 1.5)]:
        row = {column: "" for column in SENSITIVITY_COLUMNS}
        row.update(
            {
                "case_id": "case_001",
                "source_candidate_id": "segment_001",
                "independence_group": "IG001",
                "N_cells": 5,
                "parameter_name": parameter_name,
                "parameter_key": f"op_conds.{parameter_name}",
                "target_name": "T_out_K",
                "abs_normalized_sensitivity": value,
                "status": "ok",
            }
        )
        rows.append(row)

    ranking = build_parameter_ranking(rows)

    assert "normalized_sensitivity" in SENSITIVITY_COLUMNS
    assert "mean_abs_normalized_sensitivity" in RANKING_COLUMNS
    assert len(ranking) == 2
    assert ranking[0]["parameter_name"] == "oxygen_flow"
    assert math.isclose(float(ranking[0]["mean_abs_normalized_sensitivity"]), 1.5)
    summary = sensitivity_markdown(rows, ranking)
    assert "Fixed-physics sensitivity ranking diagnostic" in summary
    assert "not calibration accuracy evidence" in summary
