from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))


def test_gradient_smoke_markdown_schema_for_fixture_rows():
    from model.end_to_end_gradient_smoke import (
        END_TO_END_GRADIENT_SMOKE_COLUMNS,
        gradient_smoke_markdown,
    )

    row = {column: 0.0 for column in END_TO_END_GRADIENT_SMOKE_COLUMNS}
    row.update(
        {
            "step": 0,
            "parameter_name": "HeatLossPercent",
            "loss": 4.5,
            "next_loss": 4.0,
            "direction_check_passed": True,
            "solve_success": True,
        }
    )

    assert "parameter_name" in END_TO_END_GRADIENT_SMOKE_COLUMNS
    assert "next_loss" in END_TO_END_GRADIENT_SMOKE_COLUMNS
    summary = gradient_smoke_markdown([row])
    assert "End-to-end gradient smoke" in summary
    assert "not calibration accuracy evidence" in summary
