from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from model.revision_m1_cell0 import declared_variants, select_configuration, select_representative_windows


def test_declared_variants_preserve_length_and_freeze_reference():
    variants = declared_variants(10.3)
    assert set(variants) == {"reference", "refined_first_cv", "alt_cell0_init", "tighter_tol"}
    assert variants["reference"]["n_cells"] == 1
    assert variants["reference"]["solver_options"] == {}
    refined = variants["refined_first_cv"]
    first = refined["solver_options"]["first_cell_length"]
    rest = refined["solver_options"]["ignition_zone_res"]
    assert first + rest * (refined["n_cells"] - 1) == pytest.approx(10.3)


def test_window_selection_covers_phases_cost_and_oc_without_targets():
    runtime = pd.DataFrame(
        {
            "window_id": [f"W{i}" for i in range(8)],
            "cell0_cost": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
        }
    )
    rows = [
        {
            "window_id": f"W{i}",
            "phase_number": 1 + (i % 4),
            "split_role": "train",
            "o_c_ratio": 0.8 + 0.05 * i,
            "co_vol_pct": 100 - i,
        }
        for i in range(8)
    ]
    selected = select_representative_windows(runtime, rows, min_count=6)
    assert set(selected["phase"]) == {1, 2, 3, 4}
    joined = ";".join(selected["selection_reason"])
    assert "min_cell0_cost" in joined
    assert "max_cell0_cost" in joined
    assert "global_low_oc" in joined
    assert "global_high_oc" in joined
    assert "co_vol_pct" not in selected.columns


def test_configuration_selection_ignores_outlet_error():
    rows = []
    for config, cost, poor in (
        ("reference", 0.01, "poor"),
        ("tighter_tol", 0.001, "acceptable"),
        ("refined_first_cv", 0.02, "poor"),
    ):
        for window in ("A", "B"):
            rows.append(
                {
                    "configuration": config,
                    "window_id": window,
                    "execution_status": "completed",
                    "residual_quality_status": poor,
                    "cell0_cost": cost,
                    "outlet_rmse": 99.0 if config == "tighter_tol" else 1.0,
                }
            )
    result = select_configuration(pd.DataFrame(rows))
    assert result["selected_configuration"] == "tighter_tol"
    assert result["numerical_quality_unresolved"] is False
    assert "outlet error is not used" in result["selection_rule"]


def test_tiny_cost_difference_does_not_displace_reference():
    rows = []
    for config, cost in (("reference", 0.013925062936284), ("tighter_tol", 0.013925062936276)):
        for window in ("A", "B"):
            rows.append(
                {
                    "configuration": config,
                    "window_id": window,
                    "execution_status": "completed",
                    "residual_quality_status": "poor",
                    "cell0_cost": cost,
                }
            )
    result = select_configuration(pd.DataFrame(rows))
    assert result["selected_configuration"] == "reference"
    assert result["gate_status"] == "stable"
    assert result["numerical_quality_unresolved"] is True
