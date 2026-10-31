from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))


def test_heat_loss_gradient_parity_schema_and_values():
    pytest.importorskip("jax")

    from model.gradient_parity_diagnostic import (
        GRADIENT_PARITY_COLUMNS,
        build_gradient_parity_rows,
    )

    rows = build_gradient_parity_rows(
        parameters=("HeatLossPercent",),
        outputs=("final_T_K", "final_CO_mol_s"),
        step_fraction=1.0e-3,
    )

    assert len(rows) == 2
    assert set(rows[0]) == set(GRADIENT_PARITY_COLUMNS)
    assert {row["output"] for row in rows} == {"final_T_K", "final_CO_mol_s"}
    for row in rows:
        assert row["status"] == "ok"
        assert np.isfinite(row["fd_grad"])
        assert np.isfinite(row["ift_or_ad_grad"])
        assert row["abs_error"] < 1.0e-4
        assert row["rel_error"] < 1.0e-4


def test_coal_mass_flow_gradient_parity_schema_and_values():
    pytest.importorskip("jax")

    from model.gradient_parity_diagnostic import build_gradient_parity_rows

    rows = build_gradient_parity_rows(
        parameters=("coal_mass_flow",),
        outputs=("final_T_K", "final_dry_flow_mol_s"),
        step_fraction=1.0e-3,
    )

    assert len(rows) == 2
    assert {row["parameter"] for row in rows} == {"coal_mass_flow"}
    assert {row["output"] for row in rows} == {"final_T_K", "final_dry_flow_mol_s"}
    for row in rows:
        assert row["status"] == "ok"
        assert np.isfinite(row["fd_grad"])
        assert np.isfinite(row["ift_or_ad_grad"])
        assert row["abs_error"] < 5.0e-3
        assert row["rel_error"] < 1.0e-4
        assert "does not apply LIMS moisture" in row["notes"]
