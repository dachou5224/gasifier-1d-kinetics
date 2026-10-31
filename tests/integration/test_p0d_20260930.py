from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from model.p0d_20260930 import (
    DRY_GAP_ACCOUNTING_ATOL_MOL_S,
    ELEMENT_IDENTITY_STOP_MOL_S,
    O2_SHARE_CONTINUE_LIMIT,
    classify_downstream,
    classify_dry_gap,
    dry_gap_components,
    gap_stop_reason,
    nm3_h_to_mol_s,
    refresh_gap_identity,
)


def test_dry_gap_columns_match_total_when_o2_is_zero():
    gas = np.array([0.0, 4.0, 30.0, 20.0, 1.0, 25.0, 40.0, 8.0])
    plant_dry_nm3_h = 9000.0
    plant_dry = nm3_h_to_mol_s(plant_dry_nm3_h)
    result = dry_gap_components(
        gas,
        plant_dry_nm3_h=plant_dry_nm3_h,
        plant_co_vol_pct=40.0,
        plant_co2_vol_pct=30.0,
        plant_h2_vol_pct=20.0,
    )
    assert abs(result["gap_columns_minus_dry_gap_mol_s"]) < DRY_GAP_ACCOUNTING_ATOL_MOL_S
    assert abs(result["plant_dry_mol_s"] - plant_dry) < 1.0e-9
    expected_other = 4.0 + 1.0 + 40.0
    assert abs(result["predicted_other_dry_mol_s"] - expected_other) < 1.0e-12
    plant_co = plant_dry * 0.40
    assert abs(result["gap_CO_mol_s"] - (30.0 - plant_co)) < 1.0e-9
    assert abs(
        result["gap_CO_mol_s"] + result["gap_CO2_mol_s"] + result["gap_H2_mol_s"] + result["gap_other_dry_mol_s"]
        - result["dry_gap_mol_s"]
    ) < 1.0e-9


def test_four_columns_plus_predicted_o2_match_dry_gap():
    gas = np.array([1.5, 4.0, 30.0, 20.0, 1.0, 25.0, 40.0, 8.0])
    result = dry_gap_components(
        gas,
        plant_dry_nm3_h=9000.0,
        plant_co_vol_pct=40.0,
        plant_co2_vol_pct=30.0,
        plant_h2_vol_pct=20.0,
    )
    four = result["gap_CO_mol_s"] + result["gap_CO2_mol_s"] + result["gap_H2_mol_s"] + result["gap_other_dry_mol_s"]
    assert abs(four + 1.5 - result["dry_gap_mol_s"]) < 1.0e-9
    assert abs(result["gap_columns_minus_dry_gap_mol_s"]) < DRY_GAP_ACCOUNTING_ATOL_MOL_S
    assert result["o2_share_of_abs_dry_gap"] >= O2_SHARE_CONTINUE_LIMIT
    assert gap_stop_reason(result)


def test_small_o2_share_continues_and_old_row_can_be_refreshed():
    gas = np.array([0.004, 4.0, 300.0, 200.0, 1.0, 250.0, 400.0, 80.0])
    result = dry_gap_components(
        gas,
        plant_dry_nm3_h=120000.0,
        plant_co_vol_pct=40.0,
        plant_co2_vol_pct=20.0,
        plant_h2_vol_pct=25.0,
    )
    assert result["o2_share_of_abs_dry_gap"] < O2_SHARE_CONTINUE_LIMIT
    assert gap_stop_reason(result) == ""
    stale = dict(result)
    stale["gap_columns_sum_mol_s"] = (
        result["gap_CO_mol_s"] + result["gap_CO2_mol_s"] + result["gap_H2_mol_s"] + result["gap_other_dry_mol_s"]
    )
    stale["gap_columns_minus_dry_gap_mol_s"] = stale["gap_columns_sum_mol_s"] - result["dry_gap_mol_s"]
    refreshed = refresh_gap_identity(stale)
    assert abs(refreshed["gap_columns_minus_dry_gap_mol_s"]) < DRY_GAP_ACCOUNTING_ATOL_MOL_S
    assert gap_stop_reason(refreshed) == ""


def test_downstream_and_gap_sentences():
    assert ELEMENT_IDENTITY_STOP_MOL_S == 1.0e-2
    cells = pd.DataFrame(
        [
            {"window_id": "W1", "N_cells": 3, "cell_index": 0, "residual_cost": 0.01, "exceeds_1e-4": True},
            {"window_id": "W1", "N_cells": 3, "cell_index": 1, "residual_cost": 1.0e-8, "exceeds_1e-4": False},
            {"window_id": "W1", "N_cells": 3, "cell_index": 2, "residual_cost": 1.0e-8, "exceeds_1e-4": False},
        ]
    )
    assert classify_downstream(cells)["sentence_id"] == 1
    cells.loc[cells["cell_index"] == 2, "exceeds_1e-4"] = True
    cells.loc[cells["cell_index"] == 2, "residual_cost"] = 0.002
    downstream = classify_downstream(cells)
    assert downstream["sentence_id"] == 2
    assert downstream["cells"][0]["cell_index"] == 2

    gap = pd.DataFrame(
        [
            {"dry_gap_mol_s": -100.0, "gap_CO_mol_s": -80.0, "gap_CO2_mol_s": -10.0, "gap_H2_mol_s": -5.0, "gap_other_dry_mol_s": -5.0},
            {"dry_gap_mol_s": -200.0, "gap_CO_mol_s": -160.0, "gap_CO2_mol_s": -20.0, "gap_H2_mol_s": -10.0, "gap_other_dry_mol_s": -10.0},
        ]
    )
    assert classify_dry_gap(gap)["sentence_id"] == 3
    assert classify_dry_gap(gap)["dominant_column"] == "CO"
    mixed = gap.copy()
    mixed.loc[0, "gap_CO_mol_s"] = -20.0
    mixed.loc[0, "gap_H2_mol_s"] = -65.0
    mixed.loc[1, "gap_CO_mol_s"] = -40.0
    mixed.loc[1, "gap_H2_mol_s"] = -130.0
    split = classify_dry_gap(mixed)
    assert split["sentence_id"] == 4
    assert "CO" in split["sentence"] and "other dry" in split["sentence"]
