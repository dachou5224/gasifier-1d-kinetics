from __future__ import annotations

import math
import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from model.p0_round_20260930 import (
    POOR_CONVERGENCE_COST_THRESHOLD,
    PUBLISHED_TEST_CO_RMSE,
    REPRODUCTION_TOLERANCE,
    assert_no_local_routing,
    reaction_source_mol_s,
    reproduction_lock_result,
    select_judgment,
)


class _Cell:
    def __init__(self):
        self.op_conds = {"Combustion_CO2_Fraction": 1.0}


def test_reaction_source_matches_balance_formula():
    cell = _Cell()
    r_het = {"C+O2": 2.0, "C+H2O": 3.0, "C+CO2": 0.5, "C+H2": 0.25}
    r_homo = {"CO_Ox": 1.0, "H2_Ox": 0.4, "WGS": 0.2, "RWGS": 0.1, "CH4_Ox": 0.3, "MSR": 0.05}
    phi = 2.0
    rxn = reaction_source_mol_s(cell, r_het, r_homo, phi)
    f_co2 = 1.0
    r1, r2, r3 = 1.0, 0.4, 0.2
    r4, r5, r6 = 0.1, 0.3, 0.05
    r_comb, r_h2o, r_co2g, r_h2g = 2.0, 3.0, 0.5, 0.25
    expected_h2 = r_h2o + r3 + 3 * r6 - (2 * r_h2g + r2 + r4)
    expected_co2 = (2.0 / phi - 1.0) * r_comb + f_co2 * r1 + r3 + f_co2 * r5 - (r_co2g + r4)
    expected_o2 = -(r_comb / phi + 0.5 * f_co2 * r1 + 0.5 * r2 + (1.5 + 0.5 * f_co2) * r5)
    assert abs(rxn[5] - expected_h2) < 1e-12
    assert abs(rxn[3] - expected_co2) < 1e-12
    assert abs(rxn[0] - expected_o2) < 1e-12


def _frames_for_ratio(ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    term = pd.DataFrame(
        [
            {
                "window_id": "W1",
                "species": "H2",
                "convection_inout_mol_s": 10.0,
                "reaction_source_mol_s": -4.0,
                "inlet_source_mol_s": -4.0,
                "residual_mol_s": 2.0,
            },
            {
                "window_id": "W1",
                "species": "CO2",
                "convection_inout_mol_s": -10.0,
                "reaction_source_mol_s": 4.0,
                "inlet_source_mol_s": 4.0,
                "residual_mol_s": -2.0,
            },
        ]
    )
    element_error = 2.0 / ratio
    closure = pd.DataFrame(
        [
            {
                "window_id": "W1",
                "phase": 1,
                "cell0_residual_cost": 0.01,
                "cell0_exceeds_1e-4": True,
                "C_closure_error_mol_s": element_error,
                "H_closure_error_mol_s": element_error / 2.0,
                "O_closure_error_mol_s": 0.0,
                "N_closure_error_mol_s": 0.0,
                "S_closure_error_mol_s": 0.0,
                "energy_closure_W": -1.0,
            }
        ]
    )
    long_frame = pd.DataFrame(
        [
            {
                "window_id": "W1",
                "cell_index": 0,
                "exceeds_1e-4": True,
                "residual_cost": 0.01,
            }
        ]
    )
    return term, closure, long_frame


def test_judgment_uses_element_order_when_terms_cancel():
    same_order = select_judgment(*_frames_for_ratio(5.0))
    assert same_order["sentence_id"] == 3
    much_smaller = select_judgment(*_frames_for_ratio(1000.0))
    assert much_smaller["sentence_id"] == 4
    assert "3 个数量级" in much_smaller["sentence"]


def test_reproduction_lock_keeps_published_tolerance():
    assert POOR_CONVERGENCE_COST_THRESHOLD == 1.0e-4
    assert REPRODUCTION_TOLERANCE == 0.05
    ids = {"train": ["A"], "validation": ["B"], "test": ["C"], "all": ["A", "B", "C"]}
    rows = []
    for window_id, role, residual in (("A", "train", 1.0), ("B", "validation", 1.0), ("C", "test", 0.0)):
        rows.append(
            {
                "window_id": window_id,
                "split_role": role,
                "execution_status": "completed",
                "co_residual_vol_pct": PUBLISHED_TEST_CO_RMSE if role == "test" else residual,
            }
        )
    passed = reproduction_lock_result(pd.DataFrame(rows), ids, source_commit_hash="98ad47c")
    assert passed["passed"] is True
    rows[2]["co_residual_vol_pct"] = PUBLISHED_TEST_CO_RMSE + 0.05 + 1.0e-6
    failed = reproduction_lock_result(pd.DataFrame(rows), ids, source_commit_hash="98ad47c")
    assert failed["passed"] is False
    assert math.isclose(failed["tolerance"], 0.05)


def test_followup_stop_and_branch_rules():
    import numpy as np

    from model.p0_followup_20260930 import (
        classify_stop,
        name_h2_outlet_driver,
        rate_branch_criterion,
    )
    from model.p0_round_20260930 import element_identity_max_abs

    assert classify_stop(0.01, 1.0e-6) == "no_root_after_limits"
    assert classify_stop(0.01, 0.01) == "stopped_early_cost_still_descends"
    assert classify_stop(1.0e-8, 0.5) == "residual_below_threshold"
    assert name_h2_outlet_driver(0.0, {"H2_Ox": 80.0, "C+H2": 1.0, "RWGS": 0.2}, 1.0, 80.0) == (
        "H2_Ox_combustion_zone_instant_oxidation"
    )
    assert element_identity_max_abs(
        {"C_closure_error_mol_s": 1.0, "H_closure_error_mol_s": 2.0, "O_closure_error_mol_s": 0.0, "N_closure_error_mol_s": 0.0, "S_closure_error_mol_s": 0.0},
        {"C": 1.0, "H": 2.0, "O": 0.0, "N": 0.0, "S": 1.0e-9},
    ) == 1.0e-9

    class _Inlet:
        gas_moles = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    class _Current:
        P = 4.0e6

    class _Cell:
        def __init__(self, disabled: bool):
            self.op_conds = {"diagnostic_disable_combustion_zone": disabled}
            self.inlet = _Inlet()

    entered = rate_branch_criterion(_Cell(False), _Current(), np.zeros(8))
    disabled = rate_branch_criterion(_Cell(True), _Current(), np.zeros(8))
    assert entered["would_enter_combustion_zone"] is True
    assert entered["is_combustion"] is True
    assert disabled["would_enter_combustion_zone"] is True
    assert disabled["is_combustion"] is False


def test_artifacts_reject_local_routing_tokens():
    assert_no_local_routing("R52-C-75 reference N_cells=1")
    try:
        assert_no_local_routing("see /Users/example/workbook.xlsx")
    except RuntimeError:
        return
    raise AssertionError("local routing token was accepted")
