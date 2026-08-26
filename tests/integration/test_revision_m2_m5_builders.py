from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from model.revision_m2_wgs_effects import select_block_length, moving_block_indices, recommend_gate
from model.revision_m3_exogenous import FORBIDDEN_FEATURES, cyclic_shift, feature_contract_rows
from model.revision_m4_estimability import RANK_TOL_REL, TARGET_SCALES, singular_spectrum
from model.revision_m5_n5_parity import ABS_ERROR_NEAR_ZERO, REL_ERROR_PASS, _status


def test_block_length_uses_train_acf_only():
    residuals = [1.0, 0.8, 0.2, -0.1, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0]
    protocol = select_block_length(residuals)
    assert protocol["selection_split"] == "train"
    assert 1 <= protocol["selected_block_length"] <= 8
    rng = np.random.default_rng(20260826)
    idx = moving_block_indices(10, protocol["selected_block_length"], rng)
    assert len(idx) == 10


def test_gate_recommendation_is_predeclared():
    effect = pd.DataFrame(
        [
            {"split_role": "test", "target": "CO_dry_vol_pct", "delta_rmse": -0.2},
            {"split_role": "test", "target": "CO2_dry_vol_pct", "delta_rmse": 0.2},
        ]
    )
    assert recommend_gate(effect) == "audit_supported_candidate"


def test_exogenous_contract_excludes_syngas_flow():
    contract = feature_contract_rows("deadbeef")
    forbidden = contract[contract["allowed_in_forward_comparator"] == False]
    assert "syngas_flow_mean" in set(forbidden["feature"])
    assert "syngas_flow_mean" in FORBIDDEN_FEATURES
    assert cyclic_shift([1, 2, 3, 4], 1) == [4, 1, 2, 3]


def test_estimability_does_not_set_identifiable_true():
    sensitivity = pd.DataFrame(
        [
            {"alpha_name": "alpha_wgs", "target": target, "scaled_sensitivity": 0.2, "status": "completed"}
            for target in TARGET_SCALES
        ]
        + [
            {"alpha_name": "alpha_char_oxidation", "target": target, "scaled_sensitivity": 0.01, "status": "completed"}
            for target in TARGET_SCALES
        ]
        + [
            {"alpha_name": "alpha_volatile_oxidation", "target": target, "scaled_sensitivity": 0.01, "status": "completed"}
            for target in TARGET_SCALES
        ]
    )
    spectrum, summary = singular_spectrum(sensitivity)
    assert RANK_TOL_REL == 1.0e-3
    assert "identifiable" not in {str(v).lower() for v in summary["evidence_field"]}
    assert set(summary["identifiable_flag_forbidden"]) == {False}


def test_m5_thresholds_match_existing_conventions():
    assert REL_ERROR_PASS == 1.0e-4
    assert _status(1.0, 1.0 + 1e-8) == "pass"
    assert _status(0.0, 1.0) == "fail"
    assert ABS_ERROR_NEAR_ZERO == 1.0e-6
