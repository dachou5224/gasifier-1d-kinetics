from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))


def test_hybrid_calibration_protocol_schema_and_boundaries():
    from model.hybrid_calibration_protocol import (
        CLAIM_BOUNDARY,
        REQUIRED_PROTOCOL_KEYS,
        build_protocol,
        protocol_to_markdown,
    )

    repo_root = Path(__file__).resolve().parents[2]
    protocol = build_protocol(repo_root)
    assert tuple(protocol.keys()) == REQUIRED_PROTOCOL_KEYS
    assert CLAIM_BOUNDARY == protocol["claim_boundary"]
    assert len(protocol["candidate_alpha_groups"]) == 4
    names = {row["name"] for row in protocol["candidate_alpha_groups"]}
    assert {"alpha_wgs", "alpha_char_oxidation", "alpha_volatile_oxidation", "alpha_char_gasification"} <= names
    assert protocol["metric_schema"]["required_prediction_targets"] == [
        "CO_volpct_dry",
        "H2_volpct_dry",
        "CO2_volpct_dry",
        "H2_CO_ratio",
    ]
    assert any(item["name"] == "bounded_alpha_xt_zd" for item in protocol["baseline_matrix"])
    text = protocol_to_markdown(protocol)
    assert "Hybrid calibration protocol freeze" in text
    assert "not a trained hybrid model" in text
    assert "LIMS Weak-Label Leakage Policy" in text
    assert "alpha_wgs" in text
