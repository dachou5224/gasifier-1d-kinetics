from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))


def test_hybrid_diagnostic_bundle_schema_and_claim_guards(tmp_path):
    from model.hybrid_diagnostic_evidence_bundle import (
        BUNDLE_EVIDENCE_BOUNDARY,
        REQUIRED_METHOD_LABELS,
        build_hybrid_diagnostic_evidence_bundle,
    )

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    result = build_hybrid_diagnostic_evidence_bundle(repo_root=repo_root, out_dir=tmp_path)
    manifest = pd.read_csv(result["manifest"])
    metrics = pd.read_csv(result["metric_matrix"])
    claims = pd.read_csv(result["claim_table"])
    recommendations = pd.read_csv(result["recommendations"])
    markdown = open(result["markdown"], encoding="utf-8").read()

    assert manifest["exists"].all()
    assert (manifest["row_count"] > 0).all()
    assert set(REQUIRED_METHOD_LABELS) <= set(metrics["method_label"])
    heldout = metrics[metrics["split"].isin(["validation", "test"])]
    assert not heldout.empty
    assert set(heldout["selection_role"]) == {"held_out_transform_only"}
    assert "diagnostic-only boundary" in markdown.lower() or "diagnostic evidence bundle only" in markdown.lower()
    assert BUNDLE_EVIDENCE_BOUNDARY in markdown

    forbidden = " ".join(claims["forbidden_wording"].astype(str)).lower()
    for phrase in ["calibrated accuracy", "production control", "temporal generalization"]:
        assert phrase in forbidden
    assert {"supports_scaffold", "negative_multi_alpha", "overfit_warning", "diagnostic_only", "unsupported_for_accuracy"} <= set(claims["claim_safe_label"])
    assert "stay WGS-only" in " ".join(recommendations["current_safest_modeling_path"].astype(str)) or "stay wgs-only" in " ".join(recommendations["current_safest_modeling_path"].astype(str)).lower()
