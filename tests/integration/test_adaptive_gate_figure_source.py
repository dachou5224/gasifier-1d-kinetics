from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))


def test_adaptive_gate_figure_source_schema_and_boundaries(tmp_path):
    from model.adaptive_gate_figure_source import (
        ALLOWED_STATUSES,
        GATE_FIGURE_BOUNDARY,
        REQUIRED_FIG4_METHOD_LABELS,
        build_adaptive_gate_figure_source_bundle,
    )

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    result = build_adaptive_gate_figure_source_bundle(repo_root=repo_root, out_dir=tmp_path)

    expected_keys = {
        "fig4_metric_matrix",
        "fig4_method_ladder",
        "channel_status",
        "reaction_family_gate_map",
        "alpha_similarity",
        "multialpha_activation",
        "manifest",
    }
    for key in expected_keys:
        assert os.path.exists(result[key])
        assert result["row_counts"][key] > 0

    metrics = pd.read_csv(result["fig4_metric_matrix"])
    assert set(REQUIRED_FIG4_METHOD_LABELS) <= set(metrics["method_label"])
    heldout = metrics[metrics["split"].isin(["validation", "test"])]
    assert not heldout.empty
    assert "train_selected" not in set(heldout["evidence_role"])

    ladder = pd.read_csv(result["fig4_method_ladder"])
    assert set(REQUIRED_FIG4_METHOD_LABELS) == set(ladder["method_label"])
    assert {"allowed_claim", "forbidden_claim", "source_round", "source_artifact_path"} <= set(ladder.columns)

    channels = pd.read_csv(result["channel_status"])
    assert set(channels["gate_status"]) <= set(ALLOWED_STATUSES)
    assert "provisionally_open" in set(channels["gate_status"])
    assert "future_data_needed" in set(channels["gate_status"])
    assert {"allowed_online_policy", "forbidden_claim", "table_row_label"} <= set(channels.columns)

    reaction = pd.read_csv(result["reaction_family_gate_map"])
    assert {"channel", "gasifier_region", "dominant_process", "missing_measurement_or_prior", "gate_status"} <= set(reaction.columns)
    assert set(reaction["gate_status"]) <= set(ALLOWED_STATUSES)

    similarity = pd.read_csv(result["alpha_similarity"])
    assert {"channel_i", "channel_j", "similarity_metric", "similarity_value", "near_collinear_flag"} <= set(similarity.columns)

    activation = pd.read_csv(result["multialpha_activation"])
    assert {"selected_subset", "channel", "selected_alpha", "activated_flag", "bound_hit"} <= set(activation.columns)

    manifest = pd.read_csv(result["manifest"])
    assert manifest["exists"].all()
    assert {"source", "generated_round49"} <= set(manifest["artifact_role"])

    markdown = open(result["markdown"], encoding="utf-8").read()
    assert GATE_FIGURE_BOUNDARY in markdown
    assert "source-data extraction for a gate-diagnostic case study" in markdown
    assert "not_available_from_existing_artifact" in markdown
