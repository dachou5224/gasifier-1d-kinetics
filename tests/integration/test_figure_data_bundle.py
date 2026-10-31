from __future__ import annotations

import json
import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))


def test_figure_data_bundle_manifest_and_key_columns(tmp_path):
    from model.figure_data_bundle import build_figure_data_bundle

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    manifest = build_figure_data_bundle(repo_root=repo_root, out_dir=tmp_path)

    expected = {
        "fig2_baseline_mismatch": ["case_id", "N_cells", "target_name", "predicted_value", "target_value", "delta_pred_minus_target"],
        "fig3_gradient_parity_matrix": ["parameter", "output", "fd_grad", "ift_grad", "rel_error", "status"],
        "fig3_ift_cost_scaling": ["method", "N_cells", "status", "compile_or_cold_time_s", "warm_eval_time_s", "gradient_value"],
        "fig3_gradient_smoke_trajectory": ["step", "parameter_value", "loss", "gradient", "next_loss", "direction_check_passed"],
        "table_sensitivity_ranking": ["rank", "parameter_name", "mean_abs_normalized_sensitivity", "status"],
    }
    for artifact_name, key_columns in expected.items():
        entry = manifest["artifacts"][artifact_name]
        path = entry["file"]
        df = pd.read_csv(path)
        assert entry["row_count"] == len(df)
        assert len(df) > 0
        for column in key_columns:
            assert column in df.columns

    parity = pd.read_csv(manifest["artifacts"]["fig3_gradient_parity_matrix"]["file"])
    assert len(parity) == 20
    assert set(parity["status"]) == {"ok"}

    manifest_path = manifest["artifacts"]["manifest"]["file"]
    loaded = json.loads(open(manifest_path, encoding="utf-8").read())
    assert loaded["global_evidence_boundary"]
    assert "hybrid calibration accuracy" in loaded["global_evidence_boundary"]
    if "ift_cost_protocol" in loaded["artifacts"]:
        protocol_entry = loaded["artifacts"]["ift_cost_protocol"]
        assert protocol_entry["source_artifact"].endswith("ift_cost_protocol.md")
        assert protocol_entry["row_count"] > 0
