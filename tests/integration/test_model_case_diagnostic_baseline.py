from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from model.model_case_diagnostic_baseline import (
    REQUIRED_DIAGNOSTIC_COLUMNS,
    build_diagnostic_row,
    markdown_summary,
)


def test_model_case_diagnostic_row_has_required_columns_for_preview_row():
    pd = pytest.importorskip("pandas")
    preview = Path(
        "/Users/liuzhen/AI-projects/gasifier-data-prep/outputs/model_cases/"
        "sy3_trial_0928_feed_product.model_cases_preview.parquet"
    )
    if not preview.exists():
        pytest.skip("dataprep model_cases preview is not available")

    row = pd.read_parquet(preview).iloc[0]
    diagnostic = build_diagnostic_row(row, n_cells=1)

    assert tuple(diagnostic.keys()) == REQUIRED_DIAGNOSTIC_COLUMNS
    assert diagnostic["case_id"] == "sy3_trial_0928_feed_product__modelcase_001"
    assert diagnostic["N_cells"] == 1
    assert diagnostic["carrier_mapping_ok"] is True
    assert diagnostic["transported_moisture_percent"] == 2.0
    assert diagnostic["cell0_cost"] is not None


def test_markdown_summary_mentions_evidence_boundary():
    row = {column: 0 for column in REQUIRED_DIAGNOSTIC_COLUMNS}
    row.update(
        {
            "case_id": "case",
            "independence_group": "IG",
            "poor_convergence_cell_index": "0",
        }
    )
    text = markdown_summary([row])
    assert "diagnostic baseline" in text
    assert "not calibrated accuracy evidence" in text
