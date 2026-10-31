from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))


def test_cell0_residual_diagnostic_preview_schema_and_top_components():
    pd = pytest.importorskip("pandas")

    from model.cell0_residual_diagnostic import (
        CELL0_RESIDUAL_COLUMNS,
        build_cell0_residual_rows,
        cell0_residual_markdown,
    )

    preview = Path(
        "/Users/liuzhen/AI-projects/gasifier-data-prep/outputs/model_cases/"
        "sy3_trial_0928_feed_product.model_cases_preview.parquet"
    )
    if not preview.exists():
        pytest.skip("dataprep model_cases preview is not available")

    df = pd.read_parquet(preview)
    rows = build_cell0_residual_rows([row for _, row in df.iterrows()], n_cells_values=(1, 3, 5))

    assert len(rows) == len(df) * 3
    assert set(rows[0]) == set(CELL0_RESIDUAL_COLUMNS)
    assert {row["N_cells"] for row in rows} == {1, 3, 5}
    assert len({row["case_id"] for row in rows}) == len(df)
    for row in rows:
        assert row["cell_index"] == 0
        assert row["top1_name"]
        assert row["top2_name"]
        assert row["top3_name"]
        assert row["dominant_residual_abs_value"] >= row["top2_abs_value"]
        assert row["residual_cost"] > 0.0
        assert "0.5*sum" in row["residual_normalization"]

    summary = cell0_residual_markdown(rows)
    assert "Poor-quality threshold" in summary
    assert "Residual normalization" in summary
    assert "corrected ModelCase preview" in summary
    assert "superseded/not directly comparable" in summary
