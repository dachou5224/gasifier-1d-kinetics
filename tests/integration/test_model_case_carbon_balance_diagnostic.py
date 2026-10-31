from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from model.model_case_carbon_balance_diagnostic import (
    REQUIRED_CARBON_BALANCE_COLUMNS,
    audit_markdown_summary,
    build_carbon_balance_rows,
    markdown_summary,
)


def _preview_path() -> Path:
    return Path(
        "/Users/liuzhen/AI-projects/gasifier-data-prep/outputs/model_cases/"
        "sy3_trial_0928_feed_product.model_cases_preview.parquet"
    )


def test_model_case_carbon_balance_preview_rows_have_required_schema():
    pd = pytest.importorskip("pandas")
    preview = _preview_path()
    if not preview.exists():
        pytest.skip("dataprep model_cases preview is not available")

    df = pd.read_parquet(preview)
    rows = build_carbon_balance_rows([row for _, row in df.iterrows()])

    assert len(rows) == len(df)
    assert len(rows) >= 1
    assert tuple(rows[0].keys()) == REQUIRED_CARBON_BALANCE_COLUMNS
    for row in rows:
        assert row["case_id"]
        assert row["source_candidate_id"]
        assert row["independence_group"]
        assert (
            row["interpretation_status"]
            == "corrected_primary_coal_flow_available_field_gas_carbon_check_not_closure"
        )
        assert row["total_available_C_in_kmol_s"] > 0.0
        assert row["target_dry_gas_C_out_kmol_s"] > 0.0
        assert math.isfinite(row["apparent_gas_phase_C_recovery"])
        assert math.isfinite(row["apparent_C_gap"])
        assert math.isfinite(row["coal_only_apparent_gas_C_yield"])
        assert math.isfinite(row["gross_recovery_if_c_f_monitoring_sum_used"])
        assert row["apparent_gas_phase_C_recovery"] > row["gross_recovery_if_c_f_monitoring_sum_used"]
        assert row["c_f_monitoring_alias_warning"] == "do_not_use_c_f_monitoring_sum_as_primary_coal_feed"
        assert row["required_to_observed_dry_syngas_factor_gross_98pct"] > 1.0


def test_model_case_carbon_balance_markdown_states_boundary():
    row = {column: 0 for column in REQUIRED_CARBON_BALANCE_COLUMNS}
    row.update(
        {
            "case_id": "case",
            "source_candidate_id": "segment",
            "independence_group": "IG",
            "missing_fields": "",
            "interpretation_status": "corrected_primary_coal_flow_available_field_gas_carbon_check_not_closure",
        }
    )

    text = markdown_summary([row])

    assert "available-field dry-gas carbon check" in text
    assert "not a full carbon-balance closure" in text
    assert "excludes unconverted solid carbon" in text


def test_model_case_carbon_balance_audit_flags_low_recovery_as_suspect():
    row = {column: 0 for column in REQUIRED_CARBON_BALANCE_COLUMNS}
    row.update(
        {
            "case_id": "case",
            "source_candidate_id": "segment",
            "independence_group": "IG",
            "coal_C_kmol_s": 1.4,
            "carrier_CO2_C_kmol_s": 0.12,
            "target_dry_syngas_Nm3_h": 120000.0,
            "apparent_gas_phase_C_recovery": 0.69,
            "coal_only_apparent_gas_C_yield": 0.66,
            "required_dry_syngas_Nm3_h_for_98pct_gross_C_recovery": 171000.0,
            "required_to_observed_dry_syngas_factor_gross_98pct": 1.425,
            "required_dry_syngas_Nm3_h_for_98pct_coal_C_yield": 170000.0,
            "required_to_observed_dry_syngas_factor_coal_98pct": 1.417,
            "gross_recovery_if_c_f_monitoring_sum_used": 0.69,
            "coal_only_yield_if_c_f_monitoring_sum_used": 0.66,
            "missing_fields": "",
            "interpretation_status": "corrected_primary_coal_flow_available_field_gas_carbon_check_not_closure",
        }
    )

    text = audit_markdown_summary([row])

    assert "round-29/30 low apparent recovery values are superseded" in text
    assert "C-F monitoring sum is treated as the primary coal feed" in text
    assert "not complete carbon-balance closure" in text
