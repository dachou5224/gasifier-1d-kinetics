from __future__ import annotations

import math
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))


def test_ift_cost_schema_and_summary_with_fixture_rows():
    from model.ift_cost_diagnostic import (
        IFT_COST_COLUMNS,
        METHOD_IFT,
        METHOD_UNROLLED,
        annotate_reference_differences,
        collect_protocol_metadata,
        ift_cost_markdown,
        ift_cost_protocol_markdown,
        timeout_row,
    )

    ift_row = {column: math.nan for column in IFT_COST_COLUMNS}
    ift_row.update(
        {
            "method": METHOD_IFT,
            "N_cells": 1,
            "parameter": "HeatLossPercent",
            "output": "final_T_K",
            "newton_iterations": 8,
            "status": "ok",
            "compile_or_cold_time_s": 1.0,
            "warm_eval_time_s": 0.1,
            "warm_repeats": 2,
            "gradient_value": -1.0,
            "residual_max_abs": 1.0e-7,
        }
    )
    direct_row = {column: math.nan for column in IFT_COST_COLUMNS}
    direct_row.update(
        {
            "method": METHOD_UNROLLED,
            "N_cells": 1,
            "parameter": "HeatLossPercent",
            "output": "final_T_K",
            "newton_iterations": 8,
            "status": "ok",
            "compile_or_cold_time_s": 2.0,
            "warm_eval_time_s": 0.2,
            "warm_repeats": 2,
            "gradient_value": -1.01,
            "residual_max_abs": 1.0e-7,
        }
    )
    rows = annotate_reference_differences([ift_row, direct_row, timeout_row(METHOD_UNROLLED, 4, newton_iterations=8, timeout_s=90)])

    assert set(rows[0]) == set(IFT_COST_COLUMNS)
    assert any(row["method"] == METHOD_IFT and row["status"] == "ok" for row in rows)
    assert any(row["status"] == "timeout" for row in rows)
    direct = next(row for row in rows if row["method"] == METHOD_UNROLLED and row["N_cells"] == 1)
    assert direct["reference_method"] == METHOD_IFT
    assert math.isclose(float(direct["abs_gradient_diff"]), 0.01)
    summary = ift_cost_markdown(rows)
    assert "IFT vs direct-unrolled AD cost diagnostic" in summary
    assert "not calibration accuracy evidence" in summary

    protocol = ift_cost_protocol_markdown(collect_protocol_metadata(timeout_s=90.0, warm_repeats=2))
    assert "Machine and software" in protocol
    assert "compile_or_cold_time_s" in protocol
    assert "warm_eval_time_s" in protocol
    assert "timeout_s=90.0" in protocol
    assert "direct_unrolled_ad" in protocol
    assert "ift_custom_vjp" in protocol
    assert "not production reactor benchmarks" in protocol
