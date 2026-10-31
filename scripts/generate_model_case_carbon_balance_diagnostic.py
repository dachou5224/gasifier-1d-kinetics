#!/usr/bin/env python3
"""Generate apparent gas-phase carbon recovery artifacts from ModelCase preview."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _ensure_src_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return repo_root


def main() -> None:
    repo_root = _ensure_src_path()
    from model.model_case_carbon_balance_diagnostic import (
        REQUIRED_CARBON_BALANCE_COLUMNS,
        audit_markdown_summary,
        build_carbon_balance_rows,
        markdown_summary,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preview",
        default="/Users/liuzhen/AI-projects/gasifier-data-prep/outputs/model_cases/sy3_trial_0928_feed_product.model_cases_preview.parquet",
    )
    parser.add_argument("--out-dir", default=str(repo_root / "outputs" / "diagnostics"))
    parser.add_argument("--basename", default="model_case_carbon_balance_diagnostic")
    args = parser.parse_args()

    preview = Path(args.preview)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(preview)
    rows = build_carbon_balance_rows([row for _, row in df.iterrows()])
    table = pd.DataFrame(rows, columns=list(REQUIRED_CARBON_BALANCE_COLUMNS))

    csv_path = out_dir / f"{args.basename}.csv"
    md_path = out_dir / f"{args.basename}.md"
    audit_path = out_dir / "model_case_carbon_balance_audit.md"
    table.to_csv(csv_path, index=False)
    md_path.write_text(markdown_summary(rows), encoding="utf-8")
    audit_path.write_text(audit_markdown_summary(rows), encoding="utf-8")

    print(f"wrote_csv={csv_path}")
    print(f"wrote_markdown={md_path}")
    print(f"wrote_audit={audit_path}")
    print(f"rows={len(table)}")
    print(
        "recovery_min={:.6f}".format(
            float(table["apparent_gas_phase_C_recovery"].min())
        )
    )
    print(
        "recovery_max={:.6f}".format(
            float(table["apparent_gas_phase_C_recovery"].max())
        )
    )
    print(
        "coal_only_yield_min={:.6f}".format(
            float(table["coal_only_apparent_gas_C_yield"].min())
        )
    )
    print(
        "gross_98pct_dry_flow_factor_max={:.3f}".format(
            float(table["required_to_observed_dry_syngas_factor_gross_98pct"].max())
        )
    )


if __name__ == "__main__":
    main()
