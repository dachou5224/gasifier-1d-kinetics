#!/usr/bin/env python3
"""Generate pilot fixed-physics diagnostic baseline artifacts from ModelCase preview."""
from __future__ import annotations

import argparse
import os
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
    from model.model_case_diagnostic_baseline import (
        REQUIRED_DIAGNOSTIC_COLUMNS,
        build_diagnostic_rows,
        markdown_summary,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preview",
        default="/Users/liuzhen/AI-projects/gasifier-data-prep/outputs/model_cases/sy3_trial_0928_feed_product.model_cases_preview.parquet",
    )
    parser.add_argument("--n-cells", type=int, nargs="+", default=[5])
    parser.add_argument("--out-dir", default=str(repo_root / "outputs" / "diagnostics"))
    parser.add_argument("--basename", default="pilot_fixed_physics_diagnostic_baseline")
    args = parser.parse_args()

    preview = Path(args.preview)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(preview)
    rows = build_diagnostic_rows([row for _, row in df.iterrows()], args.n_cells)
    table = pd.DataFrame(rows, columns=list(REQUIRED_DIAGNOSTIC_COLUMNS))

    csv_path = out_dir / f"{args.basename}.csv"
    md_path = out_dir / f"{args.basename}.md"
    table.to_csv(csv_path, index=False)
    md_path.write_text(markdown_summary(rows), encoding="utf-8")

    print(f"wrote_csv={csv_path}")
    print(f"wrote_markdown={md_path}")
    print(f"rows={len(table)}")
    print(f"n_cells={','.join(str(n) for n in args.n_cells)}")


if __name__ == "__main__":
    main()
