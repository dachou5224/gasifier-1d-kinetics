#!/usr/bin/env python3
"""Generate fixed-physics sensitivity ranking diagnostics from ModelCase preview."""
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
    from model.model_case_sensitivity import (
        RANKING_COLUMNS,
        SENSITIVITY_COLUMNS,
        build_parameter_ranking,
        build_sensitivity_rows,
        sensitivity_markdown,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preview",
        default="/Users/liuzhen/AI-projects/gasifier-data-prep/outputs/model_cases/sy3_trial_0928_feed_product.model_cases_preview.parquet",
    )
    parser.add_argument("--n-cells", type=int, default=5)
    parser.add_argument("--perturbation-fraction", type=float, default=0.02)
    parser.add_argument("--out-dir", default=str(repo_root / "outputs" / "diagnostics"))
    parser.add_argument("--basename", default="fixed_physics_sensitivity")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.preview)
    model_case_rows = [row for _, row in df.iterrows()]
    rows = build_sensitivity_rows(
        model_case_rows,
        n_cells=args.n_cells,
        perturbation_fraction=args.perturbation_fraction,
    )
    ranking = build_parameter_ranking(rows)

    details_path = out_dir / f"{args.basename}_diagnostic.csv"
    ranking_path = out_dir / f"{args.basename}_ranking.csv"
    markdown_path = out_dir / f"{args.basename}_summary.md"
    pd.DataFrame(rows, columns=list(SENSITIVITY_COLUMNS)).to_csv(details_path, index=False)
    pd.DataFrame(ranking, columns=list(RANKING_COLUMNS)).to_csv(ranking_path, index=False)
    markdown_path.write_text(sensitivity_markdown(rows, ranking), encoding="utf-8")

    print(f"wrote_details_csv={details_path}")
    print(f"wrote_ranking_csv={ranking_path}")
    print(f"wrote_markdown={markdown_path}")
    print(f"rows={len(rows)}")
    print(f"ranking_rows={len(ranking)}")
    print("ranking=" + ",".join(f"{row['rank']}:{row['parameter_name']}:{row['mean_abs_normalized_sensitivity']:.6g}" for row in ranking))


if __name__ == "__main__":
    main()
