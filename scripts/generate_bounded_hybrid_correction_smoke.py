#!/usr/bin/env python3
"""Generate a bounded correction smoke artifact for selected Huayi ModelCases."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PREVIEW_SPECS = (
    (
        "phase1",
        "huayi_sy3_phase1_wgs1_shift",
        "/Users/liuzhen/AI-projects/gasifier-data-prep/outputs/model_cases/huayi_sy3_phase1_wgs1_shift.model_cases_preview.parquet",
    ),
    (
        "phase2",
        "huayi_sy3sy2_phase2_wgs1_shift",
        "/Users/liuzhen/AI-projects/gasifier-data-prep/outputs/model_cases/huayi_sy3sy2_phase2_wgs1_shift.model_cases_preview.parquet",
    ),
    (
        "phase3",
        "huayi_sy3sy2_phase3_wgs1_shift",
        "/Users/liuzhen/AI-projects/gasifier-data-prep/outputs/model_cases/huayi_sy3sy2_phase3_wgs1_shift.model_cases_preview.parquet",
    ),
    (
        "phase4",
        "huayi_sy3sy2_phase4_wgs1_shift",
        "/Users/liuzhen/AI-projects/gasifier-data-prep/outputs/model_cases/huayi_sy3sy2_phase4_wgs1_shift.model_cases_preview.parquet",
    ),
)


def _ensure_src_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return repo_root


def _load_selected_model_cases(cases_per_phase: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for phase, dataset_id, path in PREVIEW_SPECS:
        df = pd.read_parquet(path)
        if int(cases_per_phase) > 0:
            df = df.head(int(cases_per_phase))
        for _, row in df.iterrows():
            item = row.to_dict()
            item["phase"] = phase
            item["dataset_id"] = dataset_id
            rows.append(item)
    return rows


def main() -> None:
    repo_root = _ensure_src_path()
    from model.bounded_hybrid_correction_smoke import (
        BOUNDED_CORRECTION_BATCH_COLUMNS,
        BOUNDED_CORRECTION_SUMMARY_COLUMNS,
        BOUNDED_CORRECTION_TRAJECTORY_COLUMNS,
        build_bounded_correction_batch_summary,
        bounded_correction_markdown,
        run_bounded_wgs_correction_smoke,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cells", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--initial-value", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    parser.add_argument("--cases-per-phase", type=int, default=0)
    parser.add_argument("--out-dir", default=str(repo_root / "outputs" / "diagnostics"))
    parser.add_argument("--basename", default="huayi_bounded_hybrid_correction_batch_smoke")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_case_rows = _load_selected_model_cases(args.cases_per_phase)
    summaries, trajectory = run_bounded_wgs_correction_smoke(
        model_case_rows,
        n_cells=args.n_cells,
        steps=args.steps,
        initial_value=args.initial_value,
        learning_rate=args.learning_rate,
    )
    batch_rows = build_bounded_correction_batch_summary(summaries)

    summary_path = out_dir / f"{args.basename}.csv"
    trajectory_path = out_dir / f"{args.basename}_trajectory.csv"
    batch_path = out_dir / f"{args.basename}_batch_summary.csv"
    md_path = out_dir / f"{args.basename}.md"
    pd.DataFrame(summaries, columns=list(BOUNDED_CORRECTION_SUMMARY_COLUMNS)).to_csv(summary_path, index=False)
    pd.DataFrame(trajectory, columns=list(BOUNDED_CORRECTION_TRAJECTORY_COLUMNS)).to_csv(trajectory_path, index=False)
    pd.DataFrame(batch_rows, columns=list(BOUNDED_CORRECTION_BATCH_COLUMNS)).to_csv(batch_path, index=False)
    md_path.write_text(bounded_correction_markdown(summaries, trajectory, batch_rows), encoding="utf-8")

    print(f"wrote_summary={summary_path}")
    print(f"wrote_trajectory={trajectory_path}")
    print(f"wrote_batch_summary={batch_path}")
    print(f"wrote_markdown={md_path}")
    print(f"summary_rows={len(summaries)}")
    print(f"trajectory_rows={len(trajectory)}")
    print(f"batch_rows={len(batch_rows)}")
    if batch_rows:
        overall = batch_rows[0]
        print(
            "batch_summary="
            f"n_cases={overall['n_cases']},"
            f"n_loss_decreased={overall['n_loss_decreased']},"
            f"median_loss_delta={overall['median_loss_delta']:.6g},"
            f"median_relative_loss_delta={overall['median_relative_loss_delta']:.6g},"
            f"lower_hits={overall['n_hit_lower_bound']},"
            f"upper_hits={overall['n_hit_upper_bound']},"
            f"recommendation={overall['recommendation']}"
        )
    print("losses=" + ",".join(f"{row['case_id']}:{row['loss_initial']:.6g}->{row['loss_final']:.6g}" for row in summaries))


if __name__ == "__main__":
    main()
