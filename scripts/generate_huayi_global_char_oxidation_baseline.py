#!/usr/bin/env python3
"""Generate Huayi global char-oxidation alpha baseline artifacts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PREVIEW_SPECS = (
    ("phase1", "huayi_sy3_phase1_wgs1_shift", "/Users/liuzhen/AI-projects/gasifier-data-prep/outputs/model_cases/huayi_sy3_phase1_wgs1_shift.model_cases_preview.parquet"),
    ("phase2", "huayi_sy3sy2_phase2_wgs1_shift", "/Users/liuzhen/AI-projects/gasifier-data-prep/outputs/model_cases/huayi_sy3sy2_phase2_wgs1_shift.model_cases_preview.parquet"),
    ("phase3", "huayi_sy3sy2_phase3_wgs1_shift", "/Users/liuzhen/AI-projects/gasifier-data-prep/outputs/model_cases/huayi_sy3sy2_phase3_wgs1_shift.model_cases_preview.parquet"),
    ("phase4", "huayi_sy3sy2_phase4_wgs1_shift", "/Users/liuzhen/AI-projects/gasifier-data-prep/outputs/model_cases/huayi_sy3sy2_phase4_wgs1_shift.model_cases_preview.parquet"),
)


def _ensure_src_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return repo_root


def _load_model_cases() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for phase, dataset_id, path in PREVIEW_SPECS:
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            item = row.to_dict()
            item["phase"] = phase
            item["dataset_id"] = dataset_id
            rows.append(item)
    return rows


def _parse_alpha_grid(text: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in text.split(",") if item.strip())


def main() -> None:
    repo_root = _ensure_src_path()
    from model.global_char_oxidation_baseline import (
        DEFAULT_CHAR_ALPHA_GRID,
        GLOBAL_CHAR_GRID_COLUMNS,
        GLOBAL_CHAR_SUMMARY_COLUMNS,
        build_char_summary,
        char_baseline_markdown,
        char_oxidation_prediction_rows,
        select_global_char_oxidation_alpha,
    )
    from model.split_global_alpha_baseline import (
        METRIC_COLUMNS,
        PREDICTION_COLUMNS,
        attach_split_to_model_cases,
        compute_metric_rows,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cells", type=int, default=3)
    parser.add_argument("--alpha-grid", default=",".join(str(v) for v in DEFAULT_CHAR_ALPHA_GRID))
    parser.add_argument("--out-dir", default=str(repo_root / "outputs" / "diagnostics"))
    parser.add_argument("--split-manifest", default=str(repo_root / "outputs" / "diagnostics" / "huayi_hybrid_split_manifest.csv"))
    parser.add_argument("--fixed-metrics", default=str(repo_root / "outputs" / "diagnostics" / "huayi_fixed_physics_split_metrics.csv"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_rows = pd.read_csv(args.split_manifest).to_dict("records")
    model_case_rows = attach_split_to_model_cases(_load_model_cases(), split_rows)

    selected_grid_row, grid_rows = select_global_char_oxidation_alpha(
        model_case_rows,
        alpha_grid=_parse_alpha_grid(args.alpha_grid),
        n_cells=args.n_cells,
    )
    predictions = char_oxidation_prediction_rows(
        model_case_rows,
        selected_alpha=float(selected_grid_row["alpha_value"]),
        n_cells=args.n_cells,
    )
    fixed_metrics = pd.read_csv(args.fixed_metrics).to_dict("records")
    fixed_n_metrics = [row for row in fixed_metrics if int(row["N_cells"]) == int(args.n_cells)]
    metrics = compute_metric_rows(
        predictions,
        evidence_boundary="non-learned global char-oxidation scalar comparator only; not trained hybrid",
        fixed_metric_rows=fixed_n_metrics,
    )
    summary = build_char_summary(selected_grid_row, grid_rows, n_cells=args.n_cells)

    summary_path = out_dir / "huayi_global_char_oxidation_alpha_baseline.csv"
    pred_path = out_dir / "huayi_global_char_oxidation_alpha_predictions.csv"
    metric_path = out_dir / "huayi_global_char_oxidation_alpha_metrics.csv"
    grid_path = out_dir / "huayi_global_char_oxidation_alpha_selection_grid.csv"
    md_path = out_dir / "huayi_global_char_oxidation_alpha_baseline.md"
    pd.DataFrame(summary, columns=list(GLOBAL_CHAR_SUMMARY_COLUMNS)).to_csv(summary_path, index=False)
    pd.DataFrame(predictions, columns=list(PREDICTION_COLUMNS)).to_csv(pred_path, index=False)
    pd.DataFrame(metrics, columns=list(METRIC_COLUMNS)).to_csv(metric_path, index=False)
    pd.DataFrame(grid_rows, columns=list(GLOBAL_CHAR_GRID_COLUMNS)).to_csv(grid_path, index=False)
    md_path.write_text(char_baseline_markdown(summary, metrics, grid_rows), encoding="utf-8")

    print(f"wrote_summary={summary_path}")
    print(f"wrote_predictions={pred_path}")
    print(f"wrote_metrics={metric_path}")
    print(f"wrote_grid={grid_path}")
    print(f"selected_alpha={summary[0]['selected_alpha']:.6g}")
    print(f"hit_lower_bound={summary[0]['hit_lower_bound']}")
    print(f"hit_upper_bound={summary[0]['hit_upper_bound']}")
    print(f"training_objective={summary[0]['training_objective_initial']:.6g}->{summary[0]['training_objective_final']:.6g}")
    print(f"training_objective_delta={summary[0]['training_objective_delta']:.6g}")
    print(f"validation_excluded_from_selection={summary[0]['validation_excluded_from_selection']}")
    print(f"test_excluded_from_selection={summary[0]['test_excluded_from_selection']}")


if __name__ == "__main__":
    main()
