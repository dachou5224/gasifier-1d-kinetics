#!/usr/bin/env python3
"""Generate Huayi bounded global multi-alpha diagnostic artifacts."""
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


def main() -> None:
    repo_root = _ensure_src_path()
    from model.global_multialpha_diagnostic import (
        GRID_COLUMNS,
        METRIC_COLUMNS,
        PREDICTION_COLUMNS,
        SUMMARY_COLUMNS,
        add_heldout_compensation_to_summary,
        global_multialpha_metric_rows,
        global_multialpha_prediction_rows,
        multialpha_markdown,
        select_global_multialpha_subsets,
    )
    from model.split_global_alpha_baseline import attach_split_to_model_cases

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cells", type=int, default=3)
    parser.add_argument("--out-dir", default=str(repo_root / "outputs" / "diagnostics"))
    parser.add_argument("--split-manifest", default=str(repo_root / "outputs" / "diagnostics" / "huayi_hybrid_split_manifest.csv"))
    parser.add_argument("--fixed-predictions", default=str(repo_root / "outputs" / "diagnostics" / "huayi_fixed_physics_split_predictions.csv"))
    parser.add_argument("--global-wgs-predictions", default=str(repo_root / "outputs" / "diagnostics" / "huayi_global_wgs_alpha_predictions.csv"))
    parser.add_argument("--global-char-predictions", default=str(repo_root / "outputs" / "diagnostics" / "huayi_global_char_oxidation_alpha_predictions.csv"))
    parser.add_argument("--alpha-wgs-xt-predictions", default=str(repo_root / "outputs" / "diagnostics" / "huayi_alpha_wgs_xt_predictions.csv"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_rows = pd.read_csv(args.split_manifest).to_dict("records")
    model_case_rows = attach_split_to_model_cases(_load_model_cases(), split_rows)

    summary, grid_rows = select_global_multialpha_subsets(model_case_rows, n_cells=args.n_cells)
    predictions = global_multialpha_prediction_rows(
        model_case_rows,
        summary,
        pd.read_csv(args.fixed_predictions).to_dict("records"),
        pd.read_csv(args.global_wgs_predictions).to_dict("records"),
        pd.read_csv(args.global_char_predictions).to_dict("records"),
        pd.read_csv(args.alpha_wgs_xt_predictions).to_dict("records"),
        n_cells=args.n_cells,
    )
    metrics = global_multialpha_metric_rows(predictions)
    summary = add_heldout_compensation_to_summary(summary, metrics)

    summary_path = out_dir / "huayi_global_multialpha_summary.csv"
    pred_path = out_dir / "huayi_global_multialpha_predictions.csv"
    metric_path = out_dir / "huayi_global_multialpha_metrics.csv"
    grid_path = out_dir / "huayi_global_multialpha_selection_grid.csv"
    md_path = out_dir / "huayi_global_multialpha_diagnostic.md"
    pd.DataFrame(summary, columns=list(SUMMARY_COLUMNS)).to_csv(summary_path, index=False)
    pd.DataFrame(predictions, columns=list(PREDICTION_COLUMNS)).to_csv(pred_path, index=False)
    pd.DataFrame(metrics, columns=list(METRIC_COLUMNS)).to_csv(metric_path, index=False)
    pd.DataFrame(grid_rows, columns=list(GRID_COLUMNS)).to_csv(grid_path, index=False)
    md_path.write_text(multialpha_markdown(summary, metrics), encoding="utf-8")

    print(f"wrote_summary={summary_path}")
    print(f"wrote_predictions={pred_path}")
    print(f"wrote_metrics={metric_path}")
    print(f"wrote_grid={grid_path}")
    print(f"wrote_report={md_path}")
    best = min(summary, key=lambda row: float(row["train_objective_final"]))
    print(f"best_train_subset={best['subset_name']}")
    print(f"best_train_objective={best['train_objective_final']:.6g}")
    print(f"best_alphas={{'alpha_wgs': {best['alpha_wgs']}, 'alpha_char_oxidation': {best['alpha_char_oxidation']}, 'alpha_volatile_oxidation': {best['alpha_volatile_oxidation']}}}")
    print(f"summary_rows={len(summary)} prediction_rows={len(predictions)} metric_rows={len(metrics)} grid_rows={len(grid_rows)}")


if __name__ == "__main__":
    main()
