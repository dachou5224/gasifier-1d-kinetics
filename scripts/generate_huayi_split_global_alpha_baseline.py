#!/usr/bin/env python3
"""Generate Huayi chronological split and global WGS alpha baseline artifacts."""
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
    from model.split_global_alpha_baseline import (
        DEFAULT_WGS_ALPHA_GRID,
        GLOBAL_WGS_GRID_COLUMNS,
        GLOBAL_WGS_SUMMARY_COLUMNS,
        METRIC_COLUMNS,
        PREDICTION_COLUMNS,
        SPLIT_MANIFEST_COLUMNS,
        attach_split_to_model_cases,
        build_split_manifest,
        build_wgs_summary,
        compute_metric_rows,
        fixed_physics_prediction_rows,
        metrics_markdown,
        select_global_wgs_alpha,
        split_manifest_markdown,
        validate_split_manifest,
        wgs_baseline_markdown,
        wgs_prediction_rows,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cells", type=int, default=3)
    parser.add_argument("--alpha-grid", default=",".join(str(v) for v in DEFAULT_WGS_ALPHA_GRID))
    parser.add_argument("--out-dir", default=str(repo_root / "outputs" / "diagnostics"))
    parser.add_argument("--contract-audit", default=str(repo_root / "outputs" / "diagnostics" / "huayi_expanded_model_case_contract_audit.csv"))
    parser.add_argument("--fixed-baseline", default=str(repo_root / "outputs" / "diagnostics" / "huayi_expanded_fixed_physics_diagnostic_baseline.csv"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    contract_rows = pd.read_csv(args.contract_audit).to_dict("records")
    split_rows = build_split_manifest(contract_rows)
    split_validation = validate_split_manifest(split_rows)

    split_path = out_dir / "huayi_hybrid_split_manifest.csv"
    split_md_path = out_dir / "huayi_hybrid_split_manifest.md"
    pd.DataFrame(split_rows, columns=list(SPLIT_MANIFEST_COLUMNS)).to_csv(split_path, index=False)
    split_md_path.write_text(split_manifest_markdown(split_rows, split_validation), encoding="utf-8")

    fixed_rows = pd.read_csv(args.fixed_baseline).to_dict("records")
    fixed_predictions = fixed_physics_prediction_rows(fixed_rows, split_rows)
    fixed_metrics = compute_metric_rows(
        fixed_predictions,
        evidence_boundary="split-aware fixed-physics diagnostic baseline only; not calibrated accuracy",
    )
    fixed_pred_path = out_dir / "huayi_fixed_physics_split_predictions.csv"
    fixed_metric_path = out_dir / "huayi_fixed_physics_split_metrics.csv"
    fixed_md_path = out_dir / "huayi_fixed_physics_split_metrics.md"
    pd.DataFrame(fixed_predictions, columns=list(PREDICTION_COLUMNS)).to_csv(fixed_pred_path, index=False)
    pd.DataFrame(fixed_metrics, columns=list(METRIC_COLUMNS)).to_csv(fixed_metric_path, index=False)
    fixed_md_path.write_text(
        metrics_markdown("Huayi fixed-physics split metrics", fixed_metrics, "split-aware fixed-physics diagnostic baseline only; not calibrated accuracy"),
        encoding="utf-8",
    )

    model_case_rows = attach_split_to_model_cases(_load_model_cases(), split_rows)
    selected_grid_row, grid_rows = select_global_wgs_alpha(
        model_case_rows,
        alpha_grid=_parse_alpha_grid(args.alpha_grid),
        n_cells=args.n_cells,
    )
    wgs_predictions = wgs_prediction_rows(model_case_rows, selected_alpha=float(selected_grid_row["alpha_value"]), n_cells=args.n_cells)
    fixed_n3_metrics = [row for row in fixed_metrics if int(row["N_cells"]) == int(args.n_cells)]
    wgs_metrics = compute_metric_rows(
        wgs_predictions,
        evidence_boundary="non-learned global bounded WGS scalar comparator only; not trained hybrid",
        fixed_metric_rows=fixed_n3_metrics,
    )
    wgs_summary = build_wgs_summary(selected_grid_row, grid_rows, n_cells=args.n_cells)

    wgs_summary_path = out_dir / "huayi_global_wgs_alpha_baseline.csv"
    wgs_pred_path = out_dir / "huayi_global_wgs_alpha_predictions.csv"
    wgs_metric_path = out_dir / "huayi_global_wgs_alpha_metrics.csv"
    wgs_grid_path = out_dir / "huayi_global_wgs_alpha_selection_grid.csv"
    wgs_md_path = out_dir / "huayi_global_wgs_alpha_baseline.md"
    pd.DataFrame(wgs_summary, columns=list(GLOBAL_WGS_SUMMARY_COLUMNS)).to_csv(wgs_summary_path, index=False)
    pd.DataFrame(wgs_predictions, columns=list(PREDICTION_COLUMNS)).to_csv(wgs_pred_path, index=False)
    pd.DataFrame(wgs_metrics, columns=list(METRIC_COLUMNS)).to_csv(wgs_metric_path, index=False)
    pd.DataFrame(grid_rows, columns=list(GLOBAL_WGS_GRID_COLUMNS)).to_csv(wgs_grid_path, index=False)
    wgs_md_path.write_text(wgs_baseline_markdown(wgs_summary, wgs_metrics, grid_rows), encoding="utf-8")

    print(f"wrote_split={split_path}")
    print(f"wrote_fixed_predictions={fixed_pred_path}")
    print(f"wrote_fixed_metrics={fixed_metric_path}")
    print(f"wrote_wgs_summary={wgs_summary_path}")
    print(f"wrote_wgs_predictions={wgs_pred_path}")
    print(f"wrote_wgs_metrics={wgs_metric_path}")
    print(f"wrote_wgs_grid={wgs_grid_path}")
    print(f"split_counts={split_validation['split_counts']}")
    print(f"selected_alpha={wgs_summary[0]['selected_alpha']:.6g}")
    print(f"training_objective={wgs_summary[0]['training_objective_initial']:.6g}->{wgs_summary[0]['training_objective_final']:.6g}")
    print(f"validation_excluded_from_selection={wgs_summary[0]['validation_excluded_from_selection']}")
    print(f"test_excluded_from_selection={wgs_summary[0]['test_excluded_from_selection']}")


if __name__ == "__main__":
    main()
