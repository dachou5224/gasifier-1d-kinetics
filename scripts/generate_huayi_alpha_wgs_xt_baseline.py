#!/usr/bin/env python3
"""Generate Huayi minimal alpha_wgs(x_t) baseline artifacts."""
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
    from model.alpha_wgs_xt_baseline import (
        ALPHA_WGS_XT_AUDIT_COLUMNS,
        ALPHA_WGS_XT_METRIC_COLUMNS,
        ALPHA_WGS_XT_PREDICTION_COLUMNS,
        ALPHA_WGS_XT_SUMMARY_COLUMNS,
        alpha_wgs_xt_audit_rows,
        alpha_wgs_xt_markdown,
        alpha_wgs_xt_metric_rows,
        alpha_wgs_xt_prediction_rows,
        build_alpha_wgs_xt_summary,
        feature_rows_with_split,
        fit_train_feature_scaler,
        select_alpha_wgs_xt_model,
    )
    from model.split_global_alpha_baseline import attach_split_to_model_cases

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cells", type=int, default=3)
    parser.add_argument("--out-dir", default=str(repo_root / "outputs" / "diagnostics"))
    parser.add_argument("--split-manifest", default=str(repo_root / "outputs" / "diagnostics" / "huayi_hybrid_split_manifest.csv"))
    parser.add_argument("--fixed-predictions", default=str(repo_root / "outputs" / "diagnostics" / "huayi_fixed_physics_split_predictions.csv"))
    parser.add_argument("--global-wgs-predictions", default=str(repo_root / "outputs" / "diagnostics" / "huayi_global_wgs_alpha_predictions.csv"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_rows = pd.read_csv(args.split_manifest).to_dict("records")
    model_case_rows = attach_split_to_model_cases(_load_model_cases(), split_rows)
    features = feature_rows_with_split(model_case_rows)
    scaler = fit_train_feature_scaler(features)
    selected_model, candidate_rows = select_alpha_wgs_xt_model(model_case_rows, features, scaler, n_cells=args.n_cells)
    fixed_predictions = pd.read_csv(args.fixed_predictions).to_dict("records")
    global_predictions = pd.read_csv(args.global_wgs_predictions).to_dict("records")
    predictions = alpha_wgs_xt_prediction_rows(
        model_case_rows,
        features,
        scaler,
        selected_model,
        fixed_predictions,
        global_predictions,
        n_cells=args.n_cells,
    )
    metrics = alpha_wgs_xt_metric_rows(predictions)
    summary = build_alpha_wgs_xt_summary(selected_model, candidate_rows, n_cells=args.n_cells)
    audit = alpha_wgs_xt_audit_rows(scaler, selected_model, candidate_rows, predictions)

    summary_path = out_dir / "huayi_alpha_wgs_xt_baseline.csv"
    pred_path = out_dir / "huayi_alpha_wgs_xt_predictions.csv"
    metric_path = out_dir / "huayi_alpha_wgs_xt_metrics.csv"
    audit_path = out_dir / "huayi_alpha_wgs_xt_model_audit.csv"
    md_path = out_dir / "huayi_alpha_wgs_xt_baseline.md"
    pd.DataFrame(summary, columns=list(ALPHA_WGS_XT_SUMMARY_COLUMNS)).to_csv(summary_path, index=False)
    pd.DataFrame(predictions, columns=list(ALPHA_WGS_XT_PREDICTION_COLUMNS)).to_csv(pred_path, index=False)
    pd.DataFrame(metrics, columns=list(ALPHA_WGS_XT_METRIC_COLUMNS)).to_csv(metric_path, index=False)
    pd.DataFrame(audit, columns=list(ALPHA_WGS_XT_AUDIT_COLUMNS)).to_csv(audit_path, index=False)
    md_path.write_text(alpha_wgs_xt_markdown(summary, metrics, audit), encoding="utf-8")

    print(f"wrote_summary={summary_path}")
    print(f"wrote_predictions={pred_path}")
    print(f"wrote_metrics={metric_path}")
    print(f"wrote_model_audit={audit_path}")
    print(f"wrote_report={md_path}")
    print(f"selected_candidate={summary[0]['selected_candidate_name']}")
    print(f"selected_nonzero_feature={summary[0]['selected_nonzero_feature']}")
    print(f"train_objective={summary[0]['train_objective_initial']:.6g}->{summary[0]['train_objective_final']:.6g}")
    print(f"validation_excluded_from_training={summary[0]['validation_excluded_from_training']}")
    print(f"test_excluded_from_training={summary[0]['test_excluded_from_training']}")
    print(f"prediction_rows={len(predictions)}")


if __name__ == "__main__":
    main()
