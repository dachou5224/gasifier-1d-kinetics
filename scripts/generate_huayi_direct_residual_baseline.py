#!/usr/bin/env python3
"""Generate Huayi direct outlet residual correction baseline artifacts."""
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
    from model.direct_residual_baseline import (
        DIRECT_RESIDUAL_METRIC_COLUMNS,
        DIRECT_RESIDUAL_MODEL_COLUMNS,
        DIRECT_RESIDUAL_PREDICTION_COLUMNS,
        direct_residual_markdown,
        direct_residual_metric_rows,
        direct_residual_model_audit_rows,
        direct_residual_prediction_rows,
        fit_direct_residual_models,
        operating_feature_rows,
        validate_split_for_direct_residual,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(repo_root / "outputs" / "diagnostics"))
    parser.add_argument("--split-manifest", default=str(repo_root / "outputs" / "diagnostics" / "huayi_hybrid_split_manifest.csv"))
    parser.add_argument("--fixed-predictions", default=str(repo_root / "outputs" / "diagnostics" / "huayi_fixed_physics_split_predictions.csv"))
    parser.add_argument("--ridge-lambda", type=float, default=1.0e-3)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_rows = pd.read_csv(args.split_manifest).to_dict("records")
    split_validation = validate_split_for_direct_residual(split_rows)
    fixed_predictions = pd.read_csv(args.fixed_predictions).to_dict("records")
    feature_rows = operating_feature_rows(_load_model_cases())

    models = fit_direct_residual_models(fixed_predictions, feature_rows, ridge_lambda=args.ridge_lambda)
    prediction_rows = direct_residual_prediction_rows(fixed_predictions, feature_rows, models)
    metric_rows = direct_residual_metric_rows(prediction_rows)
    model_rows = direct_residual_model_audit_rows(models)

    pred_path = out_dir / "huayi_direct_residual_baseline_predictions.csv"
    metric_path = out_dir / "huayi_direct_residual_baseline_metrics.csv"
    model_path = out_dir / "huayi_direct_residual_baseline_models.csv"
    md_path = out_dir / "huayi_direct_residual_baseline.md"

    pd.DataFrame(prediction_rows, columns=list(DIRECT_RESIDUAL_PREDICTION_COLUMNS)).to_csv(pred_path, index=False)
    pd.DataFrame(metric_rows, columns=list(DIRECT_RESIDUAL_METRIC_COLUMNS)).to_csv(metric_path, index=False)
    pd.DataFrame(model_rows, columns=list(DIRECT_RESIDUAL_MODEL_COLUMNS)).to_csv(model_path, index=False)
    md_path.write_text(direct_residual_markdown(prediction_rows, metric_rows, model_rows, split_validation), encoding="utf-8")

    print(f"wrote_predictions={pred_path}")
    print(f"wrote_metrics={metric_path}")
    print(f"wrote_models={model_path}")
    print(f"wrote_summary={md_path}")
    print(f"split_counts={split_validation['split_counts']}")
    print(f"prediction_rows={len(prediction_rows)}")
    print(f"metric_rows={len(metric_rows)}")
    print(f"model_rows={len(model_rows)}")
    print(f"validation_excluded_from_fit={split_validation['validation_excluded_from_fit']}")
    print(f"test_excluded_from_fit={split_validation['test_excluded_from_fit']}")


if __name__ == "__main__":
    main()
