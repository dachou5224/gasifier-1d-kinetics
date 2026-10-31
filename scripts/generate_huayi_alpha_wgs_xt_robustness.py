#!/usr/bin/env python3
"""Generate Huayi alpha_wgs(x_t) robustness and identifiability diagnostics."""
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


def _parse_float_grid(text: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in text.split(",") if item.strip())


def main() -> None:
    repo_root = _ensure_src_path()
    from model.alpha_wgs_xt_baseline import (
        alpha_wgs_xt_prediction_rows,
        feature_rows_with_split,
        fit_train_feature_scaler,
        select_alpha_wgs_xt_model,
    )
    from model.alpha_wgs_xt_robustness import (
        FEATURE_STABILITY_COLUMNS,
        NULL_CONTROL_COLUMNS,
        REGULARIZATION_GRID_COLUMNS,
        SENSITIVITY_PROXY_COLUMNS,
        feature_stability_rows,
        null_control_rows,
        permute_train_feature_rows,
        regularization_sweep_rows,
        robustness_markdown,
        select_candidate_from_cached_rows,
        sensitivity_proxy_rows,
    )
    from model.split_global_alpha_baseline import attach_split_to_model_cases

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cells", type=int, default=3)
    parser.add_argument("--regularization-grid", default="0,0.0001,0.001,0.01,0.1")
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
    fixed_predictions = pd.read_csv(args.fixed_predictions).to_dict("records")
    global_predictions = pd.read_csv(args.global_wgs_predictions).to_dict("records")

    selected_zero, candidate_rows = select_alpha_wgs_xt_model(model_case_rows, features, scaler, n_cells=args.n_cells, regularization=0.0)
    reg_grid = _parse_float_grid(args.regularization_grid)
    selected_by_penalty = {
        float(penalty): select_candidate_from_cached_rows(candidate_rows, features, scaler, regularization=float(penalty))
        for penalty in reg_grid
    }
    unique_candidates = {str(row["candidate_name"]): row for row in selected_by_penalty.values()}
    prediction_rows_by_candidate = {
        name: alpha_wgs_xt_prediction_rows(
            model_case_rows,
            features,
            scaler,
            candidate,
            fixed_predictions,
            global_predictions,
            n_cells=args.n_cells,
        )
        for name, candidate in unique_candidates.items()
    }
    reg_rows = regularization_sweep_rows(candidate_rows, features, scaler, prediction_rows_by_candidate, regularization_grid=reg_grid)
    stability_rows = feature_stability_rows(candidate_rows, features, scaler)

    null_features = permute_train_feature_rows(features)
    null_scaler = fit_train_feature_scaler(null_features)
    selected_null, _null_candidates = select_alpha_wgs_xt_model(model_case_rows, null_features, null_scaler, n_cells=args.n_cells, regularization=0.001)
    null_rows = null_control_rows(selected_null)
    sensitivity_rows = sensitivity_proxy_rows(model_case_rows, n_cells=args.n_cells)

    reg_path = out_dir / "huayi_alpha_wgs_xt_robustness_grid.csv"
    feature_path = out_dir / "huayi_alpha_wgs_xt_feature_stability.csv"
    null_path = out_dir / "huayi_alpha_wgs_xt_null_control.csv"
    sens_path = out_dir / "huayi_alpha_wgs_sensitivity_proxy.csv"
    md_path = out_dir / "huayi_alpha_wgs_xt_robustness.md"
    pd.DataFrame(reg_rows, columns=list(REGULARIZATION_GRID_COLUMNS)).to_csv(reg_path, index=False)
    pd.DataFrame(stability_rows, columns=list(FEATURE_STABILITY_COLUMNS)).to_csv(feature_path, index=False)
    pd.DataFrame(null_rows, columns=list(NULL_CONTROL_COLUMNS)).to_csv(null_path, index=False)
    pd.DataFrame(sensitivity_rows, columns=list(SENSITIVITY_PROXY_COLUMNS)).to_csv(sens_path, index=False)
    md_path.write_text(robustness_markdown(reg_rows, stability_rows, null_rows, sensitivity_rows), encoding="utf-8")

    print(f"wrote_robustness_grid={reg_path}")
    print(f"wrote_feature_stability={feature_path}")
    print(f"wrote_null_control={null_path}")
    print(f"wrote_sensitivity_proxy={sens_path}")
    print(f"wrote_report={md_path}")
    print(f"regularization_grid={reg_grid}")
    print(f"selected_candidates={ {k: v['candidate_name'] for k, v in selected_by_penalty.items()} }")
    print(f"null_selected={null_rows[0]['selected_candidate_name']}")
    print(f"sensitivity_rows={len(sensitivity_rows)}")
    print(f"unique_prediction_candidates={list(unique_candidates)}")
    print(f"zero_reg_selected={selected_zero['candidate_name']}")


if __name__ == "__main__":
    main()
