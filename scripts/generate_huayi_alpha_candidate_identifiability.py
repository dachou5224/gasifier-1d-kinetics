#!/usr/bin/env python3
"""Generate Huayi candidate-alpha identifiability screen artifacts."""
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
    from model.alpha_candidate_identifiability import (
        RECOMMENDATION_COLUMNS,
        SENSITIVITY_COLUMNS,
        SIMILARITY_COLUMNS,
        alpha_candidate_recommendation_rows,
        alpha_candidate_sensitivity_rows,
        alpha_candidate_similarity_rows,
        identifiability_markdown,
    )
    from model.split_global_alpha_baseline import attach_split_to_model_cases

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cells", type=int, default=3)
    parser.add_argument("--out-dir", default=str(repo_root / "outputs" / "diagnostics"))
    parser.add_argument("--split-manifest", default=str(repo_root / "outputs" / "diagnostics" / "huayi_hybrid_split_manifest.csv"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_rows = pd.read_csv(args.split_manifest).to_dict("records")
    model_case_rows = attach_split_to_model_cases(_load_model_cases(), split_rows)

    sensitivity = alpha_candidate_sensitivity_rows(model_case_rows, n_cells=args.n_cells)
    similarity = alpha_candidate_similarity_rows(sensitivity)
    recommendations = alpha_candidate_recommendation_rows(sensitivity, similarity)

    sens_path = out_dir / "huayi_alpha_candidate_sensitivities.csv"
    sim_path = out_dir / "huayi_alpha_candidate_similarity.csv"
    rec_path = out_dir / "huayi_alpha_candidate_recommendations.csv"
    md_path = out_dir / "huayi_alpha_candidate_identifiability_screen.md"
    pd.DataFrame(sensitivity, columns=list(SENSITIVITY_COLUMNS)).to_csv(sens_path, index=False)
    pd.DataFrame(similarity, columns=list(SIMILARITY_COLUMNS)).to_csv(sim_path, index=False)
    pd.DataFrame(recommendations, columns=list(RECOMMENDATION_COLUMNS)).to_csv(rec_path, index=False)
    md_path.write_text(identifiability_markdown(sensitivity, similarity, recommendations), encoding="utf-8")

    print(f"wrote_sensitivities={sens_path}")
    print(f"wrote_similarity={sim_path}")
    print(f"wrote_recommendations={rec_path}")
    print(f"wrote_report={md_path}")
    print(f"sensitivity_rows={len(sensitivity)}")
    print(f"similarity_rows={len(similarity)}")
    print(f"recommendations={ {row['alpha_name']: row['recommendation'] for row in recommendations} }")


if __name__ == "__main__":
    main()
