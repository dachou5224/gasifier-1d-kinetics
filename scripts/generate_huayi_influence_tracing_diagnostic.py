#!/usr/bin/env python3
"""Generate Huayi 22-case influence-tracing diagnostics."""
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
    from model.influence_tracing_diagnostic import (
        AXIS_SUMMARY_COLUMNS,
        INFLUENCE_TRACING_COLUMNS,
        build_axis_summary_rows,
        build_influence_tracing_rows,
        influence_tracing_markdown,
    )
    from model.model_case_sensitivity import (
        RANKING_COLUMNS,
        SENSITIVITY_COLUMNS,
        build_parameter_ranking,
        build_sensitivity_rows,
        sensitivity_markdown,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cells", type=int, default=5)
    parser.add_argument("--perturbation-fraction", type=float, default=0.02)
    parser.add_argument("--out-dir", default=str(repo_root / "outputs" / "diagnostics"))
    parser.add_argument("--figure-out-dir", default=str(repo_root / "outputs" / "figures_data"))
    parser.add_argument(
        "--baseline",
        default=str(repo_root / "outputs" / "diagnostics" / "huayi_expanded_fixed_physics_diagnostic_baseline.csv"),
    )
    parser.add_argument(
        "--contract-audit",
        default=str(repo_root / "outputs" / "diagnostics" / "huayi_expanded_model_case_contract_audit.csv"),
    )
    parser.add_argument(
        "--gradient-parity",
        default=str(repo_root / "outputs" / "diagnostics" / "gradient_parity_diagnostic.csv"),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = Path(args.figure_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    model_case_rows = _load_model_cases()
    sensitivity_rows = build_sensitivity_rows(
        model_case_rows,
        n_cells=args.n_cells,
        perturbation_fraction=args.perturbation_fraction,
    )
    ranking_rows = build_parameter_ranking(sensitivity_rows)
    baseline_rows = pd.read_csv(args.baseline).to_dict("records")
    contract_rows = pd.read_csv(args.contract_audit).to_dict("records")
    parity_rows = pd.read_csv(args.gradient_parity).to_dict("records")

    influence_rows = build_influence_tracing_rows(
        sensitivity_rows,
        ranking_rows,
        parity_rows,
        baseline_rows,
        contract_rows,
    )
    axis_rows = build_axis_summary_rows(influence_rows, parity_rows)

    sensitivity_path = out_dir / "huayi_influence_tracing_sensitivity_detail.csv"
    ranking_path = out_dir / "huayi_influence_tracing_sensitivity_ranking.csv"
    diagnostic_path = out_dir / "huayi_influence_tracing_diagnostic.csv"
    summary_path = out_dir / "huayi_influence_tracing_summary.md"
    axes_path = fig_dir / "huayi_influence_tracing_axes.csv"

    pd.DataFrame(sensitivity_rows, columns=list(SENSITIVITY_COLUMNS)).to_csv(sensitivity_path, index=False)
    pd.DataFrame(ranking_rows, columns=list(RANKING_COLUMNS)).to_csv(ranking_path, index=False)
    pd.DataFrame(influence_rows, columns=list(INFLUENCE_TRACING_COLUMNS)).to_csv(diagnostic_path, index=False)
    pd.DataFrame(axis_rows, columns=list(AXIS_SUMMARY_COLUMNS)).to_csv(axes_path, index=False)
    summary_path.write_text(
        influence_tracing_markdown(influence_rows, axis_rows)
        + "\n## Sensitivity subtable\n\n"
        + sensitivity_markdown(sensitivity_rows, ranking_rows),
        encoding="utf-8",
    )

    print(f"wrote_sensitivity={sensitivity_path}")
    print(f"wrote_ranking={ranking_path}")
    print(f"wrote_diagnostic={diagnostic_path}")
    print(f"wrote_summary={summary_path}")
    print(f"wrote_axes={axes_path}")
    print(f"model_cases={len(model_case_rows)}")
    print(f"sensitivity_rows={len(sensitivity_rows)}")
    print(f"influence_rows={len(influence_rows)}")
    print("axis_ranking=" + ",".join(f"{row['aggregate_axis_rank']}:{row['axis']}:{row['mean_abs_normalized_sensitivity']:.6g}" for row in axis_rows))


if __name__ == "__main__":
    main()
