#!/usr/bin/env python3
"""Generate Huayi engineering-coordinate influence diagnostics."""
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
    from model.engineering_coordinate_influence import (
        ENGINEERING_AXIS_COLUMNS,
        ENGINEERING_COORDINATE_COLUMNS,
        ENGINEERING_COORDINATE_PARAMETERS,
        build_engineering_axis_summary,
        build_engineering_coordinate_rows,
        engineering_coordinate_markdown,
    )
    from model.model_case_sensitivity import (
        SENSITIVITY_COLUMNS,
        build_sensitivity_rows,
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
        parameters=ENGINEERING_COORDINATE_PARAMETERS,
    )
    axis_rows = build_engineering_axis_summary(sensitivity_rows)
    baseline_rows = pd.read_csv(args.baseline).to_dict("records")
    contract_rows = pd.read_csv(args.contract_audit).to_dict("records")
    coordinate_rows = build_engineering_coordinate_rows(
        sensitivity_rows,
        axis_rows,
        baseline_rows,
        contract_rows,
    )

    sensitivity_path = out_dir / "huayi_engineering_coordinate_sensitivity_detail.csv"
    diagnostic_path = out_dir / "huayi_engineering_coordinate_influence_diagnostic.csv"
    summary_path = out_dir / "huayi_engineering_coordinate_influence_summary.md"
    axes_path = fig_dir / "huayi_engineering_coordinate_axes.csv"

    pd.DataFrame(sensitivity_rows, columns=list(SENSITIVITY_COLUMNS)).to_csv(sensitivity_path, index=False)
    pd.DataFrame(coordinate_rows, columns=list(ENGINEERING_COORDINATE_COLUMNS)).to_csv(diagnostic_path, index=False)
    pd.DataFrame(axis_rows, columns=list(ENGINEERING_AXIS_COLUMNS)).to_csv(axes_path, index=False)
    summary_path.write_text(
        engineering_coordinate_markdown(coordinate_rows, axis_rows)
        + "\n## Coordinate finite-difference subtable\n\n"
        + (
            "The accompanying sensitivity-detail CSV stores the centered finite-difference probes "
            "used to build the engineering-coordinate influence table. The coordinates are "
            "reporting axes, not a claim that all underlying raw inputs are independent plant "
            "manipulated variables.\n"
        ),
        encoding="utf-8",
    )

    print(f"wrote_sensitivity={sensitivity_path}")
    print(f"wrote_diagnostic={diagnostic_path}")
    print(f"wrote_summary={summary_path}")
    print(f"wrote_axes={axes_path}")
    print(f"model_cases={len(model_case_rows)}")
    print(f"sensitivity_rows={len(sensitivity_rows)}")
    print(f"coordinate_rows={len(coordinate_rows)}")
    print(
        "coordinate_ranking="
        + ",".join(
            f"{row['aggregate_coordinate_rank']}:{row['coordinate']}:{row['mean_abs_normalized_sensitivity']:.6g}"
            for row in axis_rows
        )
    )


if __name__ == "__main__":
    main()
