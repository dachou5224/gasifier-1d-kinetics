#!/usr/bin/env python3
"""Generate Huayi O/C-centered fixed-physics response curve diagnostics."""
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


def _load_model_cases(cases_per_phase: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for phase, dataset_id, path in PREVIEW_SPECS:
        df = pd.read_parquet(path)
        if cases_per_phase > 0:
            df = df.head(cases_per_phase)
        for _, row in df.iterrows():
            item = row.to_dict()
            item["phase"] = phase
            item["dataset_id"] = dataset_id
            rows.append(item)
    return rows


def _parse_multipliers(text: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in text.split(",") if item.strip())


def main() -> None:
    repo_root = _ensure_src_path()
    from model.oc_response_curve_diagnostic import (
        OC_RESPONSE_AXIS_COLUMNS,
        OC_RESPONSE_COLUMNS,
        build_oc_response_axis_rows,
        build_oc_response_rows,
        oc_response_markdown,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cells", type=int, default=5)
    parser.add_argument("--multipliers", default="0.96,0.98,1.00,1.02,1.04")
    parser.add_argument("--cases-per-phase", type=int, default=0)
    parser.add_argument("--out-dir", default=str(repo_root / "outputs" / "diagnostics"))
    parser.add_argument("--figure-out-dir", default=str(repo_root / "outputs" / "figures_data"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = Path(args.figure_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    model_case_rows = _load_model_cases(cases_per_phase=args.cases_per_phase)
    multipliers = _parse_multipliers(args.multipliers)
    response_rows = build_oc_response_rows(model_case_rows, n_cells=args.n_cells, multipliers=multipliers)
    axis_rows = build_oc_response_axis_rows(response_rows)

    diagnostic_path = out_dir / "huayi_oc_response_curve_diagnostic.csv"
    summary_path = out_dir / "huayi_oc_response_curve_summary.md"
    axes_path = fig_dir / "huayi_oc_response_curve_axes.csv"
    phase_summary_path = fig_dir / "huayi_oc_response_curve_phase_summary.csv"

    pd.DataFrame(response_rows, columns=list(OC_RESPONSE_COLUMNS)).to_csv(diagnostic_path, index=False)
    axis_df = pd.DataFrame(axis_rows, columns=list(OC_RESPONSE_AXIS_COLUMNS))
    axis_df.to_csv(axes_path, index=False)
    phase_summary = (
        axis_df.groupby(["phase", "target_name"], dropna=False)
        .agg(
            case_count=("case_id", "nunique"),
            axis_row_count=("case_id", "size"),
            mean_local_slope_per_relative_multiplier=("local_slope_per_relative_multiplier", "mean"),
            mean_abs_local_slope_per_relative_multiplier=("local_slope_per_relative_multiplier", lambda values: values.abs().mean()),
            positive_slope_count=("local_slope_sign", lambda values: int((values == "positive").sum())),
            negative_slope_count=("local_slope_sign", lambda values: int((values == "negative").sum())),
            nonmonotonic_row_count=("curve_direction_status", lambda values: int((values == "nonmonotonic").sum())),
        )
        .reset_index()
    )
    phase_summary.to_csv(phase_summary_path, index=False)
    summary_path.write_text(oc_response_markdown(response_rows, axis_rows), encoding="utf-8")

    print(f"wrote_diagnostic={diagnostic_path}")
    print(f"wrote_summary={summary_path}")
    print(f"wrote_axes={axes_path}")
    print(f"wrote_phase_summary={phase_summary_path}")
    print(f"model_cases={len(model_case_rows)}")
    print(f"response_rows={len(response_rows)}")
    print(f"axis_rows={len(axis_rows)}")
    print(f"ok_response_rows={sum(1 for row in response_rows if row['status'] == 'ok')}")
    print("multipliers=" + ",".join(f"{value:.2f}" for value in multipliers))


if __name__ == "__main__":
    main()
