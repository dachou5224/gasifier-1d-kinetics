"""
Build figure/table-ready data bundles from existing diagnostic artifacts.

This module only reshapes already-generated diagnostics.  It does not rerun the
model, train a hybrid model, or redefine evidence metrics.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

DIAGNOSTIC_SOURCES: Mapping[str, str] = {
    "baseline": "outputs/diagnostics/pilot_fixed_physics_diagnostic_baseline.csv",
    "sensitivity_diagnostic": "outputs/diagnostics/fixed_physics_sensitivity_diagnostic.csv",
    "sensitivity_ranking": "outputs/diagnostics/fixed_physics_sensitivity_ranking.csv",
    "gradient_parity": "outputs/diagnostics/gradient_parity_diagnostic.csv",
    "ift_cost": "outputs/diagnostics/ift_cost_diagnostic.csv",
    "gradient_smoke": "outputs/diagnostics/end_to_end_gradient_smoke.csv",
}

PROTOCOL_SOURCES: Mapping[str, str] = {
    "ift_cost_protocol": "outputs/diagnostics/ift_cost_protocol.md",
}

FIGURE_DATA_FILES: Mapping[str, str] = {
    "fig2_baseline_mismatch": "fig2_baseline_mismatch.csv",
    "fig3_gradient_parity_matrix": "fig3_gradient_parity_matrix.csv",
    "fig3_ift_cost_scaling": "fig3_ift_cost_scaling.csv",
    "fig3_gradient_smoke_trajectory": "fig3_gradient_smoke_trajectory.csv",
    "table_sensitivity_ranking": "table_sensitivity_ranking.csv",
    "manifest": "figure_data_manifest.json",
    "summary": "figure_data_summary.md",
}


def _require_columns(df: pd.DataFrame, required: list[str], source: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{source} missing required columns: {missing}")


def _read_sources(repo_root: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for name, rel_path in DIAGNOSTIC_SOURCES.items():
        path = repo_root / rel_path
        if not path.exists():
            raise FileNotFoundError(f"Missing diagnostic source artifact: {path}")
        frames[name] = pd.read_csv(path)
    return frames


def _build_baseline_mismatch(baseline: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        baseline,
        [
            "case_id",
            "source_candidate_id",
            "independence_group",
            "N_cells",
            "fallback_count",
            "poor_convergence_count",
            "poor_convergence_cell_index",
            "cell0_cost",
            "cell0_max_abs_residual",
            "T_out_degC",
            "pred_CO_vol_pct",
            "target_CO_vol_pct",
            "delta_CO_vol_pct",
            "pred_H2_vol_pct",
            "target_H2_vol_pct",
            "delta_H2_vol_pct",
            "pred_CO2_vol_pct",
            "target_CO2_vol_pct",
            "delta_CO2_vol_pct",
            "pred_N2_vol_pct",
            "target_N2_vol_pct",
            "delta_N2_vol_pct",
            "pred_dry_flow_kmol_s",
            "target_dry_flow_kmol_s",
            "delta_dry_flow_kmol_s",
            "carrier_mapping_ok",
            "transported_moisture_percent",
        ],
        "pilot_fixed_physics_diagnostic_baseline.csv",
    )
    target_specs = [
        ("CO_vol_pct", "pred_CO_vol_pct", "target_CO_vol_pct", "delta_CO_vol_pct"),
        ("H2_vol_pct", "pred_H2_vol_pct", "target_H2_vol_pct", "delta_H2_vol_pct"),
        ("CO2_vol_pct", "pred_CO2_vol_pct", "target_CO2_vol_pct", "delta_CO2_vol_pct"),
        ("N2_vol_pct", "pred_N2_vol_pct", "target_N2_vol_pct", "delta_N2_vol_pct"),
        ("dry_flow_kmol_s", "pred_dry_flow_kmol_s", "target_dry_flow_kmol_s", "delta_dry_flow_kmol_s"),
    ]
    rows: list[dict[str, Any]] = []
    for _, row in baseline.iterrows():
        common = {
            "case_id": row["case_id"],
            "source_candidate_id": row["source_candidate_id"],
            "independence_group": row["independence_group"],
            "N_cells": int(row["N_cells"]),
            "fallback_count": int(row["fallback_count"]),
            "poor_convergence_count": int(row["poor_convergence_count"]),
            "poor_convergence_cell_index": row["poor_convergence_cell_index"],
            "cell0_cost": row["cell0_cost"],
            "cell0_max_abs_residual": row["cell0_max_abs_residual"],
            "T_out_degC": row["T_out_degC"],
            "carrier_mapping_ok": row["carrier_mapping_ok"],
            "transported_moisture_percent": row["transported_moisture_percent"],
        }
        for target_name, pred_col, target_col, delta_col in target_specs:
            rows.append(
                {
                    **common,
                    "target_name": target_name,
                    "predicted_value": row[pred_col],
                    "target_value": row[target_col],
                    "delta_pred_minus_target": row[delta_col],
                }
            )
    return pd.DataFrame(rows)


def _build_gradient_parity_matrix(parity: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        parity,
        [
            "parameter",
            "parameter_key",
            "parameter_base_value",
            "parameter_step",
            "output",
            "output_base_value",
            "fd_grad",
            "ift_or_ad_grad",
            "abs_error",
            "rel_error",
            "status",
        ],
        "gradient_parity_diagnostic.csv",
    )
    out = parity.copy()
    out = out.rename(columns={"ift_or_ad_grad": "ift_grad"})
    return out[
        [
            "parameter",
            "parameter_key",
            "parameter_base_value",
            "parameter_step",
            "output",
            "output_base_value",
            "fd_grad",
            "ift_grad",
            "abs_error",
            "rel_error",
            "status",
            "notes",
        ]
    ]


def _build_ift_cost_scaling(cost: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        cost,
        [
            "method",
            "N_cells",
            "parameter",
            "output",
            "status",
            "failure_reason",
            "compile_or_cold_time_s",
            "warm_eval_time_s",
            "gradient_value",
            "reference_method",
            "reference_gradient_value",
            "abs_gradient_diff",
            "rel_gradient_diff",
            "residual_max_abs",
        ],
        "ift_cost_diagnostic.csv",
    )
    return cost.copy()


def _build_gradient_smoke(smoke: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        smoke,
        [
            "step",
            "parameter_name",
            "parameter_value",
            "loss",
            "gradient",
            "step_size",
            "next_parameter_value",
            "next_loss",
            "solve_success",
            "fallback_count",
            "poor_convergence_count",
            "T_out_K",
            "T_target_K",
            "residual_max_abs",
            "direction_check_passed",
        ],
        "end_to_end_gradient_smoke.csv",
    )
    return smoke.copy()


def _build_sensitivity_ranking(ranking: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        ranking,
        [
            "rank",
            "parameter_name",
            "parameter_key",
            "row_count",
            "ok_row_count",
            "mean_abs_normalized_sensitivity",
            "median_abs_normalized_sensitivity",
            "max_abs_normalized_sensitivity",
            "status",
        ],
        "fixed_physics_sensitivity_ranking.csv",
    )
    return ranking.copy()


def _manifest_entry(path: Path, source: str, df: pd.DataFrame, key_columns: list[str], boundary: str) -> dict[str, Any]:
    return {
        "file": str(path),
        "source_artifact": source,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "key_columns": key_columns,
        "evidence_boundary": boundary,
    }


def _summary_markdown(manifest: Mapping[str, Any]) -> str:
    lines = [
        "# Figure/table-ready diagnostics bundle",
        "",
        "This bundle reshapes existing model diagnostics for manuscript plotting and tables.",
        "It does not add model functionality, train a hybrid model, or claim calibration accuracy.",
        "",
        "| artifact | rows | source | purpose |",
        "|---|---:|---|---|",
    ]
    purposes = {
        "fig2_baseline_mismatch": "fixed-physics predicted-vs-target and delta bars",
        "fig3_gradient_parity_matrix": "FD-vs-IFT parity matrix or heatmap",
        "fig3_ift_cost_scaling": "IFT vs direct/unrolled AD cost-scaling panel",
        "fig3_gradient_smoke_trajectory": "end-to-end gradient descent smoke trajectory",
        "table_sensitivity_ranking": "fixed-physics sensitivity ranking table",
    }
    for name, entry in manifest["artifacts"].items():
        lines.append(
            f"| {name} | {entry['row_count']} | {entry['source_artifact']} | {purposes.get(name, 'metadata')} |"
        )
    lines.extend(
        [
            "",
            "Can support:",
            "- pilot fixed-physics mismatch characterization",
            "- sensitivity-prioritized correction design",
            "- FD-vs-IFT/JAX gradient parity evidence",
            "- IFT/custom-VJP method-cost rationale",
            "- end-to-end gradient smoke evidence",
            "",
            "Cannot support:",
            "- hybrid calibration accuracy",
            "- trained MLP or learned multiplier performance",
            "- industrial generalization",
            "- final Nature-style figure aesthetics without separate plotting/QA.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_figure_data_bundle(
    *,
    repo_root: str | Path | None = None,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Build figure/table data CSVs, a JSON manifest, and a Markdown summary.
    """
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    output_dir = Path(out_dir) if out_dir is not None else root / "outputs" / "figures_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = _read_sources(root)

    artifacts = {
        "fig2_baseline_mismatch": (
            _build_baseline_mismatch(frames["baseline"]),
            DIAGNOSTIC_SOURCES["baseline"],
            ["case_id", "N_cells", "target_name", "predicted_value", "target_value", "delta_pred_minus_target"],
            "Pilot fixed-physics mismatch evidence; not calibrated accuracy.",
        ),
        "fig3_gradient_parity_matrix": (
            _build_gradient_parity_matrix(frames["gradient_parity"]),
            DIAGNOSTIC_SOURCES["gradient_parity"],
            ["parameter", "output", "fd_grad", "ift_grad", "rel_error", "status"],
            "Fixed-branch FD-vs-IFT/JAX parity evidence; not full-reactor training evidence.",
        ),
        "fig3_ift_cost_scaling": (
            _build_ift_cost_scaling(frames["ift_cost"]),
            DIAGNOSTIC_SOURCES["ift_cost"],
            ["method", "N_cells", "status", "compile_or_cold_time_s", "warm_eval_time_s", "gradient_value"],
            "Fixed-branch method-cost evidence; not full-reactor performance evidence.",
        ),
        "fig3_gradient_smoke_trajectory": (
            _build_gradient_smoke(frames["gradient_smoke"]),
            DIAGNOSTIC_SOURCES["gradient_smoke"],
            ["step", "parameter_value", "loss", "gradient", "next_loss", "direction_check_passed"],
            "End-to-end gradient smoke evidence; not calibration accuracy.",
        ),
        "table_sensitivity_ranking": (
            _build_sensitivity_ranking(frames["sensitivity_ranking"]),
            DIAGNOSTIC_SOURCES["sensitivity_ranking"],
            ["rank", "parameter_name", "mean_abs_normalized_sensitivity", "status"],
            "Fixed-physics sensitivity ranking evidence; not learned correction performance.",
        ),
    }

    manifest: dict[str, Any] = {
        "bundle_version": 1,
        "output_directory": str(output_dir),
        "artifacts": {},
        "source_artifacts": dict(DIAGNOSTIC_SOURCES),
        "protocol_artifacts": dict(PROTOCOL_SOURCES),
        "global_evidence_boundary": (
            "Figure data bundle for pilot diagnostics only. It does not claim hybrid calibration "
            "accuracy, trained model performance, or industrial generalization."
        ),
    }
    for name, (df, source, key_columns, boundary) in artifacts.items():
        csv_path = output_dir / FIGURE_DATA_FILES[name]
        df.to_csv(csv_path, index=False)
        manifest["artifacts"][name] = _manifest_entry(csv_path, source, df, key_columns, boundary)
    for protocol_name, rel_path in PROTOCOL_SOURCES.items():
        source_path = root / rel_path
        if source_path.exists():
            manifest["artifacts"][protocol_name] = {
                "file": str(source_path),
                "source_artifact": rel_path,
                "row_count": len(source_path.read_text(encoding="utf-8").splitlines()),
                "column_count": 0,
                "key_columns": ["protocol", "timing definitions", "timeout rule", "method definitions"],
                "evidence_boundary": "Reviewer-facing protocol metadata; not a numeric benchmark table.",
            }

    manifest_path = output_dir / FIGURE_DATA_FILES["manifest"]
    summary_path = output_dir / FIGURE_DATA_FILES["summary"]
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_path.write_text(_summary_markdown(manifest), encoding="utf-8")
    manifest["artifacts"]["manifest"] = {
        "file": str(manifest_path),
        "source_artifact": "generated manifest",
        "row_count": 1,
        "column_count": 0,
        "key_columns": ["artifacts", "source_artifacts", "global_evidence_boundary"],
        "evidence_boundary": manifest["global_evidence_boundary"],
    }
    manifest["artifacts"]["summary"] = {
        "file": str(summary_path),
        "source_artifact": "generated summary",
        "row_count": len(summary_path.read_text(encoding="utf-8").splitlines()),
        "column_count": 0,
        "key_columns": ["artifact", "rows", "source", "purpose"],
        "evidence_boundary": manifest["global_evidence_boundary"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


__all__ = [
    "DIAGNOSTIC_SOURCES",
    "FIGURE_DATA_FILES",
    "PROTOCOL_SOURCES",
    "build_figure_data_bundle",
]
