#!/usr/bin/env python3
"""Generate Huayi expanded ModelCase audit and fixed-physics diagnostics."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PREVIEW_SPECS = (
    (
        "phase1",
        "huayi_sy3_phase1_wgs1_shift",
        "v0_baseline_coal_assumption",
        "/Users/liuzhen/AI-projects/gasifier-data-prep/outputs/model_cases/huayi_sy3_phase1_wgs1_shift.model_cases_preview.parquet",
    ),
    (
        "phase2",
        "huayi_sy3sy2_phase2_wgs1_shift",
        "huayi_p2_sy3sy2_coal_assumption",
        "/Users/liuzhen/AI-projects/gasifier-data-prep/outputs/model_cases/huayi_sy3sy2_phase2_wgs1_shift.model_cases_preview.parquet",
    ),
    (
        "phase3",
        "huayi_sy3sy2_phase3_wgs1_shift",
        "huayi_p3_sy3sy2_coal_assumption",
        "/Users/liuzhen/AI-projects/gasifier-data-prep/outputs/model_cases/huayi_sy3sy2_phase3_wgs1_shift.model_cases_preview.parquet",
    ),
    (
        "phase4",
        "huayi_sy3sy2_phase4_wgs1_shift",
        "huayi_p4_sy3sy2_coal_assumption",
        "/Users/liuzhen/AI-projects/gasifier-data-prep/outputs/model_cases/huayi_sy3sy2_phase4_wgs1_shift.model_cases_preview.parquet",
    ),
)

MONITOR_LINE_COLUMNS = tuple(f"monitor_pulverized_coal_line_{idx}_kg_s_value" for idx in range(1, 5))


def _ensure_src_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return repo_root


def _load_model_cases(preview_specs: tuple[tuple[str, str, str, str], ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for phase, dataset_id, expected_feature_version, preview_path in preview_specs:
        df = pd.read_parquet(preview_path)
        for _, row in df.iterrows():
            item = row.to_dict()
            item["phase"] = phase
            item["dataset_id"] = dataset_id
            item["expected_feature_version_from_diagnostics"] = expected_feature_version
            item["source_preview_file"] = Path(preview_path).name
            rows.append(item)
    return rows


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")

    def int_value(key: str) -> int | None:
        match = re.search(rf"^\s*{re.escape(key)}:\s*(\d+)\s*$", text, flags=re.MULTILINE)
        return int(match.group(1)) if match else None

    return {
        "totals": {
            "candidate_segments": int_value("candidate_segments"),
            "independence_groups": int_value("independence_groups"),
            "ready_representative_model_cases": int_value("ready_representative_model_cases"),
        }
    }


def _load_contract_audit(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _monitoring_sum(row: dict[str, Any]) -> float:
    values = [row.get(column) for column in MONITOR_LINE_COLUMNS]
    if any(value is None or (isinstance(value, float) and math.isnan(value)) for value in values):
        return math.nan
    return float(sum(float(value) for value in values))


def _contract_audit_rows(model_case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from model.model_case_adapter import build_gasifier_inputs_from_model_case

    out: list[dict[str, Any]] = []
    for row in model_case_rows:
        built = build_gasifier_inputs_from_model_case(row)
        monitor_sum = _monitoring_sum(row)
        feature_version = row.get("feature_version_value", row.get("feature_version"))
        expected = row["expected_feature_version_from_diagnostics"]
        feature_version_matches_diagnostics = feature_version == expected
        steam_status = row.get("steam_mass_flow_source_status")
        steam_value = row.get("steam_mass_flow_value")
        steam_not_applicable = steam_status == "not_applicable" and (
            steam_value is None or (isinstance(steam_value, float) and math.isnan(steam_value))
        )
        out.append(
            {
                "phase": row["phase"],
                "dataset_id": row["dataset_id"],
                "case_id": built["case_id"],
                "source_candidate_id": built["segment_id"],
                "independence_group": row.get("independence_group_id_value", row.get("independence_group_id")),
                "model_case_ready": bool(row.get("model_case_ready")),
                "feature_version_value": feature_version,
                "expected_feature_version_from_diagnostics": expected,
                "feature_version_matches_diagnostics": feature_version_matches_diagnostics,
                "feature_version_drift_affects_adapter_inputs": False,
                "coal_mass_flow_kg_s": float(row["coal_mass_flow_value"]),
                "coal_flow_source_ref": row.get("coal_mass_flow_source_ref"),
                "coal_weighfeeder_ar_flow_t_h": float(row["coal_weighfeeder_ar_flow_value"]),
                "coal_moisture_ar_fraction": float(row["coal_moisture_ar_value"]),
                "monitor_pulverized_coal_c_f_sum_kg_s": monitor_sum,
                "monitor_to_primary_coal_flow_ratio": monitor_sum / float(row["coal_mass_flow_value"]),
                "adapter_coal_flow_kg_s": float(built["op_conds"]["coal_flow"]),
                "adapter_uses_primary_coal_flow": math.isclose(
                    float(built["op_conds"]["coal_flow"]),
                    float(row["coal_mass_flow_value"]),
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                ),
                "adapter_avoids_c_f_monitoring_sum": not math.isclose(
                    float(built["op_conds"]["coal_flow"]),
                    monitor_sum,
                    rel_tol=1.0e-9,
                    abs_tol=1.0e-9,
                ),
                "steam_mass_flow_source_status": steam_status,
                "steam_mass_flow_not_applicable": steam_not_applicable,
                "adapter_steam_flow_kg_s": float(built["op_conds"]["steam_flow"]),
                "adapter_uses_zero_steam_feed": math.isclose(
                    float(built["op_conds"]["steam_flow"]),
                    0.0,
                    rel_tol=0.0,
                    abs_tol=1.0e-15,
                ),
                "coal_C_pct_adapter": float(built["coal_props"]["Cd"]),
                "HHV_d_adapter_kj_kg": float(built["coal_props"]["HHV_d"]),
                "carrier_gas_type": built["op_conds"]["carrier_gas_type"],
                "carrier_mapping_basis": built["assumptions"]["coal_flow_basis"],
                "source_preview_file": row["source_preview_file"],
            }
        )
    return out


def _attach_provenance(rows: list[dict[str, Any]], by_case: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        meta = by_case.get(str(row.get("case_id")), {})
        for key in ("phase", "dataset_id", "source_preview_file", "feature_version_value", "expected_feature_version_from_diagnostics"):
            copied[key] = meta.get(key)
        out.append(copied)
    return out


def _markdown_summary(
    audit_rows: list[dict[str, Any]],
    carbon_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    n_cells_values: list[int],
    manifest: dict[str, Any],
    contract_audit: dict[str, Any],
) -> str:
    baseline_df = pd.DataFrame(baseline_rows)
    carbon_df = pd.DataFrame(carbon_rows)
    audit_df = pd.DataFrame(audit_rows)
    drift_count = int((~audit_df["feature_version_matches_diagnostics"]).sum())
    manifest_totals = manifest.get("totals", {})
    contract_passed = contract_audit.get("passed")
    lines = [
        "# Huayi expanded corrected ModelCase diagnostic audit",
        "",
        "This artifact audits the Huayi four-phase corrected-contract preview as a paper-facing fixed-physics diagnostic dataset candidate.",
        "",
        f"- Manifest candidate segments: {manifest_totals.get('candidate_segments', 'n/a')}",
        f"- Manifest independence groups: {manifest_totals.get('independence_groups', 'n/a')}",
        f"- Manifest READY representative ModelCases: {manifest_totals.get('ready_representative_model_cases', 'n/a')}",
        f"- Preview contract audit passed: {contract_passed}",
        f"- Ready rows: {int(audit_df['model_case_ready'].sum())}/{len(audit_df)}",
        f"- Phases: {', '.join(f'{k}={v}' for k, v in audit_df['phase'].value_counts().sort_index().items())}",
        f"- N_cells baseline coverage: {', '.join(str(v) for v in n_cells_values)}",
        f"- Adapter uses primary corrected coal flow: {int(audit_df['adapter_uses_primary_coal_flow'].sum())}/{len(audit_df)}",
        f"- Adapter avoids C-F monitoring sum: {int(audit_df['adapter_avoids_c_f_monitoring_sum'].sum())}/{len(audit_df)}",
        f"- Steam not-applicable and zero adapter feed: {int((audit_df['steam_mass_flow_not_applicable'] & audit_df['adapter_uses_zero_steam_feed']).sum())}/{len(audit_df)}",
        f"- Feature-version provenance drift rows: {drift_count}/{len(audit_df)}",
        "",
        "## Provenance note",
        "",
        "The four-phase previews now expose phase-specific `feature_version` values. The model adapter consumes the numeric coal-property columns directly; feature-version matching is recorded here as provenance protection.",
        "",
        "## Phase4 caveat",
        "",
        "Phase4 is included as expanded dry-powder diagnostic coverage. It is not sufficient by itself for generalization claims or unweighted train/validation interpretation.",
        "",
        "## Numeric summary",
        "",
        f"- fallback_count distribution: {baseline_df['fallback_count'].value_counts().sort_index().to_dict()}",
        f"- poor_convergence_count distribution: {baseline_df['poor_convergence_count'].value_counts().sort_index().to_dict()}",
        f"- cell0_cost range: {baseline_df['cell0_cost'].min():.6g} to {baseline_df['cell0_cost'].max():.6g}",
        f"- T_out_degC range: {baseline_df['T_out_degC'].min():.3f} to {baseline_df['T_out_degC'].max():.3f}",
        f"- delta_CO_vol_pct range: {baseline_df['delta_CO_vol_pct'].min():.3f} to {baseline_df['delta_CO_vol_pct'].max():.3f}",
        f"- delta_H2_vol_pct range: {baseline_df['delta_H2_vol_pct'].min():.3f} to {baseline_df['delta_H2_vol_pct'].max():.3f}",
        f"- delta_CO2_vol_pct range: {baseline_df['delta_CO2_vol_pct'].min():.3f} to {baseline_df['delta_CO2_vol_pct'].max():.3f}",
        f"- delta_dry_flow_kmol_s range: {baseline_df['delta_dry_flow_kmol_s'].min():.6g} to {baseline_df['delta_dry_flow_kmol_s'].max():.6g}",
        f"- apparent gross gas-phase C recovery range: {carbon_df['apparent_gas_phase_C_recovery'].min():.6f} to {carbon_df['apparent_gas_phase_C_recovery'].max():.6f}",
        f"- C-F-misuse gross recovery range: {carbon_df['gross_recovery_if_c_f_monitoring_sum_used'].min():.6f} to {carbon_df['gross_recovery_if_c_f_monitoring_sum_used'].max():.6f}",
        "",
        "Evidence boundary: multi-case dry-powder fixed-physics diagnostic coverage only. This is not calibrated accuracy, carbon-closure proof, hybrid validation, or Yulin/slurry generalization.",
    ]
    return "\n".join(lines) + "\n"


def _build_huayi_baseline_mismatch(baseline_rows: list[dict[str, Any]]) -> pd.DataFrame:
    target_specs = [
        ("CO_vol_pct", "pred_CO_vol_pct", "target_CO_vol_pct", "delta_CO_vol_pct"),
        ("H2_vol_pct", "pred_H2_vol_pct", "target_H2_vol_pct", "delta_H2_vol_pct"),
        ("CO2_vol_pct", "pred_CO2_vol_pct", "target_CO2_vol_pct", "delta_CO2_vol_pct"),
        ("N2_vol_pct", "pred_N2_vol_pct", "target_N2_vol_pct", "delta_N2_vol_pct"),
        ("dry_flow_kmol_s", "pred_dry_flow_kmol_s", "target_dry_flow_kmol_s", "delta_dry_flow_kmol_s"),
    ]
    rows: list[dict[str, Any]] = []
    for row in baseline_rows:
        common = {
            "phase": row.get("phase"),
            "dataset_id": row.get("dataset_id"),
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


def _repo_relative(path: Path, root: Path) -> str:
    """输出仓库相对路径；仓库外路径保留绝对形式。

    已提交的 manifest 必须可移植，不能写入作者机器的绝对路径。
    """
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _rel_entry(path: Path, root: Path, **extra: Any) -> dict[str, Any]:
    relative = _repo_relative(path, root)
    entry = {
        "file": relative,
        "path_basis": "repo_relative" if not Path(relative).is_absolute() else "absolute_outside_repo",
        "path_basis_root": ".",
    }
    entry.update(extra)
    return entry


def _write_huayi_figure_bundle(out_dir: Path, audit_rows: list[dict[str, Any]], carbon_rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_mismatch = _build_huayi_baseline_mismatch(baseline_rows)
    carbon = pd.DataFrame(carbon_rows)
    audit = pd.DataFrame(audit_rows)

    fig2_path = out_dir / "huayi_expanded_fig2_baseline_mismatch.csv"
    carbon_path = out_dir / "huayi_expanded_table_carbon_sanity.csv"
    contract_path = out_dir / "huayi_expanded_table_contract_audit.csv"
    manifest_path = out_dir / "huayi_expanded_figure_data_manifest.json"
    summary_path = out_dir / "huayi_expanded_figure_data_summary.md"

    baseline_mismatch.to_csv(fig2_path, index=False)
    carbon.to_csv(carbon_path, index=False)
    audit.to_csv(contract_path, index=False)

    manifest = {
        "evidence_boundary": "Huayi four-phase dry-powder fixed-physics diagnostic coverage only; not calibrated accuracy, hybrid validation, carbon closure, or Yulin/slurry generalization.",
        "artifacts": {
            "huayi_expanded_fig2_baseline_mismatch": {
                **_rel_entry(fig2_path, repo_root),
                "row_count": int(len(baseline_mismatch)),
                "source_artifact": "outputs/diagnostics/huayi_expanded_fixed_physics_diagnostic_baseline.csv",
                "key_columns": ["phase", "case_id", "N_cells", "target_name", "predicted_value", "target_value", "delta_pred_minus_target"],
            },
            "huayi_expanded_table_carbon_sanity": {
                **_rel_entry(carbon_path, repo_root),
                "row_count": int(len(carbon)),
                "source_artifact": "outputs/diagnostics/huayi_expanded_carbon_balance_diagnostic.csv",
                "key_columns": ["phase", "case_id", "apparent_gas_phase_C_recovery", "gross_recovery_if_c_f_monitoring_sum_used"],
            },
            "huayi_expanded_table_contract_audit": {
                **_rel_entry(contract_path, repo_root),
                "row_count": int(len(audit)),
                "source_artifact": "outputs/diagnostics/huayi_expanded_model_case_contract_audit.csv",
                "key_columns": ["phase", "case_id", "adapter_uses_primary_coal_flow", "adapter_avoids_c_f_monitoring_sum", "adapter_uses_zero_steam_feed"],
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary_path.write_text(
        "# Huayi expanded figure/table data\n\n"
        f"- baseline mismatch rows: {len(baseline_mismatch)}\n"
        f"- carbon sanity rows: {len(carbon)}\n"
        f"- contract audit rows: {len(audit)}\n\n"
        "Evidence boundary: Huayi four-phase dry-powder fixed-physics diagnostic coverage only; "
        "not calibrated accuracy, hybrid validation, carbon closure, or Yulin/slurry generalization.\n",
        encoding="utf-8",
    )
    manifest["artifacts"]["manifest"] = _rel_entry(manifest_path, repo_root, row_count=1)
    manifest["artifacts"]["summary"] = _rel_entry(summary_path, repo_root, row_count=8)
    return manifest


def main() -> None:
    repo_root = _ensure_src_path()
    from model.model_case_carbon_balance_diagnostic import (
        REQUIRED_CARBON_BALANCE_COLUMNS,
        build_carbon_balance_rows,
        markdown_summary as carbon_markdown_summary,
    )
    from model.model_case_diagnostic_baseline import (
        REQUIRED_DIAGNOSTIC_COLUMNS,
        build_diagnostic_rows,
        markdown_summary as baseline_markdown_summary,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cells", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--out-dir", default=str(repo_root / "outputs" / "diagnostics"))
    parser.add_argument("--figure-out-dir", default=str(repo_root / "outputs" / "figures_data"))
    parser.add_argument("--basename", default="huayi_expanded")
    parser.add_argument(
        "--manifest",
        default="/Users/liuzhen/AI-projects/gasifier-data-prep/outputs/model_cases/huayi_four_phase_model_case_manifest.yaml",
    )
    parser.add_argument(
        "--preview-contract-audit",
        default="/Users/liuzhen/AI-projects/gasifier-data-prep/outputs/model_cases/huayi_four_phase_preview_contract_audit.json",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(Path(args.manifest))
    contract_audit = _load_contract_audit(Path(args.preview_contract_audit))
    model_case_rows = _load_model_cases(PREVIEW_SPECS)
    audit_rows = _contract_audit_rows(model_case_rows)
    by_case = {row["case_id"]: row for row in audit_rows}

    carbon_rows = _attach_provenance(build_carbon_balance_rows(model_case_rows), by_case)
    baseline_rows = _attach_provenance(build_diagnostic_rows(model_case_rows, args.n_cells), by_case)

    audit_path = out_dir / f"{args.basename}_model_case_contract_audit.csv"
    audit_md_path = out_dir / f"{args.basename}_model_case_contract_audit.md"
    carbon_path = out_dir / f"{args.basename}_carbon_balance_diagnostic.csv"
    carbon_md_path = out_dir / f"{args.basename}_carbon_balance_diagnostic.md"
    baseline_path = out_dir / f"{args.basename}_fixed_physics_diagnostic_baseline.csv"
    baseline_md_path = out_dir / f"{args.basename}_fixed_physics_diagnostic_baseline.md"

    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
    carbon_columns = [
        "phase",
        "dataset_id",
        "source_preview_file",
        "feature_version_value",
        "expected_feature_version_from_diagnostics",
        *REQUIRED_CARBON_BALANCE_COLUMNS,
    ]
    pd.DataFrame(carbon_rows).to_csv(carbon_path, index=False, columns=carbon_columns)
    baseline_columns = [
        "phase",
        "dataset_id",
        "source_preview_file",
        "feature_version_value",
        "expected_feature_version_from_diagnostics",
        *REQUIRED_DIAGNOSTIC_COLUMNS,
    ]
    pd.DataFrame(baseline_rows).to_csv(baseline_path, index=False, columns=baseline_columns)

    audit_md_path.write_text(
        _markdown_summary(audit_rows, carbon_rows, baseline_rows, args.n_cells, manifest, contract_audit),
        encoding="utf-8",
    )
    carbon_md_path.write_text(carbon_markdown_summary(carbon_rows), encoding="utf-8")
    baseline_md_path.write_text(baseline_markdown_summary(baseline_rows), encoding="utf-8")
    figure_manifest = _write_huayi_figure_bundle(Path(args.figure_out_dir), audit_rows, carbon_rows, baseline_rows)

    baseline_df = pd.DataFrame(baseline_rows)
    carbon_df = pd.DataFrame(carbon_rows)
    audit_df = pd.DataFrame(audit_rows)
    print(f"wrote_contract_audit={audit_path}")
    print(f"wrote_contract_audit_md={audit_md_path}")
    print(f"wrote_carbon={carbon_path}")
    print(f"wrote_carbon_md={carbon_md_path}")
    print(f"wrote_baseline={baseline_path}")
    print(f"wrote_baseline_md={baseline_md_path}")
    print(f"wrote_huayi_figure_manifest={figure_manifest['artifacts']['manifest']['file']}")
    print(f"ready_rows={sum(row['model_case_ready'] for row in audit_rows)}/{len(audit_rows)}")
    print(f"feature_version_match_distribution={audit_df['feature_version_matches_diagnostics'].value_counts().sort_index().to_dict()}")
    print(f"steam_zero_feed_distribution={audit_df['adapter_uses_zero_steam_feed'].value_counts().sort_index().to_dict()}")
    print(f"baseline_rows={len(baseline_rows)}")
    print(f"fallback_distribution={baseline_df['fallback_count'].value_counts().sort_index().to_dict()}")
    print(f"poor_convergence_distribution={baseline_df['poor_convergence_count'].value_counts().sort_index().to_dict()}")
    print(f"cell0_cost_min={baseline_df['cell0_cost'].min():.12g}")
    print(f"cell0_cost_max={baseline_df['cell0_cost'].max():.12g}")
    print(f"carbon_recovery_min={carbon_df['apparent_gas_phase_C_recovery'].min():.12g}")
    print(f"carbon_recovery_max={carbon_df['apparent_gas_phase_C_recovery'].max():.12g}")


if __name__ == "__main__":
    main()
