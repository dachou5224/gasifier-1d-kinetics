"""
Hybrid calibration protocol artifact builder.

This module freezes the first training protocol for a bounded reaction-family
hybrid without running training.  It is intentionally declarative: the produced
artifacts describe candidate alpha groups, split policy, baselines, metrics,
runtime risks, and claim boundaries.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

PROTOCOL_VERSION = "round40_v1"

CLAIM_BOUNDARY = (
    "protocol freeze only; not a trained hybrid model, calibrated accuracy, "
    "production control, learned multiplier interpretability, weak-label "
    "coal-quality proof, temporal generalization result, or Yulin/slurry "
    "generalization"
)

REQUIRED_PROTOCOL_KEYS: tuple[str, ...] = (
    "protocol_version",
    "claim_boundary",
    "readiness_audit",
    "candidate_alpha_groups",
    "split_policy",
    "baseline_matrix",
    "metric_schema",
    "runtime_risks",
    "recommended_next_model_round",
)


def _artifact_status(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
    }


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _finite_range(values: pd.Series) -> list[float | None]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return [None, None]
    return [float(numeric.min()), float(numeric.max())]


def build_readiness_audit(repo_root: Path) -> dict[str, Any]:
    diagnostics = repo_root / "outputs" / "diagnostics"
    contract = _safe_read_csv(diagnostics / "huayi_expanded_model_case_contract_audit.csv")
    baseline = _safe_read_csv(diagnostics / "huayi_expanded_fixed_physics_diagnostic_baseline.csv")
    engineering = _safe_read_csv(diagnostics / "huayi_engineering_coordinate_influence_diagnostic.csv")
    oc_response = _safe_read_csv(diagnostics / "huayi_oc_response_curve_diagnostic.csv")
    bounded = _safe_read_csv(diagnostics / "huayi_bounded_hybrid_correction_batch_smoke.csv")
    bounded_batch = _safe_read_csv(diagnostics / "huayi_bounded_hybrid_correction_batch_smoke_batch_summary.csv")

    target_columns = [
        column
        for column in (
            "target_CO_vol_pct",
            "target_H2_vol_pct",
            "target_CO2_vol_pct",
            "target_N2_vol_pct",
            "target_dry_flow_kmol_s",
            "target_H2_CO_ratio",
        )
        if column in baseline.columns
    ]
    phases = contract["phase"].value_counts().sort_index().to_dict() if "phase" in contract else {}
    engineering_rank = []
    if not engineering.empty:
        engineering_rank = (
            engineering[
                [
                    "coordinate",
                    "aggregate_coordinate_rank",
                    "aggregate_mean_abs_normalized_sensitivity",
                ]
            ]
            .drop_duplicates()
            .sort_values("aggregate_coordinate_rank")
            .to_dict("records")
        )

    return {
        "repo_dirty_state": "dirty_worktree_expected_from_prior_model_rounds",
        "huayi_ready_model_cases": int(contract["model_case_ready"].sum()) if "model_case_ready" in contract else 0,
        "huayi_contract_rows": int(len(contract)),
        "phase_counts": {str(k): int(v) for k, v in phases.items()},
        "target_columns_available": target_columns,
        "targets_sufficient_for": {
            "CO_H2_CO2_H2CO": all(
                column in target_columns
                for column in ["target_CO_vol_pct", "target_H2_vol_pct", "target_CO2_vol_pct", "target_H2_CO_ratio"]
            ),
            "outlet_temperature": "T_out_degC" in baseline.columns,
            "dry_syngas_flow": "target_dry_flow_kmol_s" in target_columns,
        },
        "baseline_rows": int(len(baseline)),
        "baseline_n_cells": sorted(int(v) for v in baseline["N_cells"].dropna().unique()) if "N_cells" in baseline else [],
        "baseline_fallback_distribution": baseline["fallback_count"].value_counts().sort_index().to_dict() if "fallback_count" in baseline else {},
        "baseline_poor_convergence_distribution": baseline["poor_convergence_count"].value_counts().sort_index().to_dict() if "poor_convergence_count" in baseline else {},
        "baseline_cell0_cost_range": _finite_range(baseline["cell0_cost"]) if "cell0_cost" in baseline else [None, None],
        "engineering_coordinate_rank": engineering_rank,
        "oc_response_rows": int(len(oc_response)),
        "oc_response_ok_rows": int((oc_response["status"] == "ok").sum()) if "status" in oc_response else 0,
        "bounded_wgs_batch_rows": int(len(bounded)),
        "bounded_wgs_loss_decreased": int(bounded["loss_decreased"].sum()) if "loss_decreased" in bounded else 0,
        "bounded_wgs_final_theta_range": _finite_range(bounded["final_theta"]) if "final_theta" in bounded else [None, None],
        "bounded_wgs_batch_recommendation": bounded_batch.iloc[0]["recommendation"] if not bounded_batch.empty and "recommendation" in bounded_batch else "not_available",
        "relevant_artifacts": [
            _artifact_status(diagnostics / name)
            for name in (
                "huayi_expanded_model_case_contract_audit.csv",
                "huayi_expanded_fixed_physics_diagnostic_baseline.csv",
                "huayi_engineering_coordinate_influence_diagnostic.csv",
                "huayi_oc_response_curve_diagnostic.csv",
                "huayi_bounded_hybrid_correction_batch_smoke.csv",
                "gradient_parity_diagnostic.csv",
            )
        ],
        "split_blockers": [
            "Current paper-facing set has only 22 representative ModelCases; adequate for protocol smoke and leave-future-out diagnostics, too small for a high-capacity trained hybrid.",
            "Representative rows span four Huayi phases but are not an exhaustive steady-window training set.",
            "Sparse LIMS/coal labels require timestamp alignment rules and train-only preprocessing before they can be used as z_d weak labels.",
            "Yulin/slurry data are out of scope for this protocol unless a separate ModelCase contract is built.",
        ],
    }


def candidate_alpha_groups() -> list[dict[str, Any]]:
    return [
        {
            "name": "alpha_wgs",
            "reaction_family": "water-gas-shift net family CO+H2O <-> CO2+H2",
            "mapped_existing_parameter": "op_conds.WGS_CatalyticFactor",
            "implementation_change": "none for scalar smoke; bounded alpha(x_t,z_d) wrapper needed for training",
            "physical_rationale": "WGS directly redistributes CO/H2/CO2 and has already shown 22-case bounded scalar loss-decrease smoke.",
            "expected_target_sensitivity": "CO, H2, CO2, H2/CO strongest; moderate outlet T through reaction heat.",
            "default_bounds": [0.02, 2.0],
            "train_without_public_solver_api_change": True,
            "current_evidence": "round38/39 bounded WGS scalar smoke; 22/22 loss-decrease, no bound hits.",
            "identifiability_risk": "High with O/C, steam/water availability, CO2 carrier, and coal-quality weak labels; must report alpha distribution and bound hits.",
        },
        {
            "name": "alpha_char_oxidation",
            "reaction_family": "char oxidation / first combustion severity C+O2",
            "mapped_existing_parameter": "op_conds.CharCombustionRateFactor",
            "implementation_change": "none for scalar mainline/JAX parameter; bounded alpha wrapper required for training",
            "physical_rationale": "Controls first-cell heat release, oxygen consumption, CO/CO2 split through phi-coupled char oxidation, and persistent cell0 boundary/source-term behavior.",
            "expected_target_sensitivity": "outlet T, CO, CO2, dry flow; can interact strongly with O/C response.",
            "default_bounds": [0.05, 1.0],
            "train_without_public_solver_api_change": True,
            "current_evidence": "parameter is already wired into Cell and JAX solver contract; no 22-case bounded training smoke yet.",
            "identifiability_risk": "Very high with O/C ratio and first-cell initialization; should be introduced after WGS-only baseline and monitored with cell0 cost.",
        },
        {
            "name": "alpha_volatile_oxidation",
            "reaction_family": "combustion-zone volatile CO/H2 oxidation",
            "mapped_existing_parameter": "op_conds.CO_OxidationFactor and op_conds.H2_OxidationFactor",
            "implementation_change": "minimal if exposed as one shared alpha or two bounded alphas; JAX parity coverage should be audited before training.",
            "physical_rationale": "Existing first-cell combustion branch has explicit CO and H2 oxidation dampers that can localize H2/CO2 mismatch without inventing new source terms.",
            "expected_target_sensitivity": "H2, CO, CO2, H2/CO; possibly T through oxidation heat.",
            "default_bounds": [0.2, 1.2],
            "train_without_public_solver_api_change": True,
            "current_evidence": "mainline parameters exist; not yet included in 22-case gradient/parity or bounded batch smoke.",
            "identifiability_risk": "High with WGS and char oxidation; should use staged ablation and bound-hit reporting.",
        },
        {
            "name": "alpha_char_gasification",
            "reaction_family": "char gasification C+CO2 and C+H2O",
            "mapped_existing_parameter": "none as a clean grouped op_conds alpha",
            "implementation_change": "requires small internal interface in Cell/KineticsService to multiply selected heterogeneous rates after rate calculation.",
            "physical_rationale": "Can address downstream CO/H2 generation and carbon conversion/mass-flow mismatch beyond first-cell oxidation.",
            "expected_target_sensitivity": "CO, H2, dry syngas flow, carbon sanity; weaker direct effect on early O2 zone.",
            "default_bounds": [0.1, 5.0],
            "train_without_public_solver_api_change": False,
            "current_evidence": "design-only; implement only after WGS/char oxidation baselines are stable.",
            "identifiability_risk": "Very high with coal reactivity, particle diameter, residence time, and carbon-balance incompleteness.",
        },
    ]


def split_policy() -> dict[str, Any]:
    return {
        "minimum_training_dataset": "Use the larger Huayi steady-window ModelCase set once available; 22 representative rows are for smoke/protocol and leave-future-out diagnostics only.",
        "chronological_22_case_diagnostic_split": {
            "train": "phase1 + phase2 representative cases (12 rows)",
            "validation": "phase3 representative cases (6 rows)",
            "test": "phase4 representative cases (4 rows)",
            "rationale": "preserves future holdout by campaign phase/date; intentionally small and diagnostic-only.",
        },
        "leave_future_out": "For full steady-window set, train on earliest contiguous blocks, tune on the next chronological block, and report the latest block once. No random shuffling before split.",
        "leave_regime_out": "Not robust with only 22 representatives. Feasible only after defining regimes by O/C/load/carrier/phase and ensuring at least one complete future-like regime is held out.",
        "lims_weak_label_policy": [
            "Align LIMS/coal-quality labels by timestamp using only labels available at or before the ModelCase timestamp.",
            "No interpolation or smoothing may use future labels relative to validation/test rows.",
            "Fit normalization, PCA/latent encoders, imputation medians, and lag windows on train blocks only.",
            "Include LIMS shuffle, delay, and dropout controls before claiming weak-label value.",
        ],
        "leakage_guards": [
            "No statistics fit on all 22 rows before split.",
            "No phase-level feature_version target leakage into model features unless used only as documented metadata/control.",
            "Repeated rows from same independence group must stay in one split for larger datasets.",
        ],
    }


def baseline_matrix() -> list[dict[str, Any]]:
    return [
        {
            "name": "fixed_physics",
            "description": "Current GasifierSystem fixed physics with ModelCase adapter.",
            "implementation_effort": "existing",
            "expected_changed_files": [],
            "output_schema": "case_id, split, target, prediction, residual, solver status, cell0 diagnostics, evidence_boundary",
            "tests": "adapter tests, diagnostic baseline schema tests, git diff --check",
            "runtime_estimate": "minutes for 22 cases at N=3/5; larger steady-window set may be hours",
            "evidence_boundary": "diagnostic baseline only, not calibrated accuracy",
        },
        {
            "name": "direct_outlet_residual_correction",
            "description": "Post-solver residual model maps process features to outlet residuals without changing physics.",
            "implementation_effort": "small",
            "expected_changed_files": ["src/model/hybrid_training_protocol.py", "scripts/generate_hybrid_baselines.py"],
            "output_schema": "case_id, split, target, fixed_prediction, corrected_prediction, residual_before, residual_after, status",
            "tests": "split integrity, no future leakage, schema, deterministic fit on toy data",
            "runtime_estimate": "low; no inner differentiable solver training",
            "evidence_boundary": "statistical residual baseline, not mechanistic alpha_j evidence",
        },
        {
            "name": "direct_kinetic_parameter_correction",
            "description": "Fit a small number of global scalar kinetic multipliers with bounds.",
            "implementation_effort": "medium",
            "expected_changed_files": ["src/model/bounded_hybrid_correction_smoke.py", "scripts/generate_global_alpha_baseline.py"],
            "output_schema": "split, alpha_name, alpha_value, bound_hit, metrics, solver diagnostics",
            "tests": "bound enforcement, monotonic loss smoke, schema",
            "runtime_estimate": "medium; repeated solver calls per candidate alpha",
            "evidence_boundary": "bounded global calibration baseline, not learned hybrid interpretability",
        },
        {
            "name": "bounded_alpha_xt",
            "description": "alpha_j(x_t) uses online/process variables only with hard sigmoid/logit bounds.",
            "implementation_effort": "medium-high",
            "expected_changed_files": ["src/model/hybrid_alpha.py", "src/model/hybrid_objective.py", "scripts/train_bounded_alpha.py"],
            "output_schema": "case_id, split, alpha_j, alpha_value, prediction, metric fields, bound hit, solver diagnostics",
            "tests": "bound transform, split, deterministic tiny training, no hidden solver API change",
            "runtime_estimate": "high if trained through solver; start with small N_cells and cached fixed branches",
            "evidence_boundary": "hybrid readiness until validated on future holdout",
        },
        {
            "name": "bounded_alpha_xt_zd",
            "description": "alpha_j(x_t,z_d) adds delayed/weak coal-quality labels or coal latent descriptor.",
            "implementation_effort": "high",
            "expected_changed_files": ["src/model/hybrid_alpha.py", "src/model/lims_alignment.py", "scripts/train_bounded_alpha.py"],
            "output_schema": "all bounded_alpha_xt fields plus z_d availability, lag, source, dropout status",
            "tests": "no future leakage, LIMS delay/shuffle/dropout controls, split integrity",
            "runtime_estimate": "high; do only after x_t baseline is stable",
            "evidence_boundary": "no weak-label value claim unless controls beat x_t-only baseline on future holdout",
        },
        {
            "name": "time_embedding_only_control",
            "description": "Use time/phase/position features without process or coal labels.",
            "implementation_effort": "small-medium",
            "expected_changed_files": ["src/model/hybrid_controls.py", "scripts/train_control_baselines.py"],
            "output_schema": "case_id, split, target, prediction, metrics, control_status",
            "tests": "split integrity and leakage guard",
            "runtime_estimate": "low-medium",
            "evidence_boundary": "detects temporal memorization; not physics correction",
        },
        {
            "name": "lims_shuffle_delay_dropout_controls",
            "description": "Controls for weak-label coal features: shuffled, excessively delayed, and dropout variants.",
            "implementation_effort": "medium",
            "expected_changed_files": ["src/model/lims_alignment.py", "scripts/train_control_baselines.py"],
            "output_schema": "case_id, split, control_type, z_d_status, metrics, evidence_boundary",
            "tests": "deterministic shuffle seed, delay rule, dropout mask, no future labels",
            "runtime_estimate": "medium; repeats alpha training for controls",
            "evidence_boundary": "required before any weak-label coal-quality claim",
        },
    ]


def metric_schema() -> dict[str, Any]:
    return {
        "required_prediction_targets": ["CO_volpct_dry", "H2_volpct_dry", "CO2_volpct_dry", "H2_CO_ratio"],
        "optional_prediction_targets": ["T_out_degC", "dry_syngas_flow_kmol_s", "N2_volpct_dry"],
        "per_target_metrics": ["RMSE", "MAE", "bias", "n"],
        "diagnostic_fields": [
            "case_id",
            "phase",
            "dataset_id",
            "split",
            "N_cells",
            "fallback_count",
            "poor_convergence_count",
            "poor_convergence_cell_index",
            "cell0_cost",
            "cell0_max_abs_residual",
            "carbon_sanity_status",
            "gross_apparent_gas_phase_C_recovery",
            "bound_hit_count",
            "bound_hit_fraction",
            "alpha_min",
            "alpha_median",
            "alpha_max",
            "alpha_p05",
            "alpha_p95",
            "status",
            "evidence_boundary",
        ],
        "artifact_schema": {
            "case_predictions_csv": "case_id, split, phase, target, y_true, y_fixed, y_pred, residual_fixed, residual_hybrid, status",
            "metric_summary_csv": "split, target, baseline_name, RMSE, MAE, bias, n, status, evidence_boundary",
            "alpha_summary_csv": "split, alpha_name, min, p05, median, p95, max, bound_low, bound_high, bound_hit_fraction",
            "solver_diagnostics_csv": "case_id, split, fallback_count, poor_convergence_count, poor_convergence_cell_index, cell0_cost, carbon_sanity_status",
        },
    }


def runtime_risks() -> list[str]:
    return [
        "Training through the implicit solver may be slow; start with N_cells=3 smoke and one alpha before multi-alpha/full N_cells.",
        "Persistent cell0 residual-quality flag can dominate diagnostics; report it separately and do not hide it by threshold changes.",
        "Alpha groups are correlated with O/C, load, carrier ratio, and coal weak labels; staged ablations are mandatory.",
        "Bound hits may mimic performance gains; report alpha distribution and hit frequency by split/phase.",
        "The 22 representative rows are too small for high-capacity learned alpha networks; use larger steady-window set before accuracy claims.",
        "LIMS labels are sparse and temporally delayed; future leakage is the highest risk for z_d experiments.",
        "JAX/IFT coverage is partial; full production training may need fallback to fixed-branch probes or careful custom VJP expansion.",
    ]


def build_protocol(repo_root: Path) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "claim_boundary": CLAIM_BOUNDARY,
        "readiness_audit": build_readiness_audit(repo_root),
        "candidate_alpha_groups": candidate_alpha_groups(),
        "split_policy": split_policy(),
        "baseline_matrix": baseline_matrix(),
        "metric_schema": metric_schema(),
        "runtime_risks": runtime_risks(),
        "recommended_next_model_round": {
            "title": "Implement split manifest + fixed-physics and global bounded-alpha baseline scaffolding",
            "tasks": [
                "Create a chronological split manifest for Huayi ModelCases without random shuffling.",
                "Implement metric schema writer using existing fixed-physics outputs.",
                "Run WGS-only global bounded alpha baseline on train/validation/test split as a non-learned calibration baseline.",
                "Add leakage-guard tests before any alpha(x_t,z_d) experiment.",
            ],
            "do_not_start_yet": [
                "MLP training on all candidate alphas",
                "weak-label coal latent claims",
                "production-control framing",
            ],
        },
    }


def protocol_to_markdown(protocol: Mapping[str, Any]) -> str:
    audit = protocol["readiness_audit"]
    lines: list[str] = [
        "# Hybrid calibration protocol freeze",
        "",
        f"- Protocol version: `{protocol['protocol_version']}`",
        f"- Claim boundary: {protocol['claim_boundary']}",
        "",
        "## Readiness Audit",
        "",
        f"- Huayi READY ModelCases: {audit['huayi_ready_model_cases']}/{audit['huayi_contract_rows']}",
        f"- Phase counts: {audit['phase_counts']}",
        f"- Baseline rows: {audit['baseline_rows']} with N_cells={audit['baseline_n_cells']}",
        f"- Available target columns: {', '.join(audit['target_columns_available'])}",
        f"- Targets sufficient for CO/H2/CO2/H2CO: {audit['targets_sufficient_for']['CO_H2_CO2_H2CO']}",
        f"- Fixed-physics fallback distribution: {audit['baseline_fallback_distribution']}",
        f"- Fixed-physics poor-convergence distribution: {audit['baseline_poor_convergence_distribution']}",
        f"- Cell0 cost range: {audit['baseline_cell0_cost_range']}",
        f"- Bounded WGS batch: {audit['bounded_wgs_loss_decreased']}/{audit['bounded_wgs_batch_rows']} loss decreased; final theta range {audit['bounded_wgs_final_theta_range']}",
        "",
        "### Split Blockers",
        "",
    ]
    lines.extend(f"- {item}" for item in audit["split_blockers"])
    lines.extend(
        [
            "",
            "## Candidate Alpha Groups",
            "",
            "| alpha | existing mapping | bounds | train without public API change | sensitivity | identifiability risk |",
            "|---|---|---:|---|---|---|",
        ]
    )
    for row in protocol["candidate_alpha_groups"]:
        lines.append(
            "| {name} | {mapping} | {bounds} | {api} | {sens} | {risk} |".format(
                name=row["name"],
                mapping=row["mapped_existing_parameter"],
                bounds=row["default_bounds"],
                api=row["train_without_public_solver_api_change"],
                sens=row["expected_target_sensitivity"],
                risk=row["identifiability_risk"],
            )
        )
    lines.extend(
        [
            "",
            "## Split Policy",
            "",
            f"- Minimum dataset: {protocol['split_policy']['minimum_training_dataset']}",
            f"- Chronological 22-case diagnostic split: {protocol['split_policy']['chronological_22_case_diagnostic_split']}",
            f"- Leave-future-out: {protocol['split_policy']['leave_future_out']}",
            f"- Leave-regime-out: {protocol['split_policy']['leave_regime_out']}",
            "",
            "### LIMS Weak-Label Leakage Policy",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in protocol["split_policy"]["lims_weak_label_policy"])
    lines.extend(
        [
            "",
            "## Baseline Matrix",
            "",
            "| baseline | effort | runtime | evidence boundary |",
            "|---|---|---|---|",
        ]
    )
    for row in protocol["baseline_matrix"]:
        lines.append(
            f"| {row['name']} | {row['implementation_effort']} | {row['runtime_estimate']} | {row['evidence_boundary']} |"
        )
    schema = protocol["metric_schema"]
    lines.extend(
        [
            "",
            "## Metric Schema",
            "",
            f"- Required targets: {', '.join(schema['required_prediction_targets'])}",
            f"- Optional targets: {', '.join(schema['optional_prediction_targets'])}",
            f"- Per-target metrics: {', '.join(schema['per_target_metrics'])}",
            f"- Diagnostic fields: {', '.join(schema['diagnostic_fields'])}",
            "",
            "## Runtime Risks",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in protocol["runtime_risks"])
    next_round = protocol["recommended_next_model_round"]
    lines.extend(
        [
            "",
            "## Recommended Next Model Round",
            "",
            f"- Title: {next_round['title']}",
            "",
            "Tasks:",
        ]
    )
    lines.extend(f"- {item}" for item in next_round["tasks"])
    lines.extend(
        [
            "",
            "Do not start yet:",
        ]
    )
    lines.extend(f"- {item}" for item in next_round["do_not_start_yet"])
    lines.extend(
        [
            "",
            f"Evidence boundary: {protocol['claim_boundary']}.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_protocol_artifacts(repo_root: Path) -> tuple[Path, Path, dict[str, Any]]:
    protocol = build_protocol(repo_root)
    out_dir = repo_root / "outputs" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "hybrid_calibration_protocol.json"
    md_path = out_dir / "hybrid_calibration_protocol.md"
    json_path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(protocol_to_markdown(protocol), encoding="utf-8")
    return json_path, md_path, protocol


__all__ = [
    "CLAIM_BOUNDARY",
    "PROTOCOL_VERSION",
    "REQUIRED_PROTOCOL_KEYS",
    "baseline_matrix",
    "build_protocol",
    "candidate_alpha_groups",
    "metric_schema",
    "protocol_to_markdown",
    "runtime_risks",
    "split_policy",
    "write_protocol_artifacts",
]
