"""Figure-ready adaptive-gate source data extraction.

This module normalizes existing Huayi round41-round48 diagnostics for the
paper-side Fig. 4/Fig. 5 adaptive update gate. It only reads existing artifacts
and writes source-data tables; it does not run solvers, fit models, or search
new hyperparameters.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

GATE_FIGURE_BOUNDARY = (
    "source-data extraction for a gate-diagnostic case study only; not new "
    "modeling, final temporal generalization, production-control evidence, "
    "learned kinetic identification, weak-label coal-quality value, or "
    "cross-gasifier generalization"
)

NOT_AVAILABLE = "not_available_from_existing_artifact"

ALLOWED_STATUSES = ("provisionally_open", "locked", "audit_only", "future_data_needed")

REQUIRED_FIG4_METHOD_LABELS = (
    "fixed physics",
    "direct residual",
    "global WGS",
    "global char oxidation",
    "WGS alpha_wgs(x_t)",
    "global multi-alpha",
)

SOURCE_ARTIFACTS: tuple[dict[str, str], ...] = (
    {"artifact_key": "round41_fixed_physics_metrics", "path": "outputs/diagnostics/huayi_fixed_physics_split_metrics.csv", "round": "round41", "kind": "metric"},
    {"artifact_key": "round41_global_wgs_metrics", "path": "outputs/diagnostics/huayi_global_wgs_alpha_metrics.csv", "round": "round41", "kind": "metric"},
    {"artifact_key": "round42_direct_residual_metrics", "path": "outputs/diagnostics/huayi_direct_residual_baseline_metrics.csv", "round": "round42", "kind": "metric"},
    {"artifact_key": "round43_global_char_metrics", "path": "outputs/diagnostics/huayi_global_char_oxidation_alpha_metrics.csv", "round": "round43", "kind": "metric"},
    {"artifact_key": "round44_wgs_alpha_xt_metrics", "path": "outputs/diagnostics/huayi_alpha_wgs_xt_metrics.csv", "round": "round44", "kind": "metric"},
    {"artifact_key": "round45_wgs_robustness_grid", "path": "outputs/diagnostics/huayi_alpha_wgs_xt_robustness_grid.csv", "round": "round45", "kind": "audit"},
    {"artifact_key": "round45_wgs_null_control", "path": "outputs/diagnostics/huayi_alpha_wgs_xt_null_control.csv", "round": "round45", "kind": "audit"},
    {"artifact_key": "round45_wgs_sensitivity_proxy", "path": "outputs/diagnostics/huayi_alpha_wgs_sensitivity_proxy.csv", "round": "round45", "kind": "sensitivity"},
    {"artifact_key": "round46_candidate_similarity", "path": "outputs/diagnostics/huayi_alpha_candidate_similarity.csv", "round": "round46", "kind": "audit"},
    {"artifact_key": "round46_candidate_recommendations", "path": "outputs/diagnostics/huayi_alpha_candidate_recommendations.csv", "round": "round46", "kind": "recommendation"},
    {"artifact_key": "round47_global_multialpha_metrics", "path": "outputs/diagnostics/huayi_global_multialpha_metrics.csv", "round": "round47", "kind": "metric"},
    {"artifact_key": "round47_global_multialpha_summary", "path": "outputs/diagnostics/huayi_global_multialpha_summary.csv", "round": "round47", "kind": "selection"},
    {"artifact_key": "round48_evidence_bundle_manifest", "path": "outputs/diagnostics/huayi_hybrid_diagnostic_evidence_manifest.csv", "round": "round48", "kind": "provenance"},
    {"artifact_key": "round48_evidence_bundle_metric_matrix", "path": "outputs/diagnostics/huayi_hybrid_diagnostic_metric_matrix.csv", "round": "round48", "kind": "metric"},
    {"artifact_key": "round48_evidence_bundle_claim_table", "path": "outputs/diagnostics/huayi_hybrid_diagnostic_claim_table.csv", "round": "round48", "kind": "claim"},
    {"artifact_key": "round48_evidence_bundle_recommendations", "path": "outputs/diagnostics/huayi_hybrid_diagnostic_recommendations.csv", "round": "round48", "kind": "recommendation"},
)

GENERATED_FILES: tuple[tuple[str, str, str], ...] = (
    ("fig4_metric_matrix", "huayi_adaptive_gate_fig4_metric_matrix.csv", "Fig. 4 panel B/C source data"),
    ("fig4_method_ladder", "huayi_adaptive_gate_fig4_method_ladder.csv", "Fig. 4 panel A method/comparator ladder"),
    ("channel_status", "huayi_adaptive_gate_channel_status.csv", "Fig. 4 panel D and Table X source traceability"),
    ("reaction_family_gate_map", "huayi_reaction_family_gate_map.csv", "Fig. 5 panel A/D source data"),
    ("alpha_similarity", "huayi_alpha_similarity_for_fig5.csv", "Fig. 5 panel B source data"),
    ("multialpha_activation", "huayi_multialpha_activation_for_fig5.csv", "Fig. 5 panel C source data"),
    ("manifest", "huayi_gate_figure_source_manifest.csv", "round49 provenance and row counts"),
    ("markdown", "huayi_gate_figure_source_bundle.md", "human-readable source-data bundle"),
)


def _read_csv(repo_root: Path, rel_path: str) -> pd.DataFrame:
    path = repo_root / rel_path
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _source_row(source_round: str, source_artifact: str, evidence_role: str = "source_data_extraction") -> dict[str, str]:
    return {
        "source_round": source_round,
        "source_artifact_path": source_artifact,
        "evidence_role": evidence_role,
        "allowed_claim": "supports adaptive gate source-data extraction and traceability",
        "forbidden_claim": (
            "Do not claim new model performance, final predictive accuracy, production-control "
            "readiness, learned kinetic identification, learned multi-alpha success, weak-label "
            "coal-quality value, final temporal generalization, or cross-gasifier generalization."
        ),
        "evidence_boundary": GATE_FIGURE_BOUNDARY,
    }


def _fig4_method_label(method_label: str) -> str:
    if method_label == "fixed_physics":
        return "fixed physics"
    if method_label == "global_wgs_alpha":
        return "global WGS"
    if method_label.startswith("direct_residual"):
        return "direct residual"
    if method_label == "global_char_oxidation_alpha":
        return "global char oxidation"
    if method_label == "alpha_wgs_xt_scaffold":
        return "WGS alpha_wgs(x_t)"
    if method_label.startswith("global_multialpha"):
        return "global multi-alpha"
    return method_label


def _method_class(method_label: str) -> str:
    return {
        "fixed physics": "fixed_physics",
        "direct residual": "residual_patch",
        "global WGS": "global_alpha",
        "global char oxidation": "global_alpha",
        "WGS alpha_wgs(x_t)": "alpha_xt",
        "global multi-alpha": "multialpha_diagnostic",
    }[method_label]


def _train_selection(method_label: str) -> str:
    return {
        "fixed physics": "none",
        "direct residual": "train_only_feature",
        "global WGS": "train_only_scalar",
        "global char oxidation": "train_only_scalar",
        "WGS alpha_wgs(x_t)": "train_only_feature",
        "global multi-alpha": "train_only_subset",
    }[method_label]


def _physical_route(method_label: str) -> str:
    return {
        "fixed physics": "none",
        "direct residual": "outlet_residual",
        "global WGS": "WGS",
        "global char oxidation": "char_oxidation",
        "WGS alpha_wgs(x_t)": "WGS",
        "global multi-alpha": "multi_alpha",
    }[method_label]


def _method_source_round(method_label: str) -> str:
    return {
        "fixed physics": "round41",
        "direct residual": "round42",
        "global WGS": "round41",
        "global char oxidation": "round43",
        "WGS alpha_wgs(x_t)": "round44-round45",
        "global multi-alpha": "round47",
    }[method_label]


def _method_allowed_claim(method_label: str) -> str:
    return {
        "fixed physics": "Split-aware fixed-physics reference with solver-quality fields.",
        "direct residual": "Black-box residual-patch comparator that motivates a physical update gate.",
        "global WGS": "Simplest bounded WGS scalar comparator selected on train split only.",
        "global char oxidation": "Audit comparator for near-burner char oxidation under current measurements.",
        "WGS alpha_wgs(x_t)": "Provisionally open bounded WGS adaptive scaffold under diagnostic split.",
        "global multi-alpha": "Stopping-rule diagnostic before learned multi-alpha under current data.",
    }[method_label]


def _method_forbidden_claim(method_label: str) -> str:
    return {
        "fixed physics": "calibrated fixed-physics accuracy or industrial generalization",
        "direct residual": "physically admissible online self-update or mechanistic hybrid learning",
        "global WGS": "final WGS calibration or production online adaptation",
        "global char oxidation": "identified burner-zone kinetics or safe online char-oxidation updating",
        "WGS alpha_wgs(x_t)": "learned WGS kinetic identification or final temporal generalization",
        "global multi-alpha": "successful learned multi-alpha calibration or production-ready adaptation",
    }[method_label]


def build_fig4_method_ladder() -> list[dict[str, Any]]:
    labels = list(REQUIRED_FIG4_METHOD_LABELS)
    gate_questions = {
        "fixed physics": "What does the verified fixed reactor model explain before updates?",
        "direct residual": "What happens if outlet residuals are patched without a reactor route?",
        "global WGS": "Does one bounded WGS scalar provide a physical comparator?",
        "global char oxidation": "Can near-burner char oxidation be safely opened from outlet data?",
        "WGS alpha_wgs(x_t)": "Can a low-capacity bounded WGS channel use operating features?",
        "global multi-alpha": "Should multiple reaction-family degrees of freedom be opened now?",
    }
    rows = []
    for idx, label in enumerate(labels, start=1):
        rows.append(
            {
                "stage_order": idx,
                "method_label": label,
                "method_class": _method_class(label),
                "train_selection": _train_selection(label),
                "physical_route": _physical_route(label),
                "gate_question": gate_questions[label],
                **_source_row(_method_source_round(label), "outputs/diagnostics/huayi_hybrid_diagnostic_metric_matrix.csv"),
                "allowed_claim": _method_allowed_claim(label),
                "forbidden_claim": _method_forbidden_claim(label),
            }
        )
    return rows


def _metric_evidence_role(method_label: str, split: str) -> str:
    if split in {"validation", "test"}:
        return "held_out"
    if method_label == "fixed physics":
        return "post_hoc_diagnostic"
    return "train_selected"


def build_fig4_metric_matrix(repo_root: str | Path) -> list[dict[str, Any]]:
    root = Path(repo_root)
    metrics = _read_csv(root, "outputs/diagnostics/huayi_hybrid_diagnostic_metric_matrix.csv")
    rows: list[dict[str, Any]] = []
    for _, row in metrics.iterrows():
        label = _fig4_method_label(str(row["method_label"]))
        split = str(row["split"])
        evidence_role = _metric_evidence_role(label, split)
        baseline_method = "global_wgs" if label in {"WGS alpha_wgs(x_t)", "global multi-alpha"} else "fixed_physics"
        rows.append(
            {
                "method_label": label,
                "source_method_detail": row["method_label"],
                "split": split,
                "target": row["target"],
                "N_cells": int(row["N_cells"]),
                "n": int(row["n"]),
                "rmse": row["rmse"],
                "mae": row["mae"],
                "bias": row["bias"],
                "baseline_method": baseline_method,
                "rmse_delta_vs_fixed": row["rmse_delta_vs_fixed"],
                "rmse_delta_vs_global_wgs": row["rmse_delta_vs_global_wgs"],
                "evidence_role": evidence_role,
                "fallback_count": row["fallback_count_sum"],
                "poor_convergence_count": row["poor_convergence_count_sum"],
                **_source_row(row["source_round"], row["source_artifact"], evidence_role),
            }
        )
    return rows


def build_channel_status() -> list[dict[str, Any]]:
    rows = [
        {
            "channel": "direct_residual",
            "reactor_process": "Outlet composition residual patch outside reactor source terms.",
            "model_interface": "huayi_direct_residual_baseline_* residual correction outputs",
            "gate_status": "locked",
            "status_reason": "Round42 is useful as a black-box comparator, but validation/test behavior is mixed and it has no physical reactor route.",
            "source_rounds": "round42,round48",
            "allowed_online_policy": "Use only as comparator, alarm signal, or diagnostic upper-bound for data-driven flexibility.",
            "forbidden_claim": "Do not allow as routine online physical self-update or call it mechanistic hybrid learning.",
            "table_row_label": "Direct outlet residual correction",
            "current_huayi_evidence_short": "Train-only residual patch; mixed held-out behavior.",
            "manuscript_wording_short": "Residual patching motivates the physical update gate.",
        },
        {
            "channel": "alpha_wgs",
            "reactor_process": "Gas-phase WGS redistribution among CO, H2, and CO2.",
            "model_interface": "op_conds.WGS_CatalyticFactor",
            "gate_status": "audit_only",
            "status_reason": "Round41 global WGS scalar is a bounded comparator, but by itself is not enough for online adaptation.",
            "source_rounds": "round41,round48",
            "allowed_online_policy": "Use as bounded WGS reference channel and audit comparator.",
            "forbidden_claim": "Do not present as final calibration, learned kinetic identification, or production online adaptation.",
            "table_row_label": "Global alpha_wgs",
            "current_huayi_evidence_short": "Train-selected alpha_wgs=0.3; mixed held-out metrics.",
            "manuscript_wording_short": "A bounded WGS scalar motivates operation-dependent WGS updating.",
        },
        {
            "channel": "alpha_wgs_xt",
            "reactor_process": "Operation-dependent bounded WGS update driven by DCS operating features.",
            "model_interface": "alpha_wgs_ridge_logit(x_t) -> op_conds.WGS_CatalyticFactor",
            "gate_status": "provisionally_open",
            "status_reason": "Rounds44-45 show leakage-safe low-capacity WGS scaffold, stable selected feature, null-control warning, and mixed sensitivity signs.",
            "source_rounds": "round44,round45,round48",
            "allowed_online_policy": "Use as first bounded adaptive diagnostic channel under conservative gate language.",
            "forbidden_claim": "Do not claim learned WGS kinetic identification, calibrated accuracy, or final temporal generalization.",
            "table_row_label": "Operation-dependent alpha_wgs(x_t)",
            "current_huayi_evidence_short": "Low-capacity WGS scaffold is runnable; robustness audit warns it is compensatory.",
            "manuscript_wording_short": "WGS is provisionally open as a bounded diagnostic channel.",
        },
        {
            "channel": "alpha_char_oxidation",
            "reactor_process": "Near-burner char oxidation, oxygen consumption, heat release, and particle conversion.",
            "model_interface": "op_conds.CharCombustionRateFactor",
            "gate_status": "audit_only",
            "status_reason": "Round43/46 indicate a sensitive but compensatory channel with limited observability from outlet dry-gas targets.",
            "source_rounds": "round43,round46,round48",
            "allowed_online_policy": "Keep as audit channel and measurement-needs example.",
            "forbidden_claim": "Do not allow learned online updating or claim identified char-oxidation kinetics.",
            "table_row_label": "Global alpha_char_oxidation",
            "current_huayi_evidence_short": "Train-selected scalar improves modestly; held-out and identifiability evidence remain weak.",
            "manuscript_wording_short": "Char oxidation remains an audit channel under current measurements.",
        },
        {
            "channel": "alpha_volatile_oxidation",
            "reactor_process": "Volatile release and early gas-phase oxidation near the inlet/burner zone.",
            "model_interface": "shared scalar over op_conds.CO_OxidationFactor + op_conds.H2_OxidationFactor",
            "gate_status": "audit_only",
            "status_reason": "Round46 shows high sensitivity and mixed signs; round47 leaves volatile oxidation at baseline in selected global subsets.",
            "source_rounds": "round46,round47,round48",
            "allowed_online_policy": "Audit sensitivity only; require additional measurements or priors before online updating.",
            "forbidden_claim": "Do not claim online volatile-oxidation self-evolution or learned multi-alpha success.",
            "table_row_label": "alpha_volatile_oxidation",
            "current_huayi_evidence_short": "High leverage but not activated by conservative global multi-alpha diagnostic.",
            "manuscript_wording_short": "Volatile oxidation is high-risk and remains audit-only.",
        },
        {
            "channel": "alpha_char_gasification",
            "reactor_process": "Char gasification by CO2/H2O or related heterogeneous gasification routes.",
            "model_interface": "no clean grouped op_conds path yet",
            "gate_status": "future_data_needed",
            "status_reason": "Round40/46 keep this design-only because no clean bounded reaction-family interface exists without changing solver semantics.",
            "source_rounds": "round40,round46,round48",
            "allowed_online_policy": "Keep as design placeholder until model path and validation data exist.",
            "forbidden_claim": "Do not include as current learned channel or adaptive update result.",
            "table_row_label": "alpha_char_gasification",
            "current_huayi_evidence_short": "Design-only; no clean grouped parameter path.",
            "manuscript_wording_short": "Char gasification is outside the current gate.",
        },
        {
            "channel": "learned_multialpha",
            "reactor_process": "Simultaneous adaptive updating of multiple reaction-family multipliers.",
            "model_interface": "deferred alpha_j(x_t) multi-output learner",
            "gate_status": "future_data_needed",
            "status_reason": "Round47 conservative global multi-alpha diagnostic effectively collapses to WGS-only and recommends stopping before learned multi-alpha.",
            "source_rounds": "round46,round47,round48",
            "allowed_online_policy": "Do not train or deploy under current 22-case diagnostic split; reopen after larger leakage-safe data.",
            "forbidden_claim": "Do not claim learned multi-alpha success, final temporal generalization, or production control readiness.",
            "table_row_label": "Learned multi-alpha update",
            "current_huayi_evidence_short": "Candidate alphas separable but all include-with-caution; global multi-alpha does not justify opening.",
            "manuscript_wording_short": "The gate prevents extra kinetic degrees of freedom from becoming compensation capacity.",
        },
        {
            "channel": "z_d",
            "reactor_process": "Slow coal-quality or feedstock latent state tied to sparse LIMS labels.",
            "model_interface": "future slow latent variable; no validated current path",
            "gate_status": "future_data_needed",
            "status_reason": "Round40 protocol lists leakage guards, but no validated z_d evidence exists yet.",
            "source_rounds": "round40,round48",
            "allowed_online_policy": "Mention only as future gate requiring no-future LIMS alignment and controls.",
            "forbidden_claim": "Do not claim weak-label coal-quality value, learned latent interpretability, or temporal generalization.",
            "table_row_label": "Slow coal-quality state z_d",
            "current_huayi_evidence_short": "Protocol only; no validated z_d diagnostic evidence.",
            "manuscript_wording_short": "Coal-quality supervision remains a future gate.",
        },
    ]
    for row in rows:
        specific_forbidden_claim = row["forbidden_claim"]
        row.update(
            _source_row(
                row["source_rounds"],
                "outputs/diagnostics/huayi_hybrid_diagnostic_evidence_bundle.md; manuscript/tables/adaptive-update-policy.md",
                "policy_traceability",
            )
        )
        row["allowed_claim"] = row["allowed_online_policy"]
        row["forbidden_claim"] = specific_forbidden_claim
    return rows


def build_reaction_family_gate_map(channel_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    specs = {
        "direct_residual": ("post-reactor", "residual patch", "outlet CO/H2/CO2 residuals", "physical reactor-route measurement or source-term mapping"),
        "alpha_wgs": ("downstream gas-phase", "WGS", "outlet CO/H2/CO2 and H2/CO", "stronger split validation before online adaptation"),
        "alpha_wgs_xt": ("downstream gas-phase", "WGS", "DCS O/C, carrier ratio, outlet dry gas", "mechanistic WGS identification remains unavailable"),
        "alpha_char_oxidation": ("inlet/burner", "char oxidation", "outlet dry gas plus O/C", "burner-zone heat release, oxygen consumption, particle conversion"),
        "alpha_volatile_oxidation": ("early reaction zone", "volatile oxidation", "outlet dry gas plus O/C", "volatile release/oxidation measurements or stronger priors"),
        "alpha_char_gasification": ("whole-reactor", "char gasification", "not_available_from_existing_artifact", "clean grouped model path plus char conversion measurements"),
        "learned_multialpha": ("whole-reactor", "multi-alpha", "multiple outlet dry-gas targets", "larger leakage-safe data and channel-specific validation"),
        "z_d": ("whole-reactor", "coal-quality latent", "sparse LIMS labels", "no-future LIMS alignment and shuffle/delay/dropout controls"),
    }
    rows = []
    status_by_channel = {row["channel"]: row["gate_status"] for row in channel_rows}
    for channel, (region, process, signal, missing) in specs.items():
        rows.append(
            {
                "channel": channel,
                "gasifier_region": region,
                "dominant_process": process,
                "measured_signal": signal,
                "missing_measurement_or_prior": missing,
                "gate_status": status_by_channel[channel],
                **_source_row("round45-round48", "outputs/diagnostics/huayi_adaptive_gate_channel_status.csv", "figure_source_data"),
            }
        )
    return rows


def build_alpha_similarity_for_fig5(repo_root: str | Path) -> list[dict[str, Any]]:
    root = Path(repo_root)
    sim = _read_csv(root, "outputs/diagnostics/huayi_alpha_candidate_similarity.csv")
    rec = _read_csv(root, "outputs/diagnostics/huayi_alpha_candidate_recommendations.csv")
    rec_lookup = {row["alpha_name"]: row["recommendation"] for _, row in rec.iterrows()}
    rows = []
    for _, row in sim.iterrows():
        value = float(row["cosine_similarity"])
        alpha_i = str(row["alpha_i"])
        alpha_j = str(row["alpha_j"])
        interpretation = "self" if alpha_i == alpha_j else str(row.get("confounding_flag", ""))
        if alpha_i != alpha_j:
            interpretation = f"{rec_lookup.get(alpha_i, 'not_available_from_existing_artifact')} vs {rec_lookup.get(alpha_j, 'not_available_from_existing_artifact')}; {interpretation}"
        rows.append(
            {
                "channel_i": alpha_i,
                "channel_j": alpha_j,
                "similarity_metric": "cosine_similarity",
                "similarity_value": value,
                "near_collinear_flag": bool(abs(value) >= 0.85),
                "interpretation": interpretation,
                **_source_row("round46", "outputs/diagnostics/huayi_alpha_candidate_similarity.csv", "sensitivity_similarity"),
            }
        )
    return rows


def _bound_hit(row: Mapping[str, Any], channel: str) -> str:
    col = f"{channel}_bound_hit"
    if bool(row.get(col, False)):
        selected = float(row[channel])
        bounds = {
            "alpha_wgs": (0.02, 2.0),
            "alpha_char_oxidation": (0.05, 1.0),
            "alpha_volatile_oxidation": (0.2, 1.2),
        }[channel]
        if abs(selected - bounds[0]) <= abs(selected - bounds[1]):
            return "lower"
        return "upper"
    return "none"


def build_multialpha_activation_for_fig5(repo_root: str | Path) -> list[dict[str, Any]]:
    root = Path(repo_root)
    summary = _read_csv(root, "outputs/diagnostics/huayi_global_multialpha_summary.csv")
    bounds = {
        "alpha_wgs": "[0.02,2.0]",
        "alpha_char_oxidation": "[0.05,1.0]",
        "alpha_volatile_oxidation": "[0.2,1.2]",
    }
    rows = []
    for _, row in summary.iterrows():
        included = set(str(row["included_alphas"]).split(","))
        for channel in ("alpha_wgs", "alpha_char_oxidation", "alpha_volatile_oxidation"):
            selected = float(row[channel])
            rows.append(
                {
                    "selected_subset": row["subset_name"],
                    "included_in_subset": channel in included,
                    "channel": channel,
                    "selected_alpha": selected,
                    "baseline_alpha": 1.0,
                    "activated_flag": bool(abs(selected - 1.0) > 0.05),
                    "bounds": bounds[channel],
                    "bound_hit": _bound_hit(row, channel),
                    "source_round": "round47",
                    "selection_split": row["selection_split"],
                    "train_objective_delta": row["train_objective_delta"],
                    "compensation_flag": row["compensation_flag"],
                    **_source_row("round47", "outputs/diagnostics/huayi_global_multialpha_summary.csv", "multialpha_activation_diagnostic"),
                }
            )
    return rows


def build_source_manifest(repo_root: str | Path, out_dir: str | Path, generated_counts: Mapping[str, int]) -> list[dict[str, Any]]:
    root = Path(repo_root)
    out = Path(out_dir)
    rows: list[dict[str, Any]] = []
    for spec in SOURCE_ARTIFACTS:
        path = root / spec["path"]
        exists = path.exists()
        row_count = 0
        column_count = 0
        if exists and path.suffix == ".csv":
            df = pd.read_csv(path)
            row_count = len(df)
            column_count = len(df.columns)
        rows.append(
            {
                "artifact_key": spec["artifact_key"],
                "artifact_role": "source",
                "artifact_path": spec["path"],
                "producing_round": spec["round"],
                "artifact_kind": spec["kind"],
                "row_count": int(row_count),
                "column_count": int(column_count),
                "exists": bool(exists),
                "evidence_role": spec["kind"],
                "allowed_claim": "Existing diagnostic source artifact for adaptive-gate figure source-data extraction.",
                "forbidden_claim": "Do not use as new modeling, final accuracy, production-control, or temporal-generalization evidence.",
                "evidence_boundary": GATE_FIGURE_BOUNDARY,
            }
        )
    for key, filename, purpose in GENERATED_FILES:
        path = out / filename
        exists = True
        if key == "manifest":
            row_count = len(SOURCE_ARTIFACTS) + len(GENERATED_FILES)
        else:
            row_count = int(generated_counts.get(key, 0))
        rows.append(
            {
                "artifact_key": key,
                "artifact_role": "generated_round49",
                "artifact_path": str(Path("outputs/diagnostics") / filename),
                "producing_round": "round49",
                "artifact_kind": purpose,
                "row_count": row_count,
                "column_count": int(len(pd.read_csv(path).columns)) if path.exists() and path.suffix == ".csv" else 0,
                "exists": bool(exists),
                "evidence_role": "figure_source_data",
                "allowed_claim": "Figure/table-ready source data extracted from existing diagnostics.",
                "forbidden_claim": "Do not infer new performance, production control, kinetic identification, or generalization.",
                "evidence_boundary": GATE_FIGURE_BOUNDARY,
            }
        )
    return rows


def _csv_with_count(rows: Sequence[Mapping[str, Any]], path: Path) -> int:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return len(df)


def gate_bundle_markdown(row_counts: Mapping[str, int], not_available_fields: Sequence[str], method_labels: Sequence[str], statuses: Sequence[str]) -> str:
    lines = [
        "# Huayi adaptive gate figure source-data bundle",
        "",
        "This round49 bundle extracts figure-ready source data from existing round41-round48 diagnostics only.",
        "It does not run new solvers, train new models, search new hyperparameters, or change channel statuses.",
        "",
        "## Generated CSV row counts",
        "",
    ]
    for key, filename, _ in GENERATED_FILES:
        if filename.endswith(".csv"):
            lines.append(f"- `{filename}`: {row_counts.get(key, 0)} rows")
    lines.extend(
        [
            "",
            "## Fig. 4 method labels",
            "",
            ", ".join(method_labels),
            "",
            "## Channel statuses",
            "",
            ", ".join(statuses),
            "",
            "## Not available from existing artifacts",
            "",
        ]
    )
    for item in not_available_fields:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Evidence Boundary",
            "",
            f"This is source-data extraction for a gate-diagnostic case study: {GATE_FIGURE_BOUNDARY}.",
            "",
            "Forbidden framing: new model performance, final predictive accuracy, production-control readiness, learned WGS kinetic identification, learned multi-alpha success, weak-label coal-quality value, final temporal generalization, or Yulin/slurry/cross-gasifier generalization.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_adaptive_gate_figure_source_bundle(repo_root: str | Path, out_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(repo_root)
    out = Path(out_dir) if out_dir is not None else root / "outputs" / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)

    missing_sources = [spec["path"] for spec in SOURCE_ARTIFACTS if not (root / spec["path"]).exists()]
    if missing_sources:
        raise FileNotFoundError(f"Missing expected source artifacts: {missing_sources}")

    method_ladder = build_fig4_method_ladder()
    metric_matrix = build_fig4_metric_matrix(root)
    channel_status = build_channel_status()
    reaction_map = build_reaction_family_gate_map(channel_status)
    alpha_similarity = build_alpha_similarity_for_fig5(root)
    multialpha_activation = build_multialpha_activation_for_fig5(root)

    paths = {
        "fig4_metric_matrix": out / "huayi_adaptive_gate_fig4_metric_matrix.csv",
        "fig4_method_ladder": out / "huayi_adaptive_gate_fig4_method_ladder.csv",
        "channel_status": out / "huayi_adaptive_gate_channel_status.csv",
        "reaction_family_gate_map": out / "huayi_reaction_family_gate_map.csv",
        "alpha_similarity": out / "huayi_alpha_similarity_for_fig5.csv",
        "multialpha_activation": out / "huayi_multialpha_activation_for_fig5.csv",
        "manifest": out / "huayi_gate_figure_source_manifest.csv",
        "markdown": out / "huayi_gate_figure_source_bundle.md",
    }

    row_counts: dict[str, int] = {}
    row_counts["fig4_metric_matrix"] = _csv_with_count(metric_matrix, paths["fig4_metric_matrix"])
    row_counts["fig4_method_ladder"] = _csv_with_count(method_ladder, paths["fig4_method_ladder"])
    row_counts["channel_status"] = _csv_with_count(channel_status, paths["channel_status"])
    row_counts["reaction_family_gate_map"] = _csv_with_count(reaction_map, paths["reaction_family_gate_map"])
    row_counts["alpha_similarity"] = _csv_with_count(alpha_similarity, paths["alpha_similarity"])
    row_counts["multialpha_activation"] = _csv_with_count(multialpha_activation, paths["multialpha_activation"])

    manifest = build_source_manifest(root, out, row_counts)
    row_counts["manifest"] = _csv_with_count(manifest, paths["manifest"])

    not_available_fields = [
        "huayi_reaction_family_gate_map.measured_signal for alpha_char_gasification: not_available_from_existing_artifact because no clean grouped op_conds path exists.",
        "huayi_adaptive_gate_channel_status z_d evidence: not_available_from_existing_artifact beyond round40 protocol; no validated z_d diagnostic exists.",
        "production-control readiness, learned kinetic identification, and final temporal generalization: not_available_from_existing_artifact by design.",
    ]
    method_labels = sorted(pd.DataFrame(metric_matrix)["method_label"].unique())
    statuses = sorted(pd.DataFrame(channel_status)["gate_status"].unique())
    md_text = gate_bundle_markdown(row_counts, not_available_fields, method_labels, statuses)
    paths["markdown"].write_text(md_text, encoding="utf-8")

    return {
        **{key: str(path) for key, path in paths.items()},
        "row_counts": row_counts,
        "method_labels": method_labels,
        "channel_statuses": statuses,
        "not_available_fields": not_available_fields,
        "evidence_boundary": GATE_FIGURE_BOUNDARY,
    }


__all__ = [
    "ALLOWED_STATUSES",
    "GATE_FIGURE_BOUNDARY",
    "NOT_AVAILABLE",
    "REQUIRED_FIG4_METHOD_LABELS",
    "build_adaptive_gate_figure_source_bundle",
    "build_channel_status",
    "build_fig4_metric_matrix",
    "build_fig4_method_ladder",
]
