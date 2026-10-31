"""
Paper-facing hybrid diagnostic evidence bundle.

This module collates existing Huayi diagnostic artifacts.  It does not rerun
the model, fit new parameters, or add modeling capacity.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

BUNDLE_EVIDENCE_BOUNDARY = (
    "hybrid diagnostic evidence bundle only; not new modeling, learned "
    "multi-alpha training, weak-label coal-quality modeling, production "
    "control, or final temporal-generalization evidence"
)

MANIFEST_COLUMNS: tuple[str, ...] = (
    "artifact_key",
    "source_artifact_path",
    "producing_round",
    "artifact_kind",
    "row_count",
    "column_count",
    "exists",
    "evidence_boundary",
)

METRIC_MATRIX_COLUMNS: tuple[str, ...] = (
    "method_label",
    "source_round",
    "source_artifact",
    "split",
    "selection_role",
    "target",
    "N_cells",
    "n",
    "rmse",
    "mae",
    "bias",
    "rmse_delta_vs_fixed",
    "rmse_delta_vs_global_wgs",
    "fallback_count_sum",
    "poor_convergence_count_sum",
    "cell0_cost_min",
    "cell0_cost_max",
    "claim_safe_label",
    "evidence_boundary",
)

CLAIM_TABLE_COLUMNS: tuple[str, ...] = (
    "result_key",
    "claim_safe_label",
    "manuscript_safe_wording",
    "forbidden_wording",
    "supporting_artifacts",
    "evidence_boundary",
)

RECOMMENDATION_COLUMNS: tuple[str, ...] = (
    "recommendation_key",
    "current_safest_modeling_path",
    "stop_condition",
    "missing_evidence_before_z_d_or_multi_alpha",
    "supporting_artifacts",
    "evidence_boundary",
)

SOURCE_ARTIFACTS: tuple[dict[str, str], ...] = (
    {"artifact_key": "fixed_physics_split_metrics", "path": "outputs/diagnostics/huayi_fixed_physics_split_metrics.csv", "round": "round41", "kind": "metric"},
    {"artifact_key": "fixed_physics_split_predictions", "path": "outputs/diagnostics/huayi_fixed_physics_split_predictions.csv", "round": "round41", "kind": "prediction"},
    {"artifact_key": "global_wgs_metrics", "path": "outputs/diagnostics/huayi_global_wgs_alpha_metrics.csv", "round": "round41", "kind": "metric"},
    {"artifact_key": "global_wgs_summary", "path": "outputs/diagnostics/huayi_global_wgs_alpha_baseline.csv", "round": "round41", "kind": "selection"},
    {"artifact_key": "direct_residual_metrics", "path": "outputs/diagnostics/huayi_direct_residual_baseline_metrics.csv", "round": "round42", "kind": "metric"},
    {"artifact_key": "direct_residual_models", "path": "outputs/diagnostics/huayi_direct_residual_baseline_models.csv", "round": "round42", "kind": "audit"},
    {"artifact_key": "global_char_metrics", "path": "outputs/diagnostics/huayi_global_char_oxidation_alpha_metrics.csv", "round": "round43", "kind": "metric"},
    {"artifact_key": "global_char_summary", "path": "outputs/diagnostics/huayi_global_char_oxidation_alpha_baseline.csv", "round": "round43", "kind": "selection"},
    {"artifact_key": "alpha_wgs_xt_metrics", "path": "outputs/diagnostics/huayi_alpha_wgs_xt_metrics.csv", "round": "round44", "kind": "metric"},
    {"artifact_key": "alpha_wgs_xt_summary", "path": "outputs/diagnostics/huayi_alpha_wgs_xt_baseline.csv", "round": "round44", "kind": "selection"},
    {"artifact_key": "wgs_robustness_grid", "path": "outputs/diagnostics/huayi_alpha_wgs_xt_robustness_grid.csv", "round": "round45", "kind": "audit"},
    {"artifact_key": "wgs_null_control", "path": "outputs/diagnostics/huayi_alpha_wgs_xt_null_control.csv", "round": "round45", "kind": "audit"},
    {"artifact_key": "wgs_sensitivity_proxy", "path": "outputs/diagnostics/huayi_alpha_wgs_sensitivity_proxy.csv", "round": "round45", "kind": "sensitivity"},
    {"artifact_key": "candidate_sensitivities", "path": "outputs/diagnostics/huayi_alpha_candidate_sensitivities.csv", "round": "round46", "kind": "sensitivity"},
    {"artifact_key": "candidate_similarity", "path": "outputs/diagnostics/huayi_alpha_candidate_similarity.csv", "round": "round46", "kind": "audit"},
    {"artifact_key": "candidate_recommendations", "path": "outputs/diagnostics/huayi_alpha_candidate_recommendations.csv", "round": "round46", "kind": "recommendation"},
    {"artifact_key": "global_multialpha_metrics", "path": "outputs/diagnostics/huayi_global_multialpha_metrics.csv", "round": "round47", "kind": "metric"},
    {"artifact_key": "global_multialpha_summary", "path": "outputs/diagnostics/huayi_global_multialpha_summary.csv", "round": "round47", "kind": "selection"},
)

REQUIRED_METHOD_LABELS: tuple[str, ...] = (
    "fixed_physics",
    "global_wgs_alpha",
    "direct_residual_constant_train_mean",
    "direct_residual_linear_operating",
    "global_char_oxidation_alpha",
    "alpha_wgs_xt_scaffold",
    "global_multialpha:alpha_wgs",
    "global_multialpha:alpha_wgs+alpha_volatile_oxidation",
)


def _read_csv(repo_root: Path, rel_path: str) -> pd.DataFrame:
    path = repo_root / rel_path
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def build_artifact_manifest(repo_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in SOURCE_ARTIFACTS:
        path = repo_root / spec["path"]
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
                "source_artifact_path": spec["path"],
                "producing_round": spec["round"],
                "artifact_kind": spec["kind"],
                "row_count": int(row_count),
                "column_count": int(column_count),
                "exists": bool(exists),
                "evidence_boundary": BUNDLE_EVIDENCE_BOUNDARY,
            }
        )
    return [{column: row.get(column) for column in MANIFEST_COLUMNS} for row in rows]


def _selection_role(split: str) -> str:
    if split == "train":
        return "train_selected_or_fit"
    if split in {"validation", "test"}:
        return "held_out_transform_only"
    return "post_hoc_diagnostic"


def _metric_row(
    *,
    method_label: str,
    source_round: str,
    source_artifact: str,
    split: str,
    target: str,
    n_cells: int,
    n: int,
    rmse: float,
    mae: float,
    bias: float,
    rmse_delta_vs_fixed: float,
    rmse_delta_vs_global_wgs: float,
    fallback_count_sum: float,
    poor_convergence_count_sum: float,
    cell0_cost_min: float,
    cell0_cost_max: float,
    claim_safe_label: str,
) -> dict[str, Any]:
    return {
        "method_label": method_label,
        "source_round": source_round,
        "source_artifact": source_artifact,
        "split": split,
        "selection_role": _selection_role(split),
        "target": target,
        "N_cells": int(n_cells),
        "n": int(n),
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "rmse_delta_vs_fixed": rmse_delta_vs_fixed,
        "rmse_delta_vs_global_wgs": rmse_delta_vs_global_wgs,
        "fallback_count_sum": fallback_count_sum,
        "poor_convergence_count_sum": poor_convergence_count_sum,
        "cell0_cost_min": cell0_cost_min,
        "cell0_cost_max": cell0_cost_max,
        "claim_safe_label": claim_safe_label,
        "evidence_boundary": BUNDLE_EVIDENCE_BOUNDARY,
    }


def _lookup_fixed_and_wgs(repo_root: Path) -> tuple[dict[tuple[str, str, int], float], dict[tuple[str, str, int], float]]:
    fixed = _read_csv(repo_root, "outputs/diagnostics/huayi_fixed_physics_split_metrics.csv")
    wgs = _read_csv(repo_root, "outputs/diagnostics/huayi_global_wgs_alpha_metrics.csv")
    fixed_lookup = {(row["split"], row["target"], int(row["N_cells"])): float(row["rmse"]) for _, row in fixed.iterrows()}
    wgs_lookup = {(row["split"], row["target"], int(row["N_cells"])): float(row["rmse"]) for _, row in wgs.iterrows()}
    return fixed_lookup, wgs_lookup


def build_metric_matrix(repo_root: Path) -> list[dict[str, Any]]:
    fixed_lookup, wgs_lookup = _lookup_fixed_and_wgs(repo_root)
    rows: list[dict[str, Any]] = []

    fixed = _read_csv(repo_root, "outputs/diagnostics/huayi_fixed_physics_split_metrics.csv")
    for _, row in fixed.iterrows():
        key = (row["split"], row["target"], int(row["N_cells"]))
        rows.append(
            _metric_row(
                method_label="fixed_physics",
                source_round="round41",
                source_artifact="outputs/diagnostics/huayi_fixed_physics_split_metrics.csv",
                split=row["split"],
                target=row["target"],
                n_cells=int(row["N_cells"]),
                n=int(row["n"]),
                rmse=float(row["rmse"]),
                mae=float(row["mae"]),
                bias=float(row["bias"]),
                rmse_delta_vs_fixed=0.0,
                rmse_delta_vs_global_wgs=float(row["rmse"]) - wgs_lookup.get(key, float("nan")),
                fallback_count_sum=float(row["fallback_count_sum"]),
                poor_convergence_count_sum=float(row["poor_convergence_count_sum"]),
                cell0_cost_min=float(row["cell0_cost_min"]),
                cell0_cost_max=float(row["cell0_cost_max"]),
                claim_safe_label="fixed_physics_reference",
            )
        )

    wgs = _read_csv(repo_root, "outputs/diagnostics/huayi_global_wgs_alpha_metrics.csv")
    for _, row in wgs.iterrows():
        rows.append(
            _metric_row(
                method_label="global_wgs_alpha",
                source_round="round41",
                source_artifact="outputs/diagnostics/huayi_global_wgs_alpha_metrics.csv",
                split=row["split"],
                target=row["target"],
                n_cells=int(row["N_cells"]),
                n=int(row["n"]),
                rmse=float(row["rmse"]),
                mae=float(row["mae"]),
                bias=float(row["bias"]),
                rmse_delta_vs_fixed=float(row["rmse_delta"]),
                rmse_delta_vs_global_wgs=0.0,
                fallback_count_sum=float(row["fallback_count_sum"]),
                poor_convergence_count_sum=float(row["poor_convergence_count_sum"]),
                cell0_cost_min=float(row["cell0_cost_min"]),
                cell0_cost_max=float(row["cell0_cost_max"]),
                claim_safe_label="diagnostic_only",
            )
        )

    direct = _read_csv(repo_root, "outputs/diagnostics/huayi_direct_residual_baseline_metrics.csv")
    for _, row in direct.iterrows():
        key = (row["split"], row["target"], int(row["N_cells"]))
        method = "direct_residual_constant_train_mean" if row["baseline_variant"] == "residual_constant_train_mean" else "direct_residual_linear_operating"
        rows.append(
            _metric_row(
                method_label=method,
                source_round="round42",
                source_artifact="outputs/diagnostics/huayi_direct_residual_baseline_metrics.csv",
                split=row["split"],
                target=row["target"],
                n_cells=int(row["N_cells"]),
                n=int(row["n"]),
                rmse=float(row["corrected_rmse"]),
                mae=float(row["corrected_mae"]),
                bias=float(row["corrected_bias"]),
                rmse_delta_vs_fixed=float(row["rmse_delta"]),
                rmse_delta_vs_global_wgs=float(row["corrected_rmse"]) - wgs_lookup.get(key, float("nan")),
                fallback_count_sum=float(row["fallback_count_sum"]) if pd.notna(row["fallback_count_sum"]) else float("nan"),
                poor_convergence_count_sum=float(row["poor_convergence_count_sum"]) if pd.notna(row["poor_convergence_count_sum"]) else float("nan"),
                cell0_cost_min=float(row["cell0_cost_min"]) if pd.notna(row["cell0_cost_min"]) else float("nan"),
                cell0_cost_max=float(row["cell0_cost_max"]) if pd.notna(row["cell0_cost_max"]) else float("nan"),
                claim_safe_label="overfit_warning",
            )
        )

    char = _read_csv(repo_root, "outputs/diagnostics/huayi_global_char_oxidation_alpha_metrics.csv")
    for _, row in char.iterrows():
        key = (row["split"], row["target"], int(row["N_cells"]))
        rows.append(
            _metric_row(
                method_label="global_char_oxidation_alpha",
                source_round="round43",
                source_artifact="outputs/diagnostics/huayi_global_char_oxidation_alpha_metrics.csv",
                split=row["split"],
                target=row["target"],
                n_cells=int(row["N_cells"]),
                n=int(row["n"]),
                rmse=float(row["rmse"]),
                mae=float(row["mae"]),
                bias=float(row["bias"]),
                rmse_delta_vs_fixed=float(row["rmse_delta"]),
                rmse_delta_vs_global_wgs=float(row["rmse"]) - wgs_lookup.get(key, float("nan")),
                fallback_count_sum=float(row["fallback_count_sum"]),
                poor_convergence_count_sum=float(row["poor_convergence_count_sum"]),
                cell0_cost_min=float(row["cell0_cost_min"]),
                cell0_cost_max=float(row["cell0_cost_max"]),
                claim_safe_label="diagnostic_only",
            )
        )

    xt = _read_csv(repo_root, "outputs/diagnostics/huayi_alpha_wgs_xt_metrics.csv")
    for _, row in xt.iterrows():
        rows.append(
            _metric_row(
                method_label="alpha_wgs_xt_scaffold",
                source_round="round44",
                source_artifact="outputs/diagnostics/huayi_alpha_wgs_xt_metrics.csv",
                split=row["split"],
                target=row["target"],
                n_cells=int(row["N_cells"]),
                n=int(row["n"]),
                rmse=float(row["alpha_rmse"]),
                mae=float(row["alpha_mae"]),
                bias=float(row["alpha_bias"]),
                rmse_delta_vs_fixed=float(row["rmse_delta_vs_fixed"]),
                rmse_delta_vs_global_wgs=float(row["rmse_delta_vs_global_wgs"]),
                fallback_count_sum=float(row["fallback_count_sum"]),
                poor_convergence_count_sum=float(row["poor_convergence_count_sum"]),
                cell0_cost_min=float(row["cell0_cost_min"]),
                cell0_cost_max=float(row["cell0_cost_max"]),
                claim_safe_label="supports_scaffold",
            )
        )

    multi = _read_csv(repo_root, "outputs/diagnostics/huayi_global_multialpha_metrics.csv")
    for _, row in multi.iterrows():
        rows.append(
            _metric_row(
                method_label=f"global_multialpha:{row['subset_name']}",
                source_round="round47",
                source_artifact="outputs/diagnostics/huayi_global_multialpha_metrics.csv",
                split=row["split"],
                target=row["target"],
                n_cells=int(row["N_cells"]),
                n=int(row["n"]),
                rmse=float(row["multialpha_rmse"]),
                mae=float(row["multialpha_mae"]),
                bias=float(row["multialpha_bias"]),
                rmse_delta_vs_fixed=float(row["rmse_delta_vs_fixed"]),
                rmse_delta_vs_global_wgs=float(row["rmse_delta_vs_global_wgs"]),
                fallback_count_sum=float(row["fallback_count_sum"]),
                poor_convergence_count_sum=float(row["poor_convergence_count_sum"]),
                cell0_cost_min=float(row["cell0_cost_min"]),
                cell0_cost_max=float(row["cell0_cost_max"]),
                claim_safe_label="negative_multi_alpha",
            )
        )
    return [{column: row.get(column) for column in METRIC_MATRIX_COLUMNS} for row in rows]


def build_claim_table() -> list[dict[str, Any]]:
    rows = [
        {
            "result_key": "verification_first_split_baseline",
            "claim_safe_label": "diagnostic_only",
            "manuscript_safe_wording": "The fixed-physics model provides a split-aware diagnostic reference with solver-quality fields.",
            "forbidden_wording": "Do not call this calibrated accuracy, industrial generalization, or final temporal validation.",
            "supporting_artifacts": "huayi_fixed_physics_split_metrics.csv; huayi_fixed_physics_split_predictions.csv",
            "evidence_boundary": BUNDLE_EVIDENCE_BOUNDARY,
        },
        {
            "result_key": "wgs_alpha_xt_scaffold",
            "claim_safe_label": "supports_scaffold",
            "manuscript_safe_wording": "A minimal bounded WGS-only scaffold can be run leakage-safely and compared against global scalars.",
            "forbidden_wording": "Do not claim full hybrid accuracy, learned multiplier interpretability, or temporal generalization.",
            "supporting_artifacts": "huayi_alpha_wgs_xt_metrics.csv; huayi_alpha_wgs_xt_robustness_grid.csv",
            "evidence_boundary": BUNDLE_EVIDENCE_BOUNDARY,
        },
        {
            "result_key": "direct_residual_warning",
            "claim_safe_label": "overfit_warning",
            "manuscript_safe_wording": "Direct residual patching is a black-box comparator that improves train rows but shows mixed held-out behavior.",
            "forbidden_wording": "Do not present direct residual correction as physics-guided hybrid learning or mechanistic evidence.",
            "supporting_artifacts": "huayi_direct_residual_baseline_metrics.csv",
            "evidence_boundary": BUNDLE_EVIDENCE_BOUNDARY,
        },
        {
            "result_key": "multi_alpha_stop_signal",
            "claim_safe_label": "negative_multi_alpha",
            "manuscript_safe_wording": "The conservative global multi-alpha diagnostic collapses to WGS-only and supports stopping before learned multi-alpha on 22 representatives.",
            "forbidden_wording": "Do not claim failed final hybrid training, causal proof, or that non-WGS alphas are universally invalid.",
            "supporting_artifacts": "huayi_global_multialpha_summary.csv; huayi_global_multialpha_metrics.csv",
            "evidence_boundary": BUNDLE_EVIDENCE_BOUNDARY,
        },
        {
            "result_key": "unsupported_accuracy_claims",
            "claim_safe_label": "unsupported_for_accuracy",
            "manuscript_safe_wording": "These artifacts support diagnostic scaffolding and protocol decisions, not accuracy benchmarking.",
            "forbidden_wording": "Forbidden: calibrated accuracy, production control, final temporal generalization, weak-label coal-quality proof, Yulin/slurry generalization.",
            "supporting_artifacts": "all bundle artifacts",
            "evidence_boundary": BUNDLE_EVIDENCE_BOUNDARY,
        },
    ]
    return [{column: row.get(column) for column in CLAIM_TABLE_COLUMNS} for row in rows]


def build_recommendations() -> list[dict[str, Any]]:
    rows = [
        {
            "recommendation_key": "current_path",
            "current_safest_modeling_path": "stay WGS-only or stop before learned multi-alpha; expand data/protocol before z_d",
            "stop_condition": "Do not add learned multi-alpha if global bounded multi-alpha collapses to WGS-only or improves held-out metrics only by target tradeoffs.",
            "missing_evidence_before_z_d_or_multi_alpha": "larger steady-window training set, leakage-safe coal-label alignment, repeated leave-future-out validation, stronger regularization ablations",
            "supporting_artifacts": "round45 robustness; round46 identifiability; round47 multialpha diagnostic",
            "evidence_boundary": BUNDLE_EVIDENCE_BOUNDARY,
        },
        {
            "recommendation_key": "figure_claim_boundary",
            "current_safest_modeling_path": "present hybrid sequence as diagnostic scaffold and stopping rule, not performance benchmark",
            "stop_condition": "Avoid Results language implying calibrated hybrid accuracy or deployment readiness.",
            "missing_evidence_before_z_d_or_multi_alpha": "held-out improvement versus fixed physics across more cases and targets; model-selection protocol independent of test split",
            "supporting_artifacts": "metric matrix; claim table",
            "evidence_boundary": BUNDLE_EVIDENCE_BOUNDARY,
        },
    ]
    return [{column: row.get(column) for column in RECOMMENDATION_COLUMNS} for row in rows]


def evidence_bundle_markdown(
    manifest_rows: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
    claim_rows: Sequence[Mapping[str, Any]],
    recommendation_rows: Sequence[Mapping[str, Any]],
) -> str:
    metrics = pd.DataFrame(metric_rows)
    lines = [
        "# Huayi hybrid diagnostic evidence bundle",
        "",
        "This bundle collates existing round41-round47 diagnostics. It does not add new modeling.",
        "",
        "## Included methods",
        "",
    ]
    for label in sorted(metrics["method_label"].unique()):
        lines.append(f"- {label}")
    lines.extend(["", "## Held-out metric headline", "", "| method | held-out rows | improved vs fixed | improved vs global WGS | mean delta vs fixed | mean delta vs WGS |", "|---|---:|---:|---:|---:|---:|"])
    heldout = metrics[metrics["split"].isin(["validation", "test"])]
    for label, group in heldout.groupby("method_label"):
        lines.append(
            "| {label} | {n} | {fixed}/{n} | {wgs}/{n} | {df:.6g} | {dw:.6g} |".format(
                label=label,
                n=len(group),
                fixed=int((group["rmse_delta_vs_fixed"] < 0.0).sum()),
                wgs=int((group["rmse_delta_vs_global_wgs"] < 0.0).sum()),
                df=float(group["rmse_delta_vs_fixed"].mean()),
                dw=float(group["rmse_delta_vs_global_wgs"].mean()),
            )
        )
    lines.extend(["", "## Claim-safe labels", "", "| result | label | safe wording |", "|---|---|---|"])
    for row in claim_rows:
        lines.append(f"| {row['result_key']} | {row['claim_safe_label']} | {row['manuscript_safe_wording']} |")
    lines.extend(["", "## Recommendation", ""])
    for row in recommendation_rows:
        lines.append(f"- {row['recommendation_key']}: {row['current_safest_modeling_path']}")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            f"Evidence boundary: {BUNDLE_EVIDENCE_BOUNDARY}.",
            "",
            "Forbidden framing: calibrated accuracy, learned multiplier interpretability proof, production control, final temporal generalization, weak-label coal-quality proof, or Yulin/slurry generalization.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_hybrid_diagnostic_evidence_bundle(repo_root: str | Path, out_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(repo_root)
    out = Path(out_dir) if out_dir is not None else root / "outputs" / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)

    manifest = build_artifact_manifest(root)
    missing = [row["source_artifact_path"] for row in manifest if not row["exists"]]
    if missing:
        raise FileNotFoundError(f"Missing expected source artifacts: {missing}")
    metric_matrix = build_metric_matrix(root)
    claim_table = build_claim_table()
    recommendations = build_recommendations()

    manifest_path = out / "huayi_hybrid_diagnostic_evidence_manifest.csv"
    metric_path = out / "huayi_hybrid_diagnostic_metric_matrix.csv"
    claim_path = out / "huayi_hybrid_diagnostic_claim_table.csv"
    recommendation_path = out / "huayi_hybrid_diagnostic_recommendations.csv"
    md_path = out / "huayi_hybrid_diagnostic_evidence_bundle.md"
    pd.DataFrame(manifest, columns=list(MANIFEST_COLUMNS)).to_csv(manifest_path, index=False)
    pd.DataFrame(metric_matrix, columns=list(METRIC_MATRIX_COLUMNS)).to_csv(metric_path, index=False)
    pd.DataFrame(claim_table, columns=list(CLAIM_TABLE_COLUMNS)).to_csv(claim_path, index=False)
    pd.DataFrame(recommendations, columns=list(RECOMMENDATION_COLUMNS)).to_csv(recommendation_path, index=False)
    md_path.write_text(evidence_bundle_markdown(manifest, metric_matrix, claim_table, recommendations), encoding="utf-8")
    return {
        "manifest": str(manifest_path),
        "metric_matrix": str(metric_path),
        "claim_table": str(claim_path),
        "recommendations": str(recommendation_path),
        "markdown": str(md_path),
        "method_labels": sorted(pd.DataFrame(metric_matrix)["method_label"].unique()),
        "evidence_boundary": BUNDLE_EVIDENCE_BOUNDARY,
    }


__all__ = [
    "BUNDLE_EVIDENCE_BOUNDARY",
    "CLAIM_TABLE_COLUMNS",
    "MANIFEST_COLUMNS",
    "METRIC_MATRIX_COLUMNS",
    "RECOMMENDATION_COLUMNS",
    "REQUIRED_METHOD_LABELS",
    "build_hybrid_diagnostic_evidence_bundle",
]
