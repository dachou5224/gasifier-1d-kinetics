"""
Influence-tracing diagnostics built from existing fixed-physics evidence.

The bundle links finite-perturbation sensitivity, fixed-branch FD-vs-IFT/JAX
parity availability, and cell-0 residual localization. It is reporting-only:
no training, calibration, causal inference, or production online control is
performed here.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

EVIDENCE_BOUNDARY_NOTE = (
    "fixed-physics influence-tracing diagnostic only; candidate online-correction "
    "axis selection support, not trained hybrid, calibrated accuracy, causal proof, "
    "production online deployment, full-reactor 22-case IFT parity, or Yulin/slurry generalization"
)

INFLUENCE_TRACING_COLUMNS: tuple[str, ...] = (
    "phase",
    "dataset_id",
    "case_id",
    "source_candidate_id",
    "independence_group",
    "N_cells",
    "axis",
    "axis_key",
    "target",
    "perturbation_fraction",
    "base_parameter_value",
    "base_value",
    "minus_value",
    "plus_value",
    "raw_delta_centered",
    "raw_sensitivity_per_unit",
    "normalized_sensitivity",
    "abs_normalized_sensitivity",
    "case_target_axis_rank",
    "aggregate_axis_rank",
    "aggregate_mean_abs_normalized_sensitivity",
    "aggregate_median_abs_normalized_sensitivity",
    "aggregate_max_abs_normalized_sensitivity",
    "sensitivity_status",
    "gradient_support_status",
    "gradient_fixture_output",
    "gradient_parity_status",
    "gradient_parity_max_abs_error",
    "gradient_parity_max_rel_error",
    "cell0_cost",
    "cell0_max_abs_residual",
    "fallback_count",
    "poor_convergence_count",
    "poor_convergence_cell_index",
    "cell0_localization_status",
    "evidence_boundary_note",
)

AXIS_SUMMARY_COLUMNS: tuple[str, ...] = (
    "aggregate_axis_rank",
    "axis",
    "axis_key",
    "row_count",
    "ok_row_count",
    "mean_abs_normalized_sensitivity",
    "median_abs_normalized_sensitivity",
    "max_abs_normalized_sensitivity",
    "gradient_support_status",
    "gradient_parity_ok_rows",
    "gradient_parity_max_abs_error",
    "gradient_parity_max_rel_error",
    "dominant_cell0_status",
    "evidence_boundary_note",
)

TARGET_TO_GRADIENT_OUTPUT: Mapping[str, str] = {
    "T_out_K": "final_T_K",
    "CO_volpct_dry": "final_CO_mol_s",
    "H2_volpct_dry": "final_H2_mol_s",
    "CO2_volpct_dry": "final_CO2_mol_s",
    "dry_syngas_flow_kmol_s": "final_dry_flow_mol_s",
}


def _finite_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _index_by_case_n(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], Mapping[str, Any]]:
    out: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        out[(str(row["case_id"]), int(row["N_cells"]))] = row
    return out


def _phase_by_case(contract_rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row["case_id"]): row for row in contract_rows}


def _ranking_by_axis(ranking_rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row["parameter_name"]): row for row in ranking_rows}


def _parity_summary_by_axis_output(
    parity_rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in parity_rows:
        grouped.setdefault((str(row["parameter"]), str(row["output"])), []).append(row)

    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key, group in grouped.items():
        ok = [row for row in group if row.get("status") == "ok"]
        if ok:
            out[key] = {
                "gradient_support_status": "supported_fixed_branch_not_tested_full_model",
                "gradient_parity_status": "ok",
                "max_abs_error": max(_finite_or_nan(row.get("abs_error")) for row in ok),
                "max_rel_error": max(_finite_or_nan(row.get("rel_error")) for row in ok),
                "ok_rows": len(ok),
            }
        else:
            statuses = sorted({str(row.get("status")) for row in group})
            out[key] = {
                "gradient_support_status": "not_supported_in_fixed_branch_fixture",
                "gradient_parity_status": ";".join(statuses),
                "max_abs_error": math.nan,
                "max_rel_error": math.nan,
                "ok_rows": 0,
            }
    return out


def _cell0_status(baseline_row: Mapping[str, Any] | None) -> str:
    if baseline_row is None:
        return "baseline_not_available"
    poor = int(_finite_or_nan(baseline_row.get("poor_convergence_count")) or 0)
    poor_cell = str(baseline_row.get("poor_convergence_cell_index", ""))
    fallback = int(_finite_or_nan(baseline_row.get("fallback_count")) or 0)
    if fallback:
        return "fallback_present"
    if poor > 0 and poor_cell == "0":
        return "persistent_cell0_residual_quality_flag"
    if poor > 0:
        return "non_cell0_residual_quality_flag"
    return "no_poor_convergence_flag"


def build_influence_tracing_rows(
    sensitivity_rows: Sequence[Mapping[str, Any]],
    ranking_rows: Sequence[Mapping[str, Any]],
    parity_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    contract_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Combine sensitivity, parity, and cell0 evidence into a long table."""
    baseline_by_case_n = _index_by_case_n(baseline_rows)
    meta_by_case = _phase_by_case(contract_rows)
    rank_by_axis = _ranking_by_axis(ranking_rows)
    parity_by_axis_output = _parity_summary_by_axis_output(parity_rows)

    pre_rows: list[dict[str, Any]] = []
    for row in sensitivity_rows:
        case_id = str(row["case_id"])
        n_cells = int(row["N_cells"])
        axis = str(row["parameter_name"])
        target = str(row["target_name"])
        gradient_output = TARGET_TO_GRADIENT_OUTPUT.get(target, "")
        parity = parity_by_axis_output.get((axis, gradient_output))
        if parity is None:
            parity = {
                "gradient_support_status": "not_tested_fixed_branch_output",
                "gradient_parity_status": "not_tested",
                "max_abs_error": math.nan,
                "max_rel_error": math.nan,
                "ok_rows": 0,
            }
        baseline = baseline_by_case_n.get((case_id, n_cells))
        meta = meta_by_case.get(case_id, {})
        ranking = rank_by_axis.get(axis, {})
        pre_rows.append(
            {
                "phase": meta.get("phase"),
                "dataset_id": meta.get("dataset_id"),
                "case_id": case_id,
                "source_candidate_id": row.get("source_candidate_id"),
                "independence_group": row.get("independence_group"),
                "N_cells": n_cells,
                "axis": axis,
                "axis_key": row.get("parameter_key"),
                "target": target,
                "perturbation_fraction": row.get("perturbation_fraction"),
                "base_parameter_value": row.get("base_parameter_value"),
                "base_value": row.get("base_value"),
                "minus_value": row.get("minus_value"),
                "plus_value": row.get("plus_value"),
                "raw_delta_centered": row.get("raw_delta_centered"),
                "raw_sensitivity_per_unit": row.get("raw_sensitivity_per_unit"),
                "normalized_sensitivity": row.get("normalized_sensitivity"),
                "abs_normalized_sensitivity": row.get("abs_normalized_sensitivity"),
                "case_target_axis_rank": 0,
                "aggregate_axis_rank": ranking.get("rank"),
                "aggregate_mean_abs_normalized_sensitivity": ranking.get("mean_abs_normalized_sensitivity"),
                "aggregate_median_abs_normalized_sensitivity": ranking.get("median_abs_normalized_sensitivity"),
                "aggregate_max_abs_normalized_sensitivity": ranking.get("max_abs_normalized_sensitivity"),
                "sensitivity_status": row.get("status"),
                "gradient_support_status": parity["gradient_support_status"],
                "gradient_fixture_output": gradient_output,
                "gradient_parity_status": parity["gradient_parity_status"],
                "gradient_parity_max_abs_error": parity["max_abs_error"],
                "gradient_parity_max_rel_error": parity["max_rel_error"],
                "cell0_cost": baseline.get("cell0_cost") if baseline else math.nan,
                "cell0_max_abs_residual": baseline.get("cell0_max_abs_residual") if baseline else math.nan,
                "fallback_count": baseline.get("fallback_count") if baseline else math.nan,
                "poor_convergence_count": baseline.get("poor_convergence_count") if baseline else math.nan,
                "poor_convergence_cell_index": baseline.get("poor_convergence_cell_index") if baseline else "",
                "cell0_localization_status": _cell0_status(baseline),
                "evidence_boundary_note": EVIDENCE_BOUNDARY_NOTE,
            }
        )

    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in pre_rows:
        grouped.setdefault((str(row["case_id"]), int(row["N_cells"]), str(row["target"])), []).append(row)
    for group in grouped.values():
        group.sort(
            key=lambda item: (
                not math.isfinite(_finite_or_nan(item["abs_normalized_sensitivity"])),
                -_finite_or_nan(item["abs_normalized_sensitivity"]),
                str(item["axis"]),
            )
        )
        for rank, item in enumerate(group, start=1):
            item["case_target_axis_rank"] = rank

    pre_rows.sort(
        key=lambda item: (
            str(item["phase"]),
            str(item["case_id"]),
            int(item["N_cells"]),
            str(item["target"]),
            int(item["case_target_axis_rank"]),
        )
    )
    return [{column: row.get(column) for column in INFLUENCE_TRACING_COLUMNS} for row in pre_rows]


def build_axis_summary_rows(
    influence_rows: Sequence[Mapping[str, Any]],
    parity_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in influence_rows:
        grouped.setdefault(str(row["axis"]), []).append(row)
    parity_by_axis = {}
    for row in parity_rows:
        axis = str(row["parameter"])
        parity_by_axis.setdefault(axis, []).append(row)

    out: list[dict[str, Any]] = []
    for axis, group in grouped.items():
        ok_values = [
            _finite_or_nan(row["abs_normalized_sensitivity"])
            for row in group
            if row.get("sensitivity_status") == "ok"
            and math.isfinite(_finite_or_nan(row.get("abs_normalized_sensitivity")))
        ]
        aggregate_rank = int(_finite_or_nan(group[0].get("aggregate_axis_rank")))
        parity_ok = [row for row in parity_by_axis.get(axis, []) if row.get("status") == "ok"]
        out.append(
            {
                "aggregate_axis_rank": aggregate_rank,
                "axis": axis,
                "axis_key": group[0].get("axis_key"),
                "row_count": len(group),
                "ok_row_count": len(ok_values),
                "mean_abs_normalized_sensitivity": float(np.mean(ok_values)) if ok_values else math.nan,
                "median_abs_normalized_sensitivity": float(np.median(ok_values)) if ok_values else math.nan,
                "max_abs_normalized_sensitivity": float(np.max(ok_values)) if ok_values else math.nan,
                "gradient_support_status": (
                    "supported_fixed_branch_not_tested_full_model" if parity_ok else "not_supported_in_fixed_branch_fixture"
                ),
                "gradient_parity_ok_rows": len(parity_ok),
                "gradient_parity_max_abs_error": max((_finite_or_nan(row.get("abs_error")) for row in parity_ok), default=math.nan),
                "gradient_parity_max_rel_error": max((_finite_or_nan(row.get("rel_error")) for row in parity_ok), default=math.nan),
                "dominant_cell0_status": max(
                    sorted({str(row.get("cell0_localization_status")) for row in group}),
                    key=lambda status: sum(1 for row in group if row.get("cell0_localization_status") == status),
                ),
                "evidence_boundary_note": EVIDENCE_BOUNDARY_NOTE,
            }
        )
    out.sort(key=lambda row: (int(row["aggregate_axis_rank"]), str(row["axis"])))
    return [{column: row.get(column) for column in AXIS_SUMMARY_COLUMNS} for row in out]


def influence_tracing_markdown(
    influence_rows: Sequence[Mapping[str, Any]],
    axis_rows: Sequence[Mapping[str, Any]],
) -> str:
    ok_rows = [row for row in influence_rows if row.get("sensitivity_status") == "ok"]
    cases = sorted({str(row["case_id"]) for row in influence_rows})
    phases = sorted({str(row["phase"]) for row in influence_rows})
    n_cells = sorted({int(row["N_cells"]) for row in influence_rows})
    lines = [
        "# Huayi influence-tracing diagnostic",
        "",
        "This bundle connects 22-case fixed-physics sensitivity, fixed-branch FD-vs-IFT/JAX availability, and cell0 residual localization.",
        "",
        f"- Cases: {len(cases)}",
        f"- Phases: {', '.join(phases)}",
        f"- N_cells: {', '.join(str(n) for n in n_cells)}",
        f"- Long-form rows: {len(influence_rows)}",
        f"- Sensitivity OK rows: {len(ok_rows)}",
        "",
        "| rank | axis | mean_abs_norm_sens | max_abs_norm_sens | gradient_support | parity_ok_rows | dominant_cell0_status |",
        "|---:|---|---:|---:|---|---:|---|",
    ]
    for row in axis_rows:
        lines.append(
            "| {rank} | {axis} | {mean:.6g} | {maxv:.6g} | {grad} | {parity_ok} | {cell0} |".format(
                rank=int(row["aggregate_axis_rank"]),
                axis=row["axis"],
                mean=_finite_or_nan(row["mean_abs_normalized_sensitivity"]),
                maxv=_finite_or_nan(row["max_abs_normalized_sensitivity"]),
                grad=row["gradient_support_status"],
                parity_ok=int(row["gradient_parity_ok_rows"]),
                cell0=row["dominant_cell0_status"],
            )
        )
    lines.extend(
        [
            "",
            "Interpretation: axes are candidates for online-correction design because they are sensitivity-ranked "
            "over the Huayi dry-powder fixed-physics cases and have fixed-branch gradient parity evidence for "
            "matching feed/carrier/heat-loss directions. This is not a trained hybrid, calibrated accuracy result, "
            "causal proof, production online deployment, or full-reactor 22-case IFT parity.",
            "",
            f"Evidence boundary: {EVIDENCE_BOUNDARY_NOTE}.",
        ]
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "AXIS_SUMMARY_COLUMNS",
    "EVIDENCE_BOUNDARY_NOTE",
    "INFLUENCE_TRACING_COLUMNS",
    "TARGET_TO_GRADIENT_OUTPUT",
    "build_axis_summary_rows",
    "build_influence_tracing_rows",
    "influence_tracing_markdown",
]
