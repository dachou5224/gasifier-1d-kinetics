"""
Engineering-coordinate influence diagnostics for Huayi ModelCases.

This layer translates raw sensitivity axes into plant-relevant coordinates:
O/C ratio, load at fixed O/C, carrier/coal transport boundary, and heat-loss
calibration. It is reporting-only and does not change solver behavior.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .model_case_sensitivity import SensitivityParameter
from .physics import MOLAR_MASS

CARBON_ATOMIC_MASS_KG_PER_KMOL = 12.011

ENGINEERING_EVIDENCE_BOUNDARY = (
    "engineering-coordinate fixed-physics influence tracing only; not trained hybrid, "
    "calibrated accuracy, causal proof, production online deployment, full-reactor "
    "22-case IFT parity, independent plant manipulation of carrier gas or heat loss, "
    "or Yulin/slurry generalization"
)

ENGINEERING_COORDINATE_COLUMNS: tuple[str, ...] = (
    "phase",
    "dataset_id",
    "case_id",
    "source_candidate_id",
    "independence_group",
    "N_cells",
    "coordinate",
    "coordinate_class",
    "perturbation_rule",
    "perturbed_raw_fields",
    "target",
    "perturbation_fraction",
    "base_coordinate_value",
    "minus_coordinate_value",
    "plus_coordinate_value",
    "base_value",
    "minus_value",
    "plus_value",
    "raw_delta_centered",
    "raw_sensitivity_per_unit",
    "normalized_sensitivity",
    "abs_normalized_sensitivity",
    "case_target_coordinate_rank",
    "aggregate_coordinate_rank",
    "aggregate_mean_abs_normalized_sensitivity",
    "aggregate_median_abs_normalized_sensitivity",
    "aggregate_max_abs_normalized_sensitivity",
    "sensitivity_status",
    "gradient_support_status",
    "cell0_cost",
    "cell0_max_abs_residual",
    "fallback_count",
    "poor_convergence_count",
    "poor_convergence_cell_index",
    "evidence_boundary_note",
)

ENGINEERING_AXIS_COLUMNS: tuple[str, ...] = (
    "aggregate_coordinate_rank",
    "coordinate",
    "coordinate_class",
    "perturbation_rule",
    "perturbed_raw_fields",
    "row_count",
    "ok_row_count",
    "mean_abs_normalized_sensitivity",
    "median_abs_normalized_sensitivity",
    "max_abs_normalized_sensitivity",
    "gradient_support_status",
    "evidence_boundary_note",
)

COORDINATE_METADATA: Mapping[str, Mapping[str, str]] = {
    "O_C_ratio_at_fixed_load": {
        "coordinate_class": "plant_operating",
        "perturbation_rule": "perturb O2 flow to change molar O/C while dry coal load and carrier boundary remain fixed",
        "perturbed_raw_fields": "op_conds.o2_flow",
        "gradient_support_status": "mapped_to_supported_fixed_branch_oxygen_flow_not_full_22case_coordinate_IFT",
    },
    "load_at_fixed_O_C": {
        "coordinate_class": "load",
        "perturbation_rule": "scale dry coal, O2, and carrier gas together to preserve baseline O/C and carrier/coal ratios",
        "perturbed_raw_fields": "op_conds.coal_flow;op_conds.o2_flow;op_conds.carrier_gas_flow",
        "gradient_support_status": "composed_from_supported_fixed_branch_feed_axes_not_tested_as_full_coordinate_IFT",
    },
    "carrier_to_coal_ratio_boundary": {
        "coordinate_class": "transport_boundary",
        "perturbation_rule": "perturb carrier gas relative to fixed dry coal flow; boundary sensitivity, not independent plant priority",
        "perturbed_raw_fields": "op_conds.carrier_gas_flow",
        "gradient_support_status": "mapped_to_supported_fixed_branch_carrier_flow_not_full_22case_coordinate_IFT",
    },
    "HeatLossPercent_calibration": {
        "coordinate_class": "model_calibration",
        "perturbation_rule": "perturb HeatLossPercent calibration parameter; not a plant operating knob",
        "perturbed_raw_fields": "op_conds.HeatLossPercent",
        "gradient_support_status": "mapped_to_supported_fixed_branch_heat_loss_not_full_22case_coordinate_IFT",
    },
}


def _coal_carbon_kmol_s(built: Mapping[str, Any]) -> float:
    coal_flow = float(built["op_conds"]["coal_flow"])
    carbon_fraction = float(built["coal_props"]["Cd"]) / 100.0
    return coal_flow * carbon_fraction / CARBON_ATOMIC_MASS_KG_PER_KMOL


def _o_c_ratio(built: Mapping[str, Any]) -> float:
    o2_flow = float(built["op_conds"]["o2_flow"])
    oxygen_atom_kmol_s = 2.0 * o2_flow / MOLAR_MASS["O2"]
    carbon_kmol_s = _coal_carbon_kmol_s(built)
    return oxygen_atom_kmol_s / carbon_kmol_s


def _carrier_to_coal_ratio(built: Mapping[str, Any]) -> float:
    return float(built["op_conds"]["carrier_gas_flow"]) / float(built["op_conds"]["coal_flow"])


def _set_o_c_ratio_at_fixed_load(built: dict[str, Any], value: float) -> None:
    base_ratio = _o_c_ratio(built)
    built["op_conds"]["o2_flow"] = float(built["op_conds"]["o2_flow"]) * float(value) / base_ratio


def _set_load_at_fixed_o_c(built: dict[str, Any], value: float) -> None:
    base_load = float(built["op_conds"]["coal_flow"])
    scale = float(value) / base_load
    built["op_conds"]["coal_flow"] = float(value)
    built["op_conds"]["o2_flow"] = float(built["op_conds"]["o2_flow"]) * scale
    built["op_conds"]["carrier_gas_flow"] = float(built["op_conds"]["carrier_gas_flow"]) * scale


def _set_carrier_to_coal_ratio(built: dict[str, Any], value: float) -> None:
    built["op_conds"]["carrier_gas_flow"] = float(value) * float(built["op_conds"]["coal_flow"])


def _set_heat_loss_percent(built: dict[str, Any], value: float) -> None:
    built["op_conds"]["HeatLossPercent"] = float(value)


ENGINEERING_COORDINATE_PARAMETERS: tuple[SensitivityParameter, ...] = (
    SensitivityParameter(
        name="O_C_ratio_at_fixed_load",
        key="engineering.O_C_ratio",
        getter=_o_c_ratio,
        setter=_set_o_c_ratio_at_fixed_load,
    ),
    SensitivityParameter(
        name="load_at_fixed_O_C",
        key="engineering.dry_coal_load_kg_s",
        getter=lambda built: float(built["op_conds"]["coal_flow"]),
        setter=_set_load_at_fixed_o_c,
    ),
    SensitivityParameter(
        name="carrier_to_coal_ratio_boundary",
        key="engineering.carrier_gas_flow_per_coal_flow",
        getter=_carrier_to_coal_ratio,
        setter=_set_carrier_to_coal_ratio,
    ),
    SensitivityParameter(
        name="HeatLossPercent_calibration",
        key="engineering.HeatLossPercent_calibration",
        getter=lambda built: float(built["op_conds"].get("HeatLossPercent", 2.0)),
        setter=_set_heat_loss_percent,
    ),
)


def _finite_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _baseline_index(baseline_rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], Mapping[str, Any]]:
    return {(str(row["case_id"]), int(row["N_cells"])): row for row in baseline_rows}


def _contract_index(contract_rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row["case_id"]): row for row in contract_rows}


def build_engineering_axis_summary(sensitivity_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in sensitivity_rows:
        grouped.setdefault(str(row["parameter_name"]), []).append(row)

    rows: list[dict[str, Any]] = []
    for coordinate, group in grouped.items():
        values = [
            _finite_or_nan(row.get("abs_normalized_sensitivity"))
            for row in group
            if row.get("status") == "ok" and math.isfinite(_finite_or_nan(row.get("abs_normalized_sensitivity")))
        ]
        meta = COORDINATE_METADATA[coordinate]
        rows.append(
            {
                "aggregate_coordinate_rank": 0,
                "coordinate": coordinate,
                "coordinate_class": meta["coordinate_class"],
                "perturbation_rule": meta["perturbation_rule"],
                "perturbed_raw_fields": meta["perturbed_raw_fields"],
                "row_count": len(group),
                "ok_row_count": len(values),
                "mean_abs_normalized_sensitivity": float(np.mean(values)) if values else math.nan,
                "median_abs_normalized_sensitivity": float(np.median(values)) if values else math.nan,
                "max_abs_normalized_sensitivity": float(np.max(values)) if values else math.nan,
                "gradient_support_status": meta["gradient_support_status"],
                "evidence_boundary_note": ENGINEERING_EVIDENCE_BOUNDARY,
            }
        )
    rows.sort(
        key=lambda row: (
            not math.isfinite(_finite_or_nan(row["mean_abs_normalized_sensitivity"])),
            -_finite_or_nan(row["mean_abs_normalized_sensitivity"]),
            str(row["coordinate"]),
        )
    )
    for idx, row in enumerate(rows, start=1):
        row["aggregate_coordinate_rank"] = idx
    return [{column: row.get(column) for column in ENGINEERING_AXIS_COLUMNS} for row in rows]


def build_engineering_coordinate_rows(
    sensitivity_rows: Sequence[Mapping[str, Any]],
    axis_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    contract_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    baseline_by_key = _baseline_index(baseline_rows)
    contract_by_case = _contract_index(contract_rows)
    axis_by_name = {str(row["coordinate"]): row for row in axis_rows}
    pre_rows: list[dict[str, Any]] = []
    for row in sensitivity_rows:
        coordinate = str(row["parameter_name"])
        meta = COORDINATE_METADATA[coordinate]
        case_id = str(row["case_id"])
        n_cells = int(row["N_cells"])
        baseline = baseline_by_key.get((case_id, n_cells), {})
        contract = contract_by_case.get(case_id, {})
        axis = axis_by_name[coordinate]
        pre_rows.append(
            {
                "phase": contract.get("phase"),
                "dataset_id": contract.get("dataset_id"),
                "case_id": case_id,
                "source_candidate_id": row.get("source_candidate_id"),
                "independence_group": row.get("independence_group"),
                "N_cells": n_cells,
                "coordinate": coordinate,
                "coordinate_class": meta["coordinate_class"],
                "perturbation_rule": meta["perturbation_rule"],
                "perturbed_raw_fields": meta["perturbed_raw_fields"],
                "target": row.get("target_name"),
                "perturbation_fraction": row.get("perturbation_fraction"),
                "base_coordinate_value": row.get("base_parameter_value"),
                "minus_coordinate_value": row.get("minus_parameter_value"),
                "plus_coordinate_value": row.get("plus_parameter_value"),
                "base_value": row.get("base_value"),
                "minus_value": row.get("minus_value"),
                "plus_value": row.get("plus_value"),
                "raw_delta_centered": row.get("raw_delta_centered"),
                "raw_sensitivity_per_unit": row.get("raw_sensitivity_per_unit"),
                "normalized_sensitivity": row.get("normalized_sensitivity"),
                "abs_normalized_sensitivity": row.get("abs_normalized_sensitivity"),
                "case_target_coordinate_rank": 0,
                "aggregate_coordinate_rank": axis.get("aggregate_coordinate_rank"),
                "aggregate_mean_abs_normalized_sensitivity": axis.get("mean_abs_normalized_sensitivity"),
                "aggregate_median_abs_normalized_sensitivity": axis.get("median_abs_normalized_sensitivity"),
                "aggregate_max_abs_normalized_sensitivity": axis.get("max_abs_normalized_sensitivity"),
                "sensitivity_status": row.get("status"),
                "gradient_support_status": meta["gradient_support_status"],
                "cell0_cost": baseline.get("cell0_cost", math.nan),
                "cell0_max_abs_residual": baseline.get("cell0_max_abs_residual", math.nan),
                "fallback_count": baseline.get("fallback_count", math.nan),
                "poor_convergence_count": baseline.get("poor_convergence_count", math.nan),
                "poor_convergence_cell_index": baseline.get("poor_convergence_cell_index", ""),
                "evidence_boundary_note": ENGINEERING_EVIDENCE_BOUNDARY,
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
                str(item["coordinate"]),
            )
        )
        for idx, item in enumerate(group, start=1):
            item["case_target_coordinate_rank"] = idx

    pre_rows.sort(
        key=lambda row: (
            str(row["phase"]),
            str(row["case_id"]),
            int(row["N_cells"]),
            str(row["target"]),
            int(row["case_target_coordinate_rank"]),
        )
    )
    return [{column: row.get(column) for column in ENGINEERING_COORDINATE_COLUMNS} for row in pre_rows]


def engineering_coordinate_markdown(
    rows: Sequence[Mapping[str, Any]],
    axis_rows: Sequence[Mapping[str, Any]],
) -> str:
    cases = sorted({str(row["case_id"]) for row in rows})
    ok_count = sum(1 for row in rows if row.get("sensitivity_status") == "ok")
    lines = [
        "# Huayi engineering-coordinate influence tracing",
        "",
        "This diagnostic translates raw feed/carrier/heat-loss sensitivities into plant-relevant engineering coordinates.",
        "",
        f"- Cases: {len(cases)}",
        f"- Rows: {len(rows)}",
        f"- OK rows: {ok_count}",
        "- O/C ratio is treated as the primary plant-facing operating coordinate.",
        "- Carrier gas is treated as a coal-transport-coupled boundary, not an independent gasifier control priority.",
        "- HeatLossPercent is treated as a model-calibration coordinate, not a plant operating knob.",
        "",
        "| rank | coordinate | class | mean_abs_norm_sens | max_abs_norm_sens | rule |",
        "|---:|---|---|---:|---:|---|",
    ]
    for row in axis_rows:
        lines.append(
            "| {rank} | {coord} | {cls} | {mean:.6g} | {maxv:.6g} | {rule} |".format(
                rank=int(row["aggregate_coordinate_rank"]),
                coord=row["coordinate"],
                cls=row["coordinate_class"],
                mean=_finite_or_nan(row["mean_abs_normalized_sensitivity"]),
                maxv=_finite_or_nan(row["max_abs_normalized_sensitivity"]),
                rule=row["perturbation_rule"],
            )
        )
    lines.extend(
        [
            "",
            "Interpretation boundary: coordinate ranking supports diagnostic localization and candidate "
            "online-correction axis selection. It is not a trained hybrid, calibrated accuracy result, "
            "causal proof, production online deployment, full-reactor 22-case IFT parity, independent "
            "carrier/heat-loss plant manipulation claim, or Yulin/slurry generalization.",
            "",
            f"Evidence boundary: {ENGINEERING_EVIDENCE_BOUNDARY}.",
        ]
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "COORDINATE_METADATA",
    "ENGINEERING_AXIS_COLUMNS",
    "ENGINEERING_COORDINATE_COLUMNS",
    "ENGINEERING_COORDINATE_PARAMETERS",
    "ENGINEERING_EVIDENCE_BOUNDARY",
    "build_engineering_axis_summary",
    "build_engineering_coordinate_rows",
    "engineering_coordinate_markdown",
]
