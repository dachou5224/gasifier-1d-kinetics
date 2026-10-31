"""
O/C-centered fixed-physics response curve diagnostics.

The helpers here perturb the molar O/C coordinate by changing O2 flow at fixed
dry coal load and fixed carrier boundary. They are reporting-only and do not
change solver physics or convergence thresholds.
"""
from __future__ import annotations

import math
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .engineering_coordinate_influence import (
    ENGINEERING_EVIDENCE_BOUNDARY,
    ENGINEERING_COORDINATE_PARAMETERS,
)
from .gasifier_system import GasifierSystem
from .model_case_adapter import build_gasifier_inputs_from_model_case
from .model_case_diagnostic_baseline import (
    POOR_CONVERGENCE_COST_THRESHOLD,
    cell_residual_quality,
)

OC_MULTIPLIERS: tuple[float, ...] = (0.96, 0.98, 1.00, 1.02, 1.04)

OC_RESPONSE_COLUMNS: tuple[str, ...] = (
    "phase",
    "dataset_id",
    "case_id",
    "source_candidate_id",
    "independence_group",
    "N_cells",
    "oc_multiplier",
    "base_o_c_ratio",
    "o_c_ratio",
    "o2_flow_kg_s",
    "coal_flow_kg_s",
    "carrier_gas_flow_kg_s",
    "runtime_s",
    "status",
    "failure_reason",
    "fallback_count",
    "poor_convergence_count",
    "poor_convergence_cell_index",
    "cell0_cost",
    "cell0_max_abs_residual",
    "T_out_K",
    "T_out_degC",
    "CO_volpct_dry",
    "H2_volpct_dry",
    "CO2_volpct_dry",
    "N2_volpct_dry",
    "dry_syngas_flow_kmol_s",
    "H2_CO_ratio",
    "evidence_boundary_note",
)

OC_RESPONSE_AXIS_COLUMNS: tuple[str, ...] = (
    "phase",
    "dataset_id",
    "case_id",
    "source_candidate_id",
    "independence_group",
    "N_cells",
    "target_name",
    "oc_multiplier",
    "base_o_c_ratio",
    "o_c_ratio",
    "target_value",
    "baseline_target_value",
    "delta_from_baseline",
    "local_slope_per_O_C",
    "local_slope_per_relative_multiplier",
    "local_slope_sign",
    "curve_direction_status",
    "slope_direction_matches_curve",
    "response_status",
    "evidence_boundary_note",
)

OC_RESPONSE_TARGETS: tuple[str, ...] = (
    "T_out_K",
    "CO_volpct_dry",
    "H2_volpct_dry",
    "CO2_volpct_dry",
    "dry_syngas_flow_kmol_s",
    "H2_CO_ratio",
)

OC_RESPONSE_EVIDENCE_BOUNDARY = (
    "fixed-physics O/C response evidence on Huayi dry-powder ModelCases; "
    "supports plant-coordinate influence tracing and figure generation only; "
    "not calibrated accuracy, causal proof, trained hybrid validation, "
    "production control recommendation, or Yulin/slurry generalization"
)


def _get(row: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if hasattr(row, "get"):
        return row.get(key, default)
    return getattr(row, key, default)


def _finite_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _oc_parameter():
    for parameter in ENGINEERING_COORDINATE_PARAMETERS:
        if parameter.name == "O_C_ratio_at_fixed_load":
            return parameter
    raise RuntimeError("O_C_ratio_at_fixed_load parameter is not registered")


def _copy_built(built: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case_id": built["case_id"],
        "segment_id": built["segment_id"],
        "geometry": dict(built["geometry"]),
        "coal_props": dict(built["coal_props"]),
        "op_conds": dict(built["op_conds"]),
        "targets": dict(built["targets"]),
        "assumptions": dict(built["assumptions"]),
    }


def _dry_syngas_kpis(profile: np.ndarray) -> dict[str, float]:
    last = np.asarray(profile[-1], dtype=float)
    gas = last[:8]
    dry = float(np.sum(gas[:7]))
    denom = dry + 1.0e-12
    co = float(gas[2] / denom * 100.0)
    h2 = float(gas[5] / denom * 100.0)
    return {
        "T_out_K": float(last[10]),
        "T_out_degC": float(last[10] - 273.15),
        "CO_volpct_dry": co,
        "H2_volpct_dry": h2,
        "CO2_volpct_dry": float(gas[3] / denom * 100.0),
        "N2_volpct_dry": float(gas[6] / denom * 100.0),
        "dry_syngas_flow_kmol_s": dry / 1000.0,
        "H2_CO_ratio": h2 / max(co, 1.0e-12),
    }


def _solve_built_case(built: Mapping[str, Any], n_cells: int) -> dict[str, Any]:
    start = time.perf_counter()
    system = GasifierSystem(built["geometry"], built["coal_props"], built["op_conds"])
    profile, _z = system.solve(N_cells=n_cells, solver_method="minimize", jacobian_mode="scipy", jax_warmup=False)
    runtime_s = time.perf_counter() - start
    residual_quality = cell_residual_quality(system, profile)
    poor_cells = [
        item["cell_index"]
        for item in residual_quality
        if item["cost"] > POOR_CONVERGENCE_COST_THRESHOLD
    ]
    cell0 = residual_quality[0] if residual_quality else {}
    return {
        "kpis": _dry_syngas_kpis(profile),
        "runtime_s": runtime_s,
        "fallback_count": int(system.solve_stats.get("fallback_count", 0)),
        "poor_convergence_count": int(system.solve_stats.get("poor_convergence_count", 0)),
        "poor_convergence_cell_index": ";".join(str(i) for i in poor_cells),
        "cell0_cost": cell0.get("cost", math.nan),
        "cell0_max_abs_residual": cell0.get("max_abs_residual", math.nan),
    }


def _response_row(
    model_case_row: Mapping[str, Any],
    built: Mapping[str, Any],
    n_cells: int,
    base_o_c: float,
    multiplier: float,
) -> dict[str, Any]:
    parameter = _oc_parameter()
    perturbed = _copy_built(built)
    target_o_c = base_o_c * float(multiplier)
    parameter.setter(perturbed, target_o_c)
    common = {
        "phase": _get(model_case_row, "phase"),
        "dataset_id": _get(model_case_row, "dataset_id"),
        "case_id": built["case_id"],
        "source_candidate_id": built["segment_id"],
        "independence_group": _get(model_case_row, "independence_group_id", _get(model_case_row, "independence_group_id_value")),
        "N_cells": int(n_cells),
        "oc_multiplier": float(multiplier),
        "base_o_c_ratio": float(base_o_c),
        "o_c_ratio": parameter.getter(perturbed),
        "o2_flow_kg_s": float(perturbed["op_conds"]["o2_flow"]),
        "coal_flow_kg_s": float(perturbed["op_conds"]["coal_flow"]),
        "carrier_gas_flow_kg_s": float(perturbed["op_conds"]["carrier_gas_flow"]),
        "evidence_boundary_note": OC_RESPONSE_EVIDENCE_BOUNDARY,
    }
    try:
        solved = _solve_built_case(perturbed, n_cells)
    except Exception as exc:  # pragma: no cover - exercised only by failing probes
        row = {
            **common,
            "runtime_s": math.nan,
            "status": "failed",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "fallback_count": math.nan,
            "poor_convergence_count": math.nan,
            "poor_convergence_cell_index": "",
            "cell0_cost": math.nan,
            "cell0_max_abs_residual": math.nan,
        }
        for key in ("T_out_K", "T_out_degC", "CO_volpct_dry", "H2_volpct_dry", "CO2_volpct_dry", "N2_volpct_dry", "dry_syngas_flow_kmol_s", "H2_CO_ratio"):
            row[key] = math.nan
        return row

    row = {
        **common,
        "runtime_s": solved["runtime_s"],
        "status": "ok",
        "failure_reason": "",
        "fallback_count": solved["fallback_count"],
        "poor_convergence_count": solved["poor_convergence_count"],
        "poor_convergence_cell_index": solved["poor_convergence_cell_index"],
        "cell0_cost": solved["cell0_cost"],
        "cell0_max_abs_residual": solved["cell0_max_abs_residual"],
    }
    row.update(solved["kpis"])
    return {column: row.get(column) for column in OC_RESPONSE_COLUMNS}


def build_oc_response_rows(
    model_case_rows: Iterable[Mapping[str, Any]],
    *,
    n_cells: int = 5,
    multipliers: Sequence[float] = OC_MULTIPLIERS,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    parameter = _oc_parameter()
    for model_case_row in model_case_rows:
        built = build_gasifier_inputs_from_model_case(model_case_row)
        base_o_c = parameter.getter(built)
        for multiplier in multipliers:
            rows.append(_response_row(model_case_row, built, n_cells, base_o_c, float(multiplier)))
    return rows


def _direction(values: Sequence[float]) -> str:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if len(finite) < 3:
        return "insufficient"
    diffs = np.diff(np.asarray(finite, dtype=float))
    tol = max(1.0e-10, float(np.max(np.abs(finite))) * 1.0e-8)
    if np.all(diffs >= -tol) and np.any(diffs > tol):
        return "monotonic_increasing"
    if np.all(diffs <= tol) and np.any(diffs < -tol):
        return "monotonic_decreasing"
    if np.all(np.abs(diffs) <= tol):
        return "flat_or_indeterminate"
    return "nonmonotonic"


def _sign(value: float) -> str:
    if not math.isfinite(value):
        return "not_available"
    if value > 0.0:
        return "positive"
    if value < 0.0:
        return "negative"
    return "zero"


def build_oc_response_axis_rows(response_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in response_rows:
        grouped.setdefault((str(row["case_id"]), int(row["N_cells"])), []).append(row)

    out: list[dict[str, Any]] = []
    for (_case_id, _n_cells), group in grouped.items():
        sorted_group = sorted(group, key=lambda row: float(row["oc_multiplier"]))
        base = min(sorted_group, key=lambda row: abs(float(row["oc_multiplier"]) - 1.0))
        minus = min(sorted_group, key=lambda row: abs(float(row["oc_multiplier"]) - 0.98))
        plus = min(sorted_group, key=lambda row: abs(float(row["oc_multiplier"]) - 1.02))
        for target in OC_RESPONSE_TARGETS:
            base_value = _finite_or_nan(base.get(target))
            minus_value = _finite_or_nan(minus.get(target))
            plus_value = _finite_or_nan(plus.get(target))
            minus_o_c = _finite_or_nan(minus.get("o_c_ratio"))
            plus_o_c = _finite_or_nan(plus.get("o_c_ratio"))
            if math.isfinite(minus_value) and math.isfinite(plus_value) and plus_o_c != minus_o_c:
                slope_per_o_c = (plus_value - minus_value) / (plus_o_c - minus_o_c)
                slope_per_relative = (plus_value - minus_value) / (float(plus["oc_multiplier"]) - float(minus["oc_multiplier"]))
            else:
                slope_per_o_c = math.nan
                slope_per_relative = math.nan
            direction = _direction([_finite_or_nan(row.get(target)) for row in sorted_group if row.get("status") == "ok"])
            slope_sign = _sign(slope_per_o_c)
            if direction == "monotonic_increasing":
                matches = slope_sign == "positive"
            elif direction == "monotonic_decreasing":
                matches = slope_sign == "negative"
            elif direction in {"flat_or_indeterminate", "insufficient"}:
                matches = ""
            else:
                matches = "nonmonotonic_curve"

            for row in sorted_group:
                target_value = _finite_or_nan(row.get(target))
                out.append(
                    {
                        "phase": row.get("phase"),
                        "dataset_id": row.get("dataset_id"),
                        "case_id": row.get("case_id"),
                        "source_candidate_id": row.get("source_candidate_id"),
                        "independence_group": row.get("independence_group"),
                        "N_cells": row.get("N_cells"),
                        "target_name": target,
                        "oc_multiplier": row.get("oc_multiplier"),
                        "base_o_c_ratio": row.get("base_o_c_ratio"),
                        "o_c_ratio": row.get("o_c_ratio"),
                        "target_value": target_value,
                        "baseline_target_value": base_value,
                        "delta_from_baseline": target_value - base_value if math.isfinite(target_value) and math.isfinite(base_value) else math.nan,
                        "local_slope_per_O_C": slope_per_o_c,
                        "local_slope_per_relative_multiplier": slope_per_relative,
                        "local_slope_sign": slope_sign,
                        "curve_direction_status": direction,
                        "slope_direction_matches_curve": matches,
                        "response_status": row.get("status"),
                        "evidence_boundary_note": OC_RESPONSE_EVIDENCE_BOUNDARY,
                    }
                )
    out.sort(
        key=lambda row: (
            str(row["phase"]),
            str(row["case_id"]),
            str(row["target_name"]),
            float(row["oc_multiplier"]),
        )
    )
    return [{column: row.get(column) for column in OC_RESPONSE_AXIS_COLUMNS} for row in out]


def oc_response_markdown(
    response_rows: Sequence[Mapping[str, Any]],
    axis_rows: Sequence[Mapping[str, Any]],
) -> str:
    cases = sorted({str(row["case_id"]) for row in response_rows})
    phases = sorted({str(row.get("phase")) for row in response_rows})
    multipliers = sorted({_finite_or_nan(row["oc_multiplier"]) for row in response_rows})
    ok_rows = sum(1 for row in response_rows if row.get("status") == "ok")
    axis_df: dict[str, list[float]] = {}
    for row in axis_rows:
        if row.get("response_status") == "ok" and math.isfinite(_finite_or_nan(row.get("local_slope_per_relative_multiplier"))):
            axis_df.setdefault(str(row["target_name"]), []).append(abs(float(row["local_slope_per_relative_multiplier"])))

    lines = [
        "# Huayi O/C response curve diagnostic",
        "",
        "This diagnostic perturbs molar O/C by changing oxygen flow at fixed dry coal load and fixed carrier boundary.",
        "",
        f"- Cases: {len(cases)}",
        f"- Phases: {', '.join(phases)}",
        f"- Response rows: {len(response_rows)}",
        f"- OK response rows: {ok_rows}",
        f"- Figure-axis rows: {len(axis_rows)}",
        f"- O/C multipliers: {', '.join(f'{value:.2f}' for value in multipliers)}",
        "- Solver path: GasifierSystem.solve(solver_method='minimize', jacobian_mode='scipy')",
        "- Local slopes use the centered 0.98/1.02 O/C multiplier points around baseline.",
        "- IFT/JAX parity is not claimed for this full 22-case response grid; this is fixed-physics finite-perturbation response evidence.",
        "",
        "| target | mean_abs_local_slope_per_relative_multiplier |",
        "|---|---:|",
    ]
    for target in OC_RESPONSE_TARGETS:
        values = axis_df.get(target, [])
        mean_abs = float(np.mean(values)) if values else math.nan
        lines.append(f"| {target} | {mean_abs:.6g} |")
    lines.extend(
        [
            "",
            f"Evidence boundary: {OC_RESPONSE_EVIDENCE_BOUNDARY}.",
            f"Engineering-coordinate boundary: {ENGINEERING_EVIDENCE_BOUNDARY}.",
        ]
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "OC_MULTIPLIERS",
    "OC_RESPONSE_AXIS_COLUMNS",
    "OC_RESPONSE_COLUMNS",
    "OC_RESPONSE_EVIDENCE_BOUNDARY",
    "OC_RESPONSE_TARGETS",
    "build_oc_response_axis_rows",
    "build_oc_response_rows",
    "oc_response_markdown",
]
