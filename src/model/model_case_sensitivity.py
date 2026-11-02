"""
Fixed-physics sensitivity diagnostics for dataprep ModelCase rows.

The helpers in this module run small centered finite-difference probes on the
existing ``GasifierSystem`` path.  They are reporting-only: no calibration,
training, learned multipliers, or solver-threshold changes are performed.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .gasifier_system import GasifierSystem
from .revision_evidence_common import legacy_mesh_op_conds
from .model_case_adapter import build_gasifier_inputs_from_model_case
from .model_case_diagnostic_baseline import (
    POOR_CONVERGENCE_COST_THRESHOLD,
    cell_residual_quality,
)

TARGET_COLUMNS: tuple[str, ...] = (
    "T_out_K",
    "CO_volpct_dry",
    "H2_volpct_dry",
    "CO2_volpct_dry",
    "dry_syngas_flow_kmol_s",
)

SENSITIVITY_COLUMNS: tuple[str, ...] = (
    "case_id",
    "source_candidate_id",
    "independence_group",
    "N_cells",
    "parameter_name",
    "parameter_key",
    "perturbation_fraction",
    "base_parameter_value",
    "minus_parameter_value",
    "plus_parameter_value",
    "target_name",
    "base_value",
    "minus_value",
    "plus_value",
    "raw_delta_centered",
    "raw_sensitivity_per_unit",
    "normalized_sensitivity",
    "abs_normalized_sensitivity",
    "base_runtime_s",
    "minus_runtime_s",
    "plus_runtime_s",
    "base_fallback_count",
    "minus_fallback_count",
    "plus_fallback_count",
    "base_poor_convergence_count",
    "minus_poor_convergence_count",
    "plus_poor_convergence_count",
    "base_poor_convergence_cell_index",
    "minus_poor_convergence_cell_index",
    "plus_poor_convergence_cell_index",
    "status",
    "failure_reason",
)

RANKING_COLUMNS: tuple[str, ...] = (
    "rank",
    "parameter_name",
    "parameter_key",
    "row_count",
    "ok_row_count",
    "mean_abs_normalized_sensitivity",
    "median_abs_normalized_sensitivity",
    "max_abs_normalized_sensitivity",
    "status",
)


@dataclass(frozen=True)
class SensitivityParameter:
    name: str
    key: str
    getter: Callable[[Mapping[str, Any]], float]
    setter: Callable[[dict[str, Any], float], None]


def _get(row: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if hasattr(row, "get"):
        return row.get(key, default)
    return getattr(row, key, default)


def _set_op_cond(key: str) -> Callable[[dict[str, Any], float], None]:
    def setter(built: dict[str, Any], value: float) -> None:
        built["op_conds"][key] = float(value)

    return setter


DEFAULT_SENSITIVITY_PARAMETERS: tuple[SensitivityParameter, ...] = (
    SensitivityParameter(
        name="HeatLossPercent",
        key="op_conds.HeatLossPercent",
        getter=lambda built: float(built["op_conds"].get("HeatLossPercent", 2.0)),
        setter=_set_op_cond("HeatLossPercent"),
    ),
    SensitivityParameter(
        name="carrier_gas_flow",
        key="op_conds.carrier_gas_flow",
        getter=lambda built: float(built["op_conds"]["carrier_gas_flow"]),
        setter=_set_op_cond("carrier_gas_flow"),
    ),
    SensitivityParameter(
        name="oxygen_flow",
        key="op_conds.o2_flow",
        getter=lambda built: float(built["op_conds"]["o2_flow"]),
        setter=_set_op_cond("o2_flow"),
    ),
    SensitivityParameter(
        name="coal_mass_flow",
        key="op_conds.coal_flow",
        getter=lambda built: float(built["op_conds"]["coal_flow"]),
        setter=_set_op_cond("coal_flow"),
    ),
)


def _dry_syngas_kpis(profile: np.ndarray) -> dict[str, float]:
    last = np.asarray(profile[-1], dtype=float)
    gas = last[:8]
    dry = float(np.sum(gas[:7]))
    denom = dry + 1.0e-12
    return {
        "T_out_K": float(last[10]),
        "CO_volpct_dry": float(gas[2] / denom * 100.0),
        "H2_volpct_dry": float(gas[5] / denom * 100.0),
        "CO2_volpct_dry": float(gas[3] / denom * 100.0),
        "dry_syngas_flow_kmol_s": dry / 1000.0,
    }


def _solve_built_case(built: Mapping[str, Any], n_cells: int) -> dict[str, Any]:
    start = time.perf_counter()
    system = GasifierSystem(built["geometry"], built["coal_props"], legacy_mesh_op_conds(built["op_conds"]))
    profile, _z = system.solve(N_cells=n_cells, solver_method="minimize", jacobian_mode="scipy", jax_warmup=False)
    runtime_s = time.perf_counter() - start
    residual_quality = cell_residual_quality(system, profile)
    poor_cells = [
        item["cell_index"]
        for item in residual_quality
        if item["cost"] > POOR_CONVERGENCE_COST_THRESHOLD
    ]
    out = {
        "kpis": _dry_syngas_kpis(profile),
        "runtime_s": runtime_s,
        "fallback_count": int(system.solve_stats.get("fallback_count", 0)),
        "poor_convergence_count": int(system.solve_stats.get("poor_convergence_count", 0)),
        "poor_convergence_cell_index": ";".join(str(i) for i in poor_cells),
    }
    return out


def _copy_built_with_parameter(built: Mapping[str, Any], parameter: SensitivityParameter, value: float) -> dict[str, Any]:
    copied = {
        "case_id": built["case_id"],
        "segment_id": built["segment_id"],
        "geometry": dict(built["geometry"]),
        "coal_props": dict(built["coal_props"]),
        "op_conds": dict(built["op_conds"]),
        "targets": dict(built["targets"]),
        "assumptions": dict(built["assumptions"]),
    }
    parameter.setter(copied, value)
    return copied


def _safe_solve(built: Mapping[str, Any], n_cells: int) -> tuple[str, str, dict[str, Any] | None]:
    try:
        return "ok", "", _solve_built_case(built, n_cells)
    except Exception as exc:  # pragma: no cover - exercised only on failing probes
        return "failed", f"{type(exc).__name__}: {exc}", None


def build_sensitivity_rows(
    model_case_rows: Iterable[Mapping[str, Any]],
    *,
    n_cells: int = 5,
    perturbation_fraction: float = 0.02,
    parameters: Sequence[SensitivityParameter] = DEFAULT_SENSITIVITY_PARAMETERS,
) -> list[dict[str, Any]]:
    """Return long-form finite-difference sensitivity rows."""
    rows: list[dict[str, Any]] = []
    for model_case_row in model_case_rows:
        built = build_gasifier_inputs_from_model_case(model_case_row)
        case_id = built["case_id"]
        source_candidate_id = built["segment_id"]
        independence_group = _get(model_case_row, "independence_group_id", _get(model_case_row, "independence_group_id_value"))
        for parameter in parameters:
            p_base = parameter.getter(built)
            if not math.isfinite(p_base) or abs(p_base) < 1.0e-12:
                for target_name in TARGET_COLUMNS:
                    rows.append(
                        {
                            "case_id": case_id,
                            "source_candidate_id": source_candidate_id,
                            "independence_group": independence_group,
                            "N_cells": int(n_cells),
                            "parameter_name": parameter.name,
                            "parameter_key": parameter.key,
                            "perturbation_fraction": float(perturbation_fraction),
                            "base_parameter_value": p_base,
                            "minus_parameter_value": math.nan,
                            "plus_parameter_value": math.nan,
                            "target_name": target_name,
                            "status": "skipped_zero_or_invalid_parameter",
                            "failure_reason": "base parameter is zero, non-finite, or too small for normalized centered FD",
                        }
                    )
                continue

            p_minus = p_base * (1.0 - perturbation_fraction)
            p_plus = p_base * (1.0 + perturbation_fraction)
            base_status, base_reason, base = _safe_solve(_copy_built_with_parameter(built, parameter, p_base), n_cells)
            minus_status, minus_reason, minus = _safe_solve(_copy_built_with_parameter(built, parameter, p_minus), n_cells)
            plus_status, plus_reason, plus = _safe_solve(_copy_built_with_parameter(built, parameter, p_plus), n_cells)
            combined_status = "ok" if base_status == minus_status == plus_status == "ok" else "failed"
            failure_reason = "; ".join(
                reason
                for reason in [
                    f"base={base_reason}" if base_reason else "",
                    f"minus={minus_reason}" if minus_reason else "",
                    f"plus={plus_reason}" if plus_reason else "",
                ]
                if reason
            )

            for target_name in TARGET_COLUMNS:
                base_value = base["kpis"][target_name] if base else math.nan
                minus_value = minus["kpis"][target_name] if minus else math.nan
                plus_value = plus["kpis"][target_name] if plus else math.nan
                if combined_status == "ok":
                    raw_delta = (plus_value - minus_value) / 2.0
                    raw_sensitivity = (plus_value - minus_value) / (p_plus - p_minus)
                    if abs(base_value) > 1.0e-12:
                        normalized = raw_sensitivity * p_base / base_value
                        abs_normalized = abs(normalized)
                        row_status = "ok"
                        row_reason = ""
                    else:
                        normalized = math.nan
                        abs_normalized = math.nan
                        row_status = "near_zero_target"
                        row_reason = "base target too close to zero for normalized sensitivity"
                else:
                    raw_delta = math.nan
                    raw_sensitivity = math.nan
                    normalized = math.nan
                    abs_normalized = math.nan
                    row_status = combined_status
                    row_reason = failure_reason
                rows.append(
                    {
                        "case_id": case_id,
                        "source_candidate_id": source_candidate_id,
                        "independence_group": independence_group,
                        "N_cells": int(n_cells),
                        "parameter_name": parameter.name,
                        "parameter_key": parameter.key,
                        "perturbation_fraction": float(perturbation_fraction),
                        "base_parameter_value": p_base,
                        "minus_parameter_value": p_minus,
                        "plus_parameter_value": p_plus,
                        "target_name": target_name,
                        "base_value": base_value,
                        "minus_value": minus_value,
                        "plus_value": plus_value,
                        "raw_delta_centered": raw_delta,
                        "raw_sensitivity_per_unit": raw_sensitivity,
                        "normalized_sensitivity": normalized,
                        "abs_normalized_sensitivity": abs_normalized,
                        "base_runtime_s": base["runtime_s"] if base else math.nan,
                        "minus_runtime_s": minus["runtime_s"] if minus else math.nan,
                        "plus_runtime_s": plus["runtime_s"] if plus else math.nan,
                        "base_fallback_count": base["fallback_count"] if base else math.nan,
                        "minus_fallback_count": minus["fallback_count"] if minus else math.nan,
                        "plus_fallback_count": plus["fallback_count"] if plus else math.nan,
                        "base_poor_convergence_count": base["poor_convergence_count"] if base else math.nan,
                        "minus_poor_convergence_count": minus["poor_convergence_count"] if minus else math.nan,
                        "plus_poor_convergence_count": plus["poor_convergence_count"] if plus else math.nan,
                        "base_poor_convergence_cell_index": base["poor_convergence_cell_index"] if base else "",
                        "minus_poor_convergence_cell_index": minus["poor_convergence_cell_index"] if minus else "",
                        "plus_poor_convergence_cell_index": plus["poor_convergence_cell_index"] if plus else "",
                        "status": row_status,
                        "failure_reason": row_reason,
                    }
                )
    return [{column: row.get(column) for column in SENSITIVITY_COLUMNS} for row in rows]


def build_parameter_ranking(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate long-form rows into a parameter ranking."""
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["parameter_name"]), str(row["parameter_key"])), []).append(row)

    ranking: list[dict[str, Any]] = []
    for (name, key), group in grouped.items():
        ok_values = [
            float(row["abs_normalized_sensitivity"])
            for row in group
            if row.get("status") == "ok" and row.get("abs_normalized_sensitivity") is not None
            and math.isfinite(float(row["abs_normalized_sensitivity"]))
        ]
        if ok_values:
            status = "ok"
            mean_abs = float(np.mean(ok_values))
            median_abs = float(np.median(ok_values))
            max_abs = float(np.max(ok_values))
        else:
            status = "no_ok_rows"
            mean_abs = math.nan
            median_abs = math.nan
            max_abs = math.nan
        ranking.append(
            {
                "rank": 0,
                "parameter_name": name,
                "parameter_key": key,
                "row_count": len(group),
                "ok_row_count": len(ok_values),
                "mean_abs_normalized_sensitivity": mean_abs,
                "median_abs_normalized_sensitivity": median_abs,
                "max_abs_normalized_sensitivity": max_abs,
                "status": status,
            }
        )

    ranking.sort(
        key=lambda row: (
            not math.isfinite(float(row["mean_abs_normalized_sensitivity"])),
            -float(row["mean_abs_normalized_sensitivity"]) if math.isfinite(float(row["mean_abs_normalized_sensitivity"])) else 0.0,
            row["parameter_name"],
        )
    )
    for idx, row in enumerate(ranking, start=1):
        row["rank"] = idx
    return [{column: row.get(column) for column in RANKING_COLUMNS} for row in ranking]


def sensitivity_markdown(rows: Sequence[Mapping[str, Any]], ranking: Sequence[Mapping[str, Any]]) -> str:
    """Return a compact Markdown summary for fixed-physics sensitivity artifacts."""
    ok_rows = sum(1 for row in rows if row.get("status") == "ok")
    failed_rows = len(rows) - ok_rows
    n_cells = sorted({int(row["N_cells"]) for row in rows}) if rows else []
    cases = sorted({str(row["case_id"]) for row in rows}) if rows else []
    lines = [
        "# Fixed-physics sensitivity ranking diagnostic",
        "",
        "This is finite-difference fixed-physics sensitivity evidence, not calibration accuracy evidence.",
        "",
        f"- Cases: {len(cases)}",
        f"- Long-form rows: {len(rows)}",
        f"- OK rows: {ok_rows}",
        f"- Non-OK rows: {failed_rows}",
        f"- N_cells: {', '.join(str(n) for n in n_cells)}",
        "",
        "| rank | parameter | mean_abs_norm_sens | median_abs_norm_sens | max_abs_norm_sens | ok_rows | status |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in ranking:
        lines.append(
            "| {rank} | {param} | {mean:.6g} | {median:.6g} | {maxv:.6g} | {ok} | {status} |".format(
                rank=int(row["rank"]),
                param=row["parameter_name"],
                mean=float(row["mean_abs_normalized_sensitivity"]),
                median=float(row["median_abs_normalized_sensitivity"]),
                maxv=float(row["max_abs_normalized_sensitivity"]),
                ok=int(row["ok_row_count"]),
                status=row["status"],
            )
        )
    lines.extend(
        [
            "",
            "Evidence boundary: pilot representative ModelCases, centered finite differences, "
            "existing fixed-physics solver only. Use as a ranking/diagnostic input for later "
            "hybrid-correction design, not as fitted model accuracy.",
        ]
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "DEFAULT_SENSITIVITY_PARAMETERS",
    "RANKING_COLUMNS",
    "SENSITIVITY_COLUMNS",
    "TARGET_COLUMNS",
    "SensitivityParameter",
    "build_parameter_ranking",
    "build_sensitivity_rows",
    "sensitivity_markdown",
]
