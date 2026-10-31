"""
Cell-0 residual-quality diagnostics for pilot ModelCase fixed-physics solves.

These helpers are reporting-only.  They recompute the existing scaled residual
vector after a solve and expose the dominant components across mesh choices.
They do not change solver behavior or relax the poor-quality threshold.
"""
from __future__ import annotations

import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .gasifier_system import GasifierSystem
from .revision_evidence_common import legacy_mesh_op_conds
from .material import SPECIES_NAMES
from .model_case_adapter import build_gasifier_inputs_from_model_case
from .model_case_diagnostic_baseline import POOR_CONVERGENCE_COST_THRESHOLD

CELL0_RESIDUAL_COLUMNS: tuple[str, ...] = (
    "case_id",
    "source_candidate_id",
    "independence_group",
    "N_cells",
    "cell_index",
    "runtime_s",
    "residual_cost",
    "l2_residual",
    "max_abs_residual",
    "poor_quality_threshold",
    "is_poor_quality",
    "dominant_residual_name",
    "dominant_residual_index",
    "dominant_residual_value",
    "dominant_residual_abs_value",
    "top1_name",
    "top1_index",
    "top1_value",
    "top1_abs_value",
    "top2_name",
    "top2_index",
    "top2_value",
    "top2_abs_value",
    "top3_name",
    "top3_index",
    "top3_value",
    "top3_abs_value",
    "fallback_count",
    "poor_convergence_count",
    "poor_convergence_cell_index",
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
    "pred_dry_flow_kmol_s",
    "target_dry_flow_kmol_s",
    "delta_dry_flow_kmol_s",
    "carrier_mapping_ok",
    "transported_moisture_percent",
    "residual_normalization",
)

RESIDUAL_NAMES: tuple[str, ...] = tuple(f"gas_{name}" for name in SPECIES_NAMES) + (
    "solid_mass",
    "carbon_fraction",
    "energy",
)

RESIDUAL_NORMALIZATION_TEXT = (
    "Cell.residuals returns scaled residuals: gas species except N2 are divided by cell.ref_flow; "
    "N2 is divided by max(abs(inlet_N2)+abs(src_N2), 1e-3); solid_mass by cell.ref_solid; "
    "carbon_fraction by max(char_Xc0, 0.1); energy by max(ref_flow*35*200, 5e5). "
    "residual_cost = 0.5*sum(scaled_residual_i^2); poor-quality threshold is cost > 1e-4."
)


def _get(row: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if hasattr(row, "get"):
        return row.get(key, default)
    return getattr(row, key, default)


def _dry_syngas_kpis(profile: np.ndarray) -> dict[str, float]:
    last = np.asarray(profile[-1], dtype=float)
    gas = last[:8]
    dry = float(np.sum(gas[:7]))
    denom = dry + 1.0e-12
    return {
        "T_out_degC": float(last[10] - 273.15),
        "pred_CO_vol_pct": float(gas[2] / denom * 100.0),
        "pred_H2_vol_pct": float(gas[5] / denom * 100.0),
        "pred_CO2_vol_pct": float(gas[3] / denom * 100.0),
        "pred_dry_flow_kmol_s": dry / 1000.0,
    }


def _top_components(residual: np.ndarray, n: int = 3) -> list[dict[str, Any]]:
    order = np.argsort(-np.abs(residual))[:n]
    out: list[dict[str, Any]] = []
    for idx in order:
        out.append(
            {
                "name": RESIDUAL_NAMES[int(idx)] if int(idx) < len(RESIDUAL_NAMES) else f"residual_{int(idx)}",
                "index": int(idx),
                "value": float(residual[int(idx)]),
                "abs_value": float(abs(residual[int(idx)])),
            }
        )
    return out


def build_cell0_residual_row(model_case_row: Mapping[str, Any], n_cells: int) -> dict[str, Any]:
    """Solve one ModelCase and return one cell-0 residual decomposition row."""
    built = build_gasifier_inputs_from_model_case(model_case_row)
    system = GasifierSystem(built["geometry"], built["coal_props"], legacy_mesh_op_conds(built["op_conds"]))
    inlet = system._initialize_inlet()
    carrier_mapping_ok = bool(inlet.gas_moles[3] > 0.0 and inlet.gas_moles[6] == 0.0)

    start = time.perf_counter()
    profile, _z = system.solve(N_cells=int(n_cells), solver_method="minimize", jacobian_mode="scipy", jax_warmup=False)
    runtime_s = time.perf_counter() - start

    if not system.cells or len(profile) == 0:
        raise RuntimeError("ModelCase solve produced no cells/profile rows.")
    cell_index = 0
    residual = np.asarray(system.cells[cell_index].residuals(np.asarray(profile[cell_index], dtype=float)), dtype=float)
    cost = float(0.5 * np.sum(residual * residual))
    max_abs = float(np.max(np.abs(residual)))
    l2 = float(np.linalg.norm(residual))
    top = _top_components(residual, 3)
    while len(top) < 3:
        top.append({"name": "", "index": -1, "value": np.nan, "abs_value": np.nan})
    kpis = _dry_syngas_kpis(profile)
    targets = built["targets"]
    poor_cells = []
    for idx, (cell, row) in enumerate(zip(system.cells, profile)):
        r = np.asarray(cell.residuals(np.asarray(row, dtype=float)), dtype=float)
        if float(0.5 * np.sum(r * r)) > POOR_CONVERGENCE_COST_THRESHOLD:
            poor_cells.append(idx)

    row = {
        "case_id": built["case_id"],
        "source_candidate_id": built["segment_id"],
        "independence_group": _get(model_case_row, "independence_group_id", _get(model_case_row, "independence_group_id_value")),
        "N_cells": int(n_cells),
        "cell_index": cell_index,
        "runtime_s": runtime_s,
        "residual_cost": cost,
        "l2_residual": l2,
        "max_abs_residual": max_abs,
        "poor_quality_threshold": POOR_CONVERGENCE_COST_THRESHOLD,
        "is_poor_quality": bool(cost > POOR_CONVERGENCE_COST_THRESHOLD),
        "dominant_residual_name": top[0]["name"],
        "dominant_residual_index": top[0]["index"],
        "dominant_residual_value": top[0]["value"],
        "dominant_residual_abs_value": top[0]["abs_value"],
        "fallback_count": int(system.solve_stats.get("fallback_count", 0)),
        "poor_convergence_count": int(system.solve_stats.get("poor_convergence_count", 0)),
        "poor_convergence_cell_index": ";".join(str(i) for i in poor_cells),
        "T_out_degC": kpis["T_out_degC"],
        "pred_CO_vol_pct": kpis["pred_CO_vol_pct"],
        "target_CO_vol_pct": float(targets["outlet_co_dry_vol_pct"]),
        "delta_CO_vol_pct": kpis["pred_CO_vol_pct"] - float(targets["outlet_co_dry_vol_pct"]),
        "pred_H2_vol_pct": kpis["pred_H2_vol_pct"],
        "target_H2_vol_pct": float(targets["outlet_h2_dry_vol_pct"]),
        "delta_H2_vol_pct": kpis["pred_H2_vol_pct"] - float(targets["outlet_h2_dry_vol_pct"]),
        "pred_CO2_vol_pct": kpis["pred_CO2_vol_pct"],
        "target_CO2_vol_pct": float(targets["outlet_co2_dry_vol_pct"]),
        "delta_CO2_vol_pct": kpis["pred_CO2_vol_pct"] - float(targets["outlet_co2_dry_vol_pct"]),
        "pred_dry_flow_kmol_s": kpis["pred_dry_flow_kmol_s"],
        "target_dry_flow_kmol_s": float(targets["syngas_dry_flow_kmol_s"]),
        "delta_dry_flow_kmol_s": kpis["pred_dry_flow_kmol_s"] - float(targets["syngas_dry_flow_kmol_s"]),
        "carrier_mapping_ok": carrier_mapping_ok,
        "transported_moisture_percent": float(built["coal_props"]["Mt"]),
        "residual_normalization": RESIDUAL_NORMALIZATION_TEXT,
    }
    for i, component in enumerate(top, start=1):
        row[f"top{i}_name"] = component["name"]
        row[f"top{i}_index"] = component["index"]
        row[f"top{i}_value"] = component["value"]
        row[f"top{i}_abs_value"] = component["abs_value"]
    return {column: row.get(column) for column in CELL0_RESIDUAL_COLUMNS}


def build_cell0_residual_rows(
    model_case_rows: Iterable[Mapping[str, Any]],
    n_cells_values: Sequence[int] = (1, 3, 5),
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n_cells in n_cells_values:
        for model_case_row in model_case_rows:
            rows.append(build_cell0_residual_row(model_case_row, int(n_cells)))
    return rows


def cell0_residual_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    """Return a compact Markdown diagnostic summary."""
    if not rows:
        return "# Cell-0 residual decomposition diagnostic\n\nNo rows generated.\n"
    n_cells = sorted({int(row["N_cells"]) for row in rows})
    dominant = {}
    for row in rows:
        name = str(row["dominant_residual_name"])
        dominant[name] = dominant.get(name, 0) + 1
    lines = [
        "# Cell-0 residual decomposition diagnostic",
        "",
        "This is a fixed-physics diagnostic artifact, not a solver fix or calibration result.",
        "It is generated from the current corrected ModelCase preview; earlier round-27 "
        "three-case outputs are superseded/not directly comparable after the coal-flow "
        "boundary correction.",
        "",
        f"- Rows: {len(rows)}",
        f"- N_cells coverage: {', '.join(str(n) for n in n_cells)}",
        f"- Poor-quality threshold: residual_cost > {POOR_CONVERGENCE_COST_THRESHOLD:g}",
        f"- Residual cost definition: `0.5 * sum(scaled_residual_i^2)`",
        f"- Residual normalization: {RESIDUAL_NORMALIZATION_TEXT}",
        f"- Dominant residual counts: {', '.join(f'{k}={v}' for k, v in sorted(dominant.items()))}",
        "",
        "| case_id | IG | N | cost | max_abs | top1 | top1_value | top2 | top2_value | top3 | top3_value | poor_cells | dCO | dH2 | dCO2 | d_flow |",
        "|---|---|---:|---:|---:|---|---:|---|---:|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {case_id} | {ig} | {n} | {cost:.6g} | {max_abs:.6g} | {t1} | {v1:.6g} | {t2} | {v2:.6g} | {t3} | {v3:.6g} | {poor} | {dco:.3g} | {dh2:.3g} | {dco2:.3g} | {dflow:.3g} |".format(
                case_id=row["case_id"],
                ig=row["independence_group"],
                n=int(row["N_cells"]),
                cost=float(row["residual_cost"]),
                max_abs=float(row["max_abs_residual"]),
                t1=row["top1_name"],
                v1=float(row["top1_value"]),
                t2=row["top2_name"],
                v2=float(row["top2_value"]),
                t3=row["top3_name"],
                v3=float(row["top3_value"]),
                poor=row["poor_convergence_cell_index"],
                dco=float(row["delta_CO_vol_pct"]),
                dh2=float(row["delta_H2_vol_pct"]),
                dco2=float(row["delta_CO2_vol_pct"]),
                dflow=float(row["delta_dry_flow_kmol_s"]),
            )
        )
    lines.extend(
        [
            "",
            "Interpretation boundary: a persistent cell-0 residual-quality flag means the first-cell "
            "scaled residual is above the recorded threshold; it does not indicate fallback or hidden "
            "solver threshold relaxation. Outlet deltas are included so reviewers can separate local "
            "residual quality from fixed-physics outlet mismatch.",
        ]
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "CELL0_RESIDUAL_COLUMNS",
    "RESIDUAL_NAMES",
    "RESIDUAL_NORMALIZATION_TEXT",
    "build_cell0_residual_row",
    "build_cell0_residual_rows",
    "cell0_residual_markdown",
]
