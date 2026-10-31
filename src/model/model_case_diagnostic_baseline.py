"""
Diagnostic fixed-physics baseline helpers for dataprep ModelCase rows.

These helpers are intentionally reporting-only: they call the existing
``GasifierSystem`` path, then recompute residual quality from the solved cells.
They do not calibrate parameters or adjust solver thresholds.
"""
from __future__ import annotations

import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .gasifier_system import GasifierSystem
from .model_case_adapter import build_gasifier_inputs_from_model_case

POOR_CONVERGENCE_COST_THRESHOLD = 1.0e-4

REQUIRED_DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    "case_id",
    "source_candidate_id",
    "independence_group",
    "N_cells",
    "runtime_s",
    "fallback_count",
    "poor_convergence_count",
    "poor_convergence_cell_index",
    "cell0_cost",
    "cell0_max_abs_residual",
    "T_out_degC",
    "pred_CO_vol_pct",
    "pred_H2_vol_pct",
    "pred_CO2_vol_pct",
    "pred_N2_vol_pct",
    "pred_dry_flow_kmol_s",
    "target_CO_vol_pct",
    "target_H2_vol_pct",
    "target_CO2_vol_pct",
    "target_N2_vol_pct",
    "target_dry_flow_kmol_s",
    "delta_CO_vol_pct",
    "delta_H2_vol_pct",
    "delta_CO2_vol_pct",
    "delta_N2_vol_pct",
    "delta_dry_flow_kmol_s",
    "pred_H2_CO_ratio",
    "target_H2_CO_ratio",
    "delta_H2_CO_ratio",
    "carrier_mapping_ok",
    "transported_moisture_percent",
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
        "pred_N2_vol_pct": float(gas[6] / denom * 100.0),
        "pred_dry_flow_kmol_s": dry / 1000.0,
    }


def cell_residual_quality(system: GasifierSystem, profile: np.ndarray) -> list[dict[str, float]]:
    """Recompute scaled residual quality for each solved cell/profile row."""
    out: list[dict[str, float]] = []
    for idx, (cell, row) in enumerate(zip(system.cells, profile)):
        residual = np.asarray(cell.residuals(np.asarray(row, dtype=float)), dtype=float)
        out.append(
            {
                "cell_index": idx,
                "cost": float(0.5 * np.sum(residual * residual)),
                "max_abs_residual": float(np.max(np.abs(residual))),
                "l2_residual": float(np.linalg.norm(residual)),
            }
        )
    return out


def build_diagnostic_row(model_case_row: Mapping[str, Any], n_cells: int) -> dict[str, Any]:
    """
    Solve one ModelCase and return a flat diagnostic row for CSV/Markdown output.
    """
    built = build_gasifier_inputs_from_model_case(model_case_row)
    system = GasifierSystem(built["geometry"], built["coal_props"], built["op_conds"])

    inlet = system._initialize_inlet()
    carrier_mapping_ok = bool(inlet.gas_moles[3] > 0.0 and inlet.gas_moles[6] == 0.0)
    transported_moisture_percent = float(built["coal_props"]["Mt"])

    start = time.perf_counter()
    profile, _z = system.solve(N_cells=n_cells, solver_method="minimize", jacobian_mode="scipy", jax_warmup=False)
    runtime_s = time.perf_counter() - start

    residual_quality = cell_residual_quality(system, profile)
    poor_cells = [
        item["cell_index"]
        for item in residual_quality
        if item["cost"] > POOR_CONVERGENCE_COST_THRESHOLD
    ]
    cell0 = residual_quality[0] if residual_quality else {}
    kpis = _dry_syngas_kpis(profile)
    targets = built["targets"]

    target_co = float(targets["outlet_co_dry_vol_pct"])
    target_h2 = float(targets["outlet_h2_dry_vol_pct"])
    target_co2 = float(targets["outlet_co2_dry_vol_pct"])
    target_n2 = float(targets.get("outlet_n2_dry_vol_pct", 0.0))
    target_flow = float(targets["syngas_dry_flow_kmol_s"])
    pred_h2_co = kpis["pred_H2_vol_pct"] / max(kpis["pred_CO_vol_pct"], 1.0e-12)
    target_h2_co = target_h2 / max(target_co, 1.0e-12)

    row = {
        "case_id": built["case_id"],
        "source_candidate_id": built["segment_id"],
        "independence_group": _get(model_case_row, "independence_group_id", _get(model_case_row, "independence_group_id_value")),
        "N_cells": int(n_cells),
        "runtime_s": runtime_s,
        "fallback_count": int(system.solve_stats.get("fallback_count", 0)),
        "poor_convergence_count": int(system.solve_stats.get("poor_convergence_count", 0)),
        "poor_convergence_cell_index": ";".join(str(i) for i in poor_cells),
        "cell0_cost": cell0.get("cost"),
        "cell0_max_abs_residual": cell0.get("max_abs_residual"),
        "T_out_degC": kpis["T_out_degC"],
        "pred_CO_vol_pct": kpis["pred_CO_vol_pct"],
        "pred_H2_vol_pct": kpis["pred_H2_vol_pct"],
        "pred_CO2_vol_pct": kpis["pred_CO2_vol_pct"],
        "pred_N2_vol_pct": kpis["pred_N2_vol_pct"],
        "pred_dry_flow_kmol_s": kpis["pred_dry_flow_kmol_s"],
        "target_CO_vol_pct": target_co,
        "target_H2_vol_pct": target_h2,
        "target_CO2_vol_pct": target_co2,
        "target_N2_vol_pct": target_n2,
        "target_dry_flow_kmol_s": target_flow,
        "delta_CO_vol_pct": kpis["pred_CO_vol_pct"] - target_co,
        "delta_H2_vol_pct": kpis["pred_H2_vol_pct"] - target_h2,
        "delta_CO2_vol_pct": kpis["pred_CO2_vol_pct"] - target_co2,
        "delta_N2_vol_pct": kpis["pred_N2_vol_pct"] - target_n2,
        "delta_dry_flow_kmol_s": kpis["pred_dry_flow_kmol_s"] - target_flow,
        "pred_H2_CO_ratio": pred_h2_co,
        "target_H2_CO_ratio": target_h2_co,
        "delta_H2_CO_ratio": pred_h2_co - target_h2_co,
        "carrier_mapping_ok": carrier_mapping_ok,
        "transported_moisture_percent": transported_moisture_percent,
    }
    return {column: row.get(column) for column in REQUIRED_DIAGNOSTIC_COLUMNS}


def build_diagnostic_rows(model_case_rows: Iterable[Mapping[str, Any]], n_cells_values: Sequence[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n_cells in n_cells_values:
        for model_case_row in model_case_rows:
            rows.append(build_diagnostic_row(model_case_row, int(n_cells)))
    return rows


def markdown_summary(rows: Sequence[Mapping[str, Any]]) -> str:
    """Return a compact Markdown summary for the generated diagnostic table."""
    lines = [
        "# Pilot fixed-physics diagnostic baseline",
        "",
        "This is a diagnostic smoke/baseline artifact, not calibrated accuracy evidence.",
        "",
        f"- Rows: {len(rows)}",
        f"- N_cells coverage: {', '.join(str(n) for n in sorted({int(r['N_cells']) for r in rows}))}",
        "- Solver path: GasifierSystem.solve(solver_method='minimize', jacobian_mode='scipy')",
        "- Poor convergence threshold recorded, not relaxed: cost > 1e-4",
        "",
        "| case_id | IG | N | runtime_s | poor_cells | T_out_C | CO | H2 | CO2 | dry_flow | dCO | dH2 | dCO2 | d_flow |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {case_id} | {ig} | {n} | {runtime:.3f} | {poor} | {t:.2f} | {co:.2f} | {h2:.2f} | {co2:.2f} | {flow:.3f} | {dco:.2f} | {dh2:.2f} | {dco2:.2f} | {dflow:.3f} |".format(
                case_id=row["case_id"],
                ig=row["independence_group"],
                n=int(row["N_cells"]),
                runtime=float(row["runtime_s"]),
                poor=row["poor_convergence_cell_index"],
                t=float(row["T_out_degC"]),
                co=float(row["pred_CO_vol_pct"]),
                h2=float(row["pred_H2_vol_pct"]),
                co2=float(row["pred_CO2_vol_pct"]),
                flow=float(row["pred_dry_flow_kmol_s"]),
                dco=float(row["delta_CO_vol_pct"]),
                dh2=float(row["delta_H2_vol_pct"]),
                dco2=float(row["delta_CO2_vol_pct"]),
                dflow=float(row["delta_dry_flow_kmol_s"]),
            )
        )
    lines.extend(
        [
            "",
            "Evidence boundary: reproducible pilot fixed-physics diagnostic baseline. "
            "Do not interpret as calibrated validation or industrial generalization.",
        ]
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "POOR_CONVERGENCE_COST_THRESHOLD",
    "REQUIRED_DIAGNOSTIC_COLUMNS",
    "build_diagnostic_row",
    "build_diagnostic_rows",
    "cell_residual_quality",
    "markdown_summary",
]
