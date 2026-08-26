"""Shared helpers for the 2026-08-26 revision evidence package.

These helpers freeze dataset intake, solver-option application, residual-quality
status labels and artifact hashing. They do not change manuscript claims.
"""
from __future__ import annotations

import hashlib
import math
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .gasifier_system import GasifierSystem
from .model_case_adapter import NORMAL_M3_PER_KMOL, build_gasifier_inputs_from_model_case

DATASET_ID = "R52-C-75"
POOR_CONVERGENCE_COST_THRESHOLD = 1.0e-4
SEED = 20260826
DATA_PREP_ROOT = Path("/Users/liuzhen/AI-projects/gasifier-data-prep")
ROUND52_DIR = DATA_PREP_ROOT / "outputs" / "round52"
MODEL_ROOT = Path(__file__).resolve().parents[2]

TARGETS = (
    ("CO_dry_vol_pct", "co_vol_pct", "pred_CO_vol_pct"),
    ("H2_dry_vol_pct", "h2_vol_pct", "pred_H2_vol_pct"),
    ("CO2_dry_vol_pct", "co2_vol_pct", "pred_CO2_vol_pct"),
    ("H2_CO_ratio", "h2_co_ratio", "pred_H2_CO_ratio"),
    ("dry_syngas_flow", "syngas_flow_mean", "pred_dry_flow_nm3_h"),
)

EVIDENCE_BOUNDARY = (
    "revision evidence only; not production control, learned multi-alpha, "
    "z_d value, oxidation-channel updating, cross-gasifier generalization, "
    "or manuscript claim text"
)


def source_commit(repo: Path | None = None) -> str:
    root = repo or MODEL_ROOT
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def residual_quality_status(cell0_cost: float) -> str:
    if not math.isfinite(float(cell0_cost)):
        return "unknown"
    if float(cell0_cost) > POOR_CONVERGENCE_COST_THRESHOLD:
        return "poor"
    return "acceptable"


def execution_status_from_exception(exc: BaseException | None) -> str:
    if exc is None:
        return "completed"
    return f"error:{type(exc).__name__}"


def write_sha256_manifest(paths: Sequence[Path], dest: Path) -> Path:
    lines: list[str] = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.name}")
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest


def _read_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def coal_assumption_paths() -> dict[str, Path]:
    return {
        "v0_baseline_coal_assumption": DATA_PREP_ROOT / "config" / "coal_assumption_v0.yaml",
        "huayi_p2_sy3sy2_coal_assumption": DATA_PREP_ROOT / "config" / "coal_assumption_huayi_p2.yaml",
        "huayi_p3_sy3sy2_coal_assumption": DATA_PREP_ROOT / "config" / "coal_assumption_huayi_p3.yaml",
        "huayi_p4_sy3sy2_coal_assumption": DATA_PREP_ROOT / "config" / "coal_assumption_huayi_p4.yaml",
    }


def load_frozen_split() -> tuple[bool, list[dict[str, Any]], pd.DataFrame]:
    split_manifest = _read_yaml(ROUND52_DIR / "huayi_round52_split_freeze.yaml")
    inventory = pd.read_csv(ROUND52_DIR / "huayi_round52_window_inventory.csv")
    primary = inventory[inventory["dataset_id"] == DATASET_ID].copy()
    expected = {role: set(split_manifest["splits"]["row_ids"][role]) for role in ("train", "validation", "test")}
    frozen_consumed = all(
        set(primary.loc[primary["split_role"] == role, "window_id"]) == ids
        for role, ids in expected.items()
    )
    return frozen_consumed, model_case_rows(primary), primary


def model_case_rows(primary: pd.DataFrame) -> list[dict[str, Any]]:
    coal_by_id = {key: _read_yaml(path) for key, path in coal_assumption_paths().items()}
    rows: list[dict[str, Any]] = []
    for _, row in primary.sort_values(["phase", "start_time", "window_id"]).iterrows():
        coal = coal_by_id[str(row["coal_assumption_id"])]
        ultimate = coal["ultimate_analysis_dry"]
        proximate = coal["proximate_analysis"]
        co2_carrier_nm3_h = max(float(row["o2_flow_mean"]) * 0.24, 0.0)
        rows.append(
            {
                "case_id": row["window_id"],
                "case_id_value": row["window_id"],
                "segment_id": row["parent_segment_id"],
                "independence_group_id": row["parent_independence_group_id"],
                "feature_version": row["coal_assumption_id"],
                "model_case_ready": bool(row["model_case_ready"]),
                "phase": f"phase{int(row['phase'])}",
                "phase_number": int(row["phase"]),
                "dataset_id": row["dataset_id"],
                "split": str(row["split_role"]),
                "split_role": str(row["split_role"]),
                "window_id": row["window_id"],
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "reactor_bed_diameter_m_value": 3.8,
                "reactor_bed_height_m_value": 10.3,
                "coal_carbon_mass_fraction_adb_value": ultimate["carbon"]["value"],
                "coal_hydrogen_mass_fraction_adb_value": ultimate["hydrogen"]["value"],
                "coal_oxygen_mass_fraction_adb_value": ultimate["oxygen"]["value"],
                "coal_nitrogen_mass_fraction_adb_value": ultimate["nitrogen"]["value"],
                "coal_sulfur_mass_fraction_adb_value": ultimate["sulfur"]["value"],
                "coal_ash_proximate_d_value": proximate["ash_d"]["value"],
                "coal_volatile_matter_d_value": proximate["volatile_matter_d"]["value"],
                "coal_fixed_carbon_d_value": proximate["fixed_carbon_d"]["value"],
                "coal_heating_value_dry_kj_kg_value": coal["heating_value_dry_kj_kg"]["value"],
                "coal_mass_flow_value": float(row["coal_flow_mean"]),
                "o2_volumetric_flow_value": float(row["o2_flow_mean"]),
                "co2_carrier_volumetric_flow_value": co2_carrier_nm3_h,
                "gasifier_pressure_value": float(row["pressure_mean"]),
                "o2_inlet_temperature_c_value": 160.0,
                "co2_carrier_inlet_temperature_c_value": 80.0,
                "syngas_dry_volumetric_flow_value": float(row["syngas_flow_mean"]),
                "outlet_co_dry_vol_pct_value": float(row["co_vol_pct"]),
                "outlet_h2_dry_vol_pct_value": float(row["h2_vol_pct"]),
                "outlet_co2_dry_vol_pct_value": float(row["co2_vol_pct"]),
                "outlet_ch4_dry_vol_pct_value": float(row.get("ch4_vol_pct", 0.0)),
                "outlet_n2_dry_vol_pct_value": 0.0,
                "outlet_h2s_dry_vol_pct_value": 0.0,
                "co_vol_pct": float(row["co_vol_pct"]),
                "h2_vol_pct": float(row["h2_vol_pct"]),
                "co2_vol_pct": float(row["co2_vol_pct"]),
                "h2_co_ratio": float(row["h2_co_ratio"]),
                "syngas_flow_mean": float(row["syngas_flow_mean"]),
                "o_c_ratio": float(row["o_c_ratio"]),
                "coal_flow_mean": float(row["coal_flow_mean"]),
                "pressure_mpa": float(row["pressure_mean"]),
                "carrier_to_coal_ratio": co2_carrier_nm3_h / max(float(row["coal_flow_mean"]), 1.0e-12),
            }
        )
    return rows


def apply_solver_options(op_conds: Mapping[str, Any], options: Mapping[str, Any] | None) -> dict[str, Any]:
    out = dict(op_conds)
    if not options:
        return out
    mapping = {
        "first_cell_length": "FirstCellLength",
        "ignition_zone_res": "IgnitionZoneRes",
        "cell0_guess_temperatures": "Cell0GuessTemperatures",
        "cell0_guess_mode": "Cell0GuessMode",
        "solver_xtol": "SolverXtol",
        "solver_ftol": "SolverFtol",
        "solver_max_nfev": "SolverMaxNfev",
    }
    for src, dest in mapping.items():
        if src in options and options[src] is not None:
            out[dest] = options[src]
    return out


def _dry_syngas_kpis(profile: np.ndarray) -> dict[str, float]:
    last = np.asarray(profile[-1], dtype=float)
    gas = last[:8]
    dry = float(np.sum(gas[:7]))
    denom = dry + 1.0e-12
    co = float(gas[2] / denom * 100.0)
    h2 = float(gas[5] / denom * 100.0)
    return {
        "T_out_degC": float(last[10] - 273.15),
        "pred_CO_vol_pct": co,
        "pred_H2_vol_pct": h2,
        "pred_CO2_vol_pct": float(gas[3] / denom * 100.0),
        "pred_H2_CO_ratio": h2 / max(co, 1.0e-12),
        "pred_dry_flow_kmol_s": dry / 1000.0,
        "pred_dry_flow_nm3_h": dry / 1000.0 * NORMAL_M3_PER_KMOL * 3600.0,
    }


def solve_model_case(
    row: Mapping[str, Any],
    *,
    n_cells: int,
    alpha_wgs: float = 1.0,
    solver_options: Mapping[str, Any] | None = None,
    solver_method: str = "minimize",
) -> dict[str, Any]:
    built = build_gasifier_inputs_from_model_case(row)
    op_conds = apply_solver_options(built["op_conds"], solver_options)
    op_conds["WGS_CatalyticFactor"] = float(alpha_wgs)
    system = GasifierSystem(built["geometry"], built["coal_props"], op_conds)
    profile, _z = system.solve(
        N_cells=int(n_cells),
        solver_method=solver_method,
        jacobian_mode="scipy",
        jax_warmup=False,
    )
    kpis = _dry_syngas_kpis(profile)
    cell0 = system.cells[0]
    components = cell0.residual_components(np.asarray(profile[0], dtype=float))
    poor_cells = []
    for idx, (cell, state_row) in enumerate(zip(system.cells, profile)):
        residual = np.asarray(cell.residuals(np.asarray(state_row, dtype=float)), dtype=float)
        if float(0.5 * np.sum(residual * residual)) > POOR_CONVERGENCE_COST_THRESHOLD:
            poor_cells.append(idx)
    return {
        "execution_status": "completed",
        "residual_quality_status": residual_quality_status(components["cost"]),
        "fallback_count": int(system.solve_stats.get("fallback_count", 0)),
        "poor_convergence_count": int(system.solve_stats.get("poor_convergence_count", 0)),
        "poor_convergence_cell_index": ";".join(str(i) for i in poor_cells),
        "cell0_cost": float(components["cost"]),
        "cell0_max_abs_scaled_residual": float(components["max_abs_scaled"]),
        "cell0_max_abs_unscaled_residual": float(components["max_abs_unscaled"]),
        "cell0_dominant_scaled_name": components["dominant_scaled_name"],
        "cell0_dominant_unscaled_name": components["dominant_unscaled_name"],
        "cell0_dominant_scaled_value": float(components["dominant_scaled_value"]),
        "cell0_dominant_unscaled_value": float(components["dominant_unscaled_value"]),
        "N_cells": int(n_cells),
        "alpha_wgs": float(alpha_wgs),
        "residual_names": list(components["names"]),
        "residual_scaled": [float(v) for v in components["scaled"]],
        "residual_unscaled": [float(v) for v in components["unscaled"]],
        **kpis,
    }


def metric_rows(
    predictions: pd.DataFrame,
    *,
    method: str,
    source_commit_hash: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (split_role, target), group in predictions.groupby(["split_role", "target"], dropna=False):
        attempted = int(group["window_id"].nunique())
        completed = group[group["execution_status"] == "completed"]
        n_completed = int(completed["window_id"].nunique())
        n_timeout = int(group[group["execution_status"] == "timeout"]["window_id"].nunique())
        n_error = int(group[group["execution_status"].astype(str).str.startswith("error")]["window_id"].nunique())
        if n_completed:
            residual = completed["residual"].astype(float).to_numpy()
            rmse = float(np.sqrt(np.mean(residual * residual)))
            mae = float(np.mean(np.abs(residual)))
            bias = float(np.mean(residual))
        else:
            rmse = mae = bias = math.nan
        rows.append(
            {
                "dataset_id": DATASET_ID,
                "split_role": split_role,
                "method": method,
                "target": target,
                "rmse": rmse,
                "mae": mae,
                "bias": bias,
                "n_attempted": attempted,
                "n_completed": n_completed,
                "n_timeout": n_timeout,
                "n_error": n_error,
                "execution_status": "completed" if attempted == n_completed else "partial",
                "source_commit": source_commit_hash,
            }
        )
    return pd.DataFrame(rows).sort_values(["method", "split_role", "target"]).reset_index(drop=True)


def prediction_records(
    row: Mapping[str, Any],
    outcome: Mapping[str, Any],
    *,
    method: str,
    configuration: str,
    source_commit_hash: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for target, true_col, pred_col in TARGETS:
        y_true = float(row[true_col])
        y_pred = float(outcome[pred_col]) if outcome.get("execution_status") == "completed" and pred_col in outcome else math.nan
        records.append(
            {
                "dataset_id": row["dataset_id"],
                "window_id": row["window_id"],
                "phase": row["phase_number"],
                "split_role": row["split_role"],
                "method": method,
                "configuration": configuration,
                "N_cells": outcome.get("N_cells", math.nan),
                "target": target,
                "target_value": y_true,
                "predicted_value": y_pred,
                "residual": y_pred - y_true if math.isfinite(y_pred) else math.nan,
                "alpha_wgs": outcome.get("alpha_wgs", math.nan),
                "poor_convergence_count": outcome.get("poor_convergence_count", math.nan),
                "cell0_cost": outcome.get("cell0_cost", math.nan),
                "cell0_max_abs_scaled_residual": outcome.get("cell0_max_abs_scaled_residual", math.nan),
                "cell0_max_abs_unscaled_residual": outcome.get("cell0_max_abs_unscaled_residual", math.nan),
                "execution_status": outcome.get("execution_status", "unknown"),
                "residual_quality_status": outcome.get("residual_quality_status", "unknown"),
                "source_commit": source_commit_hash,
            }
        )
    return records


def objective_from_predictions(predictions: pd.DataFrame, split_role: str) -> float:
    work = predictions[(predictions["split_role"] == split_role) & (predictions["execution_status"] == "completed")].copy()
    scales = {
        "CO_dry_vol_pct": 10.0,
        "H2_dry_vol_pct": 10.0,
        "CO2_dry_vol_pct": 10.0,
        "H2_CO_ratio": 0.1,
        "dry_syngas_flow": 1.0e5,
    }
    total = 0.0
    count = 0
    for target, group in work.groupby("target"):
        residual = group["residual"].astype(float).to_numpy() / scales[str(target)]
        total += float(np.sqrt(np.mean(residual * residual)))
        count += 1
    return total / max(count, 1)


def iter_windows(rows: Sequence[Mapping[str, Any]], window_ids: Iterable[str]) -> list[dict[str, Any]]:
    wanted = set(window_ids)
    return [dict(row) for row in rows if str(row["window_id"]) in wanted]
