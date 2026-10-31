"""P1.5-A / P1.5-C / P2-A 共用执行层（2026-10-30）。

只做证据生产：不调参、不改动力学常数、不改氢气限幅。
所有论文轮次统一使用 ``MeshFamily="uniform_full_length"``、``SolverStrictRoot=True``、
``ParticleSizeModel="fortran_constant"``（P1.5-A 起）。
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .gasifier_system import GasifierSystem
from .model_case_adapter import NORMAL_M3_PER_KMOL, build_gasifier_inputs_from_model_case
from .p0_round_20260930 import (
    ELEMENT_KG_PER_MOL,
    ELEMENTS,
    H2O_MOLAR_MASS_G_PER_MOL,
    atoms_from_gas,
    reactor_element_flows,
    silence_solver_logs,
)
from .revision_evidence_common import source_commit, write_sha256_manifest

MESH_FAMILY = "uniform_full_length"
PARTICLE_MODEL_MAIN = "fortran_constant"
PARTICLE_MODEL_DIAG = "shrinking_legacy"

# 论文轮次固定的求解选项（新增开关，不改任何物理常数）
PAPER_OP_CONDS: dict[str, Any] = {
    "MeshFamily": MESH_FAMILY,
    "ParticleSizeModel": PARTICLE_MODEL_MAIN,
    "SolverStrictRoot": True,
    "AllowTruncatedMesh": False,
}

LIMIT_KEYS = (
    "C_O2_LimitedByO2",
    "C_H2O_LimitedByH2O",
    "C_CO2_LimitedByCO2",
    "C_H2_LimitedByH2",
    "WGS_NegativeTruncated",
    "MSR_Limited",
    "H2Ox_AtCap099",
)


def paper_op_conds(
    built: Mapping[str, Any],
    *,
    particle_model: str = PARTICLE_MODEL_MAIN,
    mesh_family: str = MESH_FAMILY,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    op = dict(built["op_conds"])
    op.update(PAPER_OP_CONDS)
    op["MeshFamily"] = mesh_family
    op["ParticleSizeModel"] = particle_model
    if extra:
        op.update(extra)
    return op


def dry_flow_nm3_h(gas_moles: np.ndarray) -> float:
    dry = float(np.sum(np.asarray(gas_moles)[:7]))
    return dry / 1000.0 * NORMAL_M3_PER_KMOL * 3600.0


def dry_vol_pct(gas_moles: np.ndarray, index: int) -> float:
    gas = np.asarray(gas_moles)[:7]
    return float(gas[index] / max(float(np.sum(gas)), 1e-12) * 100.0)


def _combustion_end_z(system: GasifierSystem) -> float:
    """燃烧区末端位置：最后一个入口 pO2 超过阈值的单元的下游边界。"""
    z_end = 0.0
    for diag in system.cell_diagnostics:
        if diag.get("is_combustion_zone"):
            z_end = float(sum(d["dz_m"] for d in system.cell_diagnostics[: diag["cell_index"] + 1]))
    return z_end


def _limit_length_fractions(system: GasifierSystem, L: float) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in LIMIT_KEYS:
        active = sum(float(d["dz_m"]) for d in system.cell_diagnostics if d.get(key))
        out[f"limit_len_frac_{key}"] = active / max(L, 1e-12)
    return out


def _winning_guess(system: GasifierSystem) -> tuple[float, float]:
    branches = [b for b in getattr(system, "cell0_branches", []) if np.isfinite(b.get("cost", np.nan))]
    if not branches:
        return float("nan"), float("nan")
    best = min(branches, key=lambda b: b["cost"])
    return float(best["guess_T"]), float(best["cost"])


def run_case(
    row: Mapping[str, Any],
    *,
    n_cells: int,
    particle_model: str = PARTICLE_MODEL_MAIN,
    mesh_family: str = MESH_FAMILY,
    extra_op: Mapping[str, Any] | None = None,
    solver_method: str = "minimize",
    keep_system: bool = False,
) -> dict[str, Any]:
    """求解一个窗口，返回 P2-A 所需的逐行指标。"""
    built = build_gasifier_inputs_from_model_case(row)
    op = paper_op_conds(built, particle_model=particle_model, mesh_family=mesh_family, extra=extra_op)
    system = GasifierSystem(built["geometry"], built["coal_props"], op)
    t0 = time.perf_counter()
    profile, _z = system.solve(
        N_cells=int(n_cells), solver_method=solver_method, jacobian_mode="scipy", jax_warmup=False
    )
    wall = time.perf_counter() - t0

    L = float(built["geometry"]["L"])
    C_inlet = float(system.W_dry * (float(system.coal_props.get("Cd", 0.0)) / 100.0) / ELEMENT_KG_PER_MOL["C"])
    rec: dict[str, Any] = {
        "window_id": str(row["window_id"]),
        "phase": f"phase{int(row.get('phase_number', 0))}",
        "split_role": str(row.get("split_role", "")),
        "N_cells": int(n_cells),
        "ParticleSizeModel": particle_model,
        "MeshFamily": mesh_family,
        "sum_dz_over_L": float(system.mesh_sum_dz / L),
        "execution_status": str(system.execution_status),
        "no_root_cell_index": system.no_root_cell_index,
        "n_cells_solved": len(system.results),
        "n_cells_above_1e-4": int(system.n_cells_above_cost_threshold),
        "max_residual_cost": float(system.max_residual_cost),
        "poor_convergence_count": int(system.solve_stats.get("poor_convergence_count", 0)),
        "fallback_count": int(system.solve_stats.get("fallback_count", 0)),
        "cell0_n_distinct_roots": int(system.cell0_n_distinct_roots),
        "cell0_winning_guess_T": _winning_guess(system)[0],
        "cell0_winning_cost": _winning_guess(system)[1],
        "z_comb_end_m": _combustion_end_z(system),
        "C_inlet_coal_mol_s": C_inlet,
        "solve_wall_time_s": wall,
        "plant_CO_vol_pct": float(row.get("co_vol_pct", np.nan)),
        "plant_H2_vol_pct": float(row.get("h2_vol_pct", np.nan)),
        "plant_CO2_vol_pct": float(row.get("co2_vol_pct", np.nan)),
        "plant_dry_flow_nm3_h": float(row.get("syngas_flow_mean", np.nan)),
    }
    rec.update(_limit_length_fractions(system, L))

    if system.execution_status == "completed" and len(system.results) > 0:
        out = system.results[-1]
        gas = np.asarray(out.gas_moles, dtype=float)
        rec.update(
            {
                "T_out_K": float(out.T),
                "T_out_degC": float(out.T - 273.15),
                "CO_vol_pct": dry_vol_pct(gas, 2),
                "H2_vol_pct": dry_vol_pct(gas, 5),
                "CO2_vol_pct": dry_vol_pct(gas, 3),
                "CH4_vol_pct": dry_vol_pct(gas, 1),
                "dry_flow_nm3_h": dry_flow_nm3_h(gas),
                "outlet_solid_carbon_mol_s": float(out.solid_mass * out.carbon_fraction / ELEMENT_KG_PER_MOL["C"]),
            }
        )
        rec["X_C_implied"] = float(
            1.0 - rec["outlet_solid_carbon_mol_s"] / max(rec["C_inlet_coal_mol_s"], 1e-12)
        )
        rec["dry_flow_ratio"] = float(
            rec["dry_flow_nm3_h"] / max(rec["plant_dry_flow_nm3_h"], 1e-12)
        )
        flows = reactor_element_flows(system, out)
        rec["element_species_identity_max_abs_mol_s"] = float(
            max(abs(flows[f"{el}_closure_error_mol_s"]) for el in ELEMENTS)
        )
        rec["S_error_mol_s"] = float(flows["S_closure_error_mol_s"])
        rec["S_closure_fraction"] = float(flows["S_closure_fraction"])
    else:
        rec.update(
            {
                "T_out_K": np.nan,
                "T_out_degC": np.nan,
                "CO_vol_pct": np.nan,
                "H2_vol_pct": np.nan,
                "CO2_vol_pct": np.nan,
                "CH4_vol_pct": np.nan,
                "dry_flow_nm3_h": np.nan,
                "outlet_solid_carbon_mol_s": np.nan,
                "X_C_implied": np.nan,
                "dry_flow_ratio": np.nan,
                "element_species_identity_max_abs_mol_s": np.nan,
                "S_error_mol_s": np.nan,
            }
        )
    if keep_system:
        rec["_system"] = system
        rec["_profile"] = profile
    return rec


# ---------------------------------------------------------------------------
# P1.5-A 检查
# ---------------------------------------------------------------------------

def mesh_check_rows(L: float = 10.3, n_values: Sequence[int] = (1, 2, 3, 5, 9, 15, 20, 21, 31, 40, 80, 160)):
    from .grid_service import AdaptiveMeshGenerator, MeshConfig

    rows = []
    for family in ("legacy", MESH_FAMILY):
        for N in n_values:
            dz, _z = AdaptiveMeshGenerator(
                MeshConfig(
                    total_length=L,
                    n_cells=N,
                    ignition_zone_length=0.1,
                    ignition_zone_res=0.1,
                    min_grid_size=0.001,
                    mesh_family=family,
                )
            ).generate()
            total = float(np.sum(dz))
            rows.append(
                {
                    "mesh_family": family,
                    "N_cells": int(N),
                    "L_m": L,
                    "sum_dz_m": total,
                    "sum_dz_over_L": total / L,
                    "dz_min_m": float(np.min(dz)),
                    "dz_max_m": float(np.max(dz)),
                    "assert_pass_without_override": bool(abs(total - L) <= 1e-9 * L),
                    "requires_allow_truncated": bool(abs(total - L) > 1e-9 * L),
                }
            )
    return rows


JANAF_CHECK = {
    "CO": (33.18, 38.85),
    "CO2": (54.31, 61.71),
    "H2": (30.20, 36.29),
    "H2O": (41.27, 48.15),
    "N2": (32.70, 38.40),
    "O2": (34.87, 40.60),
}


def thermo_check_rows():
    from .physics import SHOMATE_DB, calculate_cp, get_enthalpy_molar

    rows = []
    for species, data in SHOMATE_DB.items():
        for cut in data["T_cuts"]:
            h_jump = get_enthalpy_molar(species, cut - 1e-6) - get_enthalpy_molar(species, cut + 1e-6)
            cp_low = calculate_cp(species, cut - 1e-6)
            cp_high = calculate_cp(species, cut + 1e-6)
            rows.append(
                {
                    "check": "enthalpy_jump",
                    "species": species,
                    "T_K": float(cut),
                    "value": float(h_jump),
                    "unit": "J/mol",
                    "tolerance": 20.0,
                    "pass": bool(abs(h_jump) <= 20.0),
                }
            )
            rows.append(
                {
                    "check": "cp_jump",
                    "species": species,
                    "T_K": float(cut),
                    "value": float(cp_high - cp_low),
                    "unit": "J/mol/K",
                    "tolerance": 1.0,
                    "pass": bool(abs(cp_high - cp_low) <= 1.0),
                }
            )
        if species in JANAF_CHECK:
            cp_ref, dh_ref = JANAF_CHECK[species]
            cp_calc = calculate_cp(species, 1000.0)
            dh_calc = (get_enthalpy_molar(species, 1500.0) - get_enthalpy_molar(species, 298.15)) / 1000.0
            rows.append(
                {
                    "check": "cp_at_1000K_vs_JANAF",
                    "species": species,
                    "T_K": 1000.0,
                    "value": float(cp_calc),
                    "reference": float(cp_ref),
                    "rel_dev": float((cp_calc - cp_ref) / cp_ref),
                    "unit": "J/mol/K",
                    "tolerance": 0.002,
                    "pass": bool(abs((cp_calc - cp_ref) / cp_ref) <= 0.002),
                }
            )
            rows.append(
                {
                    "check": "dH_1500_298_vs_JANAF",
                    "species": species,
                    "T_K": 1500.0,
                    "value": float(dh_calc),
                    "reference": float(dh_ref),
                    "rel_dev": float((dh_calc - dh_ref) / dh_ref),
                    "unit": "kJ/mol",
                    "tolerance": 0.002,
                    "pass": bool(abs((dh_calc - dh_ref) / dh_ref) <= 0.002),
                }
            )
    return rows


def particle_area_rows(system: GasifierSystem, profile: np.ndarray) -> list[dict[str, Any]]:
    """同一状态下两种颗粒模型的 S_total 比值与其理论值。"""
    rows = []
    for idx, (cell, state_row) in enumerate(zip(system.cells, np.asarray(profile, dtype=float))):
        from .state import StateVector

        state = StateVector.from_array(state_row, P=cell.inlet.P, z=cell.z)
        C_fed = cell.coal_flow_dry * (cell.coal_props.get("Cd", 60.0) / 100.0)
        old = cell.op_conds
        try:
            cell.op_conds = dict(old)
            cell.op_conds["ParticleSizeModel"] = PARTICLE_MODEL_MAIN
            phys_f = cell._calc_physics_props(state, C_fed)
            cell.op_conds["ParticleSizeModel"] = PARTICLE_MODEL_DIAG
            phys_s = cell._calc_physics_props(state, C_fed)
        finally:
            cell.op_conds = old
        from .input_contract import ash_mass_fraction_dry

        Ash_d = ash_mass_fraction_dry(cell.coal_props)
        Cd = float(cell.coal_props.get("Cd", 60.0)) / 100.0
        f_ash = Ash_d / (Cd + Ash_d + 1e-9)
        X = float(phys_f["X_total"])
        theory = (f_ash + (1.0 - f_ash) * (1.0 - X)) ** (-2.0 / 3.0)
        ratio = float(phys_f["S_total"] / max(phys_s["S_total"], 1e-12))
        rows.append(
            {
                "cell_index": idx,
                "X_total": X,
                "f_ash": float(f_ash),
                "S_total_fortran_constant_m2": float(phys_f["S_total"]),
                "S_total_shrinking_legacy_m2": float(phys_s["S_total"]),
                "ratio_fortran_over_shrinking": ratio,
                "ratio_theory": float(theory),
                "rel_dev": float((ratio - theory) / theory),
                "d_p_fortran_m": float(phys_f["d_p"]),
                "d_p_shrinking_m": float(phys_s["d_p"]),
                "rho_p_fortran_kg_m3": float(phys_f["rho_p"]),
                "pass": bool(abs((ratio - theory) / theory) <= 1e-6),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 产物写入
# ---------------------------------------------------------------------------

def write_round_outputs(
    out_dir: Path,
    *,
    tables: Mapping[str, pd.DataFrame],
    protocol: Mapping[str, Any],
    interpretation: str,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, df in tables.items():
        path = out_dir / name
        df.to_csv(path, index=False)
        written.append(path)
    protocol_path = out_dir / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    written.append(protocol_path)
    interp_path = out_dir / "interpretation.md"
    interp_path.write_text(interpretation, encoding="utf-8")
    written.append(interp_path)
    sha_path = out_dir / "sha256.txt"
    write_sha256_manifest(written, sha_path)
    written.append(sha_path)
    return written


def base_protocol(extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    protocol: dict[str, Any] = {
        "round": "P1.5-A / 2026-10-30",
        "model_repo": "/Users/liuzhen/AI-projects/gasifier-1d-kinetic",
        "source_commit": source_commit(),
        "mesh_family": MESH_FAMILY,
        "particle_size_model": PARTICLE_MODEL_MAIN,
        "solver_strict_root": True,
        "frozen_items": [
            "all Arrhenius constants",
            "phi",
            "f_co2 = 1.0",
            "P_O2_Combustion_atm = 0.05",
            "CharCombustionRateFactor = 0.3",
            "H2_OxidationFactor",
            "CO_OxidationFactor",
            "pyrolysis yields",
            "hydrogen post-oxidation limiting",
            "USE_RK_GILL_COMBUSTION = False",
        ],
    }
    if extra:
        protocol.update(extra)
    return protocol


__all__ = [
    "MESH_FAMILY",
    "PARTICLE_MODEL_MAIN",
    "PARTICLE_MODEL_DIAG",
    "LIMIT_KEYS",
    "paper_op_conds",
    "run_case",
    "mesh_check_rows",
    "thermo_check_rows",
    "particle_area_rows",
    "write_round_outputs",
    "base_protocol",
    "silence_solver_logs",
]
