"""P0 数值诊断：方程残差原始量纲、反应器闭合，以及 N=1 复现锁。

不改平衡方程，不放宽 residual_cost > 1e-4。
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .gasifier_system import GasifierSystem
from .material import SPECIES_NAMES, MaterialService
from .model_case_adapter import build_gasifier_inputs_from_model_case
from .revision_evidence_common import (
    DATASET_ID,
    POOR_CONVERGENCE_COST_THRESHOLD,
    apply_solver_options,
    load_frozen_split,
    source_commit,
    write_sha256_manifest,
)
from .state import StateVector

PUBLISHED_TEST_CO_RMSE = 18.7319
REPRODUCTION_TOLERANCE = 0.05
TERM_IDENTITY_ATOL_MOL_S = 1.0e-6
COST_METADATA_ATOL = 1.0e-6
HANDOFF_MODEL_HEAD = "98ad47c"

ELEMENT_KG_PER_MOL = {
    "C": 0.012011,
    "H": 0.001008,
    "O": 0.015999,
    "N": 0.028013 / 2.0,
    "S": 0.03206,
}
ELEMENTS = ("C", "H", "O", "N", "S")
# 与 GasifierSystem._initialize_inlet 中蒸发摩尔流量同一式子
H2O_MOLAR_MASS_G_PER_MOL = 18.015

RESIDUAL_UNITS = {
    **{f"gas_{name}": "mol/s" for name in SPECIES_NAMES},
    "solid_mass": "kg/s",
    "carbon_fraction": "kg/s",
    "energy": "W",
}
SCALE_UNITS = {
    **{f"gas_{name}": "mol/s" for name in SPECIES_NAMES},
    "solid_mass": "kg/s",
    "carbon_fraction": "1",
    "energy": "W",
}

# 气相残差 = (F_out - F_in) - inlet_source - reaction_source
# inlet_source 是热解/蒸发源项，reaction_source 是化学计量净生成，二者单位都是 mol/s。


def silence_solver_logs() -> None:
    logging.getLogger("model").setLevel(logging.ERROR)
    logging.getLogger().setLevel(logging.ERROR)


def reaction_source_mol_s(cell, r_het: Mapping[str, float], r_homo: Mapping[str, float], phi: float) -> np.ndarray:
    """与 Cell._calc_balances 气相式中的反应项相同，正值表示净生成。"""
    f_co2 = float(cell.op_conds.get("Combustion_CO2_Fraction", 1.0))
    r1 = float(r_homo["CO_Ox"])
    r2 = float(r_homo["H2_Ox"])
    r3 = float(r_homo["WGS"])
    r4 = float(r_homo["RWGS"])
    r5 = float(r_homo["CH4_Ox"])
    r6 = float(r_homo["MSR"])
    r_comb = float(r_het["C+O2"])
    r_h2o = float(r_het["C+H2O"])
    r_co2 = float(r_het["C+CO2"])
    r_h2 = float(r_het["C+H2"])
    phi = float(phi)
    rxn = np.zeros(8, dtype=float)
    rxn[0] = -(r_comb / phi + 0.5 * f_co2 * r1 + 0.5 * r2 + (1.5 + 0.5 * f_co2) * r5)
    rxn[1] = r_h2 - (r5 + r6)
    rxn[2] = (
        (2.0 - 2.0 / phi) * r_comb
        + r_h2o
        + 2.0 * r_co2
        + r4
        + r6
        + (1.0 - f_co2) * r5
        - (f_co2 * r1 + r3)
    )
    rxn[3] = (2.0 / phi - 1.0) * r_comb + f_co2 * r1 + r3 + f_co2 * r5 - (r_co2 + r4)
    rxn[5] = r_h2o + r3 + 3.0 * r6 - (2.0 * r_h2 + r2 + r4)
    rxn[7] = r2 + r4 + 2.0 * r5 - (r_h2o + r3 + r6)
    return rxn


def _aggregate_sources(cell) -> tuple[np.ndarray, float, float]:
    gas = np.zeros(8, dtype=float)
    solid = 0.0
    energy = 0.0
    for source in cell.sources:
        g, s_m, e = source.get_sources(cell.idx, cell.z, cell.dz)
        gas = gas + np.asarray(g, dtype=float)
        solid += float(s_m)
        energy += float(e)
    return gas, solid, energy


def _is_combustion(cell, current: StateVector, gas_src: np.ndarray) -> bool:
    inlet = np.asarray(cell.inlet.gas_moles, dtype=float)
    total = max(float(np.sum(inlet + gas_src)), 1.0e-9)
    p_o2 = float(current.P) * max(float(inlet[0] + gas_src[0]), 0.0) / total
    threshold = float(cell.op_conds.get("P_O2_Combustion_atm", 0.05)) * 101325.0
    return bool(p_o2 > threshold)


def _limit_flags(cell, current: StateVector, phys: Mapping[str, Any], gas_src: np.ndarray, ts_avg: float) -> dict[str, Any]:
    raw_het = cell.kinetics.calc_heterogeneous_rates(
        current,
        phys["d_p"],
        phys["S_total"],
        X_total=phys["X_total"],
        Re=phys["Re_p"],
        Sc=phys["Sc_p"],
        T_particle=ts_avg,
        char_combustion_factor=cell.op_conds.get("CharCombustionRateFactor"),
        use_fortran_diffusion=cell.op_conds.get("UseFortranDiffusion", False),
    )
    raw_homo = cell.kinetics.calc_homogeneous_rates(
        current,
        cell.V,
        inlet_state=cell.inlet,
        gas_src=gas_src,
        Ts_particle=ts_avg,
        wgs_rat_factor=cell.op_conds.get("WGS_RatFactor", False),
        msr_tmin_k=cell.op_conds.get("MSR_Tmin_K", 1000.0),
        wgs_catalytic_factor=cell.op_conds.get("WGS_CatalyticFactor"),
        wgs_k_factor=cell.op_conds.get("WGS_K_Factor"),
    )
    limited_het, limited_homo, _phi = cell._calc_rates(current, phys, gas_src, Ts_avg=ts_avg)
    keys = (
        ("het", "C+O2"),
        ("het", "C+H2O"),
        ("het", "C+CO2"),
        ("het", "C+H2"),
        ("homo", "CH4_Ox"),
        ("homo", "CO_Ox"),
        ("homo", "H2_Ox"),
        ("homo", "MSR"),
    )
    triggered = []
    for family, key in keys:
        raw = raw_het if family == "het" else raw_homo
        limited = limited_het if family == "het" else limited_homo
        raw_value = float(raw.get(key, 0.0))
        limited_value = float(limited.get(key, 0.0))
        scale = max(1.0, abs(raw_value))
        if abs(limited_value - raw_value) > 1.0e-6 * scale:
            triggered.append(key)
    c_fed = float(cell.coal_flow_dry) * (float(cell.coal_props.get("Cd", 60.0)) / 100.0)
    c_fed = c_fed if c_fed >= 1.0e-9 else 1.0e-9
    x_raw = 1.0 - (float(current.solid_mass) * float(current.carbon_fraction) / c_fed)
    t_ref = float(cell.op_conds.get("HeatLossRefTemp", 1800.0))
    t_weight_raw = (float(current.T) / t_ref) ** 4.0
    return {
        "reactant_limit_triggered": bool(triggered),
        "reactant_limit_keys": ";".join(triggered),
        "heat_loss_weight_clipped": bool(t_weight_raw < 0.1 or t_weight_raw > 5.0),
        "carbon_conversion_clipped": bool(x_raw < 0.0 or x_raw > 1.0),
        "particle_ct_clip": "not_computed",
    }


def evaluate_cell_state(cell, x_flat: np.ndarray) -> dict[str, Any]:
    """按 residuals() 的同一调用顺序取源项、反应项和焓，不改状态。"""
    current = StateVector.from_array(np.asarray(x_flat, dtype=float), P=cell.inlet.P, z=cell.z)
    gas_src, solid_src, energy_src = _aggregate_sources(cell)
    c_fed = float(cell.coal_flow_dry) * (float(cell.coal_props.get("Cd", 60.0)) / 100.0)
    phys = cell._calc_physics_props(current, c_fed)
    from .constants import PhysicalConstants

    tau = cell.dz / max(float(phys["v_g"]), PhysicalConstants.MIN_SLIP_VELOCITY)
    is_combustion = _is_combustion(cell, current, gas_src)
    ts_avg, ts_out = cell._calc_particle_temperature(
        current.T,
        cell.inlet.T_solid_or_gas,
        tau,
        phys["d_p"],
        current.solid_mass,
        current.carbon_fraction,
        current=current,
        phys=phys,
        gas_src=gas_src,
        is_combustion_zone=is_combustion,
    )
    r_het, r_homo, phi = cell._calc_rates(current, phys, gas_src, Ts_avg=ts_avg)
    flags = _limit_flags(cell, current, phys, gas_src, ts_avg)
    reaction = reaction_source_mol_s(cell, r_het, r_homo, phi)
    convection = np.asarray(current.gas_moles, dtype=float) - np.asarray(cell.inlet.gas_moles, dtype=float)
    h_in = float(MaterialService.get_total_enthalpy(cell.inlet, cell.coal_props))
    h_out = float(MaterialService.get_total_enthalpy(current, cell.coal_props, T_solid_override=ts_out))
    return {
        "current": current,
        "gas_src": gas_src,
        "solid_src": float(solid_src),
        "energy_src": float(energy_src),
        "reaction": reaction,
        "convection": convection,
        "is_combustion": is_combustion,
        "flags": flags,
        "H_in_W": h_in,
        "H_out_W": h_out,
        "phi": float(phi),
        "r_het": r_het,
        "r_homo": r_homo,
        "phys": phys,
        "ts_avg": float(ts_avg),
    }


def accept_multistart(sols: Sequence[Any]) -> Any | None:
    """与 GasifierSystem cell-0 multistart 相同：成功解里取最小 cost，否则用最后一次。"""
    if not sols:
        return None
    best = None
    best_cost = 1.0e9
    for sol in sols:
        if bool(getattr(sol, "success", False)) and float(sol.cost) < best_cost:
            best = sol
            best_cost = float(sol.cost)
    return best if best is not None else sols[-1]


def _solver_message(sol: Any) -> str:
    message = getattr(sol, "message", "")
    if isinstance(message, bytes):
        return message.decode("utf-8", errors="replace")
    return str(message)


def jacobian_condition_number(sol: Any) -> str:
    jac = getattr(sol, "jac", None)
    if jac is None:
        return "not_computed"
    matrix = np.asarray(jac, dtype=float)
    if matrix.ndim != 2 or matrix.size == 0 or not np.all(np.isfinite(matrix)):
        return "not_computed"
    value = float(np.linalg.cond(matrix))
    if not math.isfinite(value):
        return "not_computed"
    return f"{value:.8e}"


def _rank_components(names: Sequence[str], scaled: np.ndarray) -> tuple[str, str]:
    order = np.argsort(-np.abs(np.asarray(scaled, dtype=float)))
    primary = names[int(order[0])] if len(order) else ""
    secondary = names[int(order[1])] if len(order) > 1 else ""
    return str(primary), str(secondary)


def element_projection_of_gas_residual(gas_residual_mol_s: np.ndarray) -> dict[str, float]:
    """物种方程残差按交接单的原子系数加总，单位 mol/s。"""
    return atoms_from_gas(np.asarray(gas_residual_mol_s, dtype=float))


def element_identity_max_abs(closure: Mapping[str, float], projection: Mapping[str, float]) -> float:
    return max(
        abs(float(closure[f"{element}_closure_error_mol_s"]) - float(projection[element]))
        for element in ELEMENTS
    )


def atoms_from_gas(gas_mol_s: np.ndarray) -> dict[str, float]:
    gas = np.asarray(gas_mol_s, dtype=float)
    return {
        "C": float(gas[1] + gas[2] + gas[3]),
        "H": float(4.0 * gas[1] + 2.0 * gas[4] + 2.0 * gas[5] + 2.0 * gas[7]),
        "O": float(2.0 * gas[0] + gas[2] + 2.0 * gas[3] + gas[7]),
        "N": float(2.0 * gas[6]),
        "S": float(gas[4]),
    }


def reactor_element_flows(system: GasifierSystem, outlet: StateVector) -> dict[str, float]:
    coal = system.coal_props
    dry = float(system.W_dry)
    coal_mol = {
        element: dry * (float(coal[key]) / 100.0) / ELEMENT_KG_PER_MOL[element]
        for element, key in zip(ELEMENTS, ("Cd", "Hd", "Od", "Nd", "Sd"))
    }
    moisture_mol_s = (float(system.tmp_W_liq_evap) / H2O_MOLAR_MASS_G_PER_MOL) * 1000.0
    coal_mol["H"] += 2.0 * moisture_mol_s
    coal_mol["O"] += moisture_mol_s
    inlet_gas = atoms_from_gas(system.cells[0].inlet.gas_moles)
    outlet_gas = atoms_from_gas(outlet.gas_moles)
    outlet_solid = {element: 0.0 for element in ELEMENTS}
    outlet_solid["C"] = float(outlet.solid_mass) * float(outlet.carbon_fraction) / ELEMENT_KG_PER_MOL["C"]
    out: dict[str, float] = {}
    for element in ELEMENTS:
        inlet = coal_mol[element] + inlet_gas[element]
        outlet_total = outlet_gas[element] + outlet_solid[element]
        error = outlet_total - inlet
        out[f"{element}_inlet_mol_s"] = inlet
        out[f"{element}_inlet_coal_mol_s"] = coal_mol[element]
        out[f"{element}_inlet_gas_mol_s"] = inlet_gas[element]
        out[f"{element}_outlet_gas_mol_s"] = outlet_gas[element]
        out[f"{element}_outlet_solid_mol_s"] = outlet_solid[element]
        out[f"{element}_outlet_mol_s"] = outlet_total
        out[f"{element}_closure_error_mol_s"] = error
        out[f"{element}_closure_fraction"] = error / inlet if abs(inlet) > 1.0e-12 else math.nan
    return out


def solve_reference_case(
    row: Mapping[str, Any],
    *,
    n_cells: int = 1,
    extra_op_conds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """reference / fixed_physics：N_cells 给定，alpha_wgs=1，不改求解器容差。"""
    import model.gasifier_system as gasifier_module

    captured: list[Any] = []
    original = gasifier_module.least_squares

    def _capture(*args, **kwargs):
        sol = original(*args, **kwargs)
        captured.append(sol)
        return sol

    built = build_gasifier_inputs_from_model_case(row)
    op_conds = apply_solver_options(built["op_conds"], {})
    op_conds["WGS_CatalyticFactor"] = 1.0
    if extra_op_conds:
        op_conds.update(dict(extra_op_conds))
    system = GasifierSystem(built["geometry"], built["coal_props"], op_conds)
    gasifier_module.least_squares = _capture
    try:
        profile, _z = system.solve(
            N_cells=int(n_cells),
            solver_method="minimize",
            jacobian_mode="scipy",
            jax_warmup=False,
        )
    finally:
        gasifier_module.least_squares = original
    return {"system": system, "profile": np.asarray(profile, dtype=float), "solver_attempts": captured}


def _cell_rows_for_window(
    row: Mapping[str, Any],
    solved: Mapping[str, Any],
    *,
    source_commit_hash: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    system: GasifierSystem = solved["system"]
    profile = solved["profile"]
    accepted = accept_multistart(solved["solver_attempts"])
    long_rows: list[dict[str, Any]] = []
    term_rows: list[dict[str, Any]] = []
    closure: dict[str, Any] = {
        "dataset_id": DATASET_ID,
        "window_id": row["window_id"],
        "phase": int(row["phase_number"]),
        "split_role": row["split_role"],
        "N_cells": len(system.cells),
        "configuration": "reference",
        "alpha_wgs": 1.0,
        "execution_status": "completed",
        "fallback_count": int(system.solve_stats.get("fallback_count", 0)),
        "source_commit": source_commit_hash,
    }
    max_identity_gap = 0.0
    for cell, state_row in zip(system.cells, profile):
        components = cell.residual_components(np.asarray(state_row, dtype=float))
        evaluated = evaluate_cell_state(cell, np.asarray(state_row, dtype=float))
        names = list(components["names"])
        primary, secondary = _rank_components(names, components["scaled"])
        cost = float(components["cost"])
        metadata_ok = accepted is not None and abs(float(accepted.cost) - cost) <= COST_METADATA_ATOL
        if cell.idx == 0 and metadata_ok:
            solver_status = int(accepted.status)
            solver_message = _solver_message(accepted)
            nfev = int(accepted.nfev)
            njev = int(getattr(accepted, "njev", -1))
            jacobian = jacobian_condition_number(accepted)
        else:
            solver_status = ""
            solver_message = "not_computed" if cell.idx == 0 else "not_cell0"
            nfev = ""
            njev = ""
            jacobian = "not_computed"
        flags = evaluated["flags"]
        limit_or_piecewise = bool(
            evaluated["is_combustion"]
            or flags["reactant_limit_triggered"]
            or flags["heat_loss_weight_clipped"]
            or flags["carbon_conversion_clipped"]
        )
        for index, name in enumerate(names):
            long_rows.append(
                {
                    "dataset_id": DATASET_ID,
                    "window_id": row["window_id"],
                    "phase": int(row["phase_number"]),
                    "split_role": row["split_role"],
                    "configuration": "reference",
                    "N_cells": len(system.cells),
                    "cell_index": int(cell.idx),
                    "residual_name": name,
                    "unscaled_value": float(components["unscaled"][index]),
                    "unit": RESIDUAL_UNITS[name],
                    "scale_denominator": float(components["scales"][index]),
                    "scale_unit": SCALE_UNITS[name],
                    "scaled_value": float(components["scaled"][index]),
                    "residual_cost": cost,
                    "exceeds_1e-4": bool(cost > POOR_CONVERGENCE_COST_THRESHOLD),
                    "primary_component": primary,
                    "secondary_component": secondary,
                    "solver_status": solver_status,
                    "solver_message": solver_message,
                    "nfev": nfev,
                    "njev": njev,
                    "fallback_count": int(system.solve_stats.get("fallback_count", 0)),
                    "execution_status": "completed",
                    "is_combustion": bool(evaluated["is_combustion"]),
                    "limit_or_piecewise_triggered": limit_or_piecewise,
                    "reactant_limit_keys": flags["reactant_limit_keys"],
                    "heat_loss_weight_clipped": bool(flags["heat_loss_weight_clipped"]),
                    "carbon_conversion_clipped": bool(flags["carbon_conversion_clipped"]),
                    "particle_ct_clip": flags["particle_ct_clip"],
                    "jacobian_condition_number": jacobian,
                    "source_commit": source_commit_hash,
                }
            )
        if cell.idx == 0:
            reconstructed = evaluated["convection"] - evaluated["gas_src"] - evaluated["reaction"]
            gas_unscaled = np.asarray(components["unscaled"][:8], dtype=float)
            gap = float(np.max(np.abs(reconstructed - gas_unscaled)))
            max_identity_gap = max(max_identity_gap, gap)
            for index, species in enumerate(SPECIES_NAMES):
                term_rows.append(
                    {
                        "dataset_id": DATASET_ID,
                        "window_id": row["window_id"],
                        "phase": int(row["phase_number"]),
                        "split_role": row["split_role"],
                        "cell_index": 0,
                        "species": species,
                        "residual_name": f"gas_{species}",
                        "convection_inout_mol_s": float(evaluated["convection"][index]),
                        "reaction_source_mol_s": float(evaluated["reaction"][index]),
                        "inlet_source_mol_s": float(evaluated["gas_src"][index]),
                        "residual_mol_s": float(gas_unscaled[index]),
                        "term_sum_minus_residual_mol_s": float(reconstructed[index] - gas_unscaled[index]),
                        "source_commit": source_commit_hash,
                    }
                )
            energy_unscaled = float(components["unscaled"][names.index("energy")])
            q_loss = evaluated["H_in_W"] + evaluated["energy_src"] - evaluated["H_out_W"] + energy_unscaled
            closure.update(reactor_element_flows(system, evaluated["current"]))
            h_in = evaluated["H_in_W"]
            closure.update(
                {
                    "energy_H_in_W": h_in,
                    "energy_sources_W": evaluated["energy_src"],
                    "energy_loss_W": q_loss,
                    "energy_H_out_W": evaluated["H_out_W"],
                    "energy_closure_W": energy_unscaled,
                    "energy_closure_fraction_of_H_in": energy_unscaled / h_in if abs(h_in) > 1.0e-6 else math.nan,
                    "cell0_residual_cost": cost,
                    "cell0_exceeds_1e-4": bool(cost > POOR_CONVERGENCE_COST_THRESHOLD),
                    "term_identity_max_abs_mol_s": gap,
                }
            )
    last = profile[-1]
    gas = last[:8]
    dry = float(np.sum(gas[:7]))
    denom = dry + 1.0e-12
    closure.update(
        {
            "pred_CO_dry_vol_pct": float(gas[2] / denom * 100.0),
            "pred_CO2_dry_vol_pct": float(gas[3] / denom * 100.0),
            "pred_H2_dry_vol_pct": float(gas[5] / denom * 100.0),
            "pred_dry_flow_nm3_h": dry / 1000.0 * 22.414 * 3600.0,
            "plant_CO_dry_vol_pct": float(row["co_vol_pct"]),
            "plant_CO2_dry_vol_pct": float(row["co2_vol_pct"]),
            "plant_H2_dry_vol_pct": float(row["h2_vol_pct"]),
            "plant_dry_flow_nm3_h": float(row["syngas_flow_mean"]),
            "term_identity_max_abs_mol_s": max_identity_gap,
        }
    )
    return long_rows, term_rows, closure


def select_judgment(term_frame: pd.DataFrame, closure_frame: pd.DataFrame, long_frame: pd.DataFrame) -> dict[str, Any]:
    """四句判断的预声明规则。命中则只返回一句。"""
    costs = long_frame.drop_duplicates(["window_id", "cell_index"])
    cell0 = costs[costs["cell_index"] == 0]
    downstream = costs[costs["cell_index"] > 0]
    cell0_ok = bool((~cell0["exceeds_1e-4"].astype(bool)).all()) if len(cell0) else False
    downstream_poor = bool(downstream["exceeds_1e-4"].astype(bool).any()) if len(downstream) else False
    if downstream_poor and cell0_ok:
        return {
            "sentence_id": 1,
            "sentence": "下游单元过阈值，cell 0 不过：问题局限在入口单元。",
            "n_downstream_poor_cells": int(downstream["exceeds_1e-4"].astype(bool).sum()),
        }

    focus = term_frame[term_frame["species"].isin(["H2", "CO2"])].copy()
    classes = []
    for _, record in focus.iterrows():
        parts = {
            "convection": float(record["convection_inout_mol_s"]),
            "reaction": -float(record["reaction_source_mol_s"]),
            "inlet_source": -float(record["inlet_source_mol_s"]),
        }
        total = sum(abs(value) for value in parts.values())
        winner = max(parts, key=lambda key: abs(parts[key]))
        concentrated = total > 0.0 and abs(parts[winner]) >= 0.7 * total
        classes.append(winner if concentrated else "not_concentrated")
    unique = set(classes)
    if len(unique) == 1 and unique.pop() in {"reaction", "inlet_source"}:
        label = "反应源" if classes[0] == "reaction" else "入口源"
        column = "reaction_source_mol_s" if classes[0] == "reaction" else "inlet_source_mol_s"
        ranges = {}
        for species in ("H2", "CO2"):
            values = focus.loc[focus["species"] == species, column].astype(float)
            ranges[species] = {
                "min_mol_s": float(values.min()),
                "max_mol_s": float(values.max()),
            }
        return {
            "sentence_id": 2,
            "sentence": (
                f"三项分解显示残差集中在{label}："
                f"gas_H2 的该项为 {ranges['H2']['min_mol_s']:.6g} 至 {ranges['H2']['max_mol_s']:.6g} mol/s，"
                f"gas_CO2 的该项为 {ranges['CO2']['min_mol_s']:.6g} 至 {ranges['CO2']['max_mol_s']:.6g} mol/s。"
            ),
            "dominant_term": classes[0],
            "ranges_mol_s": ranges,
        }

    ratios = []
    for _, record in closure_frame.iterrows():
        h2 = abs(float(term_frame[(term_frame["window_id"] == record["window_id"]) & (term_frame["species"] == "H2")]["residual_mol_s"].iloc[0]))
        co2 = abs(float(term_frame[(term_frame["window_id"] == record["window_id"]) & (term_frame["species"] == "CO2")]["residual_mol_s"].iloc[0]))
        cell_scale = max(h2, co2)
        element_scale = max(abs(float(record[f"{element}_closure_error_mol_s"])) for element in ELEMENTS)
        ratios.append(
            {
                "window_id": record["window_id"],
                "cell_h2_co2_scale_mol_s": cell_scale,
                "element_closure_scale_mol_s": element_scale,
                "ratio": cell_scale / element_scale if element_scale > 1.0e-12 else math.inf,
            }
        )
    ratio_values = [float(item["ratio"]) for item in ratios]
    if ratio_values and max(ratio_values) < 10.0:
        return {
            "sentence_id": 3,
            "sentence": "元素闭合误差与 cell 残差同阶：当前状态不能支撑通道归因。",
            "window_ratios": ratios,
        }
    if ratio_values and min(ratio_values) >= 100.0:
        orders = int(math.floor(math.log10(min(ratio_values))))
        return {
            "sentence_id": 4,
            "sentence": (
                f"元素闭合误差远小于 cell 的 H2/CO2 残差：最小相差 {orders} 个数量级，"
                "避免把方程残差直接说成全炉元素不守恒。"
            ),
            "window_ratios": ratios,
            "orders_of_magnitude": orders,
        }
    return {
        "sentence_id": 0,
        "sentence": "这 9 个窗口没有落到交接单允许的四句判断里。不另写成因结论。",
        "window_ratios": ratios,
        "term_classes": classes,
    }


def _fmt(value: float) -> str:
    if not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.6g}"


def render_interpretation(judgment: Mapping[str, Any], closure_frame: pd.DataFrame, term_frame: pd.DataFrame) -> str:
    lines = [
        "# P0-A 判断",
        "",
        judgment["sentence"],
        "",
        "N_cells=1，每个窗口只有 cell 0。没有下游单元，所以第一句不由这批求解支持。",
        "",
        "阈值是 residual_cost > 1e-4。9 个窗口的 cell 0 都超过该阈值。"
        if bool(closure_frame["cell0_exceeds_1e-4"].all())
        else "阈值是 residual_cost > 1e-4。",
        "",
        "| window_id | phase | cell0 cost | H2 mol/s | CO2 mol/s | C error mol/s | H error mol/s | O error mol/s | energy W |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, record in closure_frame.sort_values(["phase", "window_id"]).iterrows():
        h2 = float(term_frame[(term_frame["window_id"] == record["window_id"]) & (term_frame["species"] == "H2")]["residual_mol_s"].iloc[0])
        co2 = float(term_frame[(term_frame["window_id"] == record["window_id"]) & (term_frame["species"] == "CO2")]["residual_mol_s"].iloc[0])
        lines.append(
            "| {window} | {phase} | {cost} | {h2} | {co2} | {c} | {h} | {o} | {energy} |".format(
                window=record["window_id"],
                phase=int(record["phase"]),
                cost=_fmt(record["cell0_residual_cost"]),
                h2=_fmt(h2),
                co2=_fmt(co2),
                c=_fmt(record["C_closure_error_mol_s"]),
                h=_fmt(record["H_closure_error_mol_s"]),
                o=_fmt(record["O_closure_error_mol_s"]),
                energy=_fmt(record["energy_closure_W"]),
            )
        )
    lines.extend(
        [
            "",
            "气相三项的符号：residual = (F_out - F_in) - inlet_source - reaction_source。",
            "inlet_source 是热解与蒸发，reaction_source 是化学计量净生成。三项相加后与未缩放气相残差的最大偏差见 protocol 的 term_identity_max_abs_mol_s。",
            "",
            "出口固相只计 carbon_fraction 对应的碳。模型固相状态不携带 H、O、N、S。",
        ]
    )
    return "\n".join(lines) + "\n"


def protocol_document(
    *,
    window_ids: Sequence[str],
    source_commit_hash: str,
    term_identity_max_abs_mol_s: float,
    judgment_id: int,
) -> dict[str, Any]:
    return {
        "dataset_id": DATASET_ID,
        "configuration": "reference",
        "N_cells": 1,
        "alpha_wgs": 1.0,
        "solver_method": "minimize",
        "jacobian_mode": "scipy",
        "window_ids": list(window_ids),
        "n_windows": len(list(window_ids)),
        "residual_cost_threshold": POOR_CONVERGENCE_COST_THRESHOLD,
        "threshold_rule": "poor when residual_cost > 1e-4",
        "units": {
            "gas": "mol/s",
            "solid_mass": "kg/s",
            "ref_solid": "kg/s, inlet dry solid mass",
            "carbon_fraction_unscaled": "kg/s carbon-mass imbalance",
            "carbon_fraction_scale": "dimensionless max(char_Xc0, 0.1)",
            "energy": "W",
            "gas_scale": "ref_flow mol/s except gas_N2, which uses max(|inlet N2| + |pyrolysis N2|, 1e-3) mol/s",
            "energy_scale": "max(ref_flow * 35 * 200, 5e5) W",
        },
        "term_split_sign": "residual = (F_out - F_in) - inlet_source - reaction_source",
        "term_identity_atol_mol_s": TERM_IDENTITY_ATOL_MOL_S,
        "term_identity_max_abs_mol_s": term_identity_max_abs_mol_s,
        "element_molar_mass_kg_per_mol": ELEMENT_KG_PER_MOL,
        "moisture_accounting": "coal moisture uses GasifierSystem.tmp_W_liq_evap and (mass_kg_s / 18.015) * 1000 mol/s; steam remains in inlet gas",
        "outlet_solid_elements": "carbon only; H O N S are not solid state variables",
        "jacobian_condition_number": "2-norm condition of the accepted cell-0 least_squares scaled residual Jacobian; not_computed when the accepted attempt cost does not match the stored residual or the matrix is non-finite",
        "iteration_count": "least_squares has nfev and njev, not a Newton iteration count",
        "particle_ct_clip": "not_computed; the exponential clip sits inside the particle-temperature loop",
        "judgment_sentence_id": judgment_id,
        "concentration_rule": "a term is concentrated when its absolute residual contribution is at least 70% of the sum of absolute convection, reaction and inlet contributions",
        "same_order_rule": "sentence 3 when every window has max(|H2|,|CO2|) / max(|element errors|) < 10",
        "much_smaller_rule": "sentence 4 when every window has that ratio >= 100",
        "source_commit": source_commit_hash,
        "handoff_model_head": HANDOFF_MODEL_HEAD,
        "solver_modules_match_handoff_head": source_commit_hash.startswith(HANDOFF_MODEL_HEAD),
        "excluded_reruns": ["tighter_tol", "alt_cell0_init", "refined_first_cv"],
    }


def assert_no_local_routing(text: str) -> None:
    lowered = text.lower()
    forbidden = ("/users/", "workbook", ".xlsx", "gasifier-data-prep")
    found = [token for token in forbidden if token in lowered]
    if found:
        raise RuntimeError(f"handoff artifact contains forbidden routing tokens: {found}")


def run_p0a(rows: Sequence[Mapping[str, Any]], window_ids: Sequence[str], output_dir: Path) -> dict[str, Any]:
    silence_solver_logs()
    output_dir.mkdir(parents=True, exist_ok=True)
    commit = source_commit()
    selected = [row for row in rows if str(row["window_id"]) in set(window_ids)]
    if len(selected) != len(list(window_ids)):
        raise RuntimeError("representative window ids are not all in the frozen split")
    long_rows: list[dict[str, Any]] = []
    term_rows: list[dict[str, Any]] = []
    closure_rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        print(f"P0-A {index}/{len(selected)} {row['window_id']}", flush=True)
        solved = solve_reference_case(row, n_cells=1)
        part_long, part_term, closure = _cell_rows_for_window(row, solved, source_commit_hash=commit)
        gap = float(closure["term_identity_max_abs_mol_s"])
        if gap > TERM_IDENTITY_ATOL_MOL_S:
            raise RuntimeError(
                f"cell-0 term split does not reproduce the gas residual for {row['window_id']}: max abs gap {gap} mol/s"
            )
        long_rows.extend(part_long)
        term_rows.extend(part_term)
        closure_rows.append(closure)
    long_frame = pd.DataFrame(long_rows)
    term_frame = pd.DataFrame(term_rows)
    closure_frame = pd.DataFrame(closure_rows)
    identity = float(closure_frame["term_identity_max_abs_mol_s"].max())
    if identity > TERM_IDENTITY_ATOL_MOL_S:
        raise RuntimeError(f"cell-0 term split does not reproduce the gas residual: max abs gap {identity} mol/s")
    judgment = select_judgment(term_frame, closure_frame, long_frame)
    protocol = protocol_document(
        window_ids=[row["window_id"] for row in selected],
        source_commit_hash=commit,
        term_identity_max_abs_mol_s=identity,
        judgment_id=int(judgment["sentence_id"]),
    )
    interpretation = render_interpretation(judgment, closure_frame, term_frame)
    long_path = output_dir / "p0a_cell_residual_long.csv"
    term_path = output_dir / "p0a_cell0_term_split.csv"
    closure_path = output_dir / "p0a_reactor_closure.csv"
    protocol_path = output_dir / "p0a_protocol.json"
    interpretation_path = output_dir / "p0a_interpretation.md"
    long_frame.to_csv(long_path, index=False)
    term_frame.to_csv(term_path, index=False)
    closure_frame.to_csv(closure_path, index=False)
    protocol_text = json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    assert_no_local_routing(protocol_text)
    assert_no_local_routing(interpretation)
    protocol_path.write_text(protocol_text, encoding="utf-8")
    interpretation_path.write_text(interpretation, encoding="utf-8")
    write_sha256_manifest(
        [long_path, term_path, closure_path, protocol_path, interpretation_path],
        output_dir / "p0a_sha256.txt",
    )
    return judgment


def m2_window_ids(stability_csv: Path) -> dict[str, list[str]]:
    frame = pd.read_csv(stability_csv)
    fixed = frame[(frame["method"] == "fixed_physics") & (frame["target"] == "CO_dry_vol_pct") & (frame["N_cells"] == 1)]
    out = {}
    for role, group in fixed.groupby("split_role"):
        out[str(role)] = sorted(group["window_id"].astype(str).unique())
    out["all"] = sorted(fixed["window_id"].astype(str).unique())
    return out


def test_co_rmse(predictions: pd.DataFrame) -> float:
    work = predictions[(predictions["split_role"] == "test") & (predictions["execution_status"] == "completed")]
    residual = work["co_residual_vol_pct"].astype(float).to_numpy()
    if residual.size == 0 or not np.all(np.isfinite(residual)):
        return math.nan
    return float(np.sqrt(np.mean(residual * residual)))


def reproduction_lock_result(
    predictions: pd.DataFrame,
    m2_ids: Mapping[str, Sequence[str]],
    *,
    source_commit_hash: str,
) -> dict[str, Any]:
    completed = predictions[predictions["execution_status"] == "completed"]
    reproduced = {
        role: sorted(completed.loc[completed["split_role"] == role, "window_id"].astype(str).unique())
        for role in ("train", "validation", "test")
    }
    reproduced["all"] = sorted(completed["window_id"].astype(str).unique())
    rmse = test_co_rmse(predictions)
    ids_match = all(list(reproduced[role]) == list(m2_ids[role]) for role in ("all", "test", "train", "validation"))
    rmse_ok = math.isfinite(rmse) and abs(rmse - PUBLISHED_TEST_CO_RMSE) <= REPRODUCTION_TOLERANCE
    n_test_completed = int((completed["split_role"] == "test").sum())
    passed = bool(ids_match and rmse_ok and n_test_completed == len(m2_ids["test"]) and len(predictions) == len(m2_ids["all"]))
    return {
        "dataset_id": DATASET_ID,
        "configuration": "reference",
        "N_cells": 1,
        "alpha_wgs": 1.0,
        "solver_method": "minimize",
        "target": "CO_dry_vol_pct",
        "split_role": "test",
        "published_rmse": PUBLISHED_TEST_CO_RMSE,
        "tolerance": REPRODUCTION_TOLERANCE,
        "reproduced_rmse": rmse,
        "abs_error": abs(rmse - PUBLISHED_TEST_CO_RMSE) if math.isfinite(rmse) else math.nan,
        "passed": passed,
        "window_ids_match_m2": ids_match,
        "n_test_completed": n_test_completed,
        "n_test_m2": len(list(m2_ids["test"])),
        "n_windows_completed": int(len(completed)),
        "n_windows_m2": len(list(m2_ids["all"])),
        "missing_test_window_ids": sorted(set(m2_ids["test"]) - set(reproduced["test"])),
        "extra_test_window_ids": sorted(set(reproduced["test"]) - set(m2_ids["test"])),
        "residual_cost_threshold": POOR_CONVERGENCE_COST_THRESHOLD,
        "source_commit": source_commit_hash,
        "handoff_model_head": HANDOFF_MODEL_HEAD,
        "solver_modules_match_handoff_head": source_commit_hash.startswith(HANDOFF_MODEL_HEAD),
        "mesh_comparison_executed": False,
        "stop_reason": "" if passed else "N=1 test CO RMSE or window ids do not match M2",
    }


def run_p0b_n1_lock(rows: Sequence[Mapping[str, Any]], m2_ids: Mapping[str, Sequence[str]], output_dir: Path) -> dict[str, Any]:
    silence_solver_logs()
    output_dir.mkdir(parents=True, exist_ok=True)
    commit = source_commit()
    wanted = set(m2_ids["all"])
    selected = [row for row in rows if str(row["window_id"]) in wanted]
    if len(selected) != len(wanted):
        raise RuntimeError("frozen split does not contain every M2 window id")
    records = []
    for index, row in enumerate(selected, start=1):
        print(f"P0-B N=1 {index}/{len(selected)} {row['window_id']}", flush=True)
        try:
            solved = solve_reference_case(row, n_cells=1)
            profile = solved["profile"]
            gas = profile[-1, :8]
            dry = float(np.sum(gas[:7]))
            denom = dry + 1.0e-12
            pred_co = float(gas[2] / denom * 100.0)
            records.append(
                {
                    "window_id": row["window_id"],
                    "split_role": row["split_role"],
                    "phase": int(row["phase_number"]),
                    "execution_status": "completed",
                    "pred_CO_dry_vol_pct": pred_co,
                    "co_residual_vol_pct": pred_co - float(row["co_vol_pct"]),
                }
            )
        except Exception as exc:
            records.append(
                {
                    "window_id": row["window_id"],
                    "split_role": row["split_role"],
                    "phase": int(row["phase_number"]),
                    "execution_status": f"error:{type(exc).__name__}",
                    "pred_CO_dry_vol_pct": math.nan,
                    "co_residual_vol_pct": math.nan,
                }
            )
    result = reproduction_lock_result(pd.DataFrame(records), m2_ids, source_commit_hash=commit)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    assert_no_local_routing(text)
    path = output_dir / "p0b_n1_reproduction.json"
    path.write_text(text, encoding="utf-8")
    write_sha256_manifest([path], output_dir / "p0b_sha256.txt")
    return result


def m1_reference_window_ids(variant_matrix_csv: Path) -> list[str]:
    frame = pd.read_csv(variant_matrix_csv)
    reference = frame[frame["configuration"] == "reference"].sort_values(["phase", "window_id"])
    return reference["window_id"].astype(str).tolist()
