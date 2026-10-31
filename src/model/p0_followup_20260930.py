"""P0 后续：ftol 停因、关闭燃烧区分支的诊断重解，以及 N=3/5/9 网格比较。"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .alpha_candidate_identifiability import inject_alpha
from .p0_round_20260930 import (
    DATASET_ID,
    ELEMENTS,
    HANDOFF_MODEL_HEAD,
    POOR_CONVERGENCE_COST_THRESHOLD,
    accept_multistart,
    assert_no_local_routing,
    element_identity_max_abs,
    element_projection_of_gas_residual,
    evaluate_cell_state,
    reaction_source_mol_s,
    reactor_element_flows,
    silence_solver_logs,
    solve_reference_case,
    source_commit,
    write_sha256_manifest,
)
ELEMENT_IDENTITY_ATOL_MOL_S = 1.0e-8
REFERENCE_MATCH_ATOL_MOL_S = 1.0e-6
FLAT_RELATIVE_DROP = 1.0e-3
N9_FULL_SPLIT_MAX_WALL_TIME_S = 120.0
SOLVER_STATUS = {
    -1: "improper_input",
    0: "max_nfev",
    1: "gtol",
    2: "ftol",
    3: "xtol",
    4: "ftol_and_xtol",
}
LIMITED_REACTIONS = (
    ("het", "C+O2"),
    ("het", "C+H2O"),
    ("het", "C+CO2"),
    ("het", "C+H2"),
    ("homo", "CH4_Ox"),
    ("homo", "CO_Ox"),
    ("homo", "H2_Ox"),
    ("homo", "MSR"),
)
M4_ONE_SIDED_STEP = {
    "alpha_wgs": 0.03,
    "alpha_char_oxidation": 0.03,
    "alpha_volatile_oxidation": 0.1,
}
M4_BASELINE = {
    "alpha_wgs": 1.0,
    "alpha_char_oxidation": 0.3,
    "alpha_volatile_oxidation": 1.0,
}


def solver_bounds(dry_solid_kg_s: float) -> tuple[np.ndarray, np.ndarray]:
    lower = np.zeros(11)
    upper = np.ones(11) * 3000.0
    upper[8] = float(dry_solid_kg_s) * 1.5
    upper[9] = 1.0
    lower[10] = 300.0
    upper[10] = 4000.0
    return lower, upper


def probe_relative_cost_drop(residual_fn, x: np.ndarray, grad: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """在边界内探测可行步。相对下降超过阈值则 cost 还没有平坦。"""
    x = np.asarray(x, dtype=float)
    base = 0.5 * float(np.sum(np.asarray(residual_fn(x), dtype=float) ** 2))
    best = base
    direction = np.asarray(grad, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(direction)) if direction.size == x.size else 0.0
    if norm > 0.0:
        direction = -direction / norm
        for step in (1.0e-8, 1.0e-6, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0):
            trial = np.clip(x + step * direction, lower, upper)
            cost = 0.5 * float(np.sum(np.asarray(residual_fn(trial), dtype=float) ** 2))
            best = min(best, cost)
    scales = np.maximum(np.abs(x), 1.0)
    for frac in (1.0e-6, 1.0e-4, 1.0e-3, 1.0e-2):
        for index in range(x.size):
            for sign in (-1.0, 1.0):
                trial = np.array(x, copy=True)
                trial[index] = x[index] + sign * frac * scales[index]
                trial = np.clip(trial, lower, upper)
                cost = 0.5 * float(np.sum(np.asarray(residual_fn(trial), dtype=float) ** 2))
                best = min(best, cost)
    return (base - best) / max(base, 1.0e-30)


def classify_stop(cost: float, relative_drop: float) -> str:
    if float(cost) <= POOR_CONVERGENCE_COST_THRESHOLD:
        return "residual_below_threshold"
    if float(relative_drop) >= FLAT_RELATIVE_DROP:
        return "stopped_early_cost_still_descends"
    return "no_root_after_limits"


def rate_branch_criterion(cell, current, gas_src: np.ndarray) -> dict[str, float | bool]:
    """与 Cell._calc_rates 里的燃烧区判据相同。"""
    inlet = np.asarray(cell.inlet.gas_moles, dtype=float)
    avail = np.maximum(inlet + np.asarray(gas_src, dtype=float), 0.0)
    total = max(float(np.sum(avail)), 1.0e-9)
    p_o2 = float(current.P) * float(avail[0]) / total
    threshold = float(cell.op_conds.get("P_O2_Combustion_atm", 0.05)) * 101325.0
    entered = bool(p_o2 > threshold)
    disabled = bool(cell.op_conds.get("diagnostic_disable_combustion_zone", False))
    return {
        "pO2_Pa": p_o2,
        "pO2_atm": p_o2 / 101325.0,
        "threshold_Pa": threshold,
        "threshold_atm": threshold / 101325.0,
        "would_enter_combustion_zone": entered,
        "is_combustion": False if disabled else entered,
    }


def name_h2_outlet_driver(
    outlet_h2_mol_s: float,
    consumers_mol_s: Mapping[str, float],
    kinetic_h2_ox: float,
    limited_h2_ox: float,
) -> str:
    if float(outlet_h2_mol_s) >= 1.0:
        return "outlet_h2_not_near_zero"
    top = max(consumers_mol_s, key=lambda key: float(consumers_mol_s[key]))
    replaced = abs(float(limited_h2_ox) - float(kinetic_h2_ox)) > 1.0e-6 * max(1.0, abs(float(kinetic_h2_ox)))
    if top == "H2_Ox" and replaced:
        return "H2_Ox_combustion_zone_instant_oxidation"
    return str(top)


def _raw_rates(cell, current, phys, gas_src, ts_avg: float) -> tuple[dict[str, float], dict[str, float]]:
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
    return raw_het, raw_homo


def _dry_kpis(profile: np.ndarray) -> dict[str, float]:
    gas = np.asarray(profile[-1, :8], dtype=float)
    dry = float(np.sum(gas[:7]))
    denom = dry + 1.0e-12
    co = float(gas[2] / denom * 100.0)
    h2 = float(gas[5] / denom * 100.0)
    return {
        "pred_CO_dry_vol_pct": co,
        "pred_H2_dry_vol_pct": h2,
        "pred_CO2_dry_vol_pct": float(gas[3] / denom * 100.0),
        "pred_H2_mol_s": float(gas[5]),
        "pred_H2_CO_ratio": h2 / max(co, 1.0e-12),
        "pred_dry_flow_nm3_h": dry / 1000.0 * 22.414 * 3600.0,
    }


def _identity_from_solve(solved: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, float], float, list[bool]]:
    system = solved["system"]
    profile = solved["profile"]
    projection = {element: 0.0 for element in ELEMENTS}
    combustion = []
    for cell, state_row in zip(system.cells, profile):
        components = cell.residual_components(np.asarray(state_row, dtype=float))
        evaluated = evaluate_cell_state(cell, np.asarray(state_row, dtype=float))
        part = element_projection_of_gas_residual(components["unscaled"][:8])
        for element in ELEMENTS:
            projection[element] += part[element]
        combustion.append(bool(rate_branch_criterion(cell, evaluated["current"], evaluated["gas_src"])["is_combustion"]))
    outlet = evaluated["current"]
    closure = reactor_element_flows(system, outlet)
    return closure, projection, element_identity_max_abs(closure, projection), combustion


def _cell0_diagnosis(row: Mapping[str, Any], solved: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    system = solved["system"]
    cell = system.cells[0]
    state = np.asarray(solved["profile"][0], dtype=float)
    accepted = accept_multistart(solved["solver_attempts"])
    components = cell.residual_components(state)
    evaluated = evaluate_cell_state(cell, state)
    criterion = rate_branch_criterion(cell, evaluated["current"], evaluated["gas_src"])
    raw_het, raw_homo = _raw_rates(cell, evaluated["current"], evaluated["phys"], evaluated["gas_src"], evaluated["ts_avg"])
    limited_het = evaluated["r_het"]
    limited_homo = evaluated["r_homo"]
    names = list(components["names"])
    h2_index = names.index("gas_H2")
    cost = float(components["cost"])
    grad = np.asarray(getattr(accepted, "grad", accepted.jac.T @ accepted.fun), dtype=float)
    lower, upper = solver_bounds(system.W_dry)
    relative_drop = probe_relative_cost_drop(cell.residuals, np.asarray(accepted.x, dtype=float), grad, lower, upper)
    status = int(accepted.status)
    active = int(np.asarray(accepted.active_mask, dtype=int)[5])
    kinetic_h2 = float(raw_homo.get("H2_Ox", 0.0))
    limited_h2 = float(limited_homo.get("H2_Ox", 0.0))
    consumers = {
        "H2_Ox": limited_h2,
        "C+H2": 2.0 * float(limited_het.get("C+H2", 0.0)),
        "RWGS": float(limited_homo.get("RWGS", 0.0)),
    }
    outlet_h2 = float(evaluated["current"].gas_moles[5])
    reaction = reaction_source_mol_s(cell, limited_het, limited_homo, evaluated["phi"])
    h2_target = float(cell.inlet.gas_moles[5] + evaluated["gas_src"][5] + reaction[5])
    diagnosis = {
        "dataset_id": DATASET_ID,
        "window_id": row["window_id"],
        "phase": int(row["phase_number"]),
        "split_role": row["split_role"],
        "solver_status": status,
        "solver_stop": SOLVER_STATUS.get(status, str(status)),
        "residual_cost": cost,
        "exceeds_1e-4": bool(cost > POOR_CONVERGENCE_COST_THRESHOLD),
        "nfev": int(accepted.nfev),
        "njev": int(getattr(accepted, "njev", -1)),
        "optimality": float(accepted.optimality),
        "relative_cost_drop": float(relative_drop),
        "cost_flat": bool(relative_drop < FLAT_RELATIVE_DROP),
        "stop_class": classify_stop(cost, relative_drop),
        "fallback_count": int(system.solve_stats.get("fallback_count", 0)),
        "pO2_atm": criterion["pO2_atm"],
        "threshold_atm": criterion["threshold_atm"],
        "is_combustion": criterion["is_combustion"],
        "h2_residual_mol_s": float(components["unscaled"][h2_index]),
        "h2_outlet_mol_s": outlet_h2,
        "h2_stoichiometric_target_mol_s": h2_target,
        "h2_lower_bound_active": bool(active < 0 or outlet_h2 <= 1.0e-8),
        "h2_outlet_driver": name_h2_outlet_driver(outlet_h2, consumers, kinetic_h2, limited_h2),
        "pred_H2_dry_vol_pct": _dry_kpis(solved["profile"])["pred_H2_dry_vol_pct"],
    }
    rate_rows = []
    for family, key in LIMITED_REACTIONS:
        raw = raw_het if family == "het" else raw_homo
        limited = limited_het if family == "het" else limited_homo
        raw_value = float(raw.get(key, 0.0))
        limited_value = float(limited.get(key, 0.0))
        rate_rows.append(
            {
                "dataset_id": DATASET_ID,
                "window_id": row["window_id"],
                "phase": int(row["phase_number"]),
                "reaction": key,
                "unlimited_rate_mol_s": raw_value,
                "limited_rate_mol_s": limited_value,
                "limit_binding": bool(abs(limited_value - raw_value) > 1.0e-6 * max(1.0, abs(raw_value))),
                "is_combustion": criterion["is_combustion"],
            }
        )
    return diagnosis, rate_rows


def _combustion_off_row(row: Mapping[str, Any], solved: Mapping[str, Any]) -> dict[str, Any]:
    system = solved["system"]
    cell = system.cells[0]
    state = np.asarray(solved["profile"][0], dtype=float)
    components = cell.residual_components(state)
    names = list(components["names"])
    closure, projection, gap, combustion = _identity_from_solve(solved)
    kpis = _dry_kpis(solved["profile"])
    record = {
        "dataset_id": DATASET_ID,
        "window_id": row["window_id"],
        "phase": int(row["phase_number"]),
        "split_role": row["split_role"],
        "diagnostic_disable_combustion_zone": True,
        "cell0_cost": float(components["cost"]),
        "exceeds_1e-4": bool(float(components["cost"]) > POOR_CONVERGENCE_COST_THRESHOLD),
        "h2_residual_mol_s": float(components["unscaled"][names.index("gas_H2")]),
        "pred_H2_dry_vol_pct": kpis["pred_H2_dry_vol_pct"],
        "pred_H2_mol_s": kpis["pred_H2_mol_s"],
        "is_combustion": bool(combustion[0]),
        "element_species_identity_max_abs_mol_s": gap,
        "fallback_count": int(system.solve_stats.get("fallback_count", 0)),
        "execution_status": "completed",
    }
    for element in ELEMENTS:
        record[f"{element}_closure_error_mol_s"] = closure[f"{element}_closure_error_mol_s"]
        record[f"{element}_species_projection_mol_s"] = projection[element]
        record[f"{element}_identity_delta_mol_s"] = closure[f"{element}_closure_error_mol_s"] - projection[element]
    return record


def _reference_match_gap(term_frame: pd.DataFrame, p0a_term_csv: Path) -> float:
    archived = pd.read_csv(p0a_term_csv)
    archived = archived[archived["species"] == "H2"][["window_id", "residual_mol_s"]]
    merged = term_frame.merge(archived, on="window_id", suffixes=("_new", "_p0a"))
    return float((merged["residual_mol_s_new"] - merged["residual_mol_s_p0a"]).abs().max())


def render_branch_interpretation(diagnosis: pd.DataFrame, off: pd.DataFrame, *, identity_holds: bool, identity_gap: float) -> str:
    counts = diagnosis["stop_class"].value_counts().to_dict()
    drivers = sorted(set(diagnosis["h2_outlet_driver"]))
    if not identity_holds:
        opening = (
            f"关掉燃烧区分支后，元素误差不再等于物种残差。最大绝对差额 {identity_gap:.6g} mol/s。"
            "停止，不改系数，也不做网格比较。"
        )
    elif list(counts) == ["no_root_after_limits"] or counts.get("no_root_after_limits", 0) == len(diagnosis):
        opening = (
            "9 个窗口都以 ftol 停在非零残差上，接受解邻域内的可行步不能再把 cost 降下 0.1%。"
            "非零残差来自限幅后的方程在这个可行域里没有根，不是求解器停早了。"
        )
    elif counts.get("stopped_early_cost_still_descends", 0) == len(diagnosis):
        opening = "9 个窗口的邻域里仍有能降低 cost 的可行步。求解器在几十 mol/s 的残差上停早了。"
    else:
        opening = "9 个窗口的停止类型不一致：" + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())) + "。"
    lines = [
        "# P0-C 判断",
        "",
        opening,
        "",
        f"出口 H2 的消耗主导项：{', '.join(drivers)}。",
        "",
        "| window_id | stop | cost | relative drop | pO2 atm | H2 out mol/s | H2 driver | off cost | off H2 mol/s | off H2 vol% | identity gap |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    off_by_id = off.set_index("window_id")
    for _, record in diagnosis.sort_values(["phase", "window_id"]).iterrows():
        other = off_by_id.loc[record["window_id"]]
        lines.append(
            "| {window} | {stop} | {cost:.6g} | {drop:.3g} | {po2:.4g} | {hout:.6g} | {driver} | {ocost:.6g} | {oh2:.6g} | {ovol:.6g} | {gap:.3g} |".format(
                window=record["window_id"],
                stop=record["solver_stop"],
                cost=float(record["residual_cost"]),
                drop=float(record["relative_cost_drop"]),
                po2=float(record["pO2_atm"]),
                hout=float(record["h2_outlet_mol_s"]),
                driver=record["h2_outlet_driver"],
                ocost=float(other["cell0_cost"]),
                oh2=float(other["h2_residual_mol_s"]),
                ovol=float(other["pred_H2_dry_vol_pct"]),
                gap=float(other["element_species_identity_max_abs_mol_s"]),
            )
        )
    lines.extend(
        [
            "",
            "燃烧区分支是入口可用氧分压高于 0.05 atm 时，用瞬时挥发分氧化替换 CH4/CO/H2 动力学速率。",
            "诊断重解只把这个分支关掉，平衡式、热解产率、phi、f_co2 和 H2_OxidationFactor 都没有改。",
            "cost 平坦的判据：边界内的梯度方向和各坐标相对步都不能把 0.5||r||^2 降低 0.1% 以上。",
        ]
    )
    return "\n".join(lines) + "\n"


def run_task1(rows: Sequence[Mapping[str, Any]], window_ids: Sequence[str], output_dir: Path, p0a_term_csv: Path) -> dict[str, Any]:
    silence_solver_logs()
    output_dir.mkdir(parents=True, exist_ok=True)
    commit = source_commit()
    selected = [row for row in rows if str(row["window_id"]) in set(window_ids)]
    diagnoses = []
    rate_rows = []
    reference_h2 = []
    for index, row in enumerate(selected, start=1):
        print(f"P0-C reference {index}/{len(selected)} {row['window_id']}", flush=True)
        solved = solve_reference_case(row, n_cells=1)
        diagnosis, rates = _cell0_diagnosis(row, solved)
        diagnoses.append(diagnosis)
        rate_rows.extend(rates)
        reference_h2.append({"window_id": row["window_id"], "residual_mol_s": diagnosis["h2_residual_mol_s"]})
    reference_gap = _reference_match_gap(pd.DataFrame(reference_h2), p0a_term_csv)
    if reference_gap > REFERENCE_MATCH_ATOL_MOL_S:
        raise RuntimeError(f"reference re-solve does not match P0-A H2 residual: max abs gap {reference_gap} mol/s")
    off_rows = []
    for index, row in enumerate(selected, start=1):
        print(f"P0-C combustion-off {index}/{len(selected)} {row['window_id']}", flush=True)
        solved = solve_reference_case(row, n_cells=1, extra_op_conds={"diagnostic_disable_combustion_zone": True})
        off_rows.append(_combustion_off_row(row, solved))
    diagnosis_frame = pd.DataFrame(diagnoses)
    rate_frame = pd.DataFrame(rate_rows)
    off_frame = pd.DataFrame(off_rows)
    identity_gap = float(off_frame["element_species_identity_max_abs_mol_s"].max())
    identity_holds = bool(identity_gap <= ELEMENT_IDENTITY_ATOL_MOL_S and (off_frame["is_combustion"] == False).all())
    interpretation = render_branch_interpretation(
        diagnosis_frame,
        off_frame,
        identity_holds=identity_holds,
        identity_gap=identity_gap,
    )
    protocol = {
        "dataset_id": DATASET_ID,
        "n_windows": len(selected),
        "window_ids": [row["window_id"] for row in selected],
        "N_cells": 1,
        "combustion_branch": "rate branch enters when inlet-available pO2 > 0.05 atm and replaces CH4_Ox, CO_Ox, H2_Ox with instant volatile oxidation",
        "diagnostic_switch": "diagnostic_disable_combustion_zone forces the existing gasification branch in both the rate selector and the particle-temperature selector",
        "unchanged": ["_calc_balances coefficients", "pyrolysis yields", "phi", "f_co2", "H2_OxidationFactor", "residual_cost threshold 1e-4"],
        "flat_relative_drop": FLAT_RELATIVE_DROP,
        "flat_rule": "no_root_after_limits when cost > 1e-4 and no feasible probe reduces cost by 0.1 percent; otherwise stopped_early",
        "element_identity_atol_mol_s": ELEMENT_IDENTITY_ATOL_MOL_S,
        "element_identity_holds_after_combustion_off": identity_holds,
        "element_identity_max_abs_mol_s": identity_gap,
        "reference_h2_match_max_abs_mol_s": reference_gap,
        "source_commit": commit,
        "handoff_model_head": HANDOFF_MODEL_HEAD,
        "solver_modules_match_handoff_head": commit.startswith(HANDOFF_MODEL_HEAD),
        "mesh_started": False,
    }
    diagnosis_path = output_dir / "p0c_ftol_diagnosis.csv"
    rate_path = output_dir / "p0c_limit_rates.csv"
    off_path = output_dir / "p0c_combustion_off.csv"
    protocol_path = output_dir / "p0c_protocol.json"
    interpretation_path = output_dir / "p0c_interpretation.md"
    diagnosis_frame.to_csv(diagnosis_path, index=False)
    rate_frame.to_csv(rate_path, index=False)
    off_frame.to_csv(off_path, index=False)
    protocol_text = json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    assert_no_local_routing(protocol_text)
    assert_no_local_routing(interpretation)
    protocol_path.write_text(protocol_text, encoding="utf-8")
    interpretation_path.write_text(interpretation, encoding="utf-8")
    write_sha256_manifest(
        [diagnosis_path, rate_path, off_path, protocol_path, interpretation_path],
        output_dir / "p0c_sha256.txt",
    )
    return {"identity_holds": identity_holds, "identity_gap": identity_gap, "stop_classes": diagnosis_frame["stop_class"].value_counts().to_dict()}


def archived_n1_predictions(stability_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(stability_csv)
    fixed = frame[(frame["method"] == "fixed_physics") & (frame["configuration"] == "reference") & (frame["N_cells"] == 1)]
    pivot = fixed.pivot_table(index=["window_id", "split_role", "phase"], columns="target", values="predicted_value", aggfunc="first")
    pivot = pivot.reset_index()
    rename = {
        "CO_dry_vol_pct": "n1_CO_dry_vol_pct",
        "CO2_dry_vol_pct": "n1_CO2_dry_vol_pct",
        "H2_dry_vol_pct": "n1_H2_dry_vol_pct",
        "H2_CO_ratio": "n1_H2_CO_ratio",
        "dry_syngas_flow": "n1_dry_flow_nm3_h",
    }
    return pivot.rename(columns=rename)


def _family_rates(r_het: Mapping[str, float], r_homo: Mapping[str, float]) -> dict[str, float]:
    return {
        "WGS": float(r_homo.get("WGS", 0.0)),
        "char_oxidation": float(r_het.get("C+O2", 0.0)),
        "volatile_oxidation": float(r_homo.get("CH4_Ox", 0.0)) + float(r_homo.get("CO_Ox", 0.0)) + float(r_homo.get("H2_Ox", 0.0)),
        "char_gasification": float(r_het.get("C+H2O", 0.0)) + float(r_het.get("C+CO2", 0.0)) + float(r_het.get("C+H2", 0.0)),
    }


def mesh_window_record(row: Mapping[str, Any], n_cells: int, n1: Mapping[str, float] | None) -> tuple[dict[str, Any], dict[str, float]]:
    solved = solve_reference_case(row, n_cells=n_cells)
    system = solved["system"]
    profile = solved["profile"]
    closure, projection, gap, combustion = _identity_from_solve(solved)
    cell0 = system.cells[0].residual_components(np.asarray(profile[0], dtype=float))
    names = list(cell0["names"])
    kpis = _dry_kpis(profile)
    families = {"WGS": 0.0, "char_oxidation": 0.0, "volatile_oxidation": 0.0, "char_gasification": 0.0}
    for cell, state_row in zip(system.cells, profile):
        evaluated = evaluate_cell_state(cell, np.asarray(state_row, dtype=float))
        part = _family_rates(evaluated["r_het"], evaluated["r_homo"])
        for key, value in part.items():
            families[key] += value
    downstream = [str(index) for index, flag in enumerate(combustion) if index > 0 and not flag]
    record = {
        "dataset_id": DATASET_ID,
        "window_id": row["window_id"],
        "split_role": row["split_role"],
        "phase": int(row["phase_number"]),
        "N_cells": int(n_cells),
        "execution_status": "completed",
        "fallback_count": int(system.solve_stats.get("fallback_count", 0)),
        "cell0_residual_cost": float(cell0["cost"]),
        "poor_quality": bool(float(cell0["cost"]) > POOR_CONVERGENCE_COST_THRESHOLD),
        "cell0_H2_mol_s": float(cell0["unscaled"][names.index("gas_H2")]),
        "cell0_CO2_mol_s": float(cell0["unscaled"][names.index("gas_CO2")]),
        "pred_H2_dry_vol_pct": kpis["pred_H2_dry_vol_pct"],
        "pred_CO_dry_vol_pct": kpis["pred_CO_dry_vol_pct"],
        "pred_CO2_dry_vol_pct": kpis["pred_CO2_dry_vol_pct"],
        "pred_H2_CO_ratio": kpis["pred_H2_CO_ratio"],
        "pred_dry_flow_nm3_h": kpis["pred_dry_flow_nm3_h"],
        "C_closure_error_mol_s": closure["C_closure_error_mol_s"],
        "H_closure_error_mol_s": closure["H_closure_error_mol_s"],
        "element_species_identity_max_abs_mol_s": gap,
        "cell0_is_combustion": bool(combustion[0]),
        "downstream_cells_outside_combustion": ";".join(downstream),
        "downstream_left_combustion_zone": bool(downstream),
        "plant_CO_dry_vol_pct": float(row["co_vol_pct"]),
        "plant_CO2_dry_vol_pct": float(row["co2_vol_pct"]),
        "plant_H2_dry_vol_pct": float(row["h2_vol_pct"]),
        "plant_H2_CO_ratio": float(row["h2_co_ratio"]),
        "plant_dry_flow_nm3_h": float(row["syngas_flow_mean"]),
    }
    if n1 is not None:
        record["delta_CO_vs_n1_vol_pct"] = kpis["pred_CO_dry_vol_pct"] - float(n1["n1_CO_dry_vol_pct"])
        record["delta_CO2_vs_n1_vol_pct"] = kpis["pred_CO2_dry_vol_pct"] - float(n1["n1_CO2_dry_vol_pct"])
        record["delta_H2_vs_n1_vol_pct"] = kpis["pred_H2_dry_vol_pct"] - float(n1["n1_H2_dry_vol_pct"])
        record["delta_dry_flow_vs_n1_nm3_h"] = kpis["pred_dry_flow_nm3_h"] - float(n1["n1_dry_flow_nm3_h"])
    return record, families


def effect_matrix_from_rows(frame: pd.DataFrame, *, source_commit_hash: str) -> pd.DataFrame:
    metric_targets = (
        ("CO_dry_vol_pct", "pred_CO_dry_vol_pct", "plant_CO_dry_vol_pct"),
        ("CO2_dry_vol_pct", "pred_CO2_dry_vol_pct", "plant_CO2_dry_vol_pct"),
        ("H2_dry_vol_pct", "pred_H2_dry_vol_pct", "plant_H2_dry_vol_pct"),
        ("H2_CO_ratio", "pred_H2_CO_ratio", "plant_H2_CO_ratio"),
        ("dry_syngas_flow", "pred_dry_flow_nm3_h", "plant_dry_flow_nm3_h"),
    )
    rows = []
    for (split_role, n_cells), group in frame.groupby(["split_role", "N_cells"]):
        for target, pred_col, plant_col in metric_targets:
            completed = group[group["execution_status"] == "completed"]
            residual = completed[pred_col].astype(float).to_numpy() - completed[plant_col].astype(float).to_numpy()
            residual = residual[np.isfinite(residual)]
            rows.append(
                {
                    "dataset_id": DATASET_ID,
                    "split_role": split_role,
                    "N_cells": int(n_cells),
                    "target": target,
                    "rmse": float(np.sqrt(np.mean(residual * residual))) if residual.size else math.nan,
                    "mae": float(np.mean(np.abs(residual))) if residual.size else math.nan,
                    "bias": float(np.mean(residual)) if residual.size else math.nan,
                    "n_completed": int(residual.size),
                    "source_commit": source_commit_hash,
                }
            )
    return pd.DataFrame(rows).sort_values(["N_cells", "split_role", "target"]).reset_index(drop=True)


def one_sided_channel_rows(row: Mapping[str, Any], n_cells: int, base_co: float, cell0_cost: float) -> list[dict[str, Any]]:
    records = []
    poor = bool(cell0_cost > POOR_CONVERGENCE_COST_THRESHOLD)
    for name, step in M4_ONE_SIDED_STEP.items():
        base = M4_BASELINE[name]
        plus = base + step
        op = inject_alpha({}, name, plus)
        solved = solve_reference_case(row, n_cells=n_cells, extra_op_conds=op)
        plus_co = _dry_kpis(solved["profile"])["pred_CO_dry_vol_pct"]
        records.append(
            {
                "dataset_id": DATASET_ID,
                "window_id": row["window_id"],
                "phase": int(row["phase_number"]),
                "split_role": row["split_role"],
                "N_cells": int(n_cells),
                "alpha_name": name,
                "base_alpha": base,
                "perturbation": step,
                "plus_alpha": plus,
                "base_CO_dry_vol_pct": float(base_co),
                "plus_CO_dry_vol_pct": plus_co,
                "delta_CO_vol_pct": plus_co - float(base_co),
                "one_sided_dCO_dalpha": (plus_co - float(base_co)) / step,
                "cell0_residual_cost": float(cell0_cost),
                "interpretability": "not_interpretable_due_to_residual" if poor else "interpretable",
            }
        )
    return records


def _phase_median_ids(variant_matrix_csv: Path) -> list[str]:
    frame = pd.read_csv(variant_matrix_csv)
    reference = frame[frame["configuration"] == "reference"]
    picked = []
    for phase in (1, 2, 3, 4):
        mask = reference["selection_reason"].astype(str).str.contains(f"phase{phase}_median_oc")
        ids = reference.loc[mask, "window_id"].astype(str).tolist()
        if len(ids) != 1:
            raise RuntimeError(f"expected one phase{phase} median window, found {ids}")
        picked.append(ids[0])
    return picked


def _train_ids_by_phase(rows: Sequence[Mapping[str, Any]]) -> dict[int, str]:
    picked: dict[int, str] = {}
    for row in rows:
        phase = int(row["phase_number"])
        if row["split_role"] == "train" and phase not in picked:
            picked[phase] = str(row["window_id"])
    return picked


def render_mesh_interpretation(effects: pd.DataFrame, rows: pd.DataFrame, *, finer_subset: bool) -> str:
    test = effects[(effects["split_role"] == "test") & (effects["target"].isin(["CO_dry_vol_pct", "H2_dry_vol_pct"]))]
    lines = ["# P0-B 网格判断", ""]
    poor_rates = rows.groupby("N_cells")["poor_quality"].mean()
    if bool((poor_rates >= 0.999).all()):
        lines.append("所有已跑网格的 cell-0 都超过 1e-4。不把更细网格选成更真实的主模型。后续通道归因继续停用。")
    else:
        lines.append("cell-0 超阈值的比例随网格变化，见下表。")
    if finer_subset:
        lines.append("N=9 只跑了 test 加每个 phase 一个 train 窗，不能代替全量结论。")
    lines.extend(["", "| N | split | target | RMSE | n |", "|---:|---|---|---:|---:|"])
    for _, record in test.sort_values(["N_cells", "target"]).iterrows():
        lines.append(
            f"| {int(record['N_cells'])} | {record['split_role']} | {record['target']} | {float(record['rmse']):.6g} | {int(record['n_completed'])} |"
        )
    h2 = rows.groupby("N_cells")["pred_H2_dry_vol_pct"].median()
    lines.extend(["", "出口干基 H2 的中位数（vol%）："])
    for n_cells, value in h2.items():
        lines.append(f"- N={int(n_cells)}: {float(value):.6g}")
    left = int(rows["downstream_left_combustion_zone"].sum())
    lines.append("")
    lines.append(f"下游单元离开燃烧区的窗口数：{left} / {len(rows)}。")
    return "\n".join(lines) + "\n"


def run_mesh(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    stability_csv: Path,
    variant_matrix_csv: Path,
    n1_reproduction_json: Path,
) -> dict[str, Any]:
    silence_solver_logs()
    output_dir.mkdir(parents=True, exist_ok=True)
    commit = source_commit()
    by_id = {str(row["window_id"]): row for row in rows}
    archived = archived_n1_predictions(stability_csv).set_index("window_id")
    timing_id = _train_ids_by_phase(rows)[1]
    timing_row = by_id[timing_id]
    started = time.perf_counter()
    solve_reference_case(timing_row, n_cells=1)
    n1_time = time.perf_counter() - started
    started = time.perf_counter()
    solve_reference_case(timing_row, n_cells=9)
    n9_time = time.perf_counter() - started
    finer_subset = bool(n9_time > N9_FULL_SPLIT_MAX_WALL_TIME_S)
    test_ids = [str(row["window_id"]) for row in rows if row["split_role"] == "test"]
    train_ids = list(_train_ids_by_phase(rows).values())
    median_ids = _phase_median_ids(variant_matrix_csv)
    n9_ids = sorted(set(test_ids + train_ids + median_ids)) if finer_subset else [str(row["window_id"]) for row in rows]
    plan = {3: [str(row["window_id"]) for row in rows], 5: [str(row["window_id"]) for row in rows], 9: n9_ids}
    prediction_rows = []
    reaction_rows = []
    for n_cells, window_ids in plan.items():
        for index, window_id in enumerate(window_ids, start=1):
            print(f"P0-B N={n_cells} {index}/{len(window_ids)} {window_id}", flush=True)
            row = by_id[window_id]
            n1 = archived.loc[window_id].to_dict() if window_id in archived.index else None
            try:
                record, families = mesh_window_record(row, n_cells, n1)
            except Exception as exc:
                record = {
                    "dataset_id": DATASET_ID,
                    "window_id": window_id,
                    "split_role": row["split_role"],
                    "phase": int(row["phase_number"]),
                    "N_cells": n_cells,
                    "execution_status": f"error:{type(exc).__name__}",
                    "poor_quality": True,
                    "downstream_left_combustion_zone": False,
                    "pred_CO_dry_vol_pct": math.nan,
                    "pred_CO2_dry_vol_pct": math.nan,
                    "pred_H2_dry_vol_pct": math.nan,
                    "pred_H2_CO_ratio": math.nan,
                    "pred_dry_flow_nm3_h": math.nan,
                    "plant_CO_dry_vol_pct": float(row["co_vol_pct"]),
                    "plant_CO2_dry_vol_pct": float(row["co2_vol_pct"]),
                    "plant_H2_dry_vol_pct": float(row["h2_vol_pct"]),
                    "plant_H2_CO_ratio": float(row["h2_co_ratio"]),
                    "plant_dry_flow_nm3_h": float(row["syngas_flow_mean"]),
                }
                families = {"WGS": 0.0, "char_oxidation": 0.0, "volatile_oxidation": 0.0, "char_gasification": 0.0}
            prediction_rows.append(record)
            for family, value in families.items():
                reaction_rows.append(
                    {
                        "dataset_id": DATASET_ID,
                        "window_id": window_id,
                        "split_role": row["split_role"],
                        "phase": int(row["phase_number"]),
                        "N_cells": n_cells,
                        "reaction_family": family,
                        "integrated_rate_mol_s": value,
                        "execution_status": record["execution_status"],
                    }
                )
    predictions = pd.DataFrame(prediction_rows)
    effects = effect_matrix_from_rows(predictions, source_commit_hash=commit)
    channel_rows = []
    for n_cells in plan:
        for window_id in median_ids:
            if window_id not in plan[n_cells]:
                continue
            base = predictions[(predictions["window_id"] == window_id) & (predictions["N_cells"] == n_cells)]
            if base.empty or str(base.iloc[0]["execution_status"]) != "completed":
                continue
            print(f"P0-B channel N={n_cells} {window_id}", flush=True)
            channel_rows.extend(
                one_sided_channel_rows(
                    by_id[window_id],
                    n_cells,
                    float(base.iloc[0]["pred_CO_dry_vol_pct"]),
                    float(base.iloc[0]["cell0_residual_cost"]),
                )
            )
    interpretation = render_mesh_interpretation(effects, predictions, finer_subset=finer_subset)
    protocol = {
        "dataset_id": DATASET_ID,
        "grids": [3, 5, 9],
        "n1_reproduction_rerun": False,
        "n1_metric_source": "archived fixed_physics reference predictions",
        "timing_window_id": timing_id,
        "n1_one_window_wall_time_s": n1_time,
        "n9_one_window_wall_time_s": n9_time,
        "n9_full_split_max_wall_time_s": N9_FULL_SPLIT_MAX_WALL_TIME_S,
        "finer_mesh_subset": finer_subset,
        "n9_window_count": len(n9_ids),
        "channel_windows": "one phase-median window per phase",
        "channel_difference": "one-sided, M4 perturbation magnitudes, positive direction from the fixed-physics baseline",
        "added_columns": ["cell0_residual_cost", "pred_H2_dry_vol_pct", "element_species_identity_max_abs_mol_s", "downstream_cells_outside_combustion"],
        "source_commit": commit,
        "handoff_model_head": HANDOFF_MODEL_HEAD,
        "solver_modules_match_handoff_head": commit.startswith(HANDOFF_MODEL_HEAD),
        "residual_cost_threshold": POOR_CONVERGENCE_COST_THRESHOLD,
    }
    paths = {
        "rows": output_dir / "p0b_row_predictions.csv",
        "effects": output_dir / "p0b_effect_matrix.csv",
        "reactions": output_dir / "p0b_reaction_contribution.csv",
        "channels": output_dir / "p0b_channel_response.csv",
        "protocol": output_dir / "p0b_protocol.json",
        "interpretation": output_dir / "p0b_interpretation.md",
    }
    predictions.to_csv(paths["rows"], index=False)
    effects.to_csv(paths["effects"], index=False)
    pd.DataFrame(reaction_rows).to_csv(paths["reactions"], index=False)
    pd.DataFrame(channel_rows).to_csv(paths["channels"], index=False)
    protocol_text = json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    assert_no_local_routing(protocol_text)
    assert_no_local_routing(interpretation)
    paths["protocol"].write_text(protocol_text, encoding="utf-8")
    paths["interpretation"].write_text(interpretation, encoding="utf-8")
    manifest = [paths["rows"], paths["effects"], paths["reactions"], paths["channels"], paths["protocol"], paths["interpretation"]]
    if n1_reproduction_json.exists():
        manifest.insert(0, n1_reproduction_json)
    write_sha256_manifest(manifest, output_dir / "p0b_sha256.txt")
    return {"finer_mesh_subset": finer_subset, "n9_time_s": n9_time, "n1_time_s": n1_time}
