"""
audit_reaction_heat_texaco.py
==============================

针对 Wen & Chaung (1979) 的 Texaco 工况 (I-1, I-2)，对当前 1D 模型的
逐 cell 反应热进行拆分与积分，便于检查：

  - 氧化放热: CO_Ox, H2_Ox, CH4_Ox, C+O2
  - 气化吸热: C+H2O, C+CO2
  - 其它: WGS, MSR, C+H2 (甲烷化)

注意：这里使用近似的标准反应焓 (ΔH_rxn, 298 K)，只做定性诊断，
不直接反馈到能量方程。

运行方式 (项目根目录):

  PYTHONPATH=src python tests/diagnostics/audit_reaction_heat_texaco.py
"""

import os
import sys
from typing import Dict, Any

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

from model.gasifier_system import GasifierSystem
from model.original_paper_loader import load_original_paper_cases, get_original_paper_json_path
from model.state import StateVector


# 近似标准反应焓 (J/mol)，采用“放热为正”的约定，便于直观理解：
DH = {
    # Homogeneous oxidation
    "CO_Ox": 283_000.0,      # CO + 0.5O2 -> CO2
    "H2_Ox": 241_800.0,      # H2 + 0.5O2 -> H2O(g)
    "CH4_Ox": 802_000.0,     # CH4 + 2O2 -> CO2 + 2H2O
    # Heterogeneous oxidation
    "C+O2": 393_500.0,       # C + O2 -> CO2 (等效)
    # Gasification (endothermic, 记为负值)
    "C+H2O": -131_000.0,     # C + H2O -> CO + H2
    "C+CO2": -172_000.0,     # C + CO2 -> 2CO
    # Methanation (放热)
    "C+H2": 75_000.0,        # 近似 C + 2H2 -> CH4
    # WGS / MSR (按 298K 近似)
    "WGS": 41_000.0,         # CO + H2O -> CO2 + H2 (略放热)
    "MSR": -206_000.0,       # CH4 + H2O -> CO + 3H2 (强吸热)
}


def _state_average(s_in: StateVector, s_out: StateVector, P: float, z: float) -> StateVector:
    """入口与出口状态的平均，用于代表该格内反应区的 T 与组成。"""
    return StateVector(
        gas_moles=(s_in.gas_moles + s_out.gas_moles) * 0.5,
        solid_mass=0.5 * (s_in.solid_mass + s_out.solid_mass),
        carbon_fraction=0.5 * (s_in.carbon_fraction + s_out.carbon_fraction),
        T=0.5 * (s_in.T + s_out.T),
        P=P,
        z=z,
    )


def _compute_reaction_heat_per_cell(system: GasifierSystem, results: np.ndarray) -> Dict[str, Any]:
    """
    利用已求解的 results 和 system.cells，重新调用 cell._calc_rates，
    估算每个 cell 的反应热分布。
    采用「入口与出口的平均状态」评估速率，使均相氧化（CH4/CO/H2 燃烧）在
    高温下体现；若仅用入口 T≈505 K，Arrhenius 会使均相氧化速率接近 0。
    """
    n_cells = len(system.cells)
    z = np.array(system.z_positions)

    Q_profiles: Dict[str, np.ndarray] = {k: np.zeros(n_cells) for k in DH.keys()}
    Q_potential_ox: Dict[str, np.ndarray] = {k: np.zeros(n_cells) for k in ["CO_Ox", "H2_Ox", "CH4_Ox", "C+O2"]}

    Cd_total = system.coal_props.get("Cd", 60.0) / 100.0
    C_fed_total = system.W_dry * Cd_total

    for i, cell in enumerate(system.cells):
        # 入口状态（cell 0 含 pyro + evap 源）
        if i == 0:
            x_in = cell.inlet.to_array().copy()
            gas_src = np.zeros(8)
            for s in cell.sources:
                g, _, _ = s.get_sources(cell.idx, cell.z, cell.dz)
                gas_src += g
            x_in[:8] += gas_src
            state_in = StateVector.from_array(x_in, P=cell.inlet.P, z=cell.z)
        else:
            state_in = StateVector.from_array(results[i - 1], P=cell.inlet.P, z=cell.z)

        state_out = StateVector.from_array(results[i], P=cell.inlet.P, z=cell.z)

        # 用「入口与出口的平均状态」算速率：既有 O2/可燃物，又有高温，均相氧化才会显现
        state_for_rates = _state_average(state_in, state_out, cell.inlet.P, cell.z)

        gas_src = np.zeros(8)
        solid_src = 0.0
        energy_src = 0.0
        for s in cell.sources:
            g, s_m, e = s.get_sources(cell.idx, cell.z, cell.dz)
            gas_src += g
            solid_src += s_m
            energy_src += e

        phys = cell._calc_physics_props(state_for_rates, C_fed_total)
        r_het, r_homo, phi = cell._calc_rates(state_for_rates, phys, gas_src)

        # 潜在氧化放热（仅前 3 格，避免下游格浓度异常导致量纲错误）
        if i < 3:
            r_homo_raw = cell.kinetics.calc_homogeneous_rates(
                state_for_rates, cell.V, inlet_state=cell.inlet, gas_src=gas_src
            )
            r_het_raw = cell.kinetics.calc_heterogeneous_rates(
                state_for_rates, phys["d_p"], phys["S_total"],
                X_total=phys["X_total"], Re=phys["Re_p"], Sc=phys["Sc_p"]
            )
            r_het_raw.pop("phi", None)
            for k in ["CO_Ox", "H2_Ox", "CH4_Ox"]:
                Q_potential_ox[k][i] = r_homo_raw.get(k, 0.0) * DH[k]
            Q_potential_ox["C+O2"][i] = r_het_raw.get("C+O2", 0.0) * DH["C+O2"]

        # 实际用于守恒的速率（经 O2 分配裁剪：CH4→H2→CO→C+O2）
        Q_profiles["CO_Ox"][i] = r_homo.get("CO_Ox", 0.0) * DH["CO_Ox"]
        Q_profiles["H2_Ox"][i] = r_homo.get("H2_Ox", 0.0) * DH["H2_Ox"]
        Q_profiles["CH4_Ox"][i] = r_homo.get("CH4_Ox", 0.0) * DH["CH4_Ox"]
        Q_profiles["C+O2"][i] = r_het.get("C+O2", 0.0) * DH["C+O2"]
        Q_profiles["C+H2O"][i] = r_het.get("C+H2O", 0.0) * DH["C+H2O"]
        Q_profiles["C+CO2"][i] = r_het.get("C+CO2", 0.0) * DH["C+CO2"]
        Q_profiles["C+H2"][i] = r_het.get("C+H2", 0.0) * DH["C+H2"]
        Q_profiles["WGS"][i] = r_homo.get("WGS", 0.0) * DH["WGS"]
        Q_profiles["MSR"][i] = r_homo.get("MSR", 0.0) * DH["MSR"]

    return {"z": z, "Q_profiles": Q_profiles, "Q_potential_ox": Q_potential_ox}


def _summarize_heat(
    case_name: str,
    z: np.ndarray,
    Q_profiles: Dict[str, np.ndarray],
    Q_potential_ox: Dict[str, np.ndarray] = None,
):
    """
    打印每个反应的总功率 (各 cell 功率之和 = 反应器总功率) 以及总放热/吸热。
    Q_profiles[key][i] 为第 i 个 cell 的功率 (W)，求和即该反应在全炉的总功率。
    """
    print(f"\n=== Reaction Heat Audit: {case_name} ===")

    Q_totals = {}
    for key, prof in Q_profiles.items():
        # 各格功率之和 = 该反应在全炉的总功率 (W)
        Q_totals[key] = float(np.sum(prof))

    # 输出分组统计
    def fmt_mw(W):
        return f"{W/1e6:8.2f} MW"

    Q_ox = Q_totals["CO_Ox"] + Q_totals["H2_Ox"] + Q_totals["CH4_Ox"] + Q_totals["C+O2"]
    Q_gasif = Q_totals["C+H2O"] + Q_totals["C+CO2"] + Q_totals["MSR"]
    Q_other = Q_totals["C+H2"] + Q_totals["WGS"]
    Q_net = Q_ox + Q_gasif + Q_other

    print("  Oxidation (放热，实际用于守恒):")
    print(f"    CO_Ox  : {fmt_mw(Q_totals['CO_Ox'])}  H2_Ox  : {fmt_mw(Q_totals['H2_Ox'])}  CH4_Ox : {fmt_mw(Q_totals['CH4_Ox'])}  C+O2 : {fmt_mw(Q_totals['C+O2'])}")
    print(f"    => Total Oxidation   : {fmt_mw(Q_ox)}")
    print("  [说明] O2 按动力学速率比例分配；C+O2 速率已乘以 0.3（焦炭燃烧较气相慢）。")
    print("  Gasification / Reforming (吸热为负):")
    print(f"    C+H2O  : {fmt_mw(Q_totals['C+H2O'])}")
    print(f"    C+CO2  : {fmt_mw(Q_totals['C+CO2'])}")
    print(f"    MSR    : {fmt_mw(Q_totals['MSR'])}")
    print(f"    => Total Gasification: {fmt_mw(Q_gasif)}")

    print("  Other:")
    print(f"    C+H2   : {fmt_mw(Q_totals['C+H2'])}")
    print(f"    WGS    : {fmt_mw(Q_totals['WGS'])}")
    print(f"    => Total Other       : {fmt_mw(Q_other)}")

    print(f"  ==> Net Reaction Heat (Ox + Gasif + Other): {fmt_mw(Q_net)}")
    if Q_totals.get("WGS", 0) < -0.1:
        print("  [注] WGS 为负表示净逆向 (CO2+H2→CO+H2O) 占优，吸热；若数值过大可能导致出口温度偏低。")


def main():
    json_path = get_original_paper_json_path()
    coal_db, cases, metadata = load_original_paper_cases(json_path)

    geom_meta = metadata.get("reactor_dimensions", {})
    L = float(geom_meta.get("length_m", 6.096))
    D = float(geom_meta.get("diameter_m", 1.524))
    geometry = {"L": L, "D": D}

    target_cases = ["Texaco_I-1", "Texaco_I-2"]

    for name in target_cases:
        if name not in cases:
            print(f"Case {name} not found, skip.")
            continue

        case = cases[name]
        inp = case["inputs"]
        coal_key = inp["coal"]
        coal_props = coal_db[coal_key]

        coal_flow = inp["FeedRate"] / 3600.0
        o2_flow = coal_flow * inp["Ratio_OC"]
        steam_flow = coal_flow * inp["Ratio_SC"]

        op_conds = {
            "coal_flow": coal_flow,
            "o2_flow": o2_flow,
            "steam_flow": steam_flow,
            "P": inp["P"],
            "T_in": inp["TIN"],
            "HeatLossPercent": inp.get("HeatLossPercent", 3.0),
            "SlurryConcentration": inp.get("SlurryConcentration", 100.0),
            "pilot_heat": 5.0e6,
        }

        system = GasifierSystem(geometry, coal_props, op_conds)
        results, z_grid = system.solve(N_cells=60)

        audit = _compute_reaction_heat_per_cell(system, results)
        _summarize_heat(
            name,
            audit["z"],
            audit["Q_profiles"],
            audit.get("Q_potential_ox"),
        )


if __name__ == "__main__":
    main()

