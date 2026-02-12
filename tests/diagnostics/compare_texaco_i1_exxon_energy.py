"""
compare_texaco_i1_exxon_energy.py
==================================

对 Texaco_I-1 和 Texaco_Exxon 进行：
1. 工况差异分析
2. 轴向能量平衡诊断
3. 温度突降 cell 识别

运行: PYTHONPATH=src python tests/diagnostics/compare_texaco_i1_exxon_energy.py
"""

import os
import sys
from typing import Dict, Any, List, Tuple

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

from model.gasifier_system import GasifierSystem
from model.original_paper_loader import load_original_paper_cases, get_original_paper_json_path
from model.state import StateVector
from model.material import MaterialService
from model.constants import PhysicalConstants

# 反应焓 (J/mol)，放热为正
DH = {
    "CO_Ox": 283_000.0, "H2_Ox": 241_800.0, "CH4_Ox": 802_000.0,
    "C+O2": 393_500.0,
    "C+H2O": -131_000.0, "C+CO2": -172_000.0, "C+H2": 75_000.0,
    "WGS": 41_000.0, "MSR": -206_000.0,
}


def _build_op_conds(inp: dict, coal_flow: float) -> dict:
    return {
        "coal_flow": coal_flow,
        "o2_flow": coal_flow * inp["Ratio_OC"],
        "steam_flow": coal_flow * inp["Ratio_SC"],
        "P": inp["P"],
        "T_in": inp["TIN"],
        "HeatLossPercent": inp.get("HeatLossPercent", 3.0),
        "SlurryConcentration": inp.get("SlurryConcentration", 100.0),
    }


def compare_operating_conditions(
    case1: dict, case2: dict, coal_db: dict, name1: str, name2: str
) -> None:
    """工况差异对比"""
    inp1, inp2 = case1["inputs"], case2["inputs"]
    c1, c2 = coal_db[inp1["coal"]], coal_db[inp2["coal"]]

    print("\n" + "=" * 70)
    print("工况差异分析: Texaco_I-1 vs Texaco_Exxon")
    print("=" * 70)

    def fmt(k, v1, v2, unit=""):
        diff = (v2 - v1) if isinstance(v1, (int, float)) else "N/A"
        return f"  {k:25s}: {name1} = {v1}{unit}  |  {name2} = {v2}{unit}  |  Δ = {diff}"

    coal_f1 = inp1["FeedRate"] / 3600.0
    coal_f2 = inp2["FeedRate"] / 3600.0
    print(fmt("煤进料 (kg/s)", coal_f1, coal_f2))
    print(fmt("O2/煤比", inp1["Ratio_OC"], inp2["Ratio_OC"]))
    print(fmt("蒸汽/煤比", inp1["Ratio_SC"], inp2["Ratio_SC"]))
    print(fmt("入口温度 (K)", inp1["TIN"], inp2["TIN"]))
    print(fmt("热损失 (%)", inp1.get("HeatLossPercent", 3), inp2.get("HeatLossPercent", 3)))
    print(fmt("浆液浓度 (%)", inp1.get("SlurryConcentration", 100), inp2.get("SlurryConcentration", 100)))

    o2_1 = coal_f1 * inp1["Ratio_OC"]
    o2_2 = coal_f2 * inp2["Ratio_OC"]
    print(fmt("O2 流量 (kg/s)", o2_1 * 32e-3, o2_2 * 32e-3))
    st_1 = coal_f1 * inp1["Ratio_SC"]
    st_2 = coal_f2 * inp2["Ratio_SC"]
    print(fmt("蒸汽流量 (kg/s)", st_1, st_2))

    print("\n煤质对比:")
    for k in ["Cd", "Hd", "Od", "Ad", "HHV_d"]:
        v1 = c1.get(k, 0)
        v2 = c2.get(k, 0)
        if k == "HHV_d" and v1 > 1000:
            v1, v2 = v1 / 1000, v2 / 1000
            u = " MJ/kg"
        else:
            u = ""
        print(f"  {k:10s}: {name1} = {v1}{u}  |  {name2} = {v2}{u}")

    print("\n关键差异:")
    print(f"  - Texaco_I-1: 煤量小(76.7 g/s), O2/煤比高(0.87), 无浆液水")
    print(f"  - Texaco_Exxon: 煤量大(126 g/s), O2/煤比低(0.79), 蒸汽比高(0.50)")
    print(f"  - Exxon 挥发分更低(原料为真空塔底), 燃烧区放热相对少")
    print(f"  - Exxon 碳转化 99%, 出口温度 1226°C vs I-1 实验 1370°C → I-1 出口温度更高")


def _compute_axial_energy_audit(
    system: GasifierSystem, results: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """
    逐 cell 计算: delta_H, energy_src, Q_loss, Q_rxn 分解, 温度
    """
    n = len(system.cells)
    delta_H = np.zeros(n)
    energy_src_arr = np.zeros(n)
    Q_loss_arr = np.zeros(n)
    T_arr = np.zeros(n)
    Q_rxn = {k: np.zeros(n) for k in DH.keys()}

    C_fed = system.W_dry * (system.coal_props.get("Cd", 60.0) / 100.0)
    L_total = system.geometry["L"]

    for i, cell in enumerate(system.cells):
        state_in = cell.inlet
        state_out = StateVector.from_array(results[i], P=cell.inlet.P, z=cell.z)

        g_src, s_src, e_src = np.zeros(8), 0.0, 0.0
        for s in cell.sources:
            g, sm, e = s.get_sources(cell.idx, cell.z, cell.dz)
            g_src += g
            s_src += sm
            e_src += e

        H_in = MaterialService.get_total_enthalpy(state_in, system.coal_props)
        Ts_out = getattr(state_out, "T_solid", state_out.T)
        H_out = MaterialService.get_total_enthalpy(
            state_out, system.coal_props, T_solid_override=Ts_out
        )

        delta_H[i] = H_out - H_in
        energy_src_arr[i] = e_src

        loss_pct = cell.op_conds.get("HeatLossPercent", 2.0)
        hhv = system.coal_props.get("HHV_d", 30.0)
        if hhv > 1000:
            hhv = hhv / 1000.0
        Q_total = system.op_conds["coal_flow"] * hhv * 1e6
        Q_loss_arr[i] = (loss_pct / 100.0) * Q_total * (cell.dz / L_total)

        T_arr[i] = state_out.T

        # 反应热 (用出口状态近似，因已收敛)
        phys = cell._calc_physics_props(state_out, C_fed)
        Ts_in = state_in.T_solid_or_gas
        tau = cell.dz / max(phys["v_g"], PhysicalConstants.MIN_SLIP_VELOCITY)
        Ts_avg, _ = cell._calc_particle_temperature(
            state_out.T, Ts_in, tau, phys["d_p"],
            state_out.solid_mass, state_out.carbon_fraction
        )
        r_het, r_homo, phi = cell._calc_rates(state_out, phys, g_src, Ts_avg=Ts_avg)

        Q_rxn["CO_Ox"][i] = r_homo.get("CO_Ox", 0.0) * DH["CO_Ox"]
        Q_rxn["H2_Ox"][i] = r_homo.get("H2_Ox", 0.0) * DH["H2_Ox"]
        Q_rxn["CH4_Ox"][i] = r_homo.get("CH4_Ox", 0.0) * DH["CH4_Ox"]
        Q_rxn["C+O2"][i] = r_het.get("C+O2", 0.0) * DH["C+O2"]
        Q_rxn["C+H2O"][i] = r_het.get("C+H2O", 0.0) * DH["C+H2O"]
        Q_rxn["C+CO2"][i] = r_het.get("C+CO2", 0.0) * DH["C+CO2"]
        Q_rxn["WGS"][i] = r_homo.get("WGS", 0.0) * DH["WGS"]
        Q_rxn["MSR"][i] = r_homo.get("MSR", 0.0) * DH["MSR"]

    return delta_H, energy_src_arr, Q_loss_arr, T_arr, Q_rxn, np.array(system.z_positions)


def _find_temperature_drops(T_arr: np.ndarray, z_arr: np.ndarray, threshold_K: float = 80.0) -> List[Tuple[int, float, float]]:
    """找出温度突降的 cell: (idx, dT_K, dT_per_m)"""
    drops = []
    for i in range(1, len(T_arr)):
        dT = T_arr[i] - T_arr[i - 1]
        dz = z_arr[i] - z_arr[i - 1] if i < len(z_arr) else 0.05
        if dT < -threshold_K and dz > 1e-9:
            drops.append((i, dT, dT / dz))
    return drops


def _print_energy_audit(
    name: str,
    z_arr: np.ndarray,
    delta_H: np.ndarray,
    energy_src: np.ndarray,
    Q_loss: np.ndarray,
    Q_rxn: Dict[str, np.ndarray],
    T_arr: np.ndarray,
    drops: List[Tuple[int, float, float]],
) -> None:
    """打印轴向能量平衡与突降 cell"""
    print("\n" + "=" * 70)
    print(f"轴向能量平衡: {name}")
    print("=" * 70)

    Q_ox = Q_rxn["CO_Ox"] + Q_rxn["H2_Ox"] + Q_rxn["CH4_Ox"] + Q_rxn["C+O2"]
    Q_gasif = Q_rxn["C+H2O"] + Q_rxn["C+CO2"] + Q_rxn["MSR"]

    print("\n温度突降 cell (|dT| > 80 K):")
    if not drops:
        print("  无显著突降")
    else:
        for idx, dT, dT_dz in drops[:15]:
            t_in = T_arr[idx - 1] - 273.15
            t_out = T_arr[idx] - 273.15
            print(f"  Cell {idx:3d}  z={z_arr[idx]:.3f}m:  {t_in:.0f}°C -> {t_out:.0f}°C  dT={dT:.0f}K  dT/dz={dT_dz:.0f} K/m")
            print(f"       ΔH={delta_H[idx]/1e6:.3f} MW  energy_src={energy_src[idx]/1e6:.3f} MW  Q_loss={Q_loss[idx]/1e6:.4f} MW")
            print(f"       Q_ox={Q_ox[idx]/1e6:.3f} MW  Q_gasif={Q_gasif[idx]/1e6:.3f} MW (C+H2O={Q_rxn['C+H2O'][idx]/1e6:.2f} C+CO2={Q_rxn['C+CO2'][idx]/1e6:.2f})")

    print("\n前 10 格 + 突降格 能量明细:")
    show_idx = set(range(min(10, len(z_arr))))
    for idx, _, _ in drops[:5]:
        show_idx.add(idx)
        show_idx.add(idx - 1)
    for i in sorted(show_idx):
        if i >= len(z_arr):
            continue
        bal = delta_H[i] - (energy_src[i] - Q_loss[i])
        print(f"  cell {i:3d} z={z_arr[i]:.3f} T={T_arr[i]-273.15:6.0f}°C  "
              f"ΔH={delta_H[i]/1e6:6.2f}  e_src={energy_src[i]/1e6:6.2f}  Q_loss={Q_loss[i]/1e6:5.3f}  "
              f"Q_ox={Q_ox[i]/1e6:5.2f}  Q_gasif={Q_gasif[i]/1e6:5.2f}  balance_err={bal/1e6:.3f}")

    print("\n反应热积分 (全炉):")
    Q_ox_tot = np.sum(Q_ox)
    Q_gasif_tot = np.sum(Q_gasif)
    Q_wgs = np.sum(Q_rxn["WGS"])
    print(f"  Q_ox (放热):     {Q_ox_tot/1e6:.2f} MW")
    print(f"  Q_gasif (吸热):  {Q_gasif_tot/1e6:.2f} MW")
    print(f"  Q_WGS:           {Q_wgs/1e6:.2f} MW")
    print(f"  energy_src 总和: {np.sum(energy_src)/1e6:.2f} MW (蒸发等)")
    print(f"  Q_loss 总和:     {np.sum(Q_loss)/1e6:.2f} MW")


def main():
    logging_level = getattr(__import__("logging"), "WARNING", None)
    import logging
    logging.getLogger().setLevel(logging.WARNING)
    for n in ["model", "model.gasifier_system", "model.cell"]:
        logging.getLogger(n).setLevel(logging.WARNING)

    json_path = get_original_paper_json_path()
    coal_db, cases, metadata = load_original_paper_cases(json_path)
    geom = metadata.get("reactor_dimensions", {})
    geometry = {"L": float(geom.get("length_m", 6.096)), "D": float(geom.get("diameter_m", 1.524))}

    target = ["Texaco_I-1", "Texaco_Exxon"]
    for n in target:
        if n not in cases:
            print(f"Case {n} not found")
            return

    compare_operating_conditions(
        cases["Texaco_I-1"],
        cases["Texaco_Exxon"],
        coal_db,
        "Texaco_I-1",
        "Texaco_Exxon",
    )

    for name in target:
        case = cases[name]
        inp = case["inputs"]
        coal_props = coal_db[inp["coal"]]
        coal_flow = inp["FeedRate"] / 3600.0
        op_conds = _build_op_conds(inp, coal_flow)

        system = GasifierSystem(geometry, coal_props, op_conds)
        results, z_grid = system.solve(N_cells=60)

        delta_H, energy_src, Q_loss, T_arr, Q_rxn, z_arr = _compute_axial_energy_audit(
            system, results
        )
        drops = _find_temperature_drops(T_arr, z_arr, threshold_K=80.0)

        _print_energy_audit(
            name, z_arr, delta_H, energy_src, Q_loss, Q_rxn, T_arr, drops
        )

    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    print("  - Texaco_I-1 出口 801°C vs Exxon 1226°C: I-1 轴向降温更剧烈")
    print("  - 检查突降 cell 的 Q_gasif (C+H2O, C+CO2) 是否异常")
    print("  - 若气化吸热在突降格过大，可能是求解器陷入低温解或动力学过强")


if __name__ == "__main__":
    main()
