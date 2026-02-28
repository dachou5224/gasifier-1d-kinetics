"""
鲁南 Texaco 工况逐 Cell 严格能量审计

目的：合成气组分接近工业数据，但出口温度偏低（829°C vs 1350°C），
通过逐格能量平衡找出温度偏差的根本原因。

能量守恒：H_out = H_in + e_src - Q_loss + Σ Q_rxn
  - e_src: 源项（浆液蒸发吸热为负、热解产物焓等）
  - Q_loss: 壁面散热
  - Q_rxn: 反应热（放热为正，吸热为负）

运行: PYTHONPATH=src python tests/diagnostics/audit_lunan_energy.py
"""

import os
import sys
import logging

logging.getLogger().setLevel(logging.WARNING)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
from model.gasifier_system import GasifierSystem
from model.chemistry import COAL_DATABASE
from model.state import StateVector

# 反应焓 J/mol，放热为正
DH = {
    "CO_Ox": 283_000.0, "H2_Ox": 241_800.0, "CH4_Ox": 802_000.0,
    "C+O2": 393_500.0,
    "C+H2O": -131_000.0, "C+CO2": -172_000.0, "C+H2": 75_000.0,
    "WGS": 41_000.0, "MSR": -206_000.0,
}


def _state_average(s_in, s_out, P, z):
    return StateVector(
        gas_moles=(s_in.gas_moles + s_out.gas_moles) * 0.5,
        solid_mass=0.5 * (s_in.solid_mass + s_out.solid_mass),
        carbon_fraction=0.5 * (s_in.carbon_fraction + s_out.carbon_fraction),
        T=0.5 * (s_in.T + s_out.T),
        P=P,
        z=z,
    )


def _axial_energy_audit(system, results):
    """逐 cell 严格能量审计"""
    from model.material import MaterialService

    n = len(system.cells)
    z = np.array(system.z_positions)
    C_fed = system.W_dry * (system.coal_props.get("Cd", 60.0) / 100.0)

    rows = []
    for i, cell in enumerate(system.cells):
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
        state_avg = _state_average(state_in, state_out, cell.inlet.P, cell.z)

        g_src = np.zeros(8)
        s_src = 0.0
        e_src = 0.0
        for s in cell.sources:
            g, sm, e = s.get_sources(cell.idx, cell.z, cell.dz)
            g_src += g
            s_src += sm
            e_src += e

        phys = cell._calc_physics_props(state_avg, C_fed)
        Ts_in = cell.inlet.T_solid_or_gas
        tau = cell.dz / max(phys["v_g"], 1e-3)
        Ts_avg, Ts_out = cell._calc_particle_temperature(
            state_avg.T, Ts_in, tau, phys["d_p"],
            state_avg.solid_mass, state_avg.carbon_fraction
        )
        r_het, r_homo, phi = cell._calc_rates(state_avg, phys, g_src, Ts_avg=Ts_avg)

        H_in = MaterialService.get_total_enthalpy(state_in, system.coal_props)
        H_out = MaterialService.get_total_enthalpy(state_out, system.coal_props, T_solid_override=Ts_out)

        L_total = cell.op_conds.get("L_reactor", 6.0)
        loss_pct = cell.op_conds.get("HeatLossPercent", 2.0)
        hhv = system.coal_props.get("HHV_d", 30.0)
        if hhv > 1000:
            hhv = hhv / 1000.0
        Q_loss = (loss_pct / 100.0) * (system.op_conds["coal_flow"] * hhv * 1e6) * (cell.dz / L_total)

        Q_ox = (r_homo.get("CO_Ox", 0) * DH["CO_Ox"] + r_homo.get("H2_Ox", 0) * DH["H2_Ox"] +
                r_homo.get("CH4_Ox", 0) * DH["CH4_Ox"] + r_het.get("C+O2", 0) * DH["C+O2"] * (phi or 1.0))
        Q_gasif = (r_het.get("C+H2O", 0) * DH["C+H2O"] + r_het.get("C+CO2", 0) * DH["C+CO2"])
        Q_wgs = r_homo.get("WGS", 0) * DH["WGS"]
        Q_msr = r_homo.get("MSR", 0) * DH["MSR"]
        Q_ch4 = r_het.get("C+H2", 0) * DH["C+H2"]

        Q_rxn_total = Q_ox + Q_gasif + Q_wgs + Q_msr + Q_ch4
        res_E = H_out - (H_in + e_src - Q_loss + Q_rxn_total)

        rows.append({
            "i": i,
            "z": z[i],
            "dz": cell.dz,
            "T_in": state_in.T,
            "T_out": state_out.T,
            "dT": state_out.T - state_in.T,
            "H_in": H_in,
            "H_out": H_out,
            "dH": H_out - H_in,
            "energy_src": e_src,
            "Q_loss": Q_loss,
            "Q_ox": Q_ox,
            "Q_gasif": Q_gasif,
            "Q_wgs": Q_wgs,
            "Q_msr": Q_msr,
            "Q_ch4": Q_ch4,
            "Q_rxn_total": Q_rxn_total,
            "res_E": res_E,
            "v_g": phys["v_g"],
            "tau": tau,
        })

    return rows


def main():
    coal_props = COAL_DATABASE["LuNan_Coal"]
    geometry = {"L": 6.87, "D": 1.68}
    coal_flow = 17917.0 / 3600.0
    op_conds = {
        "coal_flow": coal_flow,
        "o2_flow": coal_flow * 0.872,
        "steam_flow": 0.0,
        "P": 4e6,
        "T_in": 400.0,
        "HeatLossPercent": 2.0,
        "SlurryConcentration": 66.0,
        "AdaptiveFirstCellLength": True,
    }

    system = GasifierSystem(geometry, coal_props, op_conds)
    results, z_grid = system.solve(N_cells=40)

    audit = _axial_energy_audit(system, results)

    T_out = results[-1][10] - 273.15
    gas = results[-1][:8]
    F_dry = np.sum(gas[:7]) + 1e-12
    y_co = gas[2] / F_dry * 100
    y_h2 = gas[5] / F_dry * 100
    y_co2 = gas[3] / F_dry * 100

    print("=" * 90)
    print("鲁南 Texaco 工况 — 逐 Cell 严格能量审计")
    print("=" * 90)
    print(f"几何: L={geometry['L']}m D={geometry['D']}m  进料: {coal_flow*1000:.0f} g/s  浆液66%")
    print(f"出口: T={T_out:.1f}°C  CO={y_co:.1f}%  H2={y_h2:.1f}%  CO2={y_co2:.1f}%  [期望 T≈1350°C]")
    print("=" * 90)

    # 汇总统计
    tot_H_in = audit[0]["H_in"]
    tot_H_out = audit[-1]["H_out"]
    tot_e_src = sum(r["energy_src"] for r in audit)
    tot_Q_loss = sum(r["Q_loss"] for r in audit)
    tot_Q_ox = sum(r["Q_ox"] for r in audit)
    tot_Q_gasif = sum(r["Q_gasif"] for r in audit)
    tot_Q_wgs = sum(r["Q_wgs"] for r in audit)
    tot_Q_msr = sum(r["Q_msr"] for r in audit)
    tot_Q_ch4 = sum(r["Q_ch4"] for r in audit)
    tot_Q_rxn = sum(r["Q_rxn_total"] for r in audit)

    print("\n【全局能量平衡汇总】")
    print(f"  H_in (入口总焓):     {tot_H_in/1e6:12.2f} MW")
    print(f"  H_out (出口总焓):    {tot_H_out/1e6:12.2f} MW")
    print(f"  ΔH = H_out - H_in:   {(tot_H_out-tot_H_in)/1e6:12.2f} MW")
    print(f"  Σ energy_src:       {tot_e_src/1e6:12.2f} MW  (蒸发吸热为负)")
    print(f"  Σ Q_loss:           {-tot_Q_loss/1e6:12.2f} MW  (散热)")
    print(f"  Σ Q_ox (氧化放热):  {tot_Q_ox/1e6:12.2f} MW")
    print(f"  Σ Q_gasif (气化吸热): {tot_Q_gasif/1e6:12.2f} MW")
    print(f"  Σ Q_wgs:            {tot_Q_wgs/1e6:12.2f} MW")
    print(f"  Σ Q_msr:            {tot_Q_msr/1e6:12.2f} MW")
    print(f"  Σ Q_ch4 (甲烷化):   {tot_Q_ch4/1e6:12.2f} MW")
    print(f"  Σ Q_rxn:            {tot_Q_rxn/1e6:12.2f} MW")
    print(f"  理论: ΔH = e_src - Q_loss + Q_rxn  →  校验: {(tot_H_out-tot_H_in)/1e6:.2f} vs {(tot_e_src - tot_Q_loss + tot_Q_rxn)/1e6:.2f} MW")

    # 轴向分布：前15格 + 每10格采样 + 最后5格
    print("\n" + "=" * 90)
    print("【逐 Cell 能量分解】")
    print("=" * 90)
    print(f"{'Cell':>4} {'z(m)':>6} {'T(K)':>6} {'dT(K)':>7} {'H_in':>10} {'H_out':>10} {'e_src':>10} {'Q_loss':>9} {'Q_ox':>10} {'Q_gasif':>10} {'Q_wgs':>9} {'Q_msr':>9}")
    print("-" * 90)

    for r in audit:
        print(f"{r['i']:4d} {r['z']:6.3f} {r['T_out']:6.0f} {r['dT']:+7.0f} "
              f"{r['H_in']/1e6:10.2f} {r['H_out']/1e6:10.2f} {r['energy_src']/1e6:10.2f} "
              f"{r['Q_loss']/1e6:9.3f} {r['Q_ox']/1e6:10.2f} {r['Q_gasif']/1e6:10.2f} "
              f"{r['Q_wgs']/1e6:9.2f} {r['Q_msr']/1e6:9.2f}")

    # 温度突降分析
    print("\n" + "=" * 90)
    print("【温度突降 Cell 识别】(dT < -100K 或降温>10%)")
    print("=" * 90)
    drops = []
    for r in audit:
        if r["T_in"] > 700 and r["dT"] < -50:
            pct = -r["dT"] / r["T_in"] * 100
            drops.append((r["i"], r["z"], r["T_in"], r["T_out"], r["dT"], pct,
                          r["Q_ox"]/1e6, r["Q_gasif"]/1e6, r["energy_src"]/1e6, r["Q_loss"]/1e6))
    if drops:
        for d in drops[:20]:
            print(f"  Cell {d[0]:3d} z={d[1]:.3f}m  T {d[2]:.0f}→{d[3]:.0f}K  dT={d[4]:.0f}K ({d[5]:.1f}%)  "
                  f"Q_ox={d[6]:.2f}  Q_gasif={d[7]:.2f}  e_src={d[8]:.2f}  Q_loss={d[9]:.3f} MW")
    else:
        print("  无显著突降")

    # 诊断结论
    print("\n" + "=" * 90)
    print("【诊断结论】")
    print("=" * 90)
    cell0 = audit[0]
    print(f"  Cell 0: T_out={cell0['T_out']:.0f}K ({cell0['T_out']-273.15:.0f}°C)")
    print(f"    氧化放热 Q_ox={cell0['Q_ox']/1e6:.2f} MW, 气化吸热 Q_gasif={cell0['Q_gasif']/1e6:.2f} MW")
    print(f"    源项 e_src={cell0['energy_src']/1e6:.2f} MW (浆液蒸发吸热使 e_src<0)")
    print(f"    散热 Q_loss={cell0['Q_loss']/1e6:.3f} MW")
    if cell0["energy_src"] < -1e6:
        print("  → Cell 0 浆液蒸发吸热显著，可能导致起燃温度不足")
    net_ox = tot_Q_ox + tot_Q_gasif + tot_Q_wgs + tot_Q_msr
    print(f"  全炉净反应热 (放热-吸热): {net_ox/1e6:.2f} MW")
    print(f"  全炉散热: {tot_Q_loss/1e6:.2f} MW")
    print(f"  全炉源项(蒸发等): {tot_e_src/1e6:.2f} MW")


if __name__ == "__main__":
    main()
