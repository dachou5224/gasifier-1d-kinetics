"""
compare_i1_exxon_energy.py
===========================

对 Texaco_I-1 和 Texaco_Exxon 进行：
1. 工况差异分析
2. 轴向能量平衡诊断
3. 温度突降 cell 识别

运行方式 (项目根目录):
  PYTHONPATH=src python tests/diagnostics/compare_i1_exxon_energy.py
"""

import os
import sys
import logging

logging.getLogger().setLevel(logging.WARNING)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
from model.gasifier_system import GasifierSystem
from model.original_paper_loader import load_original_paper_cases, get_original_paper_json_path
from model.state import StateVector

# 反应焓 (J/mol)，放热为正
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
    """逐 cell 能量平衡：H_in, H_out, energy_src, Q_loss, ΔH, 反应热分解."""
    from model.material import MaterialService
    
    n = len(system.cells)
    z = np.array(system.z_positions)
    dz = np.diff(z, prepend=z[0])
    dz[0] = system.cells[0].dz
    
    rows = []
    C_fed = system.W_dry * (system.coal_props.get("Cd", 60.0) / 100.0)
    
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
        
        Q_ox = (r_homo.get("CO_Ox", 0) * 283000 + r_homo.get("H2_Ox", 0) * 241800 +
                r_homo.get("CH4_Ox", 0) * 802000 + r_het.get("C+O2", 0) * 393500 * (phi or 1.0))
        Q_gasif = (r_het.get("C+H2O", 0) * (-131000) + r_het.get("C+CO2", 0) * (-172000) +
                   r_homo.get("MSR", 0) * (-206000))
        Q_wgs = r_homo.get("WGS", 0) * 41000
        
        res_E = H_out - (H_in + e_src - Q_loss)
        
        rows.append({
            "i": i,
            "z": z[i],
            "dz": cell.dz,
            "T_in": state_in.T,
            "T_out": state_out.T,
            "H_in": H_in,
            "H_out": H_out,
            "energy_src": e_src,
            "Q_loss": Q_loss,
            "Q_ox": Q_ox,
            "Q_gasif": Q_gasif,
            "Q_wgs": Q_wgs,
            "res_E": res_E,
        })
    
    return rows


def _find_temp_drops(rows, dT_thresh_K=150, pct_thresh=0.12):
    """识别温度突降的 cell：单格降温 > dT_thresh_K 或 > pct_thresh 相对降幅."""
    drops = []
    for r in rows:
        dT = r["T_out"] - r["T_in"]
        if r["T_in"] > 800:
            pct = -dT / r["T_in"]
            if dT < -dT_thresh_K or pct > pct_thresh:
                drops.append({
                    "cell": r["i"],
                    "z": r["z"],
                    "T_in": r["T_in"],
                    "T_out": r["T_out"],
                    "dT_K": dT,
                    "pct_drop": pct * 100,
                })
    return drops


def main(n_cells=60):
    json_path = get_original_paper_json_path()
    coal_db, cases, metadata = load_original_paper_cases(json_path)
    
    geom = metadata.get("reactor_dimensions", {})
    L = float(geom.get("length_m", 6.096))
    D = float(geom.get("diameter_m", 1.524))
    geometry = {"L": L, "D": D}
    
    target_names = ["Texaco_I-1", "Texaco_Exxon"]
    
    # === 1. 工况差异分析 ===
    print("=" * 80)
    print("一、Texaco_I-1 vs Texaco_Exxon 工况差异分析")
    print("=" * 80)
    
    diff_keys = [
        "coal_feed_rate_g_s", "O2_to_fuel_ratio", "steam_to_fuel_ratio",
        "inlet_temperature_K", "heat_loss_percent", "coal_type"
    ]
    
    import json
    with open(json_path) as f:
        raw_data = json.load(f)
    raw_cases = raw_data.get("validation_cases", {})

    for name in target_names:
        if name not in cases:
            continue
        c = cases[name]
        oc = c["inputs"]
        raw_oc = raw_cases.get(name, {}).get("operating_conditions", {})
        
        coal_key = oc["coal"]
        coal_props = coal_db[coal_key]
        feed_g_s = raw_oc.get("coal_feed_rate_g_s", oc["FeedRate"] * 1000 / 3600)
        
        print(f"\n--- {name} ---")
        print(f"  煤种: {coal_key}")
        print(f"  C: {coal_props.get('Cd', 0):.1f}%, H: {coal_props.get('Hd', 0):.1f}%, "
              f"O: {coal_props.get('Od', 0):.1f}%, Ash: {coal_props.get('Ad', 0):.1f}%")
        print(f"  HHV: {coal_props.get('HHV_d', 0)/1000:.1f} MJ/kg")
        print(f"  进料: {feed_g_s:.1f} g/s ({oc['FeedRate']:.1f} kg/h)")
        print(f"  O2/煤比: {oc['Ratio_OC']:.3f}")
        print(f"  蒸汽/煤比: {oc['Ratio_SC']:.3f}")
        print(f"  入口温度: {oc['TIN']:.1f} K")
        print(f"  热损失: {oc.get('HeatLossPercent', 3):.1f}%")
        print(f"  浆液浓度: {oc.get('SlurryConcentration', 100):.1f}% (100=干粉/熔渣)")
    
    # 差异对比
    c1, c2 = cases["Texaco_I-1"]["inputs"], cases["Texaco_Exxon"]["inputs"]
    print("\n--- 关键差异对比 ---")
    print(f"  进料量:   I-1 {c1['FeedRate']:.1f} kg/h  vs  Exxon {c2['FeedRate']:.1f} kg/h  (Exxon ~{c2['FeedRate']/c1['FeedRate']:.2f}x)")
    print(f"  O2/煤比:   I-1 {c1['Ratio_OC']:.3f}  vs  Exxon {c2['Ratio_OC']:.3f}")
    print(f"  蒸汽/煤比: I-1 {c1['Ratio_SC']:.3f}  vs  Exxon {c2['Ratio_SC']:.3f}  (Exxon 蒸汽多 ~{c2['Ratio_SC']/c1['Ratio_SC']:.1f}x)")
    
    # === 2. 运行计算并轴向能量诊断 ===
    print("\n" + "=" * 80)
    print("二、轴向能量平衡诊断")
    print("=" * 80)
    
    all_results = {}
    
    for name in target_names:
        if name not in cases:
            continue
        case = cases[name]
        inp = case["inputs"]
        coal_props = coal_db[inp["coal"]]
        coal_flow = inp["FeedRate"] / 3600.0
        op_conds = {
            "coal_flow": coal_flow,
            "o2_flow": coal_flow * inp["Ratio_OC"],
            "steam_flow": coal_flow * inp["Ratio_SC"],
            "P": inp["P"],
            "T_in": inp["TIN"],
            "HeatLossPercent": inp.get("HeatLossPercent", 3.0),
            "SlurryConcentration": inp.get("SlurryConcentration", 100.0),
        }
        
        system = GasifierSystem(geometry, coal_props, op_conds)
        results, z_grid = system.solve(N_cells=n_cells)
        
        audit = _axial_energy_audit(system, results)
        drops = _find_temp_drops(audit)
        
        T_cell0 = results[0][10] - 273.15
        T_out = results[-1][10] - 273.15
        
        all_results[name] = {
            "system": system,
            "results": results,
            "audit": audit,
            "drops": drops,
            "T_cell0": T_cell0,
            "T_out": T_out,
        }
        
        print(f"\n--- {name} ---")
        print(f"  Cell 0: {T_cell0:.1f} °C,  出口: {T_out:.1f} °C")
        
        # 温度突降 cell
        if drops:
            print(f"  温度突降 cell (dT<-150K 或 >12%):")
            for d in drops[:15]:
                print(f"    Cell {d['cell']:3d} z={d['z']:.3f}m  T_in={d['T_in']:.0f}K -> T_out={d['T_out']:.0f}K  dT={d['dT_K']:.0f}K ({d['pct_drop']:.1f}%)")
            if len(drops) > 15:
                print(f"    ... 共 {len(drops)} 个")
    
    # === 3. 温度突降 cell 的详细能量分解 ===
    print("\n" + "=" * 80)
    print("三、温度突降 cell 能量分解（前 5 个最剧烈）")
    print("=" * 80)
    
    for name in target_names:
        if name not in all_results:
            continue
        rec = all_results[name]
        audit = rec["audit"]
        drops = rec["drops"]
        
        if not drops:
            print(f"\n{name}: 无显著温度突降 cell")
            continue
        
        # 按 |dT| 排序取前 5
        sorted_drops = sorted(drops, key=lambda x: x["dT_K"])[:5]
        
        print(f"\n--- {name} ---")
        for d in sorted_drops:
            i = d["cell"]
            r = audit[i]
            print(f"\n  Cell {i}  z={r['z']:.3f}m  dz={r['dz']:.4f}m")
            print(f"    T: {r['T_in']:.0f}K -> {r['T_out']:.0f}K  (dT={d['dT_K']:.0f}K)")
            print(f"    H_in={r['H_in']/1e6:.2f} MW  H_out={r['H_out']/1e6:.2f} MW  "
                  f"energy_src={r['energy_src']/1e6:.2f} MW  Q_loss={r['Q_loss']/1e6:.3f} MW")
            print(f"    放热: Q_ox={r['Q_ox']/1e6:.2f} MW  吸热: Q_gasif={r['Q_gasif']/1e6:.2f} MW  "
                  f"Q_wgs={r['Q_wgs']/1e6:.2f} MW")
            print(f"    能量残差 res_E={r['res_E']/1e6:.3f} MW")
    
    # === 4. 轴向温度与能量汇总 ===
    print("\n" + "=" * 80)
    print("四、轴向温度与能量累积（前 10 格 + 出口）")
    print("=" * 80)
    
    for name in target_names:
        if name not in all_results:
            continue
        rec = all_results[name]
        audit = rec["audit"]
        
        print(f"\n{name}:")
        print("  Cell   z(m)   T(K)   dT(K)   H_in(MW)  H_out(MW)  e_src(MW)  Q_ox(MW)  Q_gasif(MW)")
        for i, r in enumerate(audit[:10]):
            dT = r["T_out"] - r["T_in"]
            print(f"  {i:3d}  {r['z']:5.3f}  {r['T_out']:6.0f}  {dT:+6.0f}  "
                  f"{r['H_in']/1e6:8.2f}  {r['H_out']/1e6:8.2f}  {r['energy_src']/1e6:8.2f}  "
                  f"{r['Q_ox']/1e6:8.2f}  {r['Q_gasif']/1e6:8.2f}")
        r_last = audit[-1]
        print(f"  ... 出口 z={r_last['z']:.3f}m  T={r_last['T_out']:.0f}K")
    
    print("\n" + "=" * 80)
    print("诊断完成")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Texaco I-1 vs Exxon 工况差异与轴向能量诊断")
    parser.add_argument("--out", "-o", help="输出到文件")
    parser.add_argument("--cells", "-n", type=int, default=60, help="网格数 (默认 60)")
    args = parser.parse_args()
    if args.out:
        import sys
        sys.stdout = open(args.out, "w", encoding="utf-8")
    # 传递 N_cells 到 main (通过全局或修改 main 签名)
    main(n_cells=args.cells)
