#!/usr/bin/env python3
"""
分步回滚 P0-1、P0-2，验证各自对出口温度的影响。
用于定位温度偏高原因。
"""
import json
import logging
import os
import sys

import numpy as np

logging.getLogger("model").setLevel(logging.WARNING)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from model.gasifier_system import GasifierSystem
from model.chemistry import COAL_DATABASE

# Paper_Case_6 (Base) 工况
CASE = {
    "coal": "Paper_Base_Coal",
    "FeedRate_kg_h": 41670.0,
    "SlurryConc": 60.0,
    "Ratio_OC": 1.05,
    "Ratio_SC": 0.08,
    "P_MPa": 4.08,
    "T_in_K": 300.0,
    "HeatLossPercent": 3.0,
}
GEOMETRY = {"L": 6.0, "D": 2.0}
N_CELLS = 40


def run_single(op_conds, geometry, coal_props):
    """运行单次，返回出口 T (°C) 和 CO%"""
    system = GasifierSystem(geometry, coal_props, op_conds)
    results, z_grid = system.solve(N_cells=N_CELLS)
    last = results[-1]
    T_out_K = last[10]  # [F0-F7, Ws, Xc, T]
    F = last[:8]
    F_dry = float(sum(F[:7])) + 1e-12
    Y_CO = F[2] / F_dry * 100.0
    return T_out_K - 273.15, Y_CO


def main():
    coal_props = COAL_DATABASE[CASE["coal"]]
    coal_flow = CASE["FeedRate_kg_h"] / 3600.0
    op_conds = {
        "coal_flow": coal_flow,
        "o2_flow": coal_flow * CASE["Ratio_OC"],
        "steam_flow": coal_flow * CASE["Ratio_SC"],
        "P": CASE["P_MPa"] * 1e6,
        "T_in": CASE["T_in_K"],
        "HeatLossPercent": CASE["HeatLossPercent"],
        "SlurryConcentration": CASE["SlurryConc"],
        "AdaptiveFirstCellLength": True,
    }

    results = {}

    # 1. 当前（全部修改）
    print("1. 当前（P0-1/P0-2 均已修改）...")
    T, CO = run_single(op_conds, GEOMETRY, coal_props)
    results["current"] = {"T_C": T, "CO_pct": CO}
    print(f"   T_out = {T:.1f} °C, CO = {CO:.1f}%")

    # 2. 回滚 P0-1（恢复 Step 9）
    import model.pyrolysis_service as ps_mod
    orig_calc = ps_mod.PyrolysisService.calc_yields

    def calc_yields_with_step9(self, coal_props):
        Cd = coal_props.get('Cd', 0.0)
        Hd = coal_props.get('Hd', 0.0)
        Od = coal_props.get('Od', 0.0)
        Nd = coal_props.get('Nd', 0.0)
        Sd = coal_props.get('Sd', 0.0)
        FCd = coal_props.get('FCd', 50.0)
        C_vol_potential = max(Cd - FCd, 0.0)
        n_C_potential = C_vol_potential / 12.011
        n_N2 = Nd / 28.013
        n_H2S = Sd / 32.06
        H_for_S = n_H2S * 2.0 * 1.008
        Hd_avail = max(Hd - H_for_S, 0.0)
        n_H_avail = Hd_avail / 1.008
        n_O_avail = Od / 15.999
        n_CO = min(n_C_potential, n_O_avail)
        n_C_rem = max(n_C_potential - n_CO, 0.0)
        n_CH4 = min(n_C_rem, n_H_avail / 4.0)
        n_CO2 = 0.0
        n_O_excess = max(n_O_avail - n_CO, 0.0)
        n_O_used = n_CO + 2.0 * n_CO2
        n_O_resid = max(n_O_avail - n_O_used, 0.0)
        n_H2O = n_O_resid
        n_H_used_from_avail = 4.0 * n_CH4 + 2.0 * n_H2O
        n_H_resid = max(n_H_avail - n_H_used_from_avail, 0.0)
        n_H2 = n_H_resid / 2.0
        # Step 9 回滚：HHV 截断
        hhv_coal = coal_props.get('HHV_d', 30.0) * 1e3  # kJ/kg -> J/kg
        LHV_C = 110.5e3  # J/mol C (C->CO)
        d_lhv = [0, 802e3, 283e3, 0, 0, 242e3, 0, 0]  # CH4,CO,H2,H2S LHV J/mol
        lhv_basis = (Cd / 100.0) * (LHV_C / 0.012011)
        target_lhv = hhv_coal * 0.95
        target_excess = target_lhv - lhv_basis
        current_excess = n_CH4 * d_lhv[1] + n_CO * d_lhv[2] + n_H2 * d_lhv[5] + n_H2S * d_lhv[4]
        if current_excess > target_excess > 0:
            scale = target_excess / current_excess
            n_CH4 *= scale
            n_CO *= scale
            n_H2 *= scale
            n_H2S *= scale
        factor = 10.0
        molar_yields = np.zeros(8)
        molar_yields[1] = n_CH4 * factor
        molar_yields[2] = n_CO * factor
        molar_yields[3] = n_CO2 * factor
        molar_yields[4] = n_H2S * factor
        molar_yields[5] = n_H2 * factor
        molar_yields[6] = n_N2 * factor
        molar_yields[7] = n_H2O * factor
        MW = [31.998, 16.04, 28.01, 44.01, 34.08, 2.016, 28.013, 18.015]
        W_vol_calc = sum(molar_yields[i] * MW[i] for i in range(8)) / 1000.0
        return molar_yields, W_vol_calc

    ps_mod.PyrolysisService.calc_yields = calc_yields_with_step9
    print("2. 回滚 P0-1（恢复 Step 9 挥发分截断）...")
    T, CO = run_single(op_conds, GEOMETRY, coal_props)
    results["revert_P0-1"] = {"T_C": T, "CO_pct": CO}
    print(f"   T_out = {T:.1f} °C, CO = {CO:.1f}%")
    ps_mod.PyrolysisService.calc_yields = orig_calc

    # 3. 回滚 P0-2（恢复 Hf_char 经验公式）
    import model.material as mat_mod
    orig_get_solid = mat_mod.MaterialService.get_solid_enthalpy

    @staticmethod
    def get_solid_enthalpy_old(state, coal_props, T_solid_override=None):
        T_s = T_solid_override if T_solid_override is not None else state.T
        cp_s = coal_props.get('cp_char', 1300.0)
        hf_coal = coal_props.get('Hf_coal', -3e6)
        VM = 100.0 - coal_props.get('FCd', 50.0) - coal_props.get('Ashd', 6.0)
        VM = max(VM, 0.0)
        hf_char = (1.0 - VM / 100.0 * 0.7) * hf_coal
        is_char = state.carbon_fraction > 0.80
        hf = hf_char if is_char else hf_coal
        h_sensible = cp_s * (T_s - 298.15)
        h_s = h_sensible + hf
        return state.solid_mass * h_s

    mat_mod.MaterialService.get_solid_enthalpy = get_solid_enthalpy_old
    print("3. 回滚 P0-2（恢复 Hf_char 经验公式）...")
    T, CO = run_single(op_conds, GEOMETRY, coal_props)
    results["revert_P0-2"] = {"T_C": T, "CO_pct": CO}
    print(f"   T_out = {T:.1f} °C, CO = {CO:.1f}%")
    mat_mod.MaterialService.get_solid_enthalpy = orig_get_solid

    # 4. 同时回滚 P0-1 和 P0-2
    ps_mod.PyrolysisService.calc_yields = calc_yields_with_step9
    mat_mod.MaterialService.get_solid_enthalpy = get_solid_enthalpy_old
    print("4. 同时回滚 P0-1 + P0-2...")
    T, CO = run_single(op_conds, GEOMETRY, coal_props)
    results["revert_both"] = {"T_C": T, "CO_pct": CO}
    print(f"   T_out = {T:.1f} °C, CO = {CO:.1f}%")
    ps_mod.PyrolysisService.calc_yields = orig_calc
    mat_mod.MaterialService.get_solid_enthalpy = orig_get_solid

    # 汇总
    print("\n" + "=" * 60)
    print("分步回滚分析汇总 (Paper_Case_6 Base)")
    print("=" * 60)
    print(f"{'配置':<25} {'T_out (°C)':>12} {'CO (%)':>10} {'ΔT vs 当前':>12}")
    print("-" * 60)
    T_curr = results["current"]["T_C"]
    for name, r in results.items():
        dT = r["T_C"] - T_curr
        print(f"{name:<25} {r['T_C']:>12.1f} {r['CO_pct']:>10.1f} {dT:>+12.1f}")
    print("=" * 60)
    print("\n结论：")
    d1 = results["revert_P0-1"]["T_C"] - T_curr
    d2 = results["revert_P0-2"]["T_C"] - T_curr
    print(f"  - 回滚 P0-1（Step 9）: ΔT = {d1:+.1f} °C")
    print(f"  - 回滚 P0-2（Hf_char）: ΔT = {d2:+.1f} °C")
    if abs(d1) > abs(d2):
        print("  -> 温度偏高主要来自 P0-1（删除 Step 9 导致挥发分恢复、燃烧放热增加）")
    else:
        print("  -> 回滚 P0-2 使 T 升高，说明当前 P0-2（固相焓简化）实际降低了温度")
        print("  -> 回滚 P0-1 无变化，Step 9 可能未触发或挥发分已接近目标")

    out_path = os.path.join(ROOT, "docs", "rollback_p0_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
