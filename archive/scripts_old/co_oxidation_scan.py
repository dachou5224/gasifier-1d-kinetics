#!/usr/bin/env python3
"""
CO 氧化因子扫描：燃烧区 CO+0.5O2→CO2 缩放
CO_OxidationFactor=1.0 为基准（瞬时完全燃烧），<1 减弱 CO 氧化
记录 Paper_Case_6、LuNan 的 T、CO、H2、CO2
"""
import json
import logging
import os
import sys

logging.getLogger("model").setLevel(logging.WARNING)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from model.gasifier_system import GasifierSystem
from model.chemistry import COAL_DATABASE

CO_OX_FACTORS = [1.0, 0.8, 0.5, 0.3, 0.2]  # 1.0=基准，减小=减弱 CO 氧化
CASES = [
    ("Paper_Case_6", {
        "coal": "Paper_Base_Coal",
        "FeedRate_kg_h": 41670.0, "SlurryConc": 60.0, "Ratio_OC": 1.05, "Ratio_SC": 0.08,
        "P_MPa": 4.08, "T_in_K": 300.0, "HeatLossPercent": 3.0,
        "WGS_CatalyticFactor": 0.5,
        "expected": {"TOUT_C": 1370, "YCO": 61.7, "YH2": 30.3, "YCO2": 1.3},
    }),
    ("LuNan_Texaco", {
        "coal": "LuNan_Coal",
        "FeedRate_kg_h": 17917.0, "SlurryConc": 66.0, "Ratio_OC": 0.872, "Ratio_SC": 0.0,
        "P_MPa": 4.0, "T_in_K": 400.0, "HeatLossPercent": 1.5,
        "CharCombustionRateFactor": 0.35, "UseFortranDiffusion": True,
        "P_O2_Combustion_atm": 0.03, "L_m": 6.87, "D_m": 1.68,
        "expected": {"TOUT_C": 1350, "YCO": 48.82, "YH2": 36.58, "YCO2": 14.41},
    }),
]


def run_one(case_name, case, co_ox_factor, n_cells=40):
    coal_props = COAL_DATABASE[case["coal"]]
    coal_flow = case["FeedRate_kg_h"] / 3600.0
    geometry = {"L": case.get("L_m", 6.0), "D": case.get("D_m", 2.0)}
    op_conds = {
        "coal_flow": coal_flow,
        "o2_flow": coal_flow * case["Ratio_OC"],
        "steam_flow": coal_flow * case["Ratio_SC"],
        "P": case["P_MPa"] * 1e6,
        "T_in": case["T_in_K"],
        "HeatLossPercent": case["HeatLossPercent"],
        "SlurryConcentration": case["SlurryConc"],
        "AdaptiveFirstCellLength": True,
        "CO_OxidationFactor": co_ox_factor,
    }
    for k in ("CharCombustionRateFactor", "UseFortranDiffusion", "P_O2_Combustion_atm", "WGS_CatalyticFactor"):
        if k in case:
            op_conds[k] = case[k]
    system = GasifierSystem(geometry, coal_props, op_conds)
    results, _ = system.solve(N_cells=n_cells)
    last = results[-1]
    T = last[10] - 273.15
    F = last[:8]
    F_dry = sum(F[:7]) + 1e-12
    return {
        "T_C": T, "CO": F[2] / F_dry * 100, "H2": F[5] / F_dry * 100, "CO2": F[3] / F_dry * 100
    }


def main():
    print("CO 氧化因子 (CO_OxidationFactor) 扫描")
    print("1.0=基准(瞬时完全燃烧), <1=减弱 CO+0.5O2→CO2")
    print("=" * 70)
    all_results = {}
    for case_name, case in CASES:
        exp = case.get("expected", {})
        print(f"\n{case_name} (期望 T={exp.get('TOUT_C')}°C CO={exp.get('YCO')}% CO2={exp.get('YCO2')}%)")
        print("-" * 70)
        all_results[case_name] = {}
        for fac in CO_OX_FACTORS:
            r = run_one(case_name, case, fac)
            all_results[case_name][f"CO_Ox={fac}"] = r
            dT = r["T_C"] - exp.get("TOUT_C", 0)
            dCO = r["CO"] - exp.get("YCO", 0)
            dCO2 = r["CO2"] - exp.get("YCO2", 0) if exp.get("YCO2") else 0
            print(f"  CO_Ox={fac}: T={r['T_C']:.1f}°C (ΔT={dT:+.1f})  CO={r['CO']:.1f}% (ΔCO={dCO:+.1f})  CO2={r['CO2']:.1f}% (ΔCO2={dCO2:+.1f})  H2={r['H2']:.1f}%")
    print("\n" + "=" * 70)
    out = os.path.join(ROOT, "docs", "co_oxidation_scan_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"结果已保存: {out}")


if __name__ == "__main__":
    main()
