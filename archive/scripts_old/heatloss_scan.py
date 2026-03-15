#!/usr/bin/env python3
"""
热损 (HeatLossPercent) 扫描：匹配温度与气体组成。

用法: PYTHONPATH=src python scripts/heatloss_scan.py
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

# 热损扫描范围：Paper 需降 ~327°C，LuNan 需降 ~162°C
HEATLOSS_PAPER = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0]
HEATLOSS_LUNAN = [2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]

CASES = [
    ("Paper_Case_6", {
        "coal": "Paper_Base_Coal",
        "FeedRate_kg_h": 41670.0, "SlurryConc": 60.0, "Ratio_OC": 1.05, "Ratio_SC": 0.08,
        "P_MPa": 4.08, "T_in_K": 300.0,
        "WGS_CatalyticFactor": 0.5, "WGS_K_Factor": 0.2,
        "expected": {"TOUT_C": 1370, "YCO": 61.7, "YH2": 30.3, "YCO2": 1.3},
    }),
    ("LuNan_Texaco", {
        "coal": "LuNan_Coal",
        "FeedRate_kg_h": 17917.0, "SlurryConc": 66.0, "Ratio_OC": 0.872, "Ratio_SC": 0.0,
        "P_MPa": 4.0, "T_in_K": 400.0,
        "CharCombustionRateFactor": 0.35, "UseFortranDiffusion": True,
        "P_O2_Combustion_atm": 0.03, "WGS_K_Factor": 0.5,
        "L_m": 6.87, "D_m": 1.68,
        "expected": {"TOUT_C": 1350, "YCO": 48.82, "YH2": 36.58, "YCO2": 14.41},
    }),
]


def run_one(case_name, case, heatloss, n_cells=40):
    coal_props = COAL_DATABASE[case["coal"]]
    coal_flow = case["FeedRate_kg_h"] / 3600.0
    geometry = {"L": case.get("L_m", 6.0), "D": case.get("D_m", 2.0)}
    op_conds = {
        "coal_flow": coal_flow,
        "o2_flow": coal_flow * case["Ratio_OC"],
        "steam_flow": coal_flow * case["Ratio_SC"],
        "P": case["P_MPa"] * 1e6,
        "T_in": case["T_in_K"],
        "HeatLossPercent": heatloss,
        "SlurryConcentration": case["SlurryConc"],
        "AdaptiveFirstCellLength": True,
    }
    for k in ("CharCombustionRateFactor", "UseFortranDiffusion", "P_O2_Combustion_atm",
              "WGS_CatalyticFactor", "WGS_K_Factor"):
        if k in case:
            op_conds[k] = case[k]
    system = GasifierSystem(geometry, coal_props, op_conds)
    results, _ = system.solve(N_cells=n_cells)
    last = results[-1]
    T = last[10] - 273.15
    F = last[:8]
    F_dry = sum(F[:7]) + 1e-12
    return {
        "T_C": T,
        "CO": F[2] / F_dry * 100,
        "H2": F[5] / F_dry * 100,
        "CO2": F[3] / F_dry * 100,
    }


def cost(r, exp, w_T=1.0, w_CO=0.5, w_H2=0.5, w_CO2=0.3):
    """综合误差：温度优先，组分次之。CO2 期望 1.3% 异常低，权重略降。"""
    c = 0.0
    if exp.get("TOUT_C") is not None:
        c += w_T * abs(r["T_C"] - exp["TOUT_C"]) / 100.0  # 归一化 ~1 per 100°C
    if exp.get("YCO") is not None:
        c += w_CO * abs(r["CO"] - exp["YCO"])
    if exp.get("YH2") is not None:
        c += w_H2 * abs(r["H2"] - exp["YH2"])
    if exp.get("YCO2") is not None and exp["YCO2"] is not None:
        # Paper CO2=1.3% 异常低，用 min(期望, 10) 避免过度惩罚
        target = min(exp["YCO2"], 10.0) if exp["YCO2"] < 5 else exp["YCO2"]
        c += w_CO2 * abs(r["CO2"] - target)
    return c


def main():
    print("热损 (HeatLossPercent) 扫描 — 匹配温度与气体组成")
    print("=" * 75)
    heatloss_ranges = {"Paper_Case_6": HEATLOSS_PAPER, "LuNan_Texaco": HEATLOSS_LUNAN}
    best = {}
    all_results = {}

    for case_name, case in CASES:
        exp = case.get("expected", {})
        hl_range = heatloss_ranges.get(case_name, [3.0, 4.0, 5.0, 6.0, 8.0])
        print(f"\n{case_name}  期望 T={exp.get('TOUT_C')}°C  CO={exp.get('YCO')}%  H2={exp.get('YH2')}%  CO2={exp.get('YCO2')}%")
        print("-" * 75)
        print(f"{'HeatLoss%':>10} {'T(°C)':>8} {'CO%':>7} {'H2%':>7} {'CO2%':>7} {'cost':>8}")
        all_results[case_name] = {}
        best_cost = 1e9
        best_hl = None
        for hl in hl_range:
            r = run_one(case_name, case, hl)
            c = cost(r, exp)
            all_results[case_name][hl] = {**r, "cost": c}
            mark = " *" if c < best_cost else ""
            if c < best_cost:
                best_cost = c
                best_hl = hl
            print(f"{hl:>10.1f} {r['T_C']:>8.1f} {r['CO']:>7.1f} {r['H2']:>7.1f} {r['CO2']:>7.1f} {c:>8.3f}{mark}")
        best[case_name] = {"HeatLossPercent": best_hl, "result": all_results[case_name][best_hl], "cost": best_cost}
        print(f"  -> 最佳 HeatLoss={best_hl}%  cost={best_cost:.3f}")

    # 输出建议
    print("\n" + "=" * 75)
    print("建议 HeatLossPercent")
    print("=" * 75)
    for name, b in best.items():
        r = b["result"]
        exp = next(c["expected"] for n, c in CASES if n == name)
        print(f"{name}: HeatLossPercent={b['HeatLossPercent']}  ->  T={r['T_C']:.1f}°C  CO={r['CO']:.1f}%  H2={r['H2']:.1f}%  CO2={r['CO2']:.1f}%")
        print(f"      期望 T={exp.get('TOUT_C')}  CO={exp.get('YCO')}  H2={exp.get('YH2')}  CO2={exp.get('YCO2')}")

    out_path = os.path.join(ROOT, "data", "heatloss_scan_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"best": best, "all": all_results}, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
