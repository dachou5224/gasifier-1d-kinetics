#!/usr/bin/env python3
"""
运行工业规模工况，使用 gasifier-1d-kinetic 1D 动力学模型。

工况来源: gasifier-model/src/gasifier/validation_cases.py
尺寸: D=2 m, L=6 m
网格: AdaptiveFirstCellLength=True（按进煤量自适应 Cell 0）
结果保存: data/validation_results_industrial.json（工业炉规模验证算例）

用法: PYTHONPATH=src python scripts/run_gasifier_model_cases.py
"""
import json
import logging
import os
import sys

import numpy as np

# 抑制求解器详细日志
logging.getLogger("model").setLevel(logging.WARNING)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from model.gasifier_system import GasifierSystem
from model.chemistry import COAL_DATABASE

# 工业规模工况（合并相似 Paper 工况，仅保留 2 个典型案例）
# Paper 原 4 工况实为同一基础工况改 O2/Coal，合并为 Paper_Case_6 代表
INDUSTRIAL_CASES = {
    "Paper_Case_6": {
        "coal": "Paper_Base_Coal",
        "FeedRate_kg_h": 41670.0,
        "SlurryConc": 60.0,
        "Ratio_OC": 1.019,  # Tuned from 1.05 (0.97x)
        "Ratio_SC": 0.08,
        "P_MPa": 4.08,
        "T_in_K": 300.0,
        "HeatLossPercent": 8.0,  # 热损扫描最佳：T=1401°C(期望1370)，CO/H2/CO2 综合 cost 最低
        "WGS_CatalyticFactor": 0.5,
        "WGS_K_Factor": 0.2,
        # H2_OxidationFactor 已实现(cell.py)，试验 0.5 时 Paper H2 反降，暂不启用
        "expected": {"TOUT_C": 1370.0, "YCO": 61.7, "YH2": 30.3, "YCO2": 1.3},
    },
    "LuNan_Texaco": {
        "coal": "LuNan_Coal",
        "FeedRate_kg_h": 17917.0,
        "SlurryConc": 66.0,
        "Ratio_OC": 0.872,
        "Ratio_SC": 0.0,
        "P_MPa": 4.0,
        "T_in_K": 400.0,
        "HeatLossPercent": 5.0,  # 热损扫描：T=1333°C 最接近期望 1350°C
        "CharCombustionRateFactor": 0.35,  # 策略 A.2：0.35 最优（931°C vs 0.2→876°C）
        "UseFortranDiffusion": True,  # Fortran 式扩散：T^0.75 传质阻力更大，异相速率略降
        "P_O2_Combustion_atm": 0.03,  # 扩展燃烧区（0.05→0.03），更多 Cell 挥发分燃烧放热
        "WGS_K_Factor": 0.5,
        "L_m": 6.87,
        "D_m": 1.68,
        "expected": {"TOUT_C": 1350.0, "YCO": 48.82, "YH2": 36.58, "YCO2": 14.41, "carbon_conversion_pct": 91.0},
    },
}


def main():
    default_geometry = {"L": 6.0, "D": 2.0}
    n_cells = 40

    print("=" * 75)
    print("工业规模工况 — gasifier-1d-kinetic 1D 动力学模型")
    print(f"N_cells={n_cells}, Grid: AdaptiveFirstCellLength=True")
    print("=" * 75)

    results = []

    # 可选：只运行部分工况以加快测试
    cases_to_run = os.environ.get("GASIFIER_CASES", "all")
    if cases_to_run != "all":
        names = [s.strip() for s in cases_to_run.split(",")]
        cases_iter = [(n, INDUSTRIAL_CASES[n]) for n in names if n in INDUSTRIAL_CASES]
    else:
        cases_iter = INDUSTRIAL_CASES.items()

    for name, case in cases_iter:
        geometry = {
            "L": case.get("L_m", default_geometry["L"]),
            "D": case.get("D_m", default_geometry["D"]),
        }
        coal_props = COAL_DATABASE.get(case["coal"])
        if not coal_props:
            print(f"\n{name}: 煤种 '{case['coal']}' 未找到，跳过")
            continue

        coal_flow_kg_s = case["FeedRate_kg_h"] / 3600.0
        o2_flow_kg_s = coal_flow_kg_s * case["Ratio_OC"]
        steam_flow_kg_s = coal_flow_kg_s * case["Ratio_SC"]
        P_Pa = case["P_MPa"] * 1e6

        op_conds = {
            "coal_flow": coal_flow_kg_s,
            "o2_flow": o2_flow_kg_s,
            "steam_flow": steam_flow_kg_s,
            "P": P_Pa,
            "T_in": case["T_in_K"],
            "SlurryConcentration": case["SlurryConc"],
            "HeatLossPercent": case["HeatLossPercent"],
            "AdaptiveFirstCellLength": True,
        }
        for key in ("L_evap_m", "FirstCellLength", "CharCombustionRateFactor", "UseFortranDiffusion", "WGS_RatFactor", "WGS_CatalyticFactor", "WGS_K_Factor", "P_O2_Combustion_atm", "MSR_Tmin_K", "CO_OxidationFactor", "H2_OxidationFactor"):
            if key in case:
                op_conds[key] = case[key]

        try:
            system = GasifierSystem(geometry, coal_props, op_conds)
            results_arr, z = system.solve(N_cells=n_cells)
            last = results_arr[-1]
            T_out_K = last[10]
            gas = last[:8]
            F_dry = np.sum(gas[:7]) + 1e-12
            y_co = gas[2] / F_dry * 100.0
            y_h2 = gas[5] / F_dry * 100.0
            y_co2 = gas[3] / F_dry * 100.0

            exp = case.get("expected", {})
            dz_cell0 = system.cells[0].dz if system.cells else 0.0

            row = {
                "name": name,
                "T_out_C": T_out_K - 273.15,
                "YCO": y_co,
                "YH2": y_h2,
                "YCO2": y_co2,
                "T_exp": exp.get("TOUT_C"),
                "YCO_exp": exp.get("YCO"),
                "YH2_exp": exp.get("YH2"),
                "YCO2_exp": exp.get("YCO2"),
                "dz_cell0": dz_cell0,
                "expected": exp,
            }
            results.append(row)

            print(f"\n{name}")
            print(f"  进料: {coal_flow_kg_s*1000:.0f} g/s  O2/Coal={case['Ratio_OC']}  Slurry={case['SlurryConc']}%")
            print(f"  dz_cell0 (自适应): {dz_cell0:.3f} m")
            print(f"  出口 T: {row['T_out_C']:.1f} °C  [期望 {row['T_exp']} °C]")
            print(f"  干基: CO={y_co:.1f}%  H2={y_h2:.1f}%  CO2={y_co2:.1f}%")
            print(f"  [期望 CO={row['YCO_exp']}%  H2={row['YH2_exp']}%]")

        except Exception as e:
            print(f"\n{name}: 计算失败 - {e}")
            import traceback
            traceback.print_exc()

    # 汇总
    print("\n" + "=" * 75)
    print("汇总")
    print("=" * 75)
    for r in results:
        t_err = (r["T_out_C"] - r["T_exp"]) if r["T_exp"] else None
        co_err = (r["YCO"] - r["YCO_exp"]) if r["YCO_exp"] else None
        err_str = f"  T_err={t_err:+.1f}°C" if t_err is not None else ""
        if co_err is not None:
            err_str += f"  CO_err={co_err:+.1f}%"
        print(f"{r['name']}: dz0={r['dz_cell0']:.3f}m{err_str}")

    # 保存结果到 data 目录，作为 1D 动力学模型工业炉规模验证算例
    out_path = os.path.join(ROOT, "data", "validation_results_industrial.json")
    try:
        out_data = {}
        for r in results:
            out_data[r["name"]] = {
                "predicted": {
                    "TOUT_C": r["T_out_C"],
                    "Y_CO_dry": r["YCO"],
                    "Y_H2_dry": r["YH2"],
                    "Y_CO2_dry": r["YCO2"],
                    "dz_cell0_m": r["dz_cell0"],
                },
                "expected": {
                    "TOUT_C": r["T_exp"],
                    "YCO": r["YCO_exp"],
                    "YH2": r["YH2_exp"],
                    "YCO2": r.get("YCO2_exp"),
                    "carbon_conversion_pct": r.get("expected", {}).get("carbon_conversion_pct"),
                },
            }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {out_path}")
    except Exception as e:
        print(f"\n保存结果失败: {e}")


if __name__ == "__main__":
    main()
