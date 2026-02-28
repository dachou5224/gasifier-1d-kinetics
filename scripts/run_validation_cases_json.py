#!/usr/bin/env python3
"""
运行 data/validation_cases.json 中的算例。
用法: PYTHONPATH=src python scripts/run_validation_cases_json.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from model.gasifier_system import GasifierSystem
from model.chemistry import COAL_DATABASE


def main():
    json_path = os.path.join(ROOT, "data", "validation_cases.json")
    with open(json_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    cases = config.get("cases", {})
    print("=" * 70)
    print("validation_cases.json 算例计算")
    print("=" * 70)

    for name, data in cases.items():
        inputs = data["inputs"]
        coal_key = data["coal_type"]
        coal_props = COAL_DATABASE.get(coal_key)
        if not coal_props:
            print(f"\n{name}: 煤种 '{coal_key}' 未找到，跳过")
            continue

        coal_flow_kg_s = inputs["FeedRate_kg_h"] / 3600.0
        o2_flow_kg_s = coal_flow_kg_s * inputs["Ratio_OC"]
        steam_flow_kg_s = coal_flow_kg_s * inputs.get("SteamRatio_SC", 0.0)

        op_conds = {
            "coal_flow": coal_flow_kg_s,
            "o2_flow": o2_flow_kg_s,
            "steam_flow": steam_flow_kg_s,
            "P": inputs["P_Pa"],
            "T_in": inputs["T_in_K"],
            "SlurryConcentration": inputs.get("SlurryConcentration", 100.0),
        }
        geometry = {
            "L": inputs.get("L_reactor", 6.0),
            "D": inputs.get("D_reactor", 2.0),
        }

        try:
            system = GasifierSystem(geometry, coal_props, op_conds)
            results, z = system.solve(N_cells=40)
            last = results[-1]
            T_out_K = last[10]
            gas = last[:8]
            F_dry = np.sum(gas[:7]) + 1e-12
            y_co = gas[2] / F_dry * 100.0
            y_h2 = gas[5] / F_dry * 100.0
            y_co2 = gas[3] / F_dry * 100.0

            exp = data.get("expected", {})
            print(f"\n{name}")
            print(f"  进料: {coal_flow_kg_s*1000:.1f} g/s  O2/Coal={inputs['Ratio_OC']}  Slurry={inputs.get('SlurryConcentration', 100)}%")
            print(f"  出口 T: {T_out_K:.1f} K ({T_out_K-273.15:.1f} °C)  [期望 ~{exp.get('TOUT_K_approx', '-')} K]")
            print(f"  干基组成: CO={y_co:.1f}%  H2={y_h2:.1f}%  CO2={y_co2:.1f}%")
            print(f"  [期望 CO ~{exp.get('YCO_pct_approx', '-')}%]")
        except Exception as e:
            print(f"\n{name}: 计算失败 - {e}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
