"""
Run verification using validation_cases_new.json (Illinois_No6, Australia_UBE, Fluid_Coke).
Merge loaded coal DB with chemistry.COAL_DATABASE and run same flow as verify_cases.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import numpy as np
from model.gasifier_system import GasifierSystem
from model.state import StateVector
from model.chemistry import COAL_DATABASE
from model.validation_loader import load_validation_cases_from_json, get_validation_cases_new_path


def dry_mole_fraction_pct(gas_moles):
    F_dry = sum(gas_moles[:7]) + 1e-12
    return {
        "O2": gas_moles[0] / F_dry * 100,
        "CH4": gas_moles[1] / F_dry * 100,
        "CO": gas_moles[2] / F_dry * 100,
        "CO2": gas_moles[3] / F_dry * 100,
        "H2S": gas_moles[4] / F_dry * 100,
        "H2": gas_moles[5] / F_dry * 100,
        "N2": gas_moles[6] / F_dry * 100,
    }


def wet_to_dry_expected(exp):
    """
    文献 target_output 通常为湿基（含 H2O），模型输出为干基。换算：Y_dry = Y_wet / (1 - Y_H2O_wet/100)。
    """
    y_h2o = exp.get("YH2O", 0.0) / 100.0
    if y_h2o >= 1.0 or y_h2o <= 0.0:
        return {k: exp.get(k, 0.0) for k in ("YCO", "YH2", "YCO2", "YCH4")}
    scale = 1.0 / (1.0 - y_h2o)
    return {
        "YCO": exp.get("YCO", 0) * scale,
        "YH2": exp.get("YH2", 0) * scale,
        "YCO2": exp.get("YCO2", 0) * scale,
        "YCH4": exp.get("YCH4", 0) * scale,
    }


def main():
    path = get_validation_cases_new_path()
    if not os.path.isfile(path):
        print("Not found:", path)
        return
    coal_db_new, cases_new = load_validation_cases_from_json(path)
    coal_db = {**COAL_DATABASE, **coal_db_new}

    geometry = {"L": 8.0, "D": 2.6}
    n_cells = 50
    print("=" * 70)
    print("Validation (new JSON): Illinois_No6, Australia_UBE, Fluid_Coke")
    print("  sim = 模型干基 | exp = 文献干基（由 target_output 湿基换算）")
    print("=" * 70)
    for name, data in cases_new.items():
        inp = data["inputs"]
        exp_wet = data["expected"]
        exp = wet_to_dry_expected(exp_wet)
        coal_props = coal_db[inp["coal"]]
        op_conds = {
            "coal_flow": inp["FeedRate"] / 3600.0,
            "o2_flow": (inp["FeedRate"] * inp["Ratio_OC"]) / 3600.0,
            "steam_flow": (inp["FeedRate"] * inp["Ratio_SC"]) / 3600.0,
            "P": inp["P"],
            "T_in": inp["TIN"],
            "HeatLossPercent": inp.get("HeatLossPercent", 2.0),
            "SlurryConcentration": inp.get("SlurryConcentration", 100.0),
            "pilot_heat": 5.0e6,
        }
        try:
            system = GasifierSystem(geometry, coal_props, op_conds)
            results, z_grid = system.solve(N_cells=n_cells)
            last = results[-1]
            y = dry_mole_fraction_pct(last[:8])
            T_out_C = last[10] - 273.15
            print(f"\n{name}:")
            print(f"  T_out = {T_out_C:.0f} °C")
            print(f"  CO:  sim {y['CO']:.1f}%  exp(干基) {exp['YCO']:.1f}%")
            print(f"  H2:  sim {y['H2']:.1f}%  exp(干基) {exp['YH2']:.1f}%")
            print(f"  CO2: sim {y['CO2']:.1f}%  exp(干基) {exp['YCO2']:.1f}%")
            print(f"  CH4: sim {y['CH4']:.2f}%  exp(干基) {exp['YCH4']:.2f}%")
        except Exception as e:
            print(f"\n{name}: FAILED - {e}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
