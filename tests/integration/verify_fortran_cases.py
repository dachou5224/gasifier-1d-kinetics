"""
使用 reference_fortran/input_副本.txt 中的工况，
通过当前 Python 1D 气化模型计算出口 T 与干基组分。
"""

import numpy as np

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.append(os.path.join(ROOT, "src"))

from model.gasifier_system import GasifierSystem  # type: ignore
from model.state import StateVector  # type: ignore
from model.fortran_input_loader import load_fortran_cases, get_fortran_input_path  # type: ignore


def dry_mole_fraction_pct(gas_moles: np.ndarray) -> dict:
    """
    计算干基体积分数（不含 H2O）。
    Species 顺序：[O2, CH4, CO, CO2, H2S, H2, N2, H2O]
    """
    F_dry = float(np.sum(gas_moles[:7])) + 1e-12
    names = ["O2", "CH4", "CO", "CO2", "H2S", "H2", "N2"]
    Y = {}
    for i, nm in enumerate(names):
        Y[nm] = gas_moles[i] / F_dry * 100.0
    return Y


def main():
    path = get_fortran_input_path()
    coal_db, cases = load_fortran_cases(path)

    geometry = {"L": 8.0, "D": 2.6}
    n_cells = 50

    print("=" * 70)
    print("Python 1D 模型基于 Fortran input_副本.txt 的工况验证")
    print("=" * 70)

    for name, data in cases.items():
        inputs = data["inputs"]
        coal_key = inputs["coal"]
        coal_props = coal_db[coal_key]

        op_conds = {
            "coal_flow": inputs["FeedRate"] / 3600.0,
            "o2_flow": (inputs["FeedRate"] * inputs["Ratio_OC"]) / 3600.0,
            "steam_flow": (inputs["FeedRate"] * inputs["Ratio_SC"]) / 3600.0,
            "P": inputs["P"],
            "T_in": inputs["TIN"],
            "HeatLossPercent": inputs.get("HeatLossPercent", 2.0),
            "SlurryConcentration": inputs.get("SlurryConcentration", 100.0),
            "pilot_heat": 5.0e6,
        }

        try:
            system = GasifierSystem(geometry, coal_props, op_conds)
            results, z_grid = system.solve(N_cells=n_cells)
            last = results[-1]
            gas = last[:8]
            T_out_C = last[10] - 273.15
            y_dry = dry_mole_fraction_pct(gas)

            print(f"\nCase: {name}")
            print(f"  FeedRate = {inputs['FeedRate']:.1f} kg/h")
            print(f"  Ratio_OC = {inputs['Ratio_OC']:.3f}  (O2/coal mass)")
            print(f"  Ratio_SC = {inputs['Ratio_SC']:.3f}  (steam/coal mass)")
            print(f"  P        = {inputs['P']/1e5:.1f} bar")
            print(f"  T_out    = {T_out_C:.1f} °C")
            print(
                "  Y_dry(%) = "
                f"CO {y_dry['CO']:.1f}, H2 {y_dry['H2']:.1f}, "
                f"CO2 {y_dry['CO2']:.1f}, CH4 {y_dry['CH4']:.2f}, N2 {y_dry['N2']:.2f}"
            )
        except Exception as e:
            print(f"\nCase: {name} FAILED - {e}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

