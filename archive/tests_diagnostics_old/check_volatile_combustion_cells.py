"""
检查挥发分在几个 cell 内被完全燃烧消耗掉。

沿轴向输出：cell, z, pO2(atm), 燃烧区?, CH4/CO/H2 入口/出口, r_CH4_Ox, r_CO_Ox, r_H2_Ox
当 CH4 出口接近 0 且挥发分氧化速率为 0 时，可认为挥发分已消耗完毕。

运行: PYTHONPATH=src python tests/diagnostics/check_volatile_combustion_cells.py
"""
import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

from model.gasifier_system import GasifierSystem
from model.original_paper_loader import load_original_paper_cases, get_original_paper_json_path
from model.state import StateVector

THRESHOLD_PA = 5066.0  # 0.05 atm


def main():
    coal_db, cases, metadata = load_original_paper_cases(get_original_paper_json_path())
    geom = metadata.get("reactor_dimensions", {})
    L = float(geom.get("length_m", 6.096))
    D = float(geom.get("diameter_m", 1.524))
    geometry = {"L": L, "D": D}

    for name in ["Texaco_I-1", "Texaco_I-2"]:
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
            "pilot_heat": 5.0e6,
        }
        system = GasifierSystem(geometry, coal_props, op_conds)
        results, z_grid = system.solve(N_cells=60)

        n = len(results)
        z = np.array(system.z_positions)
        Cd_total = coal_props.get("Cd", 60.0) / 100.0
        C_fed = system.W_dry * Cd_total

        print(f"\n{'='*100}")
        print(f"  {name}: 挥发分燃烧沿轴向分布")
        print(f"{'='*100}")
        print(f"{'cell':<5} {'z(m)':<7} {'pO2(atm)':<10} {'zone':<12} "
              f"{'CH4_in':<8} {'CH4_out':<8} {'CO_in':<8} {'CO_out':<8} {'H2_in':<8} {'H2_out':<8} "
              f"{'O2_in':<8} {'O2_out':<8} "
              f"{'r_CH4':<7} {'r_CO':<7} {'r_H2':<7}")
        print("-" * 100)

        first_gasification = None
        ch4_depleted_cell = None

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
            gas_src = np.zeros(8)
            for s in cell.sources:
                g, _, _ = s.get_sources(cell.idx, cell.z, cell.dz)
                gas_src += g
            phys = cell._calc_physics_props(state_out, C_fed)
            r_het, r_homo, phi = cell._calc_rates(state_out, phys, gas_src)

            # 使用入口 pO2（与 Fortran poxyin 一致）判断燃烧区/气化区
            F_in = max(state_in.total_gas_moles, 1e-9)
            pO2_Pa = cell.inlet.P * (state_in.gas_moles[0] / F_in)
            pO2_atm = pO2_Pa / 101325.0
            is_comb = pO2_Pa > THRESHOLD_PA
            zone = "combustion" if is_comb else "gasification"

            if first_gasification is None and not is_comb:
                first_gasification = i
            if ch4_depleted_cell is None and state_out.gas_moles[1] < 0.01:
                ch4_depleted_cell = i

            r_CH4 = r_homo.get("CH4_Ox", 0.0)
            r_CO = r_homo.get("CO_Ox", 0.0)
            r_H2 = r_homo.get("H2_Ox", 0.0)

            print(f"{i:<5} {z[i]:<7.3f} {pO2_atm:<10.3f} {zone:<12} "
                  f"{state_in.gas_moles[1]:<8.2f} {state_out.gas_moles[1]:<8.2f} "
                  f"{state_in.gas_moles[2]:<8.2f} {state_out.gas_moles[2]:<8.2f} "
                  f"{state_in.gas_moles[5]:<8.2f} {state_out.gas_moles[5]:<8.2f} "
                  f"{state_in.gas_moles[0]:<8.2f} {state_out.gas_moles[0]:<8.2f} "
                  f"{r_CH4:<7.2f} {r_CO:<7.2f} {r_H2:<7.2f}")

        print("-" * 100)
        print(f"  [结论] 首入气化区 cell: {first_gasification}  (z={z[first_gasification]:.3f}m)" if first_gasification is not None else "  [结论] 全程燃烧区")
        print(f"  [结论] CH4 耗尽 cell: {ch4_depleted_cell}  (z={z[ch4_depleted_cell]:.3f}m)" if ch4_depleted_cell is not None else "  [结论] CH4 未耗尽")


if __name__ == "__main__":
    main()
