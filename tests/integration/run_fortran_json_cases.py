"""
从 data/validation_cases_fortran.json 读取验证工况，用当前 1D 模型计算并输出评估报告。

几何尺寸与操作条件按 validation_cases_OriginalPaper.json 的 Texaco pilot 标准：
  L = 6.096 m, D = 1.524 m（20 ft × 5 ft）
"""
import json
import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

from model.gasifier_system import GasifierSystem
from model.state import StateVector


def _get_original_paper_geometry():
    """从 validation_cases_OriginalPaper.json 读取气化炉尺寸"""
    path = os.path.join(ROOT, "data", "validation_cases_OriginalPaper.json")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f).get("metadata", {})
        geom = meta.get("reactor_dimensions", {})
        return {
            "L": float(geom.get("length_m", 6.096)),
            "D": float(geom.get("diameter_m", 1.524)),
        }
    return {"L": 6.096, "D": 1.524}


def load_cases_from_json(path: str):
    """从 validation_cases_fortran.json 加载 -> (coal_db, cases)。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("fortran_cases", data)
    coal_db = {}
    cases = {}
    for name, rec in raw.items():
        coal_key = rec["inputs"]["coal"]
        coal_db[coal_key] = rec["coal"]
        cases[name] = {"inputs": rec["inputs"], "expected": rec.get("expected", {})}
    return coal_db, cases


def dry_mole_fraction_pct(gas_moles: np.ndarray) -> dict:
    F_dry = float(np.sum(gas_moles[:7])) + 1e-12
    names = ["O2", "CH4", "CO", "CO2", "H2S", "H2", "N2"]
    return {nm: gas_moles[i] / F_dry * 100.0 for i, nm in enumerate(names)}


def main():
    json_path = os.path.join(ROOT, "data", "validation_cases_fortran.json")
    if not os.path.isfile(json_path):
        print("Not found:", json_path)
        return

    coal_db, cases = load_cases_from_json(json_path)
    geometry = _get_original_paper_geometry()
    n_cells = 60

    results_list = []

    print("=" * 72)
    print("  Fortran 工况 (validation_cases_fortran.json) — 当前模型计算结果")
    print(f"  几何尺寸: L = {geometry['L']:.3f} m, D = {geometry['D']:.3f} m (来自 OriginalPaper.json)")
    print("=" * 72)

    for name, data in cases.items():
        inp = data["inputs"]
        coal_key = inp["coal"]
        coal_props = coal_db[coal_key]

        op_conds = {
            "coal_flow": inp["FeedRate"] / 3600.0,
            "o2_flow": (inp["FeedRate"] * inp["Ratio_OC"]) / 3600.0,
            "steam_flow": (inp["FeedRate"] * inp["Ratio_SC"]) / 3600.0,
            "P": inp["P"],
            "T_in": inp["TIN"],
            "HeatLossPercent": inp.get("HeatLossPercent", 2.0),
            "SlurryConcentration": inp.get("SlurryConcentration", 60.0),
            "pilot_heat": 5.0e6,
        }

        try:
            system = GasifierSystem(geometry, coal_props, op_conds)
            results, z_grid = system.solve(N_cells=n_cells)
            last = results[-1]
            gas = last[:8]
            T_out_C = last[10] - 273.15
            y_dry = dry_mole_fraction_pct(gas)

            row = {
                "name": name,
                "FeedRate": inp["FeedRate"],
                "Ratio_OC": inp["Ratio_OC"],
                "Ratio_SC": inp["Ratio_SC"],
                "Slurry_pct": inp.get("SlurryConcentration", 60),
                "T_out_C": T_out_C,
                "CO": y_dry["CO"],
                "H2": y_dry["H2"],
                "CO2": y_dry["CO2"],
                "CH4": y_dry["CH4"],
                "N2": y_dry["N2"],
                "O2": y_dry["O2"],
                "ok": True,
            }
            results_list.append(row)

            print(f"\n  {name}")
            print(f"    FeedRate = {inp['FeedRate']:.0f} kg/h   Ratio_OC = {inp['Ratio_OC']:.3f}   Ratio_SC = {inp['Ratio_SC']:.3f}   Slurry = {row['Slurry_pct']:.0f}%")
            print(f"    T_out = {T_out_C:.1f} °C")
            print(f"    Y_dry(%)  CO {y_dry['CO']:.1f}   H2 {y_dry['H2']:.1f}   CO2 {y_dry['CO2']:.1f}   CH4 {y_dry['CH4']:.2f}   N2 {y_dry['N2']:.2f}   O2 {y_dry['O2']:.2f}")
        except Exception as e:
            results_list.append({"name": name, "ok": False, "error": str(e)})
            print(f"\n  {name}  FAILED: {e}")

    # ----- 评估 -----
    print("\n" + "=" * 72)
    print("  评估摘要")
    print("=" * 72)

    ok_rows = [r for r in results_list if r.get("ok")]
    if not ok_rows:
        print("  无成功工况。")
        return

    T_arr = [r["T_out_C"] for r in ok_rows]
    CO_arr = [r["CO"] for r in ok_rows]
    H2_arr = [r["H2"] for r in ok_rows]
    CO2_arr = [r["CO2"] for r in ok_rows]

    print(f"  成功工况数: {len(ok_rows)} / {len(results_list)}")
    print(f"  出口温度:   min = {min(T_arr):.0f} °C,  max = {max(T_arr):.0f} °C,  mean = {np.mean(T_arr):.0f} °C")
    print(f"  CO (干基):  min = {min(CO_arr):.1f} %,   max = {max(CO_arr):.1f} %,   mean = {np.mean(CO_arr):.1f} %")
    print(f"  H2 (干基):  min = {min(H2_arr):.1f} %,   max = {max(H2_arr):.1f} %,   mean = {np.mean(H2_arr):.1f} %")
    print(f"  CO2(干基):  min = {min(CO2_arr):.1f} %,   max = {max(CO2_arr):.1f} %,   mean = {np.mean(CO2_arr):.1f} %")

    print("\n  合理性检查 (典型 Texaco 气化):")
    print("    - 出口温度一般 900–1400 °C: ", end="")
    t_ok = sum(1 for t in T_arr if 900 <= t <= 1400)
    print(f"{t_ok}/{len(T_arr)} 在范围内" if t_ok == len(T_arr) else f"有 {len(T_arr)-t_ok} 个偏离")
    print("    - 干基 CO 通常 45–65 %:     ", end="")
    co_ok = sum(1 for c in CO_arr if 45 <= c <= 65)
    print(f"{co_ok}/{len(CO_arr)} 在范围内")
    print("    - 干基 H2 通常 30–42 %:     ", end="")
    h2_ok = sum(1 for h in H2_arr if 30 <= h <= 42)
    print(f"{h2_ok}/{len(H2_arr)} 在范围内")
    print("    - 干基 CO2 通常 2–15 %:     ", end="")
    co2_ok = sum(1 for c in CO2_arr if 2 <= c <= 15)
    print(f"{co2_ok}/{len(CO2_arr)} 在范围内")
    print("    - 出口 O2 应接近 0:          ", end="")
    o2_ok = sum(1 for r in ok_rows if r["O2"] < 1.0)
    print(f"{o2_ok}/{len(ok_rows)} 个 < 1%")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
