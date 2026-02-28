#!/usr/bin/env python3
"""
运行 data/validation_cases_merged.json 中的全部验证算例，输出结果到 data/validation_results_merged.json。

算例包含：Texaco 系列、煤浆、Paper_Case_6、Illinois_No6、Australia_UBE、Fluid_Coke。
几何尺寸：Texaco/煤浆用 pilot (L=6.096m, D=1.524m)；Paper_Case_6 用 L=6m, D=2m；其余默认 pilot。

用法: PYTHONPATH=src python scripts/run_merged_cases.py
      PYTHONPATH=src python scripts/run_merged_cases.py --limit 5  # 快速测试
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
from model.original_paper_loader import load_original_paper_cases


def _dry_pct(gas_moles):
    F_dry = float(np.sum(gas_moles[:7])) + 1e-12
    return {sp: float(gas_moles[i]) / F_dry * 100.0 for i, sp in enumerate(["O2", "CH4", "CO", "CO2", "H2S", "H2", "N2"])}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cells", type=int, default=40)
    args = ap.parse_args()

    path = os.path.join(ROOT, "data", "validation_cases_merged.json")
    if not os.path.isfile(path):
        print("Not found:", path)
        return

    with open(path, encoding="utf-8") as f:
        raw_data = json.load(f)
    raw_cases = raw_data.get("validation_cases", {})

    coal_db, cases, metadata = load_original_paper_cases(path)
    op_meta = metadata.get("original_paper", metadata)
    geom_default = op_meta.get("reactor_dimensions", {})
    L_default = float(geom_default.get("length_m", 6.096))
    D_default = float(geom_default.get("diameter_m", 1.524))

    if args.limit:
        cases = dict(list(cases.items())[: args.limit])
        print(f"[Limit: first {args.limit} cases]")

    print("=" * 75)
    print("validation_cases_merged.json — 1D 动力学模型验证")
    print(f"N_cells={args.cells}, 默认几何 L={L_default}m D={D_default}m")
    print("=" * 75)

    results = {}
    for name, case in cases.items():
        inp = case["inputs"]
        oc_raw = raw_cases.get(name, {}).get("operating_conditions", {})
        L = float(oc_raw.get("L_reactor_m", L_default))
        D = float(oc_raw.get("D_reactor_m", D_default))
        geometry = {"L": L, "D": D}

        coal_key = inp["coal"]
        coal_props = coal_db.get(coal_key)
        if not coal_props:
            print(f"\n{name}: 煤种 {coal_key} 未找到，跳过")
            continue

        coal_flow = inp["FeedRate"] / 3600.0
        op_conds = {
            "coal_flow": coal_flow,
            "o2_flow": coal_flow * inp["Ratio_OC"],
            "steam_flow": coal_flow * inp["Ratio_SC"],
            "P": inp["P"],
            "T_in": inp["TIN"],
            "HeatLossPercent": inp.get("HeatLossPercent", 3.0),
            "SlurryConcentration": inp.get("SlurryConcentration", 100.0),
            "AdaptiveFirstCellLength": True,
        }

        try:
            system = GasifierSystem(geometry, coal_props, op_conds)
            arr, z = system.solve(N_cells=args.cells)
            last = arr[-1]
            T_out = last[10] - 273.15
            y = _dry_pct(last[:8])
            exp = case.get("expected", {})
            exp_dry = exp.get("dry_product_gas_vol_pct") or exp.get("target_output", {})
            T_exp = exp.get("outlet_temperature_C")
            if T_exp is None and "TOUT_K_approx" in exp:
                T_exp = exp["TOUT_K_approx"] - 273.15
            if not exp_dry and "YCO_pct_approx" in exp:
                exp_dry = {"CO": exp["YCO_pct_approx"], "H2": exp.get("YH2_pct_approx"), "CO2": exp.get("YCO2_pct_approx")}

            results[name] = {
                "predicted": {
                    "TOUT_C": round(T_out, 2),
                    "Y_CO_dry": round(y["CO"], 2),
                    "Y_H2_dry": round(y["H2"], 2),
                    "Y_CO2_dry": round(y["CO2"], 2),
                    "geometry_L": L,
                    "geometry_D": D,
                },
                "expected": {
                    "TOUT_C": T_exp,
                    "YCO": exp_dry.get("CO") if isinstance(exp_dry, dict) else None,
                    "YH2": exp_dry.get("H2") if isinstance(exp_dry, dict) else None,
                    "YCO2": exp_dry.get("CO2") if isinstance(exp_dry, dict) else None,
                },
            }
            print(f"\n{name}")
            print(f"  几何: L={L}m D={D}m  进料: {coal_flow*1000:.0f} g/s")
            print(f"  出口 T: {T_out:.1f} °C" + (f"  [期望 {T_exp} °C]" if T_exp else ""))
            print(f"  干基: CO={y['CO']:.1f}%  H2={y['H2']:.1f}%  CO2={y['CO2']:.1f}%")
        except Exception as e:
            print(f"\n{name}: 失败 - {e}")
            import traceback
            traceback.print_exc()

    out_path = os.path.join(ROOT, "data", "validation_results_merged.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
