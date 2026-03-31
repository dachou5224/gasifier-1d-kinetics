#!/usr/bin/env python3
import json
import os
import sys
import time
import numpy as np

# 确保导入 src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from model.gasifier_system import GasifierSystem
from model.chemistry import COAL_DATABASE, CASE_TO_COAL_MAP

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "validation_cases_final.json")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "docs", "validation_full_report.md")

def load_cases():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def run_single_case(name, case_data, n_cells=30):
    cond = case_data["operating_conditions"]
    expected = case_data.get("expected_results", {})
    orig_feedstock = case_data.get("feedstock_type_orig", "").lower()
    
    geometry = {
        'L': cond.get('L_reactor_m', 6.0),
        'D': cond.get('D_reactor_m', 1.5)
    }
    
    # [PHASE 2] Coal Mapping
    coal_type = case_data.get("coal_type")
    if name in CASE_TO_COAL_MAP:
        mapped_coal = CASE_TO_COAL_MAP[name]
        if mapped_coal in COAL_DATABASE:
            coal_type = mapped_coal
    
    if not coal_type or coal_type not in COAL_DATABASE:
        coal_type = "Paper_Base_Coal"
    coal_props = COAL_DATABASE[coal_type]

    # [PHASE 1] PHYSICS-DRIVEN FEED DETERMINATION
    # Precise override for Residue/Texaco cases
    name_lower = name.lower()
    
    # Priority 1: Use explicit feed_type from case_data
    feed_type_explicit = case_data.get("feed_type")
    
    # Priority 2: Heuristic detection
    is_true_slurry = "slurry" in name_lower or "coal_water_slurry" in orig_feedstock
    is_residue = "residue" in orig_feedstock or "molten" in orig_feedstock
    
    if feed_type_explicit:
        feed_type = feed_type_explicit
    elif is_true_slurry and not is_residue:
        feed_type = "Slurry-fed"
    else:
        # Default to Dry-fed physics (steam injection) for Residue and Paper cases
        feed_type = "Dry-fed"

    slurry_conc = cond.get('slurry_concentration_pct', 100.0)
    coal_flow_kg_h = cond.get('coal_feed_rate_kg_hr', 100.0)
    if cond.get('coal_feed_rate_g_s'):
        coal_flow_kg_h = cond.get('coal_feed_rate_g_s') * 3.6
    coal_flow = coal_flow_kg_h / 3600.0

    ratio_oc = cond.get('O2_to_coal_ratio', cond.get('O2_to_fuel_ratio', cond.get('Ratio_OC', 0.9)))
    raw_water_ratio = cond.get('water_to_coal_ratio', cond.get('steam_to_fuel_ratio', cond.get('Ratio_SC', 0.0)))
    
    if feed_type == "Slurry-fed":
        # Carrier water physics (Latent Heat)
        if slurry_conc == 100.0 or slurry_conc is None:
            slurry_conc = 100.0 / (1.0 + raw_water_ratio)
        ratio_sc = 0.0 # All water is liquid carrier
    else:
        # Steam injection physics (No Latent Heat in Cell 0)
        ratio_sc = raw_water_ratio
        slurry_conc = 100.0

    P = cond.get('pressure_Pa', cond.get('pressure_atm', 1.0) * 101325.0)

    op_conds = cond.copy()
    op_conds.update({
        'coal_flow': coal_flow,
        'o2_flow': coal_flow * ratio_oc,
        'steam_flow': coal_flow * ratio_sc,
        'P': P,
        'T_in': cond.get('inlet_temperature_K', 400.0),
        'HeatLossPercent': cond.get('heat_loss_percent', 2.0),
        'SlurryConcentration': slurry_conc,
        'AdaptiveFirstCellLength': True,
    })
    
    # Ensure backward compatibility for key tuning params
    if 'Combustion_CO2_Fraction' not in op_conds:
        op_conds['Combustion_CO2_Fraction'] = 0.15
    if 'WGS_CatalyticFactor' not in op_conds:
        op_conds['WGS_CatalyticFactor'] = 1.5
    
    print(f"  [Input Audit] {feed_type:10} | {orig_feedstock:15} | O2/C: {ratio_oc:.3f} | S/C: {ratio_sc:.3f} | Conc: {slurry_conc:4.1f}%")

    res = {
        "name": name,
        "capacity": case_data.get("capacity", "N/A"),
        "feed": feed_type,
        "status": "Failed",
        "time": 0.0,
        "T_out": 0.0,
        "T_out_err": 0.0,
        "CO": 0.0,
        "CO_err": 0.0
    }

    try:
        system = GasifierSystem(geometry, coal_props, op_conds)
        start_t = time.time()
        profile, info = system.solve(N_cells=n_cells, solver_method='jax_pure')
        res["time"] = time.time() - start_t
        last = profile[-1]
        res["T_out"] = last[10] - 273.15
        res["status"] = "Success" if res["T_out"] > 500 else "Cold-Exit"
        
        # Composition Dry
        gas_flow = last[:8]
        total_gas_dry = np.sum(gas_flow[:7]) + 1e-12
        y_dry = (gas_flow / total_gas_dry) * 100
        res["CO"] = y_dry[2]
    except Exception as e:
        res["status"] = f"Error: {str(e)[:15]}"
        
    return res

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cells", type=int, default=40)
    args = ap.parse_args()

    data = load_cases()
    results = []
    
    print(f"开始验证所有算例 (N_cells={args.cells}, Feedstock-Type Aware)...")
    
    for cap in ["Pilot", "Industrial"]:
        for ftype in ["Slurry-fed", "Dry-fed"]:
            cases = data[cap][ftype]
            if not cases: continue
            print(f"\n正在处理 {cap} | {ftype} ({len(cases)} 算例)...")
            for name, case_data in cases.items():
                if args.limit and len(results) >= args.limit: break
                r = run_single_case(name, case_data, n_cells=args.cells)
                results.append(r)
                print(f"  - {name}: {r['status']} (T={r['T_out']:.0f}C, CO={r['CO']:.1f}%)")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# 验证算例全量测试报告\n\n")
        f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("> **注**: 已启用 feedstock_type_orig 深度感知。molten_residue 强制按 Dry-fed (蒸汽) 物理模型处理。\n\n")
        header = "| 算例名称 | 状态 | T_out(C) | CO% | 投料判定 |\n"
        sep    = "| :--- | :--- | :--- | :--- | :--- |\n"
        f.write(header + sep)
        for r in results:
            f.write(f"| {r['name']} | {r['status']} | {r['T_out']:.1f} | {r['CO']:.2f} | {r['feed']} |\n")
        
    print(f"\n报告已生成: {REPORT_PATH}")

if __name__ == "__main__":
    main()
