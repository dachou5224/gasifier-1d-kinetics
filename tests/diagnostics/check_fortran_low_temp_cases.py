"""
检查 Fortran 工况中出口温度低于 1200°C 的工况：
1. 检查入口是否重复输入煤浆水
2. 检查 Steam/Coal 比例是否在合理范围
3. 特别关注标注 "texaco" 的工况
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from model.fortran_input_loader import load_fortran_cases, get_fortran_input_path
from model.gasifier_system import GasifierSystem
import numpy as np


def dry_mole_fraction_pct(gas_moles):
    F_dry = sum(gas_moles[:7]) + 1e-12
    return {
        "CO": gas_moles[2] / F_dry * 100,
        "H2": gas_moles[5] / F_dry * 100,
        "CO2": gas_moles[3] / F_dry * 100,
    }


def main():
    path = get_fortran_input_path()
    coal_db, cases = load_fortran_cases(path)
    
    geometry = {"L": 8.0, "D": 2.6}
    n_cells = 50
    
    print("=" * 80)
    print("Fortran 工况温度与输入条件检查（重点关注 T_out < 1200°C）")
    print("=" * 80)
    
    low_temp_cases = []
    all_results = []
    
    for name, data in cases.items():
        inp = data["inputs"]
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
            T_out_C = last[10] - 273.15
            y = dry_mole_fraction_pct(last[:8])
            
            # 计算实际水输入
            coal_flow_kg_s = op_conds["coal_flow"]
            steam_flow_kg_s = op_conds["steam_flow"]
            Mt = coal_props.get("Mt", 0.0) / 100.0
            coal_moisture_kg_s = coal_flow_kg_s * Mt
            
            # 如果 SlurryConcentration < 100，还有浆液水
            slurry_conc = op_conds.get("SlurryConcentration", 100.0)
            W_dry = coal_flow_kg_s * (1 - Mt)
            if slurry_conc < 100.0:
                W_slurry_h2o = (W_dry / (slurry_conc/100.0)) - W_dry
            else:
                W_slurry_h2o = 0.0
            
            total_h2o_kg_s = coal_moisture_kg_s + W_slurry_h2o + steam_flow_kg_s
            steam_coal_ratio = steam_flow_kg_s / coal_flow_kg_s if coal_flow_kg_s > 0 else 0
            
            result = {
                "name": name,
                "T_out": T_out_C,
                "Ratio_SC": inp["Ratio_SC"],
                "steam_flow_kg_s": steam_flow_kg_s,
                "coal_moisture_kg_s": coal_moisture_kg_s,
                "slurry_h2o_kg_s": W_slurry_h2o,
                "total_h2o_kg_s": total_h2o_kg_s,
                "steam_coal_ratio": steam_coal_ratio,
                "SlurryConcentration": slurry_conc,
                "CO": y["CO"],
                "H2": y["H2"],
                "CO2": y["CO2"],
            }
            all_results.append(result)
            
            if T_out_C < 1200.0:
                low_temp_cases.append(result)
                
        except Exception as e:
            print(f"\n{name}: FAILED - {e}")
    
    # 打印所有结果
    print("\n所有工况汇总：")
    print(f"{'工况':<20} {'T_out(°C)':<12} {'Ratio_SC':<10} {'Steam/Coal':<12} {'浆液水(kg/s)':<15} {'总水(kg/s)':<12}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['name']:<20} {r['T_out']:<12.1f} {r['Ratio_SC']:<10.3f} {r['steam_coal_ratio']:<12.3f} "
              f"{r['slurry_h2o_kg_s']:<15.2f} {r['total_h2o_kg_s']:<12.2f}")
    
    # 重点检查低温工况
    print("\n" + "=" * 80)
    print("温度 < 1200°C 的工况详细分析：")
    print("=" * 80)
    
    for r in low_temp_cases:
        is_texaco = "texaco" in r["name"].lower()
        print(f"\n【{r['name']}】{' (Texaco)' if is_texaco else ''}")
        print(f"  出口温度: {r['T_out']:.1f} °C")
        print(f"  Ratio_SC (fsteam): {r['Ratio_SC']:.3f}  (Steam/Coal 质量比)")
        print(f"  煤水分 (xmois): {r['coal_moisture_kg_s']:.3f} kg/s")
        print(f"  浆液水 (SlurryConcentration={r['SlurryConcentration']:.0f}%): {r['slurry_h2o_kg_s']:.3f} kg/s")
        print(f"  外加蒸汽 (steam_flow): {r['steam_flow_kg_s']:.3f} kg/s")
        print(f"  总水输入: {r['total_h2o_kg_s']:.3f} kg/s")
        print(f"  实际 Steam/Coal 比: {r['steam_coal_ratio']:.3f}")
        
        # 合理性检查
        issues = []
        warnings = []
        
        # 检查重复输入
        if r['slurry_h2o_kg_s'] > 0.1 and r['steam_flow_kg_s'] > 0.1:
            issues.append("⚠️  可能重复输入：既有浆液水又有外加蒸汽")
        
        # Texaco 典型 steam/coal: 0.1-0.3 (文献范围)
        if is_texaco:
            if r['Ratio_SC'] > 0.5:
                issues.append(f"⚠️  Texaco 工况 Steam/Coal={r['Ratio_SC']:.3f} 严重偏高（典型 0.1-0.3）")
            elif r['Ratio_SC'] > 0.35:
                warnings.append(f"⚠️  Texaco 工况 Steam/Coal={r['Ratio_SC']:.3f} 偏高（典型 0.1-0.3）")
            elif r['Ratio_SC'] > 0.3:
                warnings.append(f"ℹ️  Texaco 工况 Steam/Coal={r['Ratio_SC']:.3f} 略高于典型上限（0.3）")
            
            if r['slurry_h2o_kg_s'] > 0.1:
                issues.append("⚠️  Texaco 应为干粉或高固含浆态，不应有大量液态浆液水")
            
            # 温度检查
            if r['T_out'] < 1000:
                issues.append(f"⚠️  出口温度 {r['T_out']:.1f}°C 过低（Texaco 通常 >1100°C）")
            elif r['T_out'] < 1100:
                warnings.append(f"⚠️  出口温度 {r['T_out']:.1f}°C 偏低（Texaco 通常 >1100°C）")
        
        if issues:
            print("  问题：")
            for issue in issues:
                print(f"    {issue}")
        if warnings:
            print("  警告：")
            for warning in warnings:
                print(f"    {warning}")
        if not issues and not warnings:
            print("  ✓ 输入条件合理")
    
    print("\n" + "=" * 80)
    print("建议：")
    print("  1. Texaco 工况：fsteam 通常 0.1-0.3，若 >0.5 需核实")
    print("  2. 若 fsteam 已包含煤浆水（作为蒸汽），SlurryConcentration 应设为 100")
    print("  3. 若 SlurryConcentration < 100，fsteam 应为纯外加蒸汽（不含浆水）")
    print("=" * 80)


if __name__ == "__main__":
    main()
