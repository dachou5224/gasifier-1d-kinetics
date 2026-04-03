#!/usr/bin/env python3
import json
import os
import sys
import time
import numpy as np

# 确保导入 src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from model.gasifier_system import GasifierSystem
from model.chemistry import COAL_DATABASE

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "validation_cases_final.json")

def load_final_cases():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def run_test(name, case_data, n_cells=50):
    print(f"\n[测试算例] {name}")
    print(f"  描述: {case_data['description']}")
    
    cond = case_data["operating_conditions"]
    expected = case_data.get("expected_results", {})
    
    # 构造 GasifierSystem 的输入
    geometry = {
        'L': cond.get('L_reactor_m', 6.096),
        'D': cond.get('D_reactor_m', 1.524)
    }
    
    # 使用 COAL_DATABASE 中的现有煤种以确保鲁棒性
    coal_type = case_data.get("coal_type", "LuNan_Coal")
    if coal_type in COAL_DATABASE:
        coal_props = COAL_DATABASE[coal_type]
    else:
        # 尝试手动构造
        coal_data = case_data.get("coal", {})
        coal_props = {
            'Cd': coal_data.get('C', 70),
            'Hd': coal_data.get('H', 5),
            'Od': coal_data.get('O', 10),
            'Nd': coal_data.get('N', 1),
            'Sd': coal_data.get('S', 0.5),
            'Ad': coal_data.get('Ash', 10),
            'Vd': coal_data.get('Vd', 30),
            'FCd': coal_data.get('FCd', 60),
            'Mt': coal_data.get('Mt', 0),
            'HHV_d': coal_data.get('HHV_d', 30000)
        }

    # 计算流量 (转换为 kg/s)
    coal_flow_kg_hr = cond.get('coal_feed_rate_kg_hr', 100)
    coal_flow = coal_flow_kg_hr / 3600.0
    ratio_oc = cond.get('O2_to_fuel_ratio', 0.8)
    ratio_sc = cond.get('steam_to_fuel_ratio', 0.0)
    o2_purity = cond.get('O2_purity', 0.98)
    
    op_conds = {
        'coal_flow': coal_flow,
        'o2_flow': (coal_flow * ratio_oc) / o2_purity,
        'n2_flow': (coal_flow * ratio_oc) / o2_purity * (1.0 - o2_purity),
        'steam_flow': coal_flow * ratio_sc,
        'P': cond.get('pressure_Pa', 2.4e6),
        'T_in': 1200.0, # 强力辅助点火
        'HeatLossPercent': 0.0,
        'SlurryConcentration': cond.get('slurry_concentration_pct', 100.0),
        'AdaptiveFirstCellLength': True,
        'L_evap_m': 1.5 if cond.get('slurry_concentration_pct', 100.0) < 99.0 else 0.2
    }
    
    try:
        # 初始化系统
        system = GasifierSystem(geometry, coal_props, op_conds)
        
        # 第 1 次运行
        print("  正在执行 newton_fd 求解 (N_cells=100)...")
        start_time = time.time()
        profile_jax, info_jax = system.solve(N_cells=100, solver_method='newton_fd', jacobian_mode='centered_fd')
        compile_and_run_time = time.time() - start_time
        print(f"  [耗时] 第 1 次运行: {compile_and_run_time:.4f}s")

        # 第 2 次运行 (热启动)
        start_time = time.time()
        profile_jax2, info_jax2 = system.solve(N_cells=n_cells, solver_method='newton_fd', jacobian_mode='centered_fd')
        hot_run_time = time.time() - start_time
        print(f"  [耗时] 第 2 次运行 (热启动): {hot_run_time:.4f}s")

        # 结果对比
        last_cell = profile_jax[-1]
        gas_flow = last_cell[:7]
        total_gas = np.sum(gas_flow) + 1e-12
        dry_vol_pct = (gas_flow / total_gas) * 100
        exit_T_C = last_cell[8] - 273.15
        
        print(f"  [结果] 出口温度: {exit_T_C:.1f} °C (预期: {expected.get('outlet_temperature_C', 'N/A')} °C)")
        print(f"  [结果] 干基组成 (CO/H2/CO2): {dry_vol_pct[0]:.2f}/{dry_vol_pct[1]:.2f}/{dry_vol_pct[2]:.2f}")
        if "dry_product_gas_vol_pct" in expected:
            exp_pct = expected["dry_product_gas_vol_pct"]
            print(f"  [预期] 干基组成 (CO/H2/CO2): {exp_pct.get('CO')}/{exp_pct.get('H2')}/{exp_pct.get('CO2')}")
            
    except Exception as e:
        print(f"  [错误] 求解失败: {e}")

def main():
    data = load_final_cases()
    
    # 选取代表性算例
    if "LuNan_Texaco" in data["Industrial"]["Slurry-fed"]:
        run_test("LuNan_Texaco", data["Industrial"]["Slurry-fed"]["LuNan_Texaco"])
    
    if "Texaco_I-1" in data["Pilot"]["Dry-fed"]:
        run_test("Texaco_I-1", data["Pilot"]["Dry-fed"]["Texaco_I-1"])

if __name__ == "__main__":
    main()
