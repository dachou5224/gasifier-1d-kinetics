"""
全量 Profile 一致性测试：逐格对比 JAX-JIT 与 Scipy-Minimize 的组分与温度分布。
"""
import os
import sys
import numpy as np
import time

# 将 src 目录添加到 Python 路径
sys.path.append(os.path.join(os.getcwd(), 'gasifier-1d-kinetic', 'src'))

from model.gasifier_system import GasifierSystem

def run_profile_comparison():
    # 典型 Texaco 工况
    geometry = {'L': 6.0, 'D': 1.0}
    coal_props = {
        "Cd": 71.0, "Hd": 5.1, "Od": 10.2, "Nd": 1.3, "Sd": 0.5, "Ad": 11.9,
        "Vd": 35.0, "FCd": 53.1, "Mt": 0.0, "HHV_d": 30500.0
    }
    op_conds = {
        "coal_flow": 11.575, # kg/s
        "o2_flow": 10.0,     # kg/s
        "P": 24.0 * 101325.0,
        "T_in": 500.0,
        "Ratio_SC": 0.2,
        "HeatLossPercent": 3.0,
        "AdaptiveFirstCellLength": True
    }

    N_CELLS = 20
    print(f"\nRunning Full Profile Comparison ({N_CELLS} cells)...")

    # 1. Scipy 求解
    sys_scipy = GasifierSystem(geometry, coal_props, op_conds)
    res_scipy, z_scipy = sys_scipy.solve(N_cells=N_CELLS, solver_method='minimize')

    # 2. JAX-JIT 求解
    sys_jax = GasifierSystem(geometry, coal_props, op_conds)
    res_jax, z_jax = sys_jax.solve(N_cells=N_CELLS, solver_method='jax_jit')

    # 3. 组分映射
    names = ['O2', 'CH4', 'CO', 'CO2', 'H2S', 'H2', 'N2', 'H2O', 'Ws', 'Xc', 'T']
    
    print("\n" + "="*80)
    print(f"{'Component':<10} | {'Max Diff':<12} | {'Avg Diff':<12} | {'Outlet (S)':<12} | {'Outlet (J)':<12}")
    print("-" * 80)

    for i, name in enumerate(names):
        s_prof = res_scipy[:, i]
        j_prof = res_jax[:, i]
        
        diff = np.abs(s_prof - j_prof)
        max_diff = np.max(diff)
        avg_diff = np.mean(diff)
        
        print(f"{name:<10} | {max_diff:12.4e} | {avg_diff:12.4e} | {s_prof[-1]:12.2f} | {j_prof[-1]:12.2f}")

    print("="*80)

    # 4. 关键点详细诊断 (前 5 格)
    print("\nDetailed Diagnostic (First 5 Cells Temperature):")
    print(f"{'Cell':<6} | {'Z(m)':<8} | {'T_Scipy(K)':<12} | {'T_JAX(K)':<12} | {'Diff(K)':<10}")
    for i in range(5):
        print(f"{i:<6} | {z_scipy[i]:<8.3f} | {res_scipy[i, 10]:<12.1f} | {res_jax[i, 10]:<12.1f} | {res_scipy[i, 10]-res_jax[i, 10]:<10.1f}")

if __name__ == "__main__":
    run_profile_comparison()
