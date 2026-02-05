import sys
import os
import numpy as np
import logging

# Configure basic logging for tests (INFO level to suppress DEBUG noise)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# 添加 src 到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from model.gasifier_system import GasifierSystem

def test_solver():
    # 使用 Paper Case 6 进行对标 (刘臻 论文)
    coal_props = {
        'Cd': 80.19,
        'Hd': 4.83,
        'Od': 9.76,
        'Ad': 7.35,
        'Hf': -0.6e6 # 估算值 J/kg
    }
    
    # Paper Case 6
    coal_flow = 41670.0 / 3600.0 # kg/s (~11.57)
    operating_conds = {
        'coal_flow': coal_flow,
        'o2_flow': coal_flow * 1.05,
        'steam_flow': coal_flow * 0.08,
        'P': 4.08e6,
        'T_in': 300.0
    }
    
    geometry = {'L': 6.0, 'D': 2.0}
    
    # Use GasifierSystem instead of GasifierSolver1D
    solver = GasifierSystem(geometry, coal_props, operating_conds)
    
    print("Starting 1D Solver Test...")
    # GasifierSystem.solve returns (results_array, z_positions)
    results, z_positions = solver.solve(N_cells=25)
    
    print(f"Solver finished. Number of cells solved: {len(results)}")
    
    # Calculate initial Carbon
    C_inlet = coal_flow * (coal_props['Cd']/100.0)
    
    print("\nAxial Profile Data:")
    print(f"{'Cell':<4} | {'Tg (K)':<8} | {'Tp(es)(K)':<10} | {'Xc (%)':<8} | {'O2':<8} | {'CO':<8} | {'H2':<8} | {'CO2':<8}")
    print("-" * 80)
    
    for i, res in enumerate(results):
        T = res[10]
        W_s = res[8]
        X_C_frac = res[9]
        
        # Conversion
        C_remaining = W_s * X_C_frac
        conversion = (1 - C_remaining/C_inlet) * 100.0
        
        # Estimate Tp (Manual Check)
        F_gas = res[:8]
        F_total = sum(F_gas) + 1e-9
        P = operating_conds['P']
        R = 8.314
        C_O2_kmol = (max(F_gas[0],0)/F_total) * P / (R * T) / 1000.0
        Tp_est = T + 6.6e4 * C_O2_kmol
        
        print(f"{i:<4d} | {T:<8.1f} | {Tp_est:<10.1f} | {conversion:<8.1f} | {F_gas[0]:<8.2f} | {F_gas[2]:<8.2f} | {F_gas[5]:<8.2f} | {F_gas[3]:<8.2f}")

if __name__ == "__main__":
    test_solver()
