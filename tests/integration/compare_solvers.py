import logging
import time
import numpy as np
import json
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from model.gasifier_system import GasifierSystem
from model.chemistry import COAL_DATABASE

# Configure condensed logging
logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s') # Suppress INFO logs
logger = logging.getLogger(__name__)

def load_case_data(case_name):
    json_path = os.path.join(os.path.dirname(__file__), '../../data/validation_cases.json')
    with open(json_path, 'r') as f:
        config = json.load(f)
    if 'cases' in config and case_name in config['cases']:
        return config['cases'][case_name]
    # 兼容当前仓库以 model.chemistry.VALIDATION_CASES 为主的数据组织
    from model.chemistry import VALIDATION_CASES
    if case_name in VALIDATION_CASES:
        return VALIDATION_CASES[case_name]
    raise KeyError(f"Case '{case_name}' not found in validation_cases.json or VALIDATION_CASES")

def run_case(case_name, solver_method, **solve_kwargs):
    extra = f" {solve_kwargs}" if solve_kwargs else ""
    method_name = solver_method if not solve_kwargs else f"{solver_method}+{','.join(sorted(solve_kwargs.keys()))}"
    print(f"\n--- Running {case_name} with {solver_method.upper()}{extra} ---")
    
    # 1. Setup
    case_data = load_case_data(case_name)
    inputs = case_data['inputs']
    
    # Coal Props
    coal_key = case_data.get('coal_type', inputs.get('coal'))
    coal_props = COAL_DATABASE[coal_key].copy()
    
    # Op Conds Calculation
    feed_rate = inputs.get('FeedRate_kg_h', inputs.get('FeedRate'))
    coal_flow_kg_s = feed_rate / 3600.0
    o2_flow_kg_s = coal_flow_kg_s * inputs.get('Ratio_OC', 1.0)
    steam_flow_kg_s = coal_flow_kg_s * inputs.get('SteamRatio_SC', inputs.get('Ratio_SC', 0.0))
    
    op_conds = {
        'coal_flow': coal_flow_kg_s,
        'o2_flow': o2_flow_kg_s,
        'steam_flow': steam_flow_kg_s,
        'P': inputs.get('P_Pa', inputs.get('P')),
        'T_in': inputs.get('T_in_K', inputs.get('TIN')),
        'HeatLossPercent': inputs.get('HeatLossPercent', 1.0),
        'epsilon': inputs.get('Voidage', 1.0) # Default voidage
    }
    
    # Case 6 Override (Critical)
    if case_name == 'Paper_Case_6':
        op_conds['SlurryConcentration'] = inputs.get('SlurryConcentration', 62.0)
        op_conds['steam_flow'] = 0.0 # 保持与历史基准一致
    
    geometry = {
        'L': inputs.get('L_reactor', 6.0),
        'D': inputs.get('D_reactor', 2.0)
    }
    
    # 2. Run
    system = GasifierSystem(geometry, coal_props, op_conds)
    
    start_time = time.time()
    try:
        results, z = system.solve(N_cells=20, solver_method=solver_method, **solve_kwargs)
        success = True
    except Exception as e:
        print(f"Solver Failed: {e}")
        success = False
        results = None
    
    duration = time.time() - start_time
    
    # 3. Metrics
    if success:
        T_out = results[-1][10]
        # Calculate Y_CO
        res_out = results[-1]
        gas_moles = res_out[0:8]
        total = np.sum(gas_moles)
        y_co = (res_out[2]/total)*100 if total > 0 else 0
        
        # Calculate Residual Norm for Final Cell
        # We need to construct the residual vector
        final_cell = system.cells[-1]
        final_res_vec = final_cell.residuals(results[-1])
        res_norm = np.max(np.abs(final_res_vec))
        
        print(f"Result: Success in {duration:.2f}s")
        print(f"  T_out: {T_out:.1f} K")
        print(f"  Y_CO:  {y_co:.1f} %")
        print(f"  Res:   {res_norm:.1e}")
        return {
            'method': method_name,
            'time': duration,
            'T_out': T_out,
            'Y_CO': y_co,
            'Res': res_norm,
            'status': 'Success'
        }
    else:
        return {
            'method': method_name,
            'time': duration,
            'status': 'Failed'
        }

if __name__ == "__main__":
    print("=== Solver Comparison Benchmark ===")
    
    case = "Paper_Case_6"
    
    res_min = run_case(case, 'minimize')
    res_new = run_case(case, 'newton')
    res_jac = run_case(case, 'minimize', use_jax_jacobian=True)
    res_jaxn = run_case(case, 'jax_newton', jax_warmup=True)
    
    print("\n=== Summary Table ===")
    print(f"{'Method':<22} | {'Status':<10} | {'Time (s)':<10} | {'T_out (K)':<10} | {'CO (%)':<8} | {'Res Norm':<10}")
    print("-" * 90)
    for r in [res_min, res_new, res_jac, res_jaxn]:
        if r['status'] == 'Success':
            print(f"{r['method']:<22} | {r['status']:<10} | {r['time']:<10.2f} | {r['T_out']:<10.1f} | {r['Y_CO']:<8.1f} | {r['Res']:<10.1e}")
        else:
            print(f"{r['method']:<22} | {r['status']:<10} | {r['time']:<10.2f} | N/A        | N/A      | N/A")
    print("=================================")
