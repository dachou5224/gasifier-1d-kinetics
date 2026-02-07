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
    return config['cases'][case_name]

def run_case(case_name, solver_method):
    print(f"\n--- Running {case_name} with {solver_method.upper()} ---")
    
    # 1. Setup
    case_data = load_case_data(case_name)
    inputs = case_data['inputs']
    
    # Coal Props
    coal_key = case_data['coal_type']
    coal_props = COAL_DATABASE[coal_key].copy()
    
    # Op Conds Calculation
    coal_flow_kg_s = inputs['FeedRate_kg_h'] / 3600.0
    o2_flow_kg_s = coal_flow_kg_s * inputs['Ratio_OC']
    steam_flow_kg_s = coal_flow_kg_s * inputs.get('SteamRatio_SC', 0.0)
    
    op_conds = {
        'coal_flow': coal_flow_kg_s,
        'o2_flow': o2_flow_kg_s,
        'steam_flow': steam_flow_kg_s,
        'P': inputs['P_Pa'],
        'T_in': inputs['T_in_K'],
        'HeatLossPercent': inputs.get('HeatLossPercent', 1.0),
        'epsilon': inputs.get('Voidage', 1.0) # Default voidage
    }
    
    # Case 6 Override (Critical)
    if case_name == 'Paper_Case_6':
        op_conds['SlurryConcentration'] = 62.0
        op_conds['steam_flow'] = 0.0 # Clean up as inputs.get('SteamRatio_SC') might be 0.12 or 0.0 depending on version
    
    geometry = {
        'L': inputs.get('L_reactor', 6.0),
        'D': inputs.get('D_reactor', 2.0)
    }
    
    # 2. Run
    system = GasifierSystem(geometry, coal_props, op_conds)
    
    start_time = time.time()
    try:
        results, z = system.solve(N_cells=20, solver_method=solver_method)
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
            'method': solver_method,
            'time': duration,
            'T_out': T_out,
            'Y_CO': y_co,
            'Res': res_norm,
            'status': 'Success'
        }
    else:
        return {
            'method': solver_method,
            'time': duration,
            'status': 'Failed'
        }

if __name__ == "__main__":
    print("=== Solver Comparison Benchmark ===")
    
    case = "Paper_Case_6"
    
    res_min = run_case(case, 'minimize')
    res_new = run_case(case, 'newton')
    
    print("\n=== Summary Table ===")
    print(f"{'Method':<15} | {'Status':<10} | {'Time (s)':<10} | {'T_out (K)':<10} | {'CO (%)':<8} | {'Res Norm':<10}")
    print("-" * 80)
    for r in [res_min, res_new]:
        if r['status'] == 'Success':
            print(f"{r['method']:<15} | {r['status']:<10} | {r['time']:<10.2f} | {r['T_out']:<10.1f} | {r['Y_CO']:<8.1f} | {r['Res']:<10.1e}")
        else:
            print(f"{r['method']:<15} | {r['status']:<10} | {r['time']:<10.2f} | N/A        | N/A      | N/A")
    print("=================================")
