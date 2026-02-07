import logging
import sys
import os
import json

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from model.gasifier_system import GasifierSystem
from model.chemistry import COAL_DATABASE

# Configure Logging to show INFO (for Cell Audit)
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def load_case_data(case_name):
    json_path = os.path.join(os.path.dirname(__file__), '../../data/validation_cases.json')
    with open(json_path, 'r') as f:
        config = json.load(f)
    return config['cases'][case_name]

if __name__ == "__main__":
    case_name = "Paper_Case_6"
    print(f"--- Diagnosing Energy Balance for {case_name} ---")
    
    # Load Data
    case_data = load_case_data(case_name)
    inputs = case_data['inputs']
    
    # Op Conds
    coal_flow_kg_s = inputs['FeedRate_kg_h'] / 3600.0
    o2_flow_kg_s = coal_flow_kg_s * inputs['Ratio_OC']
    
    op_conds = {
        'coal_flow': coal_flow_kg_s,
        'o2_flow': o2_flow_kg_s,
        'steam_flow': 0.0, # Explicitly 0 for Case 6
        'P': inputs['P_Pa'],
        'T_in': inputs['T_in_K'],
        'SlurryConcentration': 62.0,
        'HeatLossPercent': 1.0,
        'epsilon': inputs.get('Voidage', 1.0)
    }
    
    # Coal Props
    coal_props = COAL_DATABASE[case_data['coal_type']].copy()
    
    # Initialize System with WARNING logging to suppress iteration logs
    logging.getLogger('model.cell').setLevel(logging.WARNING)
    logging.getLogger('model.gasifier_system').setLevel(logging.WARNING)
    
    system = GasifierSystem(geometry, coal_props, op_conds)
    
    print("Running solve(N_cells=1)...")
    try:
        results, z = system.solve(N_cells=1, solver_method='minimize') 
        
        # Now enable INFO and run residuals on the result to see the Final Balance
        print("\n=== CONVERGED STATE DIAGNOSTICS ===")
        logging.getLogger('model.cell').setLevel(logging.INFO)
        
        cell0 = system.cells[0]
        sol0 = results[0]
        
        # Force a residual check to trigger logs
        res = cell0.residuals(sol0)
        
        print(f"Final Residual Vector Norm: {np.linalg.norm(res):.2e}")
        
    except Exception as e:
        print(f"Details: {e}")
