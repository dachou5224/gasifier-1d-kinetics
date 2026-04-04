import logging
import sys
import os
import json
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from model.gasifier_system import GasifierSystem
from model.chemistry import COAL_DATABASE
from model.state import StateVector
from model.material import SPECIES_NAMES
from model.validation_loader import CASE_NAME_ALIASES_FINAL

# Configure Logging to show INFO (for Cell Audit)
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def load_case_data(case_name):
    data_dir = os.path.join(os.path.dirname(__file__), '../../data')
    primary_path = os.path.join(data_dir, 'validation_cases_pilot.json')
    archive_path = os.path.join(data_dir, 'archive', 'validation_cases_pilot.json')
    json_path = primary_path if os.path.exists(primary_path) else archive_path
    with open(json_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    lookup = case_name
    reverse_aliases = {canonical: alias for alias, canonical in CASE_NAME_ALIASES_FINAL.items()}
    if lookup not in config['fortran_cases']:
        lookup = reverse_aliases.get(case_name, case_name)
    return config['fortran_cases'][lookup], config['coal_database']

if __name__ == "__main__":
    case_name = "Texaco_I-1"
    print(f"--- Diagnosing Energy Balance for {case_name} ---")
    
    # Load Data
    case_data, coal_db = load_case_data(case_name)
    inputs = case_data['inputs']
    
    # Op Conds
    coal_flow_kg_s = inputs['FeedRate'] / 3600.0
    o2_flow_kg_s = coal_flow_kg_s * inputs['Ratio_OC']
    steam_flow_kg_s = coal_flow_kg_s * inputs.get('Ratio_SC', 0.0)
    
    op_conds = {
        'coal_flow': coal_flow_kg_s,
        'o2_flow': o2_flow_kg_s,
        'steam_flow': steam_flow_kg_s,
        'P': inputs['P'],
        'T_in': inputs['TIN'],
        'SlurryConcentration': inputs.get('SlurryConcentration', 100.0),
        'HeatLossPercent': inputs.get('HeatLossPercent', 4.0),
        'epsilon': inputs.get('Voidage', 1.0)
    }
    
    geometry = {
        'L': inputs.get('L', 2.0),
        'D': inputs.get('D', 0.2)
    }
    
    # Coal Props
    coal_props = coal_db[inputs['coal']].copy()
    
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
        sol0_arr = results[0]
        sol0 = StateVector.from_array(sol0_arr, P=cell0.inlet.P, z=0.0)
        
        # Force a residual check to trigger logs
        res = cell0.residuals(sol0_arr)
        
        t_in_val = float(np.atleast_1d(cell0.inlet.T_solid_or_gas)[0])
        p_in_val = float(np.atleast_1d(cell0.inlet.P)[0]) / 1e5
        t_out_val = float(np.atleast_1d(sol0.T)[0])
        solid_out_val = float(np.atleast_1d(sol0.solid_mass)[0])
        
        print(f"T_in = {t_in_val:.1f} K, P_in = {p_in_val:.1f} bar")
        print(f"T_out = {t_out_val:.1f} K, Solid_mass = {solid_out_val:.4f} kg/s")
        print("Inlet Gas Moles:")
        for i, sp in enumerate(SPECIES_NAMES):
            val = float(np.atleast_1d(cell0.inlet.gas_moles[i])[0])
            if val > 1e-4:
                print(f"  {sp:4}: {val:.4f} mol/s")
        print("Gas Source (Pyrolysis):")
        for s in cell0.sources:
            g, sm, e = s.get_sources(0, cell0.z, cell0.dz)
            for i, sp in enumerate(SPECIES_NAMES):
                val = float(np.atleast_1d(g[i])[0])
                if abs(val) > 1e-4:
                    print(f"  {sp:4}: {val:.4f} mol/s")
        
    except Exception as e:
        print(f"Details: {e}")
