import json
import os
import sys
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from model.gasifier_system import GasifierSystem
from model.material import MaterialService, SPECIES_NAMES

def audit_cell_0_fluids():
    # 1. Load Case 6
    json_path = os.path.join(os.path.dirname(__file__), 'validation_cases.json')
    with open(json_path, 'r') as f:
        config = json.load(f)
    
    data = config['cases']['Paper_Case_6']
    inputs = data['inputs']
    
    coal_props = {
        'Cd': 80.19, 'Hd': 4.83, 'Od': 9.76, 'Ad': 7.35,
        'Nd': 0.0, 'Sd': 0.0, 'Mt': 0.0, 'Hf': -0.6e6
    }
    
    coal_flow_kg_s = inputs['FeedRate_kg_h'] / 3600.0
    o2_flow_kg_s = coal_flow_kg_s * inputs['Ratio_OC']
    steam_flow_kg_s = coal_flow_kg_s * inputs.get('SteamRatio_SC', 0.0)
    
    op_conds = {
        'coal_flow': coal_flow_kg_s,
        'o2_flow': o2_flow_kg_s,
        'steam_flow': steam_flow_kg_s,
        'P': inputs['P_Pa'],
        'T_in': inputs['T_in_K'],
        'L_reactor': inputs.get('L_reactor', 10.0),
        'D_reactor': inputs.get('D_reactor', 2.0)
    }
    
    geometry = {
        'L': inputs.get('L_reactor', 10.0),
        'D': inputs.get('D_reactor', 2.0)
    }
    
    # 2. Setup System
    system = GasifierSystem(geometry, coal_props, op_conds)
    N = 25
    system.dz = geometry['L'] / N
    
    # 3. Audit Inlet State (Cell 0)
    inlet = system.inlet_state
    W_gas_in = np.sum(inlet.gas_moles * np.array([31.998, 16.04, 28.01, 44.01, 34.08, 2.016, 28.013, 18.015])) * 1e-3
    total_moles_in = np.sum(inlet.gas_moles)
    avg_mw_in = W_gas_in * 1000.0 / max(total_moles_in, 1e-9)
    rho_in = (inlet.P * avg_mw_in * 1e-3) / (8.314 * inlet.T)
    A = np.pi * (geometry['D']/2.0)**2
    v_in = W_gas_in / (rho_in * A)
    
    print(f"--- Cell 0 INLET AUDIT ---")
    print(f"  T_in: {inlet.T} K, P: {inlet.P} Pa")
    print(f"  Gas Moles: {inlet.gas_moles}")
    print(f"  Total Gas Mass Flow (W_gas_in): {W_gas_in:.4f} kg/s")
    print(f"  Avg MW (g/mol): {avg_mw_in:.2f}")
    print(f"  Density (rho_in): {rho_in:.2f} kg/m3")
    print(f"  Area (A): {A:.2f} m2")
    print(f"  Inlet Velocity (v_in): {v_in:.4f} m/s")
    
    # Check if a 0.07 m/s value appears anywhere
    # Maybe coal mass flow was included in total? No, W_gas is gas only.
    
audit_cell_0_fluids()
