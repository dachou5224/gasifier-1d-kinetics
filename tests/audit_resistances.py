import json
import os
import sys
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from model.gasifier_system import GasifierSystem
from model.material import MaterialService, SPECIES_NAMES
from model.physics import R_CONST

def audit_kinetics_limitations():
    # 1. Load Case 6
    json_path = os.path.join(os.path.dirname(__file__), 'validation_cases.json')
    if not os.path.exists(json_path):
        json_path = '/Users/liuzhen/AI-projects/gasifier-1d-kinetic/tests/validation_cases.json'
    with open(json_path, 'r') as f:
        config = json.load(f)
    
    data = config['cases']['Paper_Case_6']
    inputs = data['inputs']
    
    coal_props = {
        'Cd': 80.19, 'Hd': 4.83, 'Od': 9.76, 'Ad': 7.35,
        'Nd': 0.0, 'Sd': 0.0, 'Mt': 0.0, 'Hf': -0.6e6,
        'HHV_d': 30.0
    }
    
    coal_flow_kg_s = inputs['FeedRate_kg_h'] / 3600.0
    op_conds = {
        'coal_flow': coal_flow_kg_s,
        'o2_flow': coal_flow_kg_s * inputs['Ratio_OC'],
        'steam_flow': coal_flow_kg_s * inputs.get('SteamRatio_SC', 0.0),
        'P': inputs['P_Pa'],
        'T_in': inputs['T_in_K'],
        'L_reactor': inputs.get('L_reactor', 6.0),
        'D_reactor': inputs.get('D_reactor', 2.0)
    }
    
    system = GasifierSystem({'L': 6.0, 'D': 2.0}, coal_props, op_conds)
    results, z = system.solve(N_cells=25)
    
    print(f"{'Cell':<4} | {'T':<6} | {'u_g':<6} | {'tau':<6} | {'X':<5} | {'R_kin %':<8} | {'R_diff %':<8} | {'Rate(Het)'}")
    print("-" * 75)
    
    for i in range(len(system.cells)):
        cell = system.cells[i]
        sol_x = results[i]
        from model.state import StateVector
        current = StateVector.from_array(sol_x, P=op_conds['P'], z=z[i])
        
        # Physics and Rates
        g_src = np.zeros(8)
        for s in cell.sources:
            g, _, _ = s.get_sources(cell.idx, cell.z, cell.dz)
            g_src += g
            
        phys = cell._calc_physics_props(current, coal_flow_kg_s * (coal_props['Cd']/100.0))
        # Internal audit of kinetics resistances
        from model.physics import calculate_diffusion_coefficient, calculate_gas_viscosity
        
        # Re-eval C+H2O for resistance breakdown
        rxn = 'C+H2O'
        T, P = current.T, current.P
        d_p = phys['d_p']
        Y = (1.0 - phys['X_total'])**(1/3.0)
        Y = max(Y, 1e-4)
        
        D_i = calculate_diffusion_coefficient(T, P, 'H2O')
        Sh = 2.0 # simplified
        k_d = (Sh * D_i) / d_p
        k_ash = k_d * (0.75**2.5)
        
        A_p = cell.kinetics.het_model.params[rxn]['A']
        E_p = cell.kinetics.het_model.params[rxn]['E']
        k_s = A_p * np.exp(-E_p / (R_CONST * T))
        
        r_diff = 1.0 / k_d
        r_ash = (1-Y) / (k_ash * Y)
        r_kin = 1.0 / (k_s * Y**2)
        r_total = r_diff + r_ash + r_kin
        
        pct_kin = (r_kin / r_total) * 100
        pct_diff = (r_diff / r_total) * 100
        
        v_g = phys['v_g']
        tau = cell.dz / max(v_g, 1e-3)
        
        # Get C+H2O rate (mol/s)
        r_het, _, _ = cell._calc_rates(current, phys, g_src)
        rate_h2o = r_het.get('C+H2O', 0.0)
        
        print(f"{i:<4} | {T:6.0f} | {v_g:6.2f} | {tau:6.3f} | {phys['X_total']:5.3f} | {pct_kin:8.1f} | {pct_diff:8.1f} | {rate_h2o:8.1f}")

if __name__ == '__main__':
    audit_kinetics_limitations()
