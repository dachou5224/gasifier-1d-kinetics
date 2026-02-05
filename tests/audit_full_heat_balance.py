import json
import os
import sys
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from model.gasifier_system import GasifierSystem
from model.material import MaterialService, SPECIES_NAMES

def audit_full_heat_balance():
    # 1. Load Case 6
    json_path = os.path.join(os.path.dirname(__file__), 'validation_cases.json')
    with open(json_path, 'r') as f:
        config = json.load(f)
    
    data = config['cases']['Paper_Case_6']
    inputs = data['inputs']
    
    coal_props = {
        'Cd': 80.19, 'Hd': 4.83, 'Od': 9.76, 'Ad': 7.35,
        'Nd': 0.0, 'Sd': 0.0, 'Mt': 0.0, 'Hf': -0.6e6,
        'HHV_d': 30.0 # MJ/kg
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
        'L_reactor': inputs.get('L_reactor', 6.0),
        'D_reactor': inputs.get('D_reactor', 2.0),
        'HeatLossPercent': 1.0
    }
    
    geometry = {
        'L': inputs.get('L_reactor', 6.0),
        'D': inputs.get('D_reactor', 2.0)
    }
    
    # 2. Run System
    system = GasifierSystem(geometry, coal_props, op_conds)
    results, z = system.solve(N_cells=25)
    
    print(f"{'Cell':<5} | {'T (K)':<8} | {'H_gas (MW)':<12} | {'H_sol (MW)':<12} | {'Ws (kg/s)':<10} | {'Xc':<6} | {'Net_Q_Reac'}")
    print("-" * 90)
    
    # Track cell by cell
    for i in range(len(system.cells)):
        cell = system.cells[i]
        sol_x = results[i]
        
        # State at exit
        from model.state import StateVector
        current = StateVector.from_array(sol_x, P=op_conds['P'], z=z[i])
        
        # Enthalpy Fluxes
        H_gas = MaterialService.get_gas_enthalpy(current) / 1e6
        H_sol = MaterialService.get_solid_enthalpy(current, coal_props) / 1e6
        
        # Get Rates
        g_src = np.zeros(8)
        for s in cell.sources:
            g, _, _ = s.get_sources(cell.idx, cell.z, cell.dz)
            g_src += g
            
        phys = cell._calc_physics_props(current, coal_flow_kg_s * (coal_props['Cd']/100.0))
        r_het, r_homo, phi = cell._calc_rates(current, phys, g_src)
        
        Q_reac = (r_het['C+O2'] * 0.3935) - (r_het['C+H2O'] * 0.1313) - (r_het['C+CO2'] * 0.1725)
        Q_homo = (r_homo['CO_Ox'] * 0.283) + (r_homo['H2_Ox'] * 0.242) + (r_homo['CH4_Ox'] * 0.802)
        
        print(f"{i:<5} | {current.T:8.1f} | {H_gas:12.2f} | {H_sol:12.2f} | {current.solid_mass:10.4f} | {current.carbon_fraction:6.3f} | {Q_reac+Q_homo:12.2f} MW")

if __name__ == '__main__':
    audit_full_heat_balance()
