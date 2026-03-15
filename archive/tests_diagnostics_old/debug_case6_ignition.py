import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from model.gasifier_system import GasifierSystem
from model.chemistry import VALIDATION_CASES, COAL_DATABASE

def debug_ignition():
    case_name = "Paper_Case_6"
    print(f"Debugging {case_name} Ignition Propagation...")
    
    case_data = VALIDATION_CASES[case_name]
    inputs = case_data['inputs']
    coal_props = COAL_DATABASE[inputs['coal']]
    
    op_conds = {
        'coal_flow': inputs['FeedRate'] / 3600.0,
        'o2_flow': (inputs['FeedRate'] * inputs['Ratio_OC']) / 3600.0,
        'steam_flow': (inputs['FeedRate'] * inputs['Ratio_SC']) / 3600.0,
        'P': inputs['P'],
        'T_in': inputs['TIN'],
        'HeatLossPercent': 3.0,
        'SlurryConcentration': inputs.get('SlurryConcentration', 100.0),
        'particle_diameter': 100e-6,
        'Q_total_MW': 0.0
    }
    geometry = {'L': 8.0, 'D': 2.6} 

    system = GasifierSystem(geometry, coal_props, op_conds)
    system.solve(N_cells=50)
    
    for i in range(10):
        cell = system.cells[i]
        state = system.results[i]
        dz = cell.dz
        C_fed = system.W_dry * system.Cd_total
        Xc = 1.0 - (state.solid_mass * state.carbon_fraction / (C_fed + 1e-9))
        print(f"Cell {i}: Z={cell.z:.3f}, dz={dz:.3f}, T={state.T:.1f}K, Xc={Xc:.2%}")
        gas = state.gas_moles
        print(f"  Gas (mol/s): O2={gas[0]:.2f}, CH4={gas[1]:.2f}, CO={gas[2]:.2f}, H2={gas[5]:.2f}, H2O={gas[7]:.2f}, CO2={gas[3]:.2f}")

if __name__ == "__main__":
    debug_ignition()
