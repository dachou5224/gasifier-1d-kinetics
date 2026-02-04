
import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from model.gasifier_system import GasifierSystem
from model.chemistry import VALIDATION_CASES, COAL_DATABASE

def debug_case(case_name):
    print(f"\n{'='*40}")
    print(f"DEBUGGING CASE: {case_name}")
    print(f"{'='*40}")
    
    data = VALIDATION_CASES[case_name]
    inputs = data['inputs']
    coal_props = COAL_DATABASE[inputs['coal']]
    
    op_conds = {
        'coal_flow': inputs['FeedRate'] / 3600.0,
        'o2_flow': (inputs['FeedRate'] * inputs['Ratio_OC']) / 3600.0,
        'steam_flow': (inputs['FeedRate'] * inputs['Ratio_SC']) / 3600.0,
        'P': inputs['P'],
        'T_in': inputs['TIN'],
        'HeatLossPercent': inputs.get('HeatLossPercent', 0.0),
        'pilot_heat': 0.1e6 
    }
    
    geometry = {'L': 8.0, 'D': 2.6}
    system = GasifierSystem(geometry, coal_props, op_conds)
    
    # Run solver
    results, z = system.solve(N_cells=50)
    
    # Check first cell
    from model.state import StateVector
    from model.cell import Cell
    
    # Reconstruct Cell 0
    # Logic from GasifierSystem.solve
    W_wet = op_conds['coal_flow']
    Mt = coal_props.get('Mt', 0.0)
    W_dry = W_wet * (1 - Mt/100.0)
    W_h2o_liq = W_wet * (Mt/100.0)
    W_steam = op_conds.get('steam_flow', 0.0)
    
    molar_yields, Y_vm = system.pyrolysis.calc_yields(coal_props)
    W_vol = Y_vm * W_dry
    W_char = W_dry - W_vol - (coal_props['Ad']/100.0)*W_dry
    
    # Gas Inlet
    gas_in = np.zeros(8)
    gas_in[0] = op_conds['o2_flow'] / 0.031998  # O2
    gas_in[7] = (W_steam + W_h2o_liq) / 0.018015 # H2O
    
    # Cell 0 Instant Pyrolysis
    gas0 = gas_in + molar_yields * W_dry
    char0 = W_char
    inlet0 = StateVector(P=op_conds['P'], T=op_conds['T_in'], gas_moles=gas0, solid_mass=char0, carbon_fraction=1.0)
    
    print(f"Cell 0 Inlet: T={inlet0.T:.1f}, O2={inlet0.gas_moles[0]:.3f} mol/s, Char={inlet0.solid_mass:.3f} kg/s")
    
    # Exit state
    exit0 = StateVector.from_array(results[0], P=op_conds['P'], z=z[0])
    print(f"Cell 0 Exit:  T={exit0.T:.1f}, O2={exit0.gas_moles[0]:.3f} mol/s, Xc={exit0.carbon_fraction:.3f}")
    
    # Check energy balance for Cell 0
    # Cell(index, z, dz, V, A, inlet, kinetics, pyrolysis, coal_props, op_conds)
    from model.material import MaterialService
    H_in = MaterialService.get_total_enthalpy(inlet0, coal_props)
    H_out = MaterialService.get_total_enthalpy(exit0, coal_props)
    print(f"Enthalpy Balance: H_in={H_in/1e6:.2f} MW, H_out={H_out/1e6:.2f} MW")
    
if __name__ == "__main__":
    debug_case('Paper_Case_6')
    debug_case('Paper_Case_1')
    debug_case('Paper_Case_2')
