import sys
import os
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from model.gasifier_system import GasifierSystem
from model.state import StateVector
from model.chemistry import VALIDATION_CASES, COAL_DATABASE
from model.cell import Cell
from model.material import MaterialService

def run_energy_audit():
    print("[Audit] Detailed Homogeneous Energy Breakdown...")
    case_name = 'Paper_Case_6'
    data = VALIDATION_CASES[case_name]
    inputs = data['inputs']
    coal_props = COAL_DATABASE[inputs['coal']]
    
    op_conds = {
        'coal_flow': inputs['FeedRate'] / 3600.0,
        'o2_flow': (inputs['FeedRate'] * inputs['Ratio_OC']) / 3600.0,
        'steam_flow': (inputs['FeedRate'] * inputs['Ratio_SC']) / 3600.0,
        'T_in': inputs['TIN'],
        'P': inputs['P'],
        'HeatLossPercent': inputs.get('HeatLossPercent', 3.0),
        'SlurryConcentration': inputs.get('SlurryConcentration', 100.0),
        'pilot_heat': 5.0e6
    }
    
    geometry = {'L': 8.0, 'D': 2.6}
    system = GasifierSystem(geometry, coal_props, op_conds)
    results, z_grid = system.solve(N_cells=50)
    
    harvest = []
    current_inlet = system._initialize_inlet()
    
    for i, state_arr in enumerate(results):
        state = StateVector.from_array(state_arr, P=op_conds['P'], z=z_grid[i])
        z = z_grid[i]
        dz = 8.0 / 50.0
        A = np.pi * (2.6/2.0)**2
        V = A * dz
        
        cell_ops = op_conds.copy()
        cell_ops['pyrolysis_done'] = True
        if i == 0: cell_ops['Q_source_term'] = system.evap_heat_load + 5.0e6
        else: cell_ops['Q_source_term'] = 0.0
        
        cell = Cell(i, z, dz, V, A, current_inlet, system.kinetics, system.pyrolysis, coal_props, cell_ops)
        info = cell.get_snapshot(state)
        r_homo = info['Rates_Homo']
        
        # Exact Enthalpies at local T (J/mol)
        # MSR: CH4 + H2O -> CO + 3H2 (Endo)
        v_msr_p = StateVector(gas_moles=np.array([0,0,1,0,0,3,0,0]), solid_mass=0, carbon_fraction=0, T=state.T, P=state.P, z=0)
        v_msr_r = StateVector(gas_moles=np.array([0,1,0,0,0,0,0,1]), solid_mass=0, carbon_fraction=0, T=state.T, P=state.P, z=0)
        dH_MSR = MaterialService.get_gas_enthalpy(v_msr_p) - MaterialService.get_gas_enthalpy(v_msr_r)
        
        # RWGS: CO2 + H2 -> CO + H2O (Endo)
        v_rw_p = StateVector(gas_moles=np.array([0,0,1,0,0,0,0,1]), solid_mass=0, carbon_fraction=0, T=state.T, P=state.P, z=0)
        v_rw_r = StateVector(gas_moles=np.array([0,0,0,1,0,1,0,0]), solid_mass=0, carbon_fraction=0, T=state.T, P=state.P, z=0)
        dH_RWGS = MaterialService.get_gas_enthalpy(v_rw_p) - MaterialService.get_gas_enthalpy(v_rw_r)

        Q_MSR_W = r_homo['MSR'] * dH_MSR
        Q_RWGS_W = r_homo['RWGS'] * dH_RWGS
        Q_WGS_W = r_homo['WGS'] * (-dH_RWGS) # WGS is reverse of RWGS

        snap = {
            'z': z,
            'T_C': state.T - 273.15,
            'Xc': state.carbon_fraction,
            'Rate_MSR': r_homo['MSR'],
            'Rate_RWGS': r_homo['RWGS'],
            'Rate_WGS': r_homo['WGS'],
            'Q_MSR_kW': Q_MSR_W / 1000.0,
            'Q_RWGS_kW': Q_RWGS_W / 1000.0,
            'Q_WGS_kW': Q_WGS_W / 1000.0,
            'F_CH4': state.gas_moles[1],
            'F_CO2': state.gas_moles[3],
            'F_CO': state.gas_moles[2]
        }
        harvest.append(snap)
        current_inlet = state

    df = pd.DataFrame(harvest)
    print("\n--- Homo Energy Audit (Cell 0 to 5) ---")
    print(df[['z', 'T_C', 'Rate_MSR', 'Q_MSR_kW', 'Q_RWGS_kW', 'Q_WGS_kW', 'F_CH4']].iloc[0:6])
    
    print("\n[Conclusion] If Q_MSR and Q_RWGS are near zero after Cell 0, CH4/CO2 are exhausted or rates are too high.")

if __name__ == "__main__":
    run_energy_audit()
