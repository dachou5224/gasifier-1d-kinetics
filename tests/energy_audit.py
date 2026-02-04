import sys
import os
import numpy as np
import pandas as pd
import json

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from model.gasifier_system import GasifierSystem
from model.state import StateVector
from model.chemistry import VALIDATION_CASES, COAL_DATABASE
from model.cell import Cell
from model.material import MaterialService, SPECIES_NAMES

def run_diagnostics():
    print("[Diagnostic] Initializing Analysis...")
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
    scaling = {'kinetics': {'comb': 1.0, 'gas': 1.0}}
    
    system = GasifierSystem(geometry, coal_props, op_conds, scaling)
    results, z_grid = system.solve(N_cells=50)
    
    print("[Diagnostic] Simulation Solved. Harvesting Energy Data...")
    
    harvest = []
    current_inlet = system._initialize_inlet()
    
    for i, state_arr in enumerate(results):
        state = StateVector.from_array(state_arr, P=op_conds['P'], z=z_grid[i])
        z = z_grid[i]
        dz = geometry['L'] / 50.0
        V_cell = (np.pi * (geometry['D']/2)**2) * dz
        A = np.pi * (geometry['D']/2)**2
        
        # Prepare cell to access its physics methods
        cell_ops = op_conds.copy()
        cell_ops['pyrolysis_done'] = True
        if i == 0:
            cell_ops['Q_source_term'] = system.evap_heat_load + op_conds['pilot_heat']
        else:
            cell_ops['Q_source_term'] = 0.0
            
        cell = Cell(i, z, dz, V_cell, A, current_inlet, system.kinetics, system.pyrolysis, coal_props, cell_ops)
        info = cell.get_snapshot(state)
        
        # Energy Breakdown
        r_het = info['Rates_Het']
        r_homo = info['Rates_Homo']
        
        # Approx Enthalpies (J/mol)
        dH_C_O2 = -110500.0 # C -> CO
        dH_C_H2O = 131300.0 # C -> CO+H2
        dH_C_CO2 = 172500.0 # C -> 2CO
        
        # Terms in Watts (J/s)
        Q_Oxidation = r_het['C+O2'] * abs(dH_C_O2)
        Q_Reduction = r_het['C+H2O'] * dH_C_H2O + r_het['C+CO2'] * dH_C_CO2
        
        # Homogeneous (approximate)
        Q_Homo_Ox = (r_homo['CO_Ox'] * 283000.0 + r_homo['H2_Ox'] * 241800.0 + r_homo['CH4_Ox'] * 802300.0)
        Q_WGS = r_homo['WGS'] * 35000.0 # Exothermic
        
        H_gas = MaterialService.get_gas_enthalpy(state)
        H_solid = MaterialService.get_solid_enthalpy(state, coal_props)
        
        Q_loss = (inputs['HeatLossPercent']/100.0) * system.op_conds['Q_total_MW'] * 1e6 * (dz / geometry['L'])
        
        snap = {
            'z': z,
            'T_C': state.T - 273.15,
            'Xc': (1.0 - (state.solid_mass*state.carbon_fraction)/(current_inlet.solid_mass*current_inlet.carbon_fraction + 1e-9))*100.0,
            'Rate_Het_O2': r_het['C+O2'],
            'Rate_Het_H2O': r_het['C+H2O'],
            'Q_Ox_MW': Q_Oxidation / 1e6,
            'Q_Red_MW': Q_Reduction / 1e6,
            'Q_Homo_Ox_MW': Q_Homo_Ox / 1e6,
            'Q_loss_MW': Q_loss / 1e6,
            'H_total_MW': (H_gas + H_solid) / 1e6,
            'F_O2': state.gas_moles[0]
        }
        harvest.append(snap)
        current_inlet = state

    df = pd.DataFrame(harvest)
    df.to_csv('energy_audit.csv', index=False)
    print("[Diagnostic] Energy Audit saved to energy_audit.csv")
    
    # Analyze Cell 0
    print("\n--- Cell 0 Audit ---")
    c0 = df.iloc[0]
    print(f"Temperature: {c0['T_C']:.1f} C")
    print(f"Carbon Conversion: {c0['Xc']:.2f} %")
    print(f"Heat Release (Oxid): {c0['Q_Ox_MW']:.2f} MW")
    print(f"Heat Sink (Reduc): {c0['Q_Red_MW']:.2f} MW")
    print(f"Residual O2: {c0['F_O2']:.4f} mol/s")

if __name__ == "__main__":
    run_diagnostics()
