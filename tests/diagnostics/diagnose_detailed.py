
import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from model.gasifier_system import GasifierSystem
from model.cell import Cell
from model.state import StateVector

def run_detailed_diagnostics():
    print("[Diagnostic] Initializing Gasifier Simulation...")
    
    # 1. Properties (Paper Case 6 - Industrial)
    coal_props = {
        'Ashd': 6.0,  'VMd': 32.5,  'FCd': 61.5,
        'Cd': 78.4,   'Hd': 5.4,    'Od': 9.2,   'Nd': 1.4,   'Sd': 0.60,
        'Mt': 5.0,
        'HHV_d': 31.4,      # MJ/kg (Matches paper ~31400 kJ/kg)
        'dp': 100e-6        # 100 micron
    }
    
    op_conds = {
        'coal_flow': 11.57,  # kg/s
        'o2_flow': 12.16,    # kg/s
        'steam_flow': 2.21,  # kg/s
        'T_in': 400.0,       # K
        'P': 4.0e6,          # Pa
        'SlurryConcentration': 62.0,
        'pyrolysis_done': True # Instant Pyrolysis Mode
    }
    
    geometry = {
        'L': 8.0,
        'D': 2.8
    }
    
    # Scaling (Thesis Values)
    # comb=1.0, gas=0.0035 (to account for flux vs holdup difference likely)
    scaling = {
        'kinetics': {
            'comb': 1.0, 
            'gas': 1.0,
            'mixing': 0.4
        }
    }
    op_conds['pilot_heat'] = 5.0e6 # Added pilot heat for easier numerical ignition
    
    # 2. Run Simulation
    system = GasifierSystem(geometry, coal_props, op_conds, scaling)
    results, z_grid = system.solve(N_cells=50)
    
    print("[Diagnostic] Simulation Solved. Generating Detailed Report...")
    
    # 3. Harvest Data
    data = []
    
    # Prepare Inlet for Cell 0
    current_inlet = system._initialize_inlet()
    
    # 3.1 Prepend Inlet State (z=0, Before Pyrolysis)
    # Total Carbon in Coal
    W_dry = op_conds['coal_flow'] * (1 - coal_props.get('Mt', 0.0)/100.0)
    C_fed = W_dry * (coal_props.get('Cd', 0.0)/100.0)
    
    inlet_snap = {
        'Cell': -1,
        'z (m)': 0.0,
        'T (K)': op_conds['T_in'],
        'P (Pa)': op_conds['P'],
        'SolidMass (kg/s)': W_dry, # Before devolatilization
        'CarbonFrac': coal_props.get('Cd', 0.0)/100.0,
        'Conversion_Coal (%)': 0.0,
        'Phi': 1.0, 
        'S_total': 0.0,
        'd_p (um)': coal_props.get('dp', 100e-6) * 1e6,
        'v_g (m/s)': 0.0,
        'Sh': 2.0,
        'rho_gas': 0.0
    }
    specs = ['O2', 'CH4', 'CO', 'CO2', 'H2S', 'H2', 'N2', 'H2O']
    for idx, sp in enumerate(specs):
        inlet_snap[f'F_{sp}'] = current_inlet.gas_moles[idx]
    # Rates/Src as zero for z=0
    for rxn in ['C+O2', 'C+H2O', 'C+CO2', 'C+H2']: inlet_snap[f'Rate_Het_{rxn}'] = 0.0
    for rxn in ['CO_Ox', 'H2_Ox', 'WGS', 'RWGS', 'CH4_Ox', 'MSR']: inlet_snap[f'Rate_Homo_{rxn}'] = 0.0
    for sp in specs: inlet_snap[f'Src_Pyro_{sp}'] = 0.0
    data.append(inlet_snap)
    
    # Re-calculate physics for each cell
    for i, res in enumerate(results):
        # Determine Inlet for this cell
        if i == 0:
            inlet = current_inlet
        else:
            inlet = results[i-1]
            
        # Re-create Cell instance to access physics methods
        # Need Geometry
        dz = system.z_positions[1] - system.z_positions[0] # Uniform Grid
        V_cell = (np.pi * (geometry['D']/2)**2) * dz
        A = np.pi * (geometry['D']/2)**2
        
        cell_ops = op_conds.copy()
        cell_ops['L_reactor'] = geometry['L']
        cell_ops['pyrolysis_done'] = True
        
        # Q_source (Evap) for Cell 0
        if i == 0:
            cell_ops['Q_source_term'] = system.evap_heat_load
        else:
            cell_ops['Q_source_term'] = 0.0
            
        cell = Cell(i, z_grid[i], dz, V_cell, A, inlet, system.kinetics, system.pyrolysis, coal_props, cell_ops)
        cell.coal_flow_dry = system.W_dry
        cell.Cd_total = system.Cd_total
        cell.char_Xc0 = system.char_Xc0
        
        # Get Snapshot
        res_state = StateVector.from_array(res, P=op_conds['P'], z=z_grid[i])
        snap = cell.get_snapshot(res_state)
        
        # Flatten Dict
        C_sol = snap['SolidMass'] * snap['CarbonFrac']
        X_coal = (C_fed - C_sol) / (C_fed + 1e-9) * 100.0
        
        row = {
            'Cell': i,
            'z (m)': snap['z'],
            'T (K)': snap['T'],
            'P (Pa)': snap['P'],
            'SolidMass (kg/s)': snap['SolidMass'],
            'CarbonFrac': snap['CarbonFrac'],
            'Conversion_Coal (%)': X_coal,
            'Phi': snap['Phi'],
            'S_total': snap['S_total'],
            'd_p (um)': snap['d_p'] * 1e6,
            'v_g (m/s)': snap['v_g'],
            'Re_p': snap['Re_p'],
            'Sh': snap['Sh'],
            'rho_gas': snap.get('rho_gas', 0.0)
        }
        
        # Species
        specs = ['O2', 'CH4', 'CO', 'CO2', 'H2S', 'H2', 'N2', 'H2O']
        for idx, sp in enumerate(specs):
            row[f'F_{sp}'] = snap['GasMoles'][idx]
            
        # Rates (Het)
        for rxn, rate in snap['Rates_Het'].items():
            row[f'Rate_Het_{rxn}'] = rate
            
        # Rates (Homo)
        for rxn, rate in snap['Rates_Homo'].items():
            row[f'Rate_Homo_{rxn}'] = rate
            
        # Pyrolysis Src
        for idx, sp in enumerate(specs):
            row[f'Src_Pyro_{sp}'] = snap['Src_Pyrolysis'][idx]
            
        data.append(row)
        
    # 4. Export
    df = pd.DataFrame(data)
    csv_path = 'diagnostic_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"[Diagnostic] Data saved to {csv_path}")
    
    # 5. Print Summary Table (Cells 0, 10, 25, 49)
    print("\n--- Summary Report ---")
    cols = ['z (m)', 'T (K)', 'v_g (m/s)', 'rho_gas', 'Sh', 'Conversion_Coal (%)', 'd_p (um)']
    indices = [0, 1, 5, 25, 49]
    print(df.loc[indices, cols].to_string())

if __name__ == "__main__":
    run_detailed_diagnostics()
