import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from model.gasifier_system import GasifierSystem
from model.chemistry import VALIDATION_CASES, COAL_DATABASE

def run_and_report():
    cases_to_run = ["Paper_Case_6", "Paper_Case_1", "Paper_Case_2"]
    
    for case_name in cases_to_run:
        print(f"\n{'='*80}")
        print(f"FULL REPORT: {case_name}")
        print(f"{'='*80}")
        
        # 1. Setup
        case_data = VALIDATION_CASES[case_name]
        inputs = case_data['inputs']
        coal_props = COAL_DATABASE[inputs['coal']]
        
        op_conds = {
            'coal_flow': inputs['FeedRate'] / 3600.0,
            'o2_flow': (inputs['FeedRate'] * inputs['Ratio_OC']) / 3600.0,
            'steam_flow': (inputs['FeedRate'] * inputs['Ratio_SC']) / 3600.0,
            'P': inputs['P'],
            'T_in': inputs['TIN'],
            'HeatLossPercent': 3.0, # Standard assumption
            'SlurryConcentration': inputs.get('SlurryConcentration', 100.0),
            'particle_diameter': 100e-6
        }
        
        # Reactor Geometry (Standard)
        geometry = {'L': 8.0, 'D': 2.6} 
        
        # 2. Run Simulation
        system = GasifierSystem(geometry, coal_props, op_conds)
        try:
            results = system.solve(N_cells=50)
            
            # 3. Extract Profile Data
            # Show first 10 cells (Critical Zone) + sparse downstream
            indices = list(range(0, 10)) + list(range(10, 50, 5)) + [49]
            indices = sorted(list(set(indices))) # unique sorted
            
            data = []
            for i in indices:
                cell = system.cells[i]
                state = system.results[i]
                y = state.gas_fractions
                
                # Calculate Xc manually since it's missing from snapshot dict
                C_local = state.solid_mass * state.carbon_fraction
                C_fed = system.W_dry * system.Cd_total
                Xc = 1.0 - (C_local / (C_fed + 1e-9))
                
                # Z, T, Xc, O2, CO, H2, CO2, H2O, CH4
                row = [
                    f"{cell.z:.2f}",
                    f"{state.T:.1f}",
                    f"{Xc*100:.1f}%",
                ]
                # Gas Fracs (O2, CH4, CO, CO2, H2S, H2, N2, H2O)
                # Show key species: O2(0), CO(2), CO2(3), H2(5), H2O(7), CH4(1)
                row.append(f"{y[0]*100:.2f}") # O2
                row.append(f"{y[2]*100:.1f}") # CO
                row.append(f"{y[5]*100:.1f}") # H2
                row.append(f"{y[3]*100:.1f}") # CO2
                row.append(f"{y[7]*100:.1f}") # H2O
                row.append(f"{y[1]*100:.2f}") # CH4
                
                data.append(row)
                
                data.append(row)
                
            headers = ["Z(m)", "T(K)", "Xc(%)", "O2(%)", "CO(%)", "H2(%)", "CO2(%)", "H2O(%)", "CH4(%)"]
            df = pd.DataFrame(data, columns=headers)
            print(df.to_string(index=False))
            
            # 4. Result Summary
            print("\n[Outlet Summary]")
            out = system.results[-1]
            y_out = out.gas_fractions
            print(f"Exit Temperature: {out.T - 273.15:.1f} C")
            print(f"Syngas (H2+CO): {(y_out[2]+y_out[5])*100:.1f}% (Exp: {case_data['exp_results'].get('CO',0) + case_data['exp_results'].get('H2',0):.1f}%)")
            
        except Exception as e:
            print(f"Simulation Crashed: {e}")

if __name__ == "__main__":
    run_and_report()
