import sys
import os
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from model.gasifier_system import GasifierSystem
from model.state import StateVector
from model.chemistry import VALIDATION_CASES, COAL_DATABASE

def calculate_dry_mole_fraction(gas_moles, total_gas_moles):
    """Calculate dry basis volume percentage."""
    # Species: [O2, CH4, CO, CO2, H2S, H2, N2, H2O]
    # Indices 0-6 are dry (excluding H2O at index 7)
    
    F_dry_total = sum(gas_moles[:7]) + 1e-9
    
    Y = {}
    species_names = ['O2', 'CH4', 'CO', 'CO2', 'H2S', 'H2', 'N2']
    for i, name in enumerate(species_names):
        Y[name] = (gas_moles[i] / F_dry_total) * 100.0
        
    return Y

def run_verification():
    print(f"{'='*80}")
    print(f"1D Kinetic Model - Multi-Case Verification Suite")
    print(f"{'='*80}\n")
    
    summary_data = []
    case_data_store = {}
    
    for case_name, data in VALIDATION_CASES.items():
        print(f"Running Case: {case_name}...")
        
        # 1. Setup
        inputs = data['inputs']
        expected = data['expected']
        
        coal_key = inputs['coal']
        coal_props = COAL_DATABASE.get(coal_key)
        if not coal_props:
            print(f"  [Error] Coal '{coal_key}' not found!")
            continue
            
        op_conds = {
            'coal_flow': inputs['FeedRate'] / 3600.0,
            'o2_flow': (inputs['FeedRate'] * inputs['Ratio_OC']) / 3600.0,
            'steam_flow': (inputs['FeedRate'] * inputs['Ratio_SC']) / 3600.0,
            'P': inputs['P'],
            'T_in': inputs['TIN'],
            'HeatLossPercent': inputs.get('HeatLossPercent', 0.0),
            'SlurryConcentration': inputs.get('SlurryConcentration', 100.0),
            'pilot_heat': 5.0e6 # Standardized for industrial burner ignition
        }
        
        # Geometry (Industrial Scale)
        geometry = {'L': 8.0, 'D': 2.6} 
        run_N = 50
            
        # Scaling
        scaling = {'kinetics': {'comb': 1.0, 'gas': 1.0, 'mixing': 1.0}}
            
        # 2. Solver
        system = GasifierSystem(geometry, coal_props, op_conds, scaling)
        try:
            results, z_grid = system.solve(N_cells=run_N)
            
            # 3. Analyze Results
            last_state_arr = results[-1]
            last_state = StateVector.from_array(last_state_arr, P=op_conds['P'], z=z_grid[-1])
            T_out = last_state.T
            
            Y_calc = calculate_dry_mole_fraction(last_state.gas_moles, last_state.total_gas_moles)
            
            # Calc Conversion
            C_in_kg = op_conds['coal_flow'] * (coal_props['Cd']/100.0)
            C_out_kg = last_state.solid_mass * last_state.carbon_fraction
            Xc_calc = (1.0 - C_out_kg / (C_in_kg + 1e-9)) * 100.0
            
            # Compare
            row = {'Case': case_name}
            
            # CO
            row['CO_Exp'] = expected.get('YCO', np.nan)
            row['CO_Sim'] = Y_calc['CO']
            row['CO_Err'] = row['CO_Sim'] - row['CO_Exp']
            
            # H2
            row['H2_Exp'] = expected.get('YH2', np.nan)
            row['H2_Sim'] = Y_calc['H2']
            row['H2_Err'] = row['H2_Sim'] - row['H2_Exp']
            
            # CO2
            row['CO2_Exp'] = expected.get('YCO2', np.nan)
            row['CO2_Sim'] = Y_calc['CO2']
            row['CO2_Err'] = row['CO2_Sim'] - row['CO2_Exp']
            
            # Xc
            exp_Xc = expected.get('Xc', np.nan)
            row['Xc_Exp'] = exp_Xc
            row['Xc_Sim'] = Xc_calc
            row['Xc_Err'] = Xc_calc - exp_Xc if not np.isnan(exp_Xc) else np.nan
            
            # Temperature Check
            exp_T = expected.get('TOUT_C', np.nan)
            row['T_Exp'] = exp_T
            row['T_Sim'] = T_out - 273.15 # Convert K to C
            row['T_Err'] = row['T_Sim'] - exp_T if not np.isnan(exp_T) else np.nan
            
            # Store full profile for plotting
            # Calculate CO vol% for the profile
            total_moles = np.sum(results[:, 0:8], axis=1)
            co_vol_pct = (results[:, 2] / (total_moles + 1e-9)) * 100.0
            
            # Axial Carbon Conversion (relative to coal carbon fed)
            W_dry_kg_s = (inputs['FeedRate']/3600.0) * (1 - coal_props.get('Mt', 0.0)/100.0)
            C_fed = W_dry_kg_s * (coal_props['Cd']/100.0)
            C_sol_prof = results[:, 8] * results[:, 9]
            Xc_prof = (C_fed - C_sol_prof) / (C_fed + 1e-9) * 100.0
            
            # Prepend Absolute Inlet (z=0, X=0, T=Tin)
            z_full = np.concatenate(([0.0], z_grid))
            T_full = np.concatenate(([op_conds['T_in'] - 273.15], results[:, 10] - 273.15))
            Xc_full = np.concatenate(([0.0], Xc_prof))
            CO_full = np.concatenate(([0.0], co_vol_pct))
            
            # Harvest O2 and dp profiles
            o2_prof = results[:, 0]
            dp_prof = []
            for i in range(run_N):
                # Use persisted cells and local state
                st = StateVector.from_array(results[i], P=op_conds['P'], z=z_grid[i])
                s = system.cells[i].get_snapshot(st)
                dp_prof.append(s['d_p'] * 1e6) # um
                
            case_data_store[case_name] = {
                'z': z_full,
                'T_C': T_full,
                'Xc': Xc_full,
                'CO_vol': CO_full,
                'O2_mol': np.concatenate(([op_conds['o2_flow']/32.0*1000.0], o2_prof)), # Approximate inlet
                'dp_um': np.concatenate(([coal_props.get('dp', 100e-6)*1e6], np.array(dp_prof)))
            }
            
            summary_data.append(row)
            print(f"  -> Done. T={row['T_Sim']:.1f}C, Xc={Xc_calc:.1f}%, CO={Y_calc['CO']:.1f}%")
            
        except Exception as e:
            print(f"  [FAILED] {str(e)}")
            import traceback
            traceback.print_exc()

    # 4. Report
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY REPORT")
    print("="*80)
    
    df = pd.DataFrame(summary_data)
    cols = ['Case', 'T_Exp', 'T_Sim', 'T_Err', 'Xc_Sim', 'CO_Sim', 'CO2_Sim', 'H2_Sim']
    
    # Format for clean printing
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', lambda x: '%.1f' % x)
    
    print(df[cols])
    
    # Validation Criteria
    print("\n[Pass/Fail Check]")
    all_pass = True
    for idx, row in df.iterrows():
        # Check Error magnitude
        err_xc = abs(row['Xc_Err']) if not np.isnan(row['Xc_Err']) else 0
        err_co = abs(row['CO_Err'])
        
        status = "PASS"
        if err_xc > 5.0 or err_co > 5.0: # 5% tolerance
            status = "WARN"
            all_pass = False
        print(f"  {row['Case']}: {status} (Xc Err: {err_xc:.1f}, CO Err: {err_co:.1f})")

    # ==============================================================================
    # 5. Plotting
    # ==============================================================================
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(20, 10))
        
        # Subplot 1: Temperature
        plt.subplot(2, 2, 1)
        for case, data in case_data_store.items():
            plt.plot(data['z'], data['T_C'], label=case, linewidth=2.0)
        plt.xlabel('Axial Position (m)')
        plt.ylabel('Temperature (°C)')
        plt.title('Axial Temperature Profile')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(fontsize=8)
        
        # Subplot 2: Conversion
        plt.subplot(2, 2, 2)
        for case, data in case_data_store.items():
            plt.plot(data['z'], data['Xc'], label=case, linewidth=2.0, linestyle='--')
        plt.xlabel('Axial Position (m)')
        plt.ylabel('Percentage (%)')
        plt.title('Axial Carbon Conversion')
        plt.ylim(0, 105)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(fontsize=8)

        # Subplot 3: CO Concentration
        plt.subplot(2, 2, 3)
        for case, data in case_data_store.items():
            plt.plot(data['z'], data['CO_vol'], label=case, linewidth=2.0)
        plt.xlabel('Axial Position (m)')
        plt.ylabel('Percentage (%)')
        plt.title('Axial CO Volume Fraction')
        plt.ylim(0, 70)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(fontsize=8)
        
        # Subplot 4: Particle Diameter vs O2
        plt.subplot(2, 2, 4)
        for case, data in case_data_store.items():
            # Plot Dp on left y
            ax1 = plt.gca()
            p, = ax1.plot(data['z'], data['dp_um'], label=f"{case} dp", linewidth=1.5)
            # Find a way to color O2 similarly
            color = p.get_color()
            # Plot O2 on right y (This is tricky in a loop)
            # Maybe just plot them as separate lines for now
            plt.plot(data['z'], data['O2_mol'], label=f"{case} O2", linewidth=1.0, linestyle=':', color=color)
            
        plt.xlabel('Axial Position (m)')
        plt.ylabel('dp (um) / O2 (mol/s)')
        plt.title('Dp (solid) & O2 (dotted)')
        plt.yscale('log') # Better for seeing O2 decay
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(fontsize=8, ncol=2)
        
        output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../docs/verification_profiles.png'))
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"\n[Plot Saved]: {output_path}")
        
    except ImportError:
        print("[Warning] Matplotlib not found - install to see plots.")

if __name__ == "__main__":
    run_verification()
