import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from model.gasifier_system import GasifierSystem
from model.state import StateVector
from model.cell import Cell
from model.chemistry import VALIDATION_CASES, COAL_DATABASE

def diagnose_cell0():
    print("Diagnosing Cell 0 Energy Balance Landscape...")
    
    # Setup Case 6 (Slurry)
    case_data = VALIDATION_CASES["Paper_Case_6"]
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
    }
    geometry = {'L': 8.0, 'D': 2.6} 
    system = GasifierSystem(geometry, coal_props, op_conds)
    
    # Initialize Inlet
    # Note: solve() modifies system state, we just need _initialize_inlet equivalent logic
    # But since I refactored _init_inlet to likely be private or simple...
    # Let's use the public method if avail or recreate. system._initialize_inlet() is avail.
    
    inlet = system._initialize_inlet()
    
    # Setup Cell 0
    L = 8.0
    N = 50
    dz = L / N
    D = 2.6
    A = np.pi * (D/2)**2
    V_cell = A * dz
    
    cell_ops = op_conds.copy()
    cell_ops['L_reactor'] = L
    cell_ops['Q_source_term'] = system.evap_heat_load # Negative
    
    print(f"Evap Heat Load: {system.evap_heat_load / 1e6:.2f} MW")
    
    cell = Cell(0, dz/2, dz, V_cell, A, inlet, system.kinetics, system.pyrolysis, coal_props, cell_ops)
    cell.coal_flow_dry = system.W_dry
    cell.Cd_total = system.Cd_total
    cell.char_Xc0 = system.char_Xc0
    
    # Sweep Temperature and Check Residuals
    T_range = np.linspace(300, 3000, 50)
    res_E_list = []
    Q_rxn_list = []
    
    # We need to solve/guess other variables (gas composition) at each T to get a meaningful "semi-equilibrium" rate?
    # Or just freeze composition at inlet and see reaction heat POTENTIAL?
    # Kinetics depend on Concentration. If we freeze inlet comp, we see initial rate.
    # The actual solver finds a state where Conservation AND Energy are satisfied.
    # A true 1D scan is hard because 11 defined variables.
    
    # Simplified approach: take the inlet composition, assume reaction proceeds for time 'dt' (residence time),
    # calculate Heat Release vs Heat Loss.
    
    # Residence Time
    # Density approx at T
    # rho = P / (R_gas * T) * MW
    # u = F / (rho * A). dt = dz / u.
    
    print(f"\nScanning T (K) | Res_E (MW?) | Q_rxn (MW) | Q_source (MW)")
    print("-" * 60)
    
    for T in T_range:
        # Create a dummy state at T, with inlet composition
        # (This is wrong because composition changes, but shows instantaneous tendency)
        # Better: Solve the *Species* balance at fixed T, then check Energy Residual.
        
        # 1. Fix T. Solve 10 variables (O2...Xc).
        # We can use the cell.residuals function but override the Energy residual or T variable.
        
        # Or, just evaluate residuals at x = inlet, but replace T.
        # This shows "Initial Driving Force".
        
        x_test = inlet.to_array()
        x_test[10] = T
        
        # We need to manually call the components used in residuals to get Q values debugging
        # But cell.residuals returns the final error vector.
        # We want the 'res_E' component value (index 10 in standard array? No, index 10 is T var).
        # In cell.py, residuals return 11-15 values.
        # res_E is usually index 10?
        # Let's check cell.py residuals return structure.
        # It ends with: res_Ws, res_Xc, res_E, res_atoms...
        
        res = cell.residuals(x_test)
        
        # Unpack res_E (Index 10)
        # Wait, Step 239 showed:
        # [res_O2 ... res_H2O (8), res_Ws(9), res_Xc(10), res_E(11), atoms(12..15)]
        # So Res_E is index 10 or 11.
        # 0-7 gases. 8 Ws. 9 Xc. 10 E.
        
        r_E = res[10] # Scaled by 1e6 usually
        
        res_E_list.append(r_E)
        
        # Also estimating Q_rxn physically
        # Rate calculation inside cell is not exposed.
        # But r_E ~ (H_out - H_in - Q_source - Q_rxn).
        # If r_E is positive, it means H_out > ... (Needs cooling? or T too high?)
        # If T is high, H_out is high. r_E should be positive (unless reaction balances it).
        
        if T == 3000.0:
            print(f"\n[DEBUG 3000K] Checking Rates...")
            x_test_3000 = x_test.copy()
            # Construct state manually to check concentrations
            # Note: cell.kinetics.calculate_rates takes (state, cell_geom, ...)
            
            # Need to create StateVector 'state_curr'
            state_curr = StateVector.from_array(x_test_3000, P=inlet.P, z=dz/2)
            
            # Print Concentrations
            vol_fracs = state_curr.gas_fractions
            print(f"  Gas Fractions: O2={vol_fracs[0]:.4f}, CH4={vol_fracs[1]:.4f}, CO={vol_fracs[2]:.4f}")
            
            # Calc Rates
            r_homo = system.kinetics.calc_homogeneous_rates(state_curr, V_cell)
            print(f"  Homo Rates (mol/s):")
            for k, v in r_homo.items():
                print(f"    {k}: {v:.4e}")
            
            # Heterogeneous requires more complex inputs (d_p, S_total)
            # Skip for now as gas phase is primary for ignition stability

            
        if T % 500 == 0 or T==300:
             print(f"{T:.0f}   | {r_E:.4e}")

    plt.figure()
    plt.plot(T_range, res_E_list)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Energy Residual (Scaled)')
    plt.title('Cell 0 Energy Residual Landscape (Fixed Inlet Comp)')
    plt.grid(True)
    plt.savefig('docs/cell0_diagnostic.png')
    print("\nSaved plot to docs/cell0_diagnostic.png")

if __name__ == "__main__":
    diagnose_cell0()
