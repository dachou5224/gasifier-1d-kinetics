import sys
import os
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from model.gasifier_system import GasifierSystem
from model.state import StateVector
from model.chemistry import VALIDATION_CASES, COAL_DATABASE
from model.material import SPECIES_NAMES

def get_atoms_from_state(state, coal_props):
    """
    Calculate total moles of atoms (C, H, O, N) in a given state (Gas + Solid).
    """
    # 1. Gas Phase
    # [O2, CH4, CO, CO2, H2S, H2, N2, H2O]
    gas_c = state.gas_moles[1] + state.gas_moles[2] + state.gas_moles[3]
    gas_h = 4*state.gas_moles[1] + 2*state.gas_moles[4] + 2*state.gas_moles[5] + 2*state.gas_moles[7]
    gas_o = 2*state.gas_moles[0] + state.gas_moles[2] + 2*state.gas_moles[3] + state.gas_moles[7]
    gas_n = 2*state.gas_moles[6]
    
    # 2. Solid Phase
    # Solid is Char. Char contains C and maybe some residual H/O/N/S if not fully pyrolyzed?
    # In this model, Char is modeled as pure Carbon + Ash usually, or Fixed Carbon.
    # state.carbon_fraction tracks the Carbon fraction.
    # We implicitly assume the 'solid_mass' is Char + Ash.
    # C_mass = solid_mass * carbon_fraction.
    # The non-carbon part of Char is Ash. 
    # Wait, the model's Pyrolysis step separates Volatiles (CHONS) from Char(C_fixed).
    # So Char is predominantly Carbon.
    
    c_solid_moles = (state.solid_mass * state.carbon_fraction) / 0.012011
    
    return {
        'C': gas_c + c_solid_moles,
        'H': gas_h,
        'O': gas_o,
        'N': gas_n
    }

def get_atoms_from_feed(op_conds, coal_props):
    """
    Calculate effective atom feed rate (mol/s) from Coal + Oxidant + Steam.
    """
    # 1. Coal Atoms
    # coal_flow is wet or dry? 
    # op_conds['coal_flow'] is usually As Received or Wet if it includes moisture.
    # The System class handles this breakdown. 
    # Let's rely on the System's inlet initialization logic for consistency?
    # Or recalculate from first principles to verify the System inlet logic too.
    
    W_wet = op_conds['coal_flow'] # kg/s
    Mt = coal_props.get('Mt', 0.0)
    W_dry = W_wet * (1 - Mt/100.0)
    
    # Coal Ultimate Analysis (Dry Basis, wt%)
    Cd = coal_props.get('Cd', 0.0)
    Hd = coal_props.get('Hd', 0.0)
    Od = coal_props.get('Od', 0.0)
    Nd = coal_props.get('Nd', 0.0)
    
    mol_C = (W_dry * Cd/100.0) / 0.012011
    mol_H = (W_dry * Hd/100.0) / 0.001008
    mol_O = (W_dry * Od/100.0) / 0.015999
    mol_N = (W_dry * Nd/100.0) / 0.014007
    
    # Note: Coal Moisture (Mt) is usually handled as H2O input.
    # W_moist = W_wet * (Mt/100.0)
    # The model adds this to input gas.
    # Plus Slurry Water...
    
    # 2. Slurry Water
    slurry_conc = op_conds.get('SlurryConcentration', 100.0)
    if slurry_conc < 100.0:
        W_slurry_tot = W_dry / (slurry_conc/100.0)
        W_slurry_h2o = W_slurry_tot - W_dry
    else:
        W_slurry_h2o = 0.0
        
    W_coal_moist = W_wet * (Mt/100.0)
    W_total_h2o_liq = W_coal_moist + W_slurry_h2o
    
    # 3. Steam Feed
    W_steam = op_conds.get('steam_flow', 0.0)
    
    W_h2o_all = W_total_h2o_liq + W_steam
    mol_h2o = W_h2o_all / 0.018015
    
    # Add H2O atoms
    mol_H += 2 * mol_h2o
    mol_O += 1 * mol_h2o
    
    # 4. Oxidant
    o2_flow = op_conds.get('o2_flow', 0.0) # kg/s
    mol_o2 = o2_flow / 0.031998
    mol_O += 2 * mol_o2
    
    n2_flow = op_conds.get('n2_flow', 0.0)
    mol_n2 = n2_flow / 0.028013
    mol_N += 2 * mol_n2
    
    return {'C': mol_C, 'H': mol_H, 'O': mol_O, 'N': mol_N}

def run_audit():
    print(f"{'='*100}")
    print(f"GASIFIER PHYSICS AUDIT - CONSERVATION & PLAUSIBILITY CHECKS")
    print(f"{'='*100}\n")
    
    audit_results = []
    
    for case_name, data in VALIDATION_CASES.items():
        print(f"Auditing Case: {case_name}...")
        
        inputs = data['inputs']
        expected = data['expected']
        coal_key = inputs['coal']
        coal_props = COAL_DATABASE.get(coal_key)
        
        # Setup Simulation
        op_conds = {
            'coal_flow': inputs['FeedRate'] / 3600.0,
            'o2_flow': (inputs['FeedRate'] * inputs['Ratio_OC']) / 3600.0,
            'steam_flow': (inputs['FeedRate'] * inputs['Ratio_SC']) / 3600.0,
            'P': inputs['P'],
            'T_in': inputs['TIN'],
            'HeatLossPercent': inputs.get('HeatLossPercent', 0.0),
            'SlurryConcentration': inputs.get('SlurryConcentration', 100.0),
        }
        geometry = {'L': 8.0, 'D': 2.6} 
        scaling = {'kinetics': {'comb': 1.0, 'gas': 1.0, 'mixing': 1.0}}
        
        system = GasifierSystem(geometry, coal_props, op_conds, scaling)
        
        # 0. Pre-Check Feed Atoms (First Principles)
        atoms_feed = get_atoms_from_feed(op_conds, coal_props)
        
        try:
            results, z_grid = system.solve(N_cells=50)
            
            # 1. Conservation Audit
            last_state_arr = results[-1]
            last_state = StateVector.from_array(last_state_arr, P=op_conds['P'], z=z_grid[-1])
            
            atoms_out = get_atoms_from_state(last_state, coal_props)
            
            # Check Inlet State atoms (System calculation) for debugging
            inlet_state = system.cells[0].inlet
            atoms_sys_in = get_atoms_from_state(inlet_state, coal_props)
            
            # Conservation Logic
            # The system introduces moisture/oxidant into the 'inlet' gas stream.
            # So Comparing atoms_sys_in vs atoms_out checks the SOLVER conservation.
            # Comparing atoms_feed vs atoms_out checks the INITIALIZATION + SOLVER conservation.
            
            balance_data = {}
            has_leak = False
            print(f"  [Audit Trace] Element Conservation (mol/s):")
            for a in ['C', 'H', 'O', 'N']:
                vin = atoms_feed[a]
                vout = atoms_out[a]
                # Scale acceptable error by magnitude
                err = abs(vin - vout)
                pct_err = (err / (vin + 1e-9)) * 100.0
                
                print(f"    {a}: In={vin:.4f}, Out={vout:.4f}, Err={pct_err:.2f}%")
                
                status = "PASS"
                if pct_err > 0.1: # 0.1% strict threshold
                    status = "FAIL"
                    has_leak = True
                
                balance_data[a] = pct_err
                
            # 2. Process Plausibility Audit
            T_out_C = last_state.T - 273.15
            T_status = "PASS"
            if not (800 <= T_out_C <= 2000):
                T_status = f"FAIL ({T_out_C:.1f}C)"
                
            # H2/CO Ratio
            # Dry Gas Basis
            F_dry = sum(last_state.gas_moles[:7])
            Y_CO = last_state.gas_moles[2] / F_dry
            Y_H2 = last_state.gas_moles[5] / F_dry
            Y_CH4 = last_state.gas_moles[1] / F_dry
            
            ratio_H2_CO = Y_H2 / (Y_CO + 1e-9)
            
            # Typical Entrained Flow: H2/CO ~ 0.5 - 1.0 (Coal)
            # Slurry feed shifts WGS, increasing H2. Can be 0.8 - 1.2.
            # Audit thresholds (loose checking for "plausibility")
            Ratio_status = "PASS"
            if ratio_H2_CO < 0.3 or ratio_H2_CO > 2.0:
                 Ratio_status = f"WARN ({ratio_H2_CO:.2f})"
                 
            CH4_status = "PASS"
            if Y_CH4 > 0.01: # Entrained flow usually < 1% CH4 exiting (unless partial quench/low T)
                 CH4_status = f"WARN ({Y_CH4*100:.2f}%)"
            
            audit_results.append({
                'Case': case_name,
                'Cons_C(%)': balance_data['C'],
                'Cons_H(%)': balance_data['H'],
                'Cons_O(%)': balance_data['O'],
                'Cons_N(%)': balance_data['N'],
                'T_Out': T_status,
                'H2/CO': Ratio_status,
                'CH4_Level': CH4_status
            })
            
        except Exception as e:
            print(f"  [ERROR] Audit crashed: {e}")
    
    # Print Report
    print("\nGASIFIER AUDIT SUMMARY")
    print("-" * 100)
    df = pd.DataFrame(audit_results)
    
    # Format floats
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', lambda x: '%.4f' % x if isinstance(x, float) else str(x))
    
    print(df)
    
    # Final Judgement
    print("\nOVERALL AUDIT VERDICT:")
    if df['Cons_C(%)'].max() > 0.1 or df['Cons_H(%)'].max() > 0.1 or df['Cons_O(%)'].max() > 0.1:
        print("❌ FAILED: Mass conservation violation detected (>0.1%).")
    elif df['T_Out'].str.contains("FAIL").any():
         print("❌ FAILED: Temperature out of physical range (800-2000C).")
    else:
        print("✅ PASSED: Physics constraints satisfied.")

if __name__ == "__main__":
    run_audit()
