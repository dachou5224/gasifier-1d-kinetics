
import numpy as np
import sys
import os

# Adjust path to include src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from model.physics import calculate_enthalpy
from model.constants import PhysicalConstants

def test_methane_combustion():
    print("\n=== Verification: CH4 + 2O2 -> CO2 + 2H2O (Gas) ===")
    T_ref = 298.15
    T_high = 2000.0
    
    # Method A: Formation Enthalpy Difference (State Function)
    # H = Hf + (H(T) - H(298))
    
    # Reactants: 1 CH4, 2 O2
    H_CH4_298 = calculate_enthalpy('CH4', T_ref)
    H_O2_298  = calculate_enthalpy('O2', T_ref)
    H_in_298 = 1.0 * H_CH4_298 + 2.0 * H_O2_298
    
    H_CH4_2000 = calculate_enthalpy('CH4', T_high)
    H_O2_2000  = calculate_enthalpy('O2', T_high)
    H_in_2000 = 1.0 * H_CH4_2000 + 2.0 * H_O2_2000
    
    # Products: 1 CO2, 2 H2O
    H_CO2_298 = calculate_enthalpy('CO2', T_ref)
    H_H2O_298 = calculate_enthalpy('H2O', T_ref)
    H_out_298 = 1.0 * H_CO2_298 + 2.0 * H_H2O_298
    
    H_CO2_2000 = calculate_enthalpy('CO2', T_high)
    H_H2O_2000 = calculate_enthalpy('H2O', T_high)
    H_out_2000 = 1.0 * H_CO2_2000 + 2.0 * H_H2O_2000
    
    # Delta H (Heat of Combustion)
    # Note: Exothermic means H_prod < H_react, so Delta H is negative.
    Delta_H_298_MethodA = H_out_298 - H_in_298
    Delta_H_2000_MethodA = H_out_2000 - H_in_2000
    
    print(f"Method A (NIST Shomate State Function):")
    print(f"  Delta H_comb (298K):  {Delta_H_298_MethodA/1e3:.2f} kJ/mol")
    print(f"  Delta H_comb (2000K): {Delta_H_2000_MethodA/1e3:.2f} kJ/mol")
    
    # Method B: Standard Heat of Reaction + Sensible Heat Correction
    # Delta H(T) = Delta H(298) + Integral(Delta Cp dT)
    # Standard theoretical value for CH4 LHV is approx -802 kJ/mol
    LHV_CH4 = -802340.0 # J/mol (Table value)
    
    difference_298 = abs(Delta_H_298_MethodA - LHV_CH4)
    print(f"  vs Standard LHV (-802.3 kJ): Error = {difference_298/1e3:.2f} kJ ({difference_298/abs(LHV_CH4)*100:.2f}%)")
    
    if difference_298 < 5000.0: # 5 kJ tolerance
        print(">> VERDICT: Method A MATCHES Standard Heat of Reaction.")
    else:
        print(">> VERDICT: Method A DISAGREES with Standard. Check Formation Enthalpies!")
        
def test_char_combustion():
    print("\n=== Verification: C(s) + O2 -> CO2 (Heterogeneous) ===")
    T_ref = 298.15
    
    # Reactants: 1 C(s), 1 O2
    # C(s) enthalpy model in material.py: Hf + Cp(T-298)
    # Hf_coal assumed ~0 for pure Carbon (Graphite)? Or Coal Hf?
    # Let's test Pure Carbon (Graphite) limit.
    Hf_C = 0.0
    Cp_C = 1200.0 # J/kgK approx mean? No, let's use code value constant
    # Wait, code uses coal_props. Let's assume ideal Graphite.
    
    H_C_298 = Hf_C 
    H_O2_298 = calculate_enthalpy('O2', T_ref)
    H_in = H_C_298 + H_O2_298
    
    # Products: 1 CO2
    H_CO2_298 = calculate_enthalpy('CO2', T_ref)
    H_out = H_CO2_298
    
    Delta_H = H_out - H_in
    print(f"Method A (State Function):")
    print(f"  Delta H (C+O2->CO2, 298K): {Delta_H/1e3:.2f} kJ/mol")
    
    Standard_H = -393510.0
    print(f"  vs Standard (-393.5 kJ): {Delta_H - Standard_H:.2f} J")

if __name__ == "__main__":
    test_methane_combustion()
    test_char_combustion()
