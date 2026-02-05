
import numpy as np
import sys
import os

# Adjust path to include src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from model.physics import R_CONST

def test_rate_magnitude():
    print("\n=== Verification: CH4 Oxidation Rate Magnitude ===")
    
    # Parameters from KineticsService
    A = 1.6e10 # m^3/(kmol s) ?? Assuming units based on concentrations
    E = 9.304e4 # J/mol
    
    T_test = 2000.0 # K
    P_test = 40.0 * 101325.0 # 40 atm ~ 4e6 Pa
    
    # Calculate k
    k = A * np.exp(-E / (R_CONST * T_test))
    print(f"Temperature: {T_test} K")
    print(f"Reaction Constant k: {k:.4e}")
    
    # Calculate Concentration (kmol/m3)
    # Assume 10% CH4, 20% O2
    y_CH4 = 0.1
    y_O2 = 0.2
    
    # C = P/RT (mol/m3) / 1000 -> kmol/m3
    C_total_mol = P_test / (R_CONST * T_test)
    C_total_kmol = C_total_mol / 1000.0
    
    C_CH4 = y_CH4 * C_total_kmol
    C_O2  = y_O2 * C_total_kmol
    
    print(f"Total Concentration: {C_total_kmol:.4e} kmol/m3")
    print(f"C_CH4: {C_CH4:.4e} kmol/m3")
    print(f"C_O2:  {C_O2:.4e} kmol/m3")
    
    # Rate Calculation
    # r = k * Ca * Cb
    rate_kmol_m3_s = k * C_CH4 * C_O2
    rate_mol_m3_s = rate_kmol_m3_s * 1000.0
    
    print(f"Rate (kmol/m3.s): {rate_kmol_m3_s:.4e}")
    print(f"Rate (mol/m3.s):  {rate_mol_m3_s:.4e}")
    
    # Volume for Cell 0 (approx from logs)
    # D=2.0m, L=0.4m -> V = pi*1^2*0.4 = 1.257 m3
    Vol = 1.257 
    Rate_mol_s = rate_mol_m3_s * Vol
    
    print(f"Total Rate in Cell 0 (mol/s): {Rate_mol_s:.4e}")
    
    # Heat Release
    Delta_H = -802000.0 # J/mol
    Q_rel_MW = (Rate_mol_s * -Delta_H) / 1e6
    print(f"Heat Release (MW): {Q_rel_MW:.4f} MW")

if __name__ == "__main__":
    test_rate_magnitude()
