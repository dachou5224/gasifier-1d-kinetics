import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from model.pyrolysis_service import PyrolysisService

def test_pyrolysis():
    print("Testing PyrolysisService Mass Balance...")
    
    # Test Coal (Paper Base)
    # Cd=80.19, Hd=4.83, Od=9.76, Nd=0.85, Sd=0.41, Ad=7.35, Vd=31.24, FCd=61.41
    # Sum: 80.19+4.83+9.76+0.85+0.41 = 96.04. Ash=7.35. Total=103.39??
    # Ah, Ultimate often doesn't sum to 100 perfectly with Ash/Moisture if data sources mix.
    # Usually: C+H+O+N+S+Ash = 100 on Dry Basis.
    # Let's check this coal: 96.04 + 7.35 = 103.39.
    # This means the provided `Paper_Base_Coal` data is mathematically inconsistent (Sum > 100).
    # If the Input Coal sums to > 100, then Pyrolysis conserving atoms (CHONS) + Ash will sum to > 100.
    # W_dry = 1 kg.
    # Atoms mass = 0.96 kg. Ash = 0.07 kg. Total = 1.03 kg.
    # The System assumes W_dry = 1.0.
    # If Atoms+Ash > 1.0, then Mass Balance Check (Input=1.0) vs (Output=1.03) will FAIL.
    # Errors: 96/100 -> ~4% error?
    # Wait, N error was 24%.
    
    coal = {
        "Cd": 80.19, "Hd": 4.83, "Od": 9.76, "Nd": 0.85, "Sd": 0.41,
        "Ashd": 7.35, "FCd": 61.41, "VMd": 31.24
    }
    
    # Normalize Coal to 100%?
    total_raw = coal['Cd'] + coal['Hd'] + coal['Od'] + coal['Nd'] + coal['Sd'] + coal['Ashd']
    print(f"Coal Sum (Dry): {total_raw:.2f}%")
    
    # Pyrolysis Calculation
    pyro = PyrolysisService()
    yields, W_vol = pyro.calc_yields(coal)
    
    print(f"Calculated Yields (mol/kg):")
    MW = [32.0, 16.04, 28.01, 44.01, 34.08, 2.016, 28.013, 18.015]
    names = ['O2', 'CH4', 'CO', 'CO2', 'H2S', 'H2', 'N2', 'H2O']
    
    total_mass_gas = 0.0
    total_C = 0.0
    total_N = 0.0
    
    for i in range(8):
        y = yields[i]
        mass = y * MW[i] / 1000.0 # kg/kg_dry
        total_mass_gas += mass
        print(f"  {names[i]:<5}: {y:.4f} mol/kg  ({mass*100:.2f} wt%)")
        
        # Carbon
        if i == 1: total_C += y * 1 # CH4
        if i == 2: total_C += y * 1 # CO
        if i == 3: total_C += y * 1 # CO2
        
        # Nitrogen
        if i == 6: total_N += y * 2 # N2 (2N)
        
    print(f"\nTotal Volatile Mass (Yields): {total_mass_gas:.4f} kg/kg")
    print(f"Service W_vol: {W_vol:.4f} kg/kg")
    
    # Balance Check
    # Carbon
    C_in = coal['Cd'] / 100.0
    C_gas = total_C * 12.011 / 1000.0
    C_rem = C_in - C_gas
    print(f"Carbon: In={C_in:.4f}, Gas={C_gas:.4f}, Rem={C_rem:.4f}")
    
    # Nitrogen
    N_in = coal['Nd'] / 100.0 # kg/kg
    N_gas = total_N * 14.007 / 1000.0 # kg/kg
    print(f"Nitrogen: In={N_in:.4f}, Gas={N_gas:.4f}, Err={(N_gas-N_in)/N_in*100 if N_in>0 else 0:.2f}%")
    
    # Check if N_in calculation logic matches Pyrolysis logic
    # Pyrolysis: n_N2 (kmol/100) = Nd/28. n_N2 (mol/kg) = Nd/2.8.
    # Mass N2 = (Nd/2.8) * 28 = Nd * 10 g/kg = Nd/100 kg/kg.
    # Should be exact.

if __name__ == "__main__":
    test_pyrolysis()
