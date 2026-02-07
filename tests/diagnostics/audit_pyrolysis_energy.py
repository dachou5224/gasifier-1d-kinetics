import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from model.pyrolysis_service import PyrolysisService
from model.physics import get_lhv

def audit_pyrolysis_energy():
    # Paper Case 6 Coal
    coal_props = {
        'Cd': 80.19, 'Hd': 4.83, 'Od': 9.76, 'Ad': 7.35,
        'Nd': 0.0, 'Sd': 0.0, 'HHV_d': 30.0, # MJ/kg
        'FCd': 60.0 # Standard assumption if not provided
    }
    
    pyro = PyrolysisService()
    molar_yields, W_vol_kg = pyro.calc_yields(coal_props)
    
    # Species heating values (LHV J/mol)
    species = ['O2', 'CH4', 'CO', 'CO2', 'H2S', 'H2', 'N2', 'H2O']
    lhv_volatiles = 0.0
    for i, sp in enumerate(species):
        lhv_volatiles += molar_yields[i] * get_lhv(sp)
    
    # Residual Char (Solid Carbon)
    # W_dry = 1.0 kg coal
    # Mass of Carbon in Volatiles
    C_in_vol_kg = (molar_yields[1] + molar_yields[2] + molar_yields[3]) * 12.011 / 1000.0
    C_total_kg = coal_props['Cd'] / 100.0
    C_char_kg = C_total_kg - C_in_vol_kg
    lhv_char = C_char_kg * (393510 / 0.012011) # J/kg_C * kg_C
    
    total_lhv_prod = (lhv_volatiles + lhv_char) / 1e6 # MJ/kg coal
    
    print(f"--- Pyrolysis Energy Audit (Case 6) ---")
    print(f"  Coal HHV (Dry): {coal_props['HHV_d']:.2f} MJ/kg")
    # Estimate LHV coal manually or just compare products
    print(f"  Volatile LHV:  {lhv_volatiles/1e6:10.2f} MJ/kg_coal")
    print(f"  Char LHV:      {lhv_char/1e6:10.2f} MJ/kg_coal")
    print(f"  Total Product LHV: {total_lhv_prod:10.2f} MJ/kg_coal")
    
    if total_lhv_prod > coal_props['HHV_d']:
        print(f"  WARNING: Energy Created! Yields must be normalized.")
    else:
        print(f"  ENERGY STATUS: OK (Gap covers heat of pyrolysis)")

if __name__ == '__main__':
    audit_pyrolysis_energy()
