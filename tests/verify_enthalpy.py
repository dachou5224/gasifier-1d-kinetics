import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath('src'))

from model.physics import get_enthalpy_molar, SHOMATE_DB

# Standard Enthalpy of Formation at 298.15 K (kJ/mol) from NIST WebBook
NIST_HF_298 = {
    'CO': -110.53,
    'CO2': -393.51,
    'H2': 0.0,
    'H2O': -241.83,
    'CH4': -74.87,
    'N2': 0.0,
    'O2': 0.0,
    'H2S': -20.50,
}

print(f"{'Species':<10} | {'Code H(298.15) (kJ/mol)':<25} | {'NIST Hf (kJ/mol)':<15} | {'Diff':<10}")
print("-" * 70)

for species, nist_val in NIST_HF_298.items():
    code_h_j_mol = get_enthalpy_molar(species, 298.15)
    code_h_kj_mol = code_h_j_mol / 1000.0
    diff = code_h_kj_mol - nist_val
    print(f"{species:<10} | {code_h_kj_mol:<25.4f} | {nist_val:<15.2f} | {diff:<10.4f}")
