import os, sys
sys.path.insert(0, os.path.abspath('src'))
from model.physics import get_enthalpy_molar
from model.material import SPECIES_NAMES

print("=== Standard Enthalpy of Formation at 298.15K (J/mol) ===")
for sp in SPECIES_NAMES:
    h298 = get_enthalpy_molar(sp, 298.15)
    print(f"{sp:5}: {h298:.2f}")

print("\n=== Check CH4 Oxidation Heat ===")
# CH4 + 2O2 -> CO2 + 2H2O
h_ch4 = get_enthalpy_molar('CH4', 298.15)
h_o2 = get_enthalpy_molar('O2', 298.15)
h_co2 = get_enthalpy_molar('CO2', 298.15)
h_h2o = get_enthalpy_molar('H2O', 298.15) # gas

delta_H_rxn = (h_co2 + 2 * h_h2o) - (h_ch4 + 2 * h_o2)
print(f"Reaction Heat (CH4 + 2O2 -> CO2 + 2H2O, T=298K) = {delta_H_rxn:.2f} J/mol")
