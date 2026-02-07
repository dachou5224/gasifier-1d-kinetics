
import numpy as np
from model.kinetics import HeterogeneousKinetics
from model.physics import R_CONST

def test_rates():
    het = HeterogeneousKinetics()
    T = 600.0
    P = 4.0e6
    P_i = 1.6e6 # 40% CO2
    d_p = 100e-6
    Y = 1.0
    
    print(f"--- Kinetic Rate Debug (T={T}K, P={P/1e6}MPa) ---")
    for rxn, p in het.params.items():
        A = p['A']
        E = p['E']
        print(f"\nReaction: {rxn}")
        print(f"  A: {A:.2e}")
        print(f"  E: {E:.2e} J/mol")
        
        # ks
        ks = A * np.exp(-E / (R_CONST * T))
        print(f"  ks: {ks:.2e} m/s")
        
        # kd (approx)
        species_map = {'C+O2': 'O2', 'C+H2O': 'H2O', 'C+CO2': 'CO2', 'C+H2': 'H2'}
        # For debug, we just call the method
        rate = het.calculate_total_rate(rxn, T, P, P_i, d_p, Y)
        print(f"  Total Rate (kmol/m2.s): {rate:.2e}")

if __name__ == "__main__":
    test_rates()
