import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath('src'))

from model.gasifier_system import GasifierSystem
from model.state import StateVector
from model.material import SPECIES_NAMES

# Define geometry and coal properties (using Paper_Case_6 as reference)
geom = {'L': 8.0, 'D': 3.4}
coal = {
    'Cd': 60.0, 'Hd': 4.0, 'Nd': 1.0, 'Sd': 0.5, 'Od': 10.0, 'Ashd': 6.0, 'FCd': 50.0, 'VMD': 30.0, 'Mt': 10.0,
    'HHV_d': 32.0, 'Hf_coal': -3.37e6
}
ops = {
    'coal_flow': 45.0, # kg/s
    'o2_flow': 35.0, # kg/s
    'steam_flow': 5.0, # kg/s
    'T_in': 500.0,
    'P': 3.0e6,
    'SlurryConcentration': 65.0,
    'HeatLossPercent': 1.0,
    'particle_diameter': 100e-6
}

system = GasifierSystem(geom, coal, ops)
results, z = system.solve(N_cells=50)

print(f"{'Cell':<5} | {'Z':<8} | {'T (C)':<10} | {'Xc':<8} | {'O2 (mol/s)':<12} | {'H2O (mol/s)':<12}")
print("-" * 75)
for i in range(10): # First 10 cells
    F = results[i]
    T_c = F[10] - 273.15
    print(f"{i:<5} | {z[i]:<8.3f} | {T_c:<10.2f} | {F[9]:<8.4f} | {F[0]:<12.4f} | {F[7]:<12.4f}")
