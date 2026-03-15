import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath('src'))

from model.physics import get_enthalpy_molar, calculate_cp, SHOMATE_DB

print(f"{'Species':<10} | {'T_cut':<8} | {'Delta H (J/mol)':<15} | {'Delta Cp (J/mol*K)':<15}")
print("-" * 60)

for species, data in SHOMATE_DB.items():
    t_cut = data['T_cut']
    
    # Calculate Low T values at cut-off
    # We temporarily force the selection by modifying SHOMATE_DB or using internal logic
    # Re-implementing a quick calc here to bypass the automatic selection in the functions
    
    def calc_val(coeffs, T, prop_type):
        t = T / 1000.0
        A, B, C, D, E, F, G, H = coeffs
        if prop_type == 'Cp':
            return A + B*t + C*t**2 + D*t**3 + E/(t**2)
        elif prop_type == 'H':
            return (A*t + B*(t**2)/2 + C*(t**3)/3 + D*(t**4)/4 - E/t + F) * 1000.0
            
    h_low = calc_val(data['Low'], t_cut, 'H')
    h_high = calc_val(data['High'], t_cut, 'H')
    cp_low = calc_val(data['Low'], t_cut, 'Cp')
    cp_high = calc_val(data['High'], t_cut, 'Cp')
    
    delta_h = h_high - h_low
    delta_cp = cp_high - cp_low
    
    print(f"{species:<10} | {t_cut:<8} | {delta_h:<15.2f} | {delta_cp:<15.4f}")
