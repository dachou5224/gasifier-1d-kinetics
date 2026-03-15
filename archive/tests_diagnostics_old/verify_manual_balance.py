
import numpy as np

def manual_heat_balance():
    print("=== Manual Heat Balance: Paper Case 6 (Cell 0) ===")
    
    # 1. Inputs
    FeedRate_kg_s = 41670.0 / 3600.0 # 11.575 kg/s
    OC_Ratio = 1.05
    SC_Ratio = 0.08
    O2_Flow_kg_s = FeedRate_kg_s * OC_Ratio # 12.15 kg/s
    Steam_Flow_kg_s = FeedRate_kg_s * SC_Ratio # 0.926 kg/s
    
    # Carbon Flow
    C_frac = 0.8019 # From test_validation_config.py
    C_flow_kg_s = FeedRate_kg_s * C_frac # 9.28 kg/s
    C_flow_mol_s = C_flow_kg_s / 0.012011 # 772.6 mol/s
    
    # Oxygen Flow
    O2_flow_mol_s = O2_Flow_kg_s / 0.031998 # 379.8 mol/s
    
    print(f"Feed Rate: {FeedRate_kg_s:.3f} kg/s")
    print(f"C Flow:    {C_flow_mol_s:.1f} mol/s")
    print(f"O2 Flow:   {O2_flow_mol_s:.1f} mol/s")
    
    # 2. Heat Load (Heating + Evaporation)
    # Heating coal/oxidant from 300K to 1300K (Typical T_gasifier)
    Delta_T = 1300 - 300
    Cp_solid = 1200.0 # J/kgK
    Cp_gas = 1100.0 # J/kgK
    
    Q_heat_solid = FeedRate_kg_s * Cp_solid * Delta_T / 1e6 # MW
    Q_heat_gas = O2_Flow_kg_s * Cp_gas * Delta_T / 1e6 # MW
    
    # Moisture Evaporation (14.69 MW from logs)
    Q_evap = 14.69
    
    Total_Load = Q_heat_solid + Q_heat_gas + Q_evap
    print(f"\nHeat Load (to reach 1300K):")
    print(f"  Solid Heating:    {Q_heat_solid:.2f} MW")
    print(f"  Gas Heating:      {Q_heat_gas:.2f} MW")
    print(f"  Moisture/Evap:    {Q_evap:.2f} MW")
    print(f"  TOTAL LOAD:       {Total_Load:.2f} MW")
    
    # 3. Potential Heat Release (Combustion)
    # Reaction: C + O2 -> CO2 (Delta H = -393.5 kJ/mol)
    Delta_H_C = 393522.0 # J/mol
    Q_rel_max = (O2_flow_mol_s * Delta_H_C) / 1e6
    
    print(f"\nPotential Heat Release:")
    print(f"  Max C+O2 Heat:   {Q_rel_max:.2f} MW")
    print(f"  Margin (Rel-Load): {Q_rel_max - Total_Load:.2f} MW")
    
    # 4. Homogeneous Kinetics Check (The "Explosion" factor)
    # A = 1.6e10 [m3/(kmol s)] ??
    # Ca = 0.2 kmol/m3
    # r = 1.6e10 * 0.2 * 0.2 = 6.4e8 kmol/m3.s
    # V = 1.25 m3
    # R_total = 8e8 kmol/s = 8e11 mol/s
    # Q = 8e11 * 800000 = 6.4e17 Watts = 6.4e11 MW
    
    A_code = 1.6e10
    C_test = 0.2
    Rate_kmol_m3_s = A_code * C_test * C_test 
    Q_code_MW = (Rate_kmol_m3_s * 1.25 * 802340.0 * 1000.0) / 1e6
    
    print(f"\nKinetics Stiffness Check (T=2000K, A=1.6e10):")
    print(f"  Code-calculated Potential: {Q_code_MW:.2e} MW")
    print(f"  Wait, that's {Q_code_MW/1e6:.1f} TeraWatts!!")

if __name__ == "__main__":
    manual_heat_balance()
