import os, sys
sys.path.insert(0, os.path.abspath('src'))
from model.gasifier_system import GasifierSystem
from model.chemistry import COAL_DATABASE, VALIDATION_CASES
import numpy as np
import pandas as pd

case_data = VALIDATION_CASES["Paper_Case_6"]["inputs"]
coal_name = case_data.get("coal", "Paper_Base_Coal")
coal = COAL_DATABASE.get(coal_name, {})

feed_kg_h = case_data.get("FeedRate", 41670.0)
ratio_oc = case_data.get("Ratio_OC", 1.05)
ratio_sc = case_data.get("Ratio_SC", 0.08)
coal_flow = feed_kg_h / 3600.0

geometry = {
    "L": case_data.get("L", 6.0), 
    "D": case_data.get("D", 2.0)
}
coal_props = {
    "Cd": coal.get("Cd", 80.19), "Hd": coal.get("Hd", 4.83), "Od": coal.get("Od", 9.76), "Ad": coal.get("Ad", 7.35),
    "HHV_d": 29800.0
}
op_conds = {
    "coal_flow": coal_flow,
    "o2_flow": coal_flow * ratio_oc,
    "steam_flow": coal_flow * ratio_sc,
    "P": case_data.get("P", 4.08e6),
    "T_in": case_data.get("TIN", 300.0),
    "SlurryConcentration": case_data.get("SlurryConc", 60.0),
    "HeatLossPercent": case_data.get("HeatLossPercent", 3.0),
    "AdaptiveFirstCellLength": True
}

print(f"Running Paper Case 6 with Heat Loss: {op_conds['HeatLossPercent']}%")
system = GasifierSystem(geometry, coal_props, op_conds)
try:
    results, z = system.solve(N_cells=40)
    
    df = pd.DataFrame(results, columns=["O2", "CH4", "CO", "CO2", "H2S", "H2", "N2", "H2O", "W_solid", "X_C", "T"])
    df["z"] = z
    df["T_C"] = df["T"] - 273.15
    print("\n========= EXTREMES =========")
    print(f"Max T: {df['T_C'].max():.1f} °C at z={df.iloc[df['T_C'].idxmax()]['z']:.2f}m")
    print(f"Exit T: {df['T_C'].iloc[-1]:.1f} °C")
    print("\n========= PROFILE =========")
    print(df[["z", "T_C", "O2", "CO", "CH4", "X_C"]].to_string(index=False))
except Exception as e:
    import traceback
    traceback.print_exc()
