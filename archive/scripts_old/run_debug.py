import os, sys
sys.path.insert(0, os.path.abspath('src'))
from model.gasifier_system import GasifierSystem
from model.chemistry import COAL_DATABASE, VALIDATION_CASES
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

case_data = VALIDATION_CASES["Paper_Case_6"]["inputs"]
coal_name = case_data.get("coal", "Paper_Base_Coal")
coal = COAL_DATABASE.get(coal_name, {})

feed_kg_h = case_data.get("FeedRate", 41670.0)
ratio_oc = case_data.get("Ratio_OC", 1.05)
ratio_sc = case_data.get("Ratio_SC", 0.08)
coal_flow = feed_kg_h / 3600.0

geometry = {"L": 6.0, "D": 2.0}
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

system = GasifierSystem(geometry, coal_props, op_conds)
try:
    results, z = system.solve(N_cells=10) # 跑10格看看开头能量积聚
    print("\n=== SOLVER RESULTS ===")
    for i in range(min(5, len(results))):
        st = results[i]
        print(f"Cell {i}: z={z[i]:.2f}m, T={st.T:.1f}K, CH4={st.gas_moles[1]:.2f}, CO={st.gas_moles[2]:.2f}")
except Exception as e:
    import traceback
    traceback.print_exc()
