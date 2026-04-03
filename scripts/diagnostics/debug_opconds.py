#!/usr/bin/env python3
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from model.chemistry import VALIDATION_CASES
from model.gasifier_system import GasifierSystem

print("=== 原始硬编码算例 (Paper_Case_6) ===")
case_inputs = VALIDATION_CASES["Paper_Case_6"]["inputs"]
coal_flow_kg_s = float(case_inputs["FeedRate"]) / 3600.0
o2_flow_kg_s = coal_flow_kg_s * float(case_inputs.get("Ratio_OC", 1.05))
steam_flow_kg_s = coal_flow_kg_s * float(case_inputs.get("Ratio_SC", 0.0))
op_conds_orig = {
    "coal_flow": coal_flow_kg_s,
    "o2_flow": o2_flow_kg_s,
    "steam_flow": steam_flow_kg_s,
    "P": float(case_inputs["P"]),
    "T_in": float(case_inputs["TIN"]),
    "SlurryConcentration": float(case_inputs.get("SlurryConcentration", 60.0)),
    "HeatLossPercent": float(case_inputs.get("HeatLossPercent", 1.0)),
}
for k, v in op_conds_orig.items():
    print(f"  {k}: {v}")

print("\n=== JSON 解析算例 (LuNan_Texaco) ===")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "validation_cases_final.json")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
cond = data["Industrial"]["Slurry-fed"]["LuNan_Texaco"]["operating_conditions"]

coal_flow = cond.get('coal_feed_rate_kg_hr', 100) / 3600.0
ratio_oc = cond.get('O2_to_fuel_ratio', cond.get('O2_to_coal_ratio', 0.9))
ratio_sc = cond.get('steam_to_fuel_ratio', cond.get('water_to_coal_ratio', 0.0))

op_conds_new = {
    'coal_flow': coal_flow,
    'o2_flow': coal_flow * ratio_oc,
    'steam_flow': coal_flow * ratio_sc,
    'P': cond.get('pressure_Pa', 2.4e6),
    'T_in': cond.get('inlet_temperature_K', 400.0),
    'HeatLossPercent': cond.get('heat_loss_percent', 2.0),
    'SlurryConcentration': cond.get('slurry_concentration_pct', 100.0),
}
for k, v in op_conds_new.items():
    print(f"  {k}: {v}")

