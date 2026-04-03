#!/usr/bin/env python3
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from model.chemistry import COAL_DATABASE

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "validation_cases_final.json")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
    
case_data = data["Industrial"]["Slurry-fed"]["LuNan_Texaco"]
coal_type = case_data.get("coal_type", "Paper_Base_Coal")
print(f"JSON 指定的煤种: '{coal_type}'")
print(f"JSON 中是否包含 'coal' 数据字典: {'Yes' if case_data.get('coal') else 'No'}")

coal_props = COAL_DATABASE.get(coal_type, COAL_DATABASE["Paper_Base_Coal"])
print(f"\n加载到的煤质数据:")
for k, v in coal_props.items():
    print(f"  {k}: {v}")
    
print("\n原始正确的 LuNan_Coal 数据:")
for k, v in COAL_DATABASE["LuNan_Coal"].items():
    print(f"  {k}: {v}")

