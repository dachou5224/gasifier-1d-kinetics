#!/usr/bin/env python3
"""
重新组织和规范化验证案例。

任务:
1. 加载所有 JSON 算例文件 (pilot, industrial, OriginalPaper, new)。
2. 标准化字段: FeedRate, SlurryConc, P, T, CoalAnalysis。
3. 按 Capacity (Pilot, Industrial) 分类。
4. 按 Feed (Slurry-fed, Dry-fed) 分类。
5. 去重: 基于工况哈希或名称映射。
6. 输出 validation_cases_final.json。
"""
import json
import os
import hashlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")

FILE_LIST = [
    "validation_cases_OriginalPaper.json",
    "validation_cases_new.json",
    "validation_cases_industrial.json",
    "validation_cases_pilot.json"
]

# 用于去重的哈希映射 (避免重复算例)
SEEN_CASES = {}

def load_json(name):
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_hash(rec):
    """根据煤质和核心工况生成哈希，用于去重"""
    inputs = rec.get("operating_conditions", {})
    key_str = f"{inputs.get('coal_feed_rate_kg_hr', 0):.2f}_" \
              f"{inputs.get('O2_to_fuel_ratio', 0):.3f}_" \
              f"{inputs.get('steam_to_fuel_ratio', 0):.3f}_" \
              f"{inputs.get('slurry_concentration_pct', 100):.1f}_" \
              f"{inputs.get('pressure_Pa', 0):.0f}"
    return hashlib.md5(key_str.encode()).hexdigest()

def normalize_case(name, rec, source):
    """将不同来源的算例转换为统一格式"""
    norm = {
        "name": name,
        "source": source,
        "description": rec.get("description", ""),
        "operating_conditions": {},
        "expected_results": rec.get("expected_results", rec.get("expected", rec.get("target_output", {}))),
        "coal": rec.get("coal", {}),
        "feedstock_type_orig": rec.get("feedstock_type", "")
    }
    
    # 提取工况
    inp = rec.get("operating_conditions", rec.get("input", rec.get("inputs", {})))
    
    if "FeedRate" in inp: # Pilot 风格
        norm["operating_conditions"] = {
            "coal_feed_rate_kg_hr": inp.get("FeedRate", 0),
            "O2_to_fuel_ratio": inp.get("Ratio_OC", 0),
            "steam_to_fuel_ratio": inp.get("Ratio_SC", 0),
            "pressure_Pa": inp.get("P", 101325),
            "inlet_temperature_K": inp.get("TIN", 298.15),
            "slurry_concentration_pct": inp.get("SlurryConcentration", 100),
            "heat_loss_percent": inp.get("HeatLossPercent", 0)
        }
    elif "slurry_conc" in inp and "O2_coal_ratio_daf" in inp: # validation_cases_new.json 风格
        default_feed_kg_h = 41670.0
        norm["operating_conditions"] = {
            "coal_feed_rate_kg_hr": default_feed_kg_h,
            "O2_to_fuel_ratio": inp.get("O2_coal_ratio_daf", 0),
            "steam_to_fuel_ratio": 0.0,
            "pressure_Pa": inp.get("pressure", 4.083e6),
            "inlet_temperature_K": 400.0,
            "slurry_concentration_pct": inp.get("slurry_conc", 0.65) * 100 if inp.get("slurry_conc", 1) <= 1 else inp.get("slurry_conc"),
            "heat_loss_percent": 2.0
        }
        norm["coal"] = inp.get("coal_dry", {})
    else:
        # 复制所有字段
        norm["operating_conditions"] = dict(inp)
        # 补全缺失的关键字段映射
        if "coal_feed_rate_kg_hr" not in norm["operating_conditions"]:
            if "FeedRate" in norm["operating_conditions"]:
                norm["operating_conditions"]["coal_feed_rate_kg_hr"] = norm["operating_conditions"]["FeedRate"]
            elif "coal_feed_rate_g_s" in norm["operating_conditions"]:
                norm["operating_conditions"]["coal_feed_rate_kg_hr"] = norm["operating_conditions"]["coal_feed_rate_g_s"] * 3.6
        
        if "pressure_Pa" not in norm["operating_conditions"]:
            if "pressure_atm" in norm["operating_conditions"]:
                norm["operating_conditions"]["pressure_Pa"] = norm["operating_conditions"]["pressure_atm"] * 101325
            elif "P" in norm["operating_conditions"]:
                 norm["operating_conditions"]["pressure_Pa"] = norm["operating_conditions"]["P"]
        
        if "slurry_concentration_pct" not in norm["operating_conditions"]:
            if "SlurryConcentration" in norm["operating_conditions"]:
                norm["operating_conditions"]["slurry_concentration_pct"] = norm["operating_conditions"]["SlurryConcentration"]
            elif "slurry_conc" in norm["operating_conditions"]:
                 s = norm["operating_conditions"]["slurry_conc"]
                 norm["operating_conditions"]["slurry_concentration_pct"] = s * 100 if s <= 1 else s
            elif "water_to_coal_ratio" in norm["operating_conditions"]:
                 w_ratio = norm["operating_conditions"]["water_to_coal_ratio"]
                 norm["operating_conditions"]["slurry_concentration_pct"] = 100 / (1 + w_ratio)

    # 分类逻辑
    feed_rate = float(norm["operating_conditions"].get("coal_feed_rate_kg_hr", 0))
    slurry = float(norm["operating_conditions"].get("slurry_concentration_pct", 100))
    feedstock = norm["feedstock_type_orig"].lower()
    
    # Capacity
    if feed_rate >= 1000:
        norm["capacity"] = "Industrial"
    elif feed_rate > 0:
        norm["capacity"] = "Pilot"
    else:
        if "pilot" in norm["description"].lower() or "pilot" in source.lower():
            norm["capacity"] = "Pilot"
        elif "industrial" in norm["description"].lower() or "industrial" in source.lower():
            norm["capacity"] = "Industrial"
        else:
            norm["capacity"] = "Pilot" 
        
    # Feed Type
    if slurry < 99.9 or "slurry" in feedstock or "water" in feedstock:
        norm["feed_type"] = "Slurry-fed"
    else:
        norm["feed_type"] = "Dry-fed"
        
    return norm

def main():
    final_data = {
        "metadata": {
            "title": "Normalized Gasification Validation Cases",
            "categories": ["Capacity (Pilot/Industrial)", "Feed (Slurry/Dry)"]
        },
        "Pilot": {
            "Slurry-fed": {},
            "Dry-fed": {}
        },
        "Industrial": {
            "Slurry-fed": {},
            "Dry-fed": {}
        }
    }
    
    coal_db = {}

    for fname in FILE_LIST:
        data = load_json(fname)
        if not data: continue
        
        if "coal_database" in data:
            coal_db.update(data["coal_database"])
            
        cases_dict = data.get("validation_cases", data.get("fortran_cases", data.get("cases", {})))
        
        for name, rec in cases_dict.items():
            norm = normalize_case(name, rec, fname)
            
            h = get_hash(norm)
            if h in SEEN_CASES:
                if "OriginalPaper" in fname:
                    pass # 原则上保留 OriginalPaper
                else:
                    continue
            
            SEEN_CASES[h] = name
            
            cap = norm["capacity"]
            ftype = norm["feed_type"]
            
            if cap in final_data and ftype in final_data[cap]:
                final_data[cap][ftype][name] = norm

    out_path = os.path.join(DATA, "validation_cases_final.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
        
    print(f"整理完成！输出文件: {out_path}")
    for cap in ["Pilot", "Industrial"]:
        for ftype in ["Slurry-fed", "Dry-fed"]:
            count = len(final_data[cap][ftype])
            print(f"  - {cap} | {ftype}: {count} 算例")

if __name__ == "__main__":
    main()
