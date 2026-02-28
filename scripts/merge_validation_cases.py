#!/usr/bin/env python3
"""
合并 data/ 下四个验证算例文件，查重去重后输出 validation_cases_merged.json。

源文件:
  - validation_cases.json          小规模验证算例 (Paper_Case_6)
  - validation_cases_OriginalPaper.json  Wen & Chaung (1979) 原始论文算例
  - validation_cases_fortran.json   Fortran 对标算例 (与 OriginalPaper 重复)
  - validation_cases_new.json       新增算例 (Illinois_No6, Australia_UBE, Fluid_Coke)

查重规则:
  - fortran "texaco i-1" = OriginalPaper "Texaco_I-1" (同工况，保留 OriginalPaper)
  - fortran "slurry western" = OriginalPaper "Coal_Water_Slurry_Western"
  - Paper_Case_6 = Illinois_No6（同一算例，以 Illinois_No6 为准）

用法: python scripts/merge_validation_cases.py
"""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")

# Fortran 算例名 → OriginalPaper 算例名（重复项，以 OriginalPaper 为准）
FORTRAN_TO_ORIGINAL = {
    "texaco i-1": "Texaco_I-1",
    "texaco i-2": "Texaco_I-2",
    "texaco i-5c": "Texaco_I-5C",
    "texaco i-10": "Texaco_I-10",
    "texaco exxon": "Texaco_Exxon",
    "slurry western": "Coal_Water_Slurry_Western",
    "slurry eastern": "Coal_Water_Slurry_Eastern",
}

def load_json(name: str) -> dict:
    with open(os.path.join(DATA, name), "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    op = load_json("validation_cases_OriginalPaper.json")
    new_ = load_json("validation_cases_new.json")

    merged = {
        "metadata": {
            "source": "Merged from validation_cases.json, validation_cases_OriginalPaper.json, "
                     "validation_cases_fortran.json, validation_cases_new.json",
            "merged_from": [
                "validation_cases.json (小规模)",
                "validation_cases_OriginalPaper.json (原始论文)",
                "validation_cases_fortran.json (Fortran 对标)",
                "validation_cases_new.json (新增)",
            ],
            "deduplication": "Fortran 算例与 OriginalPaper 重复，以 OriginalPaper 为准",
            "original_paper": op.get("metadata", {}),
        },
        "coal_database": dict(op.get("coal_database", {})),
        "validation_cases": dict(op.get("validation_cases", {})),
    }

    # 添加 validation_cases_new 中的算例（Illinois_No6, Australia_UBE, Fluid_Coke）
    # 默认进料 41670 kg/h = 11575 g/s，与 validation_loader 一致
    default_feed_kg_h = 41670.0
    default_feed_g_s = default_feed_kg_h / 3600 * 1000
    new_cases = new_.get("validation_cases", new_)
    for name, rec in new_cases.items():
        if name in merged["validation_cases"]:
            continue  # 避免覆盖
        inp = rec.get("input", rec.get("inputs", {}))
        cd = inp.get("coal_dry", {})
        coal_key = f"{name}_Coal"
        slurry = float(inp.get("slurry_conc", 0.65))
        slurry_pct = slurry * 100 if slurry <= 1 else slurry
        ratio_oc = float(inp.get("O2_coal_ratio_daf", 0.9))
        P_Pa = float(inp.get("pressure", 4.083e6))
        merged["coal_database"][coal_key] = {
            "description": f"Coal for {name} (from validation_cases_new.json)",
            "ultimate_analysis_wt_pct_dry": {
                "C": cd.get("C", 70), "H": cd.get("H", 5), "O": cd.get("O", 10),
                "N": cd.get("N", 1), "S": cd.get("S", 0.5), "Ash": cd.get("Ash", 10),
            },
        }
        oc = {
            "coal_feed_rate_g_s": default_feed_g_s,
            "coal_feed_rate_kg_hr": default_feed_kg_h,
            "O2_to_fuel_ratio": ratio_oc,
            "steam_to_fuel_ratio": 0.0,
            "pressure_atm": P_Pa / 101325.0,
            "pressure_Pa": P_Pa,
            "inlet_temperature_K": 400.0,
            "heat_loss_percent": 2.0,
            "slurry_concentration_pct": slurry_pct,
            "O2_purity": float(inp.get("O2_purity", 0.98)),
            "coal_dry": cd,
        }
        # Illinois_No6 为工业规模 (11575 g/s)，几何 L=6m D=2m
        if name == "Illinois_No6":
            oc["L_reactor_m"] = 6.0
            oc["D_reactor_m"] = 2.0
        merged["validation_cases"][name] = {
            "description": f"From validation_cases_new.json",
            "coal_type": coal_key,
            "feedstock_type": "coal_water_slurry",
            "source_file": "validation_cases_new.json",
            "operating_conditions": oc,
            "expected_results": {"target_output": rec.get("target_output", {})},
        }

    # Paper_Case_6 = Illinois_No6（同一算例，以 Illinois_No6 为准）
    if "Illinois_No6" in merged["validation_cases"]:
        illinois = merged["validation_cases"]["Illinois_No6"].copy()
        illinois["description"] = "Illinois No.6 coal (same as Illinois_No6, industrial scale)"
        illinois["source_file"] = "validation_cases.json + validation_cases_new.json (Illinois_No6)"
        merged["validation_cases"]["Paper_Case_6"] = illinois

    out_path = os.path.join(DATA, "validation_cases_merged.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    n_cases = len(merged["validation_cases"])
    n_coals = len(merged["coal_database"])
    print(f"合并完成: {out_path}")
    print(f"  算例数: {n_cases} (去重后)")
    print(f"  煤种数: {n_coals}")
    print("  算例列表:", ", ".join(sorted(merged["validation_cases"].keys())))


if __name__ == "__main__":
    main()
