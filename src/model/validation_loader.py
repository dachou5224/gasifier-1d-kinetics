# validation_loader.py
# Load validation cases from JSON (e.g. data/validation_cases_new.json) into
# the same format as chemistry.VALIDATION_CASES and COAL_DATABASE.

import json
import os
from typing import Dict, Any, Tuple


def _dulong_hhv_mj_kg(C: float, H: float, O: float, S: float) -> float:
    """Rough HHV (MJ/kg) from ultimate analysis (wt%). Dulong-style."""
    c, h, o, s = C / 100.0, H / 100.0, O / 100.0, S / 100.0
    return 33.5 * c + 144.0 * h - 18.0 * o + 10.0 * s


def load_validation_cases_from_json(
    path: str,
    default_feed_rate_kg_h: float = 41670.0,
    default_TIN_K: float = 300.0,
    default_heat_loss_pct: float = 2.0,  # 文献: 散热损失为入炉煤HHV的2%
) -> Tuple[Dict[str, Dict], Dict[str, Dict[str, Any]]]:
    """
    Load validation_cases_new.json format into (COAL_DATABASE, VALIDATION_CASES).

    JSON format:
      validation_cases[name].input: slurry_conc (0-1), O2_purity, O2_coal_ratio_daf,
        pressure, coal_dry: {C, H, O, N, S, Ash}
      validation_cases[name].target_output: CO, H2, CO2, H2O, CH4 (vol% 或 mol%)
        通常为湿基（含 H2O）。与模型干基对比时需换算：Y_dry = Y_wet / (1 - Y_H2O/100)。

    Returns:
      coal_db: name_Coal -> {Cd, Hd, Od, Nd, Sd, Ad, Mt, Vd, FCd, HHV_d}
      cases:  name -> {inputs: {coal, FeedRate, Ratio_OC, ...}, expected: {YCO, YH2, ...}}
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cases_raw = data.get("validation_cases", data)
    coal_db = {}
    cases = {}
    for name, rec in cases_raw.items():
        inp = rec.get("input", rec.get("inputs", {}))
        out = rec.get("target_output", rec.get("expected", {}))
        cd = inp.get("coal_dry", {})
        C = float(cd.get("C", 70))
        H = float(cd.get("H", 5))
        O = float(cd.get("O", 10))
        N = float(cd.get("N", 1))
        S = float(cd.get("S", 0.5))
        Ash = float(cd.get("Ash", 10))
        Vd = min(45.0, max(25.0, 100.0 - Ash - 35.0))  # typical Vd
        FCd = max(0.0, 100.0 - Ash - Vd)
        coal_key = f"{name}_Coal"
        coal_db[coal_key] = {
            "Cd": C, "Hd": H, "Od": O, "Nd": N, "Sd": S, "Ad": Ash,
            "Vd": Vd, "FCd": FCd, "Mt": 0.0,
            "HHV_d": _dulong_hhv_mj_kg(C, H, O, S) * 1000.0,  # kJ/kg
        }
        slurry = float(inp.get("slurry_conc", 0.65))
        if slurry <= 1.0:
            slurry_pct = slurry * 100.0
        else:
            slurry_pct = slurry
        cases[name] = {
            "inputs": {
                "coal": coal_key,
                "FeedRate": default_feed_rate_kg_h,
                "Ratio_OC": float(inp.get("O2_coal_ratio_daf", 0.9)),
                "Ratio_SC": 0.0,
                "P": float(inp.get("pressure", 4.08e6)),
                "TIN": default_TIN_K,
                "HeatLossPercent": default_heat_loss_pct,
                "SlurryConcentration": slurry_pct,
            },
            "expected": {
                "YCO": float(out.get("CO", 0)),
                "YH2": float(out.get("H2", 0)),
                "YCO2": float(out.get("CO2", 0)),
                "YH2O": float(out.get("H2O", 0)),
                "YCH4": float(out.get("CH4", 0)),
            },
        }
    return coal_db, cases


def get_validation_cases_new_path() -> str:
    """Default path to validation_cases_new.json (project data dir)."""
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base, "data", "validation_cases_new.json")


def get_project_data_dir() -> str:
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base, "data")


def get_validation_cases_final_path() -> str:
    return os.path.join(get_project_data_dir(), "validation_cases_final.json")


def get_validation_cases_legacy_path() -> str:
    return os.path.join(get_project_data_dir(), "validation_cases.json")


def get_validation_cases_pilot_archive_path() -> str:
    return os.path.join(get_project_data_dir(), "archive", "validation_cases_pilot.json")


# 已合并到 validation_cases_final.json 的大写 canonical 名；小写 pilot 键仅作别名（文件内已删重复行）
CASE_NAME_ALIASES_FINAL: Dict[str, str] = {
    "texaco i-1": "Texaco_I-1",
    "texaco i-2": "Texaco_I-2",
    "texaco i-5c": "Texaco_I-5C",
    "texaco i-10": "Texaco_I-10",
    "texaco exxon": "Texaco_Exxon",
}


def iter_validation_cases_final(data: Dict[str, Any]):
    """Yield `(name, case_data)` pairs from normalized `validation_cases_final.json`."""
    for cap, by_feed in data.items():
        if cap == "metadata" or not isinstance(by_feed, dict):
            continue
        for _feed, cases in by_feed.items():
            if not isinstance(cases, dict):
                continue
            for name, case_data in cases.items():
                yield name, case_data


def normalize_final_case_to_legacy(case_name: str, case_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert one normalized final-case record into the legacy `{inputs, expected}` shape
    used by older tests and scripts.
    """
    cond = case_data.get("operating_conditions", {})
    expected_results = case_data.get("expected_results", {})
    dry = expected_results.get("dry_product_gas_vol_pct", {})

    feed_rate_kg_h = cond.get("coal_feed_rate_kg_hr")
    if feed_rate_kg_h is None and cond.get("coal_feed_rate_g_s") is not None:
        feed_rate_kg_h = float(cond["coal_feed_rate_g_s"]) * 3.6

    ratio_oc = cond.get("O2_to_coal_ratio", cond.get("O2_to_fuel_ratio"))
    ratio_sc = cond.get("water_to_coal_ratio", cond.get("steam_to_fuel_ratio", 0.0))
    pressure_pa = cond.get("pressure_Pa", cond.get("P"))
    t_in_k = cond.get("inlet_temperature_K", cond.get("TIN"))
    heat_loss = cond.get("heat_loss_percent", cond.get("HeatLossPercent", 1.0))
    slurry_pct = cond.get("slurry_concentration_pct", cond.get("SlurryConcentration", 100.0))
    coal_type = case_data.get("coal_type")
    if not coal_type:
        from model.chemistry import CASE_TO_COAL_MAP

        coal_type = CASE_TO_COAL_MAP.get(case_name)

    return {
        "name": case_name,
        "coal_type": coal_type,
        "inputs": {
            "coal": coal_type,
            "FeedRate_kg_h": feed_rate_kg_h,
            "FeedRate": feed_rate_kg_h,
            "Ratio_OC": ratio_oc,
            "O2_to_fuel_ratio": ratio_oc,
            "SteamRatio_SC": ratio_sc,
            "Ratio_SC": ratio_sc,
            "P_Pa": pressure_pa,
            "P": pressure_pa,
            "T_in_K": t_in_k,
            "TIN": t_in_k,
            "HeatLossPercent": heat_loss,
            "SlurryConcentration": slurry_pct,
            "Voidage": cond.get("Voidage", 1.0),
            "L_reactor": cond.get("L_reactor_m", cond.get("L_reactor", 6.0)),
            "D_reactor": cond.get("D_reactor_m", cond.get("D_reactor", 2.0)),
        },
        "expected": {
            "TOUT_C": expected_results.get("outlet_temperature_C"),
            "YCO": dry.get("CO"),
            "YH2": dry.get("H2"),
            "YCO2": dry.get("CO2"),
        },
        "expected_results": expected_results,
        "capacity": case_data.get("capacity"),
        "feed_type": case_data.get("feed_type"),
    }


def load_case_from_repo(case_name: str) -> Dict[str, Any]:
    """
    Load one validation case from repository data sources.

    Priority:
    1. Legacy `data/validation_cases.json` if present.
    2. Normalized `data/validation_cases_final.json`（benchmark；小写 pilot 别名见 CASE_NAME_ALIASES_FINAL）。
    3. `chemistry.VALIDATION_CASES` fallback.
    """
    legacy_path = get_validation_cases_legacy_path()
    if os.path.exists(legacy_path):
        with open(legacy_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        if "cases" in config and case_name in config["cases"]:
            return config["cases"][case_name]

    # 小写 pilot 名不是 canonical 等价物；优先回到 archive 中保留原始 Fortran 口径。
    pilot_archive_path = get_validation_cases_pilot_archive_path()
    if case_name in CASE_NAME_ALIASES_FINAL and os.path.exists(pilot_archive_path):
        with open(pilot_archive_path, "r", encoding="utf-8") as f:
            pilot_data = json.load(f)
        fortran_cases = pilot_data.get("fortran_cases", {})
        if case_name in fortran_cases:
            case = dict(fortran_cases[case_name])
            coal_db = pilot_data.get("coal_database", {})
            coal_key = case.get("inputs", {}).get("coal")
            if coal_key in coal_db:
                case["coal_type"] = coal_key
            return case

    final_path = get_validation_cases_final_path()
    if os.path.exists(final_path):
        with open(final_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for name, case_data in iter_validation_cases_final(data):
            if name == case_name:
                return normalize_final_case_to_legacy(case_name, case_data)

    from model.chemistry import VALIDATION_CASES

    if case_name in VALIDATION_CASES:
        return VALIDATION_CASES[case_name]
    raise KeyError(f"Case '{case_name}' not found in repository validation data")
