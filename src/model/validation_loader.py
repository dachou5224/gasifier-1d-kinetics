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
