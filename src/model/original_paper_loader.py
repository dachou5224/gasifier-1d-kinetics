"""
original_paper_loader.py
=========================

从 `data/validation_cases_OriginalPaper.json` 读取 Wen & Chaung (1979)
原始 Texaco / 煤水浆验证工况，转换为与
`validation_loader.load_validation_cases_from_json` /
`fortran_input_loader.load_fortran_cases` 兼容的结构:

  - `coal_db`:  name -> {Cd, Hd, Od, Nd, Sd, Ad, Vd, FCd, Mt, HHV_d}
  - `cases`:    name -> {
        "inputs":   {coal, FeedRate, Ratio_OC, Ratio_SC, P, TIN,
                     HeatLossPercent, SlurryConcentration},
        "expected": 原始 JSON 中的 expected_results 块（含干/湿基组成等）
    }

注意:
  * 文献中热损失为 3% (`heat_loss_percent: 3.0`)，本 loader 直接写入
    `HeatLossPercent=3.0`，便于“严格复现 Wen & Chaung 模型”。
  * 合成气组成的干/湿基信息保留在 `expected` 原始结构中，由上层
    脚本 (如 `run_original_paper_cases.py`) 统一做干湿基换算。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple


def _dulong_hhv_mj_kg(C: float, H: float, O: float, S: float) -> float:
    """
    Dulong 近似公式，和 `validation_loader` / `fortran_input_loader` 保持一致。
    输入为干基质量百分数 (wt%)。
    """
    c, h, o, s = C / 100.0, H / 100.0, O / 100.0, S / 100.0
    return 33.5 * c + 144.0 * h - 18.0 * o + 10.0 * s


def get_original_paper_json_path() -> str:
    """
    返回 `validation_cases_OriginalPaper.json` 的默认路径。
    """
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base, "data", "validation_cases_OriginalPaper.json")


def _build_coal_db(raw_coal_db: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    将 JSON 中的 `coal_database` 转为内部 coal_db 结构。

    约定:
      - 一律使用 ultimate_analysis_wt_pct_dry 中的 C,H,O,N,S,Ash
      - 挥发/固定碳 (Vd, FCd) 无法直接获得，采用与 validation_loader 一致的经验值:
          Vd 在 25–45% 之间，且保证 Vd + FCd + Ash ≈ 100。
      - 水分 Mt: 若给出 moisture_wt_pct，则直接使用，否则为 0。
      - HHV: 若给出 HHV_kJ_kg，则直接使用；否则用 Dulong 估算。
    """
    coal_db: Dict[str, Dict[str, float]] = {}

    for name, rec in raw_coal_db.items():
        ua = rec.get("ultimate_analysis_wt_pct_dry", {})
        C = float(ua.get("C", 70.0))
        H = float(ua.get("H", 5.0))
        O = float(ua.get("O", 10.0))
        N = float(ua.get("N", 1.0))
        S = float(ua.get("S", 0.5))
        Ash = float(ua.get("Ash", 10.0))

        # Mt: as-received 总水分 (wt%)
        Mt = float(rec.get("moisture_wt_pct", 0.0))

        # Vd/FCd: 参考 validation_loader 的经验公式
        Vd = min(45.0, max(25.0, 100.0 - Ash - 35.0))
        FCd = max(0.0, 100.0 - Ash - Vd)

        # HHV: 优先采用 JSON 中给出的 HHV_kJ_kg，否则用 Dulong 估算
        if "HHV_kJ_kg" in rec:
            HHV_d_kJ_kg = float(rec["HHV_kJ_kg"])
        else:
            HHV_d_kJ_kg = _dulong_hhv_mj_kg(C, H, O, S) * 1000.0

        coal_db[name] = {
            "Cd": C,
            "Hd": H,
            "Od": O,
            "Nd": N,
            "Sd": S,
            "Ad": Ash,
            "Vd": Vd,
            "FCd": FCd,
            "Mt": Mt,
            "HHV_d": HHV_d_kJ_kg,
        }

    return coal_db


def _atm_to_pa(p_atm: float) -> float:
    """简易 atm → Pa 转换。"""
    return float(p_atm) * 101325.0


def _build_cases(
    raw_cases: Dict[str, Any],
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """
    将 JSON 中的 validation_cases 转为 cases 结构。

    每个 case:
      cases[name] = {
        "coal": coal_type,          # 便于上层追溯
        "feedstock_type": str,
        "inputs": {
            "coal": coal_type,      # 与 coal_db key 对应
            "FeedRate": ... kg/h,
            "Ratio_OC": ...,
            "Ratio_SC": ...,
            "P": ... Pa,
            "TIN": ... K,
            "HeatLossPercent": ...,
            "SlurryConcentration": ... (%),
        },
        "expected": expected_results 原始块,
      }
    """
    cases: Dict[str, Dict[str, Any]] = {}

    # 全局默认压力 (若单个 case 未给出)
    default_p_atm = 24.0
    if metadata:
        default_p_atm = float(metadata.get("operating_pressure_atm", default_p_atm))

    for name, rec in raw_cases.items():
        coal_type = rec.get("coal_type")
        feedstock_type = rec.get("feedstock_type", "").lower()
        oc = rec.get("operating_conditions", {})
        expected = rec.get("expected_results", {})

        # 1) 煤质 / 质量基准
        coal_feed_g_s = float(oc.get("coal_feed_rate_g_s", 0.0))
        coal_feed_kg_h = coal_feed_g_s * 3.6  # kg/h，与其它 loader 保持一致

        # 2) O2 / Steam / Water 比例
        # Texaco 残渣工况: O2_to_fuel_ratio / steam_to_fuel_ratio
        # 煤浆工况: O2_to_coal_ratio / water_to_coal_ratio
        if "O2_to_fuel_ratio" in oc:
            ratio_oc = float(oc.get("O2_to_fuel_ratio", 0.0))
        else:
            ratio_oc = float(oc.get("O2_to_coal_ratio", 0.0))

        if "steam_to_fuel_ratio" in oc:
            ratio_sc = float(oc.get("steam_to_fuel_ratio", 0.0))
        else:
            ratio_sc = float(oc.get("water_to_coal_ratio", 0.0))

        # 3) 压力
        if "pressure_Pa" in oc:
            P = float(oc["pressure_Pa"])
        else:
            P_atm = float(oc.get("pressure_atm", default_p_atm))
            P = _atm_to_pa(P_atm)

        # 4) 入口温度 (煤入口气温度，与 Fortran ta 对应)
        T_in = float(oc.get("inlet_temperature_K", 500.0))

        # 5) 散热损失
        heat_loss_pct = float(oc.get("heat_loss_percent", 3.0))

        # 6) 浆液浓度:
        #   - molten_residue: 视为无额外浆液水，SlurryConcentration=100
        #   - coal_water_slurry: 使用文献给出的 slurry_concentration_pct
        if "slurry_concentration_pct" in oc:
            slurry_pct = float(oc.get("slurry_concentration_pct", 100.0))
        else:
            # 熔融残渣 (Texaco) 进料近似干粉，无显式浆液水
            slurry_pct = 100.0

        inputs = {
            "coal": coal_type,
            "FeedRate": coal_feed_kg_h,
            "Ratio_OC": ratio_oc,
            "Ratio_SC": ratio_sc,
            "P": P,
            "TIN": T_in,
            "HeatLossPercent": heat_loss_pct,
            "SlurryConcentration": slurry_pct,
        }

        cases[name] = {
            "coal": coal_type,
            "feedstock_type": feedstock_type,
            "inputs": inputs,
            "expected": expected,
        }

    return cases


def load_original_paper_cases(
    path: str | None = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    加载 `validation_cases_OriginalPaper.json`，返回:

      coal_db, cases, metadata

    其中:
      - coal_db: 见模块顶部说明
      - cases:   见 `_build_cases` 说明
      - metadata: JSON 顶部的全局 `metadata` 字段
    """
    if path is None:
        path = get_original_paper_json_path()

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_coal_db = data.get("coal_database", {})
    raw_cases = data.get("validation_cases", {})
    metadata = data.get("metadata", {})

    coal_db = _build_coal_db(raw_coal_db)
    cases = _build_cases(raw_cases, metadata)

    return coal_db, cases, metadata


__all__ = [
    "get_original_paper_json_path",
    "load_original_paper_cases",
]

