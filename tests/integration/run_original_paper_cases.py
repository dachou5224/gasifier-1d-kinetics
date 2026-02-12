"""
run_original_paper_cases.py
===========================

从 `data/validation_cases_OriginalPaper.json` 加载 Wen & Chaung (1979)
的原始 Texaco / 煤水浆工况，使用当前 1D 模型计算，并与文献给出的：

  - 实验/目标值 (expected_results)
  - 原 Wen & Chaung 模型预测 (expected_results.model_predictions)

进行三方对比：
  - 出口温度 (若给出)
  - 碳转化率 (若给出)
  - 干基合成气组成 (CO, H2, CO2, CH4 如有)

运行方式 (项目根目录):

  PYTHONPATH=src python tests/integration/run_original_paper_cases.py
"""

import json
import os
import sys
from typing import Dict, Any

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

from model.gasifier_system import GasifierSystem
from model.original_paper_loader import load_original_paper_cases, get_original_paper_json_path


def _dry_mole_fraction_pct(gas_moles: np.ndarray) -> Dict[str, float]:
    """
    将 GasifierSystem 返回的气相摩尔流 (前 8 个分量) 转为干基体积分数 (%).

    约定索引: [O2, CH4, CO, CO2, H2S, H2, N2, H2O]
    """
    F_dry = float(np.sum(gas_moles[:7])) + 1e-12
    names = ["O2", "CH4", "CO", "CO2", "H2S", "H2", "N2"]
    return {nm: float(gas_moles[i]) / F_dry * 100.0 for i, nm in enumerate(names)}


def _wet_to_dry(vol_pct_wet: Dict[str, float]) -> Dict[str, float]:
    """
    将文献给出的湿基体积分数 (含 H2O) 换算为干基:
      Y_dry = Y_wet / (1 - Y_H2O/100)
    仅对除 H2O 外的组分进行缩放。
    """
    y_h2o = float(vol_pct_wet.get("H2O", 0.0))
    if y_h2o <= 0.0 or y_h2o >= 100.0:
        # 没有 H2O 或数据异常，直接按干基处理
        return dict(vol_pct_wet)

    scale = 1.0 / (1.0 - y_h2o / 100.0)
    dry = {}
    for k, v in vol_pct_wet.items():
        if k.upper() == "H2O":
            continue
        dry[k] = float(v) * scale
    return dry


def _extract_expected_dry(expected: Dict[str, Any]) -> Dict[str, float]:
    """
    从 expected_results 中提取“实验/目标值”的干基气体组成 (如有)。
    支持两种形式:
      - dry_product_gas_vol_pct: 已为干基
      - wet_product_gas_vol_pct: 含 H2O，需换算为干基
    """
    if not expected:
        return {}

    if "dry_product_gas_vol_pct" in expected:
        return {k: float(v) for k, v in expected["dry_product_gas_vol_pct"].items()}

    if "wet_product_gas_vol_pct" in expected:
        return _wet_to_dry({k: float(v) for k, v in expected["wet_product_gas_vol_pct"].items()})

    return {}


def _extract_model_pred_dry(expected: Dict[str, Any]) -> Dict[str, float]:
    """
    从 expected_results.model_predictions 中提取 Wen & Chaung 模型预测的干基组成。
    结构与 _extract_expected_dry 类似。
    """
    mp = expected.get("model_predictions", {})
    if not mp:
        return {}

    if "dry_product_gas_vol_pct" in mp:
        return {k: float(v) for k, v in mp["dry_product_gas_vol_pct"].items()}

    if "wet_product_gas_vol_pct" in mp:
        return _wet_to_dry({k: float(v) for k, v in mp["wet_product_gas_vol_pct"].items()})

    return {}


def main():
    json_path = get_original_paper_json_path()
    if not os.path.isfile(json_path):
        print("Not found:", json_path)
        return

    coal_db, cases, metadata = load_original_paper_cases(json_path)

    # 反应器尺寸: 优先使用 metadata 中的 Texaco pilot 尺寸
    geom_meta = metadata.get("reactor_dimensions", {})
    L = float(geom_meta.get("length_m", 6.096))  # 20 ft
    D = float(geom_meta.get("diameter_m", 1.524))  # 5 ft
    geometry = {"L": L, "D": D}
    n_cells = 60

    print("=" * 80)
    print("Wen & Chaung (1979) 原始工况 — 当前 1D 模型验证")
    print(f"Reactor geometry: L = {L:.3f} m, D = {D:.3f} m")
    print("=" * 80)

    rows = []

    for name, case in cases.items():
        inp = case["inputs"]
        coal_key = inp["coal"]
        coal_props = coal_db[coal_key]
        expected = case.get("expected", {})

        # Operating conditions for GasifierSystem
        coal_flow = inp["FeedRate"] / 3600.0  # kg/h -> kg/s
        o2_flow = coal_flow * inp["Ratio_OC"]
        steam_flow = coal_flow * inp["Ratio_SC"]

        op_conds = {
            "coal_flow": coal_flow,
            "o2_flow": o2_flow,
            "steam_flow": steam_flow,
            "P": inp["P"],
            "T_in": inp["TIN"],
            # 使用文献 3% 热损失进行“原始 paper 复现”
            "HeatLossPercent": inp.get("HeatLossPercent", 3.0),
            "SlurryConcentration": inp.get("SlurryConcentration", 100.0),
            "pilot_heat": 5.0e6,
        }

        try:
            system = GasifierSystem(geometry, coal_props, op_conds)
            results, z_grid = system.solve(N_cells=n_cells)
            last = results[-1]
            gas = last[:8]
            T_out_C = float(last[10] - 273.15)
            y_dry = _dry_mole_fraction_pct(gas)

            # --- 提取文献 experimental / 原模型 结果 ---
            exp_T = expected.get("outlet_temperature_C", None)
            exp_conv = expected.get("carbon_conversion_pct", None)
            exp_dry = _extract_expected_dry(expected)

            mp = expected.get("model_predictions", {})
            mp_T = mp.get("outlet_temperature_C", None)
            mp_conv = mp.get("carbon_conversion_pct", None)
            mp_dry = _extract_model_pred_dry(expected)

            # 聚合成一行，后面汇总用
            row = {
                "name": name,
                "FeedRate_g_s": coal_flow * 1000.0,
                "Ratio_OC": inp["Ratio_OC"],
                "Ratio_SC": inp["Ratio_SC"],
                "Slurry_pct": inp.get("SlurryConcentration", 100.0),
                "HeatLossPercent": inp.get("HeatLossPercent", 3.0),
                "T_exp": exp_T,
                "T_paper": mp_T,
                "T_model": T_out_C,
                "Xc_exp": exp_conv,
                "Xc_paper": mp_conv,
                "y_exp": exp_dry,
                "y_paper": mp_dry,
                "y_model": y_dry,
            }
            rows.append(row)

            # ---- 打印逐工况详细对比 ----
            print(f"\nCase: {name}")
            print(f"  Feed: {row['FeedRate_g_s']:.2f} g/s  O2/Coal={row['Ratio_OC']:.3f}  Steam/Coal={row['Ratio_SC']:.3f}  Slurry={row['Slurry_pct']:.1f}%")
            if exp_T is not None or mp_T is not None:
                print("  Outlet T (°C):", end=" ")
                if exp_T is not None:
                    print(f"Exp={exp_T:.1f}", end="  ")
                if mp_T is not None:
                    print(f"PaperModel={mp_T:.1f}", end="  ")
                print(f"PythonModel={T_out_C:.1f}")
            else:
                print(f"  Outlet T (°C): PythonModel={T_out_C:.1f}  (no explicit paper value)")

            if exp_conv is not None or mp_conv is not None:
                print("  Carbon conversion (%):", end=" ")
                if exp_conv is not None:
                    print(f"Exp={exp_conv:.2f}", end="  ")
                if mp_conv is not None:
                    print(f"PaperModel={mp_conv:.2f}", end="  ")
                # 目前 1D 模型内部尚未直接输出全局碳转化率，这里暂不重复计算
                print("")

            # 组成对比（CO, H2, CO2, CH4 如果有）
            species = ["CO", "H2", "CO2", "CH4"]
            print("  Dry syngas (vol%, dry basis):")
            header = "    " + "{:<6} {:>10} {:>12} {:>14}".format("Comp", "Exp", "PaperModel", "PythonModel")
            print(header)
            for sp in species:
                exp_val = row["y_exp"].get(sp)
                pap_val = row["y_paper"].get(sp)
                mod_val = row["y_model"].get(sp, None)
                print(
                    "    {:<6} {:>10} {:>12} {:>14}".format(
                        sp,
                        f"{exp_val:.2f}" if exp_val is not None else "-",
                        f"{pap_val:.2f}" if pap_val is not None else "-",
                        f"{mod_val:.2f}" if mod_val is not None else "-",
                    )
                )

        except Exception as e:
            print(f"\nCase: {name}  FAILED: {e}")

    # ---- 全局简要统计 ----
    print("\n" + "=" * 80)
    print("Summary (only cases with valid PythonModel results):")
    print("=" * 80)

    T_list = [r["T_model"] for r in rows if isinstance(r.get("T_model"), (int, float))]
    if T_list:
        print(
            f"  PythonModel Outlet T (°C): min={min(T_list):.0f}, max={max(T_list):.0f}, mean={np.mean(T_list):.0f}"
        )

    # 简单统计 CO/H2/CO2 范围
    for sp in ["CO", "H2", "CO2"]:
        vals = [r["y_model"].get(sp, None) for r in rows]
        vals = [v for v in vals if isinstance(v, (int, float))]
        if not vals:
            continue
        print(
            f"  PythonModel {sp} (dry, vol%): min={min(vals):.1f}, max={max(vals):.1f}, mean={np.mean(vals):.1f}"
        )


if __name__ == "__main__":
    main()

