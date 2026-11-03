#!/usr/bin/env python3
"""P1.7-A 热值口径、锚点热解热与协调煤量下的新基线（2026-11-03 卡）。

执行顺序（卡第 1 → 5 节）：
  1  热值口径（LHV_d）+ 热解热模型（from_heating_value / zero）  → unit_checks.csv
  2  锚点模式 F = P1.6-A 模式 F + PyrolysisHeatModel="zero"       → anchor_criteria.csv / anchor_attribution.csv
  3  H_in_with_moisture 只计 2% 输煤水分                          → unit_checks.csv 的 hydrogen_budget 段
  4  新基线：9 窗 × 3 煤量口径 × N=40 + 主口径 N=80               → baseline.csv / baseline_cells.csv /
                                                                   post_reactor_closure.csv
  5  预声明判据 G1–G4                                              → gates.csv

第 1 节任一单元级检查不通过即停止（不进入第 2、4 节），失败细节写进 interpretation.md。

用法：
  .venv/bin/python scripts/generate_p1_7a_20261103.py [--stage unit|all]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model.fortran_anchor_20261030 import (  # noqa: E402
    DRY_POWDER_CASES,
    anchor_cases,
    parse_fortran_log,
    solve_fortran_case,
)
from model.gasifier_system import GasifierSystem, compute_hf_coal_j_kg  # noqa: E402
from model.input_contract import (  # noqa: E402
    PYROLYSIS_HEAT_MODEL_DEFAULT,
    PYROLYSIS_HEAT_MODEL_ZERO,
    coal_hhv_kj_kg,
    coal_lhv_kj_kg,
)
from model.material import SPECIES_NAMES  # noqa: E402
from model.model_case_adapter import (  # noqa: E402
    TRANSPORTED_COAL_MOISTURE_PERCENT,
    build_gasifier_inputs_from_model_case,
    wet_coal_flow_kg_s,
)
from model.p0_round_20260930 import ELEMENT_KG_PER_MOL, m1_reference_window_ids  # noqa: E402
from model.p1_5_round_20261030 import (  # noqa: E402
    base_protocol,
    run_case,
    silence_solver_logs,
    write_round_outputs,
)
from model.physics import get_enthalpy_molar  # noqa: E402
from model.pyrolysis_service import PyrolysisService  # noqa: E402
from model.revision_evidence_common import (  # noqa: E402
    load_frozen_split,
    source_commit,
    write_sha256_manifest,
)
from model.source_terms import PyrolysisSource  # noqa: E402

OUT_DIR = ROOT / "outputs" / "diagnostics" / "round-p1.7a-20261103"
P17D_DIR = Path("/Users/liuzhen/AI-projects/gasifier-data-prep/outputs/diagnostics/p1.7d-20261103")
P15C_DIR = ROOT / "outputs" / "diagnostics" / "round-p1.5c-wen-anchor-20261030"
P16A_DIR = ROOT / "outputs" / "diagnostics" / "round-p1.6a-baseline-20261031"
M1_MATRIX = ROOT / "outputs" / "diagnostics" / "revision-evidence-20260826" / "M1-cell0" / "revision_m1_cell0_variant_matrix.csv"

N_BASE = 40
N_RECHECK = 80
N_ANCHOR = (40, 80, 160)
N_ANCHOR_REF = 160

ANCHOR_TOL = {
    "X_C_pp": 2.0,
    "CO_vol_pct": 2.0,
    "H2_vol_pct": 2.0,
    "CO2_vol_pct": 2.0,
    "T_gas_K": 50.0,
    "O2_drop_rel": 0.20,
}

MOISTURE = TRANSPORTED_COAL_MOISTURE_PERCENT / 100.0

# 三种煤量口径：CSV 列 + 该口径对应的装置碳平衡 X_C 列（原始 / 标定合成气流量）
COAL_BASES = {
    "weigh_per_window": {
        "coal_col": "coal_dry_weigh_kg_s",
        "xc_raw_col": "xc_weigh_raw_pct",
        "xc_cal_col": "xc_weigh_cal_pct",
        "role": "对照口径（称重给煤机逐窗）",
    },
    "recon_phase": {
        "coal_col": "coal_dry_recon_phase_kg_s",
        "xc_raw_col": "xc_recon_phase_raw_pct",
        "xc_cal_col": "xc_recon_phase_cal_pct",
        "role": "主口径（协调-阶段）",
    },
    "recon_24h": {
        "coal_col": "coal_dry_recon_24h_kg_s",
        "xc_raw_col": "xc_recon_24h_raw_pct",
        "xc_cal_col": "xc_recon_24h_cal_pct",
        "role": "敏感性口径（协调-24 h；两窗 insufficient）",
    },
}
MAIN_BASIS = "recon_phase"

# P1 合同煤质（config/coal_assumption_v0.yaml），仅用于第 1 节单元检查
P1_CONTRACT_COAL = {
    "Cd": 73.01, "Hd": 4.324, "Od": 10.855, "Nd": 0.94, "Sd": 0.371,
    "Ad": 10.5, "FCd": 56.727, "Mt": TRANSPORTED_COAL_MOISTURE_PERCENT,
    "cp_char": 1300.0,
}
P1_LHV_KJ_KG = 27720.0


# ---------------------------------------------------------------------------
# 第 1 节：热值口径与热解热
# ---------------------------------------------------------------------------

def _pyrolysis_heat_kj_kg(coal_props: dict, T_pyro: float) -> dict:
    """按 PyrolysisSource 的口径算热解热 [kJ/kg 干煤] 及其三项分拆。"""
    molar_yields, w_vol = PyrolysisService().calc_yields(coal_props)
    source = PyrolysisSource(
        np.zeros(len(SPECIES_NAMES)), 0.0, target_cell_idx=0, T_pyro=float(T_pyro),
        coal_mass_flow_kg_s=1.0, coal_props=coal_props,
    )
    h_vol = sum(
        float(molar_yields[i]) * float(get_enthalpy_molar(sp, float(T_pyro)))
        for i, sp in enumerate(SPECIES_NAMES) if molar_yields[i] > 0.0
    )
    cp = float(coal_props.get("cp_char", 1300.0))
    dT = float(T_pyro) - 298.15
    m_coal = 1.0
    m_char = m_coal - float(w_vol)
    h_coal = m_coal * (cp * dT + float(coal_props["Hf_coal"]))
    h_char = m_char * cp * dT
    return {
        "H_coal_W": h_coal, "H_char_W": h_char, "H_vol_W": h_vol,
        "Q_pyrolysis_W": h_coal - h_char - h_vol,
        "Q_pyrolysis_kj_kg": (h_coal - h_char - h_vol) / 1000.0,
        "W_vol_kg_per_kg": float(w_vol),
        "volatile_source_energy_W": float(source.energy_src_W),
    }


def _hf_lhv_roundtrip_rel(hf_j_kg: float, coal_props: dict, lhv_kj_kg: float) -> float:
    """由 Hf_coal 反推低位热值，核对口径闭合（相对差）。"""
    cd = float(coal_props["Cd"]) / 100.0
    hd = float(coal_props["Hd"]) / 100.0
    sd = float(coal_props["Sd"]) / 100.0
    comb = (
        (cd / 0.012011) * (-393.51)
        + (hd / 0.001008) * 0.5 * (-241.83)
        + (sd / 0.03206) * (-296.81)
    )
    lhv_back = hf_j_kg / 1000.0 - comb
    return abs(lhv_back - lhv_kj_kg) / abs(lhv_kj_kg)


def unit_checks(by_id: dict, window_ids: list[str], p17d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    ctx: dict = {}

    # ---- 1.1 Hf_coal 与热解热：LHV 口径 ----
    props_lhv = {**P1_CONTRACT_COAL, "LHV_d": P1_LHV_KJ_KG}
    hf_lhv, model_lhv = compute_hf_coal_j_kg(props_lhv, {"T_in": 298.15})
    props_lhv["Hf_coal"] = hf_lhv
    pyro_lhv = _pyrolysis_heat_kj_kg(props_lhv, 298.15)
    ctx["hf_lhv_kj_kg"] = hf_lhv / 1000.0
    ctx["q_pyro_lhv_kj_kg"] = pyro_lhv["Q_pyrolysis_kj_kg"]
    rows.append({
        "check_group": "heating_value_contract", "context": "P1 合同煤质", "metric": "Hf_coal_lhv_kj_kg",
        "value": hf_lhv / 1000.0, "reference": -1421.1, "tolerance": 1.0,
        "pass": bool(abs(hf_lhv / 1000.0 + 1421.1) <= 1.0),
        "note": f"LHV_d={P1_LHV_KJ_KG:.0f} kJ/kg 直接进公式，不再扣 9·Hd·2442；模型标记 {model_lhv}",
    })
    rows.append({
        "check_group": "heating_value_contract", "context": "P1 合同煤质", "metric": "pyrolysis_heat_lhv_kj_kg",
        "value": pyro_lhv["Q_pyrolysis_kj_kg"], "reference": -161.8, "tolerance": 2.0,
        "pass": bool(abs(pyro_lhv["Q_pyrolysis_kj_kg"] + 161.8) <= 2.0),
        "note": "PyrolysisService 产率 + PyrolysisSource 能量口径，在 298.15 K 求值（<0 吸热）",
    })

    # ---- 1.2 旧口径复现（HHV 当高位） ----
    props_hhv = {**P1_CONTRACT_COAL, "HHV_d": P1_LHV_KJ_KG}
    hf_hhv, _ = compute_hf_coal_j_kg(props_hhv, {"T_in": 298.15})
    props_hhv["Hf_coal"] = hf_hhv
    pyro_hhv = _pyrolysis_heat_kj_kg(props_hhv, 298.15)
    ctx["hf_hhv_kj_kg"] = hf_hhv / 1000.0
    ctx["q_pyro_hhv_kj_kg"] = pyro_hhv["Q_pyrolysis_kj_kg"]
    rows.append({
        "check_group": "heating_value_contract", "context": "P1 合同煤质", "metric": "Hf_coal_legacy_hhv_kj_kg",
        "value": hf_hhv / 1000.0, "reference": -2371.4, "tolerance": 1.0,
        "pass": bool(abs(hf_hhv / 1000.0 + 2371.4) <= 1.0),
        "note": "只给 HHV_d 时保持修改前行为（按高位处理、扣潜热）——旧口径复现",
    })
    rows.append({
        "check_group": "heating_value_contract", "context": "P1 合同煤质", "metric": "pyrolysis_heat_legacy_hhv_kj_kg",
        "value": pyro_hhv["Q_pyrolysis_kj_kg"], "reference": -1112.1, "tolerance": 2.0,
        "pass": bool(abs(pyro_hhv["Q_pyrolysis_kj_kg"] + 1112.1) <= 2.0),
        "note": "旧口径在 298.15 K 的热解热；与 LHV 口径相差约 950 kJ/kg 干煤（潜热扣两次）",
    })

    # ---- 1.3 两者都给必须报错 ----
    raised = False
    try:
        compute_hf_coal_j_kg({**P1_CONTRACT_COAL, "LHV_d": P1_LHV_KJ_KG, "HHV_d": 28670.0}, {"T_in": 298.15})
    except ValueError:
        raised = True
    rows.append({
        "check_group": "heating_value_contract", "context": "LHV_d 与 HHV_d 同时给出", "metric": "raises_value_error",
        "value": float(raised), "reference": 1.0, "tolerance": 0.0, "pass": bool(raised),
        "note": "不做二选一猜测：低位与高位口径不同（Hf_coal 用低位、热损基准用高位）",
    })

    # ---- 1.4 热损基准的高位：只给低位时按定义推 ----
    hhv_derived = coal_hhv_kj_kg(props_lhv)
    hd_frac = P1_CONTRACT_COAL["Hd"] / 100.0
    expected_hhv = P1_LHV_KJ_KG + 9.0 * hd_frac * 2442.0
    ctx["hhv_derived_kj_kg"] = hhv_derived
    rows.append({
        "check_group": "heating_value_contract", "context": "P1 合同煤质", "metric": "hhv_derived_kj_kg",
        "value": hhv_derived, "reference": expected_hhv, "tolerance": 1e-9,
        "pass": bool(abs(hhv_derived - expected_hhv) <= 1e-9 * expected_hhv),
        "note": "热损基准用高位；只给 LHV_d 时按定义加回 9·Hd·2442，不引入新假设",
    })

    # ---- 1.5 Hf_coal → LHV 闭合 ----
    lhv_back_rel = _hf_lhv_roundtrip_rel(hf_lhv, props_lhv, P1_LHV_KJ_KG)
    rows.append({
        "check_group": "heating_value_contract", "context": "P1 合同煤质", "metric": "lhv_roundtrip_rel",
        "value": lhv_back_rel, "reference": 0.0, "tolerance": 1e-9, "pass": bool(lhv_back_rel <= 1e-9),
        "note": "由 Hf_coal 减去元素→氧化物生成焓反推低位热值，应与 LHV_d 一致",
    })

    # ---- 1.6 PyrolysisHeatModel="zero"：与热值无关且热解热恰为 0 ----
    zero_a, _ = compute_hf_coal_j_kg({**P1_CONTRACT_COAL, "LHV_d": P1_LHV_KJ_KG},
                                    {"T_in": 500.0, "PyrolysisHeatModel": PYROLYSIS_HEAT_MODEL_ZERO})
    zero_b, _ = compute_hf_coal_j_kg({**P1_CONTRACT_COAL, "HHV_d": 34000.0},
                                     {"T_in": 500.0, "PyrolysisHeatModel": PYROLYSIS_HEAT_MODEL_ZERO})
    ctx["hf_zero_kj_kg"] = zero_a / 1000.0
    rows.append({
        "check_group": "heating_value_contract", "context": "P1 合同煤质（T_in=500 K）", "metric": "zero_model_hf_hv_independent_rel",
        "value": abs(zero_a - zero_b) / max(abs(zero_a), 1e-12), "reference": 0.0, "tolerance": 1e-12,
        "pass": bool(abs(zero_a - zero_b) <= 1e-9),
        "note": f"Hf_coal(zero)={zero_a/1000:.2f} kJ/kg；LHV_d 与 HHV_d=34 两条输入给出同一值",
    })
    props_zero = {**P1_CONTRACT_COAL, "Hf_coal": zero_a, "cp_char": 1300.0}
    pyro_zero = _pyrolysis_heat_kj_kg(props_zero, 500.0)
    ctx["q_pyro_zero_kj_kg"] = pyro_zero["Q_pyrolysis_kj_kg"]
    rows.append({
        "check_group": "heating_value_contract", "context": "P1 合同煤质（T_in=500 K）", "metric": "pyrolysis_heat_zero_kj_kg",
        "value": pyro_zero["Q_pyrolysis_kj_kg"], "reference": 0.0, "tolerance": 1e-6,
        "pass": bool(abs(pyro_zero["Q_pyrolysis_kj_kg"]) <= 1e-6),
        "note": "zero 口径下 H_coal = H_char + H_vol，热解热在定义上恰为 0",
    })

    # ---- 1.7 H_in_with_moisture 只计 2% 输煤水分 ----
    h_in = 1.0 * (P1_CONTRACT_COAL["Hd"] / 100.0) / ELEMENT_KG_PER_MOL["H"]
    w_water = 1.0 * MOISTURE / (1.0 - MOISTURE)
    h_water = 2.0 * w_water / 0.018015
    ratio = (h_in + h_water) / h_in
    ctx["h_in_moisture_ratio"] = ratio
    rows.append({
        "check_group": "hydrogen_budget", "context": "P1 合同煤质", "metric": "H_in_with_moisture_over_H_in",
        "value": ratio, "reference": 1.053, "tolerance": 0.002,
        "pass": bool(abs(ratio - 1.053) <= 0.002),
        "note": "只计 2% 输煤水分（H2O → 2 H，÷0.018015 kg/mol），不计原煤收到基水分 M_ar；"
                "修改前把水的质量直接除以 H 的摩尔质量（÷0.001008），放大约 18 倍",
    })
    # 旧实现（把水的质量当氢的质量）作对照
    ratio_legacy = (h_in + w_water / ELEMENT_KG_PER_MOL["H"]) / h_in
    ctx["h_in_moisture_ratio_legacy"] = ratio_legacy
    rows.append({
        "check_group": "hydrogen_budget", "context": "P1 合同煤质", "metric": "H_in_with_moisture_over_H_in_legacy_bug",
        "value": ratio_legacy, "reference": float("nan"), "tolerance": float("nan"), "pass": None,
        "note": "修改前列的公式（水质量 ÷ H 摩尔质量）在 2%、Hd=4.324% 下给出该值；P1.6-A 的 1.44–1.47 "
                "即由此而来，不是 M_ar（M_ar=15.77% 会给到 5.3）",
    })

    # ---- 1.8 修正前后热解热在 P0-A 窗口上的 W 量级（信息行） ----
    for wid in window_ids:
        row = by_id[wid]
        rec = p17d[p17d["window_id"] == wid].iloc[0]
        built = build_gasifier_inputs_from_model_case(row, coal_flow_dry_kg_s=float(rec["coal_dry_recon_phase_kg_s"]))
        cp = {k: v for k, v in built["coal_props"].items() if k != "LHV_d"}
        cp["LHV_d"] = built["coal_props"]["LHV_d"]
        T_in = float(built["op_conds"]["T_in"])
        m_dry = float(rec["coal_dry_recon_phase_kg_s"])
        hf_new, _ = compute_hf_coal_j_kg(cp, {"T_in": T_in})
        hf_old, _ = compute_hf_coal_j_kg({**{k: v for k, v in cp.items() if k != "LHV_d"}, "HHV_d": cp["LHV_d"]}, {"T_in": T_in})
        hf_zero, _ = compute_hf_coal_j_kg(cp, {"T_in": T_in, "PyrolysisHeatModel": PYROLYSIS_HEAT_MODEL_ZERO})
        for tag, hf in (("legacy_hhv", hf_old), ("lhv_fixed", hf_new), ("zero_model", hf_zero)):
            q = _pyrolysis_heat_kj_kg({**cp, "Hf_coal": hf}, T_in)
            rows.append({
                "check_group": "pyrolysis_heat_scale",
                "context": f"{rec['phase']} {wid.split('__')[-2][:30]}",
                "metric": f"Q_pyrolysis_{tag}_MW",
                "value": q["Q_pyrolysis_W"] * m_dry / 1e6,
                "reference": q["Q_pyrolysis_kj_kg"], "tolerance": float("nan"), "pass": None,
                "note": f"cell 0 热解热 [MW]（干煤 {m_dry:.4f} kg/s，T_in={T_in:.1f} K）；"
                        f"reference 列是比热解热 [kJ/kg 干煤]；Hf_coal={hf/1e6:.4f} MJ/kg",
            })
    return pd.DataFrame(rows), ctx


# ---------------------------------------------------------------------------
# 第 2 节：锚点（模式 F = P1.6-A 模式 F + PyrolysisHeatModel="zero"）
# ---------------------------------------------------------------------------

def _strip(rec: dict) -> dict:
    for key in ("_profile", "_z", "_system"):
        rec.pop(key, None)
    return rec


def match_fortran(cases: list[dict], fortran_out: list[dict]) -> dict[str, dict]:
    out = {}
    for case in cases:
        name = case["case_name"]
        hits = [f for f in fortran_out if f["case_name"].startswith(name[:12])]
        if not hits:
            continue
        out[name] = hits[0]
    return out


def anchor_outlet_rows(cases, geometry, fortran_by_case) -> pd.DataFrame:
    rows = []
    for case in cases:
        name = case["case_name"]
        for mode in ("F", "M"):
            for n_cells in N_ANCHOR:
                rec = solve_fortran_case(case, n_cells=n_cells, mode=mode, geometry=geometry)
                _strip(rec)
                rec["is_dry_powder"] = bool(case["is_dry_powder"])
                f = fortran_by_case.get(name, {})
                rec["fortran_T_gas_K"] = float(f.get("T_gas_K", np.nan))
                rec["fortran_X_C_pct"] = float(f.get("X_C_pct", np.nan))
                rows.append(rec)
    return pd.DataFrame(rows)


def anchor_criteria_rows(out: pd.DataFrame, fortran_by_case) -> pd.DataFrame:
    rows = []
    ref = out[out["N_cells"] == N_ANCHOR_REF]
    for name, f in fortran_by_case.items():
        if name not in DRY_POWDER_CASES:
            continue
        for mode in ("F", "M"):
            hit = ref[(ref["case_name"] == name) & (ref["mode"] == mode)]
            if hit.empty:
                continue
            r = hit.iloc[0]
            checks = {
                "X_C_pp": (r.get("X_C_pct"), f["X_C_pct"]),
                "CO_vol_pct": (r.get("CO_vol_pct"), f["CO_vol_pct"]),
                "H2_vol_pct": (r.get("H2_vol_pct"), f["H2_vol_pct"]),
                "CO2_vol_pct": (r.get("CO2_vol_pct"), f["CO2_vol_pct"]),
                "T_gas_K": (r.get("T_gas_K"), f["T_gas_K"]),
                "O2_drop_rel": (r.get("O2_drop_to_1pct_z_m"), f.get("O2_drop_to_1pct_z_m")),
            }
            for metric, (pv, fv) in checks.items():
                pv = float(pv) if pv is not None else float("nan")
                fv = float(fv) if fv is not None else float("nan")
                evaluable = metric != "O2_drop_rel"
                diff = abs(pv - fv)
                note = ""
                if metric == "O2_drop_rel":
                    diff = diff / max(abs(fv), 1e-9)
                    note = ("not evaluable: Fortran 只在收敛后打印少数气室，第一次打印处 O2 已为 0；"
                            "Python 在 cell 0 即降到入口 1% 以下")
                rows.append({
                    "case_name": name, "mode": mode, "N_cells": N_ANCHOR_REF, "metric": metric,
                    "python_value": pv, "fortran_value": fv, "abs_diff": diff,
                    "tolerance": ANCHOR_TOL[metric], "evaluable": bool(evaluable),
                    "pass": (bool(np.isfinite(diff) and diff <= ANCHOR_TOL[metric])
                             if evaluable else None),
                    "note": note,
                })
    return pd.DataFrame(rows)


def anchor_attribution_rows(cases, geometry, fortran_by_case) -> pd.DataFrame:
    toggles = [
        ("WGS_Keq_Model_new_vs_legacy_fit", {"WGS_Keq_Model": "legacy_fit"}),
        ("SolidEnthalpy_1a_on_vs_off", {"SolidEnthalpyModel": "legacy_coal_hf"}),
        ("UseFortranDiffusion", {"UseFortranDiffusion": True}),
        ("WGS_RatFactor", {"WGS_RatFactor": True}),
        ("CharCombustionRateFactor", {"CharCombustionRateFactor": 1.0}),
        # P1.7-A 新增：热解热口径
        ("PyrolysisHeatModel_zero_vs_from_heating_value",
         {"PyrolysisHeatModel": PYROLYSIS_HEAT_MODEL_ZERO}),
    ]
    rows = []
    for case in cases:
        if not case["is_dry_powder"]:
            continue
        name = case["case_name"]
        base = _strip(solve_fortran_case(case, n_cells=N_ANCHOR_REF, mode="M", geometry=geometry))
        for item, switches in toggles:
            alt = _strip(solve_fortran_case(
                case, n_cells=N_ANCHOR_REF, mode="M", geometry=geometry, switches=switches))
            rows.append({
                "case_name": name, "mode": "M", "N_cells": N_ANCHOR_REF, "item": item,
                "d_X_C_pp": alt.get("X_C_pct", np.nan) - base.get("X_C_pct", np.nan),
                "d_CO_vol_pct": alt.get("CO_vol_pct", np.nan) - base.get("CO_vol_pct", np.nan),
                "d_H2_vol_pct": alt.get("H2_vol_pct", np.nan) - base.get("H2_vol_pct", np.nan),
                "d_CO2_vol_pct": alt.get("CO2_vol_pct", np.nan) - base.get("CO2_vol_pct", np.nan),
                "d_T_gas_K": alt.get("T_gas_K", np.nan) - base.get("T_gas_K", np.nan),
                "alt_execution_status": alt.get("execution_status"),
                "fortran_T_gas_K": float(fortran_by_case.get(name, {}).get("T_gas_K", np.nan)),
                "base_T_gas_K": base.get("T_gas_K", np.nan),
                "base_X_C_pct": base.get("X_C_pct", np.nan),
                "fortran_X_C_pct": float(fortran_by_case.get(name, {}).get("X_C_pct", np.nan)),
            })

    # 修订第 4 条 + P1.7-A：I-1 的模式 F 热值无关性（HHV_d 30 → 34）
    i1 = next(c for c in cases if c["case_name"] == "texaco i-1")
    f_ref = float(fortran_by_case.get("texaco i-1", {}).get("T_gas_K", np.nan))
    f_xc = float(fortran_by_case.get("texaco i-1", {}).get("X_C_pct", np.nan))
    variants = [
        ("I-1 模式 F zero，HHV_d=30（热损基准占位，基准行）", {}, {}),
        ("I-1 模式 F zero，HHV_d=34", {"HHV_d": 34000.0}, {}),
        ("I-1 模式 F zero，HHV_d=33.74（组分关联式）", {"HHV_d": 33740.0}, {}),
        ("I-1 模式 F zero，HHV_d=30 + HeatLossPercent=0", {}, {"HeatLossPercent": 0.0}),
        ("I-1 模式 F zero，HHV_d=34 + HeatLossPercent=0", {"HHV_d": 34000.0}, {"HeatLossPercent": 0.0}),
        ("I-1 模式 F from_heating_value（HHV_d=30，旧 Hf_coal 口径）", {},
         {"PyrolysisHeatModel": PYROLYSIS_HEAT_MODEL_DEFAULT}),
    ]
    base_alt = None
    for tag, coal_ov, switches in variants:
        alt = _strip(solve_fortran_case(
            i1, n_cells=N_ANCHOR_REF, mode="F", geometry=geometry,
            coal_overrides=coal_ov or None, switches=switches or None))
        if base_alt is None:
            base_alt = alt
        rows.append({
            "case_name": "texaco i-1", "mode": "F", "N_cells": N_ANCHOR_REF, "item": tag,
            "d_X_C_pp": alt.get("X_C_pct", np.nan) - float(base_alt.get("X_C_pct", np.nan)),
            "d_CO_vol_pct": alt.get("CO_vol_pct", np.nan) - float(base_alt.get("CO_vol_pct", np.nan)),
            "d_H2_vol_pct": alt.get("H2_vol_pct", np.nan) - float(base_alt.get("H2_vol_pct", np.nan)),
            "d_CO2_vol_pct": alt.get("CO2_vol_pct", np.nan) - float(base_alt.get("CO2_vol_pct", np.nan)),
            "d_T_gas_K": alt.get("T_gas_K", np.nan) - float(base_alt.get("T_gas_K", np.nan)),
            "alt_execution_status": alt.get("execution_status"),
            "fortran_T_gas_K": f_ref,
            "base_T_gas_K": alt.get("T_gas_K", np.nan),
            "base_X_C_pct": alt.get("X_C_pct", np.nan),
            "fortran_X_C_pct": f_xc,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 第 4 节：新基线
# ---------------------------------------------------------------------------

def _limit_zone_stats(system: GasifierSystem) -> dict:
    comb = [d for d in system.cell_diagnostics if d.get("is_combustion_zone")]
    gasif = [d for d in system.cell_diagnostics if not d.get("is_combustion_zone")]
    dz_comb = sum(float(d["dz_m"]) for d in comb)
    dz_gas = sum(float(d["dz_m"]) for d in gasif)
    bind_comb = sum(float(d["dz_m"]) for d in comb if d.get("WGS_NegativeTruncated"))
    bind_gas = sum(float(d["dz_m"]) for d in gasif if d.get("WGS_NegativeTruncated"))
    n_bind = sum(1 for d in system.cell_diagnostics if d.get("WGS_NegativeTruncated"))
    return {
        "wgs_neg_bind_cells": n_bind,
        "wgs_neg_bind_len_frac_combustion": bind_comb / dz_comb if dz_comb > 0 else float("nan"),
        "wgs_neg_bind_len_frac_gasification": bind_gas / dz_gas if dz_gas > 0 else float("nan"),
        "wgs_neg_bind_len_frac_overall": (bind_comb + bind_gas) / max(dz_comb + dz_gas, 1e-12),
        "dz_combustion_m": dz_comb,
        "dz_gasification_m": dz_gas,
    }


def _hydrogen_budget(system: GasifierSystem, gas: np.ndarray, dry_kg_s: float) -> dict:
    """氢预算：进料氢 = 干基 Hd；含水分口径只计合同 2% 输煤水分（不含 M_ar）。"""
    hd = float(system.coal_props.get("Hd", 0.0)) / 100.0
    h_in = dry_kg_s * hd / ELEMENT_KG_PER_MOL["H"]
    water_kg_s = dry_kg_s * MOISTURE / (1.0 - MOISTURE)
    h_moist = h_in + 2.0 * water_kg_s / 0.018015
    h_out = 2.0 * float(gas[5]) + 4.0 * float(gas[1]) + 2.0 * float(gas[4])
    return {
        "H_in_mol_s": h_in,
        "H_in_with_moisture_mol_s": h_moist,
        "H_out_dry_mol_s": h_out,
        "H_out_over_H_in": h_out / max(h_in, 1e-12),
        "H_out_over_H_in_with_moisture": h_out / max(h_moist, 1e-12),
        "H_in_with_moisture_over_H_in": h_moist / max(h_in, 1e-12),
        "moisture_water_kg_s": water_kg_s,
    }


def baseline_stage(window_ids, by_id, p17d: pd.DataFrame):
    rows, cells = [], []
    skipped: list[dict] = []
    m = p17d.set_index("window_id")
    plan = [(wid, basis, N_BASE) for wid in window_ids for basis in COAL_BASES]
    plan += [(wid, MAIN_BASIS, N_RECHECK) for wid in window_ids]
    for wid, basis, n_cells in plan:
        spec = COAL_BASES[basis]
        rec_csv = m.loc[wid]
        dry = float(rec_csv[spec["coal_col"]])
        row = by_id[wid]
        if not np.isfinite(dry):
            # 24 h 口径两窗 insufficient：该口径没有值，按卡不缩短时段、不填充，直接不产行。
            skipped.append({
                "window_id": wid, "coal_basis": basis, "N_cells": n_cells,
                "reason": f"r_24h_status={rec_csv.get('r_24h_status', 'no_value')}",
            })
            continue
        rec = run_case(row, n_cells=n_cells, coal_flow_dry_kg_s=dry, keep_system=True)
        system: GasifierSystem = rec.pop("_system")
        rec.pop("_profile", None)
        rec.update(_limit_zone_stats(system))
        rec.update({
            "coal_basis": basis,
            "coal_basis_role": spec["role"],
            "coal_dry_kg_s": dry,
            "coal_wet_kg_s": wet_coal_flow_kg_s(dry),
            "coal_flow_op_kg_s": float(system.op_conds["coal_flow"]),
            "coal_dry_effective_kg_s": float(system.W_dry),
            "Hf_coal_j_kg": float(system.coal_props["Hf_coal"]),
            "Hf_coal_kj_kg": float(system.coal_props["Hf_coal"]) / 1000.0,
            "hf_coal_source": system.hf_coal_source,
            "pyrolysis_heat_model": system.pyrolysis_heat_model,
            "LHV_d_kj_kg": float(coal_lhv_kj_kg(system.coal_props)),
            "HHV_d_derived_kj_kg": float(coal_hhv_kj_kg(system.coal_props)),
            "plant_xc_carbon_balance_raw_pct": float(rec_csv[spec["xc_raw_col"]]),
            "plant_xc_carbon_balance_cal_pct": float(rec_csv[spec["xc_cal_col"]]),
            "plant_dry_flow_raw_nm3_h": float(rec_csv["syngas_dry_raw_nm3_h"]),
            "plant_dry_flow_cal_nm3_h": float(rec_csv["syngas_dry_cal_nm3_h"]),
            "pipeline_over_weigh": float(rec_csv["pipeline_over_weigh"]),
            "liu2023_xc_phase_pct": float(rec_csv["liu2023_xc_phase_pct"]),
            "r_phase": float(rec_csv["r_phase"]),
            "s_phase": float(rec_csv["s_phase"]),
            "co2_carrier_carbon_j_kg_h": float(rec_csv["co2_carrier_carbon_j_kg_h"]),
            "dry_syngas_carbon_raw_kg_h": float(rec_csv["dry_syngas_carbon_raw_kg_h"]),
        })
        # 装置侧绝对流量（原始 / 标定）
        plant_raw_mol = float(rec_csv["syngas_dry_raw_nm3_h"]) / 22.414 / 3600.0 * 1000.0
        plant_cal_mol = float(rec_csv["syngas_dry_cal_nm3_h"]) / 22.414 / 3600.0 * 1000.0
        y_h2 = float(rec["plant_H2_vol_pct"]) / 100.0
        rec.update({
            "plant_dry_total_mol_s_raw": plant_raw_mol,
            "plant_dry_total_mol_s_cal": plant_cal_mol,
            "H_plant_mol_s_raw": 2.0 * plant_raw_mol * y_h2,
            "H_plant_mol_s_cal": 2.0 * plant_cal_mol * y_h2,
        })
        if rec["execution_status"] == "completed" and system.results:
            state = system.results[-1]
            gas = np.asarray(state.gas_moles, dtype=float)
            total = float(np.sum(gas))
            dry_gas = float(np.sum(gas[:7]))
            for idx, name in ((2, "CO"), (3, "CO2"), (5, "H2"), (1, "CH4"), (4, "H2S"),
                              (6, "N2"), (0, "O2"), (7, "H2O")):
                rec[f"F_{name}_mol_s"] = float(gas[idx])
            y = gas / max(total, 1e-12)
            q_wgs = (y[3] * y[5]) / max(y[2] * y[7], 1e-30)
            from model.kinetics import calculate_wgs_equilibrium
            k_thermo = float(calculate_wgs_equilibrium(float(state.T), "thermo"))
            hb = _hydrogen_budget(system, gas, dry)
            pyro = system._pyrolysis_source()
            rec.update({
                "dry_total_mol_s": dry_gas,
                "Q_WGS": float(q_wgs), "K_thermo_T_out": k_thermo, "Q_over_K": float(q_wgs / k_thermo),
                "pyrolysis_heat_cell0_W": float(pyro.pyrolysis_heat_W),
                "pyrolysis_heat_cell0_kj_kg": float(pyro.pyrolysis_heat_W) / max(dry, 1e-12) / 1000.0,
                "CO2_gap_vs_plant_vol_pct": float(gas[3] / max(dry_gas, 1e-12) * 100.0) - float(rec["plant_CO2_vol_pct"]),
                "dry_flow_ratio_raw": float(rec["dry_flow_nm3_h"]) / max(float(rec_csv["syngas_dry_raw_nm3_h"]), 1e-12),
                "dry_flow_ratio_cal": float(rec["dry_flow_nm3_h"]) / max(float(rec_csv["syngas_dry_cal_nm3_h"]), 1e-12),
                **hb,
            })
            rec["H_plant_over_H_in_raw"] = rec["H_plant_mol_s_raw"] / max(hb["H_in_mol_s"], 1e-12)
            rec["H_plant_over_H_in_cal"] = rec["H_plant_mol_s_cal"] / max(hb["H_in_mol_s"], 1e-12)
            rec["H_plant_over_H_in_with_moisture_raw"] = rec["H_plant_mol_s_raw"] / max(hb["H_in_with_moisture_mol_s"], 1e-12)
            rec["H_plant_over_H_in_with_moisture_cal"] = rec["H_plant_mol_s_cal"] / max(hb["H_in_with_moisture_mol_s"], 1e-12)
            rec["model_minus_plant_xc_main_cal_pp"] = float(rec["X_C_implied"]) * 100.0 - float(rec_csv[COAL_BASES[MAIN_BASIS]["xc_cal_col"]])
        else:
            rec.update({
                "dry_total_mol_s": np.nan, "pyrolysis_heat_cell0_W": np.nan,
                "pyrolysis_heat_cell0_kj_kg": np.nan, "CO2_gap_vs_plant_vol_pct": np.nan,
                "dry_flow_ratio_raw": np.nan, "dry_flow_ratio_cal": np.nan,
                "H_in_mol_s": np.nan, "H_in_with_moisture_mol_s": np.nan,
                "H_out_dry_mol_s": np.nan, "H_out_over_H_in": np.nan,
                "H_out_over_H_in_with_moisture": np.nan,
                "H_in_with_moisture_over_H_in": np.nan,
                "H_plant_over_H_in_raw": np.nan, "H_plant_over_H_in_cal": np.nan,
                "H_plant_over_H_in_with_moisture_raw": np.nan,
                "H_plant_over_H_in_with_moisture_cal": np.nan,
                "model_minus_plant_xc_main_cal_pp": np.nan,
            })
        rows.append(rec)
        for diag in system.cell_diagnostics:
            d = dict(diag)
            d.update({"window_id": wid, "coal_basis": basis, "N_cells": n_cells})
            cells.append(d)
    return pd.DataFrame(rows), pd.DataFrame(cells), skipped


# ---------------------------------------------------------------------------
# 反应器后单参数闭合
# ---------------------------------------------------------------------------

def closure_rows(base: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in base.iterrows():
        if r["execution_status"] != "completed":
            rows.append({
                "window_id": r["window_id"], "coal_basis": r["coal_basis"], "N_cells": r["N_cells"],
                "phase": r.get("phase"), "execution_status": r["execution_status"],
                "xi_card": np.nan, "pred_CO_vol_pct": np.nan, "pred_H2_vol_pct": np.nan,
                "plant_CO_vol_pct": r.get("plant_CO_vol_pct", np.nan),
                "plant_H2_vol_pct": r.get("plant_H2_vol_pct", np.nan),
                "d_CO": np.nan, "d_H2": np.nan,
                "pipeline_over_weigh": r.get("pipeline_over_weigh", np.nan),
            })
            continue
        dry = float(r["dry_total_mol_s"])
        f_co2, f_co, f_h2 = float(r["F_CO2_mol_s"]), float(r["F_CO_mol_s"]), float(r["F_H2_mol_s"])
        y_co2 = float(r["plant_CO2_vol_pct"]) / 100.0
        # 卡面口径：(CO2 + ξ)/(干气 + ξ) = 装置 CO2 体积分数（干气总量增加 ξ，2 个干气摩尔/ξ）
        xi = (y_co2 * dry - f_co2) / (1.0 - y_co2)
        denom = dry + xi
        pred_co = (f_co - xi) / denom * 100.0
        pred_h2 = (f_h2 + xi) / denom * 100.0
        rows.append({
            "window_id": r["window_id"], "coal_basis": r["coal_basis"], "N_cells": r["N_cells"],
            "phase": r.get("phase"), "execution_status": r["execution_status"],
            "dry_total_mol_s": dry, "F_CO2_mol_s": f_co2, "F_CO_mol_s": f_co, "F_H2_mol_s": f_h2,
            "plant_CO2_vol_pct": r["plant_CO2_vol_pct"],
            "xi_card": xi, "pred_CO_vol_pct": pred_co, "plant_CO_vol_pct": r["plant_CO_vol_pct"],
            "d_CO": pred_co - float(r["plant_CO_vol_pct"]),
            "pred_H2_vol_pct": pred_h2, "plant_H2_vol_pct": r["plant_H2_vol_pct"],
            "d_H2": pred_h2 - float(r["plant_H2_vol_pct"]),
            "d_CO2": float(r["CO2_vol_pct"]) - float(r["plant_CO2_vol_pct"]),
            "pipeline_over_weigh": r.get("pipeline_over_weigh", np.nan),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 判据
# ---------------------------------------------------------------------------

def gates_stage(base: pd.DataFrame, clos: pd.DataFrame, p17d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    extras: dict = {}
    main40 = base[(base["coal_basis"] == MAIN_BASIS) & (base["N_cells"] == N_BASE)
                  & (base["execution_status"] == "completed")]
    main_clos = clos[(clos["coal_basis"] == MAIN_BASIS) & (clos["N_cells"] == N_BASE)
                     & (clos["execution_status"] == "completed")]

    # G1：CO 残差与 pipeline_over_weigh 的 Spearman
    pair = main_clos[["window_id", "d_CO", "pipeline_over_weigh"]].dropna()
    from scipy import stats as _stats
    rho, pval = _stats.spearmanr(pair["d_CO"], pair["pipeline_over_weigh"])
    pearson = float(np.corrcoef(pair["d_CO"], pair["pipeline_over_weigh"])[0, 1])
    extras["g1_pair"] = pair.to_dict(orient="records")
    rows.append({
        "gate": "G1", "description": "单参数闭合的 CO 残差与 pipeline_over_weigh 的 Spearman |ρ| < 0.5",
        "metric": "spearman_rho", "value": float(rho), "threshold": 0.5,
        "pass": bool(abs(float(rho)) < 0.5),
        "detail": f"n={len(pair)}；Spearman ρ={rho:+.3f}（p={pval:.3f}）；Pearson r={pearson:+.3f}（附报，不作判据）",
    })

    # G2：CO、H2 残差中位绝对值
    med_co = float(main_clos["d_CO"].abs().median())
    med_h2 = float(main_clos["d_H2"].abs().median())
    rows.append({
        "gate": "G2a", "description": "CO 残差中位绝对值 ≤ 1.0 vol%",
        "metric": "median_abs_d_CO", "value": med_co, "threshold": 1.0,
        "pass": bool(med_co <= 1.0),
        "detail": f"n={len(main_clos)}；中位带符号 {float(main_clos['d_CO'].median()):+.2f}；"
                  f"最大绝对 {float(main_clos['d_CO'].abs().max()):.2f}",
    })
    rows.append({
        "gate": "G2b", "description": "H2 残差中位绝对值 ≤ 1.0 vol%",
        "metric": "median_abs_d_H2", "value": med_h2, "threshold": 1.0,
        "pass": bool(med_h2 <= 1.0),
        "detail": f"n={len(main_clos)}；中位带符号 {float(main_clos['d_H2'].median()):+.2f}；"
                  f"最大绝对 {float(main_clos['d_H2'].abs().max()):.2f}",
    })

    # G3：各阶段模型 X_C 中位 vs 刘文阶段值
    phase_map = {"phase1": "P1", "phase2": "P2", "phase3": "P3", "phase4": "P4"}
    m = p17d.set_index("window_id")
    liu = {}
    for _, r in base[base["coal_basis"] == MAIN_BASIS].iterrows():
        liu[r["phase"]] = float(m.loc[r["window_id"], "liu2023_xc_phase_pct"])
    g3_rows = []
    for ph, grp in main40.groupby("phase"):
        med = float(grp["X_C_implied"].median()) * 100.0
        ref = liu.get(ph, float("nan"))
        diff = med - ref
        g3_rows.append((ph, len(grp), med, ref, diff))
        rows.append({
            "gate": "G3", "description": f"{ph} 模型 X_C 中位与刘文阶段值之差 ≤ 3 pp",
            "metric": "median_X_C_minus_liu_pp", "value": float(diff), "threshold": 3.0,
            "pass": bool(abs(diff) <= 3.0),
            "detail": f"n={len(grp)}；模型中位 {med:.2f}%、刘文 {ref:.2f}%",
        })
    extras["g3_rows"] = g3_rows

    # G4：装置干气氢 / 进料氢（含 2% 水分、协调煤量）≤ 1
    g4_base = main40.dropna(subset=["H_plant_over_H_in_with_moisture_cal"])
    over = g4_base[g4_base["H_plant_over_H_in_with_moisture_cal"] > 1.0]
    rows.append({
        "gate": "G4", "description": "装置干气氢（标定流量）/ 进料氢（含 2% 水分、协调煤量）各窗 ≤ 1",
        "metric": "n_windows_over_1", "value": float(len(over)), "threshold": 0.0,
        "pass": bool(len(over) == 0),
        "detail": "全部 ≤ 1" if over.empty else "超 1 的窗口：" + "、".join(
            f"{w.split('__')[-2][:28]}({v:.3f})" for w, v in
            zip(over["window_id"], over["H_plant_over_H_in_with_moisture_cal"])),
    })
    rows.append({
        "gate": "G4_raw", "description": "同上，但用原始合成气流量口径（并列报告，不单独设闸门）",
        "metric": "n_windows_over_1_raw", "value": float((g4_base["H_plant_over_H_in_with_moisture_raw"] > 1.0).sum()),
        "threshold": float("nan"), "pass": None,
        "detail": "原始流量口径下超 1 的窗口数；逐窗数值见 baseline.csv 的 H_plant_over_H_in_with_moisture_raw",
    })

    # 仅报告：模型 X_C 与装置碳平衡 X_C（主口径）之差
    for _, r in main40.iterrows():
        rows.append({
            "gate": "report_model_vs_plant_XC", "description": "模型 X_C − 装置碳平衡 X_C（主口径：协调-阶段 + 标定流量）",
            "metric": "window", "value": float(r["model_minus_plant_xc_main_cal_pp"]),
            "threshold": float("nan"), "pass": None,
            "detail": f"{r['window_id'].split('__')[-2][:34]}；模型 {float(r['X_C_implied'])*100:.2f}%，"
                      f"装置 {float(r['plant_xc_carbon_balance_cal_pct']):.2f}%",
        })
    return pd.DataFrame(rows), extras


# ---------------------------------------------------------------------------
# 载气口径敏感度（J 随煤量同比缩放 R_phase）
# ---------------------------------------------------------------------------

def carrier_sensitivity(p17d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in p17d.iterrows():
        c_coal = float(r["coal_dry_recon_phase_kg_s"]) * float(r["w_c"]) * 3600.0
        c_syn = float(r["dry_syngas_carbon_raw_kg_h"])
        c_car = float(r["co2_carrier_carbon_j_kg_h"])
        xc_raw = (c_syn - c_car) / c_coal * 100.0
        xc_scaled = (c_syn - c_car * float(r["r_phase"])) / c_coal * 100.0
        rows.append({
            "window_id": r["window_id"], "phase": r["phase"],
            "xc_recon_phase_raw_pct_csv": float(r["xc_recon_phase_raw_pct"]),
            "xc_recon_phase_raw_pct_recomputed": xc_raw,
            "recompute_max_abs_diff_pp": abs(xc_raw - float(r["xc_recon_phase_raw_pct"])),
            "xc_j_scaled_by_r_phase_pct": xc_scaled,
            "delta_pp": xc_scaled - xc_raw,
            "r_phase": float(r["r_phase"]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# interpretation.md
# ---------------------------------------------------------------------------

def _fmt(x, nd=2):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(v):
        return "nan"
    return f"{v:.{nd}f}"


def _table(df: pd.DataFrame, cols: list[str], fmt: dict | None = None) -> str:
    fmt = fmt or {}
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in ("window_id", "case_name", "item", "execution_status", "phase", "reaction", "metric",
                     "coal_basis", "gate", "description", "detail"):
                s = str(v)
                if c == "window_id" and "__" in s:
                    s = f"`{s.split('__')[-2][:30]}…{s[-4:]}`"
                cells.append(s)
            elif isinstance(v, (bool, np.bool_)):
                cells.append(str(bool(v)))
            elif isinstance(v, (int, np.integer)):
                cells.append(str(int(v)))
            else:
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    cells.append(str(v))
                else:
                    cells.append(f"{fv:.{fmt.get(c, 2)}f}" if np.isfinite(fv) else "nan")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _unit_only_interpretation(unit_df, ctx, unit_pass, stage) -> str:
    lines = ["# P1.7-A 热值口径 —— interpretation（第 1 节闸门，未通过）\n"]
    lines.append(f"第 1 节单元级检查整体：**{'通过' if unit_pass else '未通过'}**。"
                 "按卡停止条件，未进入第 2、4 节。\n")
    lines.append("| 组 | 上下文 | 指标 | 值 | 参考 | 容差 | 通过 |")
    lines.append("|---|---|---|---:|---:|---:|---|")
    for _, r in unit_df.iterrows():
        lines.append(f"| {r['check_group']} | {r['context']} | {r['metric']} | "
                     f"{_fmt(r['value'], 6)} | {_fmt(r['reference'], 6)} | {_fmt(r['tolerance'], 0)} | {r['pass']} |")
    lines.append("")
    return "\n".join(lines)


def _git_status(root: Path | None = None) -> str:
    """`git status --porcelain`，排除本轮产物目录本身（它正在被写）。"""
    import subprocess
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(root or ROOT), text=True
        ).strip()
    except Exception as exc:  # noqa: BLE001
        return f"unavailable ({type(exc).__name__})"
    keep = [ln for ln in out.splitlines() if "round-p1.7a-20261103" not in ln]
    return "\n".join(keep) if keep else "clean"


def full_interpretation(unit_df, ctx, anchor_crit, anchor_attr, base, clos, gates, carrier,
                        window_ids, protocol) -> str:
    L: list[str] = []
    L.append("# P1.7-A 热值口径、锚点热解热与协调煤量下的新基线 —— interpretation\n")
    L.append(
        "本轮在模型仓按 2026-11-03 卡执行：①热值口径（合同 `heating_value_dry_kj_kg` 是干燥基**低位**，"
        "修改前被当高位并在 `Hf_coal` 里再扣一次汽化潜热）；②锚点模式 F 改成“热解热 = 0”"
        "（`PyrolysisHeatModel=\"zero\"`）；③`H_in_with_moisture_mol_s` 只计 2% 输煤水分；"
        "④在 P1.7-D 的三种煤量口径下重跑 P0-A 九窗基线。冻结项一项未动：全部 Arrhenius 常数、`phi`、"
        "`f_co2 = 1.0`、0.05 atm、`CharCombustionRateFactor = 0.3`、氧化因子、热解产率、氢气限幅、"
        "`USE_RK_GILL_COMBUSTION = False`、`fortran_constant`、蒸汽进料 = 0。负向 WGS 截断保留并照常统计绑定。\n"
    )
    L.append("## 0. 提交与工作树\n")
    L.append(f"- 生成本轮产物的**代码提交**（写入 `protocol.json` 的 `source_commit`）：`{protocol.get('source_commit')}`。")
    L.append("- 产物随后以一次只含 `outputs/diagnostics/round-p1.7a-20261103/` 与脚本的提交落盘；"
             "两个哈希不能相同（`protocol.json` 要写入那个哈希，互为自指），与 2026-10-30 起各轮的约定一致。")
    L.append(f"- 生成时的 `git status --porcelain`（已排除本轮产物目录本身）：`{_git_status()}`。"
             "`clean` 表示源码与测试都没有未提交改动；若此处出现改动，说明产物不是在干净的代码树上生成的。\n")

    # ---------------- 1 ----------------
    L.append("## 1. 第 1 节的单元检查\n")
    hv = unit_df[unit_df["check_group"] == "heating_value_contract"]
    hb = unit_df[unit_df["check_group"] == "hydrogen_budget"]
    L.append("| 组 | 上下文 | 指标 | 值 | 参考 | 容差 | 通过 |")
    L.append("|---|---|---|---:|---:|---:|---|")
    for _, r in pd.concat([hv, hb]).iterrows():
        L.append(f"| {r['check_group']} | {r['context']} | `{r['metric']}` | {_fmt(r['value'], 6)} | "
                 f"{_fmt(r['reference'], 6)} | {_fmt(r['tolerance'], 0)} | {r['pass']} |")
    L.append("")
    L.append(f"- `Hf_coal`（P1 合同煤质，`LHV_d = 27720` kJ/kg）：**{ctx['hf_lhv_kj_kg']:.1f} kJ/kg**"
             f"（目标 −1421.1 ±1）；旧口径（`HHV_d = 27720` 当高位、再扣潜热）**{ctx['hf_hhv_kj_kg']:.1f}**"
             "（−2371.4 ±1）——两者相差 950.3 kJ/kg 干煤，正是被扣两次的汽化潜热 `9·Hd·2442`。")
    L.append(f"- 298.15 K 的热解热：修正后 **{ctx['q_pyro_lhv_kj_kg']:.1f} kJ/kg**（目标 −161.8 ±2），"
             f"修正前 **{ctx['q_pyro_hhv_kj_kg']:.1f}**（−1112.1 ±2）。改正后属“略吸热”的正常量级。")
    L.append(f"- 热损基准：只给 `LHV_d` 时按定义推 `HHV_d = {ctx['hhv_derived_kj_kg']:.1f} kJ/kg`"
             "（+950.3），因此 2% 热损的绝对功率不变；若误用低位作热损基准，热损会小 0.068 个百分点的煤功率。")
    L.append(f"- `zero` 口径：`Hf_coal = {ctx['hf_zero_kj_kg']:.1f} kJ/kg`（T_in = 500 K），"
             f"cell 0 热解热 **{ctx['q_pyro_zero_kj_kg']:.2e} kJ/kg**（定义上恰为 0）；"
             "同一 T_in 下 `LHV_d = 27720` 与 `HHV_d = 34 MJ/kg` 给出逐位相同的 `Hf_coal`。")
    L.append(f"- `H_in_with_moisture/H_in` = **{ctx['h_in_moisture_ratio']:.4f}**（目标 ≈1.053）。"
             f"修改前的列在同一输入下是 **{ctx['h_in_moisture_ratio_legacy']:.4f}**。\n")

    L.append("### 1.1 修正前后热解热在 P0-A 窗口上的 W 量级\n")
    scale = unit_df[(unit_df["check_group"] == "pyrolysis_heat_scale")]
    if len(scale):
        piv = scale.pivot_table(index="context", columns="metric", values="value", aggfunc="first")
        cols = [c for c in ("Q_pyrolysis_legacy_hhv_MW", "Q_pyrolysis_lhv_fixed_MW", "Q_pyrolysis_zero_model_MW")
                if c in piv.columns]
        L.append("| 窗口 | " + " | ".join(c.replace("Q_pyrolysis_", "").replace("_MW", " (MW)") for c in cols) + " |")
        L.append("|---" * (len(cols) + 1) + "|")
        for idx, row in piv.iterrows():
            L.append(f"| {idx} | " + " | ".join(_fmt(row[c], 2) if np.isfinite(row[c]) else "nan" for c in cols) + " |")
        L.append("")
        if len(cols) == 3:
            d = (piv[cols[1]] - piv[cols[0]])
            L.append(f"换到低位口径让 cell 0 的热解热从 **{piv[cols[0]].min():.1f} – {piv[cols[0]].max():.1f} MW** "
                     f"变成 **{piv[cols[1]].min():.1f} – {piv[cols[1]].max():.1f} MW**"
                     f"（吸热量减少 {d.min():.1f} – {d.max():.1f} MW）。`zero` 口径恒为 0 MW。"
                     "这些是入口温度下的值；298.15 K 的值见上一节的 −161.8 / −1112.1 kJ/kg。\n")

    # ---------------- 2 ----------------
    L.append("## 2. 锚点：模式 F = P1.6-A 模式 F + `PyrolysisHeatModel=\"zero\"`\n")
    f_df = anchor_crit[(anchor_crit["mode"] == "F") & (anchor_crit["evaluable"].astype(bool))]
    n_pass = int(f_df.groupby("case_name")["pass"].all().sum()) if len(f_df) else 0
    n_cases = int(f_df["case_name"].nunique()) if len(f_df) else 0
    L.append(f"5 个干粉算例中，模式 F 的**全部可评估判据**都通过的：**{n_pass}/{n_cases}**"
             "（判据与 P1.5-C 相同：X_C 2 pp、CO/H2/CO2 2 vol%、T 50 K，N=160）。\n")
    L.append("| 算例 | 判据 | Python | Fortran | 绝对差 | 容差 | 通过 |")
    L.append("|---|---|---:|---:|---:|---:|---|")
    for _, r in f_df.iterrows():
        L.append(f"| {r['case_name']} | {r['metric']} | {_fmt(r['python_value'], 2)} | {_fmt(r['fortran_value'], 2)} | "
                 f"{_fmt(r['abs_diff'], 2)} | {r['tolerance']} | {r['pass']} |")
    L.append("")
    for name, grp in f_df.groupby("case_name"):
        bad = grp[~grp["pass"].astype(bool)]
        ok = len(grp) - len(bad)
        L.append(f"- `{name}`：{ok}/{len(grp)} 通过" + ("（全部通过）" if bad.empty else
                 "；未通过 " + "、".join(bad["metric"])))
    m_df = anchor_crit[(anchor_crit["mode"] == "M") & (anchor_crit["evaluable"].astype(bool))]
    if len(m_df):
        m_pass = int(m_df.groupby("case_name")["pass"].all().sum())
        L.append(f"\n对照：模式 M（`fortran_constant` + `thermo` + 1a + 2，`PyrolysisHeatModel` 取缺省）"
                 f"**{m_pass}/{int(m_df['case_name'].nunique())}** 通过，逐行见 `anchor_criteria.csv` 的 mode=M 行。"
                 "`O2_drop_rel` 仍不可评估（Fortran 只在收敛后打印少数气室），未计入通过数。\n")

    L.append("### 2.1 `HHV_d` 无关性检查（I-1，模式 F，N=160）\n")
    hhv_dt = float("nan")
    hhv_dt_noloss = float("nan")
    i1 = anchor_attr[(anchor_attr["case_name"] == "texaco i-1") & (anchor_attr["item"].str.startswith("I-1 "))]
    if len(i1):
        L.append("| 口径 | T (K) | ΔT vs 基准 (K) | X_C (%) |")
        L.append("|---|---:|---:|---:|")
        baseT = float(i1[i1["item"].str.contains("基准行", regex=False)]["base_T_gas_K"].iloc[0])
        for _, r in i1.iterrows():
            L.append(f"| {r['item']} | {_fmt(r['base_T_gas_K'], 1)} | "
                     f"{_fmt(float(r['base_T_gas_K']) - baseT, 2)} | {_fmt(r['base_X_C_pct'], 2)} |")
        L.append("")
        h30 = float(i1[i1["item"].str.contains("HHV_d=30（热损", regex=False)]["base_T_gas_K"].iloc[0])
        h34 = float(i1[i1["item"].str.contains("HHV_d=34", regex=False)
                       & ~i1["item"].str.contains("HeatLoss", regex=False)]["base_T_gas_K"].iloc[0])
        h30n = float(i1[i1["item"].str.contains("HHV_d=30 + HeatLoss", regex=False)]["base_T_gas_K"].iloc[0])
        h34n = float(i1[i1["item"].str.contains("HHV_d=34 + HeatLoss", regex=False)]["base_T_gas_K"].iloc[0])
        hhv_dt = h34 - h30
        hhv_dt_noloss = h34n - h30n
        L.append(f"- 默认 2% 热损下，`HHV_d` 30 → 34 的出口温度变化 **{h34 - h30:+.2f} K**"
                 f"（判据 < 1 K → **{'通过' if abs(h34 - h30) < 1.0 else '未通过'}**）。")
        L.append(f"- 把热损关掉后同一对比是 **{h34n - h30n:+.2e} K**（逐位为 0）："
                 "残余的热值依赖**全部**来自热损基准 `2% × HHV_d`，与 `Hf_coal`／热解热无关。")
        L.append("- 也就是说：`zero` 口径下 `Hf_coal` 已完全不读热值，但热损项仍按设计读高位热值；"
                 "这一项要不要也去掉热值（改成按反应热比例）属于 Fortran 兼容口径的另一个决定，本卡未授权，未改。\n")

    L.append("### 2.2 逐项归因（模式 M，N=160，一次性切换）\n")
    attr_m = anchor_attr[anchor_attr["mode"] == "M"]
    pivot = attr_m.pivot_table(index="case_name", columns="item", values="d_T_gas_K", aggfunc="first")
    items = [c for c in ["SolidEnthalpy_1a_on_vs_off", "WGS_Keq_Model_new_vs_legacy_fit",
                         "PyrolysisHeatModel_zero_vs_from_heating_value",
                         "UseFortranDiffusion", "WGS_RatFactor", "CharCombustionRateFactor"] if c in pivot.columns]
    L.append("| 算例 | " + " | ".join(items) + " |")
    L.append("|---" * (len(items) + 1) + "|")
    for case, row in pivot.iterrows():
        L.append(f"| {case} | " + " | ".join(
            f"{float(row[c]):+.1f} K" if np.isfinite(float(row[c])) else "nan" for c in items) + " |")
    L.append("")
    dpi = attr_m[attr_m["item"] == "PyrolysisHeatModel_zero_vs_from_heating_value"]
    if len(dpi):
        L.append("新增行 `PyrolysisHeatModel`（从旧口径切到 zero）在模式 M 下的温度效应 "
                 f"{_fmt(dpi['d_T_gas_K'].min(), 1)} – {_fmt(dpi['d_T_gas_K'].max(), 1)} K、"
                 f"X_C {_fmt(dpi['d_X_C_pp'].min(), 2)} – {_fmt(dpi['d_X_C_pp'].max(), 2)} pp。"
                 "注意模式 M 的锚点煤没有真实热值（占位 `HHV_d = 30`），所以这一行同时含“热值口径”与"
                 "“热解热口径”两个因素，不能当成纯热解热效应读。\n")

    # ---------------- 2.3 与 P1.6-A 的锚点对比 ----------------
    L.append("### 2.3 与 P1.6-A（模式 F，HHV_d = 30 占位）的逐判据对比\n")
    old_crit_path = P16A_DIR / "anchor_criteria.csv"
    if old_crit_path.exists():
        old = pd.read_csv(old_crit_path)
        old_f = old[old["mode"] == "F"]
        new_f = anchor_crit[anchor_crit["mode"] == "F"]
        rows_cmp = []
        for metric in ("T_gas_K", "X_C_pp", "CO_vol_pct", "H2_vol_pct", "CO2_vol_pct"):
            o = old_f[old_f["metric"] == metric].set_index("case_name")
            n = new_f[new_f["metric"] == metric].set_index("case_name")
            for case in n.index:
                if case not in o.index:
                    continue
                rows_cmp.append({
                    "case_name": case, "metric": metric,
                    "P1.6A_python": float(o.loc[case, "python_value"]),
                    "P1.7A_python": float(n.loc[case, "python_value"]),
                    "fortran": float(n.loc[case, "fortran_value"]),
                    "P1.6A_abs_diff": float(o.loc[case, "abs_diff"]),
                    "P1.7A_abs_diff": float(n.loc[case, "abs_diff"]),
                    "P1.6A_pass": bool(o.loc[case, "pass"]),
                    "P1.7A_pass": bool(n.loc[case, "pass"]),
                })
        cmp_df = pd.DataFrame(rows_cmp)
        L.append(_table(cmp_df, ["case_name", "metric", "P1.6A_python", "P1.7A_python", "fortran",
                                 "P1.6A_abs_diff", "P1.7A_abs_diff", "P1.6A_pass", "P1.7A_pass"]))
        L.append("")
        old_pass = int(old_f[old_f["evaluable"].astype(bool)].groupby("case_name")["pass"].all().sum())
        L.append(f"通过“全部可评估判据”的算例：P1.6-A **{old_pass}/5** → 本轮 **{n_pass}/5**。"
                 "分歧要按判据看，不能只看合计：本轮 CO/CO2 在 I-1、I-2 上从差 5.2 / 5.3 vol% 收到 1.0 / 0.5 vol%，"
                 "I-10 与 exxon 的温度从差 77 / 156 K 收到 8 / 14 K；代价是 I-5c 的温度从差 27.7 K 变成 84.3 K、"
                 "I-10 的 CO 从差 0.43 变成 2.45 vol%，于是没有算例再同时满足全部判据（P1.6-A 唯一全过的是 I-5c）。\n")
        L.append("**这一节的口径要点**：本轮的模式 F 不再给占位热值，`Hf_coal` 由热解产物焓反推、"
                 "热解热恰为 0；所以两轮之间的差异不是“换了个 HHV 取值”，而是“把热值从方程里拿掉”。"
                 "P1.6-A 第 10 节记录的“HHV 决定温度偏差符号”这一结论在模式 F 上不再适用。\n")
    else:
        L.append("（未找到 P1.6-A 的 `anchor_criteria.csv`，跳过逐判据对比。）\n")

    # ---------------- 3 ----------------
    L.append("## 3. 三种煤量口径下的新基线（模式 M，`uniform_full_length`，`SolverStrictRoot=True`）\n")
    ok = base[base["execution_status"] == "completed"]
    n_rows = len(base)
    L.append(f"实际行数：**{n_rows}**（卡面按 9 窗算 9×3+9=36；`recon_24h` 有两窗 `insufficient`，"
             f"该口径只有 7 窗，故少 {36 - n_rows} 行）。未缩短时段、未填充。\n")
    L.append("| 煤量口径 | N | 完成窗数 | X_C 中位 | 干气流量比（原始） | 干气流量比（标定） | CO vol% | CO2 vol% | H2 vol% |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (basis, n), grp in ok.groupby(["coal_basis", "N_cells"]):
        comp = grp[grp["execution_status"] == "completed"]
        L.append(f"| `{basis}` | {int(n)} | {len(comp)}/{len(base[(base['coal_basis']==basis)&(base['N_cells']==n)])} | "
                 f"{_fmt(comp['X_C_implied'].median(), 4)} | {_fmt(comp['dry_flow_ratio_raw'].median(), 4)} | "
                 f"{_fmt(comp['dry_flow_ratio_cal'].median(), 4)} | {_fmt(comp['CO_vol_pct'].median())} | "
                 f"{_fmt(comp['CO2_vol_pct'].median())} | {_fmt(comp['H2_vol_pct'].median())} |")
    L.append("")
    L.append("按阶段的 X_C 中位（N=40，主口径 `recon_phase`）与刘文阶段值：\n")
    main40 = ok[(ok["coal_basis"] == MAIN_BASIS) & (ok["N_cells"] == N_BASE)]
    m = pd.read_csv(P17D_DIR / "coal_reconciled_by_window.csv").set_index("window_id")
    L.append("| 阶段 | n | 模型 X_C 中位 | 刘文 | 差 (pp) | 称重口径 X_C 中位 | 24 h 口径 X_C 中位 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for ph in ("phase1", "phase2", "phase3", "phase4"):
        grp = main40[main40["phase"] == ph]
        if grp.empty:
            continue
        liu = float(m.loc[grp["window_id"].iloc[0], "liu2023_xc_phase_pct"])
        w = ok[(ok["coal_basis"] == "weigh_per_window") & (ok["N_cells"] == N_BASE) & (ok["phase"] == ph)]
        r24 = ok[(ok["coal_basis"] == "recon_24h") & (ok["N_cells"] == N_BASE) & (ok["phase"] == ph)]
        L.append(f"| {ph} | {len(grp)} | {_fmt(grp['X_C_implied'].median()*100)} | {liu:.2f} | "
                 f"{_fmt(grp['X_C_implied'].median()*100 - liu, 2)} | "
                 f"{_fmt(w['X_C_implied'].median()*100) if len(w) else '—'} | "
                 f"{_fmt(r24['X_C_implied'].median()*100) if len(r24) else '—'} |")
    L.append("")
    noroot = base[base["execution_status"] != "completed"]
    L.append(f"`no_root` / 未完成：**{len(noroot)} / {len(base)}**"
             + ("。" if noroot.empty else "：" + "、".join(
                 f"{r['window_id'].split('__')[-2][:26]}({r['coal_basis']},N={int(r['N_cells'])},{r['execution_status']})"
                 for _, r in noroot.iterrows()) + "。")
             + "\n")
    L.append("**流量口径有三套，别混**：`dry_flow_ratio` 是 P1.6-A 沿用的列（分母是 round52 inventory 的 "
             "`syngas_flow_mean`）；`dry_flow_ratio_raw` 与 `dry_flow_ratio_cal` 的分母才是 P1.7-D 的 "
             "`syngas_dry_raw_nm3_h` 与 `syngas_dry_cal_nm3_h`。上表用的是后两列，`dry_flow_ratio` 只为与旧轮次对齐而保留。\n")
    L.append(f"**负向 WGS 截断的绑定**（冻结项，未改）：每行都只有 1 个单元绑定，燃烧区绑定长度占比 "
             f"{_fmt(float(base['wgs_neg_bind_len_frac_combustion'].median()), 3)}、气化区 "
             f"{_fmt(float(base['wgs_neg_bind_len_frac_gasification'].median()), 3)}、整炉 "
             f"{_fmt(float(base['wgs_neg_bind_len_frac_overall'].median()), 3)}（范围 "
             f"{_fmt(float(base['wgs_neg_bind_len_frac_overall'].min()), 3)}–"
             f"{_fmt(float(base['wgs_neg_bind_len_frac_overall'].max()), 3)}）。"
             "逐单元的绑定布尔量在 `baseline_cells.csv` 的 `WGS_NegativeTruncated` 列。\n")
    one = ok[ok["X_C_implied"] >= 0.9999]
    L.append(f"X_C ≥ 0.9999（完全转化上限）的行：**{len(one)}/{len(ok)}**；"
             "这些行的 X_C 不是有效预测值。逐窗清单见第 6 问。")
    L.append("")
    pinned_new = int((main40["X_C_implied"] >= 0.9999).sum())
    rest_new = main40[main40["X_C_implied"] < 0.9999]["X_C_implied"]
    old_pinned, old_rest_lo, old_rest_hi = 0, float("nan"), float("nan")
    ob0_path = P16A_DIR / "baseline_n40_n80.csv"
    if ob0_path.exists():
        ob0 = pd.read_csv(ob0_path)
        ob0 = ob0[ob0["N_cells"] == N_BASE]
        old_pinned = int((ob0["X_C_implied"] >= 0.9999).sum())
        _rest = ob0[ob0["X_C_implied"] < 0.9999]["X_C_implied"]
        if len(_rest):
            old_rest_lo, old_rest_hi = float(_rest.min()), float(_rest.max())
    L.append(f"**必须并列的一条**：本轮所有行都没有 `no_root`，但主口径 N=40 的 9 窗里有 {pinned_new} 窗的 X_C 贴住"
             f"完全转化上限，其余 {len(rest_new)} 窗在 {_fmt(rest_new.min(), 4)}–{_fmt(rest_new.max(), 4)}。"
             f"与 P1.6-A 相比（当时 9 窗里 {old_pinned} 窗贴顶、其余 {_fmt(old_rest_lo, 4)}–{_fmt(old_rest_hi, 4)}）"
             f"贴顶的窗口{'变多' if pinned_new > old_pinned else '没有变多'}了："
             "热值口径修正后入口焓高了 ~950 kJ/kg 干煤，反应器更热"
             f"（本轮主口径 T_out 中位 {_fmt(main40['T_out_K'].median(), 0)} K），碳更容易被完全消耗。"
             "因此本轮的 X_C 不能当“预测能力改善”的证据读，第 4 节的 G3 实际上是在比较"
             "“模型贴在物理上限时的取值”与刘文阶段值。\n")

    # ---------------- 4 ----------------
    L.append("## 4. 预声明判据 G1–G4（主口径 `recon_phase`，N=40）\n")
    gl = gates[gates["pass"].notna()]
    L.append("| 判据 | 指标 | 值 | 阈值 | 通过 | 说明 |")
    L.append("|---|---|---:|---:|---|---|")
    for _, r in gl.iterrows():
        L.append(f"| {r['gate']} | `{r['metric']}` | {_fmt(r['value'], 4)} | {_fmt(r['threshold'], 2)} | "
                 f"**{r['pass']}** | {r['detail']} |")
    L.append("")
    main_clos = clos[(clos["coal_basis"] == MAIN_BASIS) & (clos["N_cells"] == N_BASE)]
    L.append(f"单参数闭合 ξ（卡面分母 `干气 + ξ`）：中位 **{_fmt(main_clos['xi_card'].median(), 1)} mol/s**，"
             f"范围 {_fmt(main_clos['xi_card'].min(), 1)} – {_fmt(main_clos['xi_card'].max(), 1)} mol/s。"
             f"CO 残差中位 {_fmt(main_clos['d_CO'].median(), 2)} pp（|中位| {_fmt(main_clos['d_CO'].abs().median(), 2)}），"
             f"H2 残差中位 {_fmt(main_clos['d_H2'].median(), 2)} pp。\n")
    L.append("逐窗（主口径，N=40）：\n")
    cols = ["window_id", "xi_card", "pred_CO_vol_pct", "plant_CO_vol_pct", "d_CO",
            "pred_H2_vol_pct", "plant_H2_vol_pct", "d_H2"]
    L.append(_table(main_clos, cols, {"xi_card": 1}))
    L.append("")
    L.append("### 4.1 与 P1.6-A 的闭合残差对比\n")
    old_clos_path = P16A_DIR / "post_reactor_closure.csv"
    if old_clos_path.exists():
        oc = pd.read_csv(old_clos_path)
        oc = oc[(oc["N_cells"] == 40) & (oc["execution_status"] == "completed")]
        L.append("| 量 | P1.6-A（N=40，称重逐窗 + 旧热值口径） | P1.7-A（N=40，协调-阶段 + 低位口径） |")
        L.append("|---|---:|---:|")
        L.append(f"| ξ 中位 (mol/s) | {_fmt(oc['xi_card'].median(), 1)} | {_fmt(main_clos['xi_card'].median(), 1)} |")
        L.append(f"| CO 残差中位 (vol%) | {_fmt(oc['d_CO'].median(), 2)} | {_fmt(main_clos['d_CO'].median(), 2)} |")
        L.append(f"| \\|CO 残差\\| 中位 | {_fmt(oc['d_CO'].abs().median(), 2)} | {_fmt(main_clos['d_CO'].abs().median(), 2)} |")
        L.append(f"| H2 残差中位 (vol%) | {_fmt(oc['d_H2'].median(), 2)} | {_fmt(main_clos['d_H2'].median(), 2)} |")
        L.append(f"| \\|H2 残差\\| 中位 | {_fmt(oc['d_H2'].abs().median(), 2)} | {_fmt(main_clos['d_H2'].abs().median(), 2)} |")
        L.append("")
        L.append("两轮之间同时变了三件事（热值口径、煤量口径、氢诊断列），上表不能归因到其中任一项；"
                 "要分离需要按单因素重跑，本卡没有要求。需要指出的是：**G2 这一轮没通过，而且比 P1.6-A 更差**"
                 f"（|CO 残差| 中位 {_fmt(oc['d_CO'].abs().median(), 2)} → {_fmt(main_clos['d_CO'].abs().median(), 2)}，"
                 f"|H2 残差| 中位 {_fmt(oc['d_H2'].abs().median(), 2)} → {_fmt(main_clos['d_H2'].abs().median(), 2)}），"
                 "残差符号还整体翻转（验证文档 C1 记的 P1.6-A 是 CO +1.16 / H2 −1.35，按 CSV 复算是 +1.18 / −1.37，"
                 "两者一致到舍入）。第 4 节 G1 的 |ρ| 从 P1.6-A 的 0.817 掉到 0.083，说明“残差跟着煤量误差走”这条已经解除——"
                 "但残差本身没有变小，而是从“与煤量成正比”变成了“按阶段分组的固定偏移”（阶段 1–2 偏正、"
                 "阶段 3–4 偏负），这正是 G2 失败的形态。\n")
    L.append("### 4.2 G4 的读法\n")
    L.append(f"模型自身在含 2% 水分的进料氢上闭合（`H_out_dry/H_in_with_moisture` "
             f"{_fmt(main40['H_out_over_H_in_with_moisture'].min(), 3)} – "
             f"{_fmt(main40['H_out_over_H_in_with_moisture'].max(), 3)}），装置侧却有 4 窗 > 1。"
             "按 C4/C8 的结论，这不能当作“反应器后有变换”的证据：装置取样点在激冷之后，激冷水是另一个氢源；"
             "而且这 4 窗正好是 P1/P2 的窗口，同时也是合成气流量标定倍数 `S_phase` 最大的两个阶段。"
             "原始流量口径下也全部 ≤ 1（见 `baseline.csv` 的 `H_plant_over_H_in_with_moisture_raw`），"
             "标定后才有 4 窗超 1 —— 也就是说 G4 的失败与流量标定同源，不是氢账本本身的问题。\n")

    # ---------------- 5 ----------------
    L.append("## 5. 体积分数上的 CO2 缺口（模型 − 装置）\n")
    L.append("| 煤量口径 | N | 中位 CO2 缺口 (vol%) | 范围 |")
    L.append("|---|---:|---:|---:|")
    for (basis, n), grp in ok.groupby(["coal_basis", "N_cells"]):
        g = grp["CO2_gap_vs_plant_vol_pct"].dropna()
        L.append(f"| `{basis}` | {int(n)} | {_fmt(g.median())} | {_fmt(g.min())} – {_fmt(g.max())} |")
    L.append("")
    old_base_path = P16A_DIR / "baseline_n40_n80.csv"
    if old_base_path.exists():
        ob = pd.read_csv(old_base_path)
        ob = ob[ob["N_cells"] == 40]
        gap_old = (ob["CO2_vol_pct"] - ob["plant_CO2_vol_pct"]).dropna()
        gap_new = ok[(ok["coal_basis"] == MAIN_BASIS) & (ok["N_cells"] == N_BASE)]["CO2_gap_vs_plant_vol_pct"].median()
        L.append(f"与 P1.6-A（N=40，称重逐窗 + 旧热值口径）对比：当时模型 CO2 中位 "
                 f"{_fmt(ob['CO2_vol_pct'].median())} vol%、装置 {_fmt(ob['plant_CO2_vol_pct'].median())} vol%、"
                 f"缺口中位 {_fmt(gap_old.median())} vol%；本轮主口径缺口中位 {_fmt(gap_new)} vol%。"
                 "**缺口变大了**。方向与“热值修正让反应器更热”一致："
                 "更高温度把 WGS 推向反应物侧（CO↑、CO2↓），而装置的 CO2 水平没变。"
                 "这不是本轮可以调回来的东西——按卡要求，本轮不为闭合调参、也不改口径；"
                 "单参数闭合 ξ 相应从 45 mol/s 量级升到 105 mol/s 量级。\n")

    # ---------------- 6 ----------------
    L.append("## 6. 煤量口径从称重逐窗换成协调后，哪些窗口的 X_C 离开了 1.0\n")
    L.append("| 窗口 | 称重逐窗 X_C | 协调-阶段 X_C | 协调-24 h X_C |")
    L.append("|---|---:|---:|---:|")
    for wid in window_ids:
        row = ok[(ok["window_id"] == wid) & (ok["N_cells"] == N_BASE)]
        w = row[row["coal_basis"] == "weigh_per_window"]
        p = row[row["coal_basis"] == "recon_phase"]
        r24 = row[row["coal_basis"] == "recon_24h"]
        L.append(f"| `{wid.split('__')[-2][:32]}…{wid[-4:]}` | "
                 f"{_fmt(w['X_C_implied'].iloc[0], 4)} | {_fmt(p['X_C_implied'].iloc[0], 4)} | "
                 f"{_fmt(r24['X_C_implied'].iloc[0], 4) if len(r24) else '—'} |")
    L.append("")
    left = []
    for wid in window_ids:
        row = ok[(ok["window_id"] == wid) & (ok["N_cells"] == N_BASE)]
        w = row[row["coal_basis"] == "weigh_per_window"]["X_C_implied"]
        p = row[row["coal_basis"] == "recon_phase"]["X_C_implied"]
        if len(w) and len(p) and float(w.iloc[0]) >= 0.9999 and float(p.iloc[0]) < 0.9999:
            left.append((wid, float(w.iloc[0]), float(p.iloc[0])))
    if left:
        L.append("离开完全转化上限（称重 ≥ 0.9999、协调后 < 0.9999）的窗口：\n")
        for wid, wv, pv in left:
            L.append(f"- `{wid.split('__')[-2]}`：{wv:.4f} → **{pv:.4f}**")
    else:
        L.append("没有窗口离开完全转化上限（称重与协调两套口径下 X_C 均 ≥ 0.9999）。")
    entered = []
    for wid in window_ids:
        row = ok[(ok["window_id"] == wid) & (ok["N_cells"] == N_BASE)]
        w = row[row["coal_basis"] == "weigh_per_window"]["X_C_implied"]
        p = row[row["coal_basis"] == "recon_phase"]["X_C_implied"]
        if len(w) and len(p) and float(w.iloc[0]) < 0.9999 and float(p.iloc[0]) >= 0.9999:
            entered.append((wid, float(w.iloc[0]), float(p.iloc[0])))
    L.append("")
    if entered:
        L.append("反向：称重口径没贴顶、协调后**贴到** 1.0 的窗口（同一现象的另一面，也应并列）：\n")
        for wid, wv, pv in entered:
            L.append(f"- `{wid.split('__')[-2]}`：{wv:.4f} → **{pv:.4f}**")
        L.append("")
    both_cap = []
    both_low = []
    for wid in window_ids:
        row = ok[(ok["window_id"] == wid) & (ok["N_cells"] == N_BASE)]
        w = row[row["coal_basis"] == "weigh_per_window"]["X_C_implied"]
        p = row[row["coal_basis"] == "recon_phase"]["X_C_implied"]
        if not (len(w) and len(p)):
            continue
        wv, pv = float(w.iloc[0]), float(p.iloc[0])
        if wv >= 0.9999 and pv >= 0.9999:
            both_cap.append(wid)
        elif wv < 0.9999 and pv < 0.9999:
            both_low.append(wid)
    L.append(f"其余 {len(both_cap) + len(both_low)} 窗里，{len(both_cap)} 窗（"
             + "、".join(f"`{w.split('__')[-2]}`" for w in both_cap)
             + f"）两套口径**都**贴顶，{len(both_low)} 窗（"
             + "、".join(f"`{w.split('__')[-2]}`" for w in both_low)
             + "，均在 P4）两套口径都 < 0.9999。"
             "净效果是 4 个窗口在“贴顶集合”里互换位置，而协调煤量整体比称重高"
             "（`recon_phase/weigh` 1.02–1.15），进料更多碳 → 更容易把碳烧完，"
             "所以“换口径让 X_C 离开 1.0”这一问在本轮只在 2 个窗口上成立，方向并不一致。\n")

    # ---------------- 7 ----------------
    L.append("## 7. 其他读取 `HHV_d` 的位置，以及四个已知陷阱的处理\n")
    L.append("### 7.1 `HHV_d` 的其余读取点及取舍\n")
    L.append("修改前全仓有 7 类位置读 `HHV_d`。逐一列出并说明该用高位还是低位：\n")
    L.append("| 位置 | 用途 | 应用 | 本轮处理 |")
    L.append("|---|---|---|---|")
    L.append("| `gasifier_system.compute_hf_coal_j_kg` | `Hf_coal` | **低位** | 已改：优先 `LHV_d`，不再扣潜热；只给 `HHV_d` 时保持旧行为 |")
    L.append("| `cell.Cell._calc_energy_balance` | NumPy 热损基准 `2% × HHV` | **高位** | 改为 `coal_hhv_kj_kg()`：只给低位时按定义加回 9·Hd·2442 |")
    L.append("| `gasifier_system._solve_jax_v4` → `jax_residuals` | JAX 热损基准 | **高位** | 同上，同一个 helper |")
    L.append("| `revision_m5_n5_parity`（JAX 夹具） | 与主线一致的热损基准 | **高位** | 同上 |")
    L.append("| `fortran_anchor.build_case_inputs` 的 `HHV_d = 30` | 锚点热损基准占位 | **高位** | 保留：namelist 无热值；模式 F 改 `zero` 后它只影响热损 |")
    L.append("| `validation_loader` / `fortran_input_loader` / `original_paper_loader` 的 Dulong 式 | 由元素分析估热值 | **高位**（Dulong 给的是高位） | 不改：填 `HHV_d` 语义正确，走“只给高位”的旧换算 |")
    L.append("| `chemistry.py` / `constants.DEFAULT_HHV` / UI / 诊断脚本的 HHV 归一化 | 热损/归一化分母 | **高位** | 常量不动；诊断脚本的 `coal_props.get('HHV_d')` 改走 helper |")
    L.append("")
    L.append("**为什么 `Hf_coal` 必须用低位、热损必须用高位**：`Hf_coal` 要与产物侧的生成焓在同一基准上——"
             "模型里水以 H2O(g) 离开（Shomate 的生成焓就是气态），所以基准是低位；用高位等于把水的汽化潜热"
             "预先算进煤的化学能，然后在产物侧再放一次。热损则是文献口径“入炉煤 HHV 的 2%”，"
             "它描述的是供入化学能的 2%，与反应路径无关，故用高位。两者的差就是 950.3 kJ/kg（P1 煤），"
             "在 18–21 kg/s 下是 17–20 MW —— 与验证文档 C2 的量级一致。\n")
    L.append("### 7.2 四个已知陷阱\n")
    L.append("1. **干湿煤量**：`op_conds['coal_flow']` 是湿煤（`gasifier_system._initialize_inlet`："
             "`W_dry = W_wet·(1 − Mt/100)`）。数据侧三列煤量都是干煤，因此本轮的进料一律"
             "`wet = dry/(1 − Mt)`、`Mt = 2%`，由 `model_case_adapter.wet_coal_flow_kg_s` 显式换算，"
             "`baseline.csv` 同时给 `coal_dry_kg_s` / `coal_wet_kg_s` / `coal_dry_effective_kg_s`（应等于前者）。"
             "**旧路径确实错了**：`coal_flow_mean` 是干煤，却被直接填进湿煤槽，模型实际少算 2% 干煤。\n")
    L.append("2. **载气口径**：CSV 的 `xc_*` 列用 J 载气（四管粉煤推算）。本卡按 CSV 原样使用、**不缩放**；"
             "若把 J 随煤量同比乘 `R_phase`，装置 X_C 的变化为：\n")
    if len(carrier):
        per = carrier.groupby("phase")["delta_pp"].median()
        L.append("   | 阶段 | ΔXC (pp) |")
        L.append("   |---|---:|")
        for ph, v in per.items():
            L.append(f"   | {ph} | {v:+.2f} |")
        L.append("")
        L.append(f"   全部 9 窗 {carrier['delta_pp'].min():+.2f} – {carrier['delta_pp'].max():+.2f} pp"
                 f"（中位 {carrier['delta_pp'].median():+.2f}），即普涨约 1.6–1.8 pp，与 P1.7-D §6.4 一致。"
                 f"复算校验：按 CSV 碳项重算的 `xc_recon_phase_raw_pct` 与 CSV 的最大绝对差 "
                 f"{carrier['recompute_max_abs_diff_pp'].max():.4f} pp。\n")
    L.append(f"3. **24 h 口径只有 7 行**：窗口 1（P1 09-28 08:00）与窗口 5（P3 10-09 13:45）为 `insufficient`，"
             f"该口径 7 窗 × N=40 产出 7 行；全表 {n_rows} 行而不是 36 行。未缩短时段、未填充。\n")
    L.append("4. **窗口离散**：装置碳平衡 X_C 在“协调-阶段 + 标定流量”（主口径）下 9 窗极差 8.50 pp，"
             "而“协调-阶段 + 原始流量”下是 4.57 pp —— 因为 `S_phase` 是阶段常数，阶段 1 的 1.0877 把该阶段整体抬高。"
             "本报告凡是提到装置侧极差/离散，都注明用的是哪一套；模型侧 X_C 的离散与这两套无关。\n")

    # ---------------- 8 ----------------
    L.append("## 8. 结论、偏离与未做完的部分\n")
    L.append("### 8.1 五项硬检查的结论\n")
    L.append(f"1. 第 1 节的单元检查**全部通过**（{int(unit_df['pass'].notna().sum())} 项判定项，"
             f"{int(unit_df['pass'].sum())} 项 True）；旧口径的 −2371.4 / −1112.1 与验证文档 C2 逐位复现。")
    L.append(f"2. 锚点模式 F：逐算例结果见第 2 节，全部判据通过的算例 **{n_pass}/5**（P1.6-A 为 1/5）。"
             "CO、CO2 指标与 4 个算例的温度明显改善（I-10、exxon 分别从差 77 / 156 K 收到 8 / 14 K），"
             "但 I-5c 温度从差 27.7 K 变成 84.3 K、I-10 的 CO 从差 0.43 变成 2.45 vol%，没有算例全过。")
    L.append(f"3. `HHV_d` 30→34 的 I-1 出口温度变化 **{hhv_dt:+.2f} K**，"
             f"**{'通过' if abs(hhv_dt) < 1.0 else '未通过'}**卡面 < 1 K 的检查；"
             f"关掉热损后同一对比是 **{hhv_dt_noloss:+.2e} K**，说明残余依赖 100% 来自热损基准。")
    L.append(f"4. G1 **{'通过' if bool(gates[gates['gate']=='G1']['pass'].iloc[0]) else '未通过'}**"
             f"（|ρ| = {float(gates[gates['gate']=='G1']['value'].iloc[0]):.3f}）；"
             f"G2a/G2b **{'通过' if bool(gates[gates['gate']=='G2a']['pass'].iloc[0]) and bool(gates[gates['gate']=='G2b']['pass'].iloc[0]) else '未通过'}**"
             f"（{float(gates[gates['gate']=='G2a']['value'].iloc[0]):.2f} / "
             f"{float(gates[gates['gate']=='G2b']['value'].iloc[0]):.2f} vol%）；"
             f"G3 通过 {int(gates[gates['gate']=='G3']['pass'].astype(bool).sum())}/4 个阶段"
             f"（未通过：{'、'.join(gates[(gates['gate']=='G3') & (~gates['pass'].astype(bool))]['description'].str.split(' ').str[0])}）；"
             f"G4 **{'通过' if bool(gates[gates['gate']=='G4']['pass'].iloc[0]) else '未通过'}**"
             f"（标定流量下 {int(float(gates[gates['gate']=='G4']['value'].iloc[0]))} 窗 > 1）。")
    L.append("5. CO2 缺口（模型 − 装置）在主口径下中位 −5.51 vol%，比 P1.6-A 的 −2.2 vol% 更大。\n")
    L.append("### 8.2 与验证文档不一致、或需要更正的地方\n")
    L.append("1. **C6 的成因归错了**。C6 说 `H_in_with_moisture_mol_s` 的 1.44–1.47 对应“每 kg 干煤约 0.18 kg 水，"
             "即原煤收到基水分 M_ar”。数值上 0.17–0.18 kg/kg 是对的，但成因不是 M_ar：模型里 `Mt` 一直是 "
             "2%（`model_case_adapter` 的常量），1.44–1.47 来自**单位错误**——代码把水的质量直接除以 H 的摩尔质量"
             "（0.001008 kg/mol）而不是按 H2O→2H 换算（÷0.018015 再 ×2）。"
             "在 2%、Hd = 4.324% 下这条错误公式恰好给 1.4720，与 P1.6-A 的最大值一致；"
             "各窗 1.4417–1.4720 的差异来自四个阶段不同的 Hd（4.324 / 4.39 / 4.43 / 4.62%）。"
             "按 M_ar = 15.77% 算会得到约 5.3，与观察值不符。修正后比值 1.0528，与卡的 1.053 一致。")
    L.append("2. **P1.6-A 基线用的煤量列与本轮不同**。P1.6-A 从 `huayi_round52_window_inventory.csv` 的 "
             "`coal_flow_mean` 取煤量，而 P1.7-D 的 `coal_dry_weigh_kg_s` 在 5 个窗口上与它不同（0.1–1.2%，"
             "例如 P3 `D_015` 17.5416 对 17.5862）。本轮按卡以 P1.7-D 的 CSV 为准，因此“称重逐窗”这一列"
             "**不是** P1.6-A 那组数字的逐位复现。")
    L.append("3. **`anchor_attribution` 的 `PyrolysisHeatModel` 行含义不纯**：模式 M 的锚点煤没有真实热值"
             "（占位 `HHV_d = 30`），从 `from_heating_value` 切到 `zero` 同时改变了“热值输入”与“热解热口径”，"
             "所以 −168.7 – +106.9 K 的跨度不能当作纯热解热效应；纯效应见 2.1 节的 I-1 各行。")
    L.append("4. **本轮没有重跑 JAX 的 F1/F4 对照**（卡未要求）。`PyrolysisHeatModel` 与 `LHV_d` 都是"
             "求解前的输入换算，不进入 JAX 残差核；JAX 路径读到的仍是 `coal_props['Hf_coal']` 与 "
             "`coal_hhv_kj_kg()` 给出的 HHV，两条路径口径一致。")
    L.append("5. **仓库里有一个与本轮无关的既有测试失败**："
             "`tests/integration/test_validation_config.py::test_known_nonconvergent_case_reports_its_status` "
             "在 HEAD（`df4d3c0`，未含本轮改动）上同样失败——该算例的 cost 已是 1.7e-13，但优化器仍报 "
             "`success=False`，测试的 `max(cost) > 1e-4` 断言不再成立。本轮未改动它。")
    L.append("6. **`recon_24h` 两窗无值**（窗口 1、5 的 `r_24h_status = insufficient`），该口径只有 7 行；"
             "全表 34 行而不是 36 行，未缩短时段、未填充。")
    L.append("7. **X_C 贴顶**：主口径 4/9 窗、全部 34 行中 16 行 ≥ 0.9999，这些行的 X_C 不是有效预测值，"
             "G3 的“通过”只是在比贴顶值与刘文值。这是热值修正带来的后果，本轮没有为此加任何松弛。\n")
    L.append("### 8.3 判据的读法\n")
    L.append("G1 通过说明“残差跟着煤量误差走”这条已解除；但 G2 反而变差，残差从“与煤量成正比”变成"
             "“按阶段分组的固定偏移”（阶段 1–2 为正、3–4 为负），幅度 1.6–2.3 vol%。"
             "G4 的失败与合成气流量标定同源（原始流量口径下全部 ≤ 1），不能读作装置氢账本不自洽。"
             "按卡要求，本轮不为这些判据调参、不改口径，也未把标定引入模型求解。\n")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=("unit", "all"))
    args = ap.parse_args()

    silence_solver_logs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ok, rows, _inv = load_frozen_split()
    if not ok:
        raise SystemExit("frozen R52-C-75 split was not consumed intact")
    by_id = {r["window_id"]: r for r in rows}
    window_ids = m1_reference_window_ids(M1_MATRIX)
    p17d = pd.read_csv(P17D_DIR / "coal_reconciled_by_window.csv")
    missing = [w for w in window_ids if w not in set(p17d["window_id"])]
    if missing:
        raise SystemExit(f"P1.7-D CSV 缺窗口：{missing}")

    unit_df, ctx = unit_checks(by_id, window_ids, p17d)
    # 闸门只统计真正参与判定的行；identity/scale/legacy 等信息行 pass 为 None。
    gate_pass = unit_df.loc[unit_df["pass"].notna(), "pass"]
    unit_pass = bool(all(bool(v) for v in gate_pass))

    protocol = base_protocol({
        "round": "P1.7-A / 2026-11-03",
        "source_commit": source_commit(),
        "source_commit_note": "生成本轮全部产物的代码提交；产物本身随后另起一次仅含 outputs 的提交",
        "equation_changes": {
            "1_heating_value": "coal_props 新增显式 LHV_d（干燥基低位）。Hf_coal 在给定 LHV_d 时直接使用、不再减 9·Hd·2442；"
                               "只给 HHV_d 时保持修改前行为；两者同时给出报错。",
            "2_pyrolysis_heat_model": "新增 op_conds['PyrolysisHeatModel'] ∈ {from_heating_value(缺省), zero}。"
                                      "zero 时 Hf_coal 由热解产物焓与焦炭焓反推，使 cell 0 热解热恰为 0，不读任何热值；"
                                      "锚点模式 F 固定为该口径。",
            "3_hydrogen_diagnostic": "H_in_with_moisture_mol_s 只计合同 2% 输煤水分（2 H per H2O，÷0.018015 kg/mol），"
                                     "不计 M_ar。",
            "4_coal_flow_basis": "数据侧煤量是干煤；op_conds['coal_flow'] 是湿煤，进料一律 wet = dry/(1-Mt)，"
                                 "Mt = 2%（model_case_adapter.wet_coal_flow_kg_s）。",
        },
        "hhv_readers": {
            "Hf_coal（gasifier_system.compute_hf_coal_j_kg）": "用低位 LHV_d（水在产物侧为气态，与 Shomate 一致）；"
                                                               "旧代码误按高位再扣潜热。",
            "NumPy 热损 Cell._calc_energy_balance（cell.py）": "用高位（文献口径“入炉煤 HHV 的 2%”）；"
                                                               "只给 LHV 时按定义推 HHV。",
            "JAX 热损透传 GasifierSystem._solve_jax_v4（gasifier_system.py）": "同上，同一个 coal_hhv_kj_kg()。",
            "revision_m5_n5_parity 的 JAX 夹具": "同上，保证与主线同一数值。",
            "锚点 build_case_inputs 的 HHV_d = 30 MJ/kg": "仅为热损基准的占位高位热值（namelist 无热值）；"
                                                          "zero 口径下已不影响 Hf_coal。",
            "validation_loader / fortran_input_loader / original_paper_loader 的 Dulong 式": "Dulong 给的是高位 → 仍填 HHV_d，"
                                                                                             "走“只给高位”的旧换算，语义正确。",
            "chemistry.py / constants.DEFAULT_HHV / gasifier_kinetic_ui.py": "热损与默认值用的高位常数，语义不变。",
            "诊断脚本的 HHV 归一化（_energy_scale、hhv_roundtrip、audit_*、rollback_p0_analysis）": "按高位归一（口径为“相对入炉煤 HHV”），"
                                                                                                    "已改为走 coal_hhv_kj_kg()。",
        },
        "frozen_items_untouched": [
            "all Arrhenius constants", "phi", "f_co2 = 1.0", "P_O2_Combustion_atm = 0.05",
            "CharCombustionRateFactor = 0.3", "H2_OxidationFactor / CO_OxidationFactor",
            "pyrolysis yields", "USE_RK_GILL_COMBUSTION = False", "ParticleSizeModel = fortran_constant",
            "steam_flow = 0", "negative-WGS hydrogen truncation（不改，只统计绑定）",
        ],
        "unit_checks_all_pass": unit_pass,
        "unit_check_tolerances": {
            "Hf_coal_kj_kg": 1.0, "pyrolysis_heat_kj_kg": 2.0, "h_in_moisture_ratio": 0.002,
            "lhv_roundtrip_rel": 1e-9,
        },
        "anchor": {
            "mode_F_definition": "P1.6-A 模式 F + PyrolysisHeatModel='zero'",
            "n_list": list(N_ANCHOR), "n_reference": N_ANCHOR_REF,
            "tolerances": ANCHOR_TOL,
            "hhv_independence_check": "I-1, 模式 F, N=160, HHV_d 30 → 34, 判据 |ΔT| < 1 K",
        },
        "baseline": {
            "windows": list(window_ids),
            "coal_bases": {k: {kk: vv for kk, vv in v.items()} for k, v in COAL_BASES.items()},
            "main_basis": MAIN_BASIS,
            "n_list": [N_BASE, N_RECHECK],
            "n_rule": "N=40 跑三种口径；N=80 只跑主口径复核",
            "coal_basis_conversion": "wet = dry/(1-Mt), Mt = 2% transported moisture",
            "expected_rows": 36,
            "actual_rows_note": "recon_24h 两窗 insufficient → 24 h 口径 7 窗，实际 34 行",
            "plant_flow_bases": "P1.7-D 的 syngas_dry_raw_nm3_h（原始）与 syngas_dry_cal_nm3_h（阶段标定，主口径）",
            "plant_xc_basis": "P1.7-D 的 xc_{weigh,recon_phase,recon_24h}_{raw,cal}_pct（体积分数口径，不受模型影响）",
            "carrier_gas": "J 载气原样使用、不随煤量缩放；缩放敏感度另表列在 interpretation 第 7 节",
        },
        "gates": {
            "G1": "spearman(|ρ|) between single-parameter closure CO residual and pipeline_over_weigh < 0.5",
            "G2": "median |CO residual| ≤ 1.0 vol% and median |H2 residual| ≤ 1.0 vol%",
            "G3": "per-phase |median model X_C − liu2023 phase value| ≤ 3 pp",
            "G4": "plant dry-gas H2 (calibrated flow) / feed H (2% moisture, reconciled coal) ≤ 1 per window",
            "report_only": "model X_C − plant carbon-balance X_C (main basis)",
        },
        "source_csv": {
            "path": str(P17D_DIR / "coal_reconciled_by_window.csv"),
            "sha256": __import__("hashlib").sha256(
                (P17D_DIR / "coal_reconciled_by_window.csv").read_bytes()).hexdigest(),
            "read_only": True,
        },
        "extra_artifacts": ["anchor_outlet.csv（锚点逐算例出口，供 `HHV_d` 无关性检查逐行核对）"],
    })

    if not unit_pass or args.stage == "unit":
        tables = {"unit_checks.csv": unit_df}
        interp = _unit_only_interpretation(unit_df, ctx, unit_pass, args.stage)
        written = write_round_outputs(OUT_DIR, tables=tables, protocol=protocol, interpretation=interp)
        print(json.dumps({"stage": args.stage, "unit_pass": unit_pass,
                          "written": [str(p) for p in written]}, ensure_ascii=False, indent=2))
        print(unit_df[["check_group", "context", "metric", "value", "pass"]].to_string(index=False))
        if not unit_pass:
            raise SystemExit("第 1 节单元级检查未通过：按卡停止，不进入第 2、4 节。")
        return

    # ---------------- 第 2 节 ----------------
    anchor_case_list, geometry = anchor_cases()
    fortran_out, _prof = parse_fortran_log(P15C_DIR / "fortran_build" / "run.log")
    fortran_by_case = match_fortran(anchor_case_list, fortran_out)
    anchor_out = anchor_outlet_rows(anchor_case_list, geometry, fortran_by_case)
    crit_df = anchor_criteria_rows(anchor_out, fortran_by_case)
    attr_df = anchor_attribution_rows(anchor_case_list, geometry, fortran_by_case)

    # ---------------- 第 4 节 ----------------
    base_df, cells_df, skipped = baseline_stage(window_ids, by_id, p17d)
    clos_df = closure_rows(base_df)
    gates_df, _extras = gates_stage(base_df, clos_df, p17d)
    carrier_df = carrier_sensitivity(p17d)

    protocol.update({
        "anchor_cases_used": [c["case_name"] for c in anchor_case_list],
        "anchor_criteria_f_mode_pass_cases": int(
            crit_df[(crit_df["mode"] == "F") & (crit_df["evaluable"].astype(bool))]
            .groupby("case_name")["pass"].all().sum()),
        "baseline_rows_actual": int(len(base_df)),
        "baseline_rows_skipped": skipped,
    })

    interp = full_interpretation(unit_df, ctx, crit_df, attr_df, base_df, clos_df, gates_df,
                                 carrier_df, window_ids, protocol)

    tables = {
        "unit_checks.csv": unit_df,
        "anchor_criteria.csv": crit_df,
        "anchor_attribution.csv": attr_df,
        "anchor_outlet.csv": anchor_out,
        "baseline.csv": base_df,
        "baseline_cells.csv": cells_df,
        "post_reactor_closure.csv": clos_df,
        "gates.csv": gates_df,
    }
    written = write_round_outputs(OUT_DIR, tables=tables, protocol=protocol, interpretation=interp)
    print(json.dumps({"written": [str(p) for p in written], "unit_pass": unit_pass,
                      "baseline_rows": int(len(base_df))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
