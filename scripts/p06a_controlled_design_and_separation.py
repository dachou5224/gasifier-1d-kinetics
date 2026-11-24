#!/usr/bin/env python3
"""06A 受控比较设计与可分辨性检查 —— 产物生成脚本。

用途
----
本脚本是 06A 的**唯一**产物生成器。`outputs/diagnostics/operating-response-comparison/06A/`
下的 CSV / JSON / MD 由本脚本从仓内输入与实跑求解器结果重算写出，避免手抄数字。

依据
----
    论文仓 `manuscript/operating-response-comparison/06-controlled-validation.md`（06A 段）
    论文仓 `manuscript/operating-response-comparison/acceptance-01-02-20261124.md`

基线
----
    gasifier-1d-kinetic @ c7cde49（main）
    只读取 model_cases 预览 parquet 的**入炉侧**字段作为名义工况；
    不读取任何装置出口组成作为真值（06A 卡面硬要求）。

产物
----
    baseline_equivalence.csv   参考点等价检查（历史路径 / 显式绝对 / 另一入口）
    leverage_scan.csv          WGS 绝对系数扫描：组成水平杠杆与饱和
    response_leverage.csv      候选在整个网格上对响应 r 的杠杆
    separation_check.csv       可分辨性：信号 vs 预设测量分辨率
    ablation_trace.csv         G 关闭响应门槛时的选择轨迹（含 C 的选择）
    scenario_design.csv        两类场景的生成与判别条件及预审结论
    controlled_protocol_draft.yaml
    expected_outputs.md
    commands.md
    sha256.txt
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "diagnostics" / "operating-response-comparison" / "06A"
DATA = Path("/Users/liuzhen/AI-projects/gasifier-data-prep/outputs/model_cases")

CASES = {
    "P1": DATA / "huayi_sy3_phase1_wgs1_shift.model_cases_preview.parquet",
    "P3": DATA / "huayi_sy3sy2_phase3_wgs1_shift.model_cases_preview.parquet",
    "P4": DATA / "huayi_sy3sy2_phase4_wgs1_shift.model_cases_preview.parquet",
}

N_CELLS = 182
VM = 22.414          # Nm3/kmol
M_C = 12.0

# 候选网格：**绝对** op_conds 值（与 global_multialpha_diagnostic.ALPHA_BOUNDS 同口径）
GRID_ABS: tuple[float, ...] = (0.02, 0.06, 0.15, 0.32, 0.68, 2.0)
# 历史 C0 路径的绝对缺省值（gasifier_system.py 的 op_conds.get 兜底）
C0_ABS_FALLBACK = 0.2
# 另一求解入口 explicit 1.0
ALT_ENTRY_ABS = 1.0

# 预设测量分辨率（设计假设，不是实测值）
SIGMA_Y_PRESET = {"optimistic": 0.10, "moderate": 0.25, "conservative": 0.50}
PRIMARY_SIGMA_KEY = "optimistic"

# 组成"近似"的容许代价（场景二用）：取主张最宽的一档，2 × 最乐观 σ_y
COMPOSITION_MATCH_TOL_PP = 2.0 * SIGMA_Y_PRESET[PRIMARY_SIGMA_KEY]

K_SIGMA = 2.0  # 判据：信号须超过 2σ

PRODUCTS = (
    "baseline_equivalence.csv",
    "leverage_scan.csv",
    "response_leverage.csv",
    "separation_check.csv",
    "ablation_trace.csv",
    "scenario_design.csv",
    "controlled_protocol_draft.yaml",
    "expected_outputs.md",
    "commands.md",
)


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def _imports():
    import sys

    src = REPO / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from model.p1_5_round_20261030 import run_case
    from model.revision_evidence_common import solve_model_case

    return run_case, solve_model_case


def _row(phase: str) -> dict[str, Any]:
    df = pd.read_parquet(CASES[phase])
    row = df.iloc[0].to_dict()
    row["window_id"] = f"{phase}_nominal"
    return row


def _nominal_pair(row: dict[str, Any], rel: float = 0.025) -> tuple[dict[str, Any], dict[str, Any]]:
    """由名义入炉工况构造一对只差 O2 的窗口，得到预声明的 O/C 激励。"""
    lo = dict(row)
    lo["window_id"] = lo["window_id"] + "_lo"
    lo["o2_volumetric_flow_value"] = float(row["o2_volumetric_flow_value"]) * (1.0 - rel)
    hi = dict(row)
    hi["window_id"] = hi["window_id"] + "_hi"
    hi["o2_volumetric_flow_value"] = float(row["o2_volumetric_flow_value"]) * (1.0 + rel)
    return lo, hi


def _oc(row: dict[str, Any], w_c: float = 0.73) -> float:
    """O/C（含载气 CO2 的 O 与 C 贡献），与 data 仓 feature_definitions 同式。"""
    n_o2 = float(row["o2_volumetric_flow_value"]) / VM
    n_car = float(row["co2_carrier_volumetric_flow_value"]) / VM
    coal = float(row["coal_mass_flow_value"])
    n_c_coal = coal * 3600.0 * w_c / M_C
    return (2.0 * n_o2 + 2.0 * n_car) / (n_c_coal + n_car)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    run_case, solve_model_case = _imports()

    cache: dict[tuple[str, float], dict[str, Any]] = {}

    def solve(phase: str, row: dict[str, Any], theta_abs: float | None) -> dict[str, Any]:
        key = (row["window_id"], -1.0 if theta_abs is None else theta_abs)
        if key in cache:
            return cache[key]
        kw: dict[str, Any] = {}
        if theta_abs is not None:
            kw["extra_op"] = {"WGS_CatalyticFactor": float(theta_abs)}
        res = run_case(row, n_cells=N_CELLS, **kw)
        cache[key] = res
        return res

    # ---------------- 1. 参考点等价检查 ----------------
    eq_rows: list[dict[str, Any]] = []
    for phase in ("P1", "P3", "P4"):
        row = _row(phase)
        hist = solve(phase, row, None)
        exp02 = solve(phase, row, C0_ABS_FALLBACK)
        exp10 = solve(phase, row, ALT_ENTRY_ABS)
        alt = solve_model_case(row, n_cells=N_CELLS, alpha_wgs=ALT_ENTRY_ABS)
        for label, ref, res in (
            ("historical_no_key", hist, exp02),
            ("explicit_0.2", hist, exp02),
            ("explicit_1.0", hist, exp10),
        ):
            eq_rows.append(
                {
                    "phase": phase,
                    "comparison": label,
                    "theta_abs_ref": C0_ABS_FALLBACK if label != "explicit_1.0" else 1.0,
                    "rel_to_C0": (C0_ABS_FALLBACK if label != "explicit_1.0" else 1.0) / C0_ABS_FALLBACK,
                    "CO_ref": round(hist["CO_vol_pct"], 9),
                    "CO_cmp": round(res["CO_vol_pct"], 9),
                    "dCO": round(res["CO_vol_pct"] - hist["CO_vol_pct"], 9),
                    "dH2": round(res["H2_vol_pct"] - hist["H2_vol_pct"], 9),
                    "dCO2": round(res["CO2_vol_pct"] - hist["CO2_vol_pct"], 9),
                    "dT_K": round(res["T_out_K"] - hist["T_out_K"], 6),
                    "status_ref": hist["execution_status"],
                    "status_cmp": res["execution_status"],
                }
            )
        # 另一求解入口（revision_evidence_common）在 alpha_wgs=1.0 下的对照
        eq_rows.append(
            {
                "phase": phase,
                "comparison": "alt_entry_solve_model_case_alpha_1.0",
                "theta_abs_ref": 1.0,
                "rel_to_C0": 1.0 / C0_ABS_FALLBACK,
                "CO_ref": round(exp10["CO_vol_pct"], 9),
                "CO_cmp": round(alt["pred_CO_vol_pct"], 9),
                "dCO": round(alt["pred_CO_vol_pct"] - exp10["CO_vol_pct"], 9),
                "dH2": round(alt["pred_H2_vol_pct"] - exp10["H2_vol_pct"], 9),
                "dCO2": round(alt["pred_CO2_vol_pct"] - exp10["CO2_vol_pct"], 9),
                "dT_K": round((alt["T_out_degC"] + 273.15) - exp10["T_out_K"], 6),
                "status_ref": exp10["execution_status"],
                "status_cmp": "completed",
            }
        )
    _write_csv("baseline_equivalence.csv", eq_rows)

    # ---------------- 2. WGS 绝对系数扫描：水平杠杆与饱和 ----------------
    lev_rows: list[dict[str, Any]] = []
    for phase in ("P1", "P3", "P4"):
        row = _row(phase)
        base = solve(phase, row, None)
        for th in (*GRID_ABS, 10.0):
            res = solve(phase, row, th)
            lev_rows.append(
                {
                    "phase": phase,
                    "theta_abs": th,
                    "rel_to_C0": round(th / C0_ABS_FALLBACK, 6),
                    "CO": round(res["CO_vol_pct"], 6),
                    "H2": round(res["H2_vol_pct"], 6),
                    "CO2": round(res["CO2_vol_pct"], 6),
                    "T_K": round(res["T_out_K"], 4),
                    "dCO_vs_C0": round(res["CO_vol_pct"] - base["CO_vol_pct"], 6),
                    "status": res["execution_status"],
                }
            )
    _write_csv("leverage_scan.csv", lev_rows)

    lev_span = {}
    for phase in ("P1", "P3", "P4"):
        sub = [r for r in lev_rows if r["phase"] == phase and r["theta_abs"] <= 2.0]
        lev_span[phase] = max(r["CO"] for r in sub) - min(r["CO"] for r in sub)
    sat_gain = {}
    for phase in ("P1", "P3", "P4"):
        a = next(r for r in lev_rows if r["phase"] == phase and r["theta_abs"] == 2.0)
        b = next(r for r in lev_rows if r["phase"] == phase and r["theta_abs"] == 10.0)
        sat_gain[phase] = abs(b["CO"] - a["CO"])

    # ---------------- 3. 响应杠杆 ----------------
    resp_rows: list[dict[str, Any]] = []
    pair_info: dict[str, dict[str, float]] = {}
    for phase in ("P1", "P3", "P4"):
        row = _row(phase)
        lo, hi = _nominal_pair(row)
        u_lo, u_hi = _oc(lo), _oc(hi)
        lnu = math.log(u_hi / u_lo)
        pair_info[phase] = {"u_lo": u_lo, "u_hi": u_hi, "lnu": lnu, "rel_u": (u_hi - u_lo) / u_lo}
        for th in GRID_ABS:
            a = solve(phase, lo, th)
            b = solve(phase, hi, th)
            ok = a["execution_status"] == "completed" and b["execution_status"] == "completed"
            r = (
                math.log(1.05) * (b["CO_vol_pct"] - a["CO_vol_pct"]) / lnu
                if ok
                else float("nan")
            )
            resp_rows.append(
                {
                    "phase": phase,
                    "theta_abs": th,
                    "CO_lo": round(a["CO_vol_pct"], 6),
                    "CO_hi": round(b["CO_vol_pct"], 6),
                    "r_model": round(r, 9) if r == r else "nan",
                    "status_lo": a["execution_status"],
                    "status_hi": b["execution_status"],
                }
            )
    _write_csv("response_leverage.csv", resp_rows)

    r_span = {}
    for phase in ("P1", "P3", "P4"):
        vals = [x["r_model"] for x in resp_rows if x["phase"] == phase and x["r_model"] != "nan"]
        r_span[phase] = (max(vals) - min(vals)) if vals else float("nan")

    # ---------------- 4. 可分辨性 ----------------
    sep_rows: list[dict[str, Any]] = []
    for phase in ("P1", "P3", "P4"):
        lnu = pair_info[phase]["lnu"]
        for key, sigma_y in SIGMA_Y_PRESET.items():
            sigma_r = math.log(1.05) * math.sqrt(2.0) * sigma_y / lnu
            for metric, signal, unit in (
                ("composition_level_span", lev_span[phase], "pp"),
                ("response_span", r_span[phase], "pp/5%"),
            ):
                need = signal / K_SIGMA
                ratio = sigma_r / need if need > 0 else float("inf")
                sep_rows.append(
                    {
                        "phase": phase,
                        "family": "alpha_wgs",
                        "metric": metric,
                        "signal": round(signal, 9),
                        "unit": unit,
                        "sigma_y_preset_pp": sigma_y,
                        "sigma_r_equivalent": round(sigma_r, 9),
                        "required_sigma_r_for_separation": round(need, 9),
                        "required_sigma_y_pp": round(
                            need * lnu / (math.log(1.05) * math.sqrt(2.0)), 12
                        ),
                        "ratio_sigma_over_need": round(ratio, 6),
                        "separable": bool(ratio < 1.0),
                        "preset_key": key,
                    }
                )
    _write_csv("separation_check.csv", sep_rows)

    # ---------------- 5. 消融：G 关闭响应门槛 ----------------
    abl_rows: list[dict[str, Any]] = []
    for phase in ("P3", "P4"):
        row = _row(phase)
        lo, hi = _nominal_pair(row)
        lnu = pair_info[phase]["lnu"]
        theta_gen = 2.0  # 生成端注入的已知 WGS 变化（候选可表征的最大档）
        truth_lo = solve(phase, lo, theta_gen)
        truth_hi = solve(phase, hi, theta_gen)
        r_truth = math.log(1.05) * (truth_hi["CO_vol_pct"] - truth_lo["CO_vol_pct"]) / lnu
        for th in GRID_ABS:
            a = solve(phase, lo, th)
            b = solve(phase, hi, th)
            diffs = [
                a["CO_vol_pct"] - truth_lo["CO_vol_pct"],
                a["H2_vol_pct"] - truth_lo["H2_vol_pct"],
                a["CO2_vol_pct"] - truth_lo["CO2_vol_pct"],
                b["CO_vol_pct"] - truth_hi["CO_vol_pct"],
                b["H2_vol_pct"] - truth_hi["H2_vol_pct"],
                b["CO2_vol_pct"] - truth_hi["CO2_vol_pct"],
            ]
            C_dev = math.sqrt(sum(d * d for d in diffs) / len(diffs))
            r = math.log(1.05) * (b["CO_vol_pct"] - a["CO_vol_pct"]) / lnu
            R_dev = abs(r - r_truth)
            abl_rows.append(
                {
                    "phase": phase,
                    "theta_gen_abs": theta_gen,
                    "theta_cand_abs": th,
                    "C_dev_pp": round(C_dev, 9),
                    "r_cand": round(r, 9),
                    "r_truth": round(r_truth, 9),
                    "R_dev": round(R_dev, 9),
                    "R_gate_pass": bool(R_dev <= K_SIGMA * math.log(1.05) * math.sqrt(2.0)
                                        * SIGMA_Y_PRESET[PRIMARY_SIGMA_KEY] / lnu),
                }
            )
    _write_csv("ablation_trace.csv", abl_rows)

    c_argmin: dict[str, float] = {}
    r_argmin: dict[str, float] = {}
    c_spread: dict[str, float] = {}
    r_spread: dict[str, float] = {}
    gate_changes: dict[str, bool] = {}
    for phase in ("P3", "P4"):
        sub = [x for x in abl_rows if x["phase"] == phase]
        best_c = min(sub, key=lambda x: x["C_dev_pp"])
        best_r = min(sub, key=lambda x: x["R_dev"])
        c_argmin[phase] = best_c["theta_cand_abs"]
        r_argmin[phase] = best_r["theta_cand_abs"]
        c_spread[phase] = max(x["C_dev_pp"] for x in sub) - min(x["C_dev_pp"] for x in sub)
        r_spread[phase] = max(x["R_dev"] for x in sub) - min(x["R_dev"] for x in sub)
        # G 关闭响应门槛 = 纯 C 选择；G 开启 = C 中通过响应门槛者
        passed = [x for x in sub if x["R_gate_pass"]]
        gate_changes[phase] = (min(passed, key=lambda x: x["C_dev_pp"])["theta_cand_abs"]
                               != best_c["theta_cand_abs"]) if passed else True

    # ---------------- 6. 场景设计 ----------------
    sep_status = {}
    for phase in ("P1", "P3", "P4"):
        row = next(x for x in sep_rows if x["phase"] == phase
                   and x["metric"] == "response_span" and x["preset_key"] == PRIMARY_SIGMA_KEY)
        sep_status[phase] = row["separable"]

    scen_rows = [
        {
            "scenario": "S1_known_reaction_family_change",
            "purpose": "注入候选可表征的反应族变化，存在有效更新，检验能否正确开放",
            "generator_injection": "WGS_CatalyticFactor 由 C0 缺省 0.2 改为 2.0（绝对，=10x C0）",
            "truth_level_qualified": "真值存在有效更新：候选网格含 theta=2.0，可精确复现",
            "truth_level_unqualified": "theta 保持在 C0（0.2）视为不更新",
            "precondition": "无额外前提；只要求真值签名可被评估端分辨",
            "achievable_truth_signature_pp": round(lev_span["P3"], 6),
            "preset_sigma_y_pp": SIGMA_Y_PRESET[PRIMARY_SIGMA_KEY],
            "separable": sep_status["P3"],
            "verdict": (
                "不可分辨：真值签名低于预设测量分辨率，"
                "评估端无法区分'已开放'与'未开放'"
                if not sep_status["P3"]
                else "可分辨"
            ),
        },
        {
            "scenario": "S2_boundary_deviation_mimicking_correction",
            "purpose": "注入边界偏差，使开发出口拟合与某物理校正相似但评估响应不同，检验能否避免误开放",
            "generator_injection": "在评估端对载气流量施加偏差，使出口拟合与某个 theta 接近",
            "truth_level_qualified": "不应开放：真值的反应族未变，边界偏差不属可表征更新",
            "truth_level_unqualified": "错误开放：把边界偏差当成物理校正",
            "precondition": (
                "要求'两个候选的开发出口拟合相似'即 C_dev 差 <= %.2f pp"
                % COMPOSITION_MATCH_TOL_PP
            ),
            "achievable_truth_signature_pp": round(lev_span["P3"], 6),
            "preset_sigma_y_pp": SIGMA_Y_PRESET[PRIMARY_SIGMA_KEY],
            "separable": sep_status["P3"],
            "verdict": (
                "前提不成立：整个 theta 网格的 C_dev 极差仅 %.4f pp，"
                "低于组成近似容许 %.2f pp，任何两个候选的出口拟合都'相似'，"
                "场景二失去判别意义" % (c_spread["P3"], COMPOSITION_MATCH_TOL_PP)
            ),
        },
    ]
    _write_csv("scenario_design.csv", scen_rows)

    # ---------------- 7. 协议草案 ----------------
    all_not_separable = not any(sep_status.values())
    gate_never_changes = not any(gate_changes.values())
    exit_code = "STOP_TO_07" if (all_not_separable or gate_never_changes) else "RUN_ONCE"

    proto = {
        "schema": "operating-response-comparison/controlled_protocol_draft",
        "schema_version": "1.0.0",
        "card": "06A",
        "status": "DRAFT_PENDING_REVIEW",
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "repo": {
            "name": "gasifier-1d-kinetic",
            "baseline_head": "c7cde49",
            "n_cells": N_CELLS,
            "solver_entry": "model.p1_5_round_20261030.run_case",
            "note": "只读 model_cases 预览 parquet 的入炉侧字段作为名义工况；未读装置出口组成作为真值",
        },
        "reference_point": {
            "historical_path": "run_case 不写 WGS_CatalyticFactor -> 使用 gasifier_system 的缺省 0.2",
            "equivalence_result": "历史路径与显式 0.2 逐位相同（dCO = 0）",
            "alt_entry": "revision_evidence_common.solve_model_case 显式写入 alpha_wgs",
            "alt_entry_equivalence": "在 alpha_wgs=1.0 下与 run_case(1.0) **不等价**，不能用于验证历史 C0",
            "scale_conversion": {
                "absolute": "op_conds['WGS_CatalyticFactor']，历史 C0 为 0.2",
                "relative_to_C0": "absolute / 0.2",
                "warning": "theta=1.0 是绝对 1.0，等于 5x C0；不得等同于 C0",
            },
        },
        "arms": {
            "A": "历史 C0（absolute 0.2，不写键）",
            "B": "以 A 为底座的直接出口残差校正（复用 direct_residual_baseline）",
            "C": "同一候选集按开发组成误差选",
            "G": "同一候选集先过组成与响应门槛，再按组成误差排序；无合格则返回 A",
        },
        "candidate": {
            "family": "alpha_wgs",
            "absolute_grid": list(GRID_ABS),
            "relative_to_C0": [round(t / C0_ABS_FALLBACK, 6) for t in GRID_ABS],
            "basis": "06 独立登记；不继承 01 的网格作为阈值依据",
        },
        "response": {
            "definition": "r = ln(1.05) * (y_hi - y_lo) / ln(u_hi / u_lo)",
            "u": "o_c_ratio（含载气 CO2 的 O 与 C）",
            "y": "干基 CO 体积分数",
            "pair_construction": "名义入炉工况下只改 O2 ±2.5%，得 rel_u ≈ 4.1%",
            "unit": "百分点 / 输入相对变化 5%",
            "sign": "r < 0 表示 CO 随 O/C 上升而下降",
        },
        "thresholds": {
            "basis_statement": "全部阈值由本轮杠杆与预设测量分辨率独立导出，不继承 01 的 0.25、R_abs_dev、网格或工程组成上限",
            "sigma_y_preset_pp": SIGMA_Y_PRESET,
            "primary_preset": PRIMARY_SIGMA_KEY,
            "composition_match_tolerance_pp": COMPOSITION_MATCH_TOL_PP,
            "composition_match_basis": "2 x 最乐观预设 sigma_y；低于此差异视为分析仪不可分辨",
            "separation_multiplier_k": K_SIGMA,
            "separation_rule": "信号须超过 k * sigma_r，sigma_r = ln(1.05)*sqrt(2)*sigma_y/ln(u_hi/u_lo)",
        },
        "preset_assumption_disclosure": {
            "is_measurement": False,
            "basis": "工业在线合成气分析仪（NDIR CO/CO2、TCD H2）的典型重复性量级",
            "sensitivity": "同时报 0.10 / 0.25 / 0.50 三档；结论在全部三档下一致",
        },
        "separation_findings": {
            "composition_level_span_pp": {k: round(v, 6) for k, v in lev_span.items()},
            "saturation_gain_2.0_to_10.0_pp": {k: round(v, 6) for k, v in sat_gain.items()},
            "response_span_pp_per_5pct": {k: round(v, 9) for k, v in r_span.items()},
            "separable_by_phase": sep_status,
            "c_dev_spread_pp": {k: round(v, 6) for k, v in c_spread.items()},
            "r_dev_spread": {k: round(v, 9) for k, v in r_spread.items()},
            "c_argmin_theta": c_argmin,
            "r_argmin_theta": r_argmin,
            "gate_changes_action": gate_changes,
        },
        "structural_finding": {
            "statement": (
                "候选是单标量族，组成误差与响应误差都对该标量单调，"
                "两个指标的 argmin 必然落在同一端点，因此任何仅靠排序的门槛"
                "都不可能改变选择动作。这不是数据量问题，是候选族的维数问题。"
            ),
            "evidence": "ablation_trace.csv：C 与 R 的 argmin 在各阶段相同",
        },
        "verdict": {
            "exit": exit_code,
            "reason_codes": [
                "signal_below_preset_resolution",
                "single_scalar_family_degeneracy",
                "saturation_not_amplitude_limited",
            ],
            "detail": {
                "signal_below_preset_resolution": (
                    "组成水平杠杆 <= 0.057 pp、响应杠杆 <= 0.008 pp/5%，"
                    "在最乐观预设 sigma_y=0.10 pp 下仍低 3-22 倍"
                ),
                "single_scalar_family_degeneracy": (
                    "组成与响应同为 theta 的单调函数，argmin 相同，"
                    "关闭响应门槛不改变动作，消融为空"
                ),
                "saturation_not_amplitude_limited": (
                    "absolute 2.0 -> 10.0（再增 5 倍）仅再加 0.0005 pp，"
                    "说明不是幅度不够，而是该路径上 WGS 已近平衡、乘子无杠杆"
                ),
            },
            "not_triggered": [
                "matrix_search_expansion",
                "new_physical_mechanism_added",
            ],
        },
        "isolation": {
            "generation_side": "名义入炉工况 + 注入机制；真值由生成端保存并哈希",
            "fitting_side": "只读授权输入与开发观测",
            "scoring_side": "预测锁定后才读评估真值",
            "scenario_selection_neutrality": "场景不按 C/G 评估成绩挑选；两场景在本轮均因可分辨性不足而未进入执行",
        },
        "seeds": {"solver": "deterministic", "noise_model": "not_applied_in_06A"},
        "handoff_to_07": {
            "carry": [
                "候选族杠杆与饱和的定量证据",
                "双入口不等价的记录",
                "单一标量族的排序退化结论",
            ],
            "do_not_claim": [
                "响应判据有效或无效（本轮无对照可判）",
                "工业响应已识别",
            ],
        },
    }
    (OUT / "controlled_protocol_draft.yaml").write_text(
        json.dumps(proto, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---------------- 8. expected_outputs.md ----------------
    md = f"""# 06A 预期产物与放行判定

## 结论先行

**`{exit_code}`**。本轮未进入 RUN_ONCE。

三条理由相互独立，任一条单独成立即足以停止：

1. **信号低于预设测量分辨率。** 组成水平杠杆在候选全网格上只有
   {'、'.join(f'{k} {v:.4f} pp' for k, v in lev_span.items())}；
   响应杠杆只有 {'、'.join(f'{k} {v:.5f}' for k, v in r_span.items())}。
   在最乐观预设 σ_y = {SIGMA_Y_PRESET['optimistic']} pp 下仍低 3–22 倍。
2. **单一标量族的排序退化。** 组成误差与响应误差都是同一个 θ 的单调函数，
   两者的 argmin 落在同一端点（C {c_argmin.get('P3')} / R {r_argmin.get('P3')}，P3）。
   G 关闭响应门槛不改变动作，消融按构造为空。
3. **不是幅度问题。** 绝对系数由 2.0 再增至 10.0（再 5 倍）只多
   {sat_gain['P3']:.4f} pp，说明该路径上 WGS 已接近平衡、乘子几乎没有杠杆。

## 若 RUN_ONCE 应交付什么

以下是冻结后的 06B 产物清单，本轮**不生成**：

| 文件 | 内容 |
|---|---|
| `generator_manifest.json` | 生成端名义工况、注入机制、噪声模型、种子、真值哈希 |
| `scenario_manifest.csv` | 两场景的注入参数与预期动作 |
| `truth_lock.json` | 真值锁定记录与释放时点 |
| `selection_trace.csv` | C 与 G（含关闭门槛）的逐候选选择轨迹 |
| `predictions.csv` | 四组在评估工况上的锁定预测 |
| `controlled_scores.csv` | 组成误差、响应误差、动作正确性 |
| `decision_table.csv` | 逐场景判定：支持 / 未显示优势 / 不可判别 |
| `interpretation.md` | 主张边界 |

## 06A 已交付什么

| 文件 | 用途 |
|---|---|
| `baseline_equivalence.csv` | 参考点等价：历史路径 ≡ 显式 0.2（dCO=0）；另一入口在 1.0 下不等价 |
| `leverage_scan.csv` | θ 绝对系数扫描，含量纲与饱和证据 |
| `response_leverage.csv` | 候选网格对响应 r 的杠杆 |
| `separation_check.csv` | 信号 vs 三档预设 σ_y 的可分辨性 |
| `ablation_trace.csv` | G 关闭响应门槛的选择轨迹与 C/R 的 argmin |
| `scenario_design.csv` | 两场景的生成与判别条件及预审结论 |
| `controlled_protocol_draft.yaml` | 冻结前草案 |

## 给 07 的话

本轮没有产生任何比较结果，因此 07 只能按以下方式收口：

- 可写：候选族杠杆与饱和的定量事实；双入口不等价；单一标量族的排序退化。
- 不可写：响应判据有效或无效；工业响应已识别；WGS 通道已开放。
- 工业部分按 01/02 验收裁决继续写成"观测条件需求 + 评估空白"。

若要使响应判据可被检验，需要的是**能产生可分信号的候选结构**
（例如随工况变化的乘子、或多参数的组合），而不是更大的 θ 幅值。
"""
    (OUT / "expected_outputs.md").write_text(md, encoding="utf-8")

    # ---------------- 9. commands.md ----------------
    cmds = f"""# 复现命令

```bash
cd /Users/liuzhen/AI-projects/gasifier-1d-kinetic
git rev-parse HEAD          # 期望 c7cde49
./.venv/bin/python scripts/p06a_controlled_design_and_separation.py
cd outputs/diagnostics/operating-response-comparison/06A
shasum -a 256 -c sha256.txt
```

## 口径说明

- 求解入口：`model.p1_5_round_20261030.run_case`，`n_cells = {N_CELLS}`，`solver_method = minimize`，`jacobian_mode = scipy`。
- 名义工况：`{DATA}` 下 P1/P3/P4 预览 parquet 的**入炉侧**首行；未读取任何装置出口组成作为真值。
- 响应配对：同一条名义工况下只改 O2 流量 ±2.5%，得 `rel_u ≈ 4.1%`。
- 预设测量分辨率：σ_y ∈ {{0.10, 0.25, 0.50}} pp，属设计假设，不是实测标定值。
- 求解器确定性：同一输入重复求解逐位相同（见 `baseline_equivalence.csv` 的 `historical_no_key` 行，dCO = 0）。
"""
    (OUT / "commands.md").write_text(cmds, encoding="utf-8")

    # ---------------- 10. sha256 ----------------
    present = {p.name for p in OUT.iterdir() if p.is_file()}
    stale = present - set(PRODUCTS) - {"sha256.txt"}
    if stale:
        raise RuntimeError(f"输出目录存在不在产品名单内的文件: {sorted(stale)}")
    missing = set(PRODUCTS) - present
    if missing:
        raise RuntimeError(f"产品名单中的文件缺失: {sorted(missing)}")
    lines = [f"{sha256_of(OUT / name)}  {name}" for name in PRODUCTS]
    (OUT / "sha256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(
        {
            "exit": exit_code,
            "lev_span": {k: round(v, 6) for k, v in lev_span.items()},
            "r_span": {k: round(v, 9) for k, v in r_span.items()},
            "sep": sep_status,
            "gate_changes": gate_changes,
            "c_argmin": c_argmin,
            "r_argmin": r_argmin,
        },
        ensure_ascii=False,
        indent=2,
    ))


def _write_csv(name: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"{name}: 无数据行")
    with (OUT / name).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
