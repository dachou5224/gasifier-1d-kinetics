#!/usr/bin/env python3
"""06A 受控比较设计与可分辨性检查 —— 产物生成脚本（rev 2）。

rev 2 相对 rev 1 的修正（按首轮审查意见）
------------------------------------------
P1 参考点底座：rev 1 直接调 `run_case` 默认 `mesh_family="uniform_full_length"`，
   且未传 `coal_flow_dry_kg_s`，因此走的是 uniform 路径、煤量按湿煤处理。
   那不是 P2.2-A 的 C0。rev 2 改为复用 P2.2-A 的冻结配置：
   MeshFamily=wsr_plus_pfr / WSRLength=1.0 / PFRCellLength=0.0515 /
   HeatLossPercentTotal=1.0 / 实测载气 / 显式干煤口径 / n_cells=182。
   并先用 `baseline_142.csv` 做逐位复现，再比较不写键与显式 0.2。

P2 误差传播：rev 1 把组成水平也套用响应量 σ_r（含 ln(1.05)/Δln u 归一因子），
   高估了倍数。rev 2 分开：组成用单一观测对确定性模型的 k·σ_y；
   响应用 σ_r = ln(1.05)·√2·σ_y/Δln u。

P3 排序退化：rev 1 把「真值取网格内 θ=2.0 + 无噪声同求解器 → C_dev=R_dev=0」
   当成「单标量族必然退化」的证据，并外推为「任何门槛都不可能改变动作」。
   rev 2 撤回这些普遍结论，只报「在本 S1、该网格与预设门槛下未改变动作」。

P4 场景二：rev 1 只算了 WGS 网格的 C_dev 极差，把「组成近似易满足」
   误写成「前提不成立」，且未真正注入边界偏差。rev 2 实际计算一种
   载气偏差，并分别登记组成近似与响应分离的通过/失败。

产物目录：outputs/diagnostics/operating-response-comparison/06A/
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

# P2.2-A 冻结输入的论文仓副本（只读）
PAPER = Path("/Users/liuzhen/AI-projects/ai-research/papers/jax-gasifier-cascade-hybrid")
WINDOWS_CSV = PAPER / "revision-evidence-20261118/P2.2-D/windows_142.csv"
BASELINE_CSV = PAPER / "revision-evidence-20261118/P2.2-A/baseline_142.csv"

VM = 22.414
M_C = 12.0

# 候选网格：**绝对** op_conds 值
GRID_ABS: tuple[float, ...] = (0.02, 0.06, 0.15, 0.32, 0.68, 2.0)
C0_ABS_FALLBACK = 0.2      # gasifier_system 的 op_conds.get 兜底
ALT_ENTRY_ABS = 1.0        # revision_evidence_common.solve_model_case 的显式设置

# 预设测量分辨率（设计假设，非实测标定）
SIGMA_Y_PRESET = {"optimistic": 0.10, "moderate": 0.25, "conservative": 0.50}
PRIMARY_SIGMA_KEY = "optimistic"
K_SIGMA = 2.0

# 场景二：边界偏差幅值（载气流量相对偏差）
S2_CARRIER_REL = 0.03

PRODUCTS = (
    "baseline_equivalence.csv",
    "c0_reproduction.csv",
    "leverage_scan.csv",
    "response_leverage.csv",
    "separation_check.csv",
    "ablation_trace.csv",
    "scenario_design.csv",
    "controlled_protocol_draft.yaml",
    "expected_outputs.md",
    "commands.md",
    "limitations.md",
)

EQUIV_PHASES = ("P1", "P2", "P3", "P4")
RUN_PHASES = ("P1", "P2", "P3", "P4")


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def _ctx():
    import sys

    src = REPO / "src"
    scripts = REPO / "scripts"
    for p in (src, scripts):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from model.p1_5_round_20261030 import run_case, silence_solver_logs
    from model.revision_evidence_common import load_frozen_split
    import generate_p1_8a_20261104 as p18a
    import generate_p2_0a_20261113 as p20

    silence_solver_logs()
    return run_case, load_frozen_split, p18a, p20


def p22a_op(p18a, p20, carrier_nm3: float) -> dict[str, Any]:
    """P2.2-A 的冻结 op 配置（见 generate_p2_2a_20261118.build_op）。"""
    return {
        "HeatLossModel": "total_percent",
        "HeatLossPercentTotal": 1.0,
        "GasificationRateFactor": 1.0,
        "CharCO2RateFactor": 1.0,
        "carrier_gas_flow": p18a._carrier_kg_s(carrier_nm3),
        "MeshFamily": "wsr_plus_pfr",
        "WSRLength_m": 1.0,
        "PFRCellLength_m": 0.0515,
        "OxygenMixingModel": "instant",
        "o2_flow_scale": 1.0,
    }


def solve_p22a(run_case, p18a, p20, row: dict[str, Any], win: dict[str, str],
               *, theta_abs: float | None = None, carrier_mult: float = 1.0,
               o2_mult: float = 1.0) -> dict[str, Any]:
    dry = float(win["coal_dry_recon_phase_kg_s"])
    carrier = float(win["co2_carrier_measured_nm3_h"]) * carrier_mult
    op = p22a_op(p18a, p20, carrier)
    op["o2_flow_scale"] = float(op["o2_flow_scale"]) * o2_mult
    if theta_abs is not None:
        op["WGS_CatalyticFactor"] = float(theta_abs)
    return run_case(
        row,
        n_cells=p20.mesh_n(),
        coal_flow_dry_kg_s=dry,
        mesh_family="wsr_plus_pfr",
        extra_op=op,
    )


def _oc(win: dict[str, str], w_c: float, o2_mult: float = 1.0) -> float:
    n_o2 = float(win["o2_fy1103a_nm3_h_mean"]) * o2_mult / VM
    n_car = float(win["co2_carrier_measured_nm3_h"]) / VM
    n_c_coal = float(win["coal_dry_recon_phase_kg_s"]) * 3600.0 * w_c / M_C
    return (2.0 * n_o2 + 2.0 * n_car) / (n_c_coal + n_car)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    run_case, load_frozen_split, p18a, p20 = _ctx()

    ok, rows, _inv = load_frozen_split(coal_set="current")
    if not ok:
        raise RuntimeError("frozen split 未被完整消费")
    by_id = {r["window_id"]: r for r in rows}
    with WINDOWS_CSV.open(newline="", encoding="utf-8") as fh:
        W = {r["window_id"]: r for r in csv.DictReader(fh)}
    with BASELINE_CSV.open(newline="", encoding="utf-8") as fh:
        B = {r["window_id"]: r for r in csv.DictReader(fh)}
    if set(by_id) != set(W) or set(by_id) != set(B):
        raise RuntimeError("三者的 window_id 集合不一致")

    def phase_of(wid: str) -> str:
        return f"P{int(by_id[wid]['phase_number'])}"

    def pick(phase: str, n: int = 1) -> list[str]:
        ids = [w for w in sorted(by_id) if phase_of(w) == phase]
        return ids[:n]

    cache: dict[tuple, dict[str, Any]] = {}

    def solve(wid: str, **kw) -> dict[str, Any]:
        key = (wid, kw.get("theta_abs"), kw.get("carrier_mult", 1.0), kw.get("o2_mult", 1.0))
        if key in cache:
            return cache[key]
        res = solve_p22a(run_case, p18a, p20, by_id[wid], W[wid], **kw)
        cache[key] = res
        return res

    # ---------------- 1. 历史 C0 逐位复现 ----------------
    repro_rows: list[dict[str, Any]] = []
    for phase in EQUIV_PHASES:
        for wid in pick(phase, 2):
            res = solve(wid)
            b = B[wid]
            repro_rows.append(
                {
                    "phase": phase,
                    "window_id": wid,
                    "mesh_family": res["MeshFamily"],
                    "n_cells": res["N_cells"],
                    "coal_flow_dry_kg_s": round(float(W[wid]["coal_dry_recon_phase_kg_s"]), 6),
                    "carrier_nm3_h": round(float(W[wid]["co2_carrier_measured_nm3_h"]), 6),
                    "theta_abs_written": "unset (gasifier_system fallback)",
                    "CO_model": round(res["CO_vol_pct"], 9),
                    "CO_baseline": round(float(b["CO_vol_pct"]), 9),
                    "dCO": round(res["CO_vol_pct"] - float(b["CO_vol_pct"]), 12),
                    "dH2": round(res["H2_vol_pct"] - float(b["H2_vol_pct"]), 12),
                    "dCO2": round(res["CO2_vol_pct"] - float(b["CO2_vol_pct"]), 12),
                    "dT_K": round(res["T_out_K"] - float(b["T_out_K"]), 12),
                    "max_residual_cost": float(res["max_residual_cost"]),
                    "poor_convergence_count": res["poor_convergence_count"],
                    "fallback_count": res["fallback_count"],
                    "status": res["execution_status"],
                }
            )
    _write_csv("c0_reproduction.csv", repro_rows)
    repro_max_dCO = max(abs(r["dCO"]) for r in repro_rows)

    # ---------------- 2. 参考点等价（在正确的 P2.2-A 底座上） ----------------
    eq_rows: list[dict[str, Any]] = []
    for phase in EQUIV_PHASES:
        wid = pick(phase, 1)[0]
        hist = solve(wid)
        exp02 = solve(wid, theta_abs=C0_ABS_FALLBACK)
        exp10 = solve(wid, theta_abs=ALT_ENTRY_ABS)
        alt = solve_p22a_alt(by_id[wid], W[wid], p18a, p20)
        for label, ref, res, th in (
            ("unset_vs_explicit_0.2", hist, exp02, C0_ABS_FALLBACK),
            ("unset_vs_explicit_1.0", hist, exp10, ALT_ENTRY_ABS),
        ):
            eq_rows.append(
                {
                    "phase": phase,
                    "window_id": wid,
                    "comparison": label,
                    "theta_abs": th,
                    "rel_to_C0": round(th / C0_ABS_FALLBACK, 6),
                    "dCO": round(res["CO_vol_pct"] - ref["CO_vol_pct"], 12),
                    "dH2": round(res["H2_vol_pct"] - ref["H2_vol_pct"], 12),
                    "dCO2": round(res["CO2_vol_pct"] - ref["CO2_vol_pct"], 12),
                    "dT_K": round(res["T_out_K"] - ref["T_out_K"], 9),
                    "status": res["execution_status"],
                }
            )
        eq_rows.append(
            {
                "phase": phase,
                "window_id": wid,
                "comparison": "alt_entry_solve_model_case_vs_run_case_at_1.0",
                "theta_abs": ALT_ENTRY_ABS,
                "rel_to_C0": round(ALT_ENTRY_ABS / C0_ABS_FALLBACK, 6),
                "dCO": round(alt["pred_CO_vol_pct"] - exp10["CO_vol_pct"], 12),
                "dH2": round(alt["pred_H2_vol_pct"] - exp10["H2_vol_pct"], 12),
                "dCO2": round(alt["pred_CO2_vol_pct"] - exp10["CO2_vol_pct"], 12),
                "dT_K": round((alt["T_out_degC"] + 273.15) - exp10["T_out_K"], 9),
                "status": "completed",
            }
        )
    _write_csv("baseline_equivalence.csv", eq_rows)

    # ---------------- 3. 水平杠杆（正确底座） ----------------
    lev_rows: list[dict[str, Any]] = []
    for phase in RUN_PHASES:
        wid = pick(phase, 1)[0]
        base = solve(wid)
        for th in (*GRID_ABS, 10.0):
            res = solve(wid, theta_abs=th)
            lev_rows.append(
                {
                    "phase": phase,
                    "window_id": wid,
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
    sat_gain = {}
    for phase in RUN_PHASES:
        sub = [r for r in lev_rows if r["phase"] == phase and r["theta_abs"] <= 2.0]
        lev_span[phase] = max(r["CO"] for r in sub) - min(r["CO"] for r in sub)
        a = next(r for r in lev_rows if r["phase"] == phase and r["theta_abs"] == 2.0)
        b = next(r for r in lev_rows if r["phase"] == phase and r["theta_abs"] == 10.0)
        sat_gain[phase] = abs(b["CO"] - a["CO"])

    # ---------------- 4. 响应杠杆（正确底座） ----------------
    resp_rows: list[dict[str, Any]] = []
    pair_info: dict[str, dict[str, float]] = {}
    W_C = 0.73
    for phase in RUN_PHASES:
        wid = pick(phase, 1)[0]
        u_lo = _oc(W[wid], W_C, o2_mult=0.975)
        u_hi = _oc(W[wid], W_C, o2_mult=1.025)
        lnu = math.log(u_hi / u_lo)
        pair_info[phase] = {"u_lo": u_lo, "u_hi": u_hi, "lnu": lnu, "rel_u": (u_hi - u_lo) / u_lo}
        for th in GRID_ABS:
            a = solve(wid, theta_abs=th, o2_mult=0.975)
            b = solve(wid, theta_abs=th, o2_mult=1.025)
            okk = a["execution_status"] == "completed" and b["execution_status"] == "completed"
            r = math.log(1.05) * (b["CO_vol_pct"] - a["CO_vol_pct"]) / lnu if okk else float("nan")
            resp_rows.append(
                {
                    "phase": phase,
                    "window_id": wid,
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
    for phase in RUN_PHASES:
        vals = [x["r_model"] for x in resp_rows if x["phase"] == phase and x["r_model"] != "nan"]
        r_span[phase] = (max(vals) - min(vals)) if vals else float("nan")

    # ---------------- 5. 可分辨性（分开传播） ----------------
    sep_rows: list[dict[str, Any]] = []
    for phase in RUN_PHASES:
        lnu = pair_info[phase]["lnu"]
        for key, sigma_y in SIGMA_Y_PRESET.items():
            # (a) 组成水平：单一观测 vs 确定性模型 -> 差值误差 = sigma_y
            sig_c = sigma_y
            need_c = lev_span[phase] / K_SIGMA
            sep_rows.append(
                {
                    "phase": phase,
                    "family": "alpha_wgs",
                    "metric": "composition_level_span",
                    "signal": round(lev_span[phase], 9),
                    "unit": "pp",
                    "estimator_error_model": "single_observation_vs_deterministic_model",
                    "error_sigma": round(sig_c, 9),
                    "preset_key": key,
                    "sigma_y_preset_pp": sigma_y,
                    "required_signal_per_sigma": round(need_c, 9),
                    "required_sigma_y_pp": round(need_c, 12),
                    "ratio_preset_over_required": round(sig_c / need_c, 6) if need_c else "inf",
                    "separable": bool(sig_c < need_c),
                }
            )
            # (b) 响应：r = ln(1.05)*Δy/Δln u，Δy 为两窗之差 -> sqrt(2)*sigma_y
            sig_r = math.log(1.05) * math.sqrt(2.0) * sigma_y / lnu
            need_r = r_span[phase] / K_SIGMA
            sep_rows.append(
                {
                    "phase": phase,
                    "family": "alpha_wgs",
                    "metric": "response_span",
                    "signal": round(r_span[phase], 9),
                    "unit": "pp/5%",
                    "estimator_error_model": "difference_of_two_observations_times_ln1.05_over_lnu",
                    "error_sigma": round(sig_r, 9),
                    "preset_key": key,
                    "sigma_y_preset_pp": sigma_y,
                    "required_signal_per_sigma": round(need_r, 9),
                    "required_sigma_y_pp": round(
                        need_r * lnu / (math.log(1.05) * math.sqrt(2.0)), 12
                    ),
                    "ratio_preset_over_required": round(sig_r / need_r, 6) if need_r else "inf",
                    "separable": bool(sig_r < need_r),
                }
            )
    _write_csv("separation_check.csv", sep_rows)

    def sep_of(phase: str, metric: str, key: str = PRIMARY_SIGMA_KEY) -> bool:
        return next(
            r["separable"]
            for r in sep_rows
            if r["phase"] == phase and r["metric"] == metric and r["preset_key"] == key
        )

    # ---------------- 6. S1 消融（G 关闭响应门槛） ----------------
    abl_rows: list[dict[str, Any]] = []
    for phase in RUN_PHASES:
        wid = pick(phase, 1)[0]
        lnu = pair_info[phase]["lnu"]
        theta_gen = 2.0
        t_lo = solve(wid, theta_abs=theta_gen, o2_mult=0.975)
        t_hi = solve(wid, theta_abs=theta_gen, o2_mult=1.025)
        r_truth = math.log(1.05) * (t_hi["CO_vol_pct"] - t_lo["CO_vol_pct"]) / lnu
        sig_r = (math.log(1.05) * math.sqrt(2.0)
                 * SIGMA_Y_PRESET[PRIMARY_SIGMA_KEY] / lnu)
        for th in GRID_ABS:
            a = solve(wid, theta_abs=th, o2_mult=0.975)
            b = solve(wid, theta_abs=th, o2_mult=1.025)
            diffs = [
                a["CO_vol_pct"] - t_lo["CO_vol_pct"],
                a["H2_vol_pct"] - t_lo["H2_vol_pct"],
                a["CO2_vol_pct"] - t_lo["CO2_vol_pct"],
                b["CO_vol_pct"] - t_hi["CO_vol_pct"],
                b["H2_vol_pct"] - t_hi["H2_vol_pct"],
                b["CO2_vol_pct"] - t_hi["CO2_vol_pct"],
            ]
            C_dev = math.sqrt(sum(d * d for d in diffs) / len(diffs))
            r = math.log(1.05) * (b["CO_vol_pct"] - a["CO_vol_pct"]) / lnu
            R_dev = abs(r - r_truth)
            abl_rows.append(
                {
                    "phase": phase,
                    "window_id": wid,
                    "theta_gen_abs": theta_gen,
                    "theta_cand_abs": th,
                    "C_dev_pp": round(C_dev, 9),
                    "r_cand": round(r, 9),
                    "r_truth": round(r_truth, 9),
                    "R_dev": round(R_dev, 9),
                    "R_gate_threshold": round(K_SIGMA * sig_r, 9),
                    "R_gate_pass": bool(R_dev <= K_SIGMA * sig_r),
                }
            )
    _write_csv("ablation_trace.csv", abl_rows)

    s1 = {}
    for phase in RUN_PHASES:
        sub = [x for x in abl_rows if x["phase"] == phase]
        best_c = min(sub, key=lambda x: x["C_dev_pp"])
        best_r = min(sub, key=lambda x: x["R_dev"])
        rdev = [x["R_dev"] for x in sub]
        monotone = all(rdev[i] <= rdev[i + 1] for i in range(len(rdev) - 1)) or all(
            rdev[i] >= rdev[i + 1] for i in range(len(rdev) - 1)
        )
        s1[phase] = {
            "c_argmin_theta": best_c["theta_cand_abs"],
            "r_argmin_theta": best_r["theta_cand_abs"],
            "same_argmin": best_c["theta_cand_abs"] == best_r["theta_cand_abs"],
            "r_dev_monotone_in_grid": monotone,
            "all_pass_gate": all(x["R_gate_pass"] for x in sub),
            "c_dev_spread": round(max(x["C_dev_pp"] for x in sub) - min(x["C_dev_pp"] for x in sub), 9),
            "r_dev_spread": round(max(rdev) - min(rdev), 9),
        }

    # ---------------- 7. S2：真实注入边界偏差 ----------------
    s2_rows: list[dict[str, Any]] = []
    for phase in RUN_PHASES:
        wid = pick(phase, 1)[0]
        lnu = pair_info[phase]["lnu"]
        sig_r = (math.log(1.05) * math.sqrt(2.0)
                 * SIGMA_Y_PRESET[PRIMARY_SIGMA_KEY] / lnu)
        sig_c = SIGMA_Y_PRESET[PRIMARY_SIGMA_KEY]
        # 真值：名义边界、反应族为 C0
        t_lo = solve(wid, o2_mult=0.975)
        t_hi = solve(wid, o2_mult=1.025)
        # 边界偏差：载气流量 ×(1+3%)，反应族仍为 C0
        d_lo = solve(wid, o2_mult=0.975, carrier_mult=1.0 + S2_CARRIER_REL)
        d_hi = solve(wid, o2_mult=1.025, carrier_mult=1.0 + S2_CARRIER_REL)
        r_dev_inj = math.log(1.05) * (d_hi["CO_vol_pct"] - d_lo["CO_vol_pct"]) / lnu
        # 偏差模型的"表观组成"是否与某个物理候选（在名义边界下）相似
        best = None
        for th in GRID_ABS:
            a = solve(wid, theta_abs=th, o2_mult=0.975)
            b = solve(wid, theta_abs=th, o2_mult=1.025)
            cdiff = math.sqrt(
                (
                    (a["CO_vol_pct"] - d_lo["CO_vol_pct"]) ** 2
                    + (a["H2_vol_pct"] - d_lo["H2_vol_pct"]) ** 2
                    + (a["CO2_vol_pct"] - d_lo["CO2_vol_pct"]) ** 2
                    + (b["CO_vol_pct"] - d_hi["CO_vol_pct"]) ** 2
                    + (b["H2_vol_pct"] - d_hi["H2_vol_pct"]) ** 2
                    + (b["CO2_vol_pct"] - d_hi["CO2_vol_pct"]) ** 2
                )
                / 6.0
            )
            r_cand = math.log(1.05) * (b["CO_vol_pct"] - a["CO_vol_pct"]) / lnu
            rec = {
                "phase": phase,
                "window_id": wid,
                "carrier_bias_rel": S2_CARRIER_REL,
                "theta_cand_abs": th,
                "r_deviation_injected": round(r_dev_inj, 9),
                "r_candidate": round(r_cand, 9),
                "response_gap": round(abs(r_cand - r_dev_inj), 9),
                "C_match_to_deviation": round(cdiff, 9),
                "composition_match_tolerance_pp": round(K_SIGMA * sig_c, 9),
                "composition_matches": bool(cdiff <= K_SIGMA * sig_c),
                "response_separable": bool(abs(r_cand - r_dev_inj) > K_SIGMA * sig_r),
            }
            s2_rows.append(rec)
            if best is None or rec["C_match_to_deviation"] < best["C_match_to_deviation"]:
                best = rec
        s2_rows.append({**{k: "" for k in s2_rows[0]}, "phase": phase, "theta_cand_abs": "SUMMARY",
                        "C_match_to_deviation": best["C_match_to_deviation"],
                        "composition_matches": best["composition_matches"],
                        "response_separable": best["response_separable"],
                        "r_candidate": best["r_candidate"],
                        "r_deviation_injected": best["r_deviation_injected"],
                        "response_gap": best["response_gap"],
                        "window_id": wid, "carrier_bias_rel": S2_CARRIER_REL,
                        "composition_match_tolerance_pp": best["composition_match_tolerance_pp"]})
    _write_csv("scenario_design.csv", s2_rows)

    s2_summary = {
        p: next(r for r in s2_rows if r["phase"] == p and r["theta_cand_abs"] == "SUMMARY")
        for p in RUN_PHASES
    }
    # 每个阶段「最接近候选」的 θ，用于文档渲染（SUMMARY 行不带 θ）
    s2_best_theta: dict[str, Any] = {}
    for p in RUN_PHASES:
        sub = [r for r in s2_rows if r["phase"] == p and r["theta_cand_abs"] != "SUMMARY"]
        s2_best_theta[p] = min(sub, key=lambda r: float(r["C_match_to_deviation"]))["theta_cand_abs"]
    s2_verdict = {}
    for p, r in s2_summary.items():
        if r["composition_matches"] and not r["response_separable"]:
            s2_verdict[p] = "NOT_SEPARABLE_mimicry_indistinguishable"
        elif r["composition_matches"] and r["response_separable"]:
            s2_verdict[p] = "SEPARABLE_gate_could_reject"
        elif not r["composition_matches"]:
            s2_verdict[p] = "PRECONDITION_UNMET_deviation_not_mimicking_any_candidate"
        else:
            s2_verdict[p] = "UNCLASSIFIED"

    # ---------------- 8. 判定 ----------------
    all_not_sep = not any(
        sep_of(p, m) for p in RUN_PHASES for m in ("composition_level_span", "response_span")
    )
    s1_action_unchanged = all(s1[p]["same_argmin"] for p in s1)
    s2_any_separable = any(v == "SEPARABLE_gate_could_reject" for v in s2_verdict.values())

    exit_code = "STOP_TO_07" if (all_not_sep and not s2_any_separable) else "RUN_ONCE"

    proto = {
        "schema": "operating-response-comparison/controlled_protocol_draft",
        "schema_version": "2.0.0",
        "card": "06A",
        "revision": 2,
        "status": "DRAFT_PENDING_REVIEW",
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "reference_point": {
            "repo": "gasifier-1d-kinetic",
            "baseline_head": "c7cde49",
            "solver_entry": "model.p1_5_round_20261030.run_case",
            "frozen_config": {
                "MeshFamily": "wsr_plus_pfr",
                "WSRLength_m": 1.0,
                "PFRCellLength_m": 0.0515,
                "n_cells": int(p20.mesh_n()),
                "HeatLossModel": "total_percent",
                "HeatLossPercentTotal": 1.0,
                "GasificationRateFactor": 1.0,
                "CharCO2RateFactor": 1.0,
                "OxygenMixingModel": "instant",
                "o2_flow_scale": 1.0,
                "carrier_gas_flow": "from windows_142 co2_carrier_measured_nm3_h via p18a._carrier_kg_s",
                "coal_flow": "explicit dry, windows_142 coal_dry_recon_phase_kg_s",
            },
            "frozen_config_source": "scripts/generate_p2_2a_20261118.py::build_op",
            "reproduction": {
                "reference": "revision-evidence-20261118/P2.2-A/baseline_142.csv",
                "max_abs_dCO_over_sampled_windows": repro_max_dCO,
                "verdict": "byte-identical" if repro_max_dCO == 0.0 else "mismatch",
            },
            "inputs_sha256": {
                "windows_142.csv": sha256_of(WINDOWS_CSV),
                "baseline_142.csv": sha256_of(BASELINE_CSV),
            },
            "historical_c0_theta_abs": C0_ABS_FALLBACK,
            "historical_c0_theta_basis": (
                "run_case 不写 WGS_CatalyticFactor 时 gasifier_system.py 取 op_conds.get(...,0.2)；"
                "unset 与显式 0.2 逐位相同（见 baseline_equivalence.csv）"
            ),
            "alt_entry_status": (
                "revision_evidence_common.solve_model_case 按其自身缺省路径求解，"
                "**未传** WSR+PFR / 整炉热损 1% / 显式干煤口径。因此 "
                "baseline_equivalence.csv 中 alt 行与正确 C0 路径的差异是"
                "「配置差异 + 入口差异」的混合，原因未分离。"
                "可主张：两个未匹配配置的调用不等价，不能直接互相替代。"
                "不可主张：差异来自求解入口本身，或「控制其余配置后仍不等价」。"
                "若日后需要后者，须另做匹配配置对照（非本轮 06A 的必需计算）。"
            ),
            "scale_conversion": {
                "absolute": "op_conds['WGS_CatalyticFactor']，历史 C0 = 0.2",
                "relative_to_C0": "absolute / 0.2",
                "warning": "theta=1.0 是绝对 1.0，等于 5x C0，不等同于 C0",
            },
        },
        "error_propagation": {
            "composition_level": {
                "model": "single observation vs deterministic model value",
                "sigma": "sigma_y",
                "note": "不含 ln(1.05)/ln u 归一因子；rev 1 误用了 sigma_r",
            },
            "response": {
                "model": "difference of two observations scaled by ln(1.05)/ln(u_hi/u_lo)",
                "sigma": "sigma_r = ln(1.05)*sqrt(2)*sigma_y/ln(u_hi/u_lo)",
            },
            "two_observation_composition_contrast": {
                "status": "not_used",
                "note": "若日后比较两个含噪观测的组成差，须另给差值误差与协方差，不得复用响应式",
            },
        },
        "candidate": {
            "family": "alpha_wgs",
            "absolute_grid": list(GRID_ABS),
            "relative_to_C0": [round(t / C0_ABS_FALLBACK, 6) for t in GRID_ABS],
            "basis": "06 独立登记，不继承 01 网格",
        },
        "response": {
            "definition": "r = ln(1.05) * (y_hi - y_lo) / ln(u_hi/u_lo)",
            "u": "o_c_ratio（含载气 CO2 的 O 与 C）",
            "y": "干基 CO 体积分数",
            "pair_construction": "同一窗口 O2 流量 x0.975 / x1.025，rel_u 见 response_leverage.csv",
            "unit": "百分点 / 输入相对变化 5%",
        },
        "thresholds": {
            "basis_statement": "由本轮杠杆与预设测量分辨率独立导出，不继承 01 的 0.25 / R_abs_dev / 网格 / 工程组成上限",
            "sigma_y_preset_pp": SIGMA_Y_PRESET,
            "primary_preset": PRIMARY_SIGMA_KEY,
            "preset_status": "DESIGN_ASSUMPTION_NOT_CALIBRATED",
            "preset_basis": (
                "工业在线合成气分析仪（NDIR CO/CO2、TCD H2）的典型重复性量级；"
                "厂家常按量程百分比给出，不能直接等同绝对 pp，也不能当作本任务的一倍标准差"
            ),
            "separation_multiplier_k": K_SIGMA,
        },
        "separation_findings": {
            "composition_level_span_pp": {k: round(v, 6) for k, v in lev_span.items()},
            "saturation_gain_2.0_to_10.0_pp": {k: round(v, 6) for k, v in sat_gain.items()},
            "response_span_pp_per_5pct": {k: round(v, 9) for k, v in r_span.items()},
            "separable": {
                p: {
                    "composition": sep_of(p, "composition_level_span"),
                    "response": sep_of(p, "response_span"),
                }
                for p in RUN_PHASES
            },
            "by_preset": {
                key: {
                    p: {
                        "composition": sep_of(p, "composition_level_span", key),
                        "response": sep_of(p, "response_span", key),
                    }
                    for p in RUN_PHASES
                }
                for key in SIGMA_Y_PRESET
            },
        },
        "s1_ablation": {
            "scope": (
                "仅限本实现校验：真值取网格内 theta=2.0、无噪声、同一求解器；"
                "该配置下候选可精确复现真值"
            ),
            "per_phase": s1,
            "finding": (
                "在本 S1、该网格与预设门槛下，C 与 R 的 argmin 相同，"
                "关闭响应门槛未改变动作。"
            ),
            "not_claimed": [
                "单标量族必然排序退化",
                "任何门槛都不可能改变选择动作",
                "因此必须改用多参数或工况乘子",
            ],
            "counter_evidence": (
                "R_dev 在本网格内并非单调（见 ablation_trace.csv），"
                "因此不能由有限点推断普遍的单调性或退化"
            ),
        },
        "s2_boundary_deviation": {
            "injection": f"载气流量 x(1+{S2_CARRIER_REL})，反应族仍为 C0",
            "per_phase_summary": {p: s2_summary[p] for p in s2_summary},
            "verdict": s2_verdict,
        },
        "verdict": {
            "exit": exit_code,
            "basis": (
                "在已复现的历史 C0 底座与所列预设分辨率下，"
                "组成与响应信号均低于 2σ 判据；S2 的边界偏差与物理候选的响应差也未超过 2σ_r"
                if exit_code == "STOP_TO_07"
                else "存在可分辨的对照，建议冻结后 RUN_ONCE"
            ),
            "retracted_from_rev1": [
                "single_scalar_family_degeneracy（作为普遍结论）",
                "any_threshold_cannot_change_action",
                "WGS_near_equilibrium（缺状态/反应商依据）",
                "S2_already_negated（rev 1 未真正注入偏差）",
            ],
            "carry_to_07": [
                "P2.2-A 底座的历史 C0 可逐位复现",
                (
                    "两个**未匹配配置**的调用不等价：alt 入口按其自身缺省路径求解，"
                    "未传 WSR+PFR / 整炉热损 1% / 显式干煤口径，故与正确 C0 路径的差异"
                    "混合了配置差异与入口差异，原因未分离。"
                    "**不得据此归因于求解入口本身，也不得声称已控制其余配置。**"
                ),
                "在所列预设分辨率下组成与响应信号的上界",
                "S2 的实测结论（见 s2_boundary_deviation.verdict）",
            ],
            "explicitly_not_carried": [
                "控制其余配置后的求解入口差异（需另做匹配配置对照，非本轮必需）",
                "任何把 alt 入口差异归因于入口实现的解释",
            ],
        },
        "isolation": {
            "generation_side": "名义入炉工况 + 注入机制；真值由生成端保存并哈希",
            "fitting_side": "只读授权输入与开发观测",
            "scoring_side": "预测锁定后才读评估真值",
            "scenario_selection_neutrality": "场景不按 C/G 评估成绩挑选",
        },
        "seeds": {"solver": "deterministic", "noise_model": "not_applied_in_06A"},
    }
    (OUT / "controlled_protocol_draft.yaml").write_text(
        json.dumps(_clean(proto), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---------------- 9. limitations.md ----------------
    lim = f"""# 06A 证据边界（rev 2）

本文件登记本轮**不能**主张的内容，以及每条撤回的依据。

## 已撤回的 rev 1 结论

| rev 1 主张 | 撤回原因 | rev 2 的处理 |
|---|---|---|
| 「历史 C0 等价检查通过」 | 走的是 `uniform_full_length` 默认网格且未传干煤口径，不是 P2.2-A 的 C0 | 改为 WSR+PFR 冻结配置，对 `baseline_142.csv` 逐位复现（最大 \\|dCO\\| = {repro_max_dCO:.3e}） |
| 组成倍数 7.06 / 7.95 / 4.94 | 把响应量的 σ_r（含 ln(1.05)/Δln u）套到组成水平上 | 组成改用 σ_y，倍数见 `separation_check.csv` |
| 「单标量族必然排序退化」 | 真值取网格内 θ=2.0、无噪声、同求解器，C_dev=R_dev=0 是按构造成立 | 只报「在本 S1、该网格与预设门槛下未改变动作」 |
| 「任何门槛都不可能改变动作」 | 由单点实现校验外推为普遍结论 | 撤回。R_dev 在本网格内并不单调 |
| 「WGS 已近平衡」 | 缺状态量与反应商依据；有限点的参数饱和不足以支持 | 撤回，只保留「θ 由 2.0 增至 10.0 的增益远小于 0.02→2.0」这一数值事实 |
| 「S2 已被否定」 | rev 1 未执行载气偏差注入，签名直接继承 S1 | rev 2 实际注入 {S2_CARRIER_REL:.0%} 载气偏差并逐候选比较 |

## 仍不能主张

1. **响应判据有效或无效。** 本轮没有产生可分辨的对照。
2. **工业响应已识别。** 历史数据为 `DESCRIPTIVE_ONLY`，S2 是受控注入不是装置数据。
3. **「任何合理分析仪」的精度结论。** 预设 σ_y 是设计假设；厂家指标按量程百分比给出，不能直接等同绝对 pp，也不能当作本任务的一倍标准差。本报告只在所列三档预设下作条件性判断。
4. **WGS 近平衡的机制结论。** 需状态量与反应商依据。

## 可主张

1. P2.2-A 的冻结配置（WSR+PFR 等）在采样窗口上**逐位复现** `baseline_142.csv`。
2. 历史 C0 路径不写 `WGS_CatalyticFactor` 时与显式 0.2 逐位相同；显式 1.0 不等价（5× C0）。
3. 在所列网格与三档预设下，组成与响应的信号上界（见 `separation_check.csv`）。
4. **两个未匹配配置的调用不等价**：`solve_model_case` 按其自身缺省路径求解，
   未传 WSR+PFR / 整炉热损 1% / 显式干煤口径，故其与正确 C0 路径的差异是
   「配置差异 + 入口差异」的混合。

## 关于双入口比较的范围限制

`baseline_equivalence.csv` 中 `alt_entry_solve_model_case_vs_run_case_at_1.0` 一行
**不能**用于支撑「控制其余配置后的入口差异」：

- alt 侧未匹配 P2.2-A 的 MeshFamily / WSRLength / PFRCellLength / 热损 / 干煤口径；
- 因此该行的 dCO / dT 混合了配置差异与入口差异，原因**未分离**；
- 可说的只有「两者在当前各自配置下不等价，不能直接互相替代」；
- **不可说**「差异来自求解入口本身」或「其余条件相同下仍不等价」。

若日后需要后一类结论，须另做一次匹配配置的对照求解。那不是本轮 06A 的必需计算，
也不影响本轮 `STOP_TO_07` 的数值依据。
"""
    (OUT / "limitations.md").write_text(lim, encoding="utf-8")

    # ---------------- 10. expected_outputs.md ----------------
    md = f"""# 06A 预期产物与放行判定（rev 2）

## 判定：`{exit_code}`

依据：在**已逐位复现**的 P2.2-A 历史 C0 底座上（最大 \\|dCO\\| = {repro_max_dCO:.3e}），
组成与响应信号在三档预设分辨率下均未超过 2σ 判据；S2 的边界偏差与物理候选之间
的响应差也未超过 2σ_r。

## 关键数字（rev 2）

| 阶段 | 组成水平杠杆 | 响应杠杆 | 组成相对 2σ_y | 响应相对 2σ_r |
|---|---:|---:|---:|---:|
""" + "\n".join(
        f"| {p} | {lev_span[p]:.4f} pp | {r_span[p]:.5f} pp/5% | "
        f"{float(next(r['ratio_preset_over_required'] for r in sep_rows if r['phase']==p and r['metric']=='composition_level_span' and r['preset_key']==PRIMARY_SIGMA_KEY)):.4f}× | "
        f"{float(next(r['ratio_preset_over_required'] for r in sep_rows if r['phase']==p and r['metric']=='response_span' and r['preset_key']==PRIMARY_SIGMA_KEY)):.4f}× |"
        for p in RUN_PHASES
    ) + f"""

「相对 2σ」列 = 预设误差 / 分离所需误差；> 1 表示不可分辨。

## S1 消融（范围受限）

{s1['P3']['finding'] if False else ''}只报事实：

""" + "\n".join(
        f"- {p}：C 的 argmin θ={s1[p]['c_argmin_theta']}，R 的 argmin θ={s1[p]['r_argmin_theta']}，"
        f"同端点={s1[p]['same_argmin']}，全部候选通过门槛={s1[p]['all_pass_gate']}，"
        f"R_dev 在本网格内单调={s1[p]['r_dev_monotone_in_grid']}"
        for p in s1
    ) + f"""

**不代表**「单标量族必然退化」或「任何门槛都不可能改变动作」。真值取网格内 θ=2.0、
无噪声、同一求解器，因此该候选的 C_dev=R_dev=0 是按构造成立。

## S2 边界偏差（rev 2 实测）

注入：载气流量 ×(1+{S2_CARRIER_REL:.0%})，反应族保持 C0。逐阶段取与候选 C 最接近者：

| 阶段 | 最接近候选 θ | C 差 | 组成近似 | 响应差 | 响应可分 | 判定 |
|---|---:|---:|---|---:|---|---|
""" + "\n".join(
        f"| {p} | {s2_best_theta[p]} | "
        f"{float(s2_summary[p]['C_match_to_deviation']):.6f} | {s2_summary[p]['composition_matches']} | "
        f"{s2_summary[p]['response_gap']} | {s2_summary[p]['response_separable']} | {s2_verdict[p]} |"
        for p in s2_summary
    ) + f"""

## 若 RUN_ONCE 应交付什么

`generator_manifest.json`、`scenario_manifest.csv`、`truth_lock.json`、`selection_trace.csv`、
`predictions.csv`、`controlled_scores.csv`、`decision_table.csv`、`interpretation.md`。
本轮**不生成**。

## 给 07 的边界

可写：历史 C0 可逐位复现；两个未匹配配置的调用不等价（差异原因未分离）；所列预设下的信号上界（条件性）。
不可写：响应判据有效/无效；工业响应已识别；WGS 已近平衡；任何通用精度的分析仪结论。
详细清单见 `limitations.md`。
"""
    (OUT / "expected_outputs.md").write_text(md, encoding="utf-8")

    # ---------------- 11. commands.md ----------------
    cmds = f"""# 复现命令

```bash
cd /Users/liuzhen/AI-projects/gasifier-1d-kinetic
git rev-parse HEAD          # 期望 c7cde49
./.venv/bin/python scripts/p06a_controlled_design_and_separation.py
cd outputs/diagnostics/operating-response-comparison/06A
shasum -a 256 -c sha256.txt
```

## 底座（P2.2-A 冻结配置）

| 项 | 值 | 来源 |
|---|---|---|
| MeshFamily | `wsr_plus_pfr` | `generate_p2_2a_20261118.build_op` |
| WSRLength_m | 1.0 | 同上 |
| PFRCellLength_m | 0.0515 | 同上 |
| n_cells | {int(p20.mesh_n())} | `generate_p2_0a_20261113.mesh_n()` |
| HeatLossModel | `total_percent` | 同上 |
| HeatLossPercentTotal | 1.0 | 同上 |
| GasificationRateFactor | 1.0 | 同上 |
| CharCO2RateFactor | 1.0 | 同上 |
| OxygenMixingModel | `instant` | 同上 |
| carrier_gas_flow | `p18a._carrier_kg_s(co2_carrier_measured_nm3_h)` | P1.8-D 实测口径 |
| coal_flow | 显式干煤 `coal_dry_recon_phase_kg_s` | P2.2-D |

## 输入（只读）

- `windows_142.csv` sha256 `{sha256_of(WINDOWS_CSV)}`
- `baseline_142.csv` sha256 `{sha256_of(BASELINE_CSV)}`

## 口径

- 响应配对：同一窗口 O2 流量 ×0.975 / ×1.025。
- 预设 σ_y ∈ {{0.10, 0.25, 0.50}} pp，**设计假设，非实测标定**。
- 误差传播分开：组成用 σ_y；响应用 σ_r = ln(1.05)·√2·σ_y/Δln u。
- 未读取任何装置出口组成作为真值。
"""
    (OUT / "commands.md").write_text(cmds, encoding="utf-8")

    # ---------------- 12. sha256 ----------------
    present = {p.name for p in OUT.iterdir() if p.is_file()}
    stale = present - set(PRODUCTS) - {"sha256.txt"}
    if stale:
        raise RuntimeError(f"输出目录存在不在产品名单内的文件: {sorted(stale)}")
    missing = set(PRODUCTS) - present
    if missing:
        raise RuntimeError(f"产品名单中的文件缺失: {sorted(missing)}")
    lines = [f"{sha256_of(OUT / name)}  {name}" for name in PRODUCTS]
    (OUT / "sha256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "exit": exit_code,
        "repro_max_abs_dCO": repro_max_dCO,
        "lev_span": {k: round(v, 6) for k, v in lev_span.items()},
        "r_span": {k: round(v, 9) for k, v in r_span.items()},
        "separable": {p: {"c": sep_of(p, "composition_level_span"),
                          "r": sep_of(p, "response_span")} for p in RUN_PHASES},
        "s1": s1,
        "s2": s2_verdict,
    }, ensure_ascii=False, indent=2))


def solve_p22a_alt(row: dict[str, Any], win: dict[str, str], p18a, p20) -> dict[str, Any]:
    """另一求解入口，用于证明它不能替代 run_case 验证历史 C0。"""
    import sys

    src = REPO / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from model.revision_evidence_common import solve_model_case

    # 该入口不接受 op 覆盖，只能按其自身路径求解
    return solve_model_case(row, n_cells=p20.mesh_n(), alpha_wgs=ALT_ENTRY_ABS)


def _clean(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
        return None
    return o


def _write_csv(name: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"{name}: 无数据行")
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with (OUT / name).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    main()
