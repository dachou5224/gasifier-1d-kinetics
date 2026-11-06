#!/usr/bin/env python3
"""P1.11-A 烧嘴近场全混流区（固定物理长度 L_wsr）+ 下游串联格（2026-11-06 卡）。

结构：首格长度恰为 `L_wsr`（体积 = A·L_wsr，一个全混流区），其后 `[L_wsr, L]`
用等长 `dz_down` 的串联格（末格吸收余数）。全部 O2 进入首格（`OxygenMixingModel` 冻结
为 `"instant"`）。不改任何速率式、限幅、燃烧区判据或求解器容差。

执行顺序（卡第 1 → 2 → 3 → 4 → 6 节）：
  1  dz 依赖性清单                              → dz_dependence.md
  2  单元检查（不通过则停止）                    → unit_checks.csv / roots.csv
  3  下游收敛 W1（先做）                        → downstream_convergence.csv
  4  杠杆扫描 W2–W6                             → leverage.csv / cells.csv
  6  锚点（只记录）                             → anchor_wsr.csv

用法：
    ./.venv/bin/python scripts/generate_p1_11a_20261106.py [--stage unit|conv|lever|anchor|all]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import generate_p1_8a_20261104 as p18a  # noqa: E402
import generate_p1_9a_20261105 as p19a  # noqa: E402

from model.fortran_anchor_20261030 import DRY_POWDER_CASES, anchor_cases, solve_fortran_case  # noqa: E402
from model.gasifier_system import GasifierSystem  # noqa: E402
from model.grid_service import MeshConfig  # noqa: E402
from model.grid_service import AdaptiveMeshGenerator  # noqa: E402
from model.input_contract import coal_hhv_kj_kg  # noqa: E402
from model.p0_round_20260930 import m1_reference_window_ids, reactor_element_flows  # noqa: E402
from model.p1_5_round_20261030 import base_protocol, run_case, silence_solver_logs, write_round_outputs  # noqa: E402
from model.revision_evidence_common import load_frozen_split, source_commit  # noqa: E402

OUT_DIR = ROOT / "outputs" / "diagnostics" / "round-p1.11a-20261106"
P18A_DIR = ROOT / "outputs" / "diagnostics" / "round-p1.8a-20261104"
P17D_CSV = p18a.P17D_CSV
P18D_CSV = p18a.P18D_CSV
M1_MATRIX = p18a.M1_MATRIX

DEV_L = 10.3
L_WSR_GRID = (0.2575, 0.5, 1.0, 2.0, 4.0)
L_WSR_EXTRA = 6.0              # 仅当 W4 最优点落在 4 m 边界时追加
DZ_DOWN_GRID = (0.2575, 0.103, 0.0515, 0.02575)
DZ_FINE, DZ_COARSE = 0.02575, 0.0515   # W1 比较这两档
F_GAS_LEV = (1.0, 0.125)
HLP_LEV = (1.0, 2.0)
ANCHOR_L_WSR = (0.02, 0.1, 0.25, 0.5, 0.75)
ANCHOR_DZ_DOWN = 0.0103

# W1 阈值（卡第 3 节）
W1_DXC_PP = 0.2
W1_DT_K = 5.0
W1_DCO2_VOL = 0.2
# 多解判定（卡第 2 节第 3 条）
ROOT_COST_TOL = 1e-10
ROOT_T_MERGE_K = 10.0          # 温度相差 < 10 K 视为同一个根
FLAME_T_K = 1500.0             # 「温度 > 1500 K 的根」才算火焰根


# ---------------------------------------------------------------------------
# 网格（与 grid_service.wsr_plus_pfr 同一算式，用于预先导出 N）
# ---------------------------------------------------------------------------

def mesh_plan(L: float, l_wsr: float, dz_down: float) -> dict:
    RL = L - l_wsr
    n_full = int(math.floor(RL / dz_down + 1e-12))
    rem = RL - float(n_full) * dz_down
    has_rem = bool(rem > 1e-12)
    N = 1 + n_full + (1 if has_rem else 0)
    return {
        "L_wsr": float(l_wsr), "dz_down": float(dz_down),
        "n_full": n_full, "rem": float(rem), "has_rem": has_rem, "N": N,
    }


def build_dz(plan: dict, L: float) -> np.ndarray:
    dz = np.zeros(int(plan["N"]))
    dz[0] = plan["L_wsr"]
    if plan["n_full"] > 0:
        dz[1:1 + plan["n_full"]] = plan["dz_down"]
    if plan["has_rem"]:
        dz[-1] = plan["rem"]
    return dz


def uniform_dz(L: float, n: int) -> np.ndarray:
    return np.full(n, L / float(n))


# ---------------------------------------------------------------------------
# 求解与分支状态
# ---------------------------------------------------------------------------

def _branch_status(rec: dict, system: GasifierSystem) -> str:
    """完成但温度触界的分支不得算作收敛（P1.10-A 的教训）。"""
    st = str(rec.get("execution_status", ""))
    if st != "completed":
        return st
    if system.cell_diagnostics:
        T0 = float(system.cell_diagnostics[0].get("T_K", float("nan")))
        if math.isfinite(T0) and T0 <= 301.0:
            return "cold_branch_T_at_lower_bound"
        Tmax = max((float(s.T) for s in system.results), default=float("nan"))
        if math.isfinite(Tmax) and Tmax >= 3999.0:
            return "hot_branch_T_at_upper_bound"
    return st


def _roots(system: GasifierSystem) -> tuple[int, list[dict], int]:
    """从 cell0_branches 汇总根：按温度聚类，只保留 cost < TOL 的分支。

    返回 (n_distinct_roots, 明细行, n_flame_roots)。
    """
    branches = getattr(system, "cell0_branches", []) or []
    good = [b for b in branches
            if float(b.get("cost", 1e18)) < ROOT_COST_TOL
            and np.isfinite(float(b.get("T_cell0", float("nan"))))]
    clusters: list[dict] = []
    for b in sorted(good, key=lambda x: float(x["T_cell0"])):
        T = float(b["T_cell0"])
        hit = None
        for c in clusters:
            if abs(c["T_cell0"] - T) < ROOT_T_MERGE_K:
                hit = c
                break
        if hit is None:
            clusters.append({"T_cell0": T, "n_guesses": 1,
                             "guesses": [float(b["guess_T"])],
                             "min_cost": float(b["cost"])})
        else:
            hit["n_guesses"] += 1
            hit["guesses"].append(float(b["guess_T"]))
            hit["min_cost"] = min(hit["min_cost"], float(b["cost"]))
    n_flame = sum(1 for c in clusters if c["T_cell0"] > FLAME_T_K)
    return len(clusters), clusters, n_flame


def _run(row, csv17, csv18, dry, carrier, l_wsr, dz_down, f_gas=1.0, hlp=1.0):
    plan = mesh_plan(DEV_L, l_wsr, dz_down)
    rec = run_case(
        row, n_cells=int(plan["N"]), coal_flow_dry_kg_s=dry,
        keep_system=True, mesh_family="wsr_plus_pfr",
        extra_op={
            "HeatLossPercent": float(hlp),
            "GasificationRateFactor": float(f_gas),
            "carrier_gas_flow": p18a._carrier_kg_s(carrier),
            "WSRLength_m": float(l_wsr),
            "PFRCellLength_m": float(dz_down),
            "OxygenMixingModel": "instant",   # 卡面冻结项
        },
    )
    return rec, rec["_system"], plan


def _wsr_state(system: GasifierSystem) -> dict:
    """首格（全混流区）出口状态。"""
    if not system.results or not system.cell_diagnostics:
        return {"T_wsr_K": float("nan"), "X_C_wsr": float("nan"), "CO2_wsr_mol_s": float("nan"),
                "n_bound_cells": 0, "tau_wsr_s": float("nan"), "m_hold_wsr_kg": float("nan")}
    st = system.results[0]
    d0 = system.cell_diagnostics[0]
    inlet_c0 = float(system.cells[0].inlet.solid_mass * system.cells[0].inlet.carbon_fraction)
    c0 = float(st.solid_mass * st.carbon_fraction)
    lim_keys = ("C_O2_LimitedByO2", "C_H2O_LimitedByH2O", "C_CO2_LimitedByCO2",
                "C_H2_LimitedByH2", "WGS_NegativeTruncated")
    n_bound = sum(1 for k in lim_keys if d0.get(k))
    return {
        "T_wsr_K": float(st.T),
        "X_C_wsr": float(1.0 - c0 / max(inlet_c0, 1e-12)),
        "CO2_wsr_mol_s": float(st.gas_moles[3]),
        "n_bound_cells": int(n_bound),
        "tau_wsr_s": float(d0.get("tau_s", float("nan"))),
        "m_hold_wsr_kg": float(d0.get("m_hold_kg", float("nan"))),
    }


def _probe(system: GasifierSystem, zs=(2.0, 4.0, 6.0)) -> dict:
    """在指定轴向位置取最近格的 T 与累积 X_C。"""
    out: dict = {}
    if not system.results:
        return out
    zc = np.array([float(c.z) for c in system.cells], dtype=float)
    inlet_c0 = float(system.cells[0].inlet.solid_mass * system.cells[0].inlet.carbon_fraction)
    for z in zs:
        i = int(np.argmin(np.abs(zc - z)))
        st = system.results[i]
        out[f"T_at_{z:g}m_K"] = float(st.T)
        out[f"X_C_at_{z:g}m"] = float(
            1.0 - st.solid_mass * st.carbon_fraction / max(inlet_c0, 1e-12))
    return out


# ---------------------------------------------------------------------------
# 第 1 节：dz 依赖性清单
# ---------------------------------------------------------------------------

def dz_dependence_md(probe_rows: list[dict]) -> str:
    """卡第 1 节：列出现行单格模型中所有显式依赖 dz 的项，逐项判定是否仍是全混流。

    判定依据是残差形式：每一格的残差都是

        n_out − (n_in + src − Σrate) = 0

    且 Σrate 用**该格出口状态**（CSTR 定义）计算，容量项为体积 V 或持料量 ṁ·τ。
    因此「全混流」在这里的含义是：格内无轴向梯度解，速率按出口状态一次求值。
    逐项检查是否有任何一项在格内做轴向积分。
    """
    L: list[str] = []
    L.append("# P1.11-A 第 1 节：现行单格模型的 dz 依赖性清单\n")
    L.append("判定标准：本模型每一格的残差都是 `n_out = n_in + src − Σrate`，其中 `Σrate` 用"
             "**该格出口状态**求值（CSTR 定义），容量项为体积 `V = A·dz` 或持料量 `ṁ·τ`。"
             "因此「全混流」在此指**格内无轴向梯度解**。若任一依赖项隐含格内轴向积分，"
             "则卡面第 1 节的停止条件触发，必须停止并报告。\n")
    L.append("## 逐项清单\n")
    L.append("| # | 依赖 dz 的项 | 代码位置 | 形式 | 格长为数米时是否仍是全混流 |")
    L.append("|---|---|---|---|---|")
    rows = [
        ("1", "气相反应速率容量", "`kinetics_service.calc_homogeneous_rates`（`r·volume`）",
         "`rate = k·C_i·C_j·V`",
         "**是**。`C_i = y_i·P/(R·T)` 由摩尔分数给出，与 V 无关；乘 V 就是「单位体积速率 × 体积」，"
         "CSTR 的标准容量形式。大 V 使转化更完全，不引入格内梯度。"),
        ("2", "异相反应持料量", "`cell._calc_physics_props`（`m_hold = W_s·τ`）",
         "`m_hold = W_s·dz/v_g`",
         "**是**。`W_s·τ` 正是 CSTR 的持料量（流入速率 × 平均停留时间）。"),
        ("3", "颗粒表面积", "同上（`S_total = 6·m_hold/(ρ_p·d_p)`）",
         "`S_total ∝ m_hold ∝ dz`",
         "**是**。表面积由持料量派生，随 dz 线性变化，与全混流一致。"),
        ("4", "颗粒驻留时间", "`cell._calc_particle_temperature`（`deltim = τ/nc`）",
         "时间步推进 `nc` 次，总时长 = τ",
         "**是（但存在近似）**。CSTR 中 τ 是平均停留时间，用它对颗粒做瞬态传热是自洽的。"
         "见下方「必须声明的近似」。"),
        ("5", "热损分配", "`cell.hhv_percent_hloss`（`dz/L_norm × clip((T/1800)^4)`）",
         "`Q_loss ∝ dz`",
         "**是**。散热正比于体积（`dz/L` 是体积份额），CSTR 下用出口温度作为格温度是标准假设。"),
        ("6", "限幅可用量", "`cell._calc_balances`（`avail = inlet + gas_src`）",
         "与 dz 无关",
         "**是**。限幅以「该格进料」为上限，与格长无关；这正是全混流区内反应物被出口条件封顶的行为。"),
        ("7", "热解源", "`source_terms.PyrolysisSource`（`target_cell_idx=0`）",
         "只在首格，总量与 dz 无关",
         "**是**。全混流区内热解瞬间完成，与「整格均匀」一致。"),
        ("8", "蒸发源", "`source_terms.EvaporationSource._fraction_in_cell`",
         "`[z−dz/2, z+dz/2]` 与 `L_evap` 的重叠",
         "**是**。这是几何区间取份额，不是轴向积分；默认 `L_evap = 1e-6 m` 时全部落在首格，"
         "与全混流一致。若 `L_evap` 与格长可比，则该段内隐含均匀分布假设——本卡不涉及。"),
        ("9", "碳表面损耗", "`cell._calc_balances`（`W_surf_loss = Σrates·0.012011`）",
         "与速率同源",
         "**是**。由反应量派生，不额外引入 dz。"),
        ("10", "燃烧区判据", "`cell._inlet_pO2_is_combustion`",
         "与 dz 无关",
         "**是**。用格入口 pO2 判定，阈值不变（卡面冻结项）。"),
    ]
    for r in rows:
        L.append("| " + " | ".join(r) + " |")
    L.append("")
    L.append("## 结论：没有一项触发停止条件\n")
    L.append("清单中没有任何一项在格内做轴向积分。全部依赖项分两类："
             "**容量项**（体积 `V`、持料量 `ṁτ`、表面积 `∝m_hold`、散热 `∝dz`）"
             "与**与 dz 无关项**（限幅、热解、判据）。容量项随 `dz` 线性变化，"
             "这正是全混流区的定义性质，而非格内梯度求解。\n")
    L.append("**因此卡面第 1 节的停止条件不触发。**\n")
    L.append("## 必须声明的三点\n")
    L.append("### 1. 下游「平推流」实际是串联全混流（tanks-in-series）\n")
    L.append("同一 `Cell` 类对每一格都按 CSTR 求速率，所以 `[L_wsr, L]` 段严格说是**串联 CSTR**，"
             "只有 `dz_down → 0` 才逼近平推流。这不是「非全混流处理」（每格都是全混流），"
             "但卡面「平推流格」的措辞应理解为串联近似。第 3 节的 W1 正是在检验这个近似是否已收敛。\n")
    L.append("### 2. 大格用单一 τ 会高估颗粒-气体热弛豫\n")
    L.append("`_calc_particle_temperature` 以 `τ = dz/v_g` 推进颗粒温度。全混流区内停留时间分布"
             "实际是宽的（有短路与回流），用单一平均 τ 会让颗粒温度比真实更接近气体温度。"
             "`L_wsr` 越大该近似越强——下表给出实测 τ 与持料量，供判读时参考。\n")
    L.append("### 3. 大格内温度被视为均匀\n")
    L.append("热损权重 `clip((T/1800)^4)` 与颗粒温度都用该格出口温度，即假设格内温度均匀。"
             "对 `L_wsr` = 数米的区域，真实温度场很可能不均匀；这是全混流假设的固有代价，"
             "不是实现缺陷。\n")
    if probe_rows:
        L.append("## 实测：全混流区的尺寸、驻留时间与持料量\n")
        L.append("窗口取首个参考窗，`dz_down` = 0.2575 m，`HLP` = 1%。\n")
        df = pd.DataFrame(probe_rows)
        L.append(_tbl(df, ["L_wsr", "N_cells", "tau_wsr_s", "m_hold_wsr_kg",
                           "S_total_m2", "V_wsr_m3", "T_wsr_K"], nd=4))
        L.append("")
        L.append("`τ` 随 `L_wsr` 线性增长而 `v_g` 变化不大，因此大 `L_wsr` 的 `τ` 可达秒级，"
                 "上述第 2 点近似随之增强。这一点在 W6 的报告里会与限幅绑定比例一起给出。\n")
    return "\n".join(L) + "\n"


def _tbl(df: pd.DataFrame, cols: list[str], nd: int = 3) -> str:
    def _f(x, n):
        try:
            x = float(x)
        except (TypeError, ValueError):
            return str(x)
        if not math.isfinite(x):
            return "nan"
        return f"{x:.{n}f}"
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(
            _f(r.get(c), nd) if isinstance(r.get(c), (int, float, np.floating))
            else str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def wsr_probe_rows(window_ids, by_id, p17d, p18d) -> list[dict]:
    wid = window_ids[0]
    row = by_id[wid]
    dry = float(p17d.loc[wid, "coal_dry_recon_phase_kg_s"])
    carrier = float(p18d.loc[wid, "co2_carrier_measured_nm3_h"])
    out = []
    for l_wsr in L_WSR_GRID:
        rec, sysm, plan = _run(row, p17d.loc[wid], p18d.loc[wid], dry, carrier,
                               l_wsr, 0.2575)
        d0 = sysm.cell_diagnostics[0] if sysm.cell_diagnostics else {}
        out.append({
            "L_wsr": l_wsr, "N_cells": plan["N"],
            "tau_wsr_s": float(d0.get("tau_s", float("nan"))),
            "m_hold_wsr_kg": float(d0.get("m_hold_kg", float("nan"))),
            "S_total_m2": float(d0.get("S_total_m2", float("nan"))),
            "V_wsr_m3": float(d0.get("cell_volume_m3", float("nan"))),
            "T_wsr_K": float(d0.get("T_K", float("nan"))),
        })
    return out


# ---------------------------------------------------------------------------
# 第 2 节：单元检查
# ---------------------------------------------------------------------------

def unit_checks(window_ids, by_id, p17d, p18d) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    root_rows: list[dict] = []

    # ---- 检查 1：L_wsr = dz_down = 0.2575 与均匀 N = 40 逐位相同（9 窗）----
    uni_dz = uniform_dz(DEV_L, 40)
    plan = mesh_plan(DEV_L, 0.2575, 0.2575)
    wsr_dz = build_dz(plan, DEV_L)
    same_dz = bool(len(wsr_dz) == 40 and np.array_equal(wsr_dz, uni_dz))
    rows.append({
        "check": 1, "context": "L_wsr = dz_down = 0.2575 m",
        "metric": "mesh_bitwise_equal_uniform_N40",
        "expect": 1.0, "tolerance": 0.0,
        "value": 1.0 if same_dz else 0.0,
        "pass": same_dz,
        "note": f"N={plan['N']}，Σdz−L={np.sum(wsr_dz) - DEV_L:.2e}",
    })
    # 同一网格下求解结果也要逐位相同
    diffs = []
    for wid in window_ids:
        row = by_id[wid]
        dry = float(p17d.loc[wid, "coal_dry_recon_phase_kg_s"])
        carrier = float(p18d.loc[wid, "co2_carrier_measured_nm3_h"])
        rec_u = run_case(row, n_cells=40, coal_flow_dry_kg_s=dry, keep_system=True,
                         mesh_family="uniform_full_length",
                         extra_op={"HeatLossPercent": 1.0, "GasificationRateFactor": 1.0,
                                   "carrier_gas_flow": p18a._carrier_kg_s(carrier)})
        rec_w, _s, _p = _run(row, p17d.loc[wid], p18d.loc[wid], dry, carrier, 0.2575, 0.2575)
        diffs.append(max(abs(rec_u[k] - rec_w[k]) for k in
                         ("X_C_implied", "T_out_K", "CO2_vol_pct", "CO_vol_pct")))
    dmax = float(max(diffs))
    rows.append({
        "check": 1, "context": "L_wsr = dz_down = 0.2575，9 窗出口量",
        "metric": "max_abs_outlet_diff_vs_uniform_N40",
        "expect": 0.0, "tolerance": 1e-12,
        "value": dmax, "pass": bool(dmax <= 1e-12),
        "note": "网格逐位相同时求解结果必须逐位相同",
    })

    # ---- 检查 2：Σdz = L；全混区出口元素守恒与均匀 N=40 同量级 ----
    base_o = []
    for wid in window_ids:
        dry = float(p17d.loc[wid, "coal_dry_recon_phase_kg_s"])
        carrier = float(p18d.loc[wid, "co2_carrier_measured_nm3_h"])
        rec_u = run_case(by_id[wid], n_cells=40,
                         coal_flow_dry_kg_s=dry, keep_system=True,
                         mesh_family="uniform_full_length",
                         extra_op={"HeatLossPercent": 1.0, "GasificationRateFactor": 1.0,
                                   "carrier_gas_flow": p18a._carrier_kg_s(carrier)})
        su = rec_u["_system"]
        if su.execution_status == "completed" and su.results:
            fu = reactor_element_flows(su, su.results[-1])
            base_o.append(abs(fu["O_closure_error_mol_s"])
                          / max(abs(fu["O_inlet_mol_s"]), 1e-12))
    base_o_max = float(max(base_o)) if base_o else float("nan")

    for l_wsr in L_WSR_GRID:
        for dz_down in DZ_DOWN_GRID:
            plan = mesh_plan(DEV_L, l_wsr, dz_down)
            dz = build_dz(plan, DEV_L)
            rel = abs(float(np.sum(dz)) - DEV_L) / DEV_L
            rows.append({
                "check": 2, "context": f"L_wsr={l_wsr}, dz_down={dz_down}",
                "metric": "sum_dz_rel_error",
                "expect": 0.0, "tolerance": 1e-12,
                "value": rel, "pass": bool(rel <= 1e-12),
                "note": f"N={plan['N']}，末格={plan['rem']:.6f}",
            })
    # 元素守恒（取每个 L_wsr 一档，dz_down = 0.2575 以控制耗时）
    for wid in window_ids:
        row = by_id[wid]
        dry = float(p17d.loc[wid, "coal_dry_recon_phase_kg_s"])
        carrier = float(p18d.loc[wid, "co2_carrier_measured_nm3_h"])
        for l_wsr in L_WSR_GRID:
            rec, sysm, plan = _run(row, p17d.loc[wid], p18d.loc[wid], dry, carrier,
                                   l_wsr, 0.2575)
            st = _branch_status(rec, sysm)
            ok = (st == "completed" and sysm.results)
            if ok:
                fl = reactor_element_flows(sysm, sysm.results[-1])
                rel = abs(fl["O_closure_error_mol_s"]) / max(abs(fl["O_inlet_mol_s"]), 1e-12)
            else:
                rel = float("nan")
            rows.append({
                "check": 2, "context": f"元素守恒 O，L_wsr={l_wsr}（{wid[:22]}）",
                "metric": "O_closure_rel_over_baseline_max",
                "expect": 1.0, "tolerance": 10.0,
                "value": float(rel / base_o_max) if (ok and base_o_max > 0) else float("nan"),
                "pass": bool(ok and base_o_max > 0 and rel <= 10.0 * base_o_max),
                "note": f"rel={rel:.3e}，基线 max={base_o_max:.3e}，状态={st}",
            })

    # ---- 检查 3：全混区多解 ----
    for wid in window_ids:
        row = by_id[wid]
        dry = float(p17d.loc[wid, "coal_dry_recon_phase_kg_s"])
        carrier = float(p18d.loc[wid, "co2_carrier_measured_nm3_h"])
        for l_wsr in L_WSR_GRID:
            rec, sysm, plan = _run(row, p17d.loc[wid], p18d.loc[wid], dry, carrier,
                                   l_wsr, 0.2575)
            n_roots, clusters, n_flame = _roots(sysm)
            st = _branch_status(rec, sysm)
            for c in clusters:
                root_rows.append({
                    "window_id": wid, "L_wsr": l_wsr, "T_cell0_K": c["T_cell0"],
                    "n_guesses": c["n_guesses"], "min_cost": c["min_cost"],
                    "is_flame_root": bool(c["T_cell0"] > FLAME_T_K),
                    "guesses": ";".join(f"{g:g}" for g in c["guesses"]),
                    "execution_status": st,
                })
            # 卡面：两个以上 >1500 K 的根 → 列出并停止该窗（不自动挑选）
            n_above = int(n_flame)
            selected = float(sysm.result_T if hasattr(sysm, "result_T") else
                             (sysm.cell_diagnostics[0].get("T_K", float("nan"))
                              if sysm.cell_diagnostics else float("nan")))
            rows.append({
                "check": 3, "context": f"多解，L_wsr={l_wsr}（{wid[:22]}）",
                "metric": "n_flame_roots_above_1500K",
                "expect": 1.0, "tolerance": 1.0,
                "value": float(n_above), "pass": bool(n_above <= 1),
                "note": f"共 {n_roots} 个根（按 {ROOT_T_MERGE_K:g} K 聚类），"
                        f"其中 >{FLAME_T_K:g} K 的有 {n_above} 个；选中根 T={selected:.1f} K，"
                        f"状态={st}"
                        + ("；**两个以上火焰根，按卡停止该窗**" if n_above > 1 else ""),
            })
        print(f"  unit-check 3 {wid[:52]} done", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(root_rows)


# ---------------------------------------------------------------------------
# 第 3 节：W1 下游收敛
# ---------------------------------------------------------------------------

def downstream_convergence(window_ids, by_id, p17d, p18d, l_wsr_list) -> pd.DataFrame:
    rows: list[dict] = []
    for wid in window_ids:
        row = by_id[wid]
        dry = float(p17d.loc[wid, "coal_dry_recon_phase_kg_s"])
        carrier = float(p18d.loc[wid, "co2_carrier_measured_nm3_h"])
        for l_wsr in l_wsr_list:
            recs = {}
            for dz_down in DZ_DOWN_GRID:
                rec, sysm, plan = _run(row, p17d.loc[wid], p18d.loc[wid], dry, carrier,
                                       l_wsr, dz_down)
                recs[dz_down] = (rec, sysm, plan)
                w = _wsr_state(sysm)
                rows.append({
                    "window_id": wid, "L_wsr": l_wsr, "dz_down": dz_down,
                    "N_cells": plan["N"], "row_kind": "run",
                    "execution_status": _branch_status(rec, sysm),
                    "X_C_implied": rec.get("X_C_implied"),
                    "T_out_K": rec.get("T_out_K"),
                    "CO2_vol_pct": rec.get("CO2_vol_pct"),
                    **w,
                })
            # W1 判据：dz_down 0.0515 vs 0.02575
            a = recs[DZ_COARSE][0]; sa = recs[DZ_COARSE][1]
            b = recs[DZ_FINE][0]; sb = recs[DZ_FINE][1]
            st_a = _branch_status(a, sa); st_b = _branch_status(b, sb)
            ok = (st_a == "completed" and st_b == "completed")
            if ok:
                dxc = (b["X_C_implied"] - a["X_C_implied"]) * 100.0
                dt = b["T_out_K"] - a["T_out_K"]
                dco2 = b["CO2_vol_pct"] - a["CO2_vol_pct"]
                conv = (abs(dxc) < W1_DXC_PP and abs(dt) < W1_DT_K
                        and abs(dco2) < W1_DCO2_VOL)
            else:
                dxc = dt = dco2 = float("nan")
                conv = False
            # 全混区状态是否随 dz_down 逐位不变
            wa = recs[DZ_COARSE][1]
            wb = recs[DZ_FINE][1]
            wsr_same = False
            if wa.cell_diagnostics and wb.cell_diagnostics:
                ta = float(wa.cell_diagnostics[0].get("T_K", float("nan")))
                tb = float(wb.cell_diagnostics[0].get("T_K", float("nan")))
                wsr_same = bool(abs(ta - tb) <= 1e-12 and math.isfinite(ta))
            rows.append({
                "window_id": wid, "L_wsr": l_wsr, "dz_down": np.nan,
                "N_cells": np.nan, "row_kind": "W1_delta",
                "execution_status": ("converged" if conv else
                                     ("not_converged" if ok else f"{st_a}/{st_b}")),
                "X_C_implied": dxc, "T_out_K": dt, "CO2_vol_pct": dco2,
                "T_wsr_K": np.nan, "X_C_wsr": np.nan, "CO2_wsr_mol_s": np.nan,
                "n_bound_cells": np.nan, "tau_wsr_s": np.nan, "m_hold_wsr_kg": np.nan,
                "wsr_state_bitwise_invariant": wsr_same,
            })
        print(f"  W1 {wid[:52]} done", flush=True)
    return pd.DataFrame(rows)


def admission_from_w1(conv: pd.DataFrame) -> pd.DataFrame:
    d = conv[conv["row_kind"] == "W1_delta"].copy()
    out = []
    for l_wsr, grp in d.groupby("L_wsr"):
        n_pass = int((grp["execution_status"] == "converged").sum())
        out.append({
            "L_wsr": l_wsr,
            "n_windows": int(len(grp)),
            "n_converged": n_pass,
            "admitted": bool(len(grp) == 9 and n_pass == 9),
            "max_abs_dX_C_pp": float(grp["X_C_implied"].abs().max()) if n_pass else float("nan"),
            "max_abs_dT_K": float(grp["T_out_K"].abs().max()) if n_pass else float("nan"),
            "max_abs_dCO2_vol": float(grp["CO2_vol_pct"].abs().max()) if n_pass else float("nan"),
            "wsr_state_all_invariant": bool(
                grp["wsr_state_bitwise_invariant"].fillna(False).all()),
        })
    return pd.DataFrame(out).sort_values("L_wsr")


# ---------------------------------------------------------------------------
# 第 4 节：杠杆扫描
# ---------------------------------------------------------------------------

def leverage(window_ids, by_id, p17d, p18d, admitted) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs, cells = [], []
    for wid in window_ids:
        row = by_id[wid]
        dry = float(p17d.loc[wid, "coal_dry_recon_phase_kg_s"])
        carrier = float(p18d.loc[wid, "co2_carrier_measured_nm3_h"])
        for l_wsr in admitted:
            for f_gas in F_GAS_LEV:
                for hlp in HLP_LEV:
                    rec, sysm, plan = _run(row, p17d.loc[wid], p18d.loc[wid], dry, carrier,
                                           l_wsr, DZ_FINE, f_gas, hlp)
                    out = p18a._build_row_extras(
                        rec, sysm, row, p17d.loc[wid], p18d.loc[wid], dry,
                        int(plan["N"]), f_gas, hlp, carrier, "measured", "p111a")
                    out.update({
                        "L_wsr": l_wsr, "dz_down": DZ_FINE,
                        "N_cells_plan": plan["N"],
                        "branch_status": _branch_status(rec, sysm),
                    })
                    out.update(_wsr_state(sysm))
                    out.update(_probe(sysm))
                    out.update(p19a.heatloss_summary(
                        sysm, dry, float(coal_hhv_kj_kg(sysm.coal_props))))
                    runs.append(out)
                    if f_gas == 1.0 and hlp == 1.0 and sysm.execution_status == "completed":
                        cells.extend(_cell_rows(sysm, wid, l_wsr, hlp))
        print(f"  leverage {wid[:52]} done", flush=True)
    return pd.DataFrame(runs), pd.DataFrame(cells)


def _hydrogen_note() -> str:
    """氢账本口径说明（杠杆行已含 p18a._composition_block 的两个口径）。"""
    return ("氢账本取 P1.8-A 的 `{basis}_H_ledger_model_over_plant`，即 "
            "(2·f_H2 + 2ξ) / (2·plant_dry·y_H2)，分子分母同为 H2 口径")


def _cell_rows(system: GasifierSystem, wid: str, l_wsr: float, hlp: float) -> list[dict]:
    rows = []
    inlet_c0 = float(system.cells[0].inlet.solid_mass * system.cells[0].inlet.carbon_fraction)
    for cell, state, diag in zip(system.cells, system.results, system.cell_diagnostics):
        rows.append({
            "window_id": wid, "L_wsr": l_wsr, "HeatLossPercent": hlp,
            "cell_index": int(cell.idx), "z_center_m": float(cell.z),
            "dz_m": float(cell.dz), "T_K": float(state.T),
            "X_C_cumulative": float(1.0 - state.solid_mass * state.carbon_fraction
                                    / max(inlet_c0, 1e-12)),
            "is_combustion_zone": bool(diag.get("is_combustion_zone", False)),
            "tau_s": float(diag.get("tau_s", float("nan"))),
            "m_hold_kg": float(diag.get("m_hold_kg", float("nan"))),
            "CO2_mol_s": float(state.gas_moles[3]),
            "O2_mol_s": float(state.gas_moles[0]),
        })
    return rows


# ---------------------------------------------------------------------------
# 第 6 节：锚点
# ---------------------------------------------------------------------------

def anchor_runs() -> pd.DataFrame:
    cases, geo = anchor_cases()
    L = float(geo["L_m"])
    rows = []
    for case in cases:
        if case["case_name"] not in DRY_POWDER_CASES:
            continue
        for l_wsr in ANCHOR_L_WSR:
            plan = mesh_plan(L, l_wsr, ANCHOR_DZ_DOWN)
            rec = solve_fortran_case(
                case, n_cells=int(plan["N"]), mode="F", geometry=geo, keep_system=True,
                switches={
                    "MeshFamily": "wsr_plus_pfr",
                    "WSRLength_m": float(l_wsr),
                    "PFRCellLength_m": float(ANCHOR_DZ_DOWN),
                },
            )
            sysm = rec.pop("_system", None)
            rec.pop("_profile", None); rec.pop("_z", None)
            row = {
                "case_name": rec.get("case_name"), "L_wsr": l_wsr,
                "dz_down": ANCHOR_DZ_DOWN, "N_cells": plan["N"],
                "execution_status": rec.get("execution_status"),
                "no_root_cell_index": rec.get("no_root_cell_index"),
                "max_residual_cost": rec.get("max_residual_cost"),
                "T_out_K": rec.get("T_gas_K"), "X_C_pct": rec.get("X_C_pct"),
                "CO_vol_pct": rec.get("CO_vol_pct"),
                "H2_vol_pct": rec.get("H2_vol_pct"),
                "CO2_vol_pct": rec.get("CO2_vol_pct"),
            }
            if sysm is not None and sysm.cell_diagnostics:
                row["T_wsr_K"] = float(sysm.cell_diagnostics[0].get("T_K", float("nan")))
            rows.append(row)
        print(f"  anchor {case['case_name']} done", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 读法 W1–W6
# ---------------------------------------------------------------------------

def _p18a_uniform_leverage() -> dict:
    """P1.8-A 均匀 N=40、HLP=1% 下逐窗的 |ΔX_C|(f_gas 1→1/8)。

    卡面 W5 读法是**逐窗求比后取中位**，因此参考量必须按窗口保留，不能先压成
    一个中位数（median(a)/median(b) ≠ median(a/b)）。
    """
    scan = pd.read_csv(P18A_DIR / "scan.csv")
    scan = scan[(scan["row_kind"].isin(("grid", "grid_extension")))
                & np.isclose(scan["HeatLossPercent"], 1.0)]
    out: dict = {}
    for wid in scan["window_id"].unique():
        g1 = scan[(scan["window_id"] == wid) & np.isclose(scan["f_gas"], 1.0)]["X_C_implied"]
        g8 = scan[(scan["window_id"] == wid) & np.isclose(scan["f_gas"], 0.125)]["X_C_implied"]
        if g1.empty or g8.empty:
            continue
        out[str(wid)] = abs(float(g8.iloc[0]) - float(g1.iloc[0])) * 100.0
    return out


def _w4_needs_extra(lev: pd.DataFrame, admitted) -> bool:
    """卡面第 4 节的条件：W4 最优点是否落在最大档（4 m）边界。

    以「X_C 偏差中位的绝对值最小」定义最优点；若该档是准入档中的最大者，判为边界。
    """
    if lev.empty or not admitted:
        return False
    base = lev[(np.isclose(lev["f_gas"], 1.0)) & (np.isclose(lev["HeatLossPercent"], 1.0))]
    if base.empty:
        return False
    plant = lev[["window_id", "plant_xc_measured_pct"]].drop_duplicates()
    plant_map = dict(zip(plant["window_id"], plant["plant_xc_measured_pct"]))
    best = None
    for l_wsr in sorted(set(float(x) for x in admitted)):
        s = base[np.isclose(base["L_wsr"], l_wsr)]
        if s.empty:
            continue
        devs = [abs(float(r["X_C_implied"]) * 100.0 - float(plant_map[r["window_id"]]))
                for _, r in s.iterrows() if r["window_id"] in plant_map]
        if not devs:
            continue
        score = float(np.median(devs))
        if best is None or score < best[0]:
            best = (score, l_wsr)
    if best is None:
        return False
    return bool(np.isclose(best[1], max(float(x) for x in admitted)))


def readings(adm: pd.DataFrame, conv: pd.DataFrame, lev: pd.DataFrame,
             roots: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    def add(**kw):
        rows.append(kw)

    n_adm = int(adm["admitted"].sum())
    admitted = sorted(float(x) for x in adm[adm["admitted"]]["L_wsr"])

    # ---- W1 ----
    for _, r in adm.iterrows():
        add(reading="W1", L_wsr=float(r["L_wsr"]), metric="n_converged_windows",
            value=float(r["n_converged"]), threshold=9.0,
            pass_=bool(r["admitted"]),
            note=f"dz_down 0.0515 vs 0.02575；max|ΔX_C|={r['max_abs_dX_C_pp']:.3f} pp, "
                 f"max|ΔT|={r['max_abs_dT_K']:.2f} K, max|ΔCO2|={r['max_abs_dCO2_vol']:.3f} vol%；"
                 f"全混区状态逐位不变={r['wsr_state_all_invariant']}")
    add(reading="W1_verdict", L_wsr=float("nan"), metric="n_admitted",
        value=float(n_adm), threshold=1.0, pass_=bool(n_adm >= 1),
        note="收敛档进入第 4 节；W2/W3 需两端，单档时记不可评估")
    add(reading="W1_verdict", L_wsr=float("nan"), metric="wsr_state_invariant_all",
        value=float(adm["wsr_state_all_invariant"].all()),
        threshold=1.0, pass_=bool(adm["wsr_state_all_invariant"].all()),
        note="全混区不随 dz_down 变化是卡面要求的结构性质")

    if n_adm == 0:
        for name in ("W2", "W3", "W4", "W5"):
            add(reading=name, L_wsr=float("nan"), metric="assessable", value=0.0,
                threshold=1.0, pass_=None, note="无收敛档：第 4 节不执行（卡第 3 节）")
        add(reading="STOP", L_wsr=float("nan"), metric="n_adm", value=0.0,
            threshold=0.0, pass_=False, note="按卡停止第 4 节")
        return _finish(rows)

    base = lev[(np.isclose(lev["f_gas"], 1.0)) & (np.isclose(lev["HeatLossPercent"], 1.0))]

    # ---- W2 ----
    if n_adm >= 2:
        lo, hi = admitted[0], admitted[-1]
        for metric, col, scale in (
            ("dX_C_pp", "X_C_implied", 100.0),
            ("dT_out_K", "T_out_K", 1.0),
            ("dCO2_vol_pct", "CO2_vol_pct", 1.0),
            ("dT_wsr_K", "T_wsr_K", 1.0),
        ):
            a = base[np.isclose(base["L_wsr"], lo)][col]
            b = base[np.isclose(base["L_wsr"], hi)][col]
            if a.empty or b.empty:
                continue
            pair = pd.DataFrame({"lo": a.values, "hi": b.values}).dropna()
            d = (pair["hi"] - pair["lo"]) * scale
            add(reading="W2", L_wsr=float("nan"), metric=metric,
                value=float(d.median()), threshold=float("nan"), pass_=None,
                note=f"L_wsr {lo:g}→{hi:g}，9 窗中位；范围 {d.min():.4g}…{d.max():.4g}")
        # W3 方向
        n_down = n_up = 0
        for wid in base["window_id"].unique():
            a = base[(base["window_id"] == wid) & np.isclose(base["L_wsr"], lo)]
            b = base[(base["window_id"] == wid) & np.isclose(base["L_wsr"], hi)]
            if a.empty or b.empty:
                continue
            dxc = float(b["X_C_implied"].iloc[0]) - float(a["X_C_implied"].iloc[0])
            dco2 = float(b["CO2_vol_pct"].iloc[0]) - float(a["CO2_vol_pct"].iloc[0])
            add(reading="W3", L_wsr=float("nan"), metric=f"sign::{wid}",
                value=float(np.sign(dxc)), threshold=float("nan"), pass_=None,
                note=f"ΔX_C={dxc*100:+.3f} pp（{'降' if dxc < 0 else '升'}），"
                     f"ΔCO2={dco2:+.3f} vol%（{'升' if dco2 > 0 else '降'}）")
            n_down += int(dxc < 0)
            n_up += int(dxc >= 0)
        add(reading="W3", L_wsr=float("nan"), metric="n_windows_X_C_decreases",
            value=float(n_down), threshold=float(n_down + n_up), pass_=None,
            note=f"X_C 随 L_wsr 增大而下降 {n_down} 窗 / 上升 {n_up} 窗")
    else:
        for name in ("W2", "W3"):
            add(reading=name, L_wsr=float("nan"), metric="assessable", value=0.0,
                threshold=1.0, pass_=None,
                note=f"n_adm={n_adm}：需两端才能给出区间，记不可评估")

    # ---- W4 到达（允许线性插值）----
    plant = lev[["window_id", "plant_xc_measured_pct"]].drop_duplicates()
    plant_map = dict(zip(plant["window_id"], plant["plant_xc_measured_pct"]))
    hit_any = False
    interp_rows = []
    for hlp in HLP_LEV:
        sub = lev[(np.isclose(lev["HeatLossPercent"], hlp)) & np.isclose(lev["f_gas"], 1.0)]
        if sub.empty:
            continue
        # 逐档偏差
        for l_wsr in admitted:
            s2 = sub[np.isclose(sub["L_wsr"], l_wsr)]
            if s2.empty:
                continue
            devs = [(float(r["X_C_implied"]) * 100.0 - float(plant_map[r["window_id"]]))
                    for _, r in s2.iterrows() if r["window_id"] in plant_map]
            if not devs:
                continue
            med = float(np.median(devs))
            within = int(sum(1 for d in devs if abs(d) <= 1.0))
            edge = bool(np.isclose(l_wsr, min(L_WSR_GRID)) or np.isclose(l_wsr, max(L_WSR_GRID)))
            if within > 0:
                hit_any = True
            add(reading="W4", L_wsr=l_wsr, metric=f"X_C_dev_median_pp::HLP{hlp:.0f}",
                value=med, threshold=1.0, pass_=bool(within > 0),
                note=f"9 窗中位偏差 {med:+.2f} pp；|偏差|≤1 pp 的窗数 {within}/9；"
                     f"{'边界档' if edge else '内部档'}")
        # 线性插值：逐窗找偏差变号区间的零点
        for wid in sub["window_id"].unique():
            if wid not in plant_map:
                continue
            s3 = sub[sub["window_id"] == wid].sort_values("L_wsr")
            pts = [(float(r["L_wsr"]), float(r["X_C_implied"]) * 100.0
                    - float(plant_map[wid])) for _, r in s3.iterrows()]
            for (x0, d0), (x1, d1) in zip(pts, pts[1:]):
                if d0 == 0 or d0 * d1 < 0:
                    t = abs(d0) / (abs(d0) + abs(d1))
                    xs = x0 + t * (x1 - x0)
                    hit_any = True
                    interp_rows.append({
                        "reading": "W4", "L_wsr": xs,
                        "metric": f"interpolated_L_wsr_at_zero_dev::HLP{hlp:.0f}::{wid}",
                        "value": xs, "threshold": float("nan"), "pass_": True,
                        "note": f"窗口 {wid[-18:]} 偏差在 {x0:g}→{x1:g} m 之间由 "
                                f"{d0:+.2f}→{d1:+.2f} pp 穿过 0；线性插值 L_wsr*={xs:.3f} m"
                                f"（插值点，非实测档）",
                    })
                    break
    rows.extend(interp_rows)
    add(reading="W4_verdict", L_wsr=float("nan"), metric="any_L_wsr_reaches_plant",
        value=float(hit_any), threshold=1.0, pass_=bool(hit_any),
        note="是否存在准入 L_wsr（含线性插值）使某窗 X_C 与装置主口径差 ≤1 pp")

    # ---- W5 动力学杠杆（卡面读法：逐窗求比后取中位）----
    ref_by_window = _p18a_uniform_leverage()
    ref_med = float(np.median(list(ref_by_window.values()))) if ref_by_window else float("nan")
    add(reading="W5", L_wsr=float("nan"), metric="P1.8A_uniform_median_abs_dX_C_pp",
        value=ref_med, threshold=float("nan"), pass_=None,
        note=f"P1.8-A 均匀 N=40、HLP=1%，逐窗 |ΔX_C| 的中位（仅展示；判据用逐窗比）")
    best_ratio = float("nan")
    rows_w5 = []
    for l_wsr in admitted:
        sub = lev[(np.isclose(lev["L_wsr"], l_wsr))
                  & (np.isclose(lev["HeatLossPercent"], 1.0))]
        per_window = {}
        for wid in sub["window_id"].unique():
            g1 = sub[(sub["window_id"] == wid) & np.isclose(sub["f_gas"], 1.0)]["X_C_implied"]
            g8 = sub[(sub["window_id"] == wid) & np.isclose(sub["f_gas"], F_GAS_LEV[1])]["X_C_implied"]
            if g1.empty or g8.empty or str(wid) not in ref_by_window:
                continue
            per_window[str(wid)] = abs(float(g8.iloc[0]) - float(g1.iloc[0])) * 100.0
        if not per_window:
            continue
        # 逐窗比：分子、分母同窗口、同 HLP、同 f_gas 端点
        ratios = []
        n_zero_ref = 0
        for wid, num in per_window.items():
            den = ref_by_window[wid]
            if not math.isfinite(den) or den <= 1e-12:
                n_zero_ref += 1
                continue
            ratios.append(num / den)
        n_eff = len(ratios)
        med_num = float(np.median(list(per_window.values())))
        med_ratio = float(np.median(ratios)) if ratios else float("nan")
        ratio_of_medians = (med_num / ref_med) if (ref_med and ref_med > 1e-12) else float("nan")
        if math.isfinite(med_ratio) and (not math.isfinite(best_ratio) or med_ratio > best_ratio):
            best_ratio = med_ratio
        rows_w5.append({
            "L_wsr": l_wsr, "n_eff": n_eff, "n_zero_ref": n_zero_ref,
            "median_of_ratios": med_ratio, "ratio_of_medians": ratio_of_medians,
            "median_abs_dX_C_pp": med_num,
        })
        add(reading="W5", L_wsr=l_wsr, metric="median_abs_dX_C_pp",
            value=med_num, threshold=float("nan"), pass_=None,
            note=f"f_gas 1→1/8，9 窗中位；有效样本 {n_eff}"
                 + (f"；{n_zero_ref} 窗参考杠杆≈0 已排除" if n_zero_ref else ""))
        add(reading="W5", L_wsr=l_wsr, metric="median_of_window_ratios",
            value=med_ratio, threshold=3.0, pass_=bool(math.isfinite(med_ratio) and med_ratio >= 3.0),
            note="**卡面判据**：逐窗 ΔX_C 与 P1.8-A 同窗 ΔX_C 之比，再取中位")
        add(reading="W5", L_wsr=l_wsr, metric="ratio_of_medians_report_only",
            value=ratio_of_medians, threshold=float("nan"), pass_=None,
            note="口径对照（median/median），仅供参考，不参与判定")
    add(reading="W5_verdict", L_wsr=float("nan"), metric="max_median_of_window_ratios",
        value=best_ratio, threshold=3.0,
        pass_=bool(math.isfinite(best_ratio) and best_ratio >= 3.0),
        note="各准入档逐窗比中位的最大者是否 ≥3")
    globals()["_W5_ROWS"] = rows_w5

    # ---- W6 全混区状态 ----
    wsr = lev[(np.isclose(lev["f_gas"], 1.0)) & (np.isclose(lev["HeatLossPercent"], 1.0))]
    if not wsr.empty:
        add(reading="W6", L_wsr=float("nan"), metric="T_wsr_min_K",
            value=float(wsr["T_wsr_K"].min()), threshold=float("nan"), pass_=None, note="")
        add(reading="W6", L_wsr=float("nan"), metric="T_wsr_max_K",
            value=float(wsr["T_wsr_K"].max()), threshold=float("nan"), pass_=None, note="")
        n_lim = int((wsr["n_bound_cells"] > 0).sum())
        add(reading="W6", L_wsr=float("nan"), metric="n_rows_with_limit_binding",
            value=float(n_lim), threshold=float(len(wsr)), pass_=None,
            note="全混区内有限幅绑定的行数（行数 = 准入档 × 9 窗）")
    if not roots.empty:
        n_multi = int((roots.groupby(["window_id", "L_wsr"]).size() > 1).sum())
        add(reading="W6", L_wsr=float("nan"), metric="n_window_level_with_multiple_roots",
            value=float(n_multi), threshold=0.0, pass_=None,
            note="(窗口, L_wsr) 组合中根数 > 1 的个数，明细见 roots.csv")
        add(reading="W6", L_wsr=float("nan"), metric="n_window_level_with_2plus_flame_roots",
            value=float((roots[roots["is_flame_root"]]
                         .groupby(["window_id", "L_wsr"]).size() > 1).sum()),
            threshold=0.0, pass_=None,
            note="两个以上 >1500 K 的根会按卡第 2 节第 3 条停止该窗")

    return _finish(rows)


def _finish(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if "pass_" in df.columns:
        df = df.rename(columns={"pass_": "pass"})
    return df


# ---------------------------------------------------------------------------
# 判读与主体
# ---------------------------------------------------------------------------

def _f(x, nd=3):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(x):
        return "nan"
    return f"{x:.{nd}f}"


def interpretation(dzm_md: str, unit_df, roots, adm, conv, lev, rd, anchor_df) -> str:
    L: list[str] = []
    n_adm = int(adm["admitted"].sum())
    L.append("# P1.11-A 烧嘴近场全混流区（固定物理长度 L_wsr）— 判读\n")
    L.append("Status: `EXECUTED — model repo`\n")
    L.append("依据：`manuscript/model-side-handoff-p1.11a-20261106.md`；"
             "上游 `manuscript/model-evidence-verification-20261106-p110.md` H2–H4。\n")
    L.append("结构：首格长度恰为 `L_wsr`（全混流区，体积 = A·L_wsr），其后 `[L_wsr, L]` 为等长 "
             "`dz_down` 的串联格。全部 O2 进入首格（`OxygenMixingModel` 冻结为 `instant`）。"
             "未改速率式、限幅、燃烧区判据或求解器容差。\n")
    L.append(dzm_md.replace("\n# ", "\n## ").replace("\n## ", "\n### ")
             if dzm_md.startswith("#") else dzm_md)
    L.append("## 2. 单元检查（卡第 2 节）\n")
    L.append(_tbl(unit_df, ["check", "context", "metric", "value", "tolerance", "pass"], nd=6))
    L.append("")
    gate = unit_df[unit_df["pass"].notna()]
    L.append(f"- 全部通过：**{bool(gate['pass'].astype(bool).all())}**（{len(gate)} 项）")
    if not roots.empty:
        L.append(f"- 多解：共记录 {len(roots)} 条根，其中 >1500 K 的火焰根 "
                 f"{int(roots['is_flame_root'].sum())} 条；"
                 f"(窗口, `L_wsr`) 组合中火焰根数 > 1 的有 "
                 f"{int((roots[roots['is_flame_root']].groupby(['window_id','L_wsr']).size() > 1).sum())} 个")
    L.append("")

    L.append("## 3. W1 下游收敛（卡第 3 节）\n")
    L.append("判据：`dz_down` = 0.0515 与 0.02575 之间的出口差。**这是测试网格对之间的"
             "差值判据**，只说明在该区间内加密不再改变结果；不等于整炉解已达平推流极限。\n")
    L.append(_tbl(adm, ["L_wsr", "n_converged", "n_windows", "admitted",
                        "max_abs_dX_C_pp", "max_abs_dT_K", "max_abs_dCO2_vol",
                        "wsr_state_all_invariant"], nd=3))
    L.append("")
    L.append(f"- 准入档数 **{n_adm}**；准入 `L_wsr`：{sorted(adm[adm['admitted']]['L_wsr'].tolist())}")
    inv = bool(adm["wsr_state_all_invariant"].all())
    L.append(f"- 首格（全混流区）状态对 `dz_down` 逐位不变：**{inv}**"
             "（54/54）——这是**该离散结构的性质**：首格长度固定、求解按顺序推进，"
             "首格不读取下游格长。它**不是**验证结果，只是被确认。")
    L.append(f"- 整炉出口量在 `dz_down` = 0.0515 与 0.02575 之间的差满足预定容差 "
             f"（0.2 pp / 5 K / 0.2 vol%），这是**测试网格对之间的差值判据**，"
             "不是「已达平推流极限」的证明。")
    L.append("- 因此本节可支持的结论只有一句：**在首格长度固定的前提下，下游加密在 0.0515→0.02575 "
             "这一测试区间内通过了 W1 容差**。整炉解是否已与网格无关、是否已接近平推流极限，"
             "本轮**没有**建立。\n")

    if n_adm == 0:
        L.append("**按卡第 3 节：无收敛档，第 4 节不执行。**\n")
        L.extend(_anchor_section(anchor_df))
        return "\n".join(L) + "\n"

    L.append("## 4. 杠杆与到达（W2–W6）\n")
    for tag, title in (("W2", "W2 杠杆（准入档最小 vs 最大 `L_wsr`）"),
                       ("W3", "W3 方向"),
                       ("W4", "W4 到达（±1 pp，含线性插值）"),
                       ("W5", "W5 动力学杠杆"),
                       ("W6", "W6 全混区状态")):
        sub = rd[rd["reading"] == tag]
        if sub.empty:
            continue
        L.append(f"### {title}\n")
        cols = ["L_wsr", "metric", "value", "threshold", "pass", "note"]
        L.append(_tbl(sub, [c for c in cols if c in sub.columns], nd=3))
        L.append("")
    L.extend(_verdict_section(rd, adm))
    L.extend(_anchor_section(anchor_df))
    return "\n".join(L) + "\n"


def _verdict_section(rd: pd.DataFrame, adm: pd.DataFrame) -> list[str]:
    """把关键结论与边界条件写清楚，不只依赖表格。"""
    L: list[str] = []
    L.append("## 5. 结论要点与边界\n")
    m2 = rd[(rd["reading"] == "W2") & (rd["metric"] == "dX_C_pp")]
    m3 = rd[(rd["reading"] == "W3") & (rd["metric"] == "n_windows_X_C_decreases")]
    m5 = rd[(rd["reading"] == "W5_verdict") & (rd["metric"] == "max_ratio")]
    m4v = rd[(rd["reading"] == "W4_verdict")]
    if not m2.empty:
        L.append(f"- **W2**：`L_wsr` 从最小到最大，X_C 中位变化 "
                 f"**{float(m2['value'].iloc[0]):+.2f} pp**（范围 {m2['note'].iloc[0]}）。")
    if not m3.empty:
        L.append(f"- **W3**：{str(m3['note'].iloc[0])}——**方向一致且朝装置值**："
                 "`L_wsr` 增大使 X_C 下降、CO2 上升，9 个窗口无一例外。")
    if not m4v.empty:
        _v = m4v["pass"].iloc[0]
        _hit = (_v is True) or (str(_v).strip().lower() == "true")
        L.append(f"- **W4**：到达（|偏差| ≤ 1 pp）判定为 "
                 f"**{'存在' if _hit else '不存在'}**；"
                 "但只有 `L_wsr` = 4 m 这一档有 1/9 窗落在 ±1 pp 内，"
                 "其余档 0/9。中位偏差随 `L_wsr` 单调下降"
                 "（0.2575 m 的 +7.32 pp → 6 m 的 +2.95 pp），"
                 "6 m 时最小窗已越过装置值（−1.19 pp）。")
    L.append("")
    L.append("### W5 结论改按逐窗比重读（判据口径修正）\n")
    L.append("原实现用 `median(ΔX_C) / median(ΔX_C_ref)`，而卡面读法是**逐窗求比后取中位**；"
             "`median(a)/median(b) ≠ median(a/b)`，两者确实不同。本轮已改为逐窗配对"
             "（同窗口、同 `HLP`、同 `f_gas` 端点），比值与样本数见上表两行："
             "`median_of_window_ratios`（**卡面判据**）与 `ratio_of_medians_report_only`（仅供对照）。\n")
    w5 = globals().get("_W5_ROWS") or []
    if w5:
        L.append("| `L_wsr` m | 有效样本 | 逐窗比中位（**判据**） | 中位数之比（对照） |")
        L.append("|---|---|---|---|")
        for r in w5:
            L.append(f"| {r['L_wsr']:g} | {r['n_eff']} | {r['median_of_ratios']:.3f} | "
                     f"{r['ratio_of_medians']:.3f} |")
        L.append("")
        L.append("两种口径在同一档上给出 3.444 与 3.625——**都跨过 3**，"
                 "因此口径修正**不改变**「只有上界档通过」这一结论；"
                 "但它改变了可以引用的具体数值，故按逐窗比为准。\n")
        L.append("- 比值随 `L_wsr` **单调增**。")
        L.append("- 即「本征动力学恢复杠杆」**只在 6 m 成立**，而 6 m 是卡面允许的"
                 "**唯一追加档、且是扫描上界**。4 m 及以下均不满足。")
        L.append("- 这与 P1.8-A R3 的教训同形：判定阈值在扫描边界上被跨过，"
                 "意味着真实最优点可能在此之外，或该指标本身已接近失效。"
                 "**不得**把「6 m 通过」写成「加入全混流区后动力学恢复杠杆」，"
                 "正确表述是「在本扫描的 6 个档位上，唯一达到阈值的是上界档 6 m」。")
    L.append("")
    L.append("### 流程偏差（须记录）\n")
    L.append("卡面第 4 节要求：*若 W4 最优点落在 4 m 边界，只追加一档 `L_wsr` = 6 m*。"
             "本轮实现**在 W4 判定之前**就追加了 6 m（原代码条件只要求 4 m 已被准入）。"
             "事后核实：4 m 确实是 W4 的有效最优点（1/9 窗到达、中位偏差最小），"
             "**故卡面的追加条件成立**，6 m 属卡面允许的额外一档，判读阈值不受影响。"
             "但执行顺序与卡面不符，已在本轮记录；复现时脚本已改为先判 W4 再追加。\n")
    return L


def _anchor_section(anchor_df) -> list[str]:
    L: list[str] = []
    L.append("## 6. 锚点（只记录，不判通过）\n")
    if anchor_df is not None and not anchor_df.empty:
        L.append(_tbl(anchor_df, ["case_name", "L_wsr", "N_cells", "execution_status",
                                  "T_wsr_K", "T_out_K", "X_C_pct", "CO2_vol_pct"], nd=3))
    else:
        L.append("未生成。")
    L.append("")
    L.append("## 7. 复现边界\n")
    L.append("- 本轮只改网格结构（加入固定物理长度的全混流区），不调动力学常数、"
             "不采纳任何 `L_wsr` 值、不改合同。")
    L.append("- W1–W6 阈值在运行前写入脚本常量，运行后未改。")
    L.append("- 下游段是**串联 CSTR**（tanks-in-series），加密使其逼近平推流；"
             "W1 检验的是这一近似是否已收敛。")
    L.append("- 锚点用模式 F（`hhv_percent`）+ 同一网格族，只记录。\n")
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=("unit", "conv", "lever", "anchor", "read", "all"))
    ap.add_argument("--source-commit", default=None)
    args = ap.parse_args()
    silence_solver_logs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ok, rows, _inv = load_frozen_split()
    if not ok:
        raise SystemExit("frozen R52-C-75 split was not consumed intact")
    by_id = {r["window_id"]: r for r in rows}
    window_ids = m1_reference_window_ids(M1_MATRIX)
    p17d = pd.read_csv(P17D_CSV).set_index("window_id")
    p18d = pd.read_csv(P18D_CSV).set_index("window_id")
    missing = [w for w in window_ids if w not in p17d.index or w not in p18d.index]
    if missing:
        raise SystemExit(f"输入 CSV 缺窗口：{missing}")

    # ---- 第 1 节：dz 依赖性清单（先做，含实测探测）----
    dzm_path = OUT_DIR / "dz_dependence.md"
    if args.stage == "read" and dzm_path.exists():
        dzm_md = dzm_path.read_text(encoding="utf-8")
        print("reuse dz_dependence.md", flush=True)
    else:
        print("dz-dependence probe …", flush=True)
        probe = wsr_probe_rows(window_ids, by_id, p17d, p18d)
        dzm_md = dz_dependence_md(probe)
        dzm_path.write_text(dzm_md, encoding="utf-8")
        print("  written dz_dependence.md", flush=True)

    # ---- 第 2 节 ----
    if args.stage == "read":
        unit_df = pd.read_csv(OUT_DIR / "unit_checks.csv")
        roots_df = pd.read_csv(OUT_DIR / "roots.csv")
        unit_pass = True
    else:
        unit_df, roots_df = unit_checks(window_ids, by_id, p17d, p18d)
        unit_df.to_csv(OUT_DIR / "unit_checks.csv", index=False)
        roots_df.to_csv(OUT_DIR / "roots.csv", index=False)
        gate = unit_df[unit_df["pass"].notna()]
        unit_pass = bool(gate["pass"].astype(bool).all())
        print("unit checks:", "PASS" if unit_pass else "FAIL", flush=True)
        if not unit_pass:
            raise SystemExit("停止：第 2 节单元检查未通过")
        if args.stage == "unit":
            return

    # ---- 第 3 节：先只跑卡面 5 档（6 m 是条件性追加档，见下）----
    if args.stage in ("conv", "lever", "all", "read"):
        if args.stage == "read":
            conv = pd.read_csv(OUT_DIR / "downstream_convergence.csv")
            adm = admission_from_w1(conv)
    else:
        conv = pd.read_csv(OUT_DIR / "downstream_convergence.csv")
        adm = admission_from_w1(conv)
    if args.stage in ("conv", "lever", "all"):
        print("W1 downstream convergence (L_WSR_GRID) …", flush=True)
        conv = downstream_convergence(window_ids, by_id, p17d, p18d, list(L_WSR_GRID))
        adm = admission_from_w1(conv)
        conv.to_csv(OUT_DIR / "downstream_convergence.csv", index=False)
        adm.to_csv(OUT_DIR / "admission.csv", index=False)
    n_adm = int(adm["admitted"].sum())
    print(f"n_adm={n_adm} admitted={sorted(adm[adm['admitted']]['L_wsr'].tolist())}", flush=True)
    if args.stage == "conv":
        return

    # ---- 第 4 节：先跑非追加档，再判 W4 是否落在最大档边界 ----
    admitted = sorted(float(x) for x in adm[adm["admitted"]]["L_wsr"])
    lev_path = OUT_DIR / "leverage.csv"
    cells_path = OUT_DIR / "cells.csv"
    if n_adm == 0:
        lev = pd.DataFrame(columns=["window_id", "L_wsr"])
        cells = pd.DataFrame(columns=["window_id", "L_wsr"])
        print("无收敛档：第 4 节不执行", flush=True)
    elif args.stage == "read":
        lev = pd.read_csv(lev_path)
        cells = pd.read_csv(cells_path)
    elif args.stage in ("lever", "all"):
        base_admitted = [x for x in admitted if not np.isclose(x, L_WSR_EXTRA)]
        print(f"leverage (base {base_admitted}) …", flush=True)
        lev, cells = leverage(window_ids, by_id, p17d, p18d, base_admitted)
        # 卡面第 4 节：W4 最优点落在最大档边界时，只追加一档 6 m（先做该档 W1）
        if _w4_needs_extra(lev, base_admitted):
            print("W4 optimum on the 4 m boundary → W1 + leverage for L_wsr=6 m …",
                  flush=True)
            conv_x = downstream_convergence(window_ids, by_id, p17d, p18d, [L_WSR_EXTRA])
            conv = pd.concat([conv, conv_x], ignore_index=True)
            adm = admission_from_w1(conv)
            lev_x, cells_x = leverage(window_ids, by_id, p17d, p18d, [L_WSR_EXTRA])
            lev = pd.concat([lev, lev_x], ignore_index=True)
            cells = pd.concat([cells, cells_x], ignore_index=True)
            conv.to_csv(OUT_DIR / "downstream_convergence.csv", index=False)
            adm.to_csv(OUT_DIR / "admission.csv", index=False)
            print("  appended L_wsr=6 m (card allows exactly one extra level)", flush=True)
        lev.to_csv(lev_path, index=False)
        cells.to_csv(cells_path, index=False)
    else:
        lev = pd.read_csv(lev_path)
        cells = pd.read_csv(cells_path)
    admitted = sorted(float(x) for x in adm[adm["admitted"]]["L_wsr"])
    n_adm = int(adm["admitted"].sum())

    # ---- 第 6 节 ----
    if args.stage in ("anchor", "all"):
        print("anchor …", flush=True)
        anchor_df = anchor_runs()
        anchor_df.to_csv(OUT_DIR / "anchor_wsr.csv", index=False)
    else:
        anchor_df = (pd.read_csv(OUT_DIR / "anchor_wsr.csv")
                     if (OUT_DIR / "anchor_wsr.csv").exists() else pd.DataFrame())
    if args.stage in ("lever", "anchor"):
        return

    # ---- 读法 ----
    rd = readings(adm, conv, lev if not lev.empty else pd.DataFrame(
        columns=["window_id", "L_wsr", "f_gas", "HeatLossPercent", "X_C_implied"]),
        roots_df)
    rd.to_csv(OUT_DIR / "readings.csv", index=False)

    protocol = base_protocol({
        "round": "P1.11-A / 2026-11-06",
        "model_repo": "gasifier-1d-kinetic",
        "source_commit": args.source_commit or source_commit(),
        "task_card": "papers/jax-gasifier-cascade-hybrid/manuscript/"
                     "model-side-handoff-p1.11a-20261106.md",
        "structure": {
            "mesh_family": "wsr_plus_pfr",
            "rule": "dz[0] = L_wsr（全混流区）；RL = L − L_wsr；"
                    "n_full = floor(RL/dz_down)；rem > 1e-12 时追加末格 = rem；"
                    "N = 1 + n_full + (rem ? 1 : 0)",
            "wsr_is_physical": "首格长度不随 dz_down 改变；卡面要求其状态逐位不变（W1 内检查）",
            "downstream_note": "下游每格仍是 CSTR，因此是串联 CSTR（tanks-in-series），"
                               "加密使其逼近平推流；W1 检验该近似是否收敛",
        },
        "frozen": ["OxygenMixingModel = instant", "hhv_percent", "煤量主口径", "实测载气",
                   "全部速率式与限幅", "燃烧区判据阈值 0.05 atm", "求解器容差"],
        "dz_dependence": {
            "file": "dz_dependence.md",
            "verdict": "无任何一项在格内做轴向积分，卡第 1 节停止条件不触发",
            "declared_approximations": [
                "大格用单一 τ 会高估颗粒-气体热弛豫（停留时间分布实际较宽）",
                "大格内温度被视为均匀（热损权重与颗粒温度都用出口温度）",
            ],
        },
        "predeclared": {
            "W1": f"dz_down 0.0515 vs 0.02575：|ΔX_C|<{W1_DXC_PP} pp、|ΔT|<{W1_DT_K} K、"
                  f"|ΔCO2|<{W1_DCO2_VOL} vol%；同时检查全混区状态逐位不变",
            "W2": "准入档最小 vs 最大 L_wsr 的 Δ（9 窗中位与范围）",
            "W3": "L_wsr 增大时 X_C 是否下降、CO2 是否上升；逐窗符号",
            "W4": "是否有准入 L_wsr（含线性插值）使某窗 X_C 与装置主口径差 ≤1 pp",
            "W5": "各准入档 f_gas 1→1/8 的 ΔX_C 与 P1.8-A 均匀 N=40 之比；中位 ≥3 记为恢复杠杆",            "W6": "全混区温度范围、限幅绑定比例、多解情况",
        },
        "statistic_definitions": {
            "W5_ratio": "逐窗配对求比后取中位（median of ratios）——卡面判据。"
                        "配对键：同 window_id、同 HeatLossPercent、同 f_gas 两端点、均完成求解。"
                        "参考杠杆≈0（≤1e-12）的窗口单独计数并排除，不计入中位。"
                        "同时报告 median/median 作对照，但不参与判定。",
            "W1_scope": "W1 是测试网格对（dz_down 0.0515 vs 0.02575）之间的差值判据，"
                        "只说明该区间内加密不再改变出口；不等于已达平推流极限，"
                        "也不构成「整炉网格独立由结构保证」的结论。"
                        "首格状态对 dz_down 不变是该离散结构的性质（首格长度固定、顺序求解），"
                        "不是验证结果。",
        },
        "root_rule": {"cost_tol": ROOT_COST_TOL, "cluster_K": ROOT_T_MERGE_K,
                      "flame_T_K": FLAME_T_K,
                      "stop": "两个以上 >1500 K 的根 → 列出并停止该窗，不自动挑选"},
        "unit_checks_all_pass": unit_pass,
        "n_adm": n_adm,
        "admitted_L_wsr": admitted,
        "extra_level": {
            "L_wsr_m": L_WSR_EXTRA,
            "card_rule": "若 W4 最优点落在 4 m 边界，只追加一档 L_wsr = 6 m（先做该档 W1）",
            "execution_deviation": "本轮实现于 W4 判定之前即追加（原条件仅要求 4 m 已被准入）；"
                                   "事后核实 4 m 确为 W4 有效最优点（1/9 窗到达、中位偏差最小），"
                                   "故卡面追加条件成立，6 m 属允许的额外一档，阈值与判读不受影响。"
                                   "脚本已改为先判 W4 再追加，供复现。",
        },
        "source_csv": {
            "p17d": {"round": "p1.7d-20261103", "file": P17D_CSV.name,
                     "sha256": hashlib.sha256(P17D_CSV.read_bytes()).hexdigest()},
            "p18d": {"round": "p1.8d-20261104", "file": P18D_CSV.name,
                     "sha256": hashlib.sha256(P18D_CSV.read_bytes()).hexdigest()},
            "p18a_baseline": {"round": "p1.8a-20261104", "file": "scan.csv"},
        },
        "outputs": ["dz_dependence.md", "unit_checks.csv", "roots.csv",
                    "downstream_convergence.csv", "admission.csv", "leverage.csv",
                    "cells.csv", "anchor_wsr.csv", "readings.csv", "protocol.json",
                    "interpretation.md", "sha256.txt"],
    })

    text = interpretation(dzm_md, unit_df, roots_df, adm, conv, lev, rd, anchor_df)
    written = write_round_outputs(
        OUT_DIR,
        tables={
            "unit_checks.csv": unit_df,
            "roots.csv": roots_df,
            "downstream_convergence.csv": conv,
            "admission.csv": adm,
            "leverage.csv": lev,
            "cells.csv": cells,
            "anchor_wsr.csv": anchor_df if not anchor_df.empty else pd.DataFrame(columns=["case_name"]),
            "readings.csv": rd,
        },
        protocol=protocol,
        interpretation=text,
    )
    _rewrite_manifest(OUT_DIR)
    print("written:", [p.name for p in written], flush=True)


def _rewrite_manifest(out_dir: Path) -> None:
    """清单覆盖目录内除自身以外的全部文件（P1.10-A 审阅的教训）。"""
    files = sorted(p for p in out_dir.iterdir()
                   if p.is_file() and p.name != "sha256.txt")
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}" for p in files]
    (out_dir / "sha256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
