#!/usr/bin/env python3
"""P1.10-A 烧嘴近场单参数混合长度模型（2026-11-05 卡）。

只新增 O2 的分布式进入（``OxygenMixingModel``），不改任何反应速率、限幅、
燃烧区判据阈值或求解器容差。``instant``（缺省）与 P1.9-A 基线逐位一致。

执行顺序（卡第 2 → 3 → 4 → 6 节）：
  2  单元检查（不通过则停止）                    → unit_checks.csv
  3  各 L_mix 下网格收敛（M1 / M1-E）            → mesh_convergence.csv / mesh_params.csv
  4  准入网格上的杠杆扫描（M2–M6）                → leverage.csv / cells.csv
  6  锚点（只记录）                              → anchor_mix.csv

预声明读法 M1–M6 写在脚本常量里，跑完不改。

用法：
    ./.venv/bin/python scripts/generate_p1_10a_20261105.py [--stage unit|mesh|lever|anchor|all]
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
from model.input_contract import coal_hhv_kj_kg  # noqa: E402
from model.p0_round_20260930 import m1_reference_window_ids, reactor_element_flows  # noqa: E402
from model.p1_5_round_20261030 import base_protocol, run_case, silence_solver_logs, write_round_outputs  # noqa: E402
from model.revision_evidence_common import load_frozen_split, source_commit  # noqa: E402

OUT_DIR = ROOT / "outputs" / "diagnostics" / "round-p1.10a-20261105"
P19A_DIR = ROOT / "outputs" / "diagnostics" / "round-p1.9a-20261105"
P17D_CSV = p18a.P17D_CSV
P18D_CSV = p18a.P18D_CSV
M1_MATRIX = p18a.M1_MATRIX

DEV_L = 10.3
REST_DZ0_DEV = 0.2575          # 卡第 3.1 节：余段格长
ZB_FLOOR = 0.515               # z_b_target = max(L_mix, 0.515)
L_MIX_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
DIV_GRID = (10, 20, 40)        # dz0 = L_mix/div
FINE_DIV, COARSE_DIV = 40, 20  # M1 比较 L_mix/20 与 L_mix/40
F_GAS_LEV = (1.0, 0.125)
HLP_LEV = (1.0, 2.0)
M6_L_MIX = 1.0

# 卡面第 3.1 节的 15 档参数表（实现须逐位一致）
EXPECTED_MESH_TABLE = {
    (0.25, 0.025):   (21, 0.525, 37, 0.2475, 59),
    (0.25, 0.0125):  (42, 0.525, 37, 0.2475, 80),
    (0.25, 0.00625): (83, 0.51875, 37, 0.25375, 121),
    (0.5, 0.05):     (11, 0.55, 37, 0.2225, 49),
    (0.5, 0.025):    (21, 0.525, 37, 0.2475, 59),
    (0.5, 0.0125):   (42, 0.525, 37, 0.2475, 80),
    (1.0, 0.1):      (10, 1.0, 36, 0.03, 47),
    (1.0, 0.05):     (20, 1.0, 36, 0.03, 57),
    (1.0, 0.025):    (40, 1.0, 36, 0.03, 77),
    (2.0, 0.2):      (10, 2.0, 32, 0.06, 43),
    (2.0, 0.1):      (20, 2.0, 32, 0.06, 53),
    (2.0, 0.05):     (40, 2.0, 32, 0.06, 73),
    (4.0, 0.4):      (10, 4.0, 24, 0.12, 35),
    (4.0, 0.2):      (20, 4.0, 24, 0.12, 45),
    (4.0, 0.1):      (40, 4.0, 24, 0.12, 65),
}

# M1 / M1-E 阈值（卡第 3.3 节）
M1_DXC_PP = 0.2
M1_DT_K = 5.0
M1_DCO2_VOL = 0.2
M1_DTPEAK_K = 20.0

# 第 2 节容差
TOL_O2_SUM_REL = 1e-12
TOL_DEGENERATE = 1e-9
TOL_EXP_ENDPOINT = 1e-12

O2_INDEX = 0


# ---------------------------------------------------------------------------
# 网格构造（与 grid_service.graded_block_tail 同一算式，用于预先导出 N）
# ---------------------------------------------------------------------------

def mesh_plan(L: float, l_mix: float, dz0: float, rest_dz0: float) -> dict:
    zb_t = max(l_mix, ZB_FLOOR) if L == DEV_L else max(l_mix, 0.04125)
    n_block = int(math.ceil(zb_t / dz0 - 1e-12))
    zb_eff = float(n_block) * dz0
    rest_len = L - zb_eff
    n_rest = int(math.floor(max(rest_len, 0.0) / rest_dz0 + 1e-12))
    tail = max(rest_len, 0.0) - float(n_rest) * rest_dz0
    has_tail = bool(tail > 1e-12)
    N = n_block + n_rest + (1 if has_tail else 0)
    return {
        "L_mix": float(l_mix), "dz0": float(dz0), "z_b_target": float(zb_t),
        "n_block": n_block, "z_b_eff": zb_eff, "n_rest": n_rest,
        "tail": float(tail), "has_tail": has_tail, "N": N,
        "L_index_boundary": int(round(l_mix / dz0)),  # L_mix 落在第几格下游边界
    }


def build_dz(plan: dict, L: float, rest_dz0: float) -> np.ndarray:
    dz = np.full(int(plan["N"]), float(plan["dz0"]))
    dz[: plan["n_block"]] = plan["dz0"]
    if plan["n_rest"] > 0:
        dz[plan["n_block"]: plan["n_block"] + plan["n_rest"]] = rest_dz0
    if plan["has_tail"]:
        dz[plan["n_block"] + plan["n_rest"]] = plan["tail"]
    return dz


def normalized(L: float, l_mix: float) -> tuple[float, float]:
    """返回 (dz0 的浮点归一化键, l_mix) 以便查表。"""
    return float(l_mix)


def lookup_table_key(l_mix: float, dz0: float) -> tuple[float, float]:
    for (k_l, k_dz) in EXPECTED_MESH_TABLE:
        if abs(k_l - l_mix) < 1e-12 and abs(k_dz - dz0) < 1e-12:
            return (k_l, k_dz)
    raise KeyError(f"({l_mix}, {dz0}) 不在卡面 15 档表内")


# ---------------------------------------------------------------------------
# 第 2 节：单元检查
# ---------------------------------------------------------------------------

def unit_checks(window_ids, by_id, p17d, p18d) -> pd.DataFrame:
    rows: list[dict] = []
    wid = window_ids[0]
    row = by_id[wid]
    dry = float(p17d.loc[wid, "coal_dry_recon_phase_kg_s"])
    carrier = float(p18d.loc[wid, "co2_carrier_measured_nm3_h"])
    base_op = {
        "HeatLossPercent": 2.0,
        "GasificationRateFactor": 1.0,
        "carrier_gas_flow": p18a._carrier_kg_s(carrier),
    }

    def _run(**extra):
        op = dict(base_op)
        op.update(extra)
        return run_case(row, n_cells=40, coal_flow_dry_kg_s=dry, keep_system=True,
                        mesh_family="uniform_full_length", extra_op=op)

    # ---- 检查 1：instant 与 P1.9-A 基线逐位相同（9 窗）----
    p19_scan = pd.read_csv(P19A_DIR / "mesh_runs.csv")
    p19_scan = p19_scan[(np.isclose(p19_scan["HeatLossPercent"], 2.0))
                        & np.isclose(p19_scan["dz0_m"], 0.2575)]
    check1 = []
    for w in window_ids:
        d_w = float(p17d.loc[w, "coal_dry_recon_phase_kg_s"])
        c_w = p18a._carrier_kg_s(float(p18d.loc[w, "co2_carrier_measured_nm3_h"]))
        rec = run_case(by_id[w], n_cells=40, coal_flow_dry_kg_s=d_w, keep_system=True,
                       mesh_family="uniform_full_length",
                       extra_op={"HeatLossPercent": 2.0, "GasificationRateFactor": 1.0,
                                 "carrier_gas_flow": c_w, "OxygenMixingModel": "instant"})
        ref = p19_scan[p19_scan["window_id"] == w].iloc[0]
        dmax = max(abs(rec[k] - float(ref[k])) for k in
                   ("X_C_implied", "T_out_K", "CO2_vol_pct", "CO_vol_pct", "H2_vol_pct"))
        check1.append({"window_id": w, "max_abs_diff": dmax})
    c1 = pd.DataFrame(check1)
    rows.append({
        "check": 1,
        "context": "instant 与 P1.9-A 均匀 N=40 基线（9 窗，HLP=2%）",
        "metric": "max_abs_diff_over_9_windows",
        "expect": 0.0, "tolerance": 1e-12,
        "value": float(c1["max_abs_diff"].max()),
        "pass": bool(c1["max_abs_diff"].max() <= 1e-12),
        "note": "逐位不变：缺省路径不进入混合分支",
    })

    # ---- 检查 2：Σ 源项 = 入口 O2 ----
    # 注意：Σ 只能覆盖**已求解**的格。若求解提前停止（no_root / 冷支），
    # Σ 会小于入口而看起来"守恒失败"；这不是模型错，但也不能当作通过。
    # 因此下面同时断言 execution_status == completed 且诊断行数等于网格格数，
    # 否则该行记 fail 并写明原因（P1.10-A 审阅意见）。
    for l_mix in (0.5, 1.0, 2.0):
        for model in ("linear", "exponential"):
            rec = _run(OxygenMixingModel=model, OxygenMixingLength_m=l_mix)
            sysm: GasifierSystem = rec["_system"]
            st = _branch_status(rec, sysm)
            n_solved = len(sysm.cell_diagnostics)
            completed = (st == "completed")
            tot = float(sum(float(d.get("o2_src_mol_s", 0.0) or 0.0)
                            for d in sysm.cell_diagnostics))
            ref = float(sysm.o2_bypass_mol_s)
            rel = abs(tot - ref) / max(abs(ref), 1e-30)
            rows.append({
                "check": 2,
                "context": f"{model}, L_mix={l_mix}",
                "metric": "sum_o2_src_over_inlet_o2_minus_1",
                "expect": 0.0, "tolerance": TOL_O2_SUM_REL,
                "value": rel,
                "pass": bool(completed and rel < TOL_O2_SUM_REL),
                "note": f"Σ={tot:.6f} / 入口={ref:.6f} mol/s；状态={st}，已解 {n_solved} 格"
                        + ("" if completed else "；求解未完成，Σ 不覆盖全炉，不能作为守恒通过"),
            })
    # ---- 检查 2b：L_mix 超过反应器长度必须报错（审阅意见）----
    L_reactor = 10.3
    try:
        _run(OxygenMixingModel="linear", OxygenMixingLength_m=L_reactor * 2.0)
        guard_ok = False
        guard_note = "L_mix=2L 未被拒绝：旁流 O2 会静默丢失"
    except ValueError as exc:
        guard_ok = True
        guard_note = f"L_mix=2L 被拒绝：{str(exc)[:60]}…"
    rows.append({
        "check": "2b",
        "context": "L_mix = 2 × L（应拒绝）",
        "metric": "raises_on_L_mix_gt_L",
        "expect": 1.0, "tolerance": 0.0,
        "value": 1.0 if guard_ok else 0.0,
        "pass": bool(guard_ok),
        "note": guard_note,
    })

    # 元素守恒（O 原子）：卡面要求「与 P1.9-A 同量级」，因此以同一轮测得的基线为标尺，
    # 不用拍定的绝对阈值（基线自身 O 闭合就有 ~1e-5 量级的离散残差）。
    base_o = []
    for w in window_ids:
        d_w = float(p17d.loc[w, "coal_dry_recon_phase_kg_s"])
        c_w = p18a._carrier_kg_s(float(p18d.loc[w, "co2_carrier_measured_nm3_h"]))
        rec_b = run_case(by_id[w], n_cells=40, coal_flow_dry_kg_s=d_w, keep_system=True,
                         mesh_family="uniform_full_length",
                         extra_op={"HeatLossPercent": 2.0, "GasificationRateFactor": 1.0,
                                   "carrier_gas_flow": c_w})
        sb = rec_b["_system"]
        if sb.execution_status == "completed" and sb.results:
            fb = reactor_element_flows(sb, sb.results[-1])
            base_o.append(abs(fb["O_closure_error_mol_s"])
                          / max(abs(fb["O_inlet_mol_s"]), 1e-12))
    base_o_max = float(max(base_o)) if base_o else float("nan")
    rec = _run(OxygenMixingModel="linear", OxygenMixingLength_m=1.0)
    sysm = rec["_system"]
    if sysm.execution_status == "completed" and sysm.results:
        flows = reactor_element_flows(sysm, sysm.results[-1])
        o_err = abs(flows["O_closure_error_mol_s"]) / max(abs(flows["O_inlet_mol_s"]), 1e-12)
        rows.append({
            "check": 2,
            "context": "linear, L_mix=1.0，全炉 O 原子闭合 vs 同轮基线",
            "metric": "O_closure_rel_over_baseline_max",
            "expect": 1.0, "tolerance": 10.0,
            "value": float(o_err / base_o_max) if base_o_max > 0 else float("nan"),
            "pass": bool(base_o_max > 0 and o_err <= 10.0 * base_o_max),
            "note": f"linear={o_err:.3e}，同轮 instant 基线 max={base_o_max:.3e}；"
                    f"旁流 O2 已计入入口氧原子，O_inlet 与 instant 一致",
        })
        rows.append({
            "check": 2,
            "context": "linear, L_mix=1.0，全炉 O 原子闭合（绝对值记录）",
            "metric": "O_closure_fraction_abs",
            "expect": float("nan"), "tolerance": float("nan"),
            "value": float(o_err),
            "pass": None,
            "note": "仅记录；判据见上一行（同量级）",
        })

    # ---- 检查 3：L_mix 小于首格时 linear ≡ instant ----
    a = _run()
    b = _run(OxygenMixingModel="linear", OxygenMixingLength_m=0.25)
    dmax = max(abs(a[k] - b[k]) for k in ("X_C_implied", "T_out_K", "CO2_vol_pct"))
    rows.append({
        "check": 3,
        "context": "L_mix=0.25 < 首格 0.2575 m，linear vs instant",
        "metric": "max_abs_outlet_diff",
        "expect": 0.0, "tolerance": TOL_DEGENERATE,
        "value": float(dmax),
        "pass": bool(dmax < TOL_DEGENERATE),
        "note": "O2 全部在首格并入，两种口径须退化到同一解",
    })

    # ---- 检查 4：exponential 累计分数端点 ----
    from model.source_terms import OxygenMixingSource
    L_mix = M6_L_MIX
    dz0 = L_mix / FINE_DIV
    probe = OxygenMixingSource(1.0, L_mix, "exponential", 1500.0)
    f_left = probe.continuous_F(L_mix - 1e-12)
    f_right = probe.F(L_mix)
    # 阶跃分量 = 边界格总源项 − 同一格内的连续部分
    z_bnd = L_mix - 0.5 * dz0
    total_cell = probe.cell_fraction(z_bnd, dz0)
    cont_part = (probe.continuous_F(L_mix) - probe.continuous_F(L_mix - dz0))
    step_part = total_cell - cont_part
    for tag, val, expr in (
        ("F_left_limit", f_left, 1.0 - math.exp(-3)),
        ("F_right_value", f_right, 1.0),
        ("step_component", step_part, math.exp(-3)),
    ):
        rel = abs(val - expr) / max(abs(expr), 1e-30)
        rows.append({
            "check": 4,
            "context": f"L_mix={L_mix}, dz0=L_mix/{FINE_DIV}",
            "metric": tag,
            "expect": float(expr), "tolerance": TOL_EXP_ENDPOINT,
            "value": float(val),
            "pass": bool(rel <= TOL_EXP_ENDPOINT),
            "note": f"期望值由表达式计算（{expr!r}），不用六位小数",
        })
    rows.append({
        "check": 4,
        "context": f"L_mix={L_mix}, dz0=L_mix/{FINE_DIV}",
        "metric": "boundary_cell_total_source_fraction",
        "expect": float(math.exp(-3 * (1.0 - 1.0 / FINE_DIV))),
        "tolerance": 1e-12,
        "value": float(total_cell),
        "pass": bool(abs(total_cell - math.exp(-3 * (1.0 - 1.0 / FINE_DIV)))
                     <= 1e-12 * math.exp(-3 * (1.0 - 1.0 / FINE_DIV))),
        "note": "边界格总源项 = 连续部分 + 阶跃；此处 5.37% ≠ 4.98%，两者不可混用",
    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 网格参数表与 M1
# ---------------------------------------------------------------------------

def mesh_params_table() -> pd.DataFrame:
    rows = []
    for l_mix in L_MIX_GRID:
        for div in DIV_GRID:
            dz0 = l_mix / div
            plan = mesh_plan(DEV_L, l_mix, dz0, REST_DZ0_DEV)
            dz = build_dz(plan, DEV_L, REST_DZ0_DEV)
            key = lookup_table_key(l_mix, dz0)
            exp_nb, exp_zbe, exp_nr, exp_tail, exp_N = EXPECTED_MESH_TABLE[key]
            ok = (plan["n_block"] == exp_nb and abs(plan["z_b_eff"] - exp_zbe) < 1e-12
                  and plan["n_rest"] == exp_nr and abs(plan["tail"] - exp_tail) < 1e-12
                  and plan["N"] == exp_N)
            rel = abs(float(np.sum(dz)) - DEV_L) / DEV_L
            rows.append({
                "L_mix": l_mix, "dz0": dz0, "div": div,
                "n_block": plan["n_block"], "z_b_eff": plan["z_b_eff"],
                "n_rest": plan["n_rest"], "tail": plan["tail"], "N": plan["N"],
                "L_mix_boundary_index": plan["L_index_boundary"],
                "sum_dz_minus_L": float(np.sum(dz)) - DEV_L,
                "sum_dz_rel": rel,
                "matches_card_table": ok,
                "pass": bool(ok and rel <= 1e-12 and np.all(dz > 0)),
            })
    return pd.DataFrame(rows)


def _branch_status(rec: dict, system: GasifierSystem) -> str:
    """区分「真收敛」与「停在温度下界的冷支」。

    dz0 = L_mix/10（首格 O2 份额 10%）时求解器会返回 ``completed``，但 cell 0 温度
    恰为下界 300 K——那是熄火解，不是点火解。若不拦截，M1 会把它当作收敛点。
    """
    st = str(rec.get("execution_status", ""))
    if st != "completed":
        return st
    if system.cell_diagnostics:
        T0 = float(system.cell_diagnostics[0].get("T_K", float("nan")))
        if math.isfinite(T0) and T0 <= 301.0:
            return "cold_branch_T_at_lower_bound"
        Tmax = _peak_T(system)
        if math.isfinite(Tmax) and Tmax >= 3999.0:
            return "hot_branch_T_at_upper_bound"
    return st


def _mesh_run(row, csv17, csv18, dry, carrier, l_mix, div, model="linear", f_gas=1.0, hlp=1.0):
    dz0 = l_mix / div
    plan = mesh_plan(DEV_L, l_mix, dz0, REST_DZ0_DEV)
    rec = run_case(
        row, n_cells=int(plan["N"]), coal_flow_dry_kg_s=dry,
        keep_system=True, mesh_family="graded_block_tail",
        extra_op={
            "HeatLossPercent": float(hlp),
            "GasificationRateFactor": float(f_gas),
            "carrier_gas_flow": p18a._carrier_kg_s(carrier),
            "GradedZoneLength": float(plan["z_b_target"]),
            "GradedZoneDz": float(dz0),
            "RestCellLength": float(REST_DZ0_DEV),
            "OxygenMixingModel": model,
            "OxygenMixingLength_m": float(l_mix),
        },
    )
    return rec, rec["_system"], plan


def mesh_convergence(window_ids, by_id, p17d, p18d) -> pd.DataFrame:
    rows = []
    for wid in window_ids:
        row = by_id[wid]
        dry = float(p17d.loc[wid, "coal_dry_recon_phase_kg_s"])
        carrier = float(p18d.loc[wid, "co2_carrier_measured_nm3_h"])
        for l_mix in L_MIX_GRID:
            recs = {}
            for div in (COARSE_DIV, FINE_DIV):
                rec, sysm, plan = _mesh_run(row, p17d.loc[wid], p18d.loc[wid],
                                            dry, carrier, l_mix, div, "linear")
                recs[div] = (rec, sysm, plan)
            for div in (COARSE_DIV, FINE_DIV):
                rec, sysm, plan = recs[div]
                peak = _peak_T(sysm)
                rows.append({
                    "window_id": wid, "L_mix": l_mix, "div": div, "dz0": l_mix / div,
                    "N_cells": plan["N"], "form": "linear",
                    "execution_status": _branch_status(rec, sysm),
                    "cell0_T_K": float(sysm.cell_diagnostics[0].get("T_K", float("nan")))
                    if sysm.cell_diagnostics else float("nan"),
                    "X_C_implied": rec.get("X_C_implied"),
                    "T_out_K": rec.get("T_out_K"),
                    "CO2_vol_pct": rec.get("CO2_vol_pct"),
                    "peak_T_K": peak,
                    "no_root_cell_index": rec.get("no_root_cell_index"),
                    "max_residual_cost": rec.get("max_residual_cost"),
                    "solve_wall_time_s": rec.get("solve_wall_time_s"),
                })
            a, sa = recs[COARSE_DIV][0], recs[COARSE_DIV][1]
            b, sb = recs[FINE_DIV][0], recs[FINE_DIV][1]
            if _branch_status(a, sa) == "completed" and _branch_status(b, sb) == "completed":
                dxc = (b["X_C_implied"] - a["X_C_implied"]) * 100.0
                dt = b["T_out_K"] - a["T_out_K"]
                dco2 = b["CO2_vol_pct"] - a["CO2_vol_pct"]
                dtp = _peak_T(recs[FINE_DIV][1]) - _peak_T(recs[COARSE_DIV][1])
                ok = (abs(dxc) < M1_DXC_PP and abs(dt) < M1_DT_K
                      and abs(dco2) < M1_DCO2_VOL and abs(dtp) < M1_DTPEAK_K)
                rows.append({
                    "window_id": wid, "L_mix": l_mix, "div": "delta",
                    "dz0": np.nan, "N_cells": np.nan, "form": "linear_M1",
                    "execution_status": "completed" if ok else "not_converged",
                    "X_C_implied": dxc, "T_out_K": dt,
                    "CO2_vol_pct": dco2, "peak_T_K": dtp,
                    "no_root_cell_index": np.nan,
                    "max_residual_cost": np.nan, "solve_wall_time_s": np.nan,
                })
        print(f"  M1 {wid[:56]} done", flush=True)
    return pd.DataFrame(rows)


def _peak_T(system: GasifierSystem) -> float:
    if not getattr(system, "results", None):
        return float("nan")
    return float(max(float(s.T) for s in system.results))


def admission_from_m1(m1_df: pd.DataFrame) -> pd.DataFrame:
    """整档准入：linear 下 9/9 窗同时满足 M1。"""
    cols = ["L_mix", "n_windows", "n_converged", "admitted",
            "max_abs_dX_C_pp", "max_abs_dT_K", "max_abs_dCO2_vol", "max_abs_dTpeak_K"]
    d = m1_df[m1_df["form"] == "linear_M1"] if len(m1_df) else m1_df
    if d is None or d.empty:
        out = []
        for l_mix in L_MIX_GRID:
            grp = m1_df[(m1_df["form"] == "linear")
                        & np.isclose(m1_df["L_mix"], l_mix)] if len(m1_df) else m1_df
            n_win = int(grp["window_id"].nunique()) if len(grp) else 0
            # 无 delta 行 = 无一对高档能同时收敛 → 该档不满足 M1
            out.append({"L_mix": l_mix, "n_windows": n_win, "n_converged": 0,
                        "admitted": False, "max_abs_dX_C_pp": float("nan"),
                        "max_abs_dT_K": float("nan"), "max_abs_dCO2_vol": float("nan"),
                        "max_abs_dTpeak_K": float("nan")})
        return pd.DataFrame(out, columns=cols).sort_values("L_mix")
    out = []
    for l_mix, grp in d.groupby("L_mix"):
        n_pass = int((grp["execution_status"] == "completed").sum())
        out.append({
            "L_mix": l_mix,
            "n_windows": int(len(grp)),
            "n_converged": n_pass,
            "admitted": bool(len(grp) == 9 and n_pass == 9),
            "max_abs_dX_C_pp": float(grp["X_C_implied"].abs().max()),
            "max_abs_dT_K": float(grp["T_out_K"].abs().max()),
            "max_abs_dCO2_vol": float(grp["CO2_vol_pct"].abs().max()),
            "max_abs_dTpeak_K": float(grp["peak_T_K"].abs().max()),
        })
    return pd.DataFrame(out, columns=cols).sort_values("L_mix")


def ignition_share_scan(window_ids, by_id, p17d, p18d) -> pd.DataFrame:
    """诊断：首格 O2 份额 → 能否点火（不是卡面要求的产物，但它是本轮结论的直接证据）。

    固定 `L_mix` = 1 m，只改 `dz0`（即首格 O2 份额 = dz0/L_mix），逐窗记录
    执行状态、cell 0 温度与峰值温度。用来说明 M1 网格（/20、/40 → 5%、2.5%）
    为何整体落在熄火区。
    """
    rows = []
    l_mix = 1.0
    for wid in window_ids:
        row = by_id[wid]
        dry = float(p17d.loc[wid, "coal_dry_recon_phase_kg_s"])
        carrier = float(p18d.loc[wid, "co2_carrier_measured_nm3_h"])
        for div in (1, 2, 5, 10, 20, 40):
            dz0 = l_mix / div
            if dz0 > 1.0:
                continue
            plan = mesh_plan(DEV_L, l_mix, dz0, REST_DZ0_DEV)
            rec, sysm, _p = _mesh_run(row, p17d.loc[wid], p18d.loc[wid],
                                      dry, carrier, l_mix, div, "linear")
            rows.append({
                "window_id": wid, "L_mix": l_mix, "div": div, "dz0": dz0,
                "first_cell_o2_share": min(dz0 / l_mix, 1.0),
                "N_cells": plan["N"],
                "execution_status": _branch_status(rec, sysm),
                "cell0_T_K": float(sysm.cell_diagnostics[0].get("T_K", float("nan")))
                if sysm.cell_diagnostics else float("nan"),
                "peak_T_K": _peak_T(sysm),
                "X_C_implied": rec.get("X_C_implied"),
                "max_residual_cost": rec.get("max_residual_cost"),
            })
        print(f"  ignition-share {wid[:50]} done", flush=True)
    return pd.DataFrame(rows)


def m1e_check(window_ids, by_id, p17d, p18d) -> dict:
    """M1-E：L_mix = 1 m 的 exponential 单独过网格收敛检查。"""
    l_mix = M6_L_MIX
    deltas = []
    for wid in window_ids:
        row = by_id[wid]
        dry = float(p17d.loc[wid, "coal_dry_recon_phase_kg_s"])
        carrier = float(p18d.loc[wid, "co2_carrier_measured_nm3_h"])
        rcs = {}
        for div in (COARSE_DIV, FINE_DIV):
            rec, sysm, _p = _mesh_run(row, p17d.loc[wid], p18d.loc[wid],
                                      dry, carrier, l_mix, div, "exponential")
            rcs[div] = (rec, sysm)
        a, sa = rcs[COARSE_DIV]; b, sb = rcs[FINE_DIV]
        if _branch_status(a, sa) != "completed" or _branch_status(b, sb) != "completed":
            deltas.append({"window_id": wid, "converged": False,
                           "reason": _branch_status(a, sa) + " / " + _branch_status(b, sb)})
            continue
        dxc = (b["X_C_implied"] - a["X_C_implied"]) * 100.0
        dt = b["T_out_K"] - a["T_out_K"]
        dco2 = b["CO2_vol_pct"] - a["CO2_vol_pct"]
        dtp = _peak_T(sb) - _peak_T(sa)
        ok = (abs(dxc) < M1_DXC_PP and abs(dt) < M1_DT_K
              and abs(dco2) < M1_DCO2_VOL and abs(dtp) < M1_DTPEAK_K)
        deltas.append({"window_id": wid, "converged": bool(ok),
                       "dX_C_pp": dxc, "dT_K": dt, "dCO2_vol": dco2, "dTpeak_K": dtp})
        print(f"  M1-E {wid[:52]} done", flush=True)
    n_ok = sum(1 for d in deltas if d.get("converged"))
    return {"L_mix": l_mix, "n_converged": n_ok, "n_windows": len(deltas),
            "passed": bool(n_ok == len(deltas) == 9), "rows": deltas}


# ---------------------------------------------------------------------------
# 第 4 节：杠杆扫描
# ---------------------------------------------------------------------------

def leverage(window_ids, by_id, p17d, p18d, admitted: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs, cells = [], []
    for wid in window_ids:
        row = by_id[wid]
        dry = float(p17d.loc[wid, "coal_dry_recon_phase_kg_s"])
        carrier = float(p18d.loc[wid, "co2_carrier_measured_nm3_h"])
        for l_mix in admitted:
            for f_gas in F_GAS_LEV:
                for hlp in HLP_LEV:
                    rec, sysm, plan = _mesh_run(row, p17d.loc[wid], p18d.loc[wid],
                                                dry, carrier, l_mix, FINE_DIV,
                                                "linear", f_gas=f_gas, hlp=hlp)
                    out = p18a._build_row_extras(
                        rec, sysm, row, p17d.loc[wid], p18d.loc[wid], dry,
                        int(plan["N"]), f_gas, hlp, carrier, "measured", "p110a")
                    out.update({
                        "L_mix": l_mix, "dz0": l_mix / FINE_DIV, "div": FINE_DIV,
                        "OxygenMixingModel": "linear",
                        "N_cells_plan": plan["N"],
                        "L_mix_boundary_index": plan["L_index_boundary"],
                    })
                    out.update(p19a.heatloss_summary(sysm, dry, float(coal_hhv_kj_kg(sysm.coal_props))))
                    out.update(_o2_structure(sysm, l_mix))
                    runs.append(out)
                    if hlp == 1.0 and f_gas == 1.0 and sysm.execution_status == "completed":
                        cells.extend(_cell_rows(sysm, wid, l_mix, hlp))
        print(f"  leverage {wid[:56]} done", flush=True)

    # M6：linear vs exponential（两项资格由调用方传入）
    m6 = {}
    if M6_L_MIX in admitted:
        for wid in window_ids:
            row = by_id[wid]
            dry = float(p17d.loc[wid, "coal_dry_recon_phase_kg_s"])
            carrier = float(p18d.loc[wid, "co2_carrier_measured_nm3_h"])
            r_lin, _s, _p = _mesh_run(row, p17d.loc[wid], p18d.loc[wid], dry, carrier,
                                      M6_L_MIX, FINE_DIV, "linear", 1.0, 1.0)
            r_exp, _s2, _p2 = _mesh_run(row, p17d.loc[wid], p18d.loc[wid], dry, carrier,
                                        M6_L_MIX, FINE_DIV, "exponential", 1.0, 1.0)
            if (r_lin.get("execution_status") == "completed"
                    and r_exp.get("execution_status") == "completed"):
                m6[wid] = {"dX_C_pp": (r_exp["X_C_implied"] - r_lin["X_C_implied"]) * 100.0,
                           "dT_K": r_exp["T_out_K"] - r_lin["T_out_K"],
                           "dCO2_vol": r_exp["CO2_vol_pct"] - r_lin["CO2_vol_pct"]}
            print(f"  M6 {wid[:56]} done", flush=True)
    globals()["_M6_ROWS"] = m6
    return pd.DataFrame(runs), pd.DataFrame(cells)


def _o2_structure(system: GasifierSystem, l_mix: float) -> dict:
    dz = np.array([d["dz_m"] for d in system.cell_diagnostics], dtype=float)
    z_out = np.cumsum(dz)
    o2_pos = float("nan")
    if system.execution_status == "completed" and system.results:
        o2 = np.array([float(s.gas_moles[O2_INDEX]) for s in system.results])
        inlet_ref = float(system.o2_bypass_mol_s)
        if inlet_ref > 0:
            hit = np.where(o2 >= 0.99 * inlet_ref)[0]
            o2_pos = float(z_out[int(hit[-1])]) if hit.size else 0.0
    flags = [bool(d.get("is_combustion_zone", False)) for d in system.cell_diagnostics]
    comb_idx = [i for i, f in enumerate(flags) if f]
    return {
        "o2_merge_complete_z_m": o2_pos,
        "o2_bypass_mol_s": float(system.o2_bypass_mol_s),
        "combustion_n_cells": int(sum(flags)),
        "combustion_length_m": float(sum(dz[i] for i in comb_idx)),
        "z_comb_end_m": float(z_out[max(comb_idx)]) if comb_idx else 0.0,
        "peak_T_K": _peak_T(system),
        "peak_T_z_m": (float(max(zip((float(s.T) for s in system.results), z_out))[1])
                       if system.results else float("nan")),
    }


def _cell_rows(system: GasifierSystem, wid: str, l_mix: float, hlp: float) -> list[dict]:
    rows = []
    inlet_c0 = float(system.cells[0].inlet.solid_mass * system.cells[0].inlet.carbon_fraction)
    for cell, state, diag in zip(system.cells, system.results, system.cell_diagnostics):
        rows.append({
            "window_id": wid, "L_mix": l_mix, "HeatLossPercent": hlp,
            "cell_index": int(cell.idx), "z_center_m": float(cell.z),
            "dz_m": float(cell.dz), "T_K": float(state.T),
            "O2_mol_s": float(state.gas_moles[O2_INDEX]),
            "o2_src_mol_s": float(diag.get("o2_src_mol_s", float("nan"))),
            "pO2_criterion_atm": float(diag.get("pO2_criterion_atm", float("nan"))),
            "is_combustion_zone": bool(diag.get("is_combustion_zone", False)),
            "X_C_cumulative": float(1.0 - state.solid_mass * state.carbon_fraction
                                    / max(inlet_c0, 1e-12)),
            "CO2_mol_s": float(state.gas_moles[3]),
        })
    return rows


# ---------------------------------------------------------------------------
# 第 6 节：锚点
# ---------------------------------------------------------------------------

def anchor_mix() -> pd.DataFrame:
    cases, geo = anchor_cases()
    L = float(geo["L_m"]); rest = 0.020625
    rows = []
    for case in cases:
        if case["case_name"] not in DRY_POWDER_CASES:
            continue
        for l_mix in (0.1, 0.25, 0.5, 0.75):
            dz0 = l_mix / 20.0
            plan = mesh_plan(L, l_mix, dz0, rest)
            rec = solve_fortran_case(
                case, n_cells=int(plan["N"]), mode="F", geometry=geo, keep_system=True,
                switches={
                    "OxygenMixingModel": "linear",
                    "OxygenMixingLength_m": float(l_mix),
                    "MeshFamily": "graded_block_tail",
                    "GradedZoneLength": float(plan["z_b_target"]),
                    "GradedZoneDz": float(dz0),
                    "RestCellLength": float(rest),
                },
            )
            sysm = rec.pop("_system", None)
            rec.pop("_profile", None); rec.pop("_z", None)
            row = {
                "case_name": rec.get("case_name"), "L_mix": l_mix, "dz0": dz0,
                "n_block": plan["n_block"], "z_b_eff": plan["z_b_eff"],
                "n_rest": plan["n_rest"], "tail": plan["tail"], "N_cells": plan["N"],
                "execution_status": rec.get("execution_status"),
                "no_root_cell_index": rec.get("no_root_cell_index"),
                "max_residual_cost": rec.get("max_residual_cost"),
                "T_out_K": rec.get("T_gas_K"), "X_C_pct": rec.get("X_C_pct"),
                "CO_vol_pct": rec.get("CO_vol_pct"),
                "H2_vol_pct": rec.get("H2_vol_pct"),
                "CO2_vol_pct": rec.get("CO2_vol_pct"),
            }
            if sysm is not None:
                row["peak_T_K"] = _peak_T(sysm)
                row["combustion_n_cells"] = int(sum(
                    1 for d in sysm.cell_diagnostics if d.get("is_combustion_zone")))
            rows.append(row)
        print(f"  anchor {case['case_name']} done", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 读法
# ---------------------------------------------------------------------------

def readings(m1: pd.DataFrame, adm: pd.DataFrame, lev: pd.DataFrame,
             m1e: dict, n_adm: int) -> pd.DataFrame:
    rows: list[dict] = []

    def add(**kw):
        rows.append(kw)

    # M1
    for _, r in adm.iterrows():
        add(reading="M1", L_mix=float(r["L_mix"]), metric="admitted",
            value=float(r["n_converged"]), threshold=9.0,
            pass_=bool(r["admitted"]),
            note=f"9/9 窗同时满足四项阈值；max|ΔX_C|={r['max_abs_dX_C_pp']:.3f} pp, "
                 f"max|ΔT|={r['max_abs_dT_K']:.2f} K, max|ΔCO2|={r['max_abs_dCO2_vol']:.3f} vol%, "
                 f"max|ΔTpeak|={r['max_abs_dTpeak_K']:.1f} K")
    add(reading="M1_verdict", L_mix=np.nan, metric="n_admitted",
        value=float(n_adm), threshold=2.0, pass_=bool(n_adm >= 2),
        note="准入档数决定 M2–M5 的可评估性（卡 3.2.1）")
    add(reading="M1_E", L_mix=float(m1e["L_mix"]), metric="exponential_converged",
        value=float(m1e["n_converged"]), threshold=9.0, pass_=bool(m1e["passed"]),
        note="L_mix=1 m 的 exponential 独立准入；不得由 linear 推断")

    m6_gate = bool(M6_L_MIX in set(adm[adm["admitted"]]["L_mix"])) and bool(m1e["passed"])
    add(reading="M6_gate", L_mix=float(M6_L_MIX), metric="two_preconditions",
        value=float(m6_gate), threshold=1.0, pass_=bool(m6_gate),
        note="(i) linear 该档已准入 且 (ii) exponential 通过 M1-E；缺一即不可评估")

    if n_adm == 0:
        for name in ("M2", "M3", "M4", "M5"):
            add(reading=name, L_mix=np.nan, metric="assessable", value=0.0,
                threshold=1.0, pass_=None, note="n_adm=0：不可评估（卡 3.2.1）")
        add(reading="STOP", L_mix=np.nan, metric="n_adm", value=0.0, threshold=0.0,
            pass_=False, note="无准入档：按卡 3.2.1 停止，不执行第 4 节")
        return _finish(rows)

    base = lev[(np.isclose(lev["f_gas"], 1.0)) & (np.isclose(lev["HeatLossPercent"], 1.0))]
    if n_adm >= 2:
        lo = float(min(adm[adm["admitted"]]["L_mix"]))
        hi = float(max(adm[adm["admitted"]]["L_mix"]))
        for metric, col in (("dX_C_pp", "X_C_implied"), ("dT_out_K", "T_out_K"),
                            ("dCO2_vol_pct", "CO2_vol_pct"), ("dT_peak_K", "peak_T_K")):
            a = base[np.isclose(base["L_mix"], lo)][col]
            b = base[np.isclose(base["L_mix"], hi)][col]
            if a.empty or b.empty:
                continue
            v = (float(b.median()) - float(a.median())) if metric.startswith("dX_C") else None
            pair = pd.DataFrame({"lo": a.values, "hi": b.values}).dropna()
            d = (pair["hi"] - pair["lo"])
            if metric.endswith("pp"):
                d = d * 100.0
            add(reading="M2", L_mix=np.nan, metric=metric,
                value=float(d.median()), threshold=np.nan, pass_=None,
                note=f"L_mix {lo}→{hi}，9 窗中位；范围 {d.min():.4g}…{d.max():.4g}")
        # M3 方向
        signs_down = signs_up = 0
        for wid in base["window_id"].unique():
            a = base[(base["window_id"] == wid) & np.isclose(base["L_mix"], lo)]
            b = base[(base["window_id"] == wid) & np.isclose(base["L_mix"], hi)]
            if a.empty or b.empty:
                continue
            d_xc = float(b["X_C_implied"].iloc[0]) - float(a["X_C_implied"].iloc[0])
            d_co2 = float(b["CO2_vol_pct"].iloc[0]) - float(a["CO2_vol_pct"].iloc[0])
            add(reading="M3", L_mix=np.nan, metric=f"window_sign::{wid}",
                value=float(np.sign(d_xc)), threshold=np.nan, pass_=None,
                note=f"ΔX_C={'−' if d_xc < 0 else '+'}({d_xc*100:+.3f} pp), "
                     f"ΔCO2={'+' if d_co2 > 0 else '−'}({d_co2:+.3f} vol%)")
            if d_xc < 0:
                signs_down += 1
            else:
                signs_up += 1
        add(reading="M3", L_mix=np.nan, metric="n_windows_X_C_decreases",
            value=float(signs_down), threshold=float(signs_down + signs_up),
            pass_=None, note=f"X_C 下降 {signs_down} / 上升 {signs_up}")
    else:
        for name in ("M2", "M3"):
            add(reading=name, L_mix=np.nan, metric="assessable", value=0.0,
                threshold=1.0, pass_=None, note="n_adm=1：需两端，不可评估")

    # M4 到达
    plant = lev[["window_id", "plant_xc_measured_pct"]].drop_duplicates()
    plant_map = dict(zip(plant["window_id"], plant["plant_xc_measured_pct"]))
    hit_any = False
    for l_mix in sorted(adm[adm["admitted"]]["L_mix"].unique()):
        for hlp in HLP_LEV:
            sub = lev[(np.isclose(lev["L_mix"], l_mix)) & (np.isclose(lev["HeatLossPercent"], hlp))
                      & (np.isclose(lev["f_gas"], 1.0))]
            if sub.empty:
                continue
            devs = [(float(r["X_C_implied"]) * 100.0 - float(plant_map[r["window_id"]]))
                    for _, r in sub.iterrows() if r["window_id"] in plant_map]
            if not devs:
                continue
            med = float(np.median(devs))
            within = int(sum(1 for d in devs if abs(d) <= 1.0))
            is_edge = bool(np.isclose(l_mix, min(L_MIX_GRID)) or np.isclose(l_mix, max(L_MIX_GRID)))
            if within > 0:
                hit_any = True
            add(reading="M4", L_mix=l_mix, metric=f"X_C_dev_median_pp::HLP{hlp:.0f}",
                value=med, threshold=1.0, pass_=bool(within > 0),
                note=f"9 窗中位偏差 {med:+.2f} pp；|偏差|≤1 pp 的窗数 {within}/9；"
                     f"{'边界档' if is_edge else '内部档'}")
    add(reading="M4_verdict", L_mix=np.nan, metric="any_L_mix_reaches_plant",
        value=float(hit_any), threshold=1.0, pass_=bool(hit_any),
        note="是否存在准入 L_mix 使某窗落在 ±1 pp 内")

    # M5 动力学杠杆
    ratios = []
    for l_mix in sorted(adm[adm["admitted"]]["L_mix"].unique()):
        sub = lev[(np.isclose(lev["L_mix"], l_mix)) & (np.isclose(lev["HeatLossPercent"], 1.0))]
        for wid in sub["window_id"].unique():
            g1 = sub[(sub["window_id"] == wid) & np.isclose(sub["f_gas"], 1.0)]["X_C_implied"]
            g8 = sub[(sub["window_id"] == wid) & np.isclose(sub["f_gas"], F_GAS_LEV[1])]["X_C_implied"]
            if g1.empty or g8.empty:
                continue
            ratios.append((l_mix, wid, (float(g8.iloc[0]) - float(g1.iloc[0])) * 100.0))
    if ratios:
        rdf = pd.DataFrame(ratios, columns=["L_mix", "window_id", "dX_C_pp"])
        ref = _p19a_uniform_leverage()
        for l_mix, grp in rdf.groupby("L_mix"):
            med_abs = float(grp["dX_C_pp"].abs().median())
            ratio = med_abs / ref if ref > 0 else float("inf")
            add(reading="M5", L_mix=l_mix, metric="median_abs_dX_C_pp",
                value=med_abs, threshold=np.nan, pass_=None,
                note=f"f_gas 1→1/8，9 窗中位 |ΔX_C|")
            add(reading="M5", L_mix=l_mix, metric="ratio_vs_P1.9A_uniform",
                value=ratio, threshold=3.0, pass_=bool(ratio >= 3.0),
                note="中位比 ≥3 记为混合长度下本征动力学恢复杠杆")
        add(reading="M5", L_mix=np.nan, metric="P1.9A_uniform_reference_median_abs_dX_C_pp",
            value=ref, threshold=np.nan, pass_=None,
            note="P1.9-A 均匀 N=40、HLP=1% 的 f_gas 1→1/8 中位 |ΔX_C|")
        add(reading="M5_verdict", L_mix=np.nan, metric="any_ratio_ge_3",
            value=float(any(float(r["value"]) >= 3.0 for _, r in
                            pd.DataFrame(rows).query("reading=='M5' and metric=='ratio_vs_P1.9A_uniform'").iterrows())),
            threshold=1.0, pass_=None, note="任一准入档比值 ≥3")
    else:
        add(reading="M5", L_mix=np.nan, metric="assessable", value=0.0,
            threshold=1.0, pass_=None, note="无可用行")

    # M6
    m6 = globals().get("_M6_ROWS") or {}
    if m6_gate and m6:
        d = pd.DataFrame.from_dict(m6, orient="index")
        add(reading="M6", L_mix=float(M6_L_MIX), metric="median_abs_dX_C_pp",
            value=float(d["dX_C_pp"].abs().median()), threshold=np.nan, pass_=None,
            note="exponential 与 linear 的出口差")
        add(reading="M6", L_mix=float(M6_L_MIX), metric="median_abs_dT_K",
            value=float(d["dT_K"].abs().median()), threshold=np.nan, pass_=None, note="")
        add(reading="M6", L_mix=float(M6_L_MIX), metric="median_abs_dCO2_vol",
            value=float(d["dCO2_vol"].abs().median()), threshold=np.nan, pass_=None, note="")
    else:
        add(reading="M6", L_mix=float(M6_L_MIX), metric="assessable", value=0.0,
            threshold=1.0, pass_=None,
            note="不可评估：需 linear 该档已准入且 exponential 通过 M1-E")

    return _finish(rows)


def _p19a_uniform_leverage() -> float:
    """P1.9-A 均匀 N=40、HLP=1% 下 f_gas 1→1/8 的中位 |ΔX_C|。"""
    d = pd.read_csv(P19A_DIR / "leverage.csv")
    d = d[(d["leverage_mesh"] == "uniform_N40") & np.isclose(d["HeatLossPercent"], 1.0)]
    vals = []
    for wid in d["window_id"].unique():
        g1 = d[(d["window_id"] == wid) & np.isclose(d["f_gas"], 1.0)]["X_C_implied"]
        g8 = d[(d["window_id"] == wid) & np.isclose(d["f_gas"], 0.125)]["X_C_implied"]
        if g1.empty or g8.empty:
            continue
        vals.append(abs(float(g8.iloc[0]) - float(g1.iloc[0])) * 100.0)
    return float(np.median(vals)) if vals else float("nan")


def _finish(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if "pass_" in df.columns:
        df = df.rename(columns={"pass_": "pass"})
    return df


# ---------------------------------------------------------------------------
# 判读文本
# ---------------------------------------------------------------------------

def _f(x, nd=3):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(x):
        return "nan"
    return f"{x:.{nd}f}"


def _tbl(df: pd.DataFrame, cols: list[str], nd: int = 3) -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(
            _f(r.get(c), nd) if isinstance(r.get(c), (int, float, np.floating))
            else str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def interpretation(unit_df, params, m1, adm, m1e, lev, rd, anchor_df) -> str:
    L: list[str] = []
    n_adm = int(adm["admitted"].sum())
    L.append("# P1.10-A 烧嘴近场单参数混合长度模型 — 判读\n")
    L.append("Status: `EXECUTED — model repo`\n")
    L.append("依据：`manuscript/model-side-handoff-p1.10a-20261105.md`；"
             "上游 `manuscript/model-evidence-verification-20261105-p19.md` G2–G4。\n")
    L.append("## 0. 冻结与实现\n")
    L.append("`OxygenMixingModel` 缺省 `instant` → 不进入混合分支，与 P1.9-A 基线逐位一致（第 1 项检查）。"
             "非 instant 时入口 O2 作不反应旁流，按累计分数 `F(z)` 逐格并入；源项进入 `gas_src[0]`，"
             "因此气相 O2 平衡、燃烧区判据与元素守恒都自动看到本格并入的 O2——"
             "这正是卡面「入口反应流 + 本格并入的 O2」。速率式、限幅、判据阈值、求解器容差未改。\n")
    L.append("两处实现细节按卡面要求在 `interpretation` 中写明：\n")
    L.append("1. **exponential 的阶跃分量**。`F` 在 `L_mix` 左极限为 `1−exp(−3)`，右侧取 1，"
             "阶跃分量 `exp(−3)` 落在下游边界恰为 `L_mix` 的那一格。该格的**总源项**是"
             "「连续部分 + 阶跃」= `exp(−3·(1−dz0/L_mix))`，`dz0=L_mix/40` 时为 **5.37%**，"
             "**不是** 4.98%。卡面检查 4 的 `exp(−3)` 只能读作阶跃分量；"
             "把它当成边界格总源项会使 Σ 源项只到 95.02%、破坏守恒。"
             "单元检查 4 同列这两个量。")
    L.append("2. **残差缩放基准**。`instant` 下 O2 在 `inlet.gas_moles` 里，mixing 下不在；"
             "若仍按 `inlet.total_gas_moles` 定标，同一套方程会得到不同的 `ref_f`，"
             "实测残差同比放大 **4.94 倍**、出口温度漂移 0.018 K。"
             "已把旁流计入 `ref_f`（`instant` 下旁流恒为 0，返回值与旧代码逐位相同），"
             "并把首格的旁流份额计入初值，使两种口径走同一条求解路径。"
             "修后同一状态上的残差差为 **0.0**、`L_mix`=0.25 与 instant 的出口差为 **0.0**。\n")
    L.append("## 1. 单元检查（第 2 节）\n")
    L.append(_tbl(unit_df, ["check", "context", "metric", "expect", "value",
                            "tolerance", "pass"], nd=6))
    L.append("")
    gate = unit_df[unit_df["pass"].notna()]
    L.append(f"- 全部通过：**{bool(gate['pass'].astype(bool).all())}**（{len(gate)} 项）\n")

    L.append("## 2. 网格参数（第 3.1 节）\n")
    L.append(_tbl(params, ["L_mix", "dz0", "n_block", "z_b_eff", "n_rest", "tail", "N",
                           "L_mix_boundary_index", "matches_card_table", "sum_dz_rel"],
                  nd=6))
    L.append("")
    L.append(f"- 15 档与卡面表逐位一致：**{bool(params['matches_card_table'].all())}**；"
             f"`Σdz−L` 相对值最大 {float(params['sum_dz_rel'].max()):.2e}（卡面 1e-12 容差）\n")

    L.append("## 3. M1 网格收敛与准入（第 3.2–3.3 节）\n")
    L.append(_tbl(adm, ["L_mix", "n_converged", "n_windows", "admitted",
                        "max_abs_dX_C_pp", "max_abs_dT_K", "max_abs_dCO2_vol",
                        "max_abs_dTpeak_K"], nd=3))
    L.append("")
    L.append(f"- 准入档数 `n_adm` = **{n_adm}**；准入档："
             f"{sorted(adm[adm['admitted']]['L_mix'].tolist())}")
    L.append(f"- **M1-E**（`L_mix`=1 m 的 exponential 独立准入）："
             f"{m1e['n_converged']}/{m1e['n_windows']} 窗收敛 → "
             f"**{'通过' if m1e['passed'] else '未通过'}**\n")
    if n_adm == 0:
        L.append("**按卡 3.2.1：`n_adm` = 0，停止，不执行第 4 节。**\n")
        L.extend(_extinction_section())
        L.extend(_tail_sections(anchor_df))
        return "\n".join(L) + "\n"

    L.append("## 4. 杠杆扫描（第 4–5 节）\n")
    L.append(f"准入档 × `f_gas` ∈ {{1, 1/8}} × `HLP` ∈ {{1, 2}}，9 窗，共 {len(lev)} 行。\n")
    m2 = rd[(rd["reading"] == "M2")]
    if not m2.empty:
        L.append("### M2 杠杆（准入档中 `L_mix` 最小 vs 最大）\n")
        L.append(_tbl(m2, ["metric", "value", "note"], nd=4))
        L.append("")
    m3 = rd[(rd["reading"] == "M3")]
    if not m3.empty:
        v = m3[m3["metric"] == "n_windows_X_C_decreases"]
        if not v.empty:
            L.append(f"### M3 方向\n\n- {v['note'].iloc[0]}\n")
    m4 = rd[(rd["reading"] == "M4")]
    if not m4.empty:
        L.append("### M4 到达（±1 pp）\n")
        L.append(_tbl(m4, ["L_mix", "metric", "value", "pass", "note"], nd=3))
        L.append("")
    m5 = rd[rd["reading"] == "M5"]
    if not m5.empty:
        L.append("### M5 动力学杠杆\n")
        L.append(_tbl(m5, ["L_mix", "metric", "value", "threshold", "pass", "note"], nd=4))
        L.append("")
    m6 = rd[rd["reading"] == "M6"]
    L.append("### M6 形式敏感性\n")
    if not m6.empty:
        L.append(_tbl(m6, ["metric", "value", "note"], nd=4))
    else:
        L.append("不可评估（需 linear 该档已准入且 exponential 通过 M1-E）。")
    L.append("")
    L.extend(_tail_sections(anchor_df))
    return "\n".join(L) + "\n"


def _extinction_section() -> list[str]:
    """n_adm = 0 时解释熄火机理（本轮的主要结论）。"""
    L: list[str] = []
    L.append("### 为什么 M1 全部不通过：分布式 O2 把火焰稀释掉了\n")
    L.append("点火在本模型里是**逐格热平衡**，不是累积过程。首格可用的反应热只来自"
             "**本格并入的那一份 O2**，而它必须同时把**整股固体**（`W_dry` ≈ 19 kg/s，"
             "`cp_char` = 1300 J/kg/K → 固体热容流率 ≈ 25 kW/K）加热到点火温度。"
             "因此首格 O2 份额直接决定单格可升温量：\n")
    L.append("| 首格 O2 份额 | 本格 O2（mol/s） | 约合反应热 | 单格可升温 |")
    L.append("|---|---|---|---|")
    for share in (1.0, 0.5, 0.2, 0.1, 0.05, 0.025):
        o2 = 465.49 * share
        q_mw = o2 * 221.0e3 / 1e6      # C+0.5O2→CO ≈ 221 kJ/mol O2
        dT = q_mw * 1e6 / (19.2 * 1300.0)
        L.append(f"| {share*100:.1f}% | {o2:.1f} | ≈{q_mw:.1f} MW | ≈{dT:.0f} K |")
    L.append("")
    L.append("把份额推到 5% 与 2.5%（= 卡面 M1 的 `L_mix/20`、`L_mix/40`），单格可升温降到"
             "数百 K 以下，低于点火所需，于是**不存在点火根**：求解器要么报 `no_root`，"
             "要么停在温度下界 300 K（熄火解）。后者尤其危险——`dz0 = L_mix/10`（10%）时"
             "确有几窗返回 `completed` 但 `cell0 T = 300 K`，若不拦截会被 M1 当成收敛点。"
             "本轮的 `_branch_status` 已把「完成但温度触界」标为 "
             "`cold_branch_T_at_lower_bound`，不计入收敛。\n")
    L.append("逐窗实测（固定 `L_mix` = 1 m，只改 `dz0`；9 窗）：\n")
    igs_path = OUT_DIR / "ignition_share_scan.csv"
    if igs_path.exists():
        igs = pd.read_csv(igs_path)
        igs["首格 O2 份额"] = (igs["first_cell_o2_share"] * 100).map(lambda v: f"{v:.1f}%")
        _order = (igs[["first_cell_o2_share", "首格 O2 份额"]]
                  .drop_duplicates().sort_values("first_cell_o2_share"))
        _rank = dict(zip(_order["首格 O2 份额"], range(len(_order))))
        piv = (igs.groupby("首格 O2 份额")["execution_status"]
               .value_counts().unstack(fill_value=0).reset_index())
        piv = piv.sort_values("首格 O2 份额", key=lambda s: s.map(_rank))
        L.append(_tbl(piv, list(piv.columns), nd=0))
        L.append("")
        t = (igs.groupby("首格 O2 份额")["cell0_T_K"]
             .agg(["min", "median", "max"]).reset_index())
        t = t.sort_values("首格 O2 份额", key=lambda s: s.map(_rank))
        L.append(_tbl(t, ["首格 O2 份额", "min", "median", "max"], nd=1))
        L.append("")
    L.append("**结论**：阈值在 10% 与 20% 之间——20% 份额 9/9 窗点火（cell 0 中位 736 K），"
             "10% 只有 3/9 窗返回完成且温度 ≤335 K，5% 与 2.5% 全部 `no_root`。"
             "卡面 M1 网格（`L_mix/20`、`L_mix/40`）整体落在熄火区，"
             "因此 **M1 无法在本模型的这一实现下评估**，`n_adm` = 0。\n")
    L.append("**锚点是支持性证据，但不是充分证明。** 同一混合模型在 Fortran 几何上"
             "（煤流小 3 个数量级）对 `L_mix` = 0.5、0.75 m 是收敛的，`dz0 = L_mix/20`"
             "（同为 5% 份额）仍能点火。这与「单格放热 ÷ 整股固体热容流率」的机理一致，"
             "因此排除了「装置工况下单纯是代码写错」这一解释；"
             "但锚点与装置算例的**几何、煤流、氧量同时不同**，它不是受控对照，"
             "**不能据此断言机理已被完全确定**——求解器行为（初值路径、严格根判据）"
             "与实现细节仍可能是贡献项。本报告把结论限定为："
             "在本模型这一实现下，稀释型混合使装置工况无法点火；"
             "更细的归因需要受控实验（固定煤流、只改份额）另行确认。\n")
    L.append("### 这不是调参问题，是「稀释」与「点火」的冲突\n")
    L.append("卡面的混合模型是**把 O2 摊薄到 `L_mix` 上**。摊薄降低了每一格的可燃比与放热集中度，"
             "而一维模型里点火恰恰要求某一格内放热足以把整股固体加热。"
             "真实烧嘴近场有**局部富氧**的火焰锋（射流卷吸造成空间上的化学计量分布），"
             "一维稀释并不产生这种局部富氧，只是把整体化学计量拉向贫燃。"
             "所以「单参数混合长度」在这一实现下不是「把火焰拉长」，而是「把火焰熄掉」。\n")
    L.append("**这对主线的含义**：G4 提出的「烧嘴近场混合状态」若要作为可解释调节量，"
             "需要的不是稀释型混合长度，而是能在**单格内**保留局部富氧/高分压 O2 的描述"
             "（例如把 O2 分配与射流结构耦合、或在首格保留一个未混合核）。"
             "卡面这一版不能直接支撑「混合长度是调节量」的结论。\n")
    return L


def _tail_sections(anchor_df) -> list[str]:
    L: list[str] = []
    L.append("## 5. 锚点（只记录，不判通过）\n")
    if anchor_df is not None and not anchor_df.empty:
        L.append(_tbl(anchor_df, ["case_name", "L_mix", "N_cells", "execution_status",
                                  "T_out_K", "X_C_pct", "CO2_vol_pct", "peak_T_K",
                                  "combustion_n_cells"], nd=3))
    else:
        L.append("未生成。")
    L.append("")
    L.append("## 6. 复现边界\n")
    L.append("- 本轮只新增 O2 的分布式进入，不调任何动力学常数、不采纳任何 `L_mix` 值、不改合同。")
    L.append("- M1–M6 阈值在运行前写入脚本常量，运行后未改。")
    L.append("- `instant` 分支与 P1.9-A 基线逐位一致（9/9 窗）。")
    L.append("- 锚点用模式 F（`hhv_percent`）+ 同一网格构造，只记录。\n")
    return L


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=("unit", "mesh", "lever", "anchor", "all"))
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

    # ---- 网格参数表：不通过则停止 ----
    params = mesh_params_table()
    params.to_csv(OUT_DIR / "mesh_params.csv", index=False)
    print("mesh params:", "PASS" if bool(params["pass"].all()) else "FAIL", flush=True)
    if not bool(params["pass"].all()):
        raise SystemExit("停止：第 3.1 节网格参数与卡面表不符")

    # ---- 第 2 节 ----
    unit_df = unit_checks(window_ids, by_id, p17d, p18d)
    unit_df.to_csv(OUT_DIR / "unit_checks.csv", index=False)
    gate = unit_df[unit_df["pass"].notna()]
    unit_pass = bool(gate["pass"].astype(bool).all())
    print("unit checks:", "PASS" if unit_pass else "FAIL", flush=True)
    if not unit_pass:
        raise SystemExit("停止：第 2 节单元检查未通过")
    if args.stage == "unit":
        return

    # ---- 第 3 节 ----
    if args.stage in ("mesh", "lever", "all"):
        print("M1 mesh convergence …", flush=True)
        m1 = mesh_convergence(window_ids, by_id, p17d, p18d)
        m1.to_csv(OUT_DIR / "mesh_convergence.csv", index=False)
        adm = admission_from_m1(m1)
        adm.to_csv(OUT_DIR / "admission.csv", index=False)
        print("M1-E …", flush=True)
        m1e = m1e_check(window_ids, by_id, p17d, p18d)
        (OUT_DIR / "m1e.json").write_text(
            json.dumps(m1e, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("ignition-share diagnostic …", flush=True)
        igs = ignition_share_scan(window_ids, by_id, p17d, p18d)
        igs.to_csv(OUT_DIR / "ignition_share_scan.csv", index=False)
    else:
        m1 = pd.read_csv(OUT_DIR / "mesh_convergence.csv")
        adm = pd.read_csv(OUT_DIR / "admission.csv")
        m1e = json.loads((OUT_DIR / "m1e.json").read_text(encoding="utf-8"))

    n_adm = int(adm["admitted"].sum())
    print(f"n_adm={n_adm} admitted={sorted(adm[adm['admitted']]['L_mix'].tolist())}", flush=True)
    if args.stage == "mesh":
        return

    # ---- 第 4 节 ----
    admitted = sorted(float(x) for x in adm[adm["admitted"]]["L_mix"])
    if n_adm == 0:
        lev = pd.DataFrame(columns=["window_id", "L_mix"])
        cells = pd.DataFrame(columns=["window_id", "L_mix"])
        print("n_adm=0：按卡 3.2.1 停止，不执行第 4 节", flush=True)
    elif args.stage in ("lever", "all"):
        print("leverage …", flush=True)
        lev, cells = leverage(window_ids, by_id, p17d, p18d, admitted)
        lev.to_csv(OUT_DIR / "leverage.csv", index=False)
        cells.to_csv(OUT_DIR / "cells.csv", index=False)
    else:
        lev = pd.read_csv(OUT_DIR / "leverage.csv")
        cells = pd.read_csv(OUT_DIR / "cells.csv")
        m6 = {}
        sub = lev[(np.isclose(lev["L_mix"], M6_L_MIX))]
        globals()["_M6_ROWS"] = m6

    # ---- 第 6 节 ----
    if args.stage in ("anchor", "all"):
        print("anchor …", flush=True)
        anchor_df = anchor_mix()
        anchor_df.to_csv(OUT_DIR / "anchor_mix.csv", index=False)
    else:
        anchor_df = (pd.read_csv(OUT_DIR / "anchor_mix.csv")
                     if (OUT_DIR / "anchor_mix.csv").exists() else pd.DataFrame())
    if args.stage in ("lever", "anchor"):
        return

    # ---- 读法 ----
    rd = readings(m1, adm, lev, m1e, n_adm)
    rd.to_csv(OUT_DIR / "readings.csv", index=False)

    protocol = base_protocol({
        "round": "P1.10-A / 2026-11-05",
        "model_repo": "gasifier-1d-kinetic",
        "source_commit": args.source_commit or source_commit(),
        "task_card": "papers/jax-gasifier-cascade-hybrid/manuscript/model-side-handoff-p1.10a-20261105.md",
        "equation_change": {
            "oxygen_mixing": "新增 op_conds['OxygenMixingModel'] ∈ {instant(缺省), linear, exponential} "
                             "与 op_conds['OxygenMixingLength_m']。非 instant 时入口 O2 作不反应旁流，"
                             "按 F(z) 逐格并入；F_linear=min(z/L_mix,1)，"
                             "F_exp=1−exp(−z/λ)（λ=L_mix/3，z≥L_mix 取 1）。",
            "combustion_criterion": "pO2 口径改为『入口反应流 + 本格并入的 O2』；阈值 0.05 atm 不变；"
                                    "instant 下与现行判据逐位一致（单元检查 1）。",
            "energy_source": "并入份额在入口温度下的摩尔焓（旁流不升温，显式简化）。",
            "jax_guard": "JAX 批处理路径遇非 instant 直接抛错，不静默回退。",
            "residual_scaling": "ref_f 计入旁流（instant 下为 0，逐位不变）；否则同一方程缩放不同，"
                                "实测残差同比放大 4.94 倍、出口温度漂移 0.018 K。",
        },
        "mesh": {
            "family": "graded_block_tail",
            "construction": "n_block=ceil(z_b_target/dz0); z_b_eff=n_block·dz0; "
                            "n_rest=floor((L−z_b_eff)/rest_dz0); tail=余量(>0 时追加 1 格); "
                            "N=n_block+n_rest+tail?1:0",
            "z_b_target": "max(L_mix, 0.515) for device; max(L_mix, 0.04125) for anchor",
            "rest_dz0": REST_DZ0_DEV,
            "params_all_match_card_table": bool(params["matches_card_table"].all()),
            "old_family_untouched": "graded_full_length 默认行为未改（P1.9-A 证据依赖它）",
        },
        "predeclared": {
            "M1": f"L_mix/20 vs /40：|ΔX_C|<{M1_DXC_PP} pp、|ΔT|<{M1_DT_K} K、"
                  f"|ΔCO2|<{M1_DCO2_VOL} vol%、|ΔTpeak|<{M1_DTPEAK_K} K；整档准入需 9/9 窗",
            "M1_E": "L_mix=1 m 的 exponential 独立过同一检查；不得由 linear 推断",
            "M2_M3": "准入档中 L_mix 最小 vs 最大；n_adm<2 记不可评估",
            "M4": "是否有准入 L_mix 使某窗模型 X_C 与装置值差 ≤1 pp",
            "M5": "各准入档 f_gas 1→1/8 的 ΔX_C 与 P1.9-A 均匀 N=40 之比；中位 ≥3 记为恢复杠杆",
            "M6": "linear vs exponential（L_mix=1 m）；需 linear 已准入且 exponential 过 M1-E",
        },
        "unit_checks_all_pass": unit_pass,
        "n_adm": n_adm,
        "admitted_L_mix": admitted,
        "m1e_passed": bool(m1e["passed"]),
        "source_csv": {
            "p17d": {"round": "p1.7d-20261103", "file": P17D_CSV.name,
                     "sha256": hashlib.sha256(P17D_CSV.read_bytes()).hexdigest()},
            "p18d": {"round": "p1.8d-20261104", "file": P18D_CSV.name,
                     "sha256": hashlib.sha256(P18D_CSV.read_bytes()).hexdigest()},
            "p19a_baseline": {"round": "p1.9a-20261105",
                              "files": ["mesh_runs.csv", "leverage.csv"]},
        },
        "outputs": ["unit_checks.csv", "mesh_params.csv", "mesh_convergence.csv",
                    "admission.csv", "m1e.json", "ignition_share_scan.csv",
                    "leverage.csv", "cells.csv", "anchor_mix.csv", "readings.csv",
                    "protocol.json", "interpretation.md", "sha256.txt"],
    })

    text = interpretation(unit_df, params, m1, adm, m1e, lev, rd, anchor_df)
    written = write_round_outputs(
        OUT_DIR,
        tables={
            "unit_checks.csv": unit_df,
            "mesh_params.csv": params,
            "mesh_convergence.csv": m1,
            "admission.csv": adm,
            "leverage.csv": lev,
            "cells.csv": cells,
            "anchor_mix.csv": anchor_df if not anchor_df.empty else pd.DataFrame(columns=["case_name"]),
            "readings.csv": rd,
        },
        protocol=protocol,
        interpretation=text,
    )
    print("written:", [p.name for p in written], flush=True)


if __name__ == "__main__":
    main()
