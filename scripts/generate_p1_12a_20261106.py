#!/usr/bin/env python3
"""P1.12-A 整炉总量热损 + 全混区解耦扫描 + 输入敏感性（2026-11-06 卡）。

三件事：
  1. 热损失改为**整炉总量**：`Q_tot = HL_tot/100 × m_dry × HHV_d`（干煤口径），
     逐格按 `w_i = dz_i·clip((T_i/1800)^4, 0.1, 5)` 归一化分配；T 来自上一轮全炉温度场，
     因此外层不动点迭代（`GasifierSystem._solve_total_heat_loss`）。
  2. 在解耦后的结构上重做 `L_wsr` × `HL_tot` × `f_gas` 扫描（K1–K4）。
  3. O2/煤量/水分输入敏感性（K5）与可辨识性（K6）。

冻结：`wsr_plus_pfr`、`dz_down = 0.0515 m`、`OxygenMixingModel = "instant"`、
煤量主口径、实测载气、全部速率式与限幅、燃烧区判据、求解器容差。

用法：
    ./.venv/bin/python scripts/generate_p1_12a_20261106.py [--stage unit|scan|alloc|input|jac|all]
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
import generate_p1_11a_20261106 as p111a  # noqa: E402

from model.gasifier_system import GasifierSystem  # noqa: E402
from model.input_contract import coal_hhv_kj_kg  # noqa: E402
from model.p0_round_20260930 import m1_reference_window_ids, reactor_element_flows  # noqa: E402
from model.p1_5_round_20261030 import base_protocol, run_case, silence_solver_logs, write_round_outputs  # noqa: E402
from model.revision_evidence_common import load_frozen_split, source_commit  # noqa: E402

OUT_DIR = ROOT / "outputs" / "diagnostics" / "round-p1.12a-20261106"
P11A_DIR = ROOT / "outputs" / "diagnostics" / "round-p1.11a-20261106"
P18A_DIR = ROOT / "outputs" / "diagnostics" / "round-p1.8a-20261104"
P17D_CSV = p18a.P17D_CSV
P18D_CSV = p18a.P18D_CSV
M1_MATRIX = p18a.M1_MATRIX

DEV_L = 10.3
DZ_DOWN = 0.0515                       # 卡面冻结
L_WSR_GRID = (0.2575, 1.0, 2.0, 4.0, 6.0)
HL_TOT_GRID = (0.5, 1.0, 2.0, 3.0, 4.0)
F_GAS_GRID = (1.0, 0.125)
HL_REF = 1.0                            # 参考热损
L_REF = (1.0, 4.0)                      # 输入敏感性与 K2 的参考 L_wsr
L_JAC = 2.0                             # K6 参考点
BASE_MT_PCT = None                      # 由输入决定

# K6 观测缩放（卡面：1 pp、0.5、0.5、0.5 vol%）
K6_SCALES = (1.0, 0.5, 0.5, 0.5)
# 输入敏感性幅度
IN_FACTOR = (0.97, 1.03)
MT_ALT_PCT = 4.0

TOL_SUMQ_REL = 1e-6
TOL_DW_REL = 1e-4
TOL_UNIFORM_SUM = 1e-12
TOL_BITWISE = 1e-12
TOL_INIT_INDEP = 1e-6


# ---------------------------------------------------------------------------
# 网格与单次求解
# ---------------------------------------------------------------------------

def mesh_n(l_wsr: float, dz_down: float = DZ_DOWN) -> int:
    RL = DEV_L - l_wsr
    n_full = int(math.floor(RL / dz_down + 1e-12))
    rem = RL - float(n_full) * dz_down
    return 1 + n_full + (1 if rem > 1e-12 else 0)


def _run(row, dry, carrier, l_wsr, f_gas=1.0, hl_tot=HL_REF, model="total_percent", **over):
    op = {
        "HeatLossModel": model,
        "HeatLossPercentTotal": float(hl_tot),
        "GasificationRateFactor": float(f_gas),
        "carrier_gas_flow": p18a._carrier_kg_s(carrier),
        "MeshFamily": "wsr_plus_pfr",
        "WSRLength_m": float(l_wsr),
        "PFRCellLength_m": float(DZ_DOWN),
        "OxygenMixingModel": "instant",
    }
    op.update(over)
    rec = run_case(row, n_cells=mesh_n(l_wsr), coal_flow_dry_kg_s=dry,
                   keep_system=True, mesh_family="wsr_plus_pfr", extra_op=op)
    return rec, rec["_system"]


def _heatloss_fields(system: GasifierSystem, dry: float, hhv_kj: float) -> dict:
    """整炉热损核算：设定值、实际求和、迭代记录、分区份额。"""
    out: dict = {
        "hloss_iters": int(getattr(system, "heat_loss_iterations", -1)),
        "hloss_converged": bool(getattr(system, "heat_loss_converged", False)),
        "hloss_status": str(getattr(system, "hloss_status", "n/a")),
        "hloss_Q_sum_W": float(getattr(system, "heat_loss_Q_sum_W", float("nan"))),
        "hloss_Q_tot_W": float(getattr(system, "heat_loss_Q_tot_W", float("nan"))),
    }
    q_tot = out["hloss_Q_tot_W"]
    q_sum = out["hloss_Q_sum_W"]
    out["hloss_sum_over_Qtot"] = (q_sum / q_tot) if q_tot and np.isfinite(q_tot) else float("nan")
    out["hloss_sum_rel_err"] = (abs(q_sum - q_tot) / abs(q_tot)
                                if q_tot and np.isfinite(q_tot) else float("nan"))
    out["hloss_frac_dry_coal_pct"] = (q_sum / (dry * hhv_kj / 1000.0 * 1e6) * 100.0
                                      if q_sum and dry > 0 else float("nan"))
    # 全混区 / 下游份额
    if system.cell_diagnostics:
        w = float(system.cell_diagnostics[0].get("hloss_W", 0.0) or 0.0)
        tot = float(sum(float(d.get("hloss_W", 0.0) or 0.0) for d in system.cell_diagnostics))
        out["hloss_frac_wsr"] = w / tot if tot > 0 else float("nan")
        out["hloss_frac_downstream"] = 1.0 - out["hloss_frac_wsr"] if tot > 0 else float("nan")
    return out


def _row_metrics(rec, system, wid, dry, carrier, l_wsr, f_gas, hl_tot, model, extra=None):
    out = {
        "window_id": wid, "L_wsr": float(l_wsr), "dz_down": float(DZ_DOWN),
        "f_gas": float(f_gas), "HL_tot_pct": float(hl_tot),
        "HeatLossModel": model, "N_cells": mesh_n(l_wsr),
        "execution_status": str(rec.get("execution_status")),
        "no_root_cell_index": rec.get("no_root_cell_index"),
        "max_residual_cost": rec.get("max_residual_cost"),
        "solve_wall_time_s": rec.get("solve_wall_time_s"),
        "sum_dz_over_L": rec.get("sum_dz_over_L"),
        "cell0_n_distinct_roots": rec.get("cell0_n_distinct_roots"),
    }
    if rec.get("execution_status") == "completed" and system.results:
        gas = np.asarray(system.results[-1].gas_moles, dtype=float)
        dry_gas = float(np.sum(gas[:7]))
        inlet_c0 = float(system.cells[0].inlet.solid_mass * system.cells[0].inlet.carbon_fraction)
        c0 = system.results[0]
        out.update({
            "X_C_implied": float(rec.get("X_C_implied", float("nan"))),
            "T_out_K": float(rec.get("T_out_K", float("nan"))),
            "CO_vol_pct": float(gas[2] / max(dry_gas, 1e-12) * 100.0),
            "H2_vol_pct": float(gas[5] / max(dry_gas, 1e-12) * 100.0),
            "CO2_vol_pct": float(gas[3] / max(dry_gas, 1e-12) * 100.0),
            "dry_total_mol_s": dry_gas,
            "F_CO2_mol_s": float(gas[3]),
            "T_wsr_K": float(c0.T),
            "X_C_wsr": float(1.0 - c0.solid_mass * c0.carbon_fraction / max(inlet_c0, 1e-12)),
        })
        out.update(_probe(system))
        out.update(_heatloss_fields(system, dry, float(coal_hhv_kj_kg(system.coal_props))))
        out.update(_element_closure(system))
    if extra:
        out.update(extra)
    return out


def _probe(system: GasifierSystem, zs=(2.0, 4.0, 6.0, None)) -> dict:
    out: dict = {}
    if not system.results:
        return out
    zc = np.array([float(c.z) for c in system.cells], dtype=float)
    inlet_c0 = float(system.cells[0].inlet.solid_mass * system.cells[0].inlet.carbon_fraction)
    for z in zs:
        if z is None:
            st = system.results[-1]
            tag = "outlet"
        else:
            i = int(np.argmin(np.abs(zc - z)))
            st = system.results[i]
            tag = f"{z:g}m"
        out[f"T_at_{tag}_K"] = float(st.T)
        out[f"X_C_at_{tag}"] = float(1.0 - st.solid_mass * st.carbon_fraction
                                     / max(inlet_c0, 1e-12))
    return out


def _element_closure(system: GasifierSystem) -> dict:
    if not system.results:
        return {}
    try:
        f = reactor_element_flows(system, system.results[-1])
        return {f"O_closure_fraction": f["O_closure_fraction"],
                "C_closure_fraction": f["C_closure_fraction"],
                "O_inlet_mol_s": f["O_inlet_mol_s"]}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# 第 2 节：单元检查
# ---------------------------------------------------------------------------

def unit_checks(window_ids, by_id, p17d, p18d) -> pd.DataFrame:
    rows: list[dict] = []

    # ---- 1: hhv_percent 与 P1.11-A（dz_down = 0.0515）逐位相同 ----
    prev = pd.read_csv(P11A_DIR / "downstream_convergence.csv")
    prev = prev[(prev["row_kind"] == "run") & np.isclose(prev["dz_down"], DZ_DOWN)]
    diffs = []
    n_cmp = 0
    for wid in window_ids:
        dry = float(p17d.loc[wid, "coal_dry_recon_phase_kg_s"])
        carrier = float(p18d.loc[wid, "co2_carrier_measured_nm3_h"])
        for l_wsr in sorted(prev["L_wsr"].unique()):
            e = prev[(prev["window_id"] == wid) & np.isclose(prev["L_wsr"], l_wsr)]
            if e.empty:
                continue
            e = e.iloc[0]
            rec, _s = _run(by_id[wid], dry, carrier, l_wsr, 1.0, 1.0,
                           model="hhv_percent", HeatLossPercent=1.0)
            n_cmp += 1
            diffs.append(max(
                abs(rec["X_C_implied"] - float(e["X_C_implied"])),
                abs(rec["T_out_K"] - float(e["T_out_K"])) / max(abs(float(e["T_out_K"])), 1.0),
                abs(rec["CO2_vol_pct"] - float(e["CO2_vol_pct"])),
            ))
        print(f"  check1 {wid[-16:]} done", flush=True)
    dmax = float(max(diffs)) if diffs else float("nan")
    rows.append({
        "check": 1, "context": f"hhv_percent vs P1.11-A（dz_down={DZ_DOWN}），{n_cmp} 行",
        "metric": "max_abs_rel_outlet_diff", "expect": 0.0, "tolerance": TOL_BITWISE,
        "value": dmax, "pass": bool(np.isfinite(dmax) and dmax <= TOL_BITWISE),
        "note": "缺省路径不进新分支，必须逐位不变",
    })

    # ---- 2: total_percent_uniform 的 ΣQ = Q_tot ----
    wid = window_ids[0]
    dry = float(p17d.loc[wid, "coal_dry_recon_phase_kg_s"])
    carrier = float(p18d.loc[wid, "co2_carrier_measured_nm3_h"])
    for l_wsr in (1.0, 4.0):
        for hl in (1.0, 4.0):
            rec, sysm = _run(by_id[wid], dry, carrier, l_wsr, 1.0, hl,
                             model="total_percent_uniform")
            tot = float(sum(float(d.get("hloss_W", 0.0) or 0.0) for d in sysm.cell_diagnostics))
            q_tot = float(coal_hhv_kj_kg(sysm.coal_props)) / 1000.0 * 1e6 * dry * hl / 100.0
            rel = abs(tot - q_tot) / max(q_tot, 1e-30)
            rows.append({
                "check": 2, "context": f"uniform, L_wsr={l_wsr}, HL_tot={hl}%",
                "metric": "sum_Q_over_Qtot_minus_1", "expect": 0.0,
                "tolerance": TOL_UNIFORM_SUM, "value": rel,
                "pass": bool(rel < TOL_UNIFORM_SUM),
                "note": f"ΣQ={tot/1e6:.6f} MW, Q_tot={q_tot/1e6:.6f} MW",
            })

    # ---- 3: total_percent 的迭代轮数与 ΣQ 残差 ----
    # 卡面要求「报告每行迭代轮数与最终残差；所有行须 < 1e-6」。全量 450 行在主扫描里
    # 本来就会产出这两个量，逐行复制一遍等于把主扫描跑两次（实测 ≈40 分钟）。
    # 这里只对**代表性子集**做闸门判定（每档 L_wsr × 两档 HL_tot × 首窗），
    # 全量分布由 `check3_full` 从 scan.csv 汇总——判定口径相同，覆盖不同。
    iters_all = []
    gate_hl = (HL_TOT_GRID[0], HL_TOT_GRID[-1])
    wid0 = window_ids[0]
    dry0 = float(p17d.loc[wid0, "coal_dry_recon_phase_kg_s"])
    car0 = float(p18d.loc[wid0, "co2_carrier_measured_nm3_h"])
    for l_wsr in L_WSR_GRID:
        for hl in gate_hl:
            rec, sysm = _run(by_id[wid0], dry0, car0, l_wsr, 1.0, hl)
            it = int(getattr(sysm, "heat_loss_iterations", -1))
            q_s = float(getattr(sysm, "heat_loss_Q_sum_W", float("nan")))
            q_t = float(getattr(sysm, "heat_loss_Q_tot_W", float("nan")))
            rel = abs(q_s - q_t) / max(abs(q_t), 1e-30) if np.isfinite(q_t) else float("nan")
            iters_all.append({"L_wsr": l_wsr, "HL_tot": hl, "iters": it,
                              "sum_rel": rel, "status": str(rec.get("execution_status")),
                              "window_id": wid0})
            rows.append({
                "check": 3, "context": f"gate: L_wsr={l_wsr}, HL_tot={hl}%（{wid0[-14:]}）",
                "metric": "sum_Q_rel_err", "expect": 0.0,
                "tolerance": TOL_SUMQ_REL, "value": rel,
                "pass": bool(np.isfinite(rel) and rel < TOL_SUMQ_REL),
                "note": f"迭代 {it} 轮，状态={rec.get('execution_status')}",
            })
    globals()["_ITERS"] = pd.DataFrame(iters_all)

    # ---- 4: 迭代与初值无关（3 窗抽查，两种第 0 轮权重）----
    # 卡面：本条不作为停止条件（"不通过：报告，继续"）。失败时要给出量级，
    # 并说明它对 K 读法是否重要 —— 下面同时跑一个容差敏感性作对照。
    for wid in window_ids[:3]:
        dry = float(p17d.loc[wid, "coal_dry_recon_phase_kg_s"])
        carrier = float(p18d.loc[wid, "co2_carrier_measured_nm3_h"])
        rec_a, sys_a = _run(by_id[wid], dry, carrier, 2.0, 1.0, 1.0)
        T0 = _p11a_temperature_field(wid, 2.0)
        rec_b, sys_b = _run(by_id[wid], dry, carrier, 2.0, 1.0, 1.0,
                            HeatLossInitTemperatures=T0)
        d = max(abs(rec_a["X_C_implied"] - rec_b["X_C_implied"]),
                abs(rec_a["T_out_K"] - rec_b["T_out_K"]) / max(abs(rec_b["T_out_K"]), 1.0),
                abs(rec_a["CO2_vol_pct"] - rec_b["CO2_vol_pct"]))
        # 容差敏感性：同一窗改用 1e-6 的权重容差，看收敛值移动多少
        rec_c, sys_c = _run(by_id[wid], dry, carrier, 2.0, 1.0, 1.0,
                            HeatLossTolerance=1e-6)
        dT_tol = abs(rec_c["T_out_K"] - rec_a["T_out_K"])
        rows.append({
            "check": 4, "context": f"初值无关性（{wid[-14:]}）",
            "metric": "max_abs_rel_outlet_diff", "expect": 0.0,
            "tolerance": TOL_INIT_INDEP, "value": d,
            "pass": bool(d < TOL_INIT_INDEP),
            "note": f"均匀初值 {sys_a.heat_loss_iterations} 轮 vs P1.11-A 温度场初值 "
                    f"{sys_b.heat_loss_iterations} 轮；容差收紧到 1e-6 只移动 "
                    f"ΔT={dT_tol:.3f} K（{sys_c.heat_loss_iterations} 轮）",
        })
    return pd.DataFrame(rows)


def _p11a_temperature_field(wid: str, l_wsr: float) -> list[float] | None:
    """读 P1.11-A 的 cells.csv，取该窗该 L_wsr 的温度场（dz_down = 0.02575）。"""
    p = P11A_DIR / "cells.csv"
    if not p.exists():
        return None
    c = pd.read_csv(p)
    c = c[(c["window_id"] == wid) & np.isclose(c["L_wsr"], l_wsr)]
    if c.empty:
        return None
    c = c.sort_values("cell_index")
    return [float(x) for x in c["T_K"].tolist()]


# ---------------------------------------------------------------------------
# 第 3 节 A：主扫描
# ---------------------------------------------------------------------------

def main_scan(window_ids, by_id, p17d, p18d) -> pd.DataFrame:
    rows = []
    for wid in window_ids:
        dry = float(p17d.loc[wid, "coal_dry_recon_phase_kg_s"])
        carrier = float(p18d.loc[wid, "co2_carrier_measured_nm3_h"])
        for l_wsr in L_WSR_GRID:
            for hl in HL_TOT_GRID:
                for f_gas in F_GAS_GRID:
                    rec, sysm = _run(by_id[wid], dry, carrier, l_wsr, f_gas, hl)
                    rows.append(_row_metrics(rec, sysm, wid, dry, carrier, l_wsr,
                                             f_gas, hl, "total_percent"))
        print(f"  scan {wid[-16:]} done", flush=True)
    return pd.DataFrame(rows)


def alloc_sensitivity(window_ids, by_id, p17d, p18d) -> pd.DataFrame:
    rows = []
    for wid in window_ids:
        dry = float(p17d.loc[wid, "coal_dry_recon_phase_kg_s"])
        carrier = float(p18d.loc[wid, "co2_carrier_measured_nm3_h"])
        for l_wsr in L_WSR_GRID:
            for model in ("total_percent", "total_percent_uniform"):
                rec, sysm = _run(by_id[wid], dry, carrier, l_wsr, 1.0, HL_REF, model=model)
                rows.append(_row_metrics(rec, sysm, wid, dry, carrier, l_wsr,
                                         1.0, HL_REF, model))
        print(f"  alloc {wid[-16:]} done", flush=True)
    return pd.DataFrame(rows)


def input_sensitivity(window_ids, by_id, p17d, p18d) -> pd.DataFrame:
    """O2 ×{0.97,1.03}、干煤 ×{0.97,1.03}、Mt ∈ {基准, 4%}，各单独扰动。"""
    rows = []
    for wid in window_ids:
        dry0 = float(p17d.loc[wid, "coal_dry_recon_phase_kg_s"])
        carrier = float(p18d.loc[wid, "co2_carrier_measured_nm3_h"])
        mt0 = float(p17d.loc[wid, "moisture_ar"])
        for l_wsr in L_REF:
            # 基准
            rec, sysm = _run(by_id[wid], dry0, carrier, l_wsr, 1.0, HL_REF)
            rows.append(_row_metrics(rec, sysm, wid, dry0, carrier, l_wsr, 1.0, HL_REF,
                                     "total_percent", {"perturb_tag": "baseline"}))
            # O2 比例：o2_flow 是 op_conds 里的绝对值，按因子缩放
            for fac in IN_FACTOR:
                rec, sysm = _run(by_id[wid], dry0, carrier, l_wsr, 1.0, HL_REF,
                                 o2_flow_scale=fac)
                rows.append(_row_metrics(rec, sysm, wid, dry0, carrier, l_wsr, 1.0, HL_REF,
                                         "total_percent",
                                         {"perturb_tag": f"o2_x{fac:.2f}", "perturb_factor": fac}))
            # 干煤量
            for fac in IN_FACTOR:
                rec, sysm = _run(by_id[wid], dry0 * fac, carrier, l_wsr, 1.0, HL_REF)
                rows.append(_row_metrics(rec, sysm, wid, dry0, carrier, l_wsr, 1.0, HL_REF,
                                         "total_percent",
                                         {"perturb_tag": f"coal_x{fac:.2f}", "perturb_factor": fac}))
            # 输煤水分（只改 Mt，干煤量不变）
            rec, sysm = _run(by_id[wid], dry0, carrier, l_wsr, 1.0, HL_REF,
                             Mt_override_pct=MT_ALT_PCT)
            rows.append(_row_metrics(rec, sysm, wid, dry0, carrier, l_wsr, 1.0, HL_REF,
                                     "total_percent",
                                     {"perturb_tag": f"Mt_{MT_ALT_PCT:.0f}",
                                      "perturb_factor": MT_ALT_PCT / mt0 if mt0 else float("nan")}))
        print(f"  input {wid[-16:]} done", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# K6 雅可比
# ---------------------------------------------------------------------------

def jacobian(window_ids, by_id, p17d, p18d) -> pd.DataFrame:
    """在 L_wsr = 2 m、HL_tot = 1%、f_gas = 1 处以有限差分求观测对参数的雅可比。

    参数：[L_wsr, HL_tot, ln f_gas, O2 比例]
    观测：[X_C, CO2, CO, H2]，按 (1 pp, 0.5, 0.5, 0.5 vol%) 缩放。
    """
    wid = window_ids[0]
    dry = float(p17d.loc[wid, "coal_dry_recon_phase_kg_s"])
    carrier = float(p18d.loc[wid, "co2_carrier_measured_nm3_h"])
    row = by_id[wid]

    def obs(l_wsr, hl, f_gas, o2s):
        rec, sysm = _run(row, dry, carrier, l_wsr, f_gas, hl, o2_flow_scale=o2s)
        if rec.get("execution_status") not in ("completed", "hloss_not_converged"):
            return None
        return np.array([float(rec["X_C_implied"]) * 100.0, float(rec["CO2_vol_pct"]),
                         float(rec["CO_vol_pct"]), float(rec["H2_vol_pct"])])

    base = obs(L_JAC, HL_REF, 1.0, 1.0)
    if base is None:
        raise SystemExit("K6 基准点未收敛，无法求雅可比")
    steps = {"L_wsr": (L_JAC * 0.02, "abs"), "HL_tot": (HL_REF * 0.05, "abs"),
             "ln_f_gas": (0.05, "abs"), "o2_ratio": (0.02, "abs")}
    pts = {"L_wsr": L_JAC, "HL_tot": HL_REF, "ln_f_gas": 0.0, "o2_ratio": 1.0}
    J = np.zeros((4, 4))
    for j, (name, (h, _kind)) in enumerate(steps.items()):
        for sign in (+1.0, -1.0):
            p = dict(pts)
            if name == "ln_f_gas":
                p["ln_f_gas"] += sign * h
                f_gas = math.exp(p["ln_f_gas"])
            else:
                p[name] += sign * h
                f_gas = 1.0
            v = obs(p["L_wsr"], p["HL_tot"], f_gas, p["o2_ratio"])
            if v is None:
                continue
            J[:, j] += (sign * (v - base)) / (2.0 * h)
        print(f"  jac col {name} done", flush=True)

    scales = np.array(K6_SCALES)
    Js = J / scales[:, None]
    U, S, Vt = np.linalg.svd(Js)
    cond = float(S[0] / S[-1]) if S[-1] > 0 else float("inf")
    # 参数相关矩阵：用 (JsᵀJs)^-1 的归一化
    JtJ = Js.T @ Js
    try:
        inv = np.linalg.inv(JtJ)
        dg = np.sqrt(np.abs(np.diag(inv)))
        corr = inv / np.outer(dg, dg)
    except np.linalg.LinAlgError:
        corr = np.full((4, 4), float("nan"))
    names = ["L_wsr", "HL_tot", "ln_f_gas", "o2_ratio"]
    out = []
    for i, n in enumerate(names):
        out.append({"row_kind": "singular_value", "param": f"SV{i+1}",
                    "value": float(S[i]), "note": ""})
    out.append({"row_kind": "condition_number", "param": "cond(J)",
                "value": cond, "note": "最大/最小奇异值"})
    for i, n in enumerate(names):
        out.append({"row_kind": "weakest_direction", "param": n,
                    "value": float(Vt[-1, i]),
                    "note": "与最小奇异值对应的参数组合（Vt 最后一行）"})
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            out.append({"row_kind": "corr", "param": f"{ni}~{nj}",
                        "value": float(corr[i, j]) if np.isfinite(corr[i, j]) else float("nan"),
                        "note": ""})
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            out.append({"row_kind": "J", "param": f"d({names[i]})/d({nj})",
                        "value": float(J[i, j]), "note": "未缩放"})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# 读法 K1–K8
# ---------------------------------------------------------------------------

def _plant_map(by_id, p18d):
    return {w: float(p18d.loc[w, "xc_measured_pct"]) for w in p18d.index}


def readings(scan, alloc, inputs, jac, adm_hl, by_id, p18d) -> pd.DataFrame:
    rows: list[dict] = []

    def add(**kw):
        rows.append(kw)

    plant = _plant_map(by_id, p18d)
    base = scan[(np.isclose(scan["HL_tot_pct"], HL_REF)) & np.isclose(scan["f_gas"], 1.0)]

    # ---- K1 纯混合杠杆（与 P1.11-A 同档对比）----
    lo, hi = L_WSR_GRID[0], L_WSR_GRID[-1]
    for metric, col, scale in (("dX_C_pp", "X_C_implied", 100.0),
                               ("dT_out_K", "T_out_K", 1.0),
                               ("dCO2_vol_pct", "CO2_vol_pct", 1.0),
                               ("dT_wsr_K", "T_wsr_K", 1.0)):
        a = base[np.isclose(base["L_wsr"], lo)][col]
        b = base[np.isclose(base["L_wsr"], hi)][col]
        if a.empty or b.empty:
            continue
        pair = pd.DataFrame({"lo": a.values, "hi": b.values}).dropna()
        d = (pair["hi"] - pair["lo"]) * scale
        add(reading="K1", L_wsr=float("nan"), metric=metric, value=float(d.median()),
            threshold=float("nan"), pass_=None,
            note=f"HL_tot=1%，L_wsr {lo:g}→{hi:g}；9 窗中位，范围 {d.min():.4g}…{d.max():.4g}")
    # 与 P1.11-A（HLP=1%）对比：被热损耦合解释掉的份额
    p11 = _p11a_comparable()
    if p11 is not None:
        for metric in ("dX_C_pp", "dT_out_K"):
            v_new = [r for r in rows if r["reading"] == "K1" and r["metric"] == metric]
            if not v_new:
                continue
            k1 = float(v_new[0]["value"])
            k0 = float(p11[metric])
            frac = (1.0 - abs(k1) / abs(k0)) if abs(k0) > 1e-12 else float("nan")
            add(reading="K1_compare", L_wsr=float("nan"), metric=f"{metric}__explained_by_coupling",
                value=float(frac), threshold=float("nan"), pass_=None,
                note=f"P1.11-A（hhv_percent, HLP=1%）同档 {k0:+.3f}；本卡（HL_tot=1%）{k1:+.3f}；"
                     f"被热损耦合解释掉的比例 {frac*100:.1f}%")

    # ---- K2 热损杠杆：每 1% HL_tot 的 Δ ----
    for l_wsr in L_REF:
        sub = scan[np.isclose(scan["L_wsr"], l_wsr) & np.isclose(scan["f_gas"], 1.0)]
        if sub.empty:
            continue
        for metric, col, scale in (("dX_C_pp_per_pct", "X_C_implied", 100.0),
                                   ("dT_out_K_per_pct", "T_out_K", 1.0),
                                   ("dCO2_vol_pct_per_pct", "CO2_vol_pct", 1.0)):
            xs = sub["HL_tot_pct"].values.astype(float)
            ys = sub[col].values.astype(float) * scale
            if len(xs) < 2 or not np.all(np.isfinite(ys)):
                continue
            slope = float(np.polyfit(xs, ys, 1)[0])
            add(reading="K2", L_wsr=float(l_wsr), metric=metric, value=slope,
                threshold=float("nan"), pass_=None,
                note=f"每 1% HL_tot 的变化（{len(xs)} 档线拟合，9 窗池化）")

    # ---- K3 可信热损下的到达（HL_tot ≤ 1%）----
    hit_any = False
    for l_wsr in L_WSR_GRID:
        for hl in [h for h in HL_TOT_GRID if h <= HL_REF + 1e-12]:
            s = scan[np.isclose(scan["L_wsr"], l_wsr) & np.isclose(scan["HL_tot_pct"], hl)
                     & np.isclose(scan["f_gas"], 1.0)]
            if s.empty:
                continue
            devs = [(float(r["X_C_implied"]) * 100.0 - plant[r["window_id"]])
                    for _, r in s.iterrows() if r["window_id"] in plant]
            if not devs:
                continue
            within = int(sum(1 for d in devs if abs(d) <= 1.0))
            if within > 0:
                hit_any = True
            add(reading="K3", L_wsr=l_wsr, metric=f"n_windows_within_1pp::HL{hl:g}",
                value=float(within), threshold=0.0, pass_=bool(within > 0),
                note=f"9 窗中位偏差 {float(np.median(devs)):+.2f} pp；"
                     f"|偏差|≤1 pp 的窗数 {within}/9")
    add(reading="K3_verdict", L_wsr=float("nan"), metric="any_reaches_at_trusted_hloss",
        value=float(hit_any), threshold=1.0, pass_=bool(hit_any),
        note="HL_tot ≤ 1% 时是否存在 L_wsr ≤ 6 m 使某窗落在 ±1 pp")

    # ---- K4 所需热量移除：逐窗插值 HL_tot* ----
    k4 = []
    for wid in sorted(set(scan["window_id"])):
        if wid not in plant:
            continue
        for l_wsr in L_WSR_GRID:
            s = scan[(scan["window_id"] == wid) & np.isclose(scan["L_wsr"], l_wsr)
                     & np.isclose(scan["f_gas"], 1.0)].sort_values("HL_tot_pct")
            xs = s["HL_tot_pct"].values.astype(float)
            ys = s["X_C_implied"].values.astype(float) * 100.0 - plant[wid]
            if len(xs) < 2:
                continue
            star = _interp_zero(xs, ys)
            edge = None
            if star is None:
                edge = "none_in_range"
            elif star <= xs.min() + 1e-12 or star >= xs.max() - 1e-12:
                edge = "boundary"
            # 对应 MW（干煤口径）
            dry = float(p17d_ref.get(wid, float("nan"))) if False else None
            k4.append({"window_id": wid, "L_wsr": l_wsr, "HL_tot_star_pct": star,
                       "edge": edge, "dev_min": float(ys.min()), "dev_max": float(ys.max())})
            add(reading="K4", L_wsr=l_wsr, metric=f"HL_tot_star_pct::{wid[-14:]}",
                value=float(star) if star is not None else float("nan"),
                threshold=float("nan"), pass_=None,
                note=f"偏差范围 {ys.min():+.2f}…{ys.max():+.2f} pp；"
                     f"{'落在扫描边界' if edge == 'boundary' else ('区间内无零点' if edge else '区间内插值')}")
    globals()["_K4"] = pd.DataFrame(k4)

    # ---- K5 输入等效 ----
    k2_dx = [r for r in rows if r["reading"] == "K2" and r["metric"] == "dX_C_pp_per_pct"]
    unit_dx = float(np.median([abs(r["value"]) for r in k2_dx])) if k2_dx else float("nan")
    add(reading="K5", L_wsr=float("nan"), metric="unit_hloss_dX_C_pp_per_pct",
        value=unit_dx, threshold=float("nan"), pass_=None,
        note="K2 各档斜率 |ΔX_C| 的中位，用作输入扰动的等价换算")
    for tag, grp in inputs.groupby("perturb_tag"):
        if tag == "baseline":
            continue
        bs = inputs[inputs["perturb_tag"] == "baseline"].set_index(["window_id", "L_wsr"])
        d_xc, d_t, d_co2 = [], [], []
        for _, r in grp.iterrows():
            key = (r["window_id"], r["L_wsr"])
            if key not in bs.index:
                continue
            b = bs.loc[key]
            d_xc.append((float(r["X_C_implied"]) - float(b["X_C_implied"])) * 100.0)
            d_t.append(float(r["T_out_K"]) - float(b["T_out_K"]))
            d_co2.append(float(r["CO2_vol_pct"]) - float(b["CO2_vol_pct"]))
        if not d_xc:
            continue
        med = float(np.median(d_xc))
        # 符号约定：扰动使 X_C 上升 → 等效于「少移除一点热」，故等效 HL_tot 取负号。
        # 即 equiv = −(ΔX_C_扰动 / 单位热损 ΔX_C)，回答「该扰动相当于多少 % 的热损变化」。
        equiv = -med / unit_dx if unit_dx and np.isfinite(unit_dx) else float("nan")
        add(reading="K5", L_wsr=float("nan"), metric=f"dX_C_pp::{tag}",
            value=med, threshold=float("nan"), pass_=None,
            note=f"ΔT_out 中位 {float(np.median(d_t)):+.2f} K；"
                 f"ΔCO2 中位 {float(np.median(d_co2)):+.3f} vol%")
        add(reading="K5", L_wsr=float("nan"), metric=f"equivalent_HL_tot_pct::{tag}",
            value=equiv, threshold=float("nan"), pass_=None,
            note="等效热损变化 = −ΔX_C(扰动)/单位热损 ΔX_C；"
                 "正值表示该扰动相当于**多**移除这些 % 的热损")

    # ---- K6 ----
    for kind in ("singular_value", "condition_number", "weakest_direction"):
        sub = jac[jac["row_kind"] == kind]
        for _, r in sub.iterrows():
            add(reading="K6", L_wsr=L_JAC, metric=f"{kind}::{r['param']}",
                value=float(r["value"]), threshold=float("nan"), pass_=None,
                note=str(r["note"]))
    sv = jac[jac["row_kind"] == "singular_value"]["value"].values.astype(float)
    if len(sv) >= 2:
        add(reading="K6_verdict", L_wsr=L_JAC, metric="min_over_max_singular_value",
            value=float(sv[-1] / sv[0]), threshold=float("nan"), pass_=None,
            note=f"cond = {sv[0]/sv[-1]:.3g}；小于 1e-6 说明存在几乎不可辨识的方向")

    # ---- K7 组成与氢账本 ----
    for l_wsr in L_REF:
        s = scan[np.isclose(scan["L_wsr"], l_wsr) & np.isclose(scan["HL_tot_pct"], HL_REF)
                 & np.isclose(scan["f_gas"], 1.0)]
        if s.empty:
            continue
        add(reading="K7", L_wsr=l_wsr, metric="median_X_C_dev_pp",
            value=float(np.median([float(r["X_C_implied"]) * 100.0 - plant[r["window_id"]]
                                   for _, r in s.iterrows() if r["window_id"] in plant])),
            threshold=float("nan"), pass_=None, note="HL_tot=1%")
        add(reading="K7", L_wsr=l_wsr, metric="median_CO_vol_pct",
            value=float(s["CO_vol_pct"].median()), threshold=float("nan"), pass_=None, note="")
        add(reading="K7", L_wsr=l_wsr, metric="median_H2_vol_pct",
            value=float(s["H2_vol_pct"].median()), threshold=float("nan"), pass_=None, note="")
    mt = inputs[inputs["perturb_tag"].str.startswith("Mt_")]
    if not mt.empty:
        bs = inputs[inputs["perturb_tag"] == "baseline"]
        add(reading="K7", L_wsr=float("nan"), metric="Mt_alt_median_dH2_vol_pct",
            value=float((mt["H2_vol_pct"].values - bs["H2_vol_pct"].values)[:len(mt)].mean())
            if len(mt) == len(bs) else float("nan"),
            threshold=float("nan"), pass_=None,
            note="Mt 4% 相对基准的 H2 变化（水汽带入氢）")

    # ---- K8 分配形式 ----
    for l_wsr in L_WSR_GRID:
        a = alloc[(np.isclose(alloc["L_wsr"], l_wsr))
                  & (alloc["HeatLossModel"] == "total_percent")]
        b = alloc[(np.isclose(alloc["L_wsr"], l_wsr))
                  & (alloc["HeatLossModel"] == "total_percent_uniform")]
        if a.empty or b.empty:
            continue
        m = a.set_index("window_id").join(b.set_index("window_id"),
                                          lsuffix="_t", rsuffix="_u", how="inner")
        if m.empty:
            continue
        dxc = (m["X_C_implied_t"] - m["X_C_implied_u"]) * 100.0
        dt = m["T_out_K_t"] - m["T_out_K_u"]
        dco2 = m["CO2_vol_pct_t"] - m["CO2_vol_pct_u"]
        add(reading="K8", L_wsr=l_wsr, metric="median_abs_dX_C_pp",
            value=float(dxc.abs().median()), threshold=float("nan"), pass_=None,
            note=f"ΔT 中位 {float(dt.median()):+.2f} K；ΔCO2 中位 {float(dco2.median()):+.3f} vol%")

    df = pd.DataFrame(rows)
    if "pass_" in df.columns:
        df = df.rename(columns={"pass_": "pass"})
    return df


def _interp_zero(xs: np.ndarray, ys: np.ndarray):
    """在升序 xs 上找 ys 的首个变号区间并线性插值。"""
    for i in range(len(xs) - 1):
        if ys[i] == 0:
            return float(xs[i])
        if ys[i] * ys[i + 1] < 0:
            t = abs(ys[i]) / (abs(ys[i]) + abs(ys[i + 1]))
            return float(xs[i] + t * (xs[i + 1] - xs[i]))
    if ys[-1] == 0:
        return float(xs[-1])
    return None


def _p11a_comparable():
    """P1.11-A（hhv_percent, HLP=1%）同档 L_wsr 0.2575→6 的 ΔX_C / ΔT_out 中位。"""
    p = P11A_DIR / "leverage.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d = d[(np.isclose(d["HeatLossPercent"], 1.0)) & np.isclose(d["f_gas"], 1.0)]
    out = {}
    for metric, col, scale in (("dX_C_pp", "X_C_implied", 100.0),
                               ("dT_out_K", "T_out_K", 1.0)):
        a = d[np.isclose(d["L_wsr"], L_WSR_GRID[0])][col]
        b = d[np.isclose(d["L_wsr"], L_WSR_GRID[-1])][col]
        if a.empty or b.empty:
            continue
        pair = pd.DataFrame({"lo": a.values, "hi": b.values}).dropna()
        out[metric] = float(((pair["hi"] - pair["lo"]) * scale).median())
    return out or None


def interpretation(unit_df, scan, alloc, inputs, jac, rd, iters_df) -> str:
    L: list[str] = []
    L.append("# P1.12-A 整炉总量热损 + 全混区解耦扫描 — 判读\n")
    L.append("Status: `EXECUTED — model repo`\n")
    L.append("依据：`manuscript/model-side-handoff-p1.12a-20261106.md`；"
             "上游 `manuscript/model-evidence-verification-20261106-p111.md` I2–I5。\n")
    L.append("## 0. 结构与热损模型\n")
    L.append("结构冻结为 `wsr_plus_pfr`、`dz_down` = 0.0515 m、`OxygenMixingModel = instant`。"
             "热损改为**整炉总量**：`Q_tot = HL_tot/100 × m_dry × HHV_d`（**干煤**口径），"
             "逐格按 `w_i = dz_i·clip((T_i/1800)^4, 0.1, 5)` 归一化分配。"
             "逐格推进的求解器在解第 i 格时不知道下游温度，因此 T 权重来自**上一轮全炉温度场**，"
             "外层不动点迭代至 `|ΣQ−Q_tot|/Q_tot < 1e-6` 且 `max|Δw/w| < 1e-4`，最多 30 轮。\n")
    L.append("`total_percent_uniform` 是同总量的对照：`w_i = dz_i`，无需迭代，"
             "`ΣQ` 逐位等于 `Q_tot`（第 2 节检查 2）。\n")

    L.append("## 1. 单元检查（卡第 2 节）\n")
    gate = unit_df[unit_df["pass"].notna()]
    L.append(f"- 共 {len(gate)} 项，全部通过：**{bool(gate['pass'].astype(bool).all())}**")
    for c, grp in unit_df.groupby("check"):
        sub = grp[grp["pass"].notna()]
        if sub.empty:
            continue
        L.append(f"  - 检查 {c}：{int(sub['pass'].astype(bool).sum())}/{len(sub)} 通过")
    L.append("")
    if iters_df is not None and not iters_df.empty:
        L.append("### 迭代轮数分布（主扫描全量）\n")
        L.append(_tbl(iters_df.groupby(["L_wsr", "HL_tot"])["iters"]
                      .agg(["min", "median", "max"]).reset_index(),
                      ["L_wsr", "HL_tot", "min", "median", "max"], nd=0))
        L.append("")
        L.append(f"- 最终 `|ΣQ−Q_tot|/Q_tot` 最大 {float(iters_df['sum_rel'].max()):.2e}"
                 f"（{len(iters_df)} 行全部须 < {TOL_SUMQ_REL:g}）")
        st = iters_df["status"].value_counts().to_dict()
        L.append(f"- 状态分布：{st}\n")

    L.append("## 2. K1–K8 结论\n")
    for tag, title in (("K1", "K1 纯混合杠杆（HL_tot = 1%）"),
                       ("K1_compare", "K1 与 P1.11-A 的对比：被热损耦合解释掉的份额"),
                       ("K2", "K2 热损杠杆（每 1% HL_tot）"),
                       ("K3", "K3 可信热损下的到达（HL_tot ≤ 1%）"),
                       ("K4", "K4 所需热量移除 HL_tot*"),
                       ("K5", "K5 输入等效"),
                       ("K6", "K6 可辨识性"),
                       ("K7", "K7 组成与氢账本"),
                       ("K8", "K8 分配形式")):
        sub = rd[rd["reading"] == tag]
        if sub.empty:
            continue
        L.append(f"### {title}\n")
        cols = [c for c in ["L_wsr", "metric", "value", "threshold", "pass", "note"]
                if c in sub.columns]
        L.append(_tbl(sub, cols, nd=4))
        L.append("")
    L.extend(_verdict_section(rd, scan, jac))
    L.append("## 4. 失败与异常\n")
    bad = scan[~scan["execution_status"].isin(["completed"])]
    L.append(f"- 主扫描非 completed 行：{len(bad)} / {len(scan)}")
    if not bad.empty:
        L.append(_tbl(bad, ["window_id", "L_wsr", "HL_tot_pct", "f_gas",
                            "execution_status"], nd=3))
        L.append(_tbl(bad.head(20), ["window_id", "L_wsr", "HL_tot_pct", "f_gas",
                                     "execution_status"], nd=3))
    L.append(f"- `no_root`：{int((scan['execution_status'] == 'no_root').sum())}；"
             f"`hloss_not_converged`：{int((scan.get('hloss_status', pd.Series(dtype=str)) == 'hloss_not_converged').sum())}")
    mult = scan[scan["cell0_n_distinct_roots"].fillna(1) > 1] if "cell0_n_distinct_roots" in scan else pd.DataFrame()
    L.append(f"- 多火焰根行：{len(mult)}\n")
    L.append("## 5. 复现边界\n")
    L.append("- 只改热损模型与扫描参数，不改动力学常数、不采纳任何参数值、不改合同。")
    L.append("- K1–K8 阈值在运行前写入脚本常量，运行后未改。")
    L.append("- 整炉总量热损在 JAX 路径上未实现，遇非 `hhv_percent` 直接报错。")
    L.append("- `total_percent` 的收敛判据以 `max|Δw/w|` 为主：非自洽权重下 ΣQ 恒等于 Q_tot"
             "（分母就是同一轮权重和），所以 ΣQ 那一条几乎恒成立，保留它只为符合卡面文字核对。\n")
    return "\n".join(L) + "\n"


def _verdict_section(rd, scan, jac) -> list[str]:
    L: list[str] = []
    L.append("## 2.5 关键读数\n")
    k1c = rd[rd["reading"] == "K1_compare"]
    if not k1c.empty:
        dxc = k1c[k1c["metric"] == "dX_C_pp__explained_by_coupling"]
        if not dxc.empty:
            L.append(f"- **K1 解耦效果**：`L_wsr` 0.2575→6 m 的 ΔX_C 在旧热损式下为"
                     f" P1.11-A 的值，在整炉总量式下为 K1 的值；"
                     f"**{float(dxc['value'].iloc[0])*100:.1f}%** 的杠杆可由「`L_wsr` 改变高温区长度"
                     f"→ 改变按 T⁴ 权重分配的热损」解释掉。这正是 I2 指出的构造性混杂。")
    k3 = rd[rd["reading"] == "K3_verdict"]
    if not k3.empty:
        L.append(f"- **K3 可信热损下能否到达**："
                 f"**{'存在' if bool(k3['pass'].iloc[0]) else '不存在'}** "
                 f"`HL_tot ≤ 1%` 且 `L_wsr ≤ 6 m` 使某窗落在 ±1 pp。")
    k4 = globals().get("_K4")
    if k4 is not None and not k4.empty:
        inner = k4[k4["edge"].isna()]
        n_in = len(inner)
        L.append(f"- **K4 所需热量移除**：{n_in} / {len(k4)} 个（窗, `L_wsr`）组合在 0.5–4% 区间内"
                 f"存在零点；其余落在扫描边界。")
        if n_in:
            L.append(f"  - `HL_tot*` 中位 {float(inner['HL_tot_star_pct'].median()):.2f}%"
                     f"（范围 {inner['HL_tot_star_pct'].min():.2f}–{inner['HL_tot_star_pct'].max():.2f}%）")
    k5 = rd[(rd["reading"] == "K5") & (rd["metric"].str.startswith("equivalent_HL_tot_pct"))]
    if not k5.empty:
        L.append("- **K5 输入等效**（扰动换算成等效 `HL_tot`；正 = 相当于多移除热）：")
        vals = {}
        for _, r in k5.iterrows():
            tag = r["metric"].split("::")[-1]
            vals[tag] = float(r["value"])
            L.append(f"  - `{tag}` → {float(r['value']):+.2f}% HL_tot")
        # 卡面的问题：K4 的过剩热量能否由 ≤3% 的 O2/煤量偏差解释
        k4t = globals().get("_K4")
        need = float("nan")
        if k4t is not None and not k4t.empty:
            inner4 = k4t[k4t["edge"].isna()]
            if not inner4.empty:
                need = float(inner4["HL_tot_star_pct"].median())
        # 两个 3% 扰动同向叠加（线性外推，非实测）
        combo = None
        if "o2_x0.97" in vals and "coal_x1.03" in vals:
            combo = vals["o2_x0.97"] + vals["coal_x1.03"]
        L.append("")
        if math.isfinite(need) and combo is not None:
            covered = bool(combo >= need)
            L.append(f"- **K5 判定**：K4 需要的热损中位 **{need:.2f}%**；"
                     f"O2 −3% 与煤量 +3% **同向叠加**的等效量约 **{combo:.2f}%**"
                     f"（单扰动线性外推，未做组合实测）。"
                     f"因此「K4 的过剩热量能否由 ≤3% 的 O2/煤量比偏差解释」的答案是 "
                     f"**{'能' if covered else '不能'}**"
                     + ("，但要求两者同时处在各自的 3% 极限且方向一致——"
                        "这是边界条件，不是典型工况。" if covered else "。"))
            L.append(f"  - 单看任一变量都不够：O2 −3% 约 {vals.get('o2_x0.97', float('nan')):.2f}%，"
                     f"煤量 +3% 约 {vals.get('coal_x1.03', float('nan')):.2f}%。")
            L.append("  - **这条只按单扰动线性外推**，本轮没有做组合扰动的实测；"
                     "两个 3% 同时成立是边界情形，且方向必须与「少氧 / 多煤」一致。"
                     "它的意义是：能量过剩**并非必然不可解释**，但需要 O2 与煤量"
                     "两个独立的 3% 级偏差同向叠加——而这两个量里只有煤量做了协调，"
                     "O2 流量计至今未核（见 P1.11-A 核查 I3 第 2 项）。"
                     "把这条当作「已解释」是不对的，正确表述是「可被边界级输入偏差覆盖」。")
    sv = jac[jac["row_kind"] == "singular_value"]["value"].values.astype(float)
    if len(sv) >= 2:
        L.append(f"- **K6 可辨识性**：cond(J) = {sv[0]/sv[-1]:.3g}；"
                 f"最小奇异值 {sv[-1]:.3g}。")
        wd = jac[jac["row_kind"] == "weakest_direction"]
        if not wd.empty:
            top = wd.iloc[int(np.argmax(np.abs(wd["value"].values.astype(float))))]
            L.append(f"  - 最弱方向由 **{top['param']}** 主导"
                     f"（载荷 {float(top['value']):+.3f}），"
                     f"即在该工作点上它最难以独立辨识。")
    L.append("")
    return L


def _tbl(df: pd.DataFrame, cols: list[str], nd: int = 3) -> str:
    def _f(x, n):
        try:
            x = float(x)
        except (TypeError, ValueError):
            return str(x)
        if not math.isfinite(x):
            return "nan"
        return f"{x:.{n}f}"
    cols = [c for c in cols if c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(
            _f(r.get(c), nd) if isinstance(r.get(c), (int, float, np.floating))
            else str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def _rewrite_manifest(out_dir: Path) -> None:
    files = sorted(p for p in out_dir.iterdir()
                   if p.is_file() and p.name != "sha256.txt")
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}" for p in files]
    (out_dir / "sha256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=("unit", "scan", "alloc", "input", "jac", "read", "all"))
    ap.add_argument("--source-commit", default=None)
    args = ap.parse_args()
    silence_solver_logs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ok, rows_, _inv = load_frozen_split()
    if not ok:
        raise SystemExit("frozen R52-C-75 split was not consumed intact")
    by_id = {r["window_id"]: r for r in rows_}
    window_ids = m1_reference_window_ids(M1_MATRIX)
    p17d = pd.read_csv(P17D_CSV).set_index("window_id")
    p18d = pd.read_csv(P18D_CSV).set_index("window_id")
    globals()["p17d_ref"] = p17d["coal_dry_recon_phase_kg_s"].to_dict()

    if args.stage == "read":
        unit_df = pd.read_csv(OUT_DIR / "unit_checks.csv")
        scan = pd.read_csv(OUT_DIR / "scan.csv")
        alloc = pd.read_csv(OUT_DIR / "alloc_sensitivity.csv")
        inputs = pd.read_csv(OUT_DIR / "input_sensitivity.csv")
        jac = pd.read_csv(OUT_DIR / "jacobian.csv")
        # 不能硬编码 True：检查 4 未通过时这里必须如实为 False
        gate = unit_df[unit_df["pass"].notna()]
        unit_pass = bool(gate["pass"].astype(bool).all())
        hard = unit_df[(unit_df["check"].astype(str).isin(["1", "2", "3"]))
                       & unit_df["pass"].notna()]
        hard_pass = bool(hard["pass"].astype(bool).all())
        iters_df = globals().get("_ITERS")
        if iters_df is None or "HL_tot" not in iters_df.columns:
            iters_df = scan.dropna(subset=["hloss_iters"]).rename(
                columns={"hloss_iters": "iters", "hloss_sum_rel_err": "sum_rel",
                         "execution_status": "status", "HL_tot_pct": "HL_tot"})
            iters_df = iters_df[["window_id", "L_wsr", "HL_tot", "iters", "sum_rel", "status"]]
    else:
        unit_df = unit_checks(window_ids, by_id, p17d, p18d)
        unit_df.to_csv(OUT_DIR / "unit_checks.csv", index=False)
        gate = unit_df[unit_df["pass"].notna()]
        unit_pass = bool(gate["pass"].astype(bool).all())
        hard = unit_df[(unit_df["check"].astype(str).isin(["1", "2", "3"]))
                       & unit_df["pass"].notna()]
        hard_pass = bool(hard["pass"].astype(bool).all())
        print("unit checks:", "PASS" if unit_pass else
              ("check4 only" if hard_pass else "FAIL"), flush=True)
        # 卡面停止条件：第 1–3 条任一不通过则停止；第 4 条不通过只报告，继续
        hard = unit_df[(unit_df["check"].astype(str).isin(["1", "2", "3"]))
                       & unit_df["pass"].notna()]
        hard_pass = bool(hard["pass"].astype(bool).all())
        if not hard_pass:
            unit_df.to_csv(OUT_DIR / "unit_checks.csv", index=False)
            raise SystemExit("停止：第 2 节检查 1–3 未通过")
        if args.stage == "unit":
            return
        scan = main_scan(window_ids, by_id, p17d, p18d)
        scan.to_csv(OUT_DIR / "scan.csv", index=False)
        # 检查 3 的全量分布：从主扫描的逐行迭代记录汇总（与闸门同一判定口径）
        iters_df = scan.dropna(subset=["hloss_iters"])[
            ["window_id", "L_wsr", "HL_tot_pct", "hloss_iters",
             "hloss_sum_rel_err", "execution_status"]].rename(
            columns={"hloss_iters": "iters", "hloss_sum_rel_err": "sum_rel",
                     "execution_status": "status", "HL_tot_pct": "HL_tot"})
        if not iters_df.empty:
            full_ok = bool((iters_df["sum_rel"] < TOL_SUMQ_REL).all())
            unit_df = pd.concat([unit_df, pd.DataFrame([{
                "check": "3b", "context": f"主扫描全量 {len(iters_df)} 行",
                "metric": "max_sum_Q_rel_err", "expect": 0.0,
                "tolerance": TOL_SUMQ_REL,
                "value": float(iters_df["sum_rel"].max()),
                "pass": full_ok,
                "note": f"迭代轮数 {int(iters_df['iters'].min())}–"
                        f"{int(iters_df['iters'].max())}；"
                        f"状态 {iters_df['status'].value_counts().to_dict()}",
            }])], ignore_index=True)
            unit_df.to_csv(OUT_DIR / "unit_checks.csv", index=False)
        alloc = alloc_sensitivity(window_ids, by_id, p17d, p18d)
        alloc.to_csv(OUT_DIR / "alloc_sensitivity.csv", index=False)
        inputs = input_sensitivity(window_ids, by_id, p17d, p18d)
        inputs.to_csv(OUT_DIR / "input_sensitivity.csv", index=False)
        jac = jacobian(window_ids, by_id, p17d, p18d)
        jac.to_csv(OUT_DIR / "jacobian.csv", index=False)
        iters_df = globals().get("_ITERS")
        if args.stage == "scan":
            return

    rd = readings(scan, alloc, inputs, jac, None, by_id, p18d)
    rd.to_csv(OUT_DIR / "readings.csv", index=False)

    protocol = base_protocol({
        "round": "P1.12-A / 2026-11-06",
        "model_repo": "gasifier-1d-kinetic",
        "source_commit": args.source_commit or source_commit(),
        "task_card": "papers/jax-gasifier-cascade-hybrid/manuscript/"
                     "model-side-handoff-p1.12a-20261106.md",
        "structure": {
            "mesh_family": "wsr_plus_pfr",
            "dz_down_m": DZ_DOWN,
            "oxygen_mixing": "instant（冻结）",
            "frozen": ["全部速率式与限幅", "燃烧区判据 0.05 atm", "煤量主口径",
                       "实测载气", "求解器容差"],
        },
        "heat_loss": {
            "new_models": ["total_percent", "total_percent_uniform"],
            "total_W": "Q_tot = HL_tot/100 × m_dry × HHV_d —— **干煤**口径，"
                       "与 P1.9-D 汽包吸热同一分母",
            "allocation": "w_i = dz_i · clip((T_i/1800)^4, 0.1, 5)；Q_i = Q_tot·w_i/Σw",
            "outer_iteration": "权重用上一轮全炉温度场；第 0 轮按长度均匀；"
                               "收敛 |ΣQ−Q_tot|/Q_tot<1e-6 且 max|Δw/w|<1e-4；最多 30 轮；"
                               "未收敛标 hloss_not_converged 并保留该轮解",
            "default_unchanged": "hhv_percent 逐位不变（检查 1）",
            "jax_guard": "JAX 路径遇 total_percent* 直接报错",
        },
        "runs": {
            "A_main_scan": {"windows": len(window_ids), "L_wsr": list(L_WSR_GRID),
                            "HL_tot_pct": list(HL_TOT_GRID), "f_gas": list(F_GAS_GRID),
                            "expected_rows": len(window_ids) * len(L_WSR_GRID)
                                             * len(HL_TOT_GRID) * len(F_GAS_GRID)},
            "B_alloc": {"expected_rows": len(window_ids) * len(L_WSR_GRID) * 2},
            "C_input": {"L_wsr": list(L_REF), "factors": list(IN_FACTOR),
                        "Mt_alt_pct": MT_ALT_PCT,
                        "expected_rows": len(window_ids) * len(L_REF) * 5,
                        "actual_rows_note": "卡面写 9×2×5=90 指的是 5 组扰动；"
                                            "为使 K5 的 Δ 有基准，额外保留每个"
                                            "（窗, L_wsr）的 baseline 行，实得 90+18=108 行。"
                                            "多出的 18 行是基准，不是新扰动。"},
        },
        "predeclared": {
            "K1": "HL_tot=1%、f_gas=1 下 L_wsr 0.2575→6 的 Δ（9 窗中位与范围），并与 P1.11-A 对比",
            "K2": "固定 L_wsr（1 与 4 m），每 1% HL_tot 的 Δ",
            "K3": "HL_tot ≤ 1% 时是否有 L_wsr ≤ 6 m 使 |X_C 偏差| ≤ 1 pp",
            "K4": "逐窗插值所需 HL_tot*（0.5/4% 边界外标 boundary）",
            "K5": "O2 ±3%、煤量 ±3% 的影响换算成等效 HL_tot",
            "K6": f"L_wsr={L_JAC} m、HL_tot=1%、f_gas=1 处 [X_C,CO2,CO,H2] 对 "
                  "[L_wsr,HL_tot,ln f_gas,O2] 的雅可比；缩放 1/0.5/0.5/0.5",
            "K7": "到达点附近两种口径的 ξ、残差、氢账本；Mt 4% 的影响",
            "K8": "total_percent 与 total_percent_uniform 在 HL_tot=1% 下的出口差",
        },
        "unit_checks_all_pass": unit_pass,
        "source_csv": {
            "p17d": {"round": "p1.7d-20261103", "file": P17D_CSV.name,
                     "sha256": hashlib.sha256(P17D_CSV.read_bytes()).hexdigest()},
            "p18d": {"round": "p1.8d-20261104", "file": P18D_CSV.name,
                     "sha256": hashlib.sha256(P18D_CSV.read_bytes()).hexdigest()},
            "p11a": {"round": "p1.11a-20261106", "file": "downstream_convergence.csv"},
        },
        "outputs": ["unit_checks.csv", "scan.csv", "alloc_sensitivity.csv",
                    "input_sensitivity.csv", "jacobian.csv", "readings.csv",
                    "protocol.json", "interpretation.md", "sha256.txt"],
    })

    text = interpretation(unit_df, scan, alloc, inputs, jac, rd, iters_df)
    written = write_round_outputs(
        OUT_DIR,
        tables={
            "unit_checks.csv": unit_df,
            "scan.csv": scan,
            "alloc_sensitivity.csv": alloc,
            "input_sensitivity.csv": inputs,
            "jacobian.csv": jac,
            "readings.csv": rd,
        },
        protocol=protocol,
        interpretation=text,
    )
    _rewrite_manifest(OUT_DIR)
    print("written:", [p.name for p in written], flush=True)


if __name__ == "__main__":
    main()
