#!/usr/bin/env python3
"""P1.13-A 试点非线性反演：κ（O/C 协调）+ ξ（激冷段变换）+ f_gas（2026-11-07 卡）。

冻结：wsr_plus_pfr、L_wsr=1 m（名义）、dz_down=0.0515 m、total_percent HL_tot=1%、
OxygenMixingModel=instant、煤量主口径、实测载气、全部 Arrhenius（除 f_gas 作用的
C+H2O / C+CO2 共用指前因子）、限幅、判据、求解器容差。

本轮只做试点反演，不采纳任何参数值，不改合同。

用法：
    ./.venv/bin/python scripts/generate_p1_13a_20261107.py [--stage all|window|phase|robust|loo|read|smoke]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import generate_p1_8a_20261104 as p18a  # noqa: E402

from model.kinetics import calculate_wgs_equilibrium  # noqa: E402
from model.p0_round_20260930 import m1_reference_window_ids  # noqa: E402
from model.p1_5_round_20261030 import (  # noqa: E402
    base_protocol,
    run_case,
    silence_solver_logs,
    write_round_outputs,
)
from model.revision_evidence_common import load_frozen_split, source_commit  # noqa: E402

OUT_DIR = ROOT / "outputs" / "diagnostics" / "round-p1.13a-20261107"
P17A_BASE = ROOT / "outputs" / "diagnostics" / "round-p1.7a-20261103" / "baseline.csv"
P17D_CSV = p18a.P17D_CSV
P18D_CSV = p18a.P18D_CSV
M1_MATRIX = p18a.M1_MATRIX

DEV_L = 10.3
DZ_DOWN = 0.0515
L_WSR_NOM = 1.0
HL_NOM = 1.0
L_WSR_ROBUST = 4.0
HL_ROBUST = (0.5, 1.5)

BOUNDS_LO = np.array([0.88, 0.0, math.log(1.0 / 64.0)])
BOUNDS_HI = np.array([1.06, 250.0, math.log(4.0)])
STARTS = ((1.00, 0.0, 1.0), (0.97, 50.0, 0.25), (0.94, 100.0, 0.125))
START_NAMES = ("s0_nom", "s1_mid", "s2_low")

SIGMAS = np.array([1.5, 0.5, 0.5, 0.5])  # X_C pp, CO/H2/CO2 vol%
OBS_NAMES = ("X_C_pp", "CO_vol_pct", "H2_vol_pct", "CO2_vol_pct")
PARAM_NAMES = ("kappa", "xi_mol_s", "ln_f_gas")

DIFF_REL = 1e-4
STEP_FLOOR = np.array([1.0, 10.0, 1.0])  # κ, ξ mol/s, ln f_gas
# least_squares x_scale；P2 主报告用 J @ diag(X_SCALE)。原 cond(J) 保留为历史读法。
X_SCALE = np.array([0.05, 50.0, 1.0])
LEDGER_SCHEMA_ONCE = "xi_counted_once"
MAX_NFEV = 240
N_WORKERS_DEFAULT = 15
COST_REL_UNIQUE = 0.01
BOUND_FRAC = 0.01
KAPPA_PHYS = (0.92, 1.03)
FGAS_PHYS = (1.0 / 16.0, 2.0)
TEQ_PHYS = (500.0, 1300.0)
COND_OK = 100.0
CORR_OK = 0.9
RMS_OK = 1.0
FAIL_WINDOW_STOP = 3
BOUND_WINDOW_FLAG = 6

PHASE_MAP = {"P1": "phase1", "P2": "phase2", "P3": "phase3", "P4": "phase4"}

_CACHE: dict[tuple, dict] = {}
_CTX: dict = {}


def mesh_n(l_wsr: float, dz_down: float = DZ_DOWN) -> int:
    rl = DEV_L - l_wsr
    n_full = int(math.floor(rl / dz_down + 1e-12))
    rem = rl - float(n_full) * dz_down
    return 1 + n_full + (1 if rem > 1e-12 else 0)


def _short(wid: str) -> str:
    return wid[-16:]


def _phase_of(wid: str) -> str:
    raw = str(_CTX["p18d"].loc[wid, "phase"])
    if raw in PHASE_MAP:
        return PHASE_MAP[raw]
    if raw.startswith("phase"):
        return raw
    return f"phase{raw[-1]}"


def _plant_y(wid: str, basis: str) -> np.ndarray:
    xc = float(_CTX["p18d"].loc[wid, "xc_measured_pct"])
    if basis == "analyzer":
        b = _CTX["baseline"].loc[wid]
        return np.array([xc, float(b["plant_CO_vol_pct"]),
                         float(b["plant_H2_vol_pct"]),
                         float(b["plant_CO2_vol_pct"])], dtype=float)
    co, h2, co2 = p18a.LIU_PHASE_COMP[_phase_of(wid)]
    return np.array([xc, co, h2, co2], dtype=float)


def _carrier_nm3(wid: str) -> float:
    return float(_CTX["p18d"].loc[wid, "co2_carrier_measured_nm3_h"])


def _carrier_mol(wid: str) -> float:
    return p18a._nm3_h_to_mol_s(_carrier_nm3(wid))


def _plant_dry_mol(wid: str) -> float:
    return float(_CTX["p18d"].loc[wid, "syngas_dry_cal_nm3_h"]) / 22.414 / 3600.0 * 1000.0


def _run_reactor(wid: str, kappa: float, f_gas: float, l_wsr: float, hl: float) -> dict:
    key = (wid, round(float(kappa), 8), round(float(f_gas), 10),
           round(float(l_wsr), 6), round(float(hl), 6))
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    row = _CTX["by_id"][wid]
    dry = float(_CTX["p17d"].loc[wid, "coal_dry_recon_phase_kg_s"])
    op = {
        "HeatLossModel": "total_percent",
        "HeatLossPercentTotal": float(hl),
        "GasificationRateFactor": float(f_gas),
        "carrier_gas_flow": p18a._carrier_kg_s(_carrier_nm3(wid)),
        "MeshFamily": "wsr_plus_pfr",
        "WSRLength_m": float(l_wsr),
        "PFRCellLength_m": float(DZ_DOWN),
        "OxygenMixingModel": "instant",
        "o2_flow_scale": float(kappa),
    }
    print(f"    solve {_short(wid)} κ={kappa:.4f} f_gas={f_gas:.4g} "
          f"L={l_wsr:g} HL={hl:g}", flush=True)
    rec = run_case(row, n_cells=mesh_n(l_wsr), coal_flow_dry_kg_s=dry,
                   keep_system=True, mesh_family="wsr_plus_pfr", extra_op=op)
    system = rec["_system"]
    snap: dict = {
        "window_id": wid, "kappa": float(kappa), "f_gas": float(f_gas),
        "L_wsr": float(l_wsr), "HL_tot_pct": float(hl),
        "execution_status": str(rec.get("execution_status")),
        "no_root_cell_index": rec.get("no_root_cell_index"),
        "cell0_n_distinct_roots": rec.get("cell0_n_distinct_roots"),
        "solve_wall_time_s": rec.get("solve_wall_time_s"),
        "hloss_iters": int(getattr(system, "heat_loss_iterations", -1)),
        "hloss_status": str(getattr(system, "hloss_status", "n/a")),
        "hloss_raw_dw": float(getattr(system, "heat_loss_raw_dw", float("nan"))),
        "ok": rec.get("execution_status") == "completed" and bool(system.results),
    }
    if snap["ok"]:
        gas = np.asarray(system.results[-1].gas_moles, dtype=float)
        inlet_c0 = float(system.cells[0].inlet.solid_mass * system.cells[0].inlet.carbon_fraction)
        c0 = system.results[0]
        zc = np.array([float(c.z) for c in system.cells], dtype=float)
        xc_at = {}
        for z, tag in ((2.0, "2m"), (4.0, "4m"), (6.0, "6m")):
            i = int(np.argmin(np.abs(zc - z)))
            st = system.results[i]
            xc_at[tag] = float(1.0 - st.solid_mass * st.carbon_fraction / max(inlet_c0, 1e-12))
        snap.update({
            "X_C_pp": float(rec["X_C_implied"]) * 100.0,
            "T_out_K": float(rec["T_out_K"]),
            "T_wsr_K": float(c0.T),
            "X_C_wsr": float(1.0 - c0.solid_mass * c0.carbon_fraction / max(inlet_c0, 1e-12)),
            "X_C_2m": xc_at["2m"], "X_C_4m": xc_at["4m"], "X_C_6m": xc_at["6m"],
            "F_CO": float(gas[2]), "F_CO2": float(gas[3]), "F_H2": float(gas[5]),
            "F_H2O": float(gas[7]), "F_CH4": float(gas[1]), "F_H2S": float(gas[4]),
            "dry_total_mol_s": float(np.sum(gas[:7])),
        })
    _CACHE[key] = snap
    return snap


def t_eq_from_q(q: float) -> float:
    if not np.isfinite(q) or q <= 0.0:
        return float("nan")

    def f(T: float) -> float:
        k = float(calculate_wgs_equilibrium(T))
        if not np.isfinite(k) or k <= 0.0:
            return 1e6
        return math.log(k) - math.log(q)

    try:
        fa, fb = f(400.0), f(2000.0)
        if not (np.isfinite(fa) and np.isfinite(fb)) or fa * fb > 0.0:
            return float("nan")
        return float(brentq(f, 400.0, 2000.0, xtol=1e-4, maxiter=80))
    except (ValueError, RuntimeError):
        return float("nan")


def apply_xi(snap: dict, xi: float) -> dict:
    """反应器后变换。水由激冷水提供，不把 ξ 限制在出口 H2O 以内。"""
    if not snap.get("ok"):
        return {"ok": False}
    xi = float(xi)
    f_co = snap["F_CO"] - xi
    f_co2 = snap["F_CO2"] + xi
    f_h2 = snap["F_H2"] + xi
    f_h2o_minus = snap["F_H2O"] - xi
    # 激冷水提供变换所需的水：T_eq 用反应器出口 H2O（不扣 ξ），与卡面「不受出口 H2O 限制」一致
    f_h2o_teq = snap["F_H2O"]
    dry = snap["dry_total_mol_s"] + xi
    q = ((f_co2 * f_h2) / (f_co * f_h2o_teq)
         if f_co > 1e-12 and f_h2o_teq > 1e-12 else float("nan"))
    return {
        "ok": True, "xi_mol_s": xi,
        "X_C_pp": snap["X_C_pp"],
        "CO_vol_pct": f_co / max(dry, 1e-12) * 100.0,
        "H2_vol_pct": f_h2 / max(dry, 1e-12) * 100.0,
        "CO2_vol_pct": f_co2 / max(dry, 1e-12) * 100.0,
        "dry_total_mol_s": dry,
        "F_CO": f_co, "F_CO2": f_co2, "F_H2": f_h2, "F_H2O": f_h2o_minus,
        "needs_quench_water": bool(f_h2o_minus < 0.0),
        "co_exhausted": bool(f_co <= 0.0),
        "Q_wgs": q, "T_eq_K": t_eq_from_q(q),
    }


def predict_y(wid: str, theta: np.ndarray, l_wsr: float, hl: float):
    kappa, xi, ln_fg = [float(v) for v in theta]
    snap = _run_reactor(wid, kappa, math.exp(ln_fg), l_wsr, hl)
    shifted = apply_xi(snap, xi)
    if not shifted.get("ok"):
        return np.full(4, np.nan), snap, shifted
    yhat = np.array([shifted["X_C_pp"], shifted["CO_vol_pct"],
                     shifted["H2_vol_pct"], shifted["CO2_vol_pct"]], dtype=float)
    return yhat, snap, shifted


def scaled_residual(wid: str, basis: str, theta, l_wsr: float, hl: float) -> np.ndarray:
    yhat, snap, _sh = predict_y(wid, np.asarray(theta, dtype=float), l_wsr, hl)
    if not snap.get("ok") or not np.all(np.isfinite(yhat)):
        return np.full(4, 1e3)
    return (yhat - _plant_y(wid, basis)) / SIGMAS


def fd_steps(theta: np.ndarray) -> np.ndarray:
    return DIFF_REL * np.maximum(np.abs(theta), STEP_FLOOR)


def group_residual(wids, basis, theta, l_wsr, hl) -> np.ndarray:
    return np.concatenate([scaled_residual(w, basis, theta, l_wsr, hl) for w in wids])


def group_jacobian(wids, basis, theta, l_wsr, hl) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    r0 = group_residual(wids, basis, theta, l_wsr, hl)
    J = np.zeros((len(r0), 3), dtype=float)
    steps = fd_steps(theta)
    for j in range(3):
        tp = theta.copy()
        h = float(steps[j])
        tp[j] = theta[j] + h
        if tp[j] > BOUNDS_HI[j]:
            h = -h
            tp[j] = theta[j] + h
        tp[j] = min(max(float(tp[j]), float(BOUNDS_LO[j])), float(BOUNDS_HI[j]))
        h_eff = float(tp[j] - theta[j])
        if abs(h_eff) < 1e-16:
            continue
        J[:, j] = (group_residual(wids, basis, tp, l_wsr, hl) - r0) / h_eff
    return J


def _bound_hits(theta: np.ndarray) -> str:
    hits = []
    span = BOUNDS_HI - BOUNDS_LO
    for i, name in enumerate(PARAM_NAMES):
        tol = BOUND_FRAC * float(span[i])
        if theta[i] <= BOUNDS_LO[i] + tol:
            hits.append(f"{name}:lo")
        if theta[i] >= BOUNDS_HI[i] - tol:
            hits.append(f"{name}:hi")
    return ",".join(hits)


def _jac_stats(J: np.ndarray, col_scale=None):
    """加权残差雅可比的曲率诊断。col_scale 为参数列尺度时算 J @ diag(s)。"""
    nan3 = np.full(3, np.nan)
    nan33 = np.full((3, 3), np.nan)
    M = np.asarray(J, dtype=float)
    if M.size == 0 or not np.all(np.isfinite(M)):
        return nan3, nan33, float("nan")
    if col_scale is not None:
        M = M * np.asarray(col_scale, dtype=float).reshape(1, -1)
    jtj = M.T @ M
    try:
        cond = float(np.linalg.cond(M))
    except np.linalg.LinAlgError:
        cond = float("inf")
    try:
        cov = np.linalg.inv(jtj)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(jtj)
    sig = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    corr = nan33.copy()
    for i in range(3):
        for k in range(3):
            d = sig[i] * sig[k]
            corr[i, k] = cov[i, k] / d if d > 0 else float("nan")
    return sig, corr, cond


def _cov_from_jac(J: np.ndarray):
    sig, corr, cond = _jac_stats(J, None)
    sigs, corrs, conds = _jac_stats(J, X_SCALE)
    return sig, corr, cond, sigs, corrs, conds


def _theta_from_start(start) -> np.ndarray:
    k, xi, fg = start
    return np.array([k, xi, math.log(fg)], dtype=float)


def _fit_once(wids, basis, start, l_wsr, hl, seed: int) -> dict:
    np.random.seed(int(seed) % (2**31 - 1))
    x0 = _theta_from_start(start)
    x0 = np.clip(x0, BOUNDS_LO, BOUNDS_HI)

    def fun(th):
        return group_residual(wids, basis, th, l_wsr, hl)

    def jac(th):
        return group_jacobian(wids, basis, th, l_wsr, hl)

    try:
        res = least_squares(
            fun, x0, jac=jac, bounds=(BOUNDS_LO, BOUNDS_HI), method="trf",
            xtol=1e-8, ftol=1e-8, gtol=1e-8, max_nfev=MAX_NFEV,
            x_scale=X_SCALE.tolist(),
        )
        ok = bool(res.success)
        theta = np.array(res.x, dtype=float)
        r = np.asarray(res.fun, dtype=float)
        J = np.asarray(res.jac, dtype=float) if res.jac is not None else jac(theta)
        nfev, status, msg = int(res.nfev), int(res.status), str(res.message)
        cost = float(res.cost)
    except Exception as exc:  # noqa: BLE001 — 记录后继续其他起点
        return {
            "success": False, "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=4),
            "kappa": float("nan"), "xi_mol_s": float("nan"), "ln_f_gas": float("nan"),
            "f_gas": float("nan"), "cost": float("nan"), "rms_scaled": float("nan"),
            "nfev": 0, "status": -99, "message": str(exc),
        }
    sig, corr, cond, sigs, corrs, conds = _cov_from_jac(J)
    rms = float(np.sqrt(np.mean(r ** 2))) if r.size else float("nan")
    return {
        "success": ok, "error": "" if ok else msg,
        "kappa": float(theta[0]), "xi_mol_s": float(theta[1]),
        "ln_f_gas": float(theta[2]), "f_gas": float(math.exp(theta[2])),
        "cost": cost, "rss": float(np.dot(r, r)), "rms_scaled": rms,
        "n_obs": int(r.size), "nfev": nfev, "status": status, "message": msg,
        "sigma_kappa": float(sig[0]), "sigma_xi": float(sig[1]), "sigma_ln_f_gas": float(sig[2]),
        "cond_J": cond,
        "rho_kappa_xi": float(corr[0, 1]), "rho_kappa_lnfg": float(corr[0, 2]),
        "rho_xi_lnfg": float(corr[1, 2]),
        "sigma_kappa_xscale": float(sigs[0]), "sigma_xi_xscale": float(sigs[1]),
        "sigma_ln_f_gas_xscale": float(sigs[2]),
        "cond_J_xscale": conds,
        "rho_kappa_xi_xscale": float(corrs[0, 1]),
        "rho_kappa_lnfg_xscale": float(corrs[0, 2]),
        "rho_xi_lnfg_xscale": float(corrs[1, 2]),
        "bound_hits": _bound_hits(theta),
        "J": J, "r": r, "theta": theta, "sig": sig, "corr": corr,
    }


def _enrich_point(wid, basis, fit: dict, l_wsr, hl) -> dict:
    """在收敛 θ 上补出口、T_eq、氢账本、载气 CO2 净去向。"""
    out = dict(fit)
    out.pop("J", None)
    out.pop("r", None)
    out.pop("theta", None)
    out.pop("sig", None)
    out.pop("corr", None)
    if not fit.get("success"):
        return out
    theta = np.array([fit["kappa"], fit["xi_mol_s"], fit["ln_f_gas"]], dtype=float)
    yhat, snap, sh = predict_y(wid, theta, l_wsr, hl)
    y = _plant_y(wid, basis)
    out.update({
        "window_id": wid, "phase": _phase_of(wid), "composition_basis": basis,
        "L_wsr": l_wsr, "HL_tot_pct": hl,
        "model_X_C_pp": float(yhat[0]), "model_CO_vol_pct": float(yhat[1]),
        "model_H2_vol_pct": float(yhat[2]), "model_CO2_vol_pct": float(yhat[3]),
        "plant_X_C_pp": float(y[0]), "plant_CO_vol_pct": float(y[1]),
        "plant_H2_vol_pct": float(y[2]), "plant_CO2_vol_pct": float(y[3]),
        "resid_X_C_pp": float(yhat[0] - y[0]),
        "resid_CO_vol_pct": float(yhat[1] - y[1]),
        "resid_H2_vol_pct": float(yhat[2] - y[2]),
        "resid_CO2_vol_pct": float(yhat[3] - y[3]),
        "T_out_K": snap.get("T_out_K", float("nan")),
        "T_wsr_K": snap.get("T_wsr_K", float("nan")),
        "X_C_wsr": snap.get("X_C_wsr", float("nan")),
        "X_C_2m": snap.get("X_C_2m", float("nan")),
        "X_C_4m": snap.get("X_C_4m", float("nan")),
        "X_C_6m": snap.get("X_C_6m", float("nan")),
        "T_eq_K": sh.get("T_eq_K", float("nan")),
        "Q_wgs": sh.get("Q_wgs", float("nan")),
        "needs_quench_water": sh.get("needs_quench_water"),
        "co_exhausted": sh.get("co_exhausted"),
        "execution_status": snap.get("execution_status"),
        "hloss_iters": snap.get("hloss_iters"),
        "hloss_status": snap.get("hloss_status"),
        "hloss_raw_dw": snap.get("hloss_raw_dw"),
        "cell0_n_distinct_roots": snap.get("cell0_n_distinct_roots"),
        "solve_wall_time_s": snap.get("solve_wall_time_s"),
    })
    plant_h2 = 2.0 * _plant_dry_mol(wid) * (y[2] / 100.0)
    # 变换后 F_H2 已含 +ξ，2ξ 只计一次。H2-only：分子分母都是 2·H2。
    h_h2only = 2.0 * sh.get("F_H2", float("nan"))
    h_full_num = (2.0 * sh.get("F_H2", 0.0) + 4.0 * snap.get("F_CH4", 0.0)
                  + 2.0 * snap.get("F_H2S", 0.0))
    out["H_ledger_h2only"] = float(h_h2only / plant_h2) if plant_h2 > 0 else float("nan")
    out["H_ledger_full_over_plant_h2"] = (
        float(h_full_num / plant_h2) if plant_h2 > 0 else float("nan"))
    out["H_ledger_schema"] = LEDGER_SCHEMA_ONCE
    out["H_ledger_xi_counted_once"] = True
    out["H_ledger_num"] = (
        "2·F_H2_shifted = 2·(F_H2_reactor+ξ); 2ξ counted once "
        "[+ 4 CH4 + 2 H2S in full column]"
    )
    out["H_ledger_den"] = "2 · plant_dry · y_H2（装置 H2-only；无 CH4/H2S 分母）"
    out["T_eq_assumption"] = (
        "Q uses shifted CO/H2/CO2 with reactor-outlet H2O; quench water is assumed "
        "to replace exactly the ξ moles consumed (proxy, not a measured inventory)"
    )
    out["carrier_CO2_in_mol_s"] = _carrier_mol(wid)
    out["CO2_net_after_xi_mol_s"] = float(sh.get("F_CO2", float("nan")) - _carrier_mol(wid))
    k_ok = KAPPA_PHYS[0] <= fit["kappa"] <= KAPPA_PHYS[1]
    fg_ok = FGAS_PHYS[0] <= fit["f_gas"] <= FGAS_PHYS[1]
    teq = out["T_eq_K"]
    teq_ok = bool(np.isfinite(teq) and TEQ_PHYS[0] <= teq <= TEQ_PHYS[1])
    out["phys_kappa_ok"] = k_ok
    out["phys_fgas_ok"] = fg_ok
    out["phys_Teq_ok"] = teq_ok
    out["phys_any_flag"] = (not k_ok) or (not fg_ok) or (not teq_ok) or bool(fit.get("bound_hits"))
    return out


def _jac_rows(tag: dict, J: np.ndarray, wids: list[str]) -> list[dict]:
    rows = []
    if J is None or not isinstance(J, np.ndarray):
        return rows
    for i, wid in enumerate(wids):
        for o, obs in enumerate(OBS_NAMES):
            ri = i * 4 + o
            if ri >= J.shape[0]:
                continue
            rec = {
                "scope": tag.get("scope"), "window_id": wid,
                "phase": tag.get("phase"), "composition_basis": tag.get("basis"),
                "L_wsr": tag.get("L_wsr"), "HL_tot_pct": tag.get("HL_tot"),
                "start_name": tag.get("start_name", ""),
                "selected": tag.get("selected", False),
                "obs": obs, "row_index": ri,
            }
            for j, p in enumerate(PARAM_NAMES):
                rec[f"d_{p}"] = float(J[ri, j]) if j < J.shape[1] else float("nan")
            rows.append(rec)
    return rows


def _uniqueness(starts: list[dict]) -> list[dict]:
    ok = [s for s in starts if s.get("success") and np.isfinite(s.get("cost", float("nan")))]
    flags = []
    for i in range(len(ok)):
        for k in range(i + 1, len(ok)):
            a, b = ok[i], ok[k]
            cmin = min(abs(a["cost"]), abs(b["cost"]))
            if cmin <= 0:
                rel = 0.0 if abs(a["cost"] - b["cost"]) == 0 else 1.0
            else:
                rel = abs(a["cost"] - b["cost"]) / cmin
            if rel >= COST_REL_UNIQUE:
                continue
            differ = False
            for p, sa, sb in (
                ("kappa", "sigma_kappa", "sigma_kappa"),
                ("xi_mol_s", "sigma_xi", "sigma_xi"),
                ("ln_f_gas", "sigma_ln_f_gas", "sigma_ln_f_gas"),
            ):
                da = abs(float(a[p]) - float(b[p]))
                if da > float(a.get(sa, np.nan)) and da > float(b.get(sb, np.nan)):
                    if np.isfinite(a.get(sa, np.nan)) and np.isfinite(b.get(sb, np.nan)):
                        differ = True
            if differ:
                flags.append({
                    "start_a": a.get("start_name"), "start_b": b.get("start_name"),
                    "cost_rel_diff": rel, "nonunique": True,
                })
    return flags


_N_STARTS = len(STARTS)


def fit_group(wids, basis, l_wsr, hl, scope: str, label: str) -> dict:
    start_rows = []
    jac_rows = []
    seed_base = 20261107
    use = list(zip(STARTS, START_NAMES))[:max(1, int(_N_STARTS))]
    print(f"START {scope} {label[-24:] if isinstance(label, str) else label} "
          f"{basis} L={l_wsr} HL={hl} n_win={len(wids)} n_starts={len(use)}", flush=True)
    for i, (st, name) in enumerate(use):
        key = "|".join(list(wids) + [basis, name, f"{l_wsr:.6g}", f"{hl:.6g}"])
        seed = seed_base + int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % 100000
        raw = _fit_once(wids, basis, st, l_wsr, hl, seed)
        row = {
            "scope": scope, "label": label, "composition_basis": basis,
            "phase": _phase_of(wids[0]) if len(wids) == 1 else label,
            "window_id": wids[0] if len(wids) == 1 else ",".join(wids),
            "n_windows": len(wids), "start_name": name,
            "start_kappa": st[0], "start_xi": st[1], "start_f_gas": st[2],
            "L_wsr": l_wsr, "HL_tot_pct": hl,
        }
        row.update({k: v for k, v in raw.items()
                    if k not in ("J", "r", "theta", "sig", "corr")})
        start_rows.append(row)
        if raw.get("success") and raw.get("J") is not None:
            jac_rows.extend(_jac_rows(
                {"scope": scope, "phase": row["phase"], "basis": basis,
                 "L_wsr": l_wsr, "HL_tot": hl, "start_name": name},
                raw["J"], wids,
            ))
        print(f"  {scope} {label} {basis} {name} success={row.get('success')} "
              f"rms={row.get('rms_scaled')} nfev={row.get('nfev')}", flush=True)

    uniq = _uniqueness(start_rows)
    ok = [s for s in start_rows if s.get("success")]
    best = min(ok, key=lambda s: s["cost"]) if ok else None
    if best is not None:
        for jr in jac_rows:
            jr["selected"] = jr.get("start_name") == best.get("start_name")
    best_full = None
    extra_points = []
    if best is not None:
        raw_best = None
        # 再评一次最佳起点，补逐窗出口（阶段拟合要每个窗一行）
        theta = np.array([best["kappa"], best["xi_mol_s"], best["ln_f_gas"]])
        for wid in wids:
            enriched = _enrich_point(wid, basis, {
                **best, "success": True,
            }, l_wsr, hl)
            extra_points.append(enriched)
        best_full = extra_points[0] if extra_points else best
        best_full = dict(best)
        best_full["nonunique"] = bool(uniq)
        best_full["n_converged_starts"] = len(ok)
        best_full["n_starts"] = len(start_rows)
        if extra_points:
            # 阶段：把逐窗出口均值写进主行，详细在 points
            best_full.update({k: extra_points[0][k] for k in extra_points[0]
                              if k not in best_full})
            best_full["window_id"] = wids[0] if len(wids) == 1 else ",".join(wids)
            best_full["n_windows"] = len(wids)
    else:
        best_full = {
            "scope": scope, "label": label, "composition_basis": basis,
            "window_id": wids[0] if len(wids) == 1 else ",".join(wids),
            "phase": _phase_of(wids[0]) if wids else "",
            "success": False, "n_converged_starts": 0, "n_starts": len(start_rows),
            "L_wsr": l_wsr, "HL_tot_pct": hl, "nonunique": False,
        }
    return {
        "best": best_full,
        "starts": start_rows,
        "jac_rows": jac_rows,
        "points": extra_points,
        "unique_flags": uniq,
        "failed": best is None,
    }


# ---------------------------------------------------------------------------
# 进程池
# ---------------------------------------------------------------------------

def _init_worker():
    silence_solver_logs()
    ok, rows_, _inv = load_frozen_split()
    if not ok:
        raise RuntimeError("frozen R52-C-75 split was not consumed intact")
    by_id = {r["window_id"]: r for r in rows_}
    p17d = pd.read_csv(P17D_CSV).set_index("window_id")
    p18d = pd.read_csv(P18D_CSV).set_index("window_id")
    base = pd.read_csv(P17A_BASE)
    base = base[(base["coal_basis"] == "recon_phase") & (base["N_cells"] == 40)].set_index("window_id")
    _CTX.update({"by_id": by_id, "p17d": p17d, "p18d": p18d, "baseline": base})
    _CACHE.clear()


def _job_fit(payload: dict) -> dict:
    if "by_id" not in _CTX:
        _init_worker()
    return fit_group(payload["wids"], payload["basis"], payload["l_wsr"],
                     payload["hl"], payload["scope"], payload["label"])


def _windows_by_phase(window_ids: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for wid in window_ids:
        out.setdefault(_phase_of(wid), []).append(wid)
    return out


def _run_jobs(jobs: list[dict], n_workers: int) -> list[dict]:
    if not jobs:
        return []
    workers = max(1, min(n_workers, len(jobs)))
    if workers == 1:
        _init_worker()
        return [_job_fit(j) for j in jobs]
    out = [None] * len(jobs)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
        fut = {ex.submit(_job_fit, j): i for i, j in enumerate(jobs)}
        for f in as_completed(fut):
            i = fut[f]
            try:
                out[i] = f.result()
            except Exception as exc:  # noqa: BLE001
                j = jobs[i]
                print(f"JOB FAIL {j}: {exc}", flush=True)
                out[i] = {
                    "best": {"success": False, "error": str(exc), **j},
                    "starts": [], "jac_rows": [], "points": [],
                    "unique_flags": [], "failed": True,
                }
    return out


def _collect(results: list[dict]):
    bests, starts, jacs, points = [], [], [], []
    for r in results:
        if r.get("best"):
            bests.append(r["best"])
        starts.extend(r.get("starts") or [])
        jacs.extend(r.get("jac_rows") or [])
        points.extend(r.get("points") or [])
    return bests, starts, jacs, points


# ---------------------------------------------------------------------------
# 留一窗（复用逐窗 θ̂，不再拟合）
# ---------------------------------------------------------------------------

def loo_from_window_fits(window_fits: pd.DataFrame, window_ids: list[str]) -> pd.DataFrame:
    _init_worker()
    by_phase = _windows_by_phase(window_ids)
    rows = []
    theta0 = np.array([1.0, 0.0, 0.0])  # κ=1, ξ=0, ln f_gas=0
    uncorr = {}
    for wid in window_ids:
        yhat, snap, _sh = predict_y(wid, theta0, L_WSR_NOM, HL_NOM)
        uncorr[wid] = (yhat, snap.get("ok", False))
    for basis in p18a.COMP_BASES:
        sub = window_fits[(window_fits["composition_basis"] == basis)
                          & (window_fits["success"] == True)]  # noqa: E712
        for phase, wids in by_phase.items():
            for train in wids:
                tests = [w for w in wids if w != train]
                if not tests:
                    continue
                tr = sub[sub["window_id"] == train]
                if tr.empty:
                    for te in tests:
                        rows.append({
                            "phase": phase, "composition_basis": basis,
                            "train_window": train, "test_window": te,
                            "method": "inversion", "success": False,
                            "rms_scaled": float("nan"), "note": "train window fit failed",
                        })
                    continue
                trow = tr.iloc[0]
                th = np.array([trow["kappa"], trow["xi_mol_s"], trow["ln_f_gas"]], dtype=float)
                y_tr = _plant_y(train, basis)
                yhat_tr_u, ok_tr = uncorr[train]
                patch = y_tr - yhat_tr_u if ok_tr else np.full(4, np.nan)
                for te in tests:
                    y_te = _plant_y(te, basis)
                    yhat_inv, snap, _sh = predict_y(te, th, L_WSR_NOM, HL_NOM)
                    yhat_u, ok_u = uncorr[te]
                    yhat_patch = yhat_u + patch if ok_u else np.full(4, np.nan)
                    for method, yhat in (
                        ("inversion", yhat_inv),
                        ("uncorrected", yhat_u),
                        ("residual_patch", yhat_patch),
                    ):
                        if not np.all(np.isfinite(yhat)):
                            rms = float("nan")
                            ok = False
                        else:
                            r = (yhat - y_te) / SIGMAS
                            rms = float(np.sqrt(np.mean(r ** 2)))
                            ok = True
                        rows.append({
                            "phase": phase, "composition_basis": basis,
                            "train_window": train, "test_window": te,
                            "method": method, "success": ok, "rms_scaled": rms,
                            "dX_C_pp": float(yhat[0] - y_te[0]) if ok else float("nan"),
                            "dCO_vol_pct": float(yhat[1] - y_te[1]) if ok else float("nan"),
                            "dH2_vol_pct": float(yhat[2] - y_te[2]) if ok else float("nan"),
                            "dCO2_vol_pct": float(yhat[3] - y_te[3]) if ok else float("nan"),
                            "train_kappa": float(trow["kappa"]) if method == "inversion" else 1.0,
                            "train_xi": float(trow["xi_mol_s"]) if method == "inversion" else 0.0,
                            "train_f_gas": float(trow["f_gas"]) if method == "inversion" else 1.0,
                        })
    return pd.DataFrame(rows)


def _jac_blocks(jac: pd.DataFrame, n_obs: int) -> list[tuple[pd.Index, np.ndarray]]:
    """按文件顺序把同一拟合的观测行切成 n_obs 行一块，保留原索引。"""
    rows = list(jac.index)
    out = []
    for i in range(0, len(rows), n_obs):
        idx = rows[i:i + n_obs]
        if len(idx) < n_obs:
            continue
        chunk = jac.loc[idx].sort_values("row_index")
        J = chunk[["d_kappa", "d_xi_mol_s", "d_ln_f_gas"]].to_numpy(dtype=float)
        out.append((chunk.index, J))
    return out


def _n_obs_for(row) -> int:
    n_win = row.get("n_windows")
    try:
        n_win = int(n_win)
    except (TypeError, ValueError):
        n_win = 3 if str(row.get("phase")) == "phase4" else 2
    return 4 * max(n_win, 1)


def _fit_jac_mask(jac: pd.DataFrame, row, scope: str) -> pd.Series:
    if scope == "window":
        return (
            (jac["scope"] == "window")
            & (jac["window_id"] == row["window_id"])
            & (jac["composition_basis"] == row["composition_basis"])
            & np.isclose(jac["L_wsr"].astype(float), float(row["L_wsr"]))
            & np.isclose(jac["HL_tot_pct"].astype(float), float(row["HL_tot_pct"]))
        )
    return (
        (jac["scope"] == scope)
        & (jac["phase"] == row["phase"])
        & (jac["composition_basis"] == row["composition_basis"])
        & np.isclose(jac["L_wsr"].astype(float), float(row["L_wsr"]))
        & np.isclose(jac["HL_tot_pct"].astype(float), float(row["HL_tot_pct"]))
    )


def _fix_h_ledger_row(r) -> float:
    """去掉全干气分子里多加的 2ξ。分母仍是装置 H2-only。"""
    raw = float(r.get("H_ledger_full_over_plant_h2", np.nan))
    xi = float(r.get("xi_mol_s", np.nan))
    wid = str(r.get("window_id", "")).split(",")[0]
    y_h2 = float(r.get("plant_H2_vol_pct", np.nan))
    if wid not in _CTX["p18d"].index or not np.isfinite(raw) or not np.isfinite(xi) or not np.isfinite(y_h2):
        return raw
    den = 2.0 * _plant_dry_mol(wid) * (y_h2 / 100.0)
    if den <= 0:
        return raw
    return raw - 2.0 * xi / den


def _as_true_series(s: pd.Series) -> pd.Series:
    return s.map(lambda x: str(x).strip().lower() in ("true", "1", "yes"))


def _valid_ledger_mask(df: pd.DataFrame) -> pd.Series:
    v = pd.to_numeric(df["H_ledger_full_over_plant_h2"], errors="coerce")
    return v.notna() & np.isfinite(v.to_numpy(dtype=float))


def _row_marked_once(df: pd.DataFrame) -> pd.Series:
    once = pd.Series(False, index=df.index)
    if "H_ledger_schema" in df.columns:
        once = once | df["H_ledger_schema"].astype(str).str.strip().eq(LEDGER_SCHEMA_ONCE)
    if "H_ledger_xi_counted_once" in df.columns:
        once = once | _as_true_series(df["H_ledger_xi_counted_once"])
    if "H_ledger_num" in df.columns:
        once = once | df["H_ledger_num"].astype(str).str.contains("counted once", case=False, na=False)
    return once


def _classify_h_ledger(df: pd.DataFrame) -> str:
    """只看有限账本行。once / old；歧义表直接报错，不静默再扣。"""
    if df is None or df.empty or "H_ledger_full_over_plant_h2" not in df.columns:
        return "once"
    valid = _valid_ledger_mask(df)
    if not bool(valid.any()):
        return "once"
    marked = _row_marked_once(df)[valid]
    if bool(marked.all()):
        return "once"
    if bool(marked.any()):
        raise RuntimeError("ambiguous hydrogen-ledger schema among rows with a finite ledger")
    if "H_ledger_full_over_plant_h2_raw" in df.columns:
        return "once"
    if "H_ledger_num" in df.columns:
        raise RuntimeError(
            "ambiguous hydrogen-ledger: H_ledger_num present but valid rows are not marked counted-once"
        )
    return "old"


def _ledger_is_counted_once(df: pd.DataFrame) -> bool:
    return _classify_h_ledger(df) == "once"


def _stamp_ledger_once(df: pd.DataFrame) -> pd.DataFrame:
    """只给有有效账本的行打口径标记；失败行保持缺值。"""
    df = df.copy()
    valid = _valid_ledger_mask(df)
    if "H_ledger_schema" not in df.columns:
        df["H_ledger_schema"] = ""
    if "H_ledger_xi_counted_once" not in df.columns:
        df["H_ledger_xi_counted_once"] = False
    num = (
        "2·F_H2_shifted = 2·(F_H2_reactor+ξ); 2ξ counted once "
        "[+ 4 CH4 + 2 H2S in full column]"
    )
    den = "2 · plant_dry · y_H2（装置 H2-only；全干气列分子含 CH4/H2S，分母不含）"
    teq = "shifted CO/H2/CO2 with reactor-outlet H2O; quench replaces ξ moles of water"
    for col, fill in (("H_ledger_num", num), ("H_ledger_den", den), ("T_eq_assumption", teq)):
        if col not in df.columns:
            df[col] = ""
        df.loc[valid, col] = fill
    df.loc[valid, "H_ledger_schema"] = LEDGER_SCHEMA_ONCE
    df.loc[valid, "H_ledger_xi_counted_once"] = True
    return df


def _maybe_migrate_h_ledger(df: pd.DataFrame) -> pd.DataFrame:
    """只迁移确认属于旧口径且账本有限的行。失败行保留缺值。"""
    if df is None or df.empty or "H_ledger_full_over_plant_h2" not in df.columns:
        return df
    status = _classify_h_ledger(df)
    if status == "once":
        return _stamp_ledger_once(df)
    df = df.copy()
    valid = _valid_ledger_mask(df)
    if "H_ledger_full_over_plant_h2_raw" not in df.columns:
        df["H_ledger_full_over_plant_h2_raw"] = np.nan
    df.loc[valid, "H_ledger_full_over_plant_h2_raw"] = df.loc[valid, "H_ledger_full_over_plant_h2"]
    df.loc[valid, "H_ledger_full_over_plant_h2"] = [
        _fix_h_ledger_row(r) for _, r in df.loc[valid].iterrows()
    ]
    return _stamp_ledger_once(df)


def _finite_equal(a, b, msg: str) -> None:
    av = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float)
    bv = pd.to_numeric(b, errors="coerce").to_numpy(dtype=float)
    ma, mb = np.isfinite(av), np.isfinite(bv)
    if ma.shape != mb.shape or not np.array_equal(ma, mb):
        raise RuntimeError(f"{msg}: finite-mask changed (kept failed rows?)")
    if ma.any() and not np.allclose(av[ma], bv[ma], rtol=0, atol=1e-12):
        i = int(np.flatnonzero(ma)[0])
        raise RuntimeError(f"{msg}: {av[i]!r} vs {bv[i]!r}")


def _with_failed_row(df: pd.DataFrame) -> pd.DataFrame:
    """追加一条无账本的失败拟合，复现「整列 .all() 把新表当旧表」的路径。"""
    fail = df.iloc[[0]].copy()
    fail["success"] = False
    for col in list(fail.columns):
        if col == "H_ledger_schema":
            fail[col] = ""
        elif col == "H_ledger_xi_counted_once":
            fail[col] = False
        elif str(col).startswith("H_ledger") or col in ("cond_J", "cond_J_xscale"):
            fail[col] = np.nan
    return pd.concat([df, fail], ignore_index=True)


def verify_h_ledger_read_paths(win: pd.DataFrame) -> None:
    """旧表迁一次、再读保持、新表直读保持；含失败行时成功行不被动。不跑反应器。"""
    if win is None or win.empty or "H_ledger_full_over_plant_h2" not in win.columns:
        raise RuntimeError("ledger check: no H_ledger_full_over_plant_h2")
    if "by_id" not in _CTX:
        _init_worker()
    base = win.loc[_valid_ledger_mask(win)].copy()
    if base.empty:
        raise RuntimeError("ledger check: no finite ledger rows")
    truth = base["H_ledger_full_over_plant_h2"].to_numpy(dtype=float)

    old = base.copy()
    if "H_ledger_full_over_plant_h2_raw" in old.columns:
        old["H_ledger_full_over_plant_h2"] = old["H_ledger_full_over_plant_h2_raw"]
    else:
        raised = []
        for _, r in old.iterrows():
            val = float(r["H_ledger_full_over_plant_h2"])
            xi = float(r["xi_mol_s"])
            wid = str(r["window_id"]).split(",")[0]
            den = 2.0 * _plant_dry_mol(wid) * float(r["plant_H2_vol_pct"]) / 100.0
            raised.append(val + 2.0 * xi / den if den > 0 else val)
        old["H_ledger_full_over_plant_h2"] = raised
    old = old.drop(columns=[c for c in (
        "H_ledger_schema", "H_ledger_xi_counted_once", "H_ledger_full_over_plant_h2_raw",
        "H_ledger_num", "H_ledger_den", "T_eq_assumption",
    ) if c in old.columns])
    old_fail = _with_failed_row(old)
    once = _maybe_migrate_h_ledger(old_fail)
    _finite_equal(once["H_ledger_full_over_plant_h2"].iloc[:len(base)], truth,
                  "ledger check A (old migrate once, with failed row)")
    if np.isfinite(float(once["H_ledger_full_over_plant_h2"].iloc[-1])):
        raise RuntimeError("ledger check A: failed row should keep a missing ledger")

    twice = _maybe_migrate_h_ledger(once)
    _finite_equal(twice["H_ledger_full_over_plant_h2"], once["H_ledger_full_over_plant_h2"],
                  "ledger check B (re-read, with failed row)")

    fresh = base.copy()
    if "H_ledger_full_over_plant_h2_raw" in fresh.columns:
        fresh = fresh.drop(columns=["H_ledger_full_over_plant_h2_raw"])
    fresh["H_ledger_schema"] = LEDGER_SCHEMA_ONCE
    fresh["H_ledger_xi_counted_once"] = True
    fresh_fail = _with_failed_row(fresh)
    c = _maybe_migrate_h_ledger(fresh_fail)
    _finite_equal(c["H_ledger_full_over_plant_h2"].iloc[:len(base)], truth,
                  "ledger check C (new table + failed row)")
    if float(c["H_ledger_full_over_plant_h2"].iloc[0]) < float(base["H_ledger_h2only"].iloc[0]):
        raise RuntimeError("ledger check C: success row fell below H2-only (double-subtracted)")

    # 4d26a31 新生成：已有去重分子和 num 文案，但缺 schema / counted_once
    legacy = base.copy()
    for col in ("H_ledger_schema", "H_ledger_xi_counted_once", "H_ledger_full_over_plant_h2_raw"):
        if col in legacy.columns:
            legacy = legacy.drop(columns=[col])
    legacy["H_ledger_num"] = (
        "2·F_H2_shifted = 2·(F_H2_reactor+ξ); 2ξ counted once "
        "[+ 4 CH4 + 2 H2S in full column]"
    )
    d = _maybe_migrate_h_ledger(_with_failed_row(legacy))
    _finite_equal(d["H_ledger_full_over_plant_h2"].iloc[:len(base)], truth,
                  "ledger check D (4d26a31 metadata, with failed row)")
    print(
        "ledger-check PASS: old migrate once; re-read stable; new+failed stable; "
        f"4d26a31 metadata stable; H_full[0]={truth[0]:.12g}",
        flush=True,
    )


def attach_xscale_and_ledgers(win, phase, robust, jac, starts) -> tuple:
    """从已存雅可比给获选解补 x_scale 条件数，并去掉氢账本重复的 2ξ。不重跑反应器。"""
    if "by_id" not in _CTX:
        _init_worker()
    jac = jac.copy()
    if "start_name" not in jac.columns:
        jac["start_name"] = ""
    if "selected" not in jac.columns:
        jac["selected"] = False

    def _fill(df, scope: str):
        if df is None or df.empty:
            return df
        df = df.copy()
        for col in ("cond_J_xscale", "rho_kappa_xi_xscale", "rho_kappa_lnfg_xscale",
                    "rho_xi_lnfg_xscale", "sigma_kappa_xscale", "sigma_xi_xscale",
                    "sigma_ln_f_gas_xscale"):
            if col not in df.columns:
                df[col] = np.nan
        for idx, r in df.iterrows():
            n = 4 if scope == "window" else _n_obs_for(r)
            sub = jac.loc[_fit_jac_mask(jac, r, scope)]
            if sub.empty:
                continue
            native = float(r.get("cond_J", np.nan))
            best_J, best_idx, best_d = None, None, 1e99
            for idx_chunk, J in _jac_blocks(sub, n):
                cond = _jac_stats(J, None)[2]
                d = abs(cond - native) if np.isfinite(native) else 1e99
                if d < best_d:
                    best_d, best_J, best_idx = d, J, idx_chunk
            if best_J is None:
                continue
            _sig, _corr, _c = _jac_stats(best_J, None)
            sigs, corrs, conds = _jac_stats(best_J, X_SCALE)
            df.at[idx, "cond_J_xscale"] = conds
            df.at[idx, "rho_kappa_xi_xscale"] = float(corrs[0, 1])
            df.at[idx, "rho_kappa_lnfg_xscale"] = float(corrs[0, 2])
            df.at[idx, "rho_xi_lnfg_xscale"] = float(corrs[1, 2])
            df.at[idx, "sigma_kappa_xscale"] = float(sigs[0])
            df.at[idx, "sigma_xi_xscale"] = float(sigs[1])
            df.at[idx, "sigma_ln_f_gas_xscale"] = float(sigs[2])
            if best_idx is not None and r.get("start_name"):
                jac.loc[best_idx, "start_name"] = r.get("start_name")
                jac.loc[best_idx, "selected"] = True
        return _maybe_migrate_h_ledger(df)

    if starts is not None and not starts.empty and "success" in starts.columns:
        ok_starts = starts[starts["success"] == True]  # noqa: E712
        for _, st in ok_starts.iterrows():
            scope = str(st.get("scope", "window"))
            n = 4 if scope == "window" else _n_obs_for(st)
            sub = jac.loc[_fit_jac_mask(jac, st, scope)]
            native = float(st.get("cond_J", np.nan))
            if sub.empty or not np.isfinite(native):
                continue
            for idx_chunk, J in _jac_blocks(sub, n):
                named = jac.loc[idx_chunk, "start_name"].astype(str).replace({"nan": ""}).fillna("")
                if (named != "").any():
                    continue
                cond = _jac_stats(J, None)[2]
                if abs(cond - native) <= max(1e-4, 1e-8 * abs(native)):
                    jac.loc[idx_chunk, "start_name"] = st.get("start_name", "")
                    break

    win = _fill(win, "window")
    phase = _fill(phase, "phase")
    robust = _fill(robust, "robust")
    return win, phase, robust, jac


# ---------------------------------------------------------------------------
# 预声明读法
# ---------------------------------------------------------------------------

def _add(rows, **kw):
    rows.append({
        "reading": kw.get("reading"), "metric": kw.get("metric"),
        "composition_basis": kw.get("basis", ""),
        "phase": kw.get("phase", ""), "window_id": kw.get("window_id", ""),
        "value": kw.get("value", float("nan")),
        "threshold": kw.get("threshold", float("nan")),
        "pass": kw.get("pass_"), "note": kw.get("note", ""),
    })


def readings(win, phase, robust, loo, starts) -> pd.DataFrame:
    rows: list[dict] = []
    # P1
    for df, scope in ((win, "window"), (phase, "phase")):
        if df is None or df.empty:
            continue
        ok = df[df["success"] == True] if "success" in df.columns else df  # noqa: E712
        for basis in p18a.COMP_BASES:
            sub = ok[ok["composition_basis"] == basis] if not ok.empty else ok
            if sub.empty:
                _add(rows, reading="P1", metric=f"{scope}_n_success::{basis}",
                     basis=basis, value=0, pass_=False, note="无收敛解")
                continue
            rms = sub["rms_scaled"].astype(float)
            _add(rows, reading="P1", metric=f"{scope}_n_success::{basis}",
                 basis=basis, value=float(len(sub)), pass_=None,
                 note=f"/{len(df[df['composition_basis']==basis]) if 'composition_basis' in df.columns else '?'}")
            _add(rows, reading="P1", metric=f"{scope}_median_rms::{basis}",
                 basis=basis, value=float(rms.median()), threshold=RMS_OK,
                 pass_=bool((rms <= RMS_OK).all()) if scope == "window" else bool(rms.median() <= RMS_OK),
                 note="缩放残差均方根；≤1 记为达到预设拟合容差（试点权重 1.5 pp / 0.5 vol%，不是测量不确定度）")
            _add(rows, reading="P1", metric=f"{scope}_max_rms::{basis}",
                 basis=basis, value=float(rms.max()), threshold=RMS_OK,
                 pass_=bool(rms.max() <= RMS_OK))
            n_in = int((rms <= RMS_OK).sum())
            _add(rows, reading="P1", metric=f"{scope}_n_rms_le_1::{basis}",
                 basis=basis, value=float(n_in), pass_=None)
    # P2
    for df, scope in ((win, "window"), (phase, "phase")):
        if df is None or df.empty:
            continue
        ok = df[df["success"] == True]  # noqa: E712
        for _, r in ok.iterrows():
            cond_n = float(r.get("cond_J", np.nan))
            cond_x = float(r.get("cond_J_xscale", np.nan))
            rhos_n = [float(r.get(c, np.nan)) for c in
                      ("rho_kappa_xi", "rho_kappa_lnfg", "rho_xi_lnfg")]
            rhos_x = [float(r.get(c, np.nan)) for c in
                      ("rho_kappa_xi_xscale", "rho_kappa_lnfg_xscale", "rho_xi_lnfg_xscale")]
            max_n = max((abs(x) for x in rhos_n if np.isfinite(x)), default=float("nan"))
            max_x = max((abs(x) for x in rhos_x if np.isfinite(x)), default=float("nan"))
            id_n = bool(np.isfinite(cond_n) and cond_n < COND_OK
                        and np.isfinite(max_n) and max_n < CORR_OK)
            id_x = bool(np.isfinite(cond_x) and cond_x < COND_OK
                        and np.isfinite(max_x) and max_x < CORR_OK)
            names = ("kappa-xi", "kappa-lnfg", "xi-lnfg")
            pair_x = ""
            if not id_x:
                if any(np.isfinite(x) and abs(x) >= CORR_OK for x in rhos_x):
                    pair_x = names[int(np.nanargmax([abs(x) if np.isfinite(x) else -1 for x in rhos_x]))]
                elif np.isfinite(cond_x) and cond_x >= COND_OK:
                    pair_x = f"cond_xscale={cond_x:.3g}"
            _add(rows, reading="P2", metric=f"{scope}_local_threshold_xscale",
                 basis=r.get("composition_basis"), phase=r.get("phase"),
                 window_id=r.get("window_id"), value=float(id_x),
                 pass_=id_x,
                 note=("post-hoc local screen J@diag(0.05,50,1); not global identifiability; "
                       f"cond={cond_x:.4g}; max|ρ|={max_x:.3f}; pair={pair_x}"))
            _add(rows, reading="P2", metric=f"{scope}_historical_threshold_native",
                 basis=r.get("composition_basis"), phase=r.get("phase"),
                 window_id=r.get("window_id"), value=float(id_n),
                 pass_=id_n,
                 note=f"predeclared native units; cond={cond_n:.4g}; max|ρ|={max_n:.3f}")
            _add(rows, reading="P2", metric=f"{scope}_cond_J_xscale",
                 basis=r.get("composition_basis"), phase=r.get("phase"),
                 window_id=r.get("window_id"), value=cond_x, threshold=COND_OK,
                 pass_=bool(np.isfinite(cond_x) and cond_x < COND_OK))
            _add(rows, reading="P2", metric=f"{scope}_cond_J_native",
                 basis=r.get("composition_basis"), phase=r.get("phase"),
                 window_id=r.get("window_id"), value=cond_n, threshold=COND_OK,
                 pass_=bool(np.isfinite(cond_n) and cond_n < COND_OK),
                 note="historical; ξ in mol/s")
            _add(rows, reading="P2", metric=f"{scope}_max_abs_rho_xscale",
                 basis=r.get("composition_basis"), phase=r.get("phase"),
                 window_id=r.get("window_id"), value=max_x, threshold=CORR_OK,
                 pass_=bool(np.isfinite(max_x) and max_x < CORR_OK),
                 note=pair_x or "local collinearity")
            bound = str(r.get("bound_hits") or "").replace("nan", "").strip()
            _add(rows, reading="P2", metric=f"{scope}_bound_hit",
                 basis=r.get("composition_basis"), phase=r.get("phase"),
                 window_id=r.get("window_id"),
                 value=float(bool(bound)), pass_=None, note=bound)
    # P3
    if win is not None and not win.empty:
        okw = win[win["success"] == True]  # noqa: E712
        for basis in p18a.COMP_BASES:
            subb = okw[okw["composition_basis"] == basis]
            for phase_name, g in subb.groupby("phase"):
                if len(g) < 2:
                    continue
                for p, s in (("kappa", "sigma_kappa"), ("xi_mol_s", "sigma_xi"),
                             ("ln_f_gas", "sigma_ln_f_gas"), ("f_gas", None)):
                    vals = g[p].astype(float)
                    scatter = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                    sig = float(g[s].astype(float).median()) if s else float("nan")
                    _add(rows, reading="P3", metric=f"within_phase_std::{p}",
                         basis=basis, phase=phase_name, value=scatter,
                         threshold=sig,
                         pass_=bool(np.isfinite(sig) and scatter <= sig) if s else None,
                         note=f"median curvature 1σ={sig:.4g}; n={len(g)}; (JᵀJ)⁻¹ 是局部曲率，不是统计 1σ")
        for wid, g in okw.groupby("window_id"):
            if set(g["composition_basis"]) < {"analyzer", "liu_phase"}:
                continue
            a = g[g["composition_basis"] == "analyzer"].iloc[0]
            b = g[g["composition_basis"] == "liu_phase"].iloc[0]
            for p, s in (("kappa", "sigma_kappa"), ("xi_mol_s", "sigma_xi"),
                         ("ln_f_gas", "sigma_ln_f_gas")):
                d = abs(float(a[p]) - float(b[p]))
                sa, sb = float(a[s]), float(b[s])
                _add(rows, reading="P3", metric=f"basis_diff::{p}",
                     window_id=wid, value=d,
                     threshold=max(sa, sb) if np.isfinite(sa) and np.isfinite(sb) else float("nan"),
                     pass_=bool(d <= sa or d <= sb),
                     note=f"σ_an={sa:.4g}; σ_liu={sb:.4g}")
        if phase is not None and not phase.empty:
            okp = phase[phase["success"] == True]  # noqa: E712
            for basis in p18a.COMP_BASES:
                sub = okp[okp["composition_basis"] == basis]
                for p in ("kappa", "xi_mol_s", "f_gas"):
                    if sub.empty:
                        continue
                    _add(rows, reading="P3", metric=f"phase_median::{p}",
                         basis=basis, value=float(sub[p].median()),
                         note=f"range {float(sub[p].min()):.4g} … {float(sub[p].max()):.4g}; n={len(sub)}")
    # P4
    if win is not None and not win.empty:
        okw = win[win["success"] == True]  # noqa: E712
        bh = okw["bound_hits"].fillna("").astype(str).replace({"nan": "", "None": ""}) if "bound_hits" in okw.columns else pd.Series(dtype=str)
        n_bound = int(bh.str.len().gt(0).sum()) if len(bh) else 0
        _add(rows, reading="P4", metric="n_bound_hit_windows",
             value=float(n_bound), threshold=float(BOUND_WINDOW_FLAG),
             pass_=None, note=">6 窗贴边界则判读单列；不停止")
        if "needs_quench_water" in okw.columns:
            nq = int(okw["needs_quench_water"].astype(bool).sum())
            _add(rows, reading="P4", metric="n_needs_quench_water",
                 value=float(nq),
                 note=f"/{len(okw)}；T_eq 用反应器出口 H2O，假定激冷水恰好补 ξ mol")
        _add(rows, reading="P4", metric="T_eq_is_proxy",
             value=1.0, pass_=None,
             note="表观平衡温度是该水量假设下的代理量，不是测得的激冷后平衡温度")
        for col, metric in (("phys_kappa_ok", "n_kappa_in_phys"),
                            ("phys_fgas_ok", "n_fgas_in_phys"),
                            ("phys_Teq_ok", "n_Teq_in_phys")):
            if col in okw.columns:
                _add(rows, reading="P4", metric=metric,
                     value=float(okw[col].astype(bool).sum()),
                     note=f"/{len(okw)} 收敛窗（两口径合计）")
    # P5
    if robust is not None and not robust.empty and phase is not None and not phase.empty:
        okp = phase[phase["success"] == True]  # noqa: E712
        okr = robust[robust["success"] == True]  # noqa: E712
        for _, r in okr.iterrows():
            nom = okp[(okp["composition_basis"] == r["composition_basis"])
                      & (okp["phase"] == r["phase"])]
            if nom.empty:
                continue
            n = nom.iloc[0]
            for p, s in (("kappa", "sigma_kappa"), ("xi_mol_s", "sigma_xi"),
                         ("ln_f_gas", "sigma_ln_f_gas")):
                d = abs(float(r[p]) - float(n[p]))
                sig = float(n[s])
                ratio = d / sig if sig > 0 and np.isfinite(sig) else float("nan")
                _add(rows, reading="P5", metric=f"d_over_sigma::{p}",
                     basis=r["composition_basis"], phase=r["phase"],
                     value=ratio, note=f"L_wsr={r['L_wsr']} HL={r['HL_tot_pct']}; Δ={d:.4g}; σ_nom={sig:.4g}")
    # P6
    if starts is not None and not starts.empty:
        n_nu = 0
        if win is not None and "nonunique" in win.columns:
            n_nu = int(win["nonunique"].astype(bool).sum())
        _add(rows, reading="P6", metric="n_nonunique_window_fits",
             value=float(n_nu), pass_=None,
             note="代价差<1% 且至少一参数相差超过各自 1σ")
        n_conv = int(starts["success"].astype(bool).sum()) if "success" in starts.columns else 0
        _add(rows, reading="P6", metric="n_converged_starts",
             value=float(n_conv), note=f"/{len(starts)} 起点")
    # P7
    if loo is not None and not loo.empty:
        for basis in p18a.COMP_BASES:
            sub = loo[loo["composition_basis"] == basis]
            for method, g in sub.groupby("method"):
                rms = g["rms_scaled"].astype(float)
                _add(rows, reading="P7", metric=f"median_test_rms::{method}",
                     basis=basis, value=float(rms.median()) if len(rms) else float("nan"),
                     note=("n={0}; 只报告。同阶段跨窗转移；"
                           "刘文口径阶段内组成目标相同，不是独立组成预测验证".format(len(g))))
                _add(rows, reading="P7", metric=f"max_test_rms::{method}",
                     basis=basis, value=float(rms.max()) if len(rms) else float("nan"))
    # 失败
    if win is not None and not win.empty:
        n_fail = int((win["success"] != True).sum())  # noqa: E712
        _add(rows, reading="stop", metric="n_failed_window_fits",
             value=float(n_fail), threshold=float(FAIL_WINDOW_STOP),
             pass_=bool(n_fail <= FAIL_WINDOW_STOP),
             note=">3 窗失败则停止后续阶段/稳健性")
    return pd.DataFrame(rows)


def interpretation(win, phase, robust, loo, starts, rd, stopped: bool) -> str:
    L = ["# P1.13-A 试点非线性反演判读", ""]
    L.append("结构冻结 `wsr_plus_pfr`，`L_wsr = 1 m`（名义），`dz_down = 0.0515 m`，"
             "`total_percent` `HL_tot = 1%`，`OxygenMixingModel = instant`。"
             "反演参数：κ（入口 O2 乘数，整体数据协调，不归因到仪表）、"
             "ξ（反应器后变换，不进求解器）、ln `f_gas`。"
             "观测：装置 X_C（实测载气、标定合成气、协调煤量主口径）与干基 CO/H2/CO2；"
             f"σ = {SIGMAS.tolist()}（预声明试点权重，不是测量不确定度）。"
             f"算法：`least_squares` / `trf`，`x_scale` = {X_SCALE.tolist()}，"
             f"有限差分相对步长 {DIFF_REL}（ξ 步长外底 {STEP_FLOOR[1]} mol/s）。"
             "P2 的无量纲尺度门槛是复审后的局部诊断，不是预声明读法。"
             "(JᵀJ)⁻¹ 是局部曲率诊断。本轮**不采纳**任何参数值。")
    L.append("")
    if stopped:
        L.append("**停止条件触发**：逐窗失败数超过 3，未跑阶段拟合与稳健性。")
        L.append("")

    def _sel(reading, metric_contains=None):
        sub = rd[rd["reading"] == reading]
        if metric_contains:
            sub = sub[sub["metric"].astype(str).str.contains(metric_contains, regex=False)]
        return sub

    L.append("## P1 拟合质量")
    L.append("σ 是预声明试点权重，不是测量不确定度；RMS≤1 记为达到预设拟合容差。")
    for _, r in _sel("P1").iterrows():
        flag = "" if r["pass"] is None or (isinstance(r["pass"], float) and not np.isfinite(r["pass"])) else (
            " **达到预设拟合容差**" if bool(r["pass"]) else " **未达预设拟合容差**")
        L.append(f"- `{r['metric']}` {r.get('composition_basis','')} = {r['value']}"
                 f"{flag}. {r.get('note','')}")
    L.append("")
    L.append("## P2 局部条件数诊断（复审后修订，不是预声明读法）")
    L.append("单点雅可比不能回答全局唯一性。下列「通过局部门槛」指参数列按优化器 "
             f"`x_scale`={X_SCALE.tolist()} 无量纲化后 cond(J@S) < {COND_OK} "
             f"且全部 |ρ| < {CORR_OK}，只是局部曲率与共线筛查。"
             "预声明历史口径是原生单位 cond(J)（ξ 以 mol/s），结果 0/26，不推广。"
             "局部共线、贴边界与该局部门槛分开列。")
    p2x = _sel("P2", "local_threshold_xscale")
    p2n = _sel("P2", "historical_threshold_native")
    n_x = int(p2x["pass"].astype(bool).sum()) if not p2x.empty else 0
    n_n = int(p2n["pass"].astype(bool).sum()) if not p2n.empty else 0
    wx = p2x[p2x["metric"].astype(str).str.startswith("window_")] if not p2x.empty else p2x
    px = p2x[p2x["metric"].astype(str).str.startswith("phase_")] if not p2x.empty else p2x
    L.append(f"- 通过所选尺度下的局部条件数与相关门槛：{n_x} / {len(p2x)}"
             f"（逐窗 {int(wx['pass'].astype(bool).sum()) if not wx.empty else 0}/{len(wx)}"
             f"；逐阶段 {int(px['pass'].astype(bool).sum()) if not px.empty else 0}/{len(px)}）")
    L.append(f"- 历史预声明口径（原生单位）：{n_n} / {len(p2n)}（不推广）")
    p2rho = _sel("P2", "max_abs_rho_xscale")
    if not p2rho.empty:
        n_col = int(p2rho["pass"].eq(False).sum())
        L.append(f"- 局部共线（无量纲 |ρ|≥{CORR_OK}）：{n_col} / {len(p2rho)}")
    p2b = _sel("P2", "bound_hit")
    if not p2b.empty:
        n_b = int(pd.to_numeric(p2b["value"], errors="coerce").fillna(0).sum())
        L.append(f"- 贴边界：{n_b} / {len(p2b)}（单独列；边界解不并入局部门槛，也不能当作全局可辨识）")
    for _, r in p2x.iterrows():
        if not bool(r["pass"]):
            L.append(f"  - 未过局部门槛 {r.get('composition_basis')} {r.get('phase')} "
                     f"`{_short(str(r.get('window_id','')))}`：{r.get('note')}")
    L.append("")
    L.append("## P3 一致性")
    for _, r in _sel("P3").iterrows():
        L.append(f"- `{r['metric']}` {r.get('composition_basis','')} {r.get('phase','')} "
                 f"{_short(str(r.get('window_id','')))} = {r['value']} "
                 f"(threshold {r['threshold']}) {r.get('note','')}")
    L.append("")
    L.append("## P4 物理合理性（只标记，不判通过）")
    for _, r in _sel("P4").iterrows():
        L.append(f"- `{r['metric']}` = {r['value']}. {r.get('note','')}")
    if win is not None and not win.empty:
        okw = win[win["success"] == True]  # noqa: E712
        bound = okw[okw.get("bound_hits", pd.Series(dtype=str)).fillna("").astype(str).str.len() > 0] if "bound_hits" in okw.columns else okw.iloc[0:0]
        if len(bound) > BOUND_WINDOW_FLAG:
            L.append(f"- **贴边界单列**：{len(bound)} 窗（>{BOUND_WINDOW_FLAG}）。")
        for _, r in bound.iterrows():
            L.append(f"  - `{_short(str(r['window_id']))}` {r.get('composition_basis')} "
                     f"hits `{r.get('bound_hits')}` κ={r.get('kappa'):.4g} ξ={r.get('xi_mol_s'):.3g} "
                     f"f_gas={r.get('f_gas'):.4g}")
    L.append("")
    L.append("## P5 稳健性")
    L.append("逐阶段拟合在 `L_wsr=4 m`、`HL_tot` ∈ {0.5, 1.5}% 下重做，报告 Δθ / 1σ。")
    p5 = _sel("P5")
    if p5.empty:
        L.append("- （无稳健性表）")
    else:
        for _, r in p5.iterrows():
            L.append(f"- `{r['metric']}` {r.get('composition_basis')} {r.get('phase')} "
                     f"= {r['value']:.4g}. {r.get('note','')}")
    L.append("")
    L.append("## P6 多起点")
    for _, r in _sel("P6").iterrows():
        L.append(f"- `{r['metric']}` = {r['value']}. {r.get('note','')}")
    L.append("")
    L.append("## P7 留一窗预测（只报告）")
    L.append("对照：(a) 不修正 θ=(1,0,1)；(b) 训练窗常数残差补丁。"
             "同阶段跨窗转移；刘文口径在阶段内组成目标相同，不作独立组成预测验证。")
    for _, r in _sel("P7", "median_test_rms").iterrows():
        L.append(f"- `{r['metric']}` {r.get('composition_basis')} 中位测试 rms = {r['value']}")
    L.append("")
    L.append("## 必须回答")
    L.append("1. 每次拟合的 θ̂±曲率 1σ、相关、条件数与缩放残差见 `fits_window.csv` / `fits_phase.csv`；"
             "全部起点见 `multistart.csv`；加权残差雅可比见 `jacobians.csv`（含 `start_name`/`selected`）。"
             "历史口径条件数为 `cond_J`；复审后局部诊断为 `cond_J_xscale`。")
    L.append("2. P1–P7 见上。")
    L.append("3. 求解失败、贴边界、非唯一解见 `multistart.csv` 的 `success` / `bound_hits` "
             "与 fits 表的 `nonunique`。")
    L.append("")
    L.append("本轮不采纳 κ、ξ、`f_gas` 的任何数值，不改合同。")
    L.append("")
    return "\n".join(L) + "\n"


def _to_df(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    drop = {"J", "r", "theta", "sig", "corr", "traceback"}
    clean = [{k: v for k, v in r.items() if k not in drop} for r in rows]
    return pd.DataFrame(clean)


def _rewrite_manifest(out_dir: Path) -> None:
    files = sorted(p for p in out_dir.iterdir() if p.is_file() and p.name != "sha256.txt")
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}" for p in files]
    (out_dir / "sha256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _window_jobs(window_ids, bases=None):
    bases = bases or list(p18a.COMP_BASES)
    jobs = []
    for wid in window_ids:
        for basis in bases:
            jobs.append({
                "wids": [wid], "basis": basis, "l_wsr": L_WSR_NOM, "hl": HL_NOM,
                "scope": "window", "label": wid,
            })
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=("all", "window", "phase", "robust", "loo", "read", "smoke"))
    ap.add_argument("--source-commit", default=None)
    ap.add_argument("--n-workers", type=int, default=N_WORKERS_DEFAULT)
    ap.add_argument("--window-index", type=int, default=0, help="smoke 用的窗口序号")
    ap.add_argument("--basis", default="analyzer", help="smoke 用的组成口径")
    ap.add_argument("--n-starts", type=int, default=len(STARTS))
    args = ap.parse_args()
    global _N_STARTS
    _N_STARTS = max(1, min(int(args.n_starts), len(STARTS)))
    silence_solver_logs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _init_worker()
    window_ids = m1_reference_window_ids(M1_MATRIX)
    n_workers = int(args.n_workers)

    if args.stage == "read":
        win = pd.read_csv(OUT_DIR / "fits_window.csv")
        phase = pd.read_csv(OUT_DIR / "fits_phase.csv") if (OUT_DIR / "fits_phase.csv").exists() else pd.DataFrame()
        robust = pd.read_csv(OUT_DIR / "robustness.csv") if (OUT_DIR / "robustness.csv").exists() else pd.DataFrame()
        loo = pd.read_csv(OUT_DIR / "loo_prediction.csv") if (OUT_DIR / "loo_prediction.csv").exists() else pd.DataFrame()
        starts = pd.read_csv(OUT_DIR / "multistart.csv")
        jac = pd.read_csv(OUT_DIR / "jacobians.csv") if (OUT_DIR / "jacobians.csv").exists() else pd.DataFrame()
        before = (win["H_ledger_full_over_plant_h2"].to_numpy(dtype=float)
                  if "H_ledger_full_over_plant_h2" in win.columns else None)
        already_once = _ledger_is_counted_once(win)
        win, phase, robust, jac = attach_xscale_and_ledgers(win, phase, robust, jac, starts)
        if already_once and before is not None:
            after = win["H_ledger_full_over_plant_h2"].to_numpy(dtype=float)
            if not np.allclose(before, after, rtol=0, atol=1e-12):
                raise RuntimeError(
                    "attach mutated a counted-once hydrogen ledger: "
                    f"{before[0]!r} -> {after[0]!r}")
        verify_h_ledger_read_paths(win)
        stopped = False
        rd = readings(win, phase, robust, loo, starts)
        rd.to_csv(OUT_DIR / "readings.csv", index=False)
        text = interpretation(win, phase, robust, loo, starts, rd, stopped)
        protocol = _protocol(args.source_commit, stopped)
        write_round_outputs(OUT_DIR, tables={
            "fits_window.csv": win, "fits_phase.csv": phase, "multistart.csv": starts,
            "robustness.csv": robust, "loo_prediction.csv": loo, "jacobians.csv": jac,
            "readings.csv": rd,
        }, protocol=protocol, interpretation=text)
        _rewrite_manifest(OUT_DIR)
        print("written read-stage", flush=True)
        return

    if args.stage == "smoke":
        wid = window_ids[int(args.window_index)]
        jobs = [{"wids": [wid], "basis": args.basis, "l_wsr": L_WSR_NOM, "hl": HL_NOM,
                 "scope": "window", "label": wid}]
        results = _run_jobs(jobs, 1)
        print(json.dumps({k: v for k, v in results[0]["best"].items()
                          if not isinstance(v, (np.ndarray, bytes))},
                         ensure_ascii=False, default=str, indent=2))
        return

    all_starts, all_jacs = [], []
    stopped = False

    # window
    w_res = _run_jobs(_window_jobs(window_ids), n_workers)
    w_bests, w_starts, w_jacs, _pts = _collect(w_res)
    all_starts.extend(w_starts)
    all_jacs.extend(w_jacs)
    win = _to_df(w_bests)
    win.to_csv(OUT_DIR / "fits_window.csv", index=False)
    # 卡面“>3 窗失败”：某窗任一组成口径的全部起点都不收敛，即计该窗失败
    fail_wids = set()
    for wid in window_ids:
        sub = win[win["window_id"] == wid] if not win.empty else pd.DataFrame()
        if sub.empty or not bool(sub["success"].astype(bool).all()):
            fail_wids.add(wid)
    print(f"window fits: {len(win)} rows, failed windows {len(fail_wids)} {sorted(_short(w) for w in fail_wids)}",
          flush=True)
    if len(fail_wids) > FAIL_WINDOW_STOP:
        stopped = True
        print("STOP: >3 windows failed", flush=True)

    phase_df = pd.DataFrame()
    robust_df = pd.DataFrame()
    loo_df = pd.DataFrame()

    if not stopped and args.stage in ("all", "phase", "robust", "loo"):
        by_phase = _windows_by_phase(window_ids)
        p_jobs = []
        for ph, wids in by_phase.items():
            for basis in p18a.COMP_BASES:
                p_jobs.append({
                    "wids": wids, "basis": basis, "l_wsr": L_WSR_NOM, "hl": HL_NOM,
                    "scope": "phase", "label": ph,
                })
        p_res = _run_jobs(p_jobs, n_workers)
        p_bests, p_starts, p_jacs, _ppts = _collect(p_res)
        all_starts.extend(p_starts)
        all_jacs.extend(p_jacs)
        phase_df = _to_df(p_bests)
        phase_df.to_csv(OUT_DIR / "fits_phase.csv", index=False)
        print(f"phase fits: {len(phase_df)}", flush=True)

    if not stopped and args.stage in ("all", "robust"):
        by_phase = _windows_by_phase(window_ids)
        r_jobs = []
        for ph, wids in by_phase.items():
            for basis in p18a.COMP_BASES:
                for hl in HL_ROBUST:
                    r_jobs.append({
                        "wids": wids, "basis": basis, "l_wsr": L_WSR_ROBUST, "hl": hl,
                        "scope": "robust", "label": ph,
                    })
        r_res = _run_jobs(r_jobs, n_workers)
        r_bests, r_starts, r_jacs, _rpts = _collect(r_res)
        all_starts.extend(r_starts)
        all_jacs.extend(r_jacs)
        robust_df = _to_df(r_bests)
        robust_df.to_csv(OUT_DIR / "robustness.csv", index=False)
        print(f"robust fits: {len(robust_df)}", flush=True)

    if not stopped and args.stage in ("all", "loo", "window"):
        loo_df = loo_from_window_fits(win, window_ids)
        loo_df.to_csv(OUT_DIR / "loo_prediction.csv", index=False)
        print(f"loo rows: {len(loo_df)}", flush=True)

    starts_df = _to_df(all_starts)
    starts_df.to_csv(OUT_DIR / "multistart.csv", index=False)
    jac_df = _to_df(all_jacs)
    jac_df.to_csv(OUT_DIR / "jacobians.csv", index=False)

    rd = readings(win, phase_df, robust_df, loo_df, starts_df)
    rd.to_csv(OUT_DIR / "readings.csv", index=False)
    text = interpretation(win, phase_df, robust_df, loo_df, starts_df, rd, stopped)
    protocol = _protocol(args.source_commit, stopped)
    write_round_outputs(OUT_DIR, tables={
        "fits_window.csv": win, "fits_phase.csv": phase_df, "multistart.csv": starts_df,
        "robustness.csv": robust_df, "loo_prediction.csv": loo_df,
        "jacobians.csv": jac_df, "readings.csv": rd,
    }, protocol=protocol, interpretation=text)
    _rewrite_manifest(OUT_DIR)
    print("written:", sorted(p.name for p in OUT_DIR.iterdir() if p.is_file()), flush=True)


def _protocol(src_commit, stopped: bool) -> dict:
    return base_protocol({
        "round": "P1.13-A / 2026-11-07",
        "model_repo": "gasifier-1d-kinetic",
        "mesh_family": "wsr_plus_pfr",
        "source_commit": src_commit or source_commit(),
        "task_card": "papers/jax-gasifier-cascade-hybrid/manuscript/"
                     "model-side-handoff-p1.13a-20261107.md",
        "structure": {
            "mesh_family": "wsr_plus_pfr",
            "L_wsr_m": L_WSR_NOM,
            "dz_down_m": DZ_DOWN,
            "HL_tot_pct": HL_NOM,
            "oxygen_mixing": "instant",
            "frozen": ["全部 Arrhenius（除 f_gas 作用的 C+H2O/C+CO2 指前因子）",
                       "限幅", "燃烧区判据", "煤量主口径", "实测载气", "求解器容差"],
        },
        "parameters": {
            "kappa": "inlet O2 multiplier; O/C coordination factor; not attributed to a meter",
            "xi_mol_s": "post-reactor WGS extent; does not enter the reactor solve; X_C unchanged",
            "ln_f_gas": "log multiplier on shared C+H2O / C+CO2 prefactor",
            "bounds": {"kappa": list(BOUNDS_LO[0:1]) + list(BOUNDS_HI[0:1]),
                       "xi": [float(BOUNDS_LO[1]), float(BOUNDS_HI[1])],
                       "ln_f_gas": [float(BOUNDS_LO[2]), float(BOUNDS_HI[2])]},
        },
        "observations": {
            "y": list(OBS_NAMES),
            "sigma": SIGMAS.tolist(),
            "sigma_note": "predeclared pilot weights, not measurement uncertainty",
            "X_C": "xc_measured_pct from P1.8-D (measured carrier, calibrated syngas, recon coal)",
            "composition": "analyzer = P1.7-A baseline plant vol%; liu_phase = Liu 2023 table 3",
            "H_ledger": "num = 2 F_H2_shifted [+ 4 CH4 + 2 H2S]; den = 2 plant_dry y_H2; 2ξ counted once",
            "T_eq": "proxy: Q from shifted CO/H2/CO2 and reactor-outlet H2O; quench replaces ξ mol H2O",
        },
        "optimizer": {
            "algorithm": "scipy.optimize.least_squares",
            "method": "trf",
            "jacobian": "custom 2-point finite difference on scaled residuals",
            "diff_rel": DIFF_REL,
            "step_floor": {"kappa": float(STEP_FLOOR[0]), "xi_mol_s": float(STEP_FLOOR[1]),
                           "ln_f_gas": float(STEP_FLOOR[2])},
            "x_scale": X_SCALE.tolist(),
            "starts": [list(s) for s in STARTS],
            "max_nfev": MAX_NFEV,
            "covariance": "(J^T J)^{-1} in native units (local curvature); also J @ diag(x_scale)",
        },
        "predeclared": {
            "P1": "scaled RMS ≤ 1 → reaches predeclared fit tolerance (pilot weights, not measurement σ)",
            "P2": "HISTORICAL as run: cond(J)<100 and all |ρ|<0.9 in native parameter units (ξ mol/s); observation residuals scaled by σ. Result 0/26. Not generalized.",
            "P3": "within-phase scatter vs curvature 1σ; analyzer vs liu Δθ vs curvature 1σ; phase median/range",
            "P4": "flags only: κ in [0.92,1.03]; f_gas in [1/16,2]; T_eq proxy in [500,1300] K under quench-replaces-ξ water assumption; bound hits",
            "P5": "redo phase fits at L_wsr=4 m, HL_tot in {0.5,1.5}%; report Δθ/curvature 1σ",
            "P6": "non-unique if cost differs <1% but some parameter differs by > each curvature 1σ",
            "P7": "within-phase LOO vs uncorrected vs residual patch; report only; Liu targets identical across windows in a phase",
        },
        "post_hoc": {
            "P2_local_xscale": (
                "Review revision, not predeclared: cond(J @ diag(0.05,50,1))<100 and |ρ|<0.9 "
                "as a local curvature/collinearity screen. Passing this threshold is not global "
                "identifiability. Bound solutions listed separately."
            ),
            "H_ledger_schema": LEDGER_SCHEMA_ONCE,
        },
        "stopped": stopped,
        "source_csv": {
            "p17d": {"round": "p1.7d-20261103", "file": P17D_CSV.name,
                     "sha256": hashlib.sha256(P17D_CSV.read_bytes()).hexdigest()},
            "p18d": {"round": "p1.8d-20261104", "file": P18D_CSV.name,
                     "sha256": hashlib.sha256(P18D_CSV.read_bytes()).hexdigest()},
            "p17a_baseline": {"round": "p1.7a-20261103", "file": P17A_BASE.name,
                              "sha256": hashlib.sha256(P17A_BASE.read_bytes()).hexdigest()},
        },
        "outputs": ["fits_window.csv", "fits_phase.csv", "multistart.csv", "robustness.csv",
                    "loo_prediction.csv", "jacobians.csv", "readings.csv",
                    "protocol.json", "interpretation.md", "sha256.txt"],
    })


if __name__ == "__main__":
    main()
