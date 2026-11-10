#!/usr/bin/env python3
"""P1.14-A 可辨识性设计：寻找分开 κ 与 f_gas 的观测量（2026-11-07 卡）。

冻结与 P1.13-A 相同：wsr_plus_pfr、L_wsr=1 m、dz_down=0.0515 m、total_percent
HL_tot=1%、OxygenMixingModel=instant、全部 Arrhenius（除 f_gas 作用的 C+H2O/C+CO2
指前因子）。ξ 只作用于出口组成，不进求解器。

本轮不做反演、不采纳任何参数，只评估观测设计。

用法：
    ./.venv/bin/python scripts/generate_p1_14a_20261107.py [--stage all|smoke|read]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import generate_p1_13a_20261107 as p113a  # noqa: E402
import generate_p1_8a_20261104 as p18a  # noqa: E402

from model.p0_round_20260930 import m1_reference_window_ids  # noqa: E402
from model.p1_5_round_20261030 import (  # noqa: E402
    base_protocol,
    run_case,
    silence_solver_logs,
    write_round_outputs,
)
from model.revision_evidence_common import load_frozen_split, source_commit  # noqa: E402

OUT_DIR = ROOT / "outputs" / "diagnostics" / "round-p1.14a-20261107"
P114D_CSV = ROOT / "outputs" / "diagnostics" / "p1.14d-20261107" / "windows_142.csv"
M1_MATRIX = p18a.M1_MATRIX

DEV_L = p113a.DEV_L
DZ_DOWN = p113a.DZ_DOWN
L_WSR_NOM = p113a.L_WSR_NOM
HL_NOM = p113a.HL_NOM
DIFF_REL = p113a.DIFF_REL
N_WORKERS_DEFAULT = 15
FAIL_WINDOW_STOP = 10
D_WALL_M = 3.8
T_W_DEFAULT_K = 537.0

# 阶段 θ：P1.13-A 分析仪口径、L_wsr=1 m、HL_tot=1%（fits_phase.csv）
PHASE_THETA = {
    "phase1": {"kappa": 0.9636054551111352, "xi": 104.81669676840302,
               "f_gas": 0.6531139633486652},
    "phase2": {"kappa": 0.8998492036049609, "xi": 86.5403359100365,
               "f_gas": 0.3965369477333102},
    "phase3": {"kappa": 0.9238312741474483, "xi": 82.14896736151056,
               "f_gas": 0.10788183495180284},
    "phase4": {"kappa": 0.949661873527387, "xi": 56.13939397745541,
               "f_gas": 0.1776065048633667},
}
THETA0 = {"kappa": 1.0, "xi": 0.0, "f_gas": 1.0}

PHASE_MAP = {"P1": "phase1", "P2": "phase2", "P3": "phase3", "P4": "phase4"}

# 观测顺序与预声明缩放（跑完不改）
OBS_NAMES = (
    "X_C_pp", "CO_vol_pct", "H2_vol_pct", "CO2_vol_pct", "CH4_vol_pct",
    "dry_per_coal_mol_kg", "T_out_K", "Q_wall_conv", "Q_wall_rad",
)
# 加性 σ；相对/对数在 scale_jac 里单独处理
SIGMA_ADD = {
    "X_C_pp": 1.5,
    "CO_vol_pct": 0.5,
    "H2_vol_pct": 0.5,
    "CO2_vol_pct": 0.5,
    "CH4_vol_pct": 0.05,
    "T_out_K": 20.0,  # 理想设计假设，不是装置出口温测量不确定度
}
SIGMA_REL = {"dry_per_coal_mol_kg": 0.02}  # 2%
SIGMA_LOG = {"Q_wall_conv": 0.10, "Q_wall_rad": 0.10}  # 10% 对数
PARAM_NAMES = ("kappa", "ln_f_gas", "xi_mol_s")  # 卡面顺序
# 与 P1.13 一致：相对步长 1e-4，ξ 外底 10 mol/s；κ / ln f_gas 外底 1
STEP_FLOOR = np.array([1.0, 1.0, 10.0])  # κ, ln f_gas, ξ
LN2 = math.log(2.0)
RHO_OK = 0.9

SUBSETS = {
    "S0": ("X_C_pp", "CO_vol_pct", "H2_vol_pct", "CO2_vol_pct"),
    "S1": ("X_C_pp", "CO_vol_pct", "H2_vol_pct", "CO2_vol_pct", "CH4_vol_pct"),
    "S2": ("X_C_pp", "CO_vol_pct", "H2_vol_pct", "CO2_vol_pct", "dry_per_coal_mol_kg"),
    "S3_conv": ("X_C_pp", "CO_vol_pct", "H2_vol_pct", "CO2_vol_pct", "Q_wall_conv"),
    "S3_rad": ("X_C_pp", "CO_vol_pct", "H2_vol_pct", "CO2_vol_pct", "Q_wall_rad"),
    "S4_conv": (
        "X_C_pp", "CO_vol_pct", "H2_vol_pct", "CO2_vol_pct", "CH4_vol_pct",
        "dry_per_coal_mol_kg", "T_out_K", "Q_wall_conv",
    ),
    "S4_rad": (
        "X_C_pp", "CO_vol_pct", "H2_vol_pct", "CO2_vol_pct", "CH4_vol_pct",
        "dry_per_coal_mol_kg", "T_out_K", "Q_wall_rad",
    ),
    "S4_no_wall": (
        "X_C_pp", "CO_vol_pct", "H2_vol_pct", "CO2_vol_pct", "CH4_vol_pct",
        "dry_per_coal_mol_kg", "T_out_K",
    ),
}
WALL_OBS = ("Q_wall_conv", "Q_wall_rad")
WALL_SUBSETS = frozenset({"S3_conv", "S3_rad", "S4_conv", "S4_rad"})
EXPLORATORY_SUBSETS = frozenset({"S4_no_wall"})
Q2_COMPARE_SUBSETS = ("S1", "S2", "S3_conv", "S3_rad")
Q3_NOTE = (
    "扩窗同时改变 n 与工况；总尺度下降不能归因于 O/C；"
    "参考窗 S0 已过 Q1 局部门槛"
)
P34_EXPLORATORY_NOTE = "超出训练窗主分析；阶段 θ 已用同阶段观测，不是独立验证"
S4_NO_WALL_NOTE = "探索变体，不是卡面 S4；无壁面、无 ln U"
Q1_LOCAL_DESIGN = (
    "逆 Fisher 是固定阶段 θ、共享 κ/ln f_gas/ξ、对角预声明权重、窗间独立假设下的曲率尺度。"
    "装置 X_C 与 CO/CO2/CH4 及校准干气流量共享原始数据，阶段内还共享标定系数。"
    "通过的是局部设计门槛，不是统计可辨识。"
)

# P1.14-D 第 2 问（不重算）
P114D_Q2 = {
    "phase1": {"rho_oc": 0.254, "rho_coal": -0.290, "cv_ratio": 1.425},
    "phase2": {"rho_oc": 0.586, "rho_coal": -0.527, "cv_ratio": 1.677},
    "phase3": {"rho_oc": 0.478, "rho_coal": -0.229, "cv_ratio": 1.271},
}

_CTX: dict = {}
_CACHE: dict[tuple, dict] = {}


def _short(wid: str) -> str:
    return wid[-16:]


def _phase_of_row(phase_raw) -> str:
    raw = str(phase_raw)
    if raw in PHASE_MAP:
        return PHASE_MAP[raw]
    if raw.startswith("phase"):
        return raw
    return f"phase{raw[-1]}"


def _t_w_k(wid: str) -> float:
    row = _CTX["p114d"].loc[wid]
    tsat = row["T_sat_iapws_C"]
    if pd.isna(tsat):
        return T_W_DEFAULT_K
    return float(tsat) + 273.15


def _coal_dry(wid: str) -> float:
    return float(_CTX["p114d"].loc[wid, "coal_dry_recon_phase_kg_s"])


def _carrier_nm3(wid: str) -> float:
    return float(_CTX["p114d"].loc[wid, "co2_carrier_measured_nm3_h"])


def wall_indices(system, T_w: float) -> dict:
    D = float(system.geometry.get("D", D_WALL_M))
    n = min(len(getattr(system, "cells", []) or []), len(getattr(system, "results", []) or []))
    q_conv = 0.0
    q_rad = 0.0
    for i in range(n):
        dz = float(system.cells[i].dz)
        T = float(system.results[i].T)
        area = math.pi * D * dz
        q_conv += area * (T - T_w)
        q_rad += area * (T ** 4 - T_w ** 4)
    return {
        "Q_wall_conv": float(q_conv),
        "Q_wall_rad": float(q_rad),
        "D_wall_m": float(D),
        "n_wall_cells": int(n),
        "T_w_K": float(T_w),
    }


def _run_reactor(wid: str, kappa: float, f_gas: float) -> dict:
    key = (wid, round(float(kappa), 8), round(float(f_gas), 10))
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    row = _CTX["by_id"][wid]
    dry = _coal_dry(wid)
    op = {
        "HeatLossModel": "total_percent",
        "HeatLossPercentTotal": float(HL_NOM),
        "GasificationRateFactor": float(f_gas),
        "carrier_gas_flow": p18a._carrier_kg_s(_carrier_nm3(wid)),
        "MeshFamily": "wsr_plus_pfr",
        "WSRLength_m": float(L_WSR_NOM),
        "PFRCellLength_m": float(DZ_DOWN),
        "OxygenMixingModel": "instant",
        "o2_flow_scale": float(kappa),
    }
    print(f"    solve {_short(wid)} κ={kappa:.6f} f_gas={f_gas:.6g}", flush=True)
    rec = run_case(
        row, n_cells=p113a.mesh_n(L_WSR_NOM), coal_flow_dry_kg_s=dry,
        keep_system=True, mesh_family="wsr_plus_pfr", extra_op=op,
    )
    system = rec.get("_system")
    snap: dict = {
        "window_id": wid,
        "kappa": float(kappa),
        "f_gas": float(f_gas),
        "L_wsr": float(L_WSR_NOM),
        "HL_tot_pct": float(HL_NOM),
        "execution_status": str(rec.get("execution_status")),
        "no_root_cell_index": rec.get("no_root_cell_index"),
        "cell0_n_distinct_roots": rec.get("cell0_n_distinct_roots"),
        "solve_wall_time_s": rec.get("solve_wall_time_s"),
        "n_cells": rec.get("N_cells"),
        "ok": rec.get("execution_status") == "completed" and bool(getattr(system, "results", None)),
    }
    if snap["ok"] and system is not None:
        gas = np.asarray(system.results[-1].gas_moles, dtype=float)
        wall = wall_indices(system, _t_w_k(wid))
        snap.update({
            "X_C_pp": float(rec["X_C_implied"]) * 100.0,
            "T_out_K": float(rec["T_out_K"]),
            "F_CO": float(gas[2]), "F_CO2": float(gas[3]), "F_H2": float(gas[5]),
            "F_H2O": float(gas[7]), "F_CH4": float(gas[1]), "F_H2S": float(gas[4]),
            "dry_total_mol_s": float(np.sum(gas[:7])),
            **wall,
        })
    rec.pop("_system", None)
    rec.pop("_profile", None)
    _CACHE[key] = snap
    return snap


def observables(snap: dict, xi: float, coal_dry: float) -> np.ndarray | None:
    shifted = p113a.apply_xi(snap, xi)
    if not shifted.get("ok"):
        return None
    dry = float(shifted["dry_total_mol_s"])
    ch4 = float(snap["F_CH4"]) / max(dry, 1e-12) * 100.0
    return np.array([
        float(shifted["X_C_pp"]),
        float(shifted["CO_vol_pct"]),
        float(shifted["H2_vol_pct"]),
        float(shifted["CO2_vol_pct"]),
        ch4,
        dry / max(float(coal_dry), 1e-12),
        float(snap["T_out_K"]),
        float(snap["Q_wall_conv"]),
        float(snap["Q_wall_rad"]),
    ], dtype=float)


def fd_steps(theta: np.ndarray) -> np.ndarray:
    return DIFF_REL * np.maximum(np.abs(theta), STEP_FLOOR)


def theta_vec(kappa: float, f_gas: float, xi: float) -> np.ndarray:
    return np.array([float(kappa), math.log(float(f_gas)), float(xi)], dtype=float)


def predict_at(wid: str, theta: np.ndarray) -> tuple[np.ndarray | None, dict]:
    kappa, ln_fg, xi = [float(v) for v in theta]
    snap = _run_reactor(wid, kappa, math.exp(ln_fg))
    if not snap.get("ok"):
        return None, snap
    y = observables(snap, xi, _coal_dry(wid))
    return y, snap


def window_jacobian(wid: str, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    y0, snap0 = predict_at(wid, theta)
    if y0 is None or not np.all(np.isfinite(y0)):
        nanJ = np.full((len(OBS_NAMES), 3), np.nan)
        nany = np.full(len(OBS_NAMES), np.nan)
        return nanJ, nany, snap0
    J = np.zeros((len(OBS_NAMES), 3), dtype=float)
    steps = fd_steps(theta)
    for j in range(3):
        tp = theta.copy()
        h = float(steps[j])
        tp[j] = theta[j] + h
        h_eff = float(tp[j] - theta[j])
        if abs(h_eff) < 1e-16:
            J[:, j] = np.nan
            continue
        y1, snap1 = predict_at(wid, tp)
        if y1 is None or not np.all(np.isfinite(y1)) or not snap1.get("ok"):
            J[:, j] = np.nan
            continue
        J[:, j] = (y1 - y0) / h_eff
    return J, y0, snap0


def plant_obs(wid: str) -> dict:
    r = _CTX["p114d"].loc[wid]
    dry = float(r["coal_dry_recon_phase_kg_s"])
    plant_dry_mol = float(r["syngas_dry_cal_nm3_h"]) / 22.414 / 3600.0 * 1000.0
    return {
        "plant_X_C_pp": float(r["xc_recon_phase_cal_measured_pct"]),
        "plant_CO_vol_pct": float(r["co_vol_pct_mean"]),
        "plant_H2_vol_pct": float(r["h2_vol_pct_mean"]),
        "plant_CO2_vol_pct": float(r["co2_vol_pct_mean"]),
        "plant_CH4_vol_pct": float(r["ch4_vol_pct_mean"]),
        "plant_dry_per_coal_mol_kg": plant_dry_mol / max(dry, 1e-12),
        "plant_syngas_dry_cal_nm3_h": float(r["syngas_dry_cal_nm3_h"]),
    }


def _obs_row(y: np.ndarray | None, prefix: str) -> dict:
    out = {}
    for i, name in enumerate(OBS_NAMES):
        out[f"{prefix}{name}"] = float(y[i]) if y is not None and np.isfinite(y[i]) else float("nan")
    return out


def process_window(wid: str) -> dict:
    if "by_id" not in _CTX:
        _init_worker()
    row = _CTX["p114d"].loc[wid]
    phase = _phase_of_row(row["phase"])
    role = str(row["split_role"])
    th = PHASE_THETA[phase]
    theta = theta_vec(th["kappa"], th["f_gas"], th["xi"])
    theta0 = theta_vec(THETA0["kappa"], THETA0["f_gas"], THETA0["xi"])
    print(f"WINDOW {_short(wid)} {phase} {role}", flush=True)
    y0, snap0 = predict_at(wid, theta0)
    J, y_th, snap_th = window_jacobian(wid, theta)
    ok0 = bool(snap0.get("ok")) and y0 is not None and np.all(np.isfinite(y0))
    ok_th = bool(snap_th.get("ok")) and y_th is not None and np.all(np.isfinite(y_th))
    jac_ok = ok_th and np.all(np.isfinite(J))
    plant = plant_obs(wid)
    base = {
        "window_id": wid,
        "phase": phase,
        "phase_raw": str(row["phase"]),
        "split_role": role,
        "coal_dry_recon_phase_kg_s": _coal_dry(wid),
        "co2_carrier_measured_nm3_h": _carrier_nm3(wid),
        "T_w_K": _t_w_k(wid),
        "T_w_source": "T_sat_iapws_C+273.15" if pd.notna(row["T_sat_iapws_C"]) else "default_537K",
        "ok_theta0": ok0,
        "ok_theta": ok_th,
        "ok_jacobian": jac_ok,
        "execution_status_theta0": snap0.get("execution_status"),
        "execution_status_theta": snap_th.get("execution_status"),
        "solve_wall_s_theta0": snap0.get("solve_wall_time_s"),
        "solve_wall_s_theta": snap_th.get("solve_wall_time_s"),
        "n_cells": snap_th.get("n_cells", snap0.get("n_cells")),
        "D_wall_m": snap_th.get("D_wall_m", snap0.get("D_wall_m", D_WALL_M)),
        "n_wall_cells": snap_th.get("n_wall_cells", snap0.get("n_wall_cells")),
        "kappa_phase": th["kappa"],
        "xi_phase": th["xi"],
        "f_gas_phase": th["f_gas"],
        "ln_f_gas_phase": math.log(th["f_gas"]),
        **plant,
        **_obs_row(y0, "theta0_"),
        **_obs_row(y_th, "theta_"),
    }
    if ok0 and ok_th:
        for i, name in enumerate(OBS_NAMES[:5]):
            base[f"d_{name}_theta_minus_plant"] = float(y_th[i] - plant[f"plant_{name}" if name != "X_C_pp" else "plant_X_C_pp"]) if name != "X_C_pp" else float(y_th[0] - plant["plant_X_C_pp"])
        base["d_X_C_pp_theta_minus_plant"] = float(y_th[0] - plant["plant_X_C_pp"])
        base["d_CO_vol_pct_theta_minus_plant"] = float(y_th[1] - plant["plant_CO_vol_pct"])
        base["d_H2_vol_pct_theta_minus_plant"] = float(y_th[2] - plant["plant_H2_vol_pct"])
        base["d_CO2_vol_pct_theta_minus_plant"] = float(y_th[3] - plant["plant_CO2_vol_pct"])
        base["d_CH4_vol_pct_theta_minus_plant"] = float(y_th[4] - plant["plant_CH4_vol_pct"])
        base["d_X_C_pp_theta0_minus_plant"] = float(y0[0] - plant["plant_X_C_pp"])
        base["d_CO_vol_pct_theta0_minus_plant"] = float(y0[1] - plant["plant_CO_vol_pct"])
        base["d_H2_vol_pct_theta0_minus_plant"] = float(y0[2] - plant["plant_H2_vol_pct"])
        base["d_CO2_vol_pct_theta0_minus_plant"] = float(y0[3] - plant["plant_CO2_vol_pct"])
        base["d_CH4_vol_pct_theta0_minus_plant"] = float(y0[4] - plant["plant_CH4_vol_pct"])
    jac_rows = []
    if ok_th:
        for i, obs in enumerate(OBS_NAMES):
            rec = {
                "window_id": wid, "phase": phase, "split_role": role,
                "obs": obs, "obs_index": i,
                "y_theta": float(y_th[i]) if y_th is not None else float("nan"),
                "d_kappa": float(J[i, 0]),
                "d_ln_f_gas": float(J[i, 1]),
                "d_xi_mol_s": float(J[i, 2]),
            }
            jac_rows.append(rec)
        i_t = OBS_NAMES.index("T_out_K")
        base["dT_out_d_kappa"] = float(J[i_t, 0])
        base["dT_out_d_ln_f_gas"] = float(J[i_t, 1])
        s_k = np.sign(J[i_t, 0]) if np.isfinite(J[i_t, 0]) and J[i_t, 0] != 0 else 0.0
        s_f = np.sign(J[i_t, 1]) if np.isfinite(J[i_t, 1]) and J[i_t, 1] != 0 else 0.0
        base["dT_out_same_sign"] = bool(s_k == s_f and s_k != 0)
    print(
        f"  done {_short(wid)} ok0={ok0} okθ={ok_th} jac={jac_ok} "
        f"t0={snap0.get('solve_wall_time_s')} tθ={snap_th.get('solve_wall_time_s')}",
        flush=True,
    )
    return {"baseline": base, "jac_rows": jac_rows, "ok": ok_th and jac_ok}


def _init_worker():
    silence_solver_logs()
    ok, rows_, _inv = load_frozen_split()
    if not ok:
        raise RuntimeError("frozen R52-C-75 split was not consumed intact")
    by_id = {r["window_id"]: r for r in rows_}
    p114d = pd.read_csv(P114D_CSV).set_index("window_id")
    csv_ids = set(p114d.index.astype(str))
    frz_ids = set(by_id)
    if csv_ids != frz_ids:
        raise RuntimeError(
            f"windows_142 ids do not match frozen split: "
            f"csv-frz={len(csv_ids - frz_ids)} frz-csv={len(frz_ids - csv_ids)}"
        )
    _CTX.update({"by_id": by_id, "p114d": p114d})
    _CACHE.clear()
    p113a._CTX.update({"by_id": by_id, "p114d": p114d})
    p113a._CACHE.clear()


def _job(wid: str) -> dict:
    try:
        return process_window(wid)
    except Exception as exc:  # noqa: BLE001
        print(f"JOB FAIL {_short(wid)}: {exc}", flush=True)
        traceback.print_exc()
        return {
            "baseline": {
                "window_id": wid, "ok_theta": False, "ok_theta0": False,
                "ok_jacobian": False, "error": f"{type(exc).__name__}: {exc}",
            },
            "jac_rows": [], "ok": False,
        }


def scale_jac(J_raw: np.ndarray, y: np.ndarray, obs_names: tuple[str, ...]) -> np.ndarray:
    """把未缩放 ∂y/∂θ 变成加权残差雅可比（加性 / 相对 / 对数）。"""
    idx = [OBS_NAMES.index(n) for n in obs_names]
    Js = np.zeros((len(idx), J_raw.shape[1]), dtype=float)
    for k, (i, name) in enumerate(zip(idx, obs_names)):
        dy = J_raw[i, :]
        if name in SIGMA_ADD:
            Js[k, :] = dy / SIGMA_ADD[name]
        elif name in SIGMA_REL:
            scale = SIGMA_REL[name] * max(abs(float(y[i])), 1e-12)
            Js[k, :] = dy / scale
        elif name in SIGMA_LOG:
            q = float(y[i])
            if not np.isfinite(q) or q <= 0:
                Js[k, :] = np.nan
            else:
                Js[k, :] = (dy / q) / SIGMA_LOG[name]
        else:
            raise KeyError(name)
    return Js


def lnU_column(obs_names: tuple[str, ...]) -> np.ndarray:
    col = np.zeros((len(obs_names), 1), dtype=float)
    for k, name in enumerate(obs_names):
        if name in SIGMA_LOG:
            col[k, 0] = 1.0 / SIGMA_LOG[name]
    return col


def _n_wall_in(obs_names: tuple[str, ...]) -> int:
    return sum(1 for n in obs_names if n in WALL_OBS)


def _add_lnU(obs_names: tuple[str, ...]) -> bool:
    """ln U 只在恰好一个壁面代理时加入，禁止 conv+rad 同堆共用一个 ln U。"""
    return _n_wall_in(obs_names) == 1


def _window_has_wall(wid: str) -> bool:
    if "p114d" not in _CTX:
        return False
    return pd.notna(_CTX["p114d"].loc[wid, "T_sat_iapws_C"])


def _is_exploratory(phase: str, window_set: str, subset: str) -> bool:
    if subset in EXPLORATORY_SUBSETS:
        return True
    if phase in ("phase3", "phase4") and window_set == "phase_all":
        return True
    return False


def _empty_fisher_out(
    subset: str,
    n_windows: int,
    used: list[str] | None = None,
    note: str = "no finite Jacobian block",
) -> dict:
    used = used or []
    obs_names = SUBSETS[subset]
    param_names = list(PARAM_NAMES) + (["ln_U"] if _add_lnU(obs_names) else [])
    return {
        "subset": subset, "n_windows": n_windows, "n_success": len(used),
        "n_obs_rows": 0, "n_params": len(param_names),
        "param_names": ",".join(param_names),
        "used_windows": ",".join(_short(w) for w in used),
        "sigma_kappa": float("nan"), "sigma_ln_f_gas": float("nan"),
        "sigma_xi": float("nan"), "sigma_ln_U": float("nan"),
        "rho_kappa_lnfg": float("nan"), "rho_kappa_xi": float("nan"),
        "rho_lnfg_xi": float("nan"),
        "rho_kappa_lnU": float("nan"), "rho_lnfg_lnU": float("nan"),
        "cond_J": float("nan"), "cond_F": float("nan"),
        "min_eig_F": float("nan"), "min_eig_params": "",
        "q1_identifiable": False, "ok": False, "note": note,
    }


def fisher_from_windows(jac_map: dict, wids: list[str], subset: str) -> dict:
    obs_names = SUBSETS[subset]
    add_lnU = _add_lnU(obs_names)
    needs_wall = _n_wall_in(obs_names) >= 1
    used = []
    blocks = []
    for wid in wids:
        pack = jac_map.get(wid)
        if pack is None:
            continue
        J_raw, y = pack
        if not np.all(np.isfinite(J_raw)) or not np.all(np.isfinite(y)):
            continue
        if needs_wall and not _window_has_wall(wid):
            continue
        Js = scale_jac(J_raw, y, obs_names)
        if add_lnU:
            Js = np.hstack([Js, lnU_column(obs_names)])
        if not np.all(np.isfinite(Js)):
            continue
        blocks.append(Js)
        used.append(wid)
    nan_out = _empty_fisher_out(subset, len(wids), used)
    if not blocks:
        return nan_out
    ncols = {b.shape[1] for b in blocks}
    if len(ncols) != 1:
        nan_out["note"] = "Jacobian column mismatch"
        return nan_out
    J = np.vstack(blocks)
    npar = J.shape[1]
    param_names = list(PARAM_NAMES) + (["ln_U"] if npar == 4 else [])
    F = J.T @ J
    try:
        cond_J = float(np.linalg.cond(J))
        cond_F = float(np.linalg.cond(F))
    except np.linalg.LinAlgError:
        cond_J, cond_F = float("inf"), float("inf")
    try:
        evals, evecs = np.linalg.eigh(F)
        min_eig = float(evals[0])
        vmin = evecs[:, 0]
    except np.linalg.LinAlgError:
        min_eig = float("nan")
        vmin = np.full(npar, np.nan)
    try:
        cov = np.linalg.inv(F)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(F)
    sig = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    def rho(i, j):
        d = sig[i] * sig[j]
        return float(cov[i, j] / d) if d > 0 and np.isfinite(d) else float("nan")
    combo = " + ".join(
        f"{vmin[i]:+.4f} {param_names[i]}" for i in range(npar) if np.isfinite(vmin[i])
    )
    rho_kl = rho(0, 1) if npar >= 2 else float("nan")
    sig_lnfg = float(sig[1]) if npar >= 2 else float("nan")
    q1 = bool(np.isfinite(rho_kl) and abs(rho_kl) < RHO_OK
              and np.isfinite(sig_lnfg) and sig_lnfg < LN2)
    out = {
        "subset": subset, "n_windows": len(wids), "n_success": len(used),
        "n_obs_rows": int(J.shape[0]), "n_params": npar,
        "param_names": ",".join(param_names),
        "used_windows": ",".join(_short(w) for w in used),
        "sigma_kappa": float(sig[0]) if npar >= 1 else float("nan"),
        "sigma_ln_f_gas": sig_lnfg,
        "sigma_xi": float(sig[2]) if npar >= 3 else float("nan"),
        "sigma_ln_U": float(sig[3]) if npar >= 4 else float("nan"),
        "rho_kappa_lnfg": rho_kl,
        "rho_kappa_xi": rho(0, 2) if npar >= 3 else float("nan"),
        "rho_lnfg_xi": rho(1, 2) if npar >= 3 else float("nan"),
        "rho_kappa_lnU": rho(0, 3) if npar >= 4 else float("nan"),
        "rho_lnfg_lnU": rho(1, 3) if npar >= 4 else float("nan"),
        "cond_J": cond_J, "cond_F": cond_F,
        "min_eig_F": min_eig, "min_eig_params": combo,
        "q1_identifiable": q1, "ok": True, "note": "",
    }
    for i in range(npar):
        for j in range(npar):
            out[f"cov_{param_names[i]}_{param_names[j]}"] = float(cov[i, j])
    return out


def _windows_of(p114d: pd.DataFrame, phase: str, role: str | None) -> list[str]:
    sub = p114d[p114d["phase_id"] == phase]
    if role is not None:
        sub = sub[sub["split_role"] == role]
    return [str(i) for i in sub.index]


def _add(rows, **kw):
    rows.append({
        "reading": kw.get("reading"),
        "metric": kw.get("metric"),
        "phase": kw.get("phase", ""),
        "subset": kw.get("subset", ""),
        "window_set": kw.get("window_set", ""),
        "window_id": kw.get("window_id", ""),
        "value": kw.get("value", float("nan")),
        "threshold": kw.get("threshold", float("nan")),
        "pass": kw.get("pass_"),
        "note": kw.get("note", ""),
        "exploratory": bool(kw.get("exploratory", False)),
    })


def _n_failed_windows(baseline: pd.DataFrame) -> int:
    """与 rec['ok'] 一致：失败 = ~ok_theta 或 ~ok_jacobian。"""
    if baseline is None or baseline.empty:
        return 0
    ok_th = (
        baseline["ok_theta"].astype(bool)
        if "ok_theta" in baseline.columns
        else pd.Series(False, index=baseline.index)
    )
    ok_j = (
        baseline["ok_jacobian"].astype(bool)
        if "ok_jacobian" in baseline.columns
        else pd.Series(False, index=baseline.index)
    )
    return int((~ok_th | ~ok_j).sum())


def _fisher_ready(fisher: pd.DataFrame) -> bool:
    if fisher is None:
        return False
    cols = getattr(fisher, "columns", [])
    return "phase" in cols and "window_set" in cols


def build_readings(fisher: pd.DataFrame, baseline: pd.DataFrame, jac: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    if _fisher_ready(fisher):
        # Q1：主结论仅 P1/P2 train；P3/P4 phase_all 标 exploratory
        for _, r in fisher.iterrows():
            if r["window_set"] not in ("train", "phase_all"):
                continue
            if r["window_set"] == "phase_all" and r["phase"] in ("phase1", "phase2"):
                continue  # P1/P2 主结论用 train
            rho = float(r.get("rho_kappa_lnfg", np.nan))
            sig = float(r.get("sigma_ln_f_gas", np.nan))
            n_ok = int(r.get("n_success", 0) or 0)
            q1 = bool(r.get("q1_identifiable")) if n_ok > 0 else False
            exploratory = _is_exploratory(str(r["phase"]), str(r["window_set"]), str(r["subset"]))
            if n_ok == 0:
                note_n0 = "无训练窗，不作 train 集局部设计结论"
            else:
                extra = []
                if exploratory and r["phase"] in ("phase3", "phase4"):
                    extra.append(P34_EXPLORATORY_NOTE)
                if str(r["subset"]) == "S4_no_wall":
                    extra.append(S4_NO_WALL_NOTE)
                extra.append("通过=局部设计门槛，不是统计可辨识")
                note_n0 = (
                    f"|ρ(κ,ln f_gas)|={abs(rho) if np.isfinite(rho) else float('nan'):.4g} "
                    f"(thr {RHO_OK}); σ(ln f_gas)={sig:.4g} (thr ln2={LN2:.4g}); "
                    f"n={n_ok}; " + "; ".join(extra)
                )
            _add(rows, reading="Q1", metric="identifiable_f_gas",
                 phase=r["phase"], subset=r["subset"], window_set=r["window_set"],
                 value=float(q1) if n_ok > 0 else float("nan"), threshold=RHO_OK,
                 pass_=q1 if n_ok > 0 else None,
                 note=note_n0, exploratory=exploratory)
            _add(rows, reading="Q1", metric="abs_rho_kappa_lnfg",
                 phase=r["phase"], subset=r["subset"], window_set=r["window_set"],
                 value=abs(rho) if np.isfinite(rho) else float("nan"), threshold=RHO_OK,
                 pass_=bool(np.isfinite(rho) and abs(rho) < RHO_OK),
                 exploratory=exploratory)
            _add(rows, reading="Q1", metric="sigma_ln_f_gas",
                 phase=r["phase"], subset=r["subset"], window_set=r["window_set"],
                 value=sig, threshold=LN2,
                 pass_=bool(np.isfinite(sig) and sig < LN2),
                 exploratory=exploratory)
        # Q2：S1 / S2 / S3_conv / S3_rad 相对 S0；P3/P4 标 exploratory
        for phase in ("phase1", "phase2", "phase3", "phase4"):
            wset = "train" if phase in ("phase1", "phase2") else "phase_all"
            exploratory = phase in ("phase3", "phase4")
            s0 = fisher[(fisher["phase"] == phase) & (fisher["subset"] == "S0")
                        & (fisher["window_set"] == wset)]
            if s0.empty:
                continue
            sig0 = float(s0.iloc[0]["sigma_ln_f_gas"])
            best_name, best_factor = "", float("nan")
            q2_note_x = P34_EXPLORATORY_NOTE if exploratory else ""
            for sub in Q2_COMPARE_SUBSETS:
                ss = fisher[(fisher["phase"] == phase) & (fisher["subset"] == sub)
                            & (fisher["window_set"] == wset)]
                if ss.empty:
                    continue
                sigs = float(ss.iloc[0]["sigma_ln_f_gas"])
                factor = (sig0 / sigs) if (np.isfinite(sig0) and np.isfinite(sigs) and sigs > 0) else float("nan")
                note = f"σ_S0={sig0:.4g}; σ_{sub}={sigs:.4g}"
                if q2_note_x:
                    note = f"{note}; {q2_note_x}"
                _add(rows, reading="Q2", metric=f"shrink_{sub}_over_S0",
                     phase=phase, subset=sub, window_set=wset, value=factor,
                     note=note, exploratory=exploratory)
                if np.isfinite(factor) and (not np.isfinite(best_factor) or factor > best_factor):
                    best_factor, best_name = factor, sub
            _add(rows, reading="Q2", metric="largest_contributor",
                 phase=phase, window_set=wset, value=best_factor,
                 note=(best_name or "none") + (f"; {q2_note_x}" if q2_note_x else ""),
                 exploratory=exploratory)
        # Q3：stacked 总信息、每窗 σ√n、ρ 变化；不归因于 O/C
        for phase in ("phase1", "phase2", "phase3", "phase4"):
            ref = fisher[(fisher["phase"] == phase) & (fisher["subset"] == "S0")
                         & (fisher["window_set"] == "p113a_ref")]
            expanded_set = "train" if phase in ("phase1", "phase2") else "phase_all"
            exploratory = phase in ("phase3", "phase4")
            exp = fisher[(fisher["phase"] == phase) & (fisher["subset"] == "S0")
                         & (fisher["window_set"] == expanded_set)]
            if ref.empty or exp.empty:
                _add(rows, reading="Q3", metric="sigma_ln_f_gas_ref_to_expanded",
                     phase=phase, subset="S0", value=float("nan"),
                     note="missing ref or expanded Fisher", exploratory=exploratory)
                continue
            sref = float(ref.iloc[0]["sigma_ln_f_gas"])
            sexp = float(exp.iloc[0]["sigma_ln_f_gas"])
            n_ref = int(ref.iloc[0]["n_success"] or 0)
            n_exp = int(exp.iloc[0]["n_success"] or 0)
            rho_ref = float(ref.iloc[0]["rho_kappa_lnfg"])
            rho_exp = float(exp.iloc[0]["rho_kappa_lnfg"])
            ratio = (sref / sexp) if (np.isfinite(sref) and np.isfinite(sexp) and sexp > 0) else float("nan")
            sref_pw = (sref * math.sqrt(n_ref)) if (np.isfinite(sref) and n_ref > 0) else float("nan")
            sexp_pw = (sexp * math.sqrt(n_exp)) if (np.isfinite(sexp) and n_exp > 0) else float("nan")
            ratio_pw = (
                (sref_pw / sexp_pw)
                if (np.isfinite(sref_pw) and np.isfinite(sexp_pw) and sexp_pw > 0)
                else float("nan")
            )
            q3_note = Q3_NOTE + (f"；{P34_EXPLORATORY_NOTE}" if exploratory else "")
            _add(rows, reading="Q3", metric="stacked_sigma_ref",
                 phase=phase, subset="S0", window_set="p113a_ref", value=sref,
                 note=f"堆叠总信息; n={n_ref}; {q3_note}", exploratory=exploratory)
            _add(rows, reading="Q3", metric="stacked_sigma_exp",
                 phase=phase, subset="S0", window_set=expanded_set, value=sexp,
                 note=f"堆叠总信息; n={n_exp}; {q3_note}", exploratory=exploratory)
            _add(rows, reading="Q3", metric="stacked_ratio_ref_over_exp",
                 phase=phase, subset="S0", window_set=expanded_set, value=ratio,
                 note=f"σ_ref/σ_exp 堆叠总信息; σ_ref={sref:.4g} → σ_exp={sexp:.4g}; {q3_note}",
                 exploratory=exploratory)
            _add(rows, reading="Q3", metric="per_window_sigma_sqrt_n_ref",
                 phase=phase, subset="S0", window_set="p113a_ref", value=sref_pw,
                 note=f"σ√n; n={n_ref}; {q3_note}", exploratory=exploratory)
            _add(rows, reading="Q3", metric="per_window_sigma_sqrt_n_exp",
                 phase=phase, subset="S0", window_set=expanded_set, value=sexp_pw,
                 note=f"σ√n; n={n_exp}; {q3_note}", exploratory=exploratory)
            _add(rows, reading="Q3", metric="per_window_ratio_ref_over_exp",
                 phase=phase, subset="S0", window_set=expanded_set, value=ratio_pw,
                 note=(f"σ√n_ref/σ√n_exp; {sref_pw:.4g} → {sexp_pw:.4g}; {q3_note}"),
                 exploratory=exploratory)
            _add(rows, reading="Q3", metric="rho_kappa_lnfg_ref",
                 phase=phase, subset="S0", window_set="p113a_ref", value=rho_ref,
                 note=f"n={n_ref}; {q3_note}", exploratory=exploratory)
            _add(rows, reading="Q3", metric="rho_kappa_lnfg_exp",
                 phase=phase, subset="S0", window_set=expanded_set, value=rho_exp,
                 note=f"n={n_exp}; {q3_note}", exploratory=exploratory)
            _add(rows, reading="Q3", metric="rho_kappa_lnfg_delta_exp_minus_ref",
                 phase=phase, subset="S0", window_set=expanded_set,
                 value=(rho_exp - rho_ref) if (np.isfinite(rho_ref) and np.isfinite(rho_exp)) else float("nan"),
                 note=f"ρ_exp−ρ_ref; ρ_ref={rho_ref:.4g} → ρ_exp={rho_exp:.4g}; {q3_note}",
                 exploratory=exploratory)
    # Q4 符号
    okb = baseline[baseline["ok_jacobian"] == True] if "ok_jacobian" in baseline.columns else baseline  # noqa: E712
    if not okb.empty and "dT_out_same_sign" in okb.columns:
        n = len(okb)
        n_same = int(okb["dT_out_same_sign"].astype(bool).sum())
        frac = n_same / n if n else float("nan")
        _add(rows, reading="Q4", metric="n_same_sign_dT",
             value=float(n_same), threshold=0.20,
             pass_=bool(frac <= 0.20) if np.isfinite(frac) else None,
             note=f"{n_same}/{n} = {frac:.3f}; >20% 则物理差别不普遍，单列")
        _add(rows, reading="Q4", metric="frac_same_sign_dT",
             value=float(frac), threshold=0.20,
             pass_=bool(np.isfinite(frac) and frac <= 0.20))
        for phase, g in okb.groupby("phase"):
            ns = int(g["dT_out_same_sign"].astype(bool).sum())
            _add(rows, reading="Q4", metric="n_same_sign_dT",
                 phase=phase, value=float(ns),
                 note=f"{ns}/{len(g)}")
        same = okb[okb["dT_out_same_sign"].astype(bool)]
        if frac > 0.20:
            for _, r in same.iterrows():
                _add(rows, reading="Q4", metric="same_sign_window",
                     phase=r.get("phase"), window_id=r.get("window_id"),
                     value=1.0,
                     note=(f"∂T/∂κ={r.get('dT_out_d_kappa'):.4g}; "
                           f"∂T/∂ln f_gas={r.get('dT_out_d_ln_f_gas'):.4g}"))
    # Q5：P1.14-D 第 2 问，不发明新相关
    for phase, d in P114D_Q2.items():
        both_weak = abs(d["rho_oc"]) < 0.3 and abs(d["rho_coal"]) < 0.3
        cv_small = d["cv_ratio"] < 1.0
        drowned = bool(both_weak or cv_small)
        _add(rows, reading="Q5", metric="steam_rho_oc",
             phase=phase, value=d["rho_oc"], threshold=0.3,
             note="P1.14-D Q2; not recomputed")
        _add(rows, reading="Q5", metric="steam_rho_coal",
             phase=phase, value=d["rho_coal"], threshold=0.3,
             note="P1.14-D Q2; not recomputed")
        _add(rows, reading="Q5", metric="steam_cv_between_over_within",
             phase=phase, value=d["cv_ratio"], threshold=1.0,
             note="P1.14-D Q2; >1 窗间不小于窗内")
        _add(rows, reading="Q5", metric="s3_may_be_drowned",
             phase=phase, value=float(drowned), pass_=drowned,
             note=("S3 的信息在理论上存在，但数据中可能被慢变量淹没"
                   if drowned else
                   "卡面条件未触发（需双 |ρ|<0.3 或 CV间<CV内）"))
    # 覆盖
    if baseline is not None and not baseline.empty:
        n_ok = int(baseline["ok_theta"].astype(bool).sum()) if "ok_theta" in baseline.columns else 0
        n_fail = _n_failed_windows(baseline)
        _add(rows, reading="coverage", metric="n_success_theta",
             value=float(n_ok), note=f"/{len(baseline)}")
        _add(rows, reading="coverage", metric="n_failed",
             value=float(n_fail), threshold=float(FAIL_WINDOW_STOP),
             pass_=bool(n_fail <= FAIL_WINDOW_STOP),
             note="失败谓词与 rec['ok'] 一致：~ok_theta | ~ok_jacobian；>10 窗失败则停止")
    return pd.DataFrame(rows)


def interpretation(baseline, jac, fisher, rd, stopped: bool, n_fail: int) -> str:
    L = ["# P1.14-A 可辨识性设计判读", ""]
    L.append(
        "结构冻结 `wsr_plus_pfr`，`L_wsr = 1 m`，`dz_down = 0.0515 m`，"
        "`total_percent` `HL_tot = 1%`，`OxygenMixingModel = instant`。"
        "阶段 θ 取自 P1.13-A 分析仪口径（`L_wsr=1 m`，`HL_tot=1%`）；"
        "θ0 = (κ=1, ξ=0, f_gas=1)。ξ 只作用于出口组成，不进求解器。"
        "水冷壁诊断量 `Q_wall_conv` / `Q_wall_rad` 不进求解；"
        "对流与辐射分列（`S3_conv`/`S3_rad`、`S4_conv`/`S4_rad`），"
        "各自带 ln U，禁止把两种壁面代理堆进同一个 Fisher。"
        "对数残差对 ln U 的未缩放列为 1，缩放后为 1/0.10。"
        "本轮**不采纳**任何参数值，不做反演。"
    )
    L.append("")
    L.append("观测缩放（预声明，跑完未改）：X_C 1.5 pp；CO/H2/CO2 0.5 vol%；CH4 0.05 vol%；"
             "干气/干煤 2%（相对）；Q_wall 10%（对数）；"
             "T_out 20 K 为**理想设计假设**，不是装置出口温测量不确定度"
             "（用于 `S4_conv` / `S4_rad` / `S4_no_wall`）。")
    L.append("")
    if stopped:
        L.append(f"**停止条件触发**：{n_fail} 窗求解失败（>{FAIL_WINDOW_STOP}）。未完成的窗见 `baseline_142.csv`。")
        L.append("")
    n = len(baseline) if baseline is not None and not baseline.empty else 0
    n_ok = int(baseline["ok_theta"].astype(bool).sum()) if n and "ok_theta" in baseline.columns else 0
    n_j = int(baseline["ok_jacobian"].astype(bool).sum()) if n and "ok_jacobian" in baseline.columns else 0
    L.append("## 覆盖")
    L.append(f"- 142 窗基线：θ 成功 {n_ok}/{n}，雅可比有限 {n_j}/{n}，失败 {n_fail}。")
    if n and "d_X_C_pp_theta_minus_plant" in baseline.columns:
        ok = baseline[baseline["ok_theta"] == True]  # noqa: E712
        if not ok.empty:
            for col, lab in (
                ("d_X_C_pp_theta_minus_plant", "θ 下 X_C 模型−装置 (pp)"),
                ("d_CO_vol_pct_theta_minus_plant", "θ 下 CO (vol%)"),
                ("d_H2_vol_pct_theta_minus_plant", "θ 下 H2 (vol%)"),
                ("d_CO2_vol_pct_theta_minus_plant", "θ 下 CO2 (vol%)"),
                ("d_X_C_pp_theta0_minus_plant", "θ0 下 X_C 模型−装置 (pp)"),
            ):
                s = pd.to_numeric(ok[col], errors="coerce")
                L.append(f"- {lab}：中位 {s.median():.3g}，范围 {s.min():.3g} … {s.max():.3g}（n={s.notna().sum()}）")
    L.append("")
    L.append("冻结切分：P1 全部 train（37），P2 全部 train（44），P3 全部 validation（44），"
             "P4 全部 test（17）。主分析 `window_set=train` **仅 P1/P2**。"
             "P3/P4 的 `phase_all` 标 exploratory：超出训练窗主分析；阶段 θ 已用同阶段观测，"
             "不是独立验证。`S4_no_wall` 为探索变体，不是卡面 S4。")
    L.append("")

    def _sel(reading):
        return rd[rd["reading"] == reading] if rd is not None and not rd.empty else pd.DataFrame()

    L.append("## Q1 局部设计门槛（不是统计可辨识）")
    L.append(Q1_LOCAL_DESIGN)
    L.append("判据：|ρ(κ, ln f_gas)| < 0.9 **且** σ(ln f_gas) < ln 2。主结论只用 P1/P2 train。")
    for _, r in _sel("Q1").iterrows():
        if r["metric"] != "identifiable_f_gas":
            continue
        expl = bool(r.get("exploratory", False))
        if r["pass"] is None or (isinstance(r["pass"], float) and not np.isfinite(r["pass"])):
            flag = "无训练窗"
        else:
            flag = "通过局部设计门槛" if bool(r["pass"]) else "未过局部设计门槛"
        if expl:
            flag = f"探索 / {flag}"
        L.append(f"- {r.get('phase')} {r.get('subset')} [{r.get('window_set')}]：**{flag}**。{r.get('note','')}")
    L.append("")
    L.append("## Q2 信息来源（相对 S0 的 σ(ln f_gas) 缩小；S3_conv / S3_rad 分开）")
    for _, r in _sel("Q2").iterrows():
        prefix = "探索 / " if bool(r.get("exploratory", False)) else ""
        L.append(f"- {prefix}{r.get('phase')} `{r['metric']}` = {r['value']}。{r.get('note','')}")
    L.append("")
    L.append("## Q3 扩窗（S0）：堆叠总信息、每窗归一 σ√n、ρ 变化")
    L.append("扩窗同时改变样本数 n 与工况。堆叠总尺度下降不能归因于 O/C 多样性；"
             "相同阶段 θ 下参考窗 S0 已通过 Q1 局部门槛。下列分列 stacked / per-window / ρ，"
             "不跑等 n 对照反应器。")
    for _, r in _sel("Q3").iterrows():
        prefix = "探索 / " if bool(r.get("exploratory", False)) else ""
        L.append(f"- {prefix}{r.get('phase')} `{r['metric']}` [{r.get('window_set')}] = {r['value']}。{r.get('note','')}")
    L.append("")
    L.append("## Q4 符号核对 ∂T_out/∂κ 与 ∂T_out/∂ln f_gas")
    q4f = _sel("Q4")
    frac_rows = q4f[q4f["metric"] == "frac_same_sign_dT"]
    if not frac_rows.empty:
        frac = float(frac_rows.iloc[0]["value"])
        if frac > 0.20:
            L.append(f"- **同号窗 {frac:.1%} > 20%，单列**：第 0 节的物理差别在这些窗上不普遍成立。")
        else:
            L.append(f"- 同号窗 {frac:.1%} ≤ 20%，未触发单列。")
    for _, r in q4f.iterrows():
        if r["metric"] == "same_sign_window":
            L.append(f"  - `{_short(str(r.get('window_id','')))}` {r.get('phase')} {r.get('note')}")
        elif r["metric"] in ("n_same_sign_dT", "frac_same_sign_dT"):
            L.append(f"- `{r['metric']}` {r.get('phase','')} = {r['value']}. {r.get('note','')}")
    L.append("")
    L.append("## Q5 数据可用性（P1.14-D 第 2 问，未重算）")
    L.append("卡面：若汽包产汽与 O/C、煤量的 |ρ| 都 < 0.3，**或** 窗间 CV < 窗内 CV，"
             "则标注「S3 的信息在理论上存在，但数据中可能被慢变量淹没」。")
    for _, r in _sel("Q5").iterrows():
        L.append(f"- {r.get('phase')} `{r['metric']}` = {r['value']}。{r.get('note','')}")
    L.append("")
    L.append("## Fisher 摘要")
    if fisher is not None and not fisher.empty:
        show = fisher[fisher["window_set"].isin(["train", "phase_all", "p113a_ref"])]
        for _, r in show.iterrows():
            L.append(
                f"- {r['phase']} {r['subset']} [{r['window_set']}] n={r['n_success']}: "
                f"ρ(κ,ln f_gas)={r.get('rho_kappa_lnfg')}; "
                f"σ(ln f_gas)={r.get('sigma_ln_f_gas')}; "
                f"cond_J={r.get('cond_J')}; min_eig={r.get('min_eig_F')}; "
                f"方向 `{r.get('min_eig_params')}`. "
                f"Q1={'是' if r.get('q1_identifiable') else '否'}"
            )
    L.append("")
    L.append("## 必须回答")
    L.append("1. 求解状态与模型−装置偏差见 `baseline_142.csv` 与上文覆盖节。")
    L.append("2. Q1–Q5 见上，逐阶段数值见 `readings.csv` / `fisher_subsets.csv`。")
    L.append("3. 求解失败行在 `baseline_142.csv` 的 `ok_theta` / `ok_jacobian` / `error`。")
    L.append("")
    L.append("本轮不采纳 κ、ξ、`f_gas`、U 的任何数值，不改合同。")
    L.append("")
    return "\n".join(L) + "\n"


def _protocol(src_commit, stopped: bool, n_fail: int) -> dict:
    sha_in = hashlib.sha256(P114D_CSV.read_bytes()).hexdigest() if P114D_CSV.exists() else ""
    return base_protocol({
        "round": "P1.14-A / 2026-11-07",
        "model_repo": "gasifier-1d-kinetic",
        "mesh_family": "wsr_plus_pfr",
        "source_commit": src_commit or source_commit(),
        "task_card": "papers/jax-gasifier-cascade-hybrid/manuscript/"
                     "model-side-handoff-p1.14a-20261107.md",
        "structure": {
            "mesh_family": "wsr_plus_pfr",
            "L_wsr_m": L_WSR_NOM,
            "dz_down_m": DZ_DOWN,
            "HL_tot_pct": HL_NOM,
            "oxygen_mixing": "instant",
            "D_wall_m": D_WALL_M,
            "frozen": [
                "全部 Arrhenius（除 f_gas 作用的 C+H2O/C+CO2 指前因子）",
                "限幅", "燃烧区判据", "煤量主口径 windows_142.coal_dry_recon_phase_kg_s",
                "实测载气 windows_142.co2_carrier_measured_nm3_h", "求解器容差",
            ],
        },
        "parameters": {
            "kappa": "inlet O2 multiplier; held at P1.13-A analyzer phase value; not inverted",
            "xi_mol_s": "post-reactor WGS extent; does not enter the reactor solve",
            "ln_f_gas": "log multiplier on shared C+H2O / C+CO2 prefactor; held at phase value",
            "ln_U": "phase-level redundant scale for log Q_wall; column of unscaled log residual is 1",
            "theta_phase_analyzer_Lwsr1_HLtot1pct": PHASE_THETA,
            "theta0": THETA0,
        },
        "observations": {
            "y": list(OBS_NAMES),
            "sigma_additive": SIGMA_ADD,
            "sigma_relative": SIGMA_REL,
            "sigma_log": SIGMA_LOG,
            "T_out_K_sigma": SIGMA_ADD["T_out_K"],
            "T_out_K_sigma_note": (
                "理想设计假设 20 K，不是装置出口温测量不确定度；"
                "仅用于含 T_out 的子集（S4_conv / S4_rad / S4_no_wall）"
            ),
            "Q_wall": (
                "diagnostic index, not in the solve: "
                "Q_conv = Σ A_i (T_i-T_w), Q_rad = Σ A_i (T_i^4-T_w^4), A_i = π D dz_i"
            ),
            "T_w": "T_sat_iapws_C + 273.15 K; default 537 K if missing (phase 4)",
            "dry_per_coal": "shifted dry mol/s per coal_dry_recon_phase_kg_s (mol/kg)",
        },
        "jacobian": {
            "scheme": "forward finite difference at phase θ",
            "diff_rel": DIFF_REL,
            "step_floor": {"kappa": float(STEP_FLOOR[0]), "ln_f_gas": float(STEP_FLOOR[1]),
                           "xi_mol_s": float(STEP_FLOOR[2])},
            "step_rule": "h = diff_rel * max(|θ_j|, floor_j); ξ does not re-solve the reactor",
            "param_order": list(PARAM_NAMES),
        },
        "fisher": {
            "train_only": (
                "主分析 window_set=train 仅 P1/P2。"
                "P3 全为 validation、P4 全为 test；其 phase_all 标 exploratory，"
                "不是独立验证（阶段 θ 已用同阶段观测）。"
            ),
            "subsets": {k: list(v) for k, v in SUBSETS.items()},
            "S3_S4_wall": (
                "S3_conv / S3_rad / S4_conv / S4_rad 仅 phase1–3，"
                "各含恰好一个壁面代理并各自带 ln U；禁止 conv+rad 同堆。"
                "S4_no_wall 为探索变体，不是卡面 S4。"
            ),
            "ln_U_column": (
                "仅当子集含恰好一个壁面代理时加入；"
                "unscaled log residual d/d ln U = 1; scaled = 1/0.10"
            ),
            "q1": "|ρ(κ,ln f_gas)|<0.9 且 σ(ln f_gas)<ln 2；通过=局部设计门槛，不是统计可辨识",
            "q1_assumptions": Q1_LOCAL_DESIGN,
            "q3": (
                "分别报告 stacked 总信息、σ√n 每窗归一、ρ 变化；"
                "扩窗同时改变 n 与工况，总尺度下降不能归因于 O/C。"
            ),
        },
        "predeclared": {
            "Q1": (
                "固定阶段 θ、共享 κ/ln f_gas/ξ、对角预声明权重、窗间独立。"
                "|ρ|<0.9 且 σ(ln f_gas)<ln 2 → 通过局部设计门槛，不是统计可辨识。"
                "主结论仅 P1/P2 train。"
            ),
            "Q2": "P1/P2 train：S1、S2、S3_conv、S3_rad 相对 S0 的 σ 缩小；分开谁贡献最大。P3/P4 若报则标 exploratory。",
            "Q3": (
                "S0：stacked（σ_ref、σ_exp、σ_ref/σ_exp）、per-window（σ√n 及其比）、ρ 变化。"
                "不归因于 O/C；不跑等 n 对照反应器。"
            ),
            "Q4": "sign of dT_out/dκ vs dT_out/d ln f_gas; same-sign fraction >20% listed separately",
            "Q5": "P1.14-D Q2 numbers used as given; drown flag if both |ρ|<0.3 or CV_between<CV_within",
        },
        "stopped": stopped,
        "n_failed_windows": n_fail,
        "fail_stop_threshold": FAIL_WINDOW_STOP,
        "source_csv": {
            "p1.14d": {
                "round": "p1.14d-20261107",
                "file": "windows_142.csv",
                "relative_path": "outputs/diagnostics/p1.14d-20261107/windows_142.csv",
                "sha256": sha_in,
                "n_rows": 142,
            }
        },
        "outputs": [
            "baseline_142.csv", "jacobians_142.csv", "fisher_subsets.csv",
            "readings.csv", "protocol.json", "interpretation.md", "sha256.txt",
        ],
    })


def _rewrite_manifest(out_dir: Path) -> None:
    skip = {"sha256.txt", "run.log"}
    files = sorted(p for p in out_dir.iterdir() if p.is_file() and p.name not in skip)
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}" for p in files]
    (out_dir / "sha256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _jac_map_from_df(jac: pd.DataFrame) -> dict:
    out = {}
    if jac is None or jac.empty or "window_id" not in getattr(jac, "columns", []):
        return out
    for wid, g in jac.groupby("window_id"):
        g = g.sort_values("obs_index")
        if len(g) < len(OBS_NAMES):
            continue
        J = g[["d_kappa", "d_ln_f_gas", "d_xi_mol_s"]].to_numpy(dtype=float)
        y = g["y_theta"].to_numpy(dtype=float)
        out[str(wid)] = (J, y)
    return out


def _subsets_for_phase(phase: str) -> list[str]:
    if phase == "phase4":
        return [s for s in SUBSETS if s not in WALL_SUBSETS]
    return list(SUBSETS)


def _annotate_fisher_row(rec: dict, phase: str, wset: str, subset: str,
                         split_role: str = "") -> dict:
    rec = dict(rec)
    rec["phase"] = phase
    rec["window_set"] = wset
    rec["split_role_of_phase"] = split_role
    rec["exploratory"] = _is_exploratory(phase, wset, subset)
    notes = [str(rec.get("note") or "").strip()]
    if subset == "S4_no_wall":
        notes.append(S4_NO_WALL_NOTE)
    if phase in ("phase3", "phase4") and wset == "phase_all":
        notes.append(P34_EXPLORATORY_NOTE)
    rec["note"] = "; ".join(n for n in notes if n)
    return rec


def compute_fisher_table(jac: pd.DataFrame, p114d: pd.DataFrame, ref_ids: list[str]) -> pd.DataFrame:
    """jac 为空也写出带列的 fisher 行（n_success=0）。"""
    jmap = _jac_map_from_df(jac)
    rows = []
    p114d = p114d.copy()
    p114d["phase_id"] = [_phase_of_row(p) for p in p114d["phase"]]
    for phase in ("phase1", "phase2", "phase3", "phase4"):
        train_ids = _windows_of(p114d, phase, "train")
        all_ids = _windows_of(p114d, phase, None)
        ref_phase = [w for w in ref_ids if w in set(p114d.index) and _phase_of_row(p114d.loc[w, "phase"]) == phase]
        sets = [("train", train_ids), ("phase_all", all_ids), ("p113a_ref", ref_phase)]
        for wset, wids in sets:
            split_role = str(p114d.loc[wids[0], "split_role"]) if wids else ""
            for sub in _subsets_for_phase(phase):
                if not wids:
                    rec = _empty_fisher_out(sub, 0, note="no windows in this set")
                else:
                    rec = fisher_from_windows(jmap, wids, sub)
                rows.append(_annotate_fisher_row(rec, phase, wset, sub, split_role))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=("all", "smoke", "read"))
    ap.add_argument("--source-commit", default=None)
    ap.add_argument("--n-workers", type=int, default=N_WORKERS_DEFAULT)
    ap.add_argument("--n-windows", type=int, default=0, help="smoke/limit: first N windows, 0=all")
    args = ap.parse_args()
    silence_solver_logs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _init_worker()
    p114d = _CTX["p114d"].reset_index()
    p114d_idx = _CTX["p114d"]
    window_ids = [str(w) for w in p114d_idx.index]
    if args.n_windows and args.n_windows > 0:
        window_ids = window_ids[: int(args.n_windows)]
    ref_ids = m1_reference_window_ids(M1_MATRIX)

    if args.stage == "read":
        base = pd.read_csv(OUT_DIR / "baseline_142.csv")
        jac = pd.read_csv(OUT_DIR / "jacobians_142.csv")
        fisher = compute_fisher_table(jac, p114d_idx, ref_ids)
        rd = build_readings(fisher, base, jac)
        proto_path = OUT_DIR / "protocol.json"
        stopped = False
        src_commit = args.source_commit
        if proto_path.exists():
            old_proto = json.loads(proto_path.read_text(encoding="utf-8"))
            if "stopped" in old_proto:
                stopped = bool(old_proto["stopped"])
            if src_commit is None:
                src_commit = old_proto.get("source_commit")
        n_fail = _n_failed_windows(base)
        text = interpretation(base, jac, fisher, rd, stopped, n_fail)
        protocol = _protocol(src_commit, stopped, n_fail)
        # 不重写 baseline/jacobians，只更新 fisher / readings / protocol / interpretation
        write_round_outputs(OUT_DIR, tables={
            "fisher_subsets.csv": fisher, "readings.csv": rd,
        }, protocol=protocol, interpretation=text)
        _rewrite_manifest(OUT_DIR)
        print("written read-stage", flush=True)
        return

    if args.stage == "smoke":
        wid = window_ids[0]
        rec = _job(wid)
        print(json.dumps({
            "window_id": wid,
            "ok": rec.get("ok"),
            "baseline_keys": sorted(rec["baseline"].keys())[:30],
            "n_jac_rows": len(rec.get("jac_rows") or []),
            "ok_theta": rec["baseline"].get("ok_theta"),
            "T_out": rec["baseline"].get("theta_T_out_K"),
            "Q_conv": rec["baseline"].get("theta_Q_wall_conv"),
            "dT_dkappa": rec["baseline"].get("dT_out_d_kappa"),
            "dT_dlnfg": rec["baseline"].get("dT_out_d_ln_f_gas"),
        }, ensure_ascii=False, default=str, indent=2))
        return

    n_workers = max(1, min(int(args.n_workers), len(window_ids)))
    baseline_rows, jac_rows = [], []
    n_fail = 0
    stopped = False
    print(f"START n_windows={len(window_ids)} n_workers={n_workers}", flush=True)

    def _consume(rec):
        nonlocal n_fail, stopped
        baseline_rows.append(rec["baseline"])
        jac_rows.extend(rec.get("jac_rows") or [])
        ok = bool(rec.get("ok"))
        if not ok:
            n_fail += 1
            print(f"FAIL {_short(str(rec['baseline'].get('window_id')))} n_fail={n_fail}", flush=True)
        if n_fail > FAIL_WINDOW_STOP:
            stopped = True
        return ok

    if n_workers == 1:
        for wid in window_ids:
            rec = _job(wid)
            _consume(rec)
            if stopped:
                print("STOP: >10 windows failed", flush=True)
                break
    else:
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as ex:
            fut = {ex.submit(_job, wid): wid for wid in window_ids}
            for f in as_completed(fut):
                rec = f.result()
                _consume(rec)
                print(f"progress {len(baseline_rows)}/{len(window_ids)} fail={n_fail}", flush=True)
                if stopped:
                    print("STOP: >10 windows failed", flush=True)
                    ex.shutdown(wait=False, cancel_futures=True)
                    break

    base = pd.DataFrame(baseline_rows)
    jac = pd.DataFrame(jac_rows)
    base.to_csv(OUT_DIR / "baseline_142.csv", index=False)
    jac.to_csv(OUT_DIR / "jacobians_142.csv", index=False)
    fisher = compute_fisher_table(jac, p114d_idx, ref_ids)
    rd = build_readings(fisher, base, jac)
    text = interpretation(base, jac, fisher, rd, stopped, n_fail)
    protocol = _protocol(args.source_commit, stopped, n_fail)
    write_round_outputs(OUT_DIR, tables={
        "baseline_142.csv": base, "jacobians_142.csv": jac,
        "fisher_subsets.csv": fisher, "readings.csv": rd,
    }, protocol=protocol, interpretation=text)
    _rewrite_manifest(OUT_DIR)
    print("written:", sorted(p.name for p in OUT_DIR.iterdir() if p.is_file()), flush=True)
    print(f"success={int(base['ok_theta'].astype(bool).sum()) if not base.empty else 0} "
          f"fail={n_fail} stopped={stopped}", flush=True)


if __name__ == "__main__":
    main()
