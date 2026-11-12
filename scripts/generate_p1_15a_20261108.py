#!/usr/bin/env python3
"""P1.15-A 工况响应斜率诊断与机理候选（2026-11-08 卡）。

冻结与 P1.14-A 相同：wsr_plus_pfr、L_wsr=1 m、dz_down=0.0515 m、total_percent
HL_tot=1%、OxygenMixingModel=instant、全部 Arrhenius（除候选明确改动的项）。
ξ 只作用于出口，不进求解器。C1 为出口后处理，替代常数 ξ。

本轮不做反演、不采纳任何参数。run_case 走 scipy（jacobian_mode 由执行层固定）。

用法：
    ./.venv/bin/python scripts/generate_p1_15a_20261108.py [--stage all|smoke|read] [--cache-only]
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

from model.kinetics import calculate_wgs_equilibrium  # noqa: E402
from model.p1_5_round_20261030 import (  # noqa: E402
    LIMIT_KEYS,
    base_protocol,
    run_case,
    silence_solver_logs,
    write_round_outputs,
)
from model.revision_evidence_common import load_frozen_split, source_commit  # noqa: E402

OUT_DIR = ROOT / "outputs" / "diagnostics" / "round-p1.15a-20261108"
CACHE_DIR = OUT_DIR / "cache"
P114D_CSV = ROOT / "outputs" / "diagnostics" / "p1.14d-20261107" / "windows_142.csv"
P114A_CSV = ROOT / "outputs" / "diagnostics" / "round-p1.14a-20261107" / "baseline_142.csv"

DEV_L = p113a.DEV_L
DZ_DOWN = p113a.DZ_DOWN
L_WSR_NOM = p113a.L_WSR_NOM
HL_NOM = p113a.HL_NOM
N_WORKERS_DEFAULT = 15
FAIL_WINDOW_STOP = 5
XC_CEIL_PP = 99.9
FD_FRAC = 0.03
LN_P = math.log(1.0 + FD_FRAC)
LN_M = math.log(1.0 - FD_FRAC)
D_LN = LN_P - LN_M
BOOT_N = 1000
BOOT_SEED = 20261108
XI_EPS = 1e-12

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
PHASE_MAP = {"P1": "phase1", "P2": "phase2", "P3": "phase3", "P4": "phase4"}
PHASES = ("phase1", "phase2", "phase3", "phase4")

OBS_R = ("CO_vol_pct", "H2_vol_pct", "CO2_vol_pct", "CH4_vol_pct", "dry_per_coal_mol_kg")
OBS_ALL = ("X_C_pp",) + OBS_R
INPUTS = ("ln_O2", "ln_coal")
PERTS = ("nom", "o2_p", "o2_m", "coal_p", "coal_m")
PERT_SCALE = {
    "nom": (1.0, 1.0),
    "o2_p": (1.0 + FD_FRAC, 1.0),
    "o2_m": (1.0 - FD_FRAC, 1.0),
    "coal_p": (1.0, 1.0 + FD_FRAC),
    "coal_m": (1.0, 1.0 - FD_FRAC),
}
PLANT_Y = {
    "X_C_pp": "xc_recon_phase_cal_measured_pct",
    "CO_vol_pct": "co_vol_pct_mean",
    "H2_vol_pct": "h2_vol_pct_mean",
    "CO2_vol_pct": "co2_vol_pct_mean",
    "CH4_vol_pct": "ch4_vol_pct_mean",
}

C2_LEVELS = (1.0, 1.0 / 4.0, 1.0 / 16.0, 1.0 / 64.0, 1.0 / 256.0)
C3_LEVELS = (0.2575, 1.0, 4.0)
C4_LEVELS = (0.5, 1.0, 2.0)
C1_TQ = (900.0, 1000.0, 1100.0)
C1_RW = (0.3, 0.6, 1.0)

_CTX: dict = {}


def _short(wid: str) -> str:
    return wid[-16:]


def _phase_of_row(phase_raw) -> str:
    raw = str(phase_raw)
    if raw in PHASE_MAP:
        return PHASE_MAP[raw]
    if raw.startswith("phase"):
        return raw
    return f"phase{raw[-1]}"


def _f(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def _coal_dry(wid: str) -> float:
    return float(_CTX["p114d"].loc[wid, "coal_dry_recon_phase_kg_s"])


def _carrier_nm3(wid: str) -> float:
    return float(_CTX["p114d"].loc[wid, "co2_carrier_measured_nm3_h"])


def _o2_nm3(wid: str) -> float:
    return float(_CTX["p114d"].loc[wid, "o2_fy1103a_nm3_h_mean"])


def _plant_dry_per_coal(wid: str) -> float:
    row = _CTX["p114d"].loc[wid]
    dry = float(row["coal_dry_recon_phase_kg_s"])
    plant_dry_mol = float(row["syngas_dry_cal_nm3_h"]) / 22.414 / 3600.0 * 1000.0
    return plant_dry_mol / max(dry, 1e-12)


def c1_id(t_q: float, r_w: float) -> str:
    return f"C1_T{int(t_q)}_rw{r_w:g}"


def c2_id(f_co2: float) -> str:
    if abs(f_co2 - 1.0) < 1e-15:
        return "C2_f1"
    inv = int(round(1.0 / f_co2))
    return f"C2_f1over{inv}"


def c3_id(l_wsr: float) -> str:
    tag = f"{l_wsr:g}".replace(".", "p")
    return f"C3_L{tag}"


def c4_id(hl: float) -> str:
    tag = f"{hl:g}".replace(".", "p")
    return f"C4_HL{tag}"


def wgs_quench(snap: dict, t_q: float, r_w: float) -> dict:
    """出口加水后在 T_q 达 WGS 平衡。替代常数 ξ，不进反应器。"""
    out = {
        "ok": False, "feasible": False, "infeasible_reason": "",
        "T_q_K": float(t_q), "r_w": float(r_w),
        "K_wgs": float("nan"), "xi_eq": float("nan"),
        "H2O_eq_mol_s": float("nan"), "H2O_positive": False,
    }
    if not snap.get("ok"):
        out["infeasible_reason"] = "reactor_failed"
        return out
    n_co = _f(snap["F_CO"])
    n_co2 = _f(snap["F_CO2"])
    n_h2 = _f(snap["F_H2"])
    n_h2o = _f(snap["F_H2O"])
    n_ch4 = _f(snap.get("F_CH4", 0.0))
    n_h2s = _f(snap.get("F_H2S", 0.0))
    n_dry = _f(snap["dry_total_mol_s"])
    x_c = _f(snap["X_C_pp"])
    if not all(math.isfinite(v) for v in (n_co, n_co2, n_h2, n_h2o, n_dry)):
        out["infeasible_reason"] = "nonfinite_moles"
        return out
    try:
        k = float(calculate_wgs_equilibrium(float(t_q)))
    except (TypeError, ValueError, OverflowError):
        out["infeasible_reason"] = "K_eval_failed"
        return out
    out["K_wgs"] = k
    if not math.isfinite(k) or k <= 0.0:
        out["infeasible_reason"] = "K_nonpositive"
        return out
    n_h2o_p = n_h2o + float(r_w) * n_dry
    out["n_H2O_plus_rw"] = n_h2o_p
    lo = -min(n_co2, n_h2) + XI_EPS
    hi = min(n_co, n_h2o_p) - XI_EPS
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        out["infeasible_reason"] = "empty_physical_interval"
        return out
    a = k - 1.0
    b = -(k * (n_co + n_h2o_p) + (n_co2 + n_h2))
    c = k * n_co * n_h2o_p - n_co2 * n_h2
    roots: list[float] = []
    if abs(a) < 1e-14:
        if abs(b) < 1e-18:
            out["infeasible_reason"] = "degenerate_quadratic"
            return out
        roots.append(-c / b)
    else:
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            out["infeasible_reason"] = "no_real_root"
            return out
        sdisc = math.sqrt(disc)
        roots.extend(((-b + sdisc) / (2.0 * a), (-b - sdisc) / (2.0 * a)))
    cand = [xi for xi in roots if math.isfinite(xi) and lo < xi < hi]
    if not cand:
        out["infeasible_reason"] = "root_outside_interval"
        return out
    xi = min(cand, key=lambda z: abs(z))
    n_co_eq = n_co - xi
    n_co2_eq = n_co2 + xi
    n_h2_eq = n_h2 + xi
    n_h2o_eq = n_h2o_p - xi
    dry_eq = n_dry + xi
    wet = dry_eq + n_h2o_eq
    h2o_pos = bool(n_h2o_eq > 0.0)
    out.update({
        "ok": True,
        "feasible": h2o_pos and n_co_eq > 0.0 and dry_eq > 0.0,
        "infeasible_reason": "" if h2o_pos else "H2O_eq_negative",
        "xi_eq": xi,
        "H2O_eq_mol_s": n_h2o_eq,
        "H2O_positive": h2o_pos,
        "X_C_pp": x_c,
        "F_CO": n_co_eq, "F_CO2": n_co2_eq, "F_H2": n_h2_eq, "F_H2O": n_h2o_eq,
        "F_CH4": n_ch4, "F_H2S": n_h2s,
        "dry_total_mol_s": dry_eq,
        "wet_total_mol_s": wet,
        "CO_vol_pct": n_co_eq / max(dry_eq, 1e-12) * 100.0,
        "H2_vol_pct": n_h2_eq / max(dry_eq, 1e-12) * 100.0,
        "CO2_vol_pct": n_co2_eq / max(dry_eq, 1e-12) * 100.0,
        "CH4_vol_pct": n_ch4 / max(dry_eq, 1e-12) * 100.0,
        "CO_wet_pct": n_co_eq / max(wet, 1e-12) * 100.0,
        "H2_wet_pct": n_h2_eq / max(wet, 1e-12) * 100.0,
        "CO2_wet_pct": n_co2_eq / max(wet, 1e-12) * 100.0,
        "H2O_wet_pct": n_h2o_eq / max(wet, 1e-12) * 100.0,
        "CH4_wet_pct": n_ch4 / max(wet, 1e-12) * 100.0,
    })
    if n_co_eq <= 0.0:
        out["feasible"] = False
        out["infeasible_reason"] = "CO_eq_nonpositive"
    return out


def y_c0(snap: dict, xi: float, coal_dry: float) -> dict | None:
    shifted = p113a.apply_xi(snap, xi)
    if not shifted.get("ok"):
        return None
    dry = float(shifted["dry_total_mol_s"])
    if not math.isfinite(dry) or dry <= 0.0:
        return None
    return {
        "X_C_pp": float(shifted["X_C_pp"]),
        "CO_vol_pct": float(shifted["CO_vol_pct"]),
        "H2_vol_pct": float(shifted["H2_vol_pct"]),
        "CO2_vol_pct": float(shifted["CO2_vol_pct"]),
        "CH4_vol_pct": float(snap["F_CH4"]) / max(dry, 1e-12) * 100.0,
        "dry_per_coal_mol_kg": dry / max(float(coal_dry), 1e-12),
        "F_H2O": float(shifted["F_H2O"]),
        "needs_quench_water": bool(shifted.get("needs_quench_water")),
        "xi_mol_s": float(xi),
        "post": "constant_xi",
    }


def y_c1(snap: dict, t_q: float, r_w: float, coal_dry: float) -> dict | None:
    q = wgs_quench(snap, t_q, r_w)
    if not q.get("feasible"):
        return None
    return {
        "X_C_pp": float(q["X_C_pp"]),
        "CO_vol_pct": float(q["CO_vol_pct"]),
        "H2_vol_pct": float(q["H2_vol_pct"]),
        "CO2_vol_pct": float(q["CO2_vol_pct"]),
        "CH4_vol_pct": float(q["CH4_vol_pct"]),
        "dry_per_coal_mol_kg": float(q["dry_total_mol_s"]) / max(float(coal_dry), 1e-12),
        "F_H2O": float(q["H2O_eq_mol_s"]),
        "H2O_wet_pct": float(q["H2O_wet_pct"]),
        "CO_wet_pct": float(q["CO_wet_pct"]),
        "H2_wet_pct": float(q["H2_wet_pct"]),
        "CO2_wet_pct": float(q["CO2_wet_pct"]),
        "xi_eq": float(q["xi_eq"]),
        "post": "wgs_quench",
        "quench": q,
    }


def fd_from_ys(y_p: dict | None, y_m: dict | None, obs: str) -> tuple[float, float]:
    """返回 (∂y/∂ln x, 每+3% 系数)。两侧缺一则 nan。"""
    if y_p is None or y_m is None:
        return float("nan"), float("nan")
    a, b = _f(y_p.get(obs)), _f(y_m.get(obs))
    if not (math.isfinite(a) and math.isfinite(b)):
        return float("nan"), float("nan")
    slope_ln = (a - b) / D_LN
    return slope_ln, slope_ln * LN_P


def _empty_snap(wid: str, **kw) -> dict:
    rec = {"window_id": wid, "ok": False, "execution_status": "failed"}
    rec.update(kw)
    return rec


def _run_reactor(
    wid: str,
    *,
    kappa: float,
    f_gas: float,
    l_wsr: float,
    hl: float,
    o2_mult: float,
    coal_mult: float,
    f_co2: float | None,
) -> dict:
    if _CTX.get("cache_only"):
        raise RuntimeError(f"cache-only refuse solve {_short(wid)}")
    row = _CTX["by_id"][wid]
    dry = _coal_dry(wid) * float(coal_mult)
    op = {
        "HeatLossModel": "total_percent",
        "HeatLossPercentTotal": float(hl),
        "GasificationRateFactor": float(f_gas),
        "carrier_gas_flow": p18a._carrier_kg_s(_carrier_nm3(wid)),
        "MeshFamily": "wsr_plus_pfr",
        "WSRLength_m": float(l_wsr),
        "PFRCellLength_m": float(DZ_DOWN),
        "OxygenMixingModel": "instant",
        "o2_flow_scale": float(kappa) * float(o2_mult),
    }
    if f_co2 is not None:
        op["CharCO2RateFactor"] = float(f_co2)
    print(
        f"    solve {_short(wid)} κ={kappa:.6f} f_gas={f_gas:.6g} "
        f"L={l_wsr:g} HL={hl:g} o2×{o2_mult:g} coal×{coal_mult:g} "
        f"f_CO2={'default' if f_co2 is None else f'{f_co2:g}'}",
        flush=True,
    )
    rec = run_case(
        row, n_cells=p113a.mesh_n(l_wsr), coal_flow_dry_kg_s=dry,
        keep_system=True, mesh_family="wsr_plus_pfr", extra_op=op,
    )
    system = rec.get("_system")
    snap: dict = {
        "window_id": wid,
        "kappa": float(kappa),
        "f_gas": float(f_gas),
        "L_wsr": float(l_wsr),
        "HL_tot_pct": float(hl),
        "o2_mult": float(o2_mult),
        "coal_mult": float(coal_mult),
        "coal_dry_kg_s": float(dry),
        "f_CO2": float("nan") if f_co2 is None else float(f_co2),
        "char_co2_explicit": f_co2 is not None,
        "execution_status": str(rec.get("execution_status")),
        "no_root_cell_index": rec.get("no_root_cell_index"),
        "solve_wall_time_s": rec.get("solve_wall_time_s"),
        "n_cells": rec.get("N_cells"),
        "ok": rec.get("execution_status") == "completed" and bool(getattr(system, "results", None)),
        "X_C_at_ceiling": False,
        "wsr_C_CO2_LimitedByCO2": False,
        "wsr_C_H2O_LimitedByH2O": False,
        "wsr_raw_C_CO2": float("nan"),
        "wsr_raw_C_H2O": float("nan"),
        "wsr_lim_C_CO2": float("nan"),
        "wsr_is_combustion": False,
    }
    for key in LIMIT_KEYS:
        snap[f"limit_len_frac_{key}"] = _f(rec.get(f"limit_len_frac_{key}"))
    if snap["ok"] and system is not None:
        gas = np.asarray(system.results[-1].gas_moles, dtype=float)
        xc_pp = float(rec["X_C_implied"]) * 100.0
        snap.update({
            "X_C_pp": xc_pp,
            "X_C_at_ceiling": bool(xc_pp >= XC_CEIL_PP),
            "T_out_K": float(rec["T_out_K"]),
            "F_CO": float(gas[2]), "F_CO2": float(gas[3]), "F_H2": float(gas[5]),
            "F_H2O": float(gas[7]), "F_CH4": float(gas[1]), "F_H2S": float(gas[4]),
            "dry_total_mol_s": float(np.sum(gas[:7])),
        })
        diags = getattr(system, "cell_diagnostics", []) or []
        if diags:
            d0 = diags[0]
            snap["wsr_C_CO2_LimitedByCO2"] = bool(d0.get("C_CO2_LimitedByCO2", False))
            snap["wsr_C_H2O_LimitedByH2O"] = bool(d0.get("C_H2O_LimitedByH2O", False))
            snap["wsr_is_combustion"] = bool(d0.get("is_combustion_zone", False))
        try:
            cell0 = system.cells[0]
            st0 = system.results[0]
            pack = cell0.snapshot_with_limits(st0)
            raw = pack.get("raw_rates") or {}
            lim = pack.get("limited_rates") or {}
            snap["wsr_raw_C_CO2"] = _f(raw.get("C+CO2"))
            snap["wsr_raw_C_H2O"] = _f(raw.get("C+H2O"))
            snap["wsr_lim_C_CO2"] = _f(lim.get("C+CO2"))
        except Exception as exc:  # noqa: BLE001
            snap["wsr_rate_error"] = f"{type(exc).__name__}: {exc}"
    rec.pop("_system", None)
    rec.pop("_profile", None)
    return snap


def _wid_key(wid: str) -> str:
    digest = hashlib.sha1(str(wid).encode("utf-8")).hexdigest()[:10]
    return f"{_short(wid)}_{digest}"


def _safe_cache_name(token: str, group: str) -> str:
    raw = f"{token}__{group}"
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in raw)


def _cache_path(wid: str, group: str) -> Path:
    return CACHE_DIR / f"{_safe_cache_name(_wid_key(wid), group)}.json"


def _legacy_cache_path(wid: str, group: str) -> Path:
    return CACHE_DIR / f"{_safe_cache_name(_short(wid), group)}.json"


def _dump_snaps(path: Path, payload: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")


def _load_snaps(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _load_group_cache(wid: str, group: str) -> dict | None:
    """命中须核对 JSON 内 window_id，避免短名交接班窗口互相覆盖。"""
    for path in (_cache_path(wid, group), _legacy_cache_path(wid, group)):
        hit = _load_snaps(path)
        if not hit:
            continue
        if str(hit.get("window_id") or "") != str(wid):
            continue
        return hit
    return None


def reactor_group_job(spec: dict) -> dict:
    """一个窗口 × 一个反应器组：名义 + O2± + 煤±。"""
    if "by_id" not in _CTX:
        _init_worker()
    wid = str(spec["window_id"])
    group = str(spec["group"])
    cache_p = _cache_path(wid, group)
    perts = tuple(spec.get("perts") or PERTS)
    hit = _load_group_cache(wid, group)
    if (
        hit and hit.get("complete")
        and all(p in (hit.get("snaps") or {}) for p in perts)
    ):
        print(f"  cache hit {_short(wid)} {group}", flush=True)
        return hit
    if _CTX.get("cache_only"):
        print(f"  cache miss {_short(wid)} {group} (cache-only, skip solve)", flush=True)
        return {
            "window_id": wid, "group": group, "complete": False, "ok_nom": False,
            "error": "cache_miss_cache_only",
            "snaps": {
                p: _empty_snap(wid, pert=p, group=group, skipped_cache_miss=True)
                for p in perts
            },
        }
    phase = _phase_of_row(_CTX["p114d"].loc[wid, "phase"])
    th = PHASE_THETA[phase]
    snaps = {}
    print(f"GROUP {_short(wid)} {phase} {group} perts={perts}", flush=True)
    for pert in perts:
        o2_m, coal_m = PERT_SCALE[pert]
        try:
            snap = _run_reactor(
                wid,
                kappa=th["kappa"],
                f_gas=th["f_gas"],
                l_wsr=float(spec["L_wsr"]),
                hl=float(spec["HL_tot_pct"]),
                o2_mult=o2_m,
                coal_mult=coal_m,
                f_co2=spec.get("f_CO2"),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {_short(wid)} {group} {pert}: {exc}", flush=True)
            traceback.print_exc()
            snap = _empty_snap(
                wid, error=f"{type(exc).__name__}: {exc}",
                L_wsr=float(spec["L_wsr"]), HL_tot_pct=float(spec["HL_tot_pct"]),
                pert=pert,
            )
        snap["pert"] = pert
        snap["group"] = group
        snap["phase"] = phase
        snaps[pert] = snap
        if pert == "nom" and not snap.get("ok"):
            print(f"  nom fail {_short(wid)} {group} — skip remaining perts", flush=True)
            for rest in perts:
                if rest not in snaps:
                    snaps[rest] = _empty_snap(wid, pert=rest, group=group, phase=phase,
                                              skipped_after_nom_fail=True)
            break
    payload = {
        "window_id": wid, "group": group, "phase": phase,
        "complete": True, "ok_nom": bool(snaps["nom"].get("ok")),
        "snaps": snaps,
    }
    _dump_snaps(cache_p, payload)
    print(
        f"  done {_short(wid)} {group} ok_nom={payload['ok_nom']} "
        f"t={snaps['nom'].get('solve_wall_time_s')}",
        flush=True,
    )
    return payload


def _job(spec: dict) -> dict:
    try:
        return reactor_group_job(spec)
    except Exception as exc:  # noqa: BLE001
        print(f"JOB FAIL {_short(str(spec.get('window_id')))} {spec.get('group')}: {exc}", flush=True)
        traceback.print_exc()
        return {
            "window_id": spec.get("window_id"), "group": spec.get("group"),
            "complete": False, "ok_nom": False,
            "error": f"{type(exc).__name__}: {exc}",
            "snaps": {p: _empty_snap(str(spec.get("window_id")), pert=p) for p in (spec.get("perts") or PERTS)},
        }


def _init_worker(cache_only: bool = False):
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
    _CTX.update({"by_id": by_id, "p114d": p114d, "cache_only": bool(cache_only)})
    p113a._CTX.update({"by_id": by_id, "p114d": p114d})
    p113a._CACHE.clear()


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta, _res, rank, _s = np.linalg.lstsq(X, y, rcond=None)
    n, p = X.shape
    rss = float(np.sum((y - X @ beta) ** 2))
    df = n - p
    se = np.full(p, np.nan)
    if df > 0 and int(rank) >= p:
        xtx = X.T @ X
        try:
            inv = np.linalg.inv(xtx)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(xtx)
        sigma2 = rss / df
        se = np.sqrt(np.clip(np.diag(inv) * sigma2, 0.0, None))
    return beta, se


def _bootstrap_per3(X: np.ndarray, y: np.ndarray, col: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = np.empty(BOOT_N, dtype=float)
    for i in range(BOOT_N):
        idx = rng.integers(0, n, n)
        b, _se = _ols(X[idx], y[idx])
        stats[i] = float(b[col]) * LN_P
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def _design_cond(X: np.ndarray) -> float:
    ok = np.isfinite(X).all(axis=1)
    if int(ok.sum()) < X.shape[1]:
        return float("nan")
    try:
        return float(np.linalg.cond(X[ok]))
    except np.linalg.LinAlgError:
        return float("nan")


def _mark_sign_stable(df: pd.DataFrame) -> pd.DataFrame:
    """两种回归下 signed 是否一致；仅 ln_O2 / ln_coal 有对照。"""
    out = df.copy()
    if out.empty:
        out["sign_stable"] = []
        return out
    stable_map: dict[tuple[str, str, str], bool] = {}
    keys = out.loc[out["input"].isin(INPUTS), ["phase", "obs", "input"]].drop_duplicates()
    for _, k in keys.iterrows():
        sub = out[
            (out["phase"] == k["phase"])
            & (out["obs"] == k["obs"])
            & (out["input"] == k["input"])
        ]
        w = sub[sub["model"] == "with_carrier"]
        n = sub[sub["model"] == "no_carrier"]
        if w.empty or n.empty:
            continue
        stable_map[(str(k["phase"]), str(k["obs"]), str(k["input"]))] = bool(
            bool(w.iloc[0]["signed"]) == bool(n.iloc[0]["signed"])
        )
    vals = []
    for _, r in out.iterrows():
        key = (str(r["phase"]), str(r["obs"]), str(r["input"]))
        if key in stable_map:
            vals.append("Y" if stable_map[key] else "N")
        else:
            vals.append("")
    out["sign_stable"] = vals
    return out


def plant_slopes_table(p114d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    df = p114d.reset_index()
    df["phase_id"] = [_phase_of_row(p) for p in df["phase"]]
    names_with = ("intercept", "ln_O2", "ln_coal", "ln_carrier")
    names_no = ("intercept", "ln_O2", "ln_coal")
    seed_base = BOOT_SEED
    for phase in PHASES:
        sub = df[df["phase_id"] == phase].copy()
        n = len(sub)
        ln_o2 = np.log(sub["o2_fy1103a_nm3_h_mean"].to_numpy(dtype=float))
        ln_coal = np.log(sub["coal_dry_recon_phase_kg_s"].to_numpy(dtype=float))
        ln_car = np.log(sub["co2_carrier_measured_nm3_h"].to_numpy(dtype=float))
        Xw = np.column_stack([np.ones(n), ln_o2, ln_coal, ln_car])
        Xn = np.column_stack([np.ones(n), ln_o2, ln_coal])
        cond_w = _design_cond(Xw)
        cond_n = _design_cond(Xn)
        ymap = {obs: sub[col].to_numpy(dtype=float) for obs, col in PLANT_Y.items()}
        ymap["dry_per_coal_mol_kg"] = np.array(
            [_plant_dry_per_coal(str(w)) for w in sub["window_id"]], dtype=float
        )
        for model, X, names, cond_this in (
            ("with_carrier", Xw, names_with, cond_w),
            ("no_carrier", Xn, names_no, cond_n),
        ):
            for obs in OBS_ALL:
                y = ymap[obs]
                ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
                if ok.sum() < X.shape[1] + 1:
                    for inp in names[1:]:
                        rows.append({
                            "phase": phase, "obs": obs, "input": inp, "model": model,
                            "n": int(ok.sum()), "beta_ln": float("nan"),
                            "se_ln": float("nan"), "per3": float("nan"),
                            "se_per3": float("nan"), "boot_lo": float("nan"),
                            "boot_hi": float("nan"), "signed": False,
                            "used_in_R1": False,
                            "cond_X": cond_this,
                            "cond_X_with_carrier": cond_w,
                            "cond_X_no_carrier": cond_n,
                            "note": "n too small or nonfinite",
                        })
                    continue
                beta, se = _ols(X[ok], y[ok])
                for j, inp in enumerate(names):
                    if j == 0:
                        continue
                    per3 = float(beta[j]) * LN_P
                    se3 = float(se[j]) * LN_P
                    # 窗口自助：种子写入 protocol；每个 (phase,obs,model,input) 用独立流
                    col_seed = seed_base + 17 * (
                        PHASES.index(phase) + 1
                    ) + 101 * OBS_ALL.index(obs) + 7 * (0 if model == "with_carrier" else 1) + j
                    lo, hi = _bootstrap_per3(X[ok], y[ok], j, col_seed)
                    signed = bool(math.isfinite(lo) and math.isfinite(hi) and not (lo <= 0.0 <= hi))
                    used = bool(
                        model == "with_carrier"
                        and obs in OBS_R
                        and inp in INPUTS
                        and signed
                    )
                    note = ""
                    if obs == "X_C_pp":
                        note = "X_C 与煤碳公共分母，伪相关；不用于 R1–R4"
                    rows.append({
                        "phase": phase, "obs": obs, "input": inp, "model": model,
                        "n": int(ok.sum()), "beta_ln": float(beta[j]),
                        "se_ln": float(se[j]), "per3": per3, "se_per3": se3,
                        "boot_lo": lo, "boot_hi": hi, "signed": signed,
                        "used_in_R1": used, "boot_n": BOOT_N, "boot_seed": col_seed,
                        "cond_X": cond_this,
                        "cond_X_with_carrier": cond_w,
                        "cond_X_no_carrier": cond_n,
                        "note": note,
                    })
    return _mark_sign_stable(pd.DataFrame(rows))


def _coal_for_snap(snap: dict, pert: str, coal_nom: float | None) -> float:
    """单位煤量产率分母：该扰动快照的实际煤量，而不是名义煤量。"""
    v = _f((snap or {}).get("coal_dry_kg_s"))
    if math.isfinite(v) and v > 0.0:
        return v
    nom = _f(coal_nom)
    if math.isfinite(nom) and nom > 0.0:
        return float(PERT_SCALE[pert][1]) * nom
    return float("nan")


def _y_for_post(snap: dict, cand: dict, coal: float) -> dict | None:
    if not snap.get("ok"):
        return None
    if not (math.isfinite(coal) and coal > 0.0):
        return None
    kind = cand["kind"]
    if kind in ("C0", "C2", "C3", "C4"):
        return y_c0(snap, cand["xi"], coal)
    if kind in ("C1", "C5"):
        return y_c1(snap, cand["T_q"], cand["r_w"], coal)
    raise KeyError(kind)


def window_fd(snaps: dict, cand: dict, coal: float | None = None) -> dict:
    ys = {}
    for p in PERTS:
        snap = snaps.get(p) or {}
        ys[p] = _y_for_post(snap, cand, _coal_for_snap(snap, p, coal))
    nom = ys["nom"]
    out = {
        "ok_nom": nom is not None,
        "y": nom,
        "xc_ceil": bool((snaps.get("nom") or {}).get("X_C_at_ceiling")),
        "wsr_C_CO2_LimitedByCO2": bool((snaps.get("nom") or {}).get("wsr_C_CO2_LimitedByCO2")),
        "wsr_raw_C_CO2": _f((snaps.get("nom") or {}).get("wsr_raw_C_CO2")),
        "wsr_raw_C_H2O": _f((snaps.get("nom") or {}).get("wsr_raw_C_H2O")),
        "limit_len_frac_C_CO2_LimitedByCO2": _f(
            (snaps.get("nom") or {}).get("limit_len_frac_C_CO2_LimitedByCO2")
        ),
    }
    for inp, pkey, mkey in (
        ("ln_O2", "o2_p", "o2_m"),
        ("ln_coal", "coal_p", "coal_m"),
    ):
        for obs in OBS_ALL:
            sl, p3 = fd_from_ys(ys[pkey], ys[mkey], obs)
            out[f"dln_{obs}__{inp}"] = sl
            out[f"per3_{obs}__{inp}"] = p3
    return out


def _alias_group(cand: dict) -> str:
    """C2/C3/C4 名义档复用 C0 反应器组。"""
    if cand["kind"] == "C0":
        return "C0"
    if cand["kind"] == "C2":
        return "C0" if abs(cand["f_CO2"] - 1.0) < 1e-15 else c2_id(cand["f_CO2"])
    if cand["kind"] == "C3":
        return "C0" if abs(cand["L_wsr"] - L_WSR_NOM) < 1e-12 else c3_id(cand["L_wsr"])
    if cand["kind"] == "C4":
        return "C0" if abs(cand["HL_tot_pct"] - HL_NOM) < 1e-12 else c4_id(cand["HL_tot_pct"])
    if cand["kind"] == "C1":
        return "C0"
    if cand["kind"] == "C5":
        return cand["reactor_group"]
    raise KeyError(cand["kind"])


def candidate_catalog() -> list[dict]:
    rows = [{
        "id": "C0", "kind": "C0", "label": "基准（常数 ξ）",
        "f_CO2": None, "L_wsr": L_WSR_NOM, "HL_tot_pct": HL_NOM,
        "T_q": None, "r_w": None, "xi": None, "reuse_c0": False,
    }]
    for tq in C1_TQ:
        for rw in C1_RW:
            rows.append({
                "id": c1_id(tq, rw), "kind": "C1",
                "label": f"激冷冻结平衡 T_q={int(tq)} K r_w={rw:g}",
                "f_CO2": None, "L_wsr": L_WSR_NOM, "HL_tot_pct": HL_NOM,
                "T_q": float(tq), "r_w": float(rw), "xi": None, "reuse_c0": True,
            })
    for f in C2_LEVELS:
        rows.append({
            "id": c2_id(f), "kind": "C2",
            "label": f"C+CO2 指前因子 × {f:g}",
            "f_CO2": float(f), "L_wsr": L_WSR_NOM, "HL_tot_pct": HL_NOM,
            "T_q": None, "r_w": None, "xi": None,
            "reuse_c0": abs(f - 1.0) < 1e-15,
        })
    for l in C3_LEVELS:
        rows.append({
            "id": c3_id(l), "kind": "C3",
            "label": f"L_wsr={l:g} m",
            "f_CO2": None, "L_wsr": float(l), "HL_tot_pct": HL_NOM,
            "T_q": None, "r_w": None, "xi": None,
            "reuse_c0": abs(l - L_WSR_NOM) < 1e-12,
        })
    for hl in C4_LEVELS:
        rows.append({
            "id": c4_id(hl), "kind": "C4",
            "label": f"HL_tot={hl:g}%",
            "f_CO2": None, "L_wsr": L_WSR_NOM, "HL_tot_pct": float(hl),
            "T_q": None, "r_w": None, "xi": None,
            "reuse_c0": abs(hl - HL_NOM) < 1e-12,
        })
    return rows


def reactor_specs_for(cands: list[dict], wids: list[str], stopped: set[str]) -> list[dict]:
    specs = []
    seen = set()
    for wid in wids:
        phase = _phase_of_row(_CTX["p114d"].loc[wid, "phase"])
        if phase in stopped:
            continue
        for cand in cands:
            if cand["kind"] in ("C1", "C5"):
                continue
            if cand.get("reuse_c0"):
                continue
            group = _alias_group(cand)
            key = (wid, group)
            if key in seen:
                continue
            seen.add(key)
            specs.append({
                "window_id": wid,
                "group": group,
                "L_wsr": cand["L_wsr"],
                "HL_tot_pct": cand["HL_tot_pct"],
                "f_CO2": cand["f_CO2"],
            })
    return specs


def run_specs(specs: list[dict], n_workers: int, cache_only: bool = False) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    if not specs:
        return out
    _CTX["cache_only"] = bool(cache_only)
    n_workers = max(1, min(int(n_workers), len(specs)))
    print(f"START jobs={len(specs)} n_workers={n_workers} cache_only={cache_only}", flush=True)
    if n_workers == 1:
        for i, spec in enumerate(specs, 1):
            rec = _job(spec)
            out[(str(spec["window_id"]), str(spec["group"]))] = rec
            print(f"progress {i}/{len(specs)} {rec.get('group')} ok_nom={rec.get('ok_nom')}", flush=True)
        return out
    with ProcessPoolExecutor(
        max_workers=n_workers, initializer=_init_worker, initargs=(cache_only,),
    ) as ex:
        fut = {ex.submit(_job, spec): spec for spec in specs}
        done = 0
        for f in as_completed(fut):
            rec = f.result()
            spec = fut[f]
            out[(str(spec["window_id"]), str(spec["group"]))] = rec
            done += 1
            print(
                f"progress {done}/{len(specs)} {_short(str(rec.get('window_id')))} "
                f"{rec.get('group')} ok_nom={rec.get('ok_nom')}",
                flush=True,
            )
    return out


def _fill_xi(cands: list[dict]) -> list[dict]:
    out = []
    for c in cands:
        cc = dict(c)
        if cc["kind"] in ("C0", "C2", "C3", "C4"):
            cc["xi_mode"] = "phase_constant"
        else:
            cc["xi_mode"] = "replaced_by_quench"
        out.append(cc)
    return out


def signed_index(plant: pd.DataFrame) -> pd.DataFrame:
    return plant[(plant["used_in_R1"]) & (plant["model"] == "with_carrier")].copy()


def evaluate_candidate(
    cand: dict,
    store: dict[tuple[str, str], dict],
    p114d: pd.DataFrame,
    plant: pd.DataFrame,
    stopped: set[str],
) -> tuple[pd.DataFrame, dict, list[dict]]:
    """返回 (model_slopes rows, summary, window-level C1/H2O 记录)。"""
    signed = signed_index(plant)
    group = _alias_group(cand)
    slope_rows = []
    win_notes = []
    level_bias = {obs: [] for obs in OBS_ALL}
    n_h2o_neg = 0
    n_c1_infeas = 0
    n_nom_ok = 0
    n_nom_fail = 0
    per_phase_ok = {ph: [0, 0] for ph in PHASES}  # ok, fail
    fd_collect: dict[tuple[str, str, str], list[float]] = {}
    bind_collect: dict[str, list[float]] = {ph: [] for ph in PHASES}
    ceil_collect: dict[str, list[float]] = {ph: [] for ph in PHASES}
    raw_co2: dict[str, list[float]] = {ph: [] for ph in PHASES}

    for wid, row in p114d.iterrows():
        phase = _phase_of_row(row["phase"])
        if phase in stopped:
            continue
        pack = store.get((str(wid), group))
        if pack is None:
            n_nom_fail += 1
            per_phase_ok[phase][1] += 1
            continue
        snaps = pack.get("snaps") or {}
        nom = snaps.get("nom") or {}
        if not nom.get("ok"):
            n_nom_fail += 1
            per_phase_ok[phase][1] += 1
            continue
        th = PHASE_THETA[phase]
        cand_x = dict(cand)
        cand_x["xi"] = th["xi"]
        coal = _coal_dry(str(wid))
        if cand["kind"] in ("C1", "C5"):
            q = wgs_quench(nom, cand["T_q"], cand["r_w"])
            rec_n = {
                "window_id": wid, "phase": phase, "candidate": cand["id"],
                "H2O_eq_mol_s": q.get("H2O_eq_mol_s"),
                "H2O_positive": q.get("H2O_positive"),
                "feasible": q.get("feasible"),
                "infeasible_reason": q.get("infeasible_reason"),
                "xi_eq": q.get("xi_eq"),
                "H2O_wet_pct": q.get("H2O_wet_pct"),
                "CO_wet_pct": q.get("CO_wet_pct"),
                "H2_wet_pct": q.get("H2_wet_pct"),
                "CO2_wet_pct": q.get("CO2_wet_pct"),
            }
            win_notes.append(rec_n)
            if not q.get("H2O_positive"):
                n_h2o_neg += 1
            if not q.get("feasible"):
                n_c1_infeas += 1
                n_nom_fail += 1
                per_phase_ok[phase][1] += 1
                continue
        fd = window_fd(snaps, cand_x, coal)
        if not fd["ok_nom"]:
            n_nom_fail += 1
            per_phase_ok[phase][1] += 1
            continue
        n_nom_ok += 1
        per_phase_ok[phase][0] += 1
        y = fd["y"]
        plant_y = {
            "X_C_pp": float(row["xc_recon_phase_cal_measured_pct"]),
            "CO_vol_pct": float(row["co_vol_pct_mean"]),
            "H2_vol_pct": float(row["h2_vol_pct_mean"]),
            "CO2_vol_pct": float(row["co2_vol_pct_mean"]),
            "CH4_vol_pct": float(row["ch4_vol_pct_mean"]),
            "dry_per_coal_mol_kg": _plant_dry_per_coal(str(wid)),
        }
        for obs in OBS_ALL:
            level_bias[obs].append(float(y[obs]) - plant_y[obs])
        bind_collect[phase].append(1.0 if fd["wsr_C_CO2_LimitedByCO2"] else 0.0)
        ceil_collect[phase].append(1.0 if fd["xc_ceil"] else 0.0)
        raw_co2[phase].append(fd["wsr_raw_C_CO2"])
        for inp in INPUTS:
            for obs in OBS_ALL:
                fd_collect.setdefault((phase, obs, inp), []).append(
                    fd[f"per3_{obs}__{inp}"]
                )

    feasible = True
    infeas_reason = ""
    if cand["kind"] in ("C1", "C5") and n_h2o_neg > 0:
        feasible = False
        infeas_reason = f"H2O_eq<0 on {n_h2o_neg} windows"
    if cand["kind"] in ("C1", "C5") and n_nom_ok == 0:
        feasible = False
        infeas_reason = infeas_reason or "no feasible windows"

    n_signed = 0
    n_cons = 0
    ratios = []
    for _, r in signed.iterrows():
        if r["phase"] in stopped:
            continue
        n_signed += 1
        vals = [v for v in fd_collect.get((r["phase"], r["obs"], r["input"]), []) if math.isfinite(v)]
        if not vals:
            continue
        med = float(np.median(vals))
        plant_s = float(r["per3"])
        same = bool(med != 0.0 and plant_s != 0.0 and np.sign(med) == np.sign(plant_s))
        if same:
            n_cons += 1
            if plant_s != 0.0:
                ratios.append(med / plant_s)

    for phase in PHASES:
        if phase in stopped:
            continue
        for obs in OBS_ALL:
            for inp in INPUTS:
                vals = [v for v in fd_collect.get((phase, obs, inp), []) if math.isfinite(v)]
                plant_hit = plant[
                    (plant["phase"] == phase) & (plant["obs"] == obs)
                    & (plant["input"] == inp) & (plant["model"] == "with_carrier")
                ]
                plant_s = float(plant_hit.iloc[0]["per3"]) if len(plant_hit) else float("nan")
                signed_flag = bool(len(plant_hit) and bool(plant_hit.iloc[0]["used_in_R1"]))
                med = float(np.median(vals)) if vals else float("nan")
                same = (
                    signed_flag and math.isfinite(med) and math.isfinite(plant_s)
                    and med != 0.0 and plant_s != 0.0 and np.sign(med) == np.sign(plant_s)
                )
                slope_rows.append({
                    "candidate": cand["id"], "kind": cand["kind"], "label": cand["label"],
                    "phase": phase, "obs": obs, "input": inp,
                    "n_ok": len(vals),
                    "model_per3_median": med,
                    "model_per3_min": float(np.min(vals)) if vals else float("nan"),
                    "model_per3_max": float(np.max(vals)) if vals else float("nan"),
                    "plant_per3": plant_s,
                    "signed_plant": signed_flag,
                    "same_sign": same,
                    "xc_ceil_frac": float(np.mean(ceil_collect[phase])) if ceil_collect[phase] else float("nan"),
                    "wsr_C_CO2_bind_frac": float(np.mean(bind_collect[phase])) if bind_collect[phase] else float("nan"),
                    "wsr_raw_C_CO2_median": float(np.nanmedian(raw_co2[phase])) if raw_co2[phase] else float("nan"),
                    "feasible": feasible,
                })

    def _med(xs):
        xs = [x for x in xs if math.isfinite(x)]
        return float(np.median(xs)) if xs else float("nan")

    summary = {
        "candidate": cand["id"], "kind": cand["kind"], "label": cand["label"],
        "feasible": feasible, "infeasible_reason": infeas_reason,
        "n_nom_ok": n_nom_ok, "n_nom_fail": n_nom_fail,
        "n_h2o_neg": n_h2o_neg, "n_c1_infeas": n_c1_infeas,
        "n_signed": n_signed, "n_consistent": n_cons,
        "r2_ratio": (n_cons / n_signed) if n_signed else float("nan"),
        "r3_median": _med(ratios),
        "r3_min": float(np.min(ratios)) if ratios else float("nan"),
        "r3_max": float(np.max(ratios)) if ratios else float("nan"),
        "r5_d_X_C_pp": _med(level_bias["X_C_pp"]),
        "r5_d_CO_vol_pct": _med(level_bias["CO_vol_pct"]),
        "r5_d_H2_vol_pct": _med(level_bias["H2_vol_pct"]),
        "r5_d_CO2_vol_pct": _med(level_bias["CO2_vol_pct"]),
        "r5_d_CH4_vol_pct": _med(level_bias["CH4_vol_pct"]),
        "r5_d_dry_per_coal": _med(level_bias["dry_per_coal_mol_kg"]),
        "wsr_C_CO2_bind_frac": float(np.nanmean(
            [x for xs in bind_collect.values() for x in xs]
        )) if any(bind_collect.values()) else float("nan"),
        "xc_ceil_frac": float(np.nanmean(
            [x for xs in ceil_collect.values() for x in xs]
        )) if any(ceil_collect.values()) else float("nan"),
        "f_CO2": cand.get("f_CO2"), "L_wsr": cand.get("L_wsr"),
        "HL_tot_pct": cand.get("HL_tot_pct"),
        "T_q": cand.get("T_q"), "r_w": cand.get("r_w"),
        "n_ok_phase1": per_phase_ok["phase1"][0], "n_fail_phase1": per_phase_ok["phase1"][1],
        "n_ok_phase2": per_phase_ok["phase2"][0], "n_fail_phase2": per_phase_ok["phase2"][1],
        "n_ok_phase3": per_phase_ok["phase3"][0], "n_fail_phase3": per_phase_ok["phase3"][1],
        "n_ok_phase4": per_phase_ok["phase4"][0], "n_fail_phase4": per_phase_ok["phase4"][1],
    }
    return pd.DataFrame(slope_rows), summary, win_notes


def pick_best(summaries: list[dict], kind: str) -> dict | None:
    """R2 一致比例（全部有符号组合总和）；并列先 R3 中位更近 1，再 |R5| 更小。"""
    pool = [s for s in summaries if s["kind"] == kind and s.get("feasible")]
    if not pool:
        return None

    def key(s):
        r2 = s.get("r2_ratio")
        r2 = -1.0 if not math.isfinite(_f(r2)) else float(r2)
        r3 = s.get("r3_median")
        r3pen = 1e6 if not math.isfinite(_f(r3)) else abs(float(r3) - 1.0)
        r5 = np.nansum(np.abs([
            _f(s.get("r5_d_CO_vol_pct")), _f(s.get("r5_d_H2_vol_pct")),
            _f(s.get("r5_d_CO2_vol_pct")), _f(s.get("r5_d_CH4_vol_pct")),
            _f(s.get("r5_d_X_C_pp")),
        ]))
        return (-r2, r3pen, float(r5))

    return min(pool, key=key)


def build_readings(
    plant: pd.DataFrame,
    model: pd.DataFrame,
    cands: pd.DataFrame,
    c0_phase: dict,
    stopped: set[str],
    c5_note: str,
) -> pd.DataFrame:
    rows = []
    signed = signed_index(plant)
    for _, r in signed.iterrows():
        rows.append({
            "reading": "R1", "metric": "signed", "phase": r["phase"],
            "subset": r["obs"], "window_set": r["input"], "window_id": "",
            "value": r["per3"], "threshold": 0.0, "pass": True,
            "note": (
                f"装置每+3%={r['per3']:.5g} 自助95% [{r['boot_lo']:.5g},{r['boot_hi']:.5g}] "
                f"β_ln={r['beta_ln']:.5g}"
            ),
            "exploratory": False, "candidate": "",
        })
    n_signed = int(len(signed[~signed["phase"].isin(stopped)]))
    rows.append({
        "reading": "R1", "metric": "n_signed", "phase": "all", "subset": "",
        "window_set": "ln_O2,ln_coal", "window_id": "",
        "value": n_signed, "threshold": "", "pass": "",
            "note": (
                "有符号=给定含载气 OLS 与窗口自助假设下区间不含 0；"
                "(观测,输入,阶段)；X_C 不计入。不写成装置斜率可信"
            ),
        "exploratory": False, "candidate": "",
    })
    for _, s in cands.iterrows():
        rows.append({
            "reading": "R2", "metric": "consistent_ratio", "phase": "all",
            "subset": s["candidate"], "window_set": "", "window_id": "",
            "value": s["r2_ratio"], "threshold": 0.8,
            "pass": bool(s["feasible"] and _f(s["r2_ratio"]) >= 0.8),
            "note": (
                f"{int(s['n_consistent'])}/{int(s['n_signed'])} "
                f"feasible={s['feasible']} {s.get('infeasible_reason') or ''}"
            ).strip(),
            "exploratory": False, "candidate": s["candidate"],
        })
        rows.append({
            "reading": "R3", "metric": "ratio_median", "phase": "all",
            "subset": s["candidate"], "window_set": "", "window_id": "",
            "value": s["r3_median"], "threshold": "", "pass": "",
            "note": f"range [{s['r3_min']},{s['r3_max']}] 仅一致组合",
            "exploratory": False, "candidate": s["candidate"],
        })
        for obs, col in (
            ("X_C_pp", "r5_d_X_C_pp"), ("CO_vol_pct", "r5_d_CO_vol_pct"),
            ("H2_vol_pct", "r5_d_H2_vol_pct"), ("CO2_vol_pct", "r5_d_CO2_vol_pct"),
            ("CH4_vol_pct", "r5_d_CH4_vol_pct"),
            ("dry_per_coal_mol_kg", "r5_d_dry_per_coal"),
        ):
            rows.append({
                "reading": "R5", "metric": f"bias_{obs}", "phase": "all",
                "subset": s["candidate"], "window_set": "", "window_id": "",
                "value": s[col], "threshold": "", "pass": "",
                "note": "模型−装置，阶段窗中位",
                "exploratory": False, "candidate": s["candidate"],
            })
    feas = cands[cands["feasible"].astype(bool)]
    hit = feas[feas["r2_ratio"] >= 0.8] if not feas.empty else feas
    scope = (
        "比较口径限于当前候选、固定阶段 θ̂、装置含载气 OLS、模型两侧 ±3% 控制扰动；"
        "装置为条件关联，模型为控制扰动。不得推广为任何参数反演都不能外推。"
    )
    if hit is not None and not hit.empty:
        best = hit.sort_values("r2_ratio", ascending=False).iloc[0]
        r4_note = f"在该口径下 {best['candidate']} 使 R2={best['r2_ratio']:.3f}≥80%。{scope}"
        r4_pass = True
        r4_val = best["candidate"]
    else:
        r4_note = f"在该口径下现有候选无一使 R2≥80%。{scope}"
        r4_pass = False
        r4_val = ""
    rows.append({
        "reading": "R4", "metric": "attribution", "phase": "all",
        "subset": r4_val, "window_set": "", "window_id": "",
        "value": 1.0 if r4_pass else 0.0, "threshold": 0.8, "pass": r4_pass,
        "note": r4_note + (("；" + c5_note) if c5_note else ""),
        "exploratory": False, "candidate": r4_val,
    })
    for ph, rec in c0_phase.items():
        rows.append({
            "reading": "C0", "metric": "n_success", "phase": ph,
            "subset": "C0", "window_set": "", "window_id": "",
            "value": rec["n_ok"], "threshold": "", "pass": not rec["stopped"],
            "note": f"fail={rec['n_fail']} stopped={rec['stopped']}",
            "exploratory": False, "candidate": "C0",
        })
    for ph in stopped:
        rows.append({
            "reading": "stop", "metric": "stage_stopped", "phase": ph,
            "subset": "C0", "window_set": "", "window_id": "",
            "value": 1.0, "threshold": FAIL_WINDOW_STOP, "pass": False,
            "note": f"C0 失败窗 > {FAIL_WINDOW_STOP}，停止该阶段",
            "exploratory": False, "candidate": "C0",
        })
    return pd.DataFrame(rows)


def _fmt(x, nd=4):
    v = _f(x)
    if not math.isfinite(v):
        return "NA"
    return f"{v:.{nd}g}"


def interpretation(
    plant: pd.DataFrame,
    model: pd.DataFrame,
    cands: pd.DataFrame,
    readings: pd.DataFrame,
    c0_phase: dict,
    stopped: set[str],
    c1_bad: list[str],
    c5_note: str,
    c2_check: dict,
) -> str:
    L = []
    L.append("# P1.15-A 工况响应斜率诊断与机理候选（2026-11-08）")
    L.append("")
    L.append("本轮只诊断装置与模型在 θ̂ 处对 O2、煤量的组成响应方向，并扫描机理候选。")
    L.append("**不反演、不采纳任何参数。**")
    L.append("")
    L.append("## 求解覆盖（C0）")
    for ph in PHASES:
        rec = c0_phase.get(ph, {"n_ok": 0, "n_fail": 0, "n_total": 0, "stopped": False})
        flag = "停止该阶段" if rec.get("stopped") else "继续"
        L.append(
            f"- {ph}：成功 {rec.get('n_ok', 0)} / 失败 {rec.get('n_fail', 0)} "
            f"/ 共 {rec.get('n_total', 0)}。{flag}。"
        )
    if stopped:
        L.append(f"- 触发停止的阶段：{', '.join(sorted(stopped))}。")
    else:
        L.append("- 未触发分阶段停止。")
    n_fail_tot = sum(int(rec.get("n_fail", 0) or 0) for rec in c0_phase.values())
    if n_fail_tot:
        L.append(
            f"- C0 失败合计 {n_fail_tot} 窗。既有 cache 用 wid[-16:] 作文件名，"
            "交接班窗口会互相覆盖；本轮核对 JSON 内 window_id，缺档不求解、不复用他窗快照。"
        )
    L.append("")
    L.append("## 第 1 节 装置斜率（含载气；每 +3% = β_ln × ln(1.03)）")
    L.append("主模型：y ~ ln O2 + ln 干煤 + ln 实测载气。窗口自助 1000 次，种子见 protocol。")
    L.append("装置侧是阶段内有限窗口、给定协变量下的条件关联；窗口自助区间只反映该回归与重采样假设，")
    L.append("未识别时序相关、遗漏变量和内生性。有符号 = 该回归下区间不含 0，不写成「装置斜率可信」。")
    L.append("模型侧是固定其余输入的 ±3% 控制扰动。二者不是同一识别策略。")
    L.append("X_C 因与煤碳公共分母而不进入 R1–R4。并列不含载气的版本见 `plant_slopes.csv`。")
    L.append("单位煤量干气的模型斜率分母为各扰动快照的 `coal_dry_kg_s`，不是名义煤量。")
    L.append("")
    sub = plant[(plant["model"] == "with_carrier") & (plant["input"].isin(INPUTS))]
    for phase in PHASES:
        L.append(f"### {phase}")
        hit = sub[sub["phase"] == phase]
        for _, r in hit.iterrows():
            sig = "有符号" if r["signed"] else "含 0"
            L.append(
                f"- {r['obs']} ← {r['input']}：每+3%={_fmt(r['per3'])} "
                f"(se={_fmt(r['se_per3'])}) 自助95% [{_fmt(r['boot_lo'])},{_fmt(r['boot_hi'])}] "
                f"**{sig}**。"
            )
        L.append("")
    L.append("## 设计矩阵条件数与载气符号稳定性")
    L.append("cond(X) 为该阶段协变量矩阵（含截距）的 2-范数条件数；符号稳定 = 含载气 / 不含载气两种回归的 signed 一致。")
    for phase in PHASES:
        hit = plant[plant["phase"] == phase]
        if hit.empty:
            continue
        cw = _fmt(hit["cond_X_with_carrier"].iloc[0])
        cn = _fmt(hit["cond_X_no_carrier"].iloc[0])
        pairs = hit[
            (hit["model"] == "with_carrier")
            & (hit["input"].isin(INPUTS))
            & (hit["obs"].isin(OBS_R))
        ]
        n_stable = int((pairs["sign_stable"] == "Y").sum())
        n_pairs = int(len(pairs))
        L.append(f"- {phase}：cond(X_with_carrier)={cw}，cond(X_no_carrier)={cn}；")
        L.append(f"  OBS_R×{{ln_O2,ln_coal}} 符号稳定 {n_stable}/{n_pairs}。")
        unstable = pairs[pairs["sign_stable"] == "N"]
        for _, r in unstable.iterrows():
            L.append(
                f"  - 不稳定：{r['obs']} ← {r['input']} "
                f"（含载气 signed={bool(r['signed'])}）"
            )
    L.append("")
    L.append("## 第 2 节 基准模型斜率（C0，两侧 ±3%）")
    m0 = model[model["candidate"] == "C0"]
    for phase in PHASES:
        L.append(f"### {phase}")
        hit = m0[(m0["phase"] == phase) & (m0["obs"].isin(OBS_R))]
        if hit.empty:
            L.append("- 无可用窗。")
            L.append("")
            continue
        ceil = hit["xc_ceil_frac"].iloc[0]
        bind = hit["wsr_C_CO2_bind_frac"].iloc[0]
        L.append(
            f"- X_C 贴顶（≥99.9%）窗分数={_fmt(ceil)}；"
            f"全混区 `C_CO2_LimitedByCO2` 绑定分数={_fmt(bind)}。"
        )
        for _, r in hit.iterrows():
            L.append(
                f"- {r['obs']} ← {r['input']}：中位每+3%={_fmt(r['model_per3_median'])} "
                f"范围 [{_fmt(r['model_per3_min'])},{_fmt(r['model_per3_max'])}] "
                f"n={int(r['n_ok'])}；装置={_fmt(r['plant_per3'])}。"
            )
        L.append("")
    L.append("## C1 不可行档")
    if c1_bad:
        for name in c1_bad:
            L.append(f"- {name}：平衡后 H2O<0 或无可行窗，记为 infeasible。")
    else:
        L.append("- 无。全部 T_q×r_w 档在成功窗上 H2O_eq>0。")
    L.append("")
    L.append("## C2 钩子核对")
    L.append(
        f"- f_CO2=1 与 C0 出口一致：{c2_check.get('f1_matches_c0')}。"
        f"{c2_check.get('f1_note', '')}"
    )
    L.append(
        f"- 抽查窗 f_CO2=1/256 的全混区 C+CO2 毛速率下降：{c2_check.get('f256_rate_down')}。"
        f"{c2_check.get('f256_note', '')}"
    )
    L.append(f"- 全混区 C+CO2 限幅是否随 f_CO2 解除：{c2_check.get('unbind_note', '')}")
    L.append("")
    L.append("## R1 给定回归与窗口自助假设下区间不含 0（有符号组合）")
    n_signed = int(len(signed_index(plant)[~signed_index(plant)["phase"].isin(stopped)]))
    L.append(
        f"- 有符号组合数（含载气主模型，X_C 除外）= {n_signed}。"
        "这只说明该回归与窗口自助假设下区间不含 0，不写成装置斜率可信。逐条见 `readings.csv` / 第 1 节。"
    )
    L.append("")
    L.append("## R2 方向一致（一致数 / 有符号数）")
    for _, s in cands.iterrows():
        flag = "" if s["feasible"] else " [不可行]"
        L.append(
            f"- {s['candidate']}{flag}：{int(s['n_consistent'])}/{int(s['n_signed'])} "
            f"= {_fmt(s['r2_ratio'], 3)}。{s.get('infeasible_reason') or ''}"
        )
    L.append("")
    L.append("## R3 量级（一致组合中 模型/装置）")
    for _, s in cands.iterrows():
        if not s["feasible"]:
            continue
        L.append(
            f"- {s['candidate']}：中位={_fmt(s['r3_median'])} "
            f"范围 [{_fmt(s['r3_min'])},{_fmt(s['r3_max'])}]"
        )
    L.append("")
    L.append("## R4 机理归属（限于当前比较口径）")
    L.append("口径：当前候选、固定阶段 θ̂、装置含载气 OLS、模型两侧 ±3% 控制扰动。")
    L.append("不得把该口径下的方向比较推广为「任何参数反演都不能外推」。")
    r4 = readings[(readings["reading"] == "R4") & (readings["metric"] == "attribution")]
    if not r4.empty:
        L.append(f"- {r4.iloc[0]['note']}")
    if c5_note:
        L.append(f"- C5：{c5_note}")
    L.append("")
    L.append("## R5 水平代价（模型−装置，相对 C0）")
    c0 = cands[cands["candidate"] == "C0"]
    c0r = c0.iloc[0] if not c0.empty else None
    L.append("中位偏差（模型−装置）。正值=模型高于装置。")
    for _, s in cands.iterrows():
        bits = [
            f"X_C={_fmt(s['r5_d_X_C_pp'])} pp",
            f"CO={_fmt(s['r5_d_CO_vol_pct'])}",
            f"H2={_fmt(s['r5_d_H2_vol_pct'])}",
            f"CO2={_fmt(s['r5_d_CO2_vol_pct'])}",
            f"CH4={_fmt(s['r5_d_CH4_vol_pct'])}",
        ]
        extra = ""
        if c0r is not None and s["candidate"] != "C0":
            extra = (
                f"；相对 C0：ΔX_C={_fmt(_f(s['r5_d_X_C_pp']) - _f(c0r['r5_d_X_C_pp']))} "
                f"ΔCO={_fmt(_f(s['r5_d_CO_vol_pct']) - _f(c0r['r5_d_CO_vol_pct']))} "
                f"ΔH2={_fmt(_f(s['r5_d_H2_vol_pct']) - _f(c0r['r5_d_H2_vol_pct']))} "
                f"ΔCO2={_fmt(_f(s['r5_d_CO2_vol_pct']) - _f(c0r['r5_d_CO2_vol_pct']))}"
            )
        L.append(f"- {s['candidate']}：{', '.join(bits)}{extra}")
    L.append("")
    L.append("本轮不采纳 κ、ξ、f_gas、f_CO2、L_wsr、HL_tot、T_q、r_w 的任何数值，不改合同。")
    L.append("")
    return "\n".join(L) + "\n"


def _plant_design_block(plant: pd.DataFrame | None) -> dict:
    out = {}
    if plant is None or plant.empty:
        return out
    for phase in PHASES:
        hit = plant[plant["phase"] == phase]
        if hit.empty or "cond_X_with_carrier" not in hit.columns:
            continue
        pairs = hit[
            (hit["model"] == "with_carrier")
            & (hit["input"].isin(INPUTS))
            & (hit["obs"].isin(OBS_R))
        ]
        out[phase] = {
            "cond_X_with_carrier": _f(hit["cond_X_with_carrier"].iloc[0]),
            "cond_X_no_carrier": _f(hit["cond_X_no_carrier"].iloc[0]),
            "sign_stable_obs_input": int((pairs["sign_stable"] == "Y").sum()),
            "n_obs_input": int(len(pairs)),
        }
    return out


def _protocol(
    src_commit, stopped, c0_phase, c2_check, pick_rule, c5_note,
    plant=None, cache_only=False, n_cache=0,
) -> dict:
    sha_in = hashlib.sha256(P114D_CSV.read_bytes()).hexdigest() if P114D_CSV.exists() else ""
    return base_protocol({
        "round": "P1.15-A / 2026-11-08",
        "model_repo": "gasifier-1d-kinetic",
        "mesh_family": "wsr_plus_pfr",
        "source_commit": src_commit or source_commit(),
        "task_card": "papers/jax-gasifier-cascade-hybrid/manuscript/"
                     "model-side-handoff-p1.15a-20261108.md",
        "structure": {
            "mesh_family": "wsr_plus_pfr",
            "L_wsr_m": L_WSR_NOM,
            "dz_down_m": DZ_DOWN,
            "HL_tot_pct": HL_NOM,
            "oxygen_mixing": "instant",
            "frozen": [
                "全部 Arrhenius（除候选明确改动的 C+CO2 指前因子或 f_gas）",
                "限幅", "燃烧区判据",
                "煤量主口径 windows_142.coal_dry_recon_phase_kg_s",
                "载气 windows_142.co2_carrier_measured_nm3_h",
                "氧气 windows_142.o2_fy1103a_nm3_h_mean（装置）与 o2_flow_scale=κ（模型）",
                "求解器容差 jacobian_mode=scipy",
            ],
        },
        "parameters": {
            "theta_phase_analyzer_Lwsr1_HLtot1pct": PHASE_THETA,
            "xi": "C0/C2/C3/C4 用阶段常数 ξ，只作用于出口；C1/C5 用激冷平衡 ξ_eq 替代",
            "CharCO2RateFactor": "缺省 1.0；C2 为 f_gas * f_CO2 只乘 C+CO2",
            "adopted": False,
        },
        "plant_regression": {
            "y": list(OBS_ALL),
            "primary": "y ~ ln O2 + ln coal + ln carrier",
            "secondary": "y ~ ln O2 + ln coal",
            "per3": "beta_ln * ln(1.03)",
            "bootstrap": {"n": BOOT_N, "unit": "window", "seed_base": BOOT_SEED},
            "X_C_note": "公共分母（煤碳）伪相关，不用于 R1–R4",
            "identification": (
                "阶段内窗口 OLS 条件关联；不是控制扰动。"
                "有符号仅表示该回归与窗口自助假设下区间不含 0。"
            ),
            "design_cond": _plant_design_block(plant),
        },
        "model_fd": {
            "scheme": "two-sided ±3%",
            "d_ln": " (y(1.03)-y(0.97)) / ln(1.03/0.97) ",
            "per3": "slope_ln * ln(1.03)",
            "no_onesided": True,
            "dry_per_coal_denominator": (
                "each snap coal_dry_kg_s; fallback PERT_SCALE[pert][1] * nominal coal"
            ),
            "identification": "ceteris paribus ±3% control perturbation at frozen phase theta-hat",
        },
        "predeclared": {
            "R1": (
                "给定含载气 OLS 与窗口自助假设下，95% 区间不含 0 → 有符号；"
                "不写成装置斜率可信；只对有符号 (观测, ln O2 或 ln 煤量, 阶段) 比较；"
                "X_C 不进判据；CH4 若有符号则计入"
            ),
            "R2": "模型阶段中位每+3% 与装置每+3% 同号 → 一致；报 一致数/有符号数",
            "R3": "一致组合中 模型/装置 的中位与范围",
            "R4": (
                "在当前候选、固定阶段 θ̂、装置含载气 OLS、模型两侧 ±3% 控制扰动这一口径下，"
                "哪个候选使 R2≥80%；若无则记该口径下现有候选均不能复现装置响应方向。"
                "不得推广为任何反演都不能外推"
            ),
            "R5": "各候选在 θ̂ 处水平偏差（模型−装置）相对 C0 的变化",
        },
        "cache_only": bool(cache_only),
        "n_cache_json": int(n_cache),
        "c5_selection": pick_rule,
        "c5_note": c5_note,
        "c2_checks": c2_check,
        "stopped_stages": sorted(stopped),
        "c0_phase": c0_phase,
        "fail_stop_threshold_per_stage": FAIL_WINDOW_STOP,
        "n_workers": N_WORKERS_DEFAULT,
        "source_csv": {
            "p1.14d": {
                "round": "p1.14d-20261107",
                "file": "windows_142.csv",
                "relative_path": "outputs/diagnostics/p1.14d-20261107/windows_142.csv",
                "sha256": sha_in,
                "n_rows": 142,
            },
            "p1.14a_level_control": {
                "round": "p1.14a-20261107",
                "file": "baseline_142.csv",
                "relative_path": "outputs/diagnostics/round-p1.14a-20261107/baseline_142.csv",
                "note": "水平对照；不含出口 H2O，C1 用水由 C0 反应器快照提供",
            },
        },
        "outputs": [
            "plant_slopes.csv", "model_slopes.csv", "candidates.csv",
            "readings.csv", "protocol.json", "interpretation.md", "sha256.txt",
        ],
        "adopted_parameters": False,
    })


def _rewrite_manifest(out_dir: Path) -> None:
    skip = {"sha256.txt", "run.log", "run.log.prev"}
    files = sorted(
        p for p in out_dir.iterdir()
        if p.is_file() and p.name not in skip
    )
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}" for p in files]
    (out_dir / "sha256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _c2_checks(store: dict, wids: list[str]) -> dict:
    out = {
        "f1_matches_c0": False, "f1_note": "未跑独立 f_CO2=1 求解，C2_f1 复用 C0",
        "f256_rate_down": False, "f256_note": "",
        "unbind_note": "",
    }
    keys = ("F_CO", "F_CO2", "F_H2", "F_H2O", "X_C_pp", "T_out_K")
    match_n = 0
    cmp_n = 0
    down_n = 0
    cmp_r = 0
    bind0 = []
    bind256 = []
    gid256 = c2_id(1.0 / 256.0)
    for wid in wids:
        c0 = (store.get((wid, "C0")) or {}).get("snaps", {}).get("nom") or {}
        c2_1 = (store.get((wid, "C2_f1")) or {}).get("snaps", {}).get("nom")
        if c2_1 and c0.get("ok") and c2_1.get("ok"):
            cmp_n += 1
            if all(c0.get(k) == c2_1.get(k) for k in keys):
                match_n += 1
        c256 = (store.get((wid, gid256)) or {}).get("snaps", {}).get("nom") or {}
        if c0.get("ok"):
            bind0.append(1.0 if c0.get("wsr_C_CO2_LimitedByCO2") else 0.0)
        if c256.get("ok"):
            bind256.append(1.0 if c256.get("wsr_C_CO2_LimitedByCO2") else 0.0)
            r0, r1 = _f(c0.get("wsr_raw_C_CO2")), _f(c256.get("wsr_raw_C_CO2"))
            if math.isfinite(r0) and math.isfinite(r1):
                cmp_r += 1
                if r1 < r0:
                    down_n += 1
    if cmp_n:
        out["f1_matches_c0"] = match_n == cmp_n
        out["f1_note"] = f"{match_n}/{cmp_n} 窗出口逐位一致"
    if cmp_r:
        out["f256_rate_down"] = down_n == cmp_r
        out["f256_note"] = f"{down_n}/{cmp_r} 窗全混区 C+CO2 毛速率下降"
    if bind0 and bind256:
        m0, m1 = float(np.mean(bind0)), float(np.mean(bind256))
        out["unbind_note"] = (
            f"C0 绑定分数={m0:.3f}，f_CO2=1/256 绑定分数={m1:.3f}"
            + ("；绑定随 f_CO2 下降而解除" if m1 < m0 else "；绑定未随 f_CO2 下降而解除")
        )
    return out


def _count_c0(store, p114d) -> tuple[dict, set[str]]:
    c0_phase = {}
    stopped = set()
    for phase in PHASES:
        ids = [str(w) for w, r in p114d.iterrows() if _phase_of_row(r["phase"]) == phase]
        n_ok = n_fail = 0
        for wid in ids:
            pack = store.get((wid, "C0"))
            ok = bool(pack and pack.get("ok_nom"))
            if ok:
                n_ok += 1
            else:
                n_fail += 1
        stop = n_fail > FAIL_WINDOW_STOP
        if stop:
            stopped.add(phase)
        c0_phase[phase] = {
            "n_ok": n_ok, "n_fail": n_fail, "n_total": len(ids), "stopped": stop,
        }
    return c0_phase, stopped


def smoke() -> None:
    _init_worker()
    p114d = _CTX["p114d"]
    wid = str(p114d.index[0])
    phase = _phase_of_row(p114d.loc[wid, "phase"])
    th = PHASE_THETA[phase]
    print(f"SMOKE window={_short(wid)} {phase}", flush=True)
    spec0 = {"window_id": wid, "group": "C0_smoke", "L_wsr": L_WSR_NOM,
             "HL_tot_pct": HL_NOM, "f_CO2": None, "perts": ("nom",)}
    rec0 = reactor_group_job(spec0)
    snap = rec0["snaps"]["nom"]
    coal = _coal_dry(wid)
    y0 = y_c0(snap, th["xi"], coal) if snap.get("ok") else None
    c1_rows = []
    for tq in C1_TQ:
        for rw in C1_RW:
            q = wgs_quench(snap, tq, rw) if snap.get("ok") else {"feasible": False, "infeasible_reason": "no_snap"}
            c1_rows.append({
                "id": c1_id(tq, rw), "feasible": q.get("feasible"),
                "H2O_eq": q.get("H2O_eq_mol_s"), "reason": q.get("infeasible_reason"),
                "H2O_wet_pct": q.get("H2O_wet_pct"),
            })
    spec1 = {"window_id": wid, "group": "C2_f1_smoke", "L_wsr": L_WSR_NOM,
             "HL_tot_pct": HL_NOM, "f_CO2": 1.0, "perts": ("nom",)}
    rec1 = reactor_group_job(spec1)
    s1 = rec1["snaps"]["nom"]
    keys = ("F_CO", "F_CO2", "F_H2", "F_H2O", "X_C_pp", "T_out_K")
    match = bool(snap.get("ok") and s1.get("ok") and all(snap.get(k) == s1.get(k) for k in keys))
    print(json.dumps({
        "window_id": wid, "phase": phase,
        "c0_ok": rec0.get("ok_nom"), "c0_y": y0,
        "c1": c1_rows,
        "c2_f1_ok": rec1.get("ok_nom"),
        "c2_f1_matches_c0": match,
        "c0_F": {k: snap.get(k) for k in keys},
        "c2_f1_F": {k: s1.get(k) for k in keys},
        "wsr_raw_C_CO2_c0": snap.get("wsr_raw_C_CO2"),
        "wsr_C_CO2_LimitedByCO2": snap.get("wsr_C_CO2_LimitedByCO2"),
    }, ensure_ascii=False, default=str, indent=2), flush=True)
    if not match:
        raise SystemExit("smoke: C2 f_CO2=1 与 C0 出口不一致")
    print("SMOKE PASS", flush=True)


def write_all(plant, model, cand_df, readings, protocol, text) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_round_outputs(
        OUT_DIR,
        tables={
            "plant_slopes.csv": plant,
            "model_slopes.csv": model,
            "candidates.csv": cand_df,
            "readings.csv": readings,
        },
        protocol=protocol,
        interpretation=text,
    )
    _rewrite_manifest(OUT_DIR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=("all", "smoke", "read"))
    ap.add_argument("--source-commit", default=None)
    ap.add_argument("--n-workers", type=int, default=N_WORKERS_DEFAULT)
    ap.add_argument("--n-windows", type=int, default=0)
    ap.add_argument(
        "--cache-only", action="store_true",
        help="cache 未命中则跳过该组，绝不调用求解器",
    )
    args = ap.parse_args()
    silence_solver_logs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    n_cache = len(list(CACHE_DIR.glob("*.json")))
    print(
        f"preflight cache json={n_cache} cache_only={args.cache_only}",
        flush=True,
    )

    if args.stage == "smoke":
        smoke()
        return

    _init_worker(cache_only=args.cache_only)
    p114d = _CTX["p114d"]
    window_ids = [str(w) for w in p114d.index]
    if args.n_windows and args.n_windows > 0:
        window_ids = window_ids[: int(args.n_windows)]
        p114d = p114d.loc[window_ids]

    if args.stage == "read":
        plant = pd.read_csv(OUT_DIR / "plant_slopes.csv")
        model = pd.read_csv(OUT_DIR / "model_slopes.csv")
        cand_df = pd.read_csv(OUT_DIR / "candidates.csv")
        proto = json.loads((OUT_DIR / "protocol.json").read_text(encoding="utf-8"))
        stopped = set(proto.get("stopped_stages") or [])
        c0_phase = proto.get("c0_phase") or {}
        c2_check = proto.get("c2_checks") or {}
        c5_note = proto.get("c5_note") or ""
        c1_bad = [
            str(r["candidate"]) for _, r in cand_df.iterrows()
            if r["kind"] == "C1" and not bool(r["feasible"])
        ]
        readings = build_readings(plant, model, cand_df, c0_phase, stopped, c5_note)
        text = interpretation(
            plant, model, cand_df, readings, c0_phase, stopped, c1_bad, c5_note, c2_check,
        )
        protocol = _protocol(
            args.source_commit or proto.get("source_commit"),
            stopped, c0_phase, c2_check, proto.get("c5_selection"), c5_note,
            plant=plant, cache_only=args.cache_only, n_cache=n_cache,
        )
        write_all(plant, model, cand_df, readings, protocol, text)
        print("written read-stage", flush=True)
        return

    catalog = _fill_xi(candidate_catalog())
    pick_rule = (
        "C1、C2 最好档按全部阶段有符号组合的一致比例（一致数/有符号数）选取；"
        "并列时先取 R3 中位更接近 1，再取 |R5| 组成偏差更小。"
        "C5 = 该 C1 档的激冷后处理作用于该 C2 档的反应器快照；C1 全不可行则跳过 C5。"
    )
    print("compute plant slopes (no reactor)", flush=True)
    plant = plant_slopes_table(p114d)

    c0_specs = [
        {"window_id": wid, "group": "C0", "L_wsr": L_WSR_NOM,
         "HL_tot_pct": HL_NOM, "f_CO2": None}
        for wid in window_ids
    ]
    store = run_specs(c0_specs, args.n_workers, cache_only=args.cache_only)
    c0_phase, stopped = _count_c0(store, p114d)
    print(f"C0 done stopped={sorted(stopped)} {c0_phase}", flush=True)
    ok_wid = next((w for w in window_ids if store.get((w, "C0"), {}).get("ok_nom")), None)
    if ok_wid:
        rec_f1 = _job({
            "window_id": ok_wid, "group": "C2_f1", "L_wsr": L_WSR_NOM,
            "HL_tot_pct": HL_NOM, "f_CO2": 1.0, "perts": ("nom",),
        })
        store[(ok_wid, "C2_f1")] = rec_f1

    rest = [c for c in catalog if c["kind"] in ("C2", "C3", "C4") and not c.get("reuse_c0")]
    rest_specs = reactor_specs_for(rest, window_ids, stopped)
    store.update(run_specs(rest_specs, args.n_workers, cache_only=args.cache_only))

    summaries = []
    slope_frames = []
    c1_notes = []
    for cand in catalog:
        sl, sm, notes = evaluate_candidate(cand, store, p114d, plant, stopped)
        slope_frames.append(sl)
        summaries.append(sm)
        c1_notes.extend(notes)
        print(
            f"eval {cand['id']} feasible={sm['feasible']} "
            f"R2={sm['n_consistent']}/{sm['n_signed']} "
            f"{sm.get('infeasible_reason') or ''}",
            flush=True,
        )

    best_c1 = pick_best(summaries, "C1")
    best_c2 = pick_best(summaries, "C2")
    c5_note = ""
    c5_cand = None
    if best_c1 is None:
        c5_note = "C1 全部不可行，跳过 C5"
    elif best_c2 is None:
        c5_note = "C2 无可行档，跳过 C5"
    else:
        c2_full = next(c for c in catalog if c["id"] == best_c2["candidate"])
        c5_cand = {
            "id": f"C5_{best_c1['candidate']}+{best_c2['candidate']}",
            "kind": "C5",
            "label": f"{best_c1['candidate']} + {best_c2['candidate']}",
            "f_CO2": c2_full["f_CO2"],
            "L_wsr": L_WSR_NOM, "HL_tot_pct": HL_NOM,
            "T_q": next(c for c in catalog if c["id"] == best_c1["candidate"])["T_q"],
            "r_w": next(c for c in catalog if c["id"] == best_c1["candidate"])["r_w"],
            "xi": None,
            "reactor_group": _alias_group(c2_full),
        }
        sl, sm, notes = evaluate_candidate(c5_cand, store, p114d, plant, stopped)
        slope_frames.append(sl)
        summaries.append(sm)
        c1_notes.extend(notes)
        c5_note = (
            f"选用 {best_c1['candidate']}（R2={_fmt(best_c1['r2_ratio'], 3)}）+ "
            f"{best_c2['candidate']}（R2={_fmt(best_c2['r2_ratio'], 3)}）"
        )
        print(f"eval {c5_cand['id']} {c5_note}", flush=True)

    model = pd.concat(slope_frames, ignore_index=True) if slope_frames else pd.DataFrame()
    cand_df = pd.DataFrame(summaries)
    c1_bad = [s["candidate"] for s in summaries if s["kind"] == "C1" and not s["feasible"]]
    c2_check = _c2_checks(store, window_ids)
    readings = build_readings(plant, model, cand_df, c0_phase, stopped, c5_note)
    text = interpretation(
        plant, model, cand_df, readings, c0_phase, stopped, c1_bad, c5_note, c2_check,
    )
    protocol = _protocol(
        args.source_commit, stopped, c0_phase, c2_check, pick_rule, c5_note,
        plant=plant, cache_only=args.cache_only, n_cache=n_cache,
    )
    if c1_notes:
        pd.DataFrame(c1_notes).to_csv(OUT_DIR / "c1_windows.csv", index=False)
    write_all(plant, model, cand_df, readings, protocol, text)
    print("written:", sorted(p.name for p in OUT_DIR.iterdir() if p.is_file()), flush=True)
    print(f"stopped={sorted(stopped)} C1_infeasible={c1_bad}", flush=True)
    print("adopted_parameters=False", flush=True)


if __name__ == "__main__":
    main()
