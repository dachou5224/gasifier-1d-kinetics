#!/usr/bin/env python3
"""P2.0-A 碳转化率受限机理的并列比较（2026-11-13 卡）。

在 κ = 1（实测氧量）下并列比较两个使碳转化率小于 1 的最简机理：

  M1 后期反应性衰减：C+H2O、C+CO2 速率再乘 ϑ(X)，X0 ∈ {0, 0.80, 0.90}，每档只有 n 自由；
  M2 渣层截留：全混区出口截留份额 f_s 的固体（碳与灰同比例），带走其焓。

冻结：wsr_plus_pfr、L_wsr = 1 m、dz_down = 0.0515 m、total_percent HL_tot = 1%、
OxygenMixingModel = instant、实测载气、逐窗协调干煤量、全部 Wen 1979 动力学常数。
**本轮不反演、不采纳参数、不加回流、不改合同。**

用法：
    ./.venv/bin/python scripts/generate_p2_0a_20261113.py --stage unit
    ./.venv/bin/python scripts/generate_p2_0a_20261113.py --stage compare
    ./.venv/bin/python scripts/generate_p2_0a_20261113.py --stage perturb
    ./.venv/bin/python scripts/generate_p2_0a_20261113.py --stage hl
    ./.venv/bin/python scripts/generate_p2_0a_20261113.py --stage read
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import generate_p1_8a_20261104 as p18a  # noqa: E402

from model.gasifier_system import GasifierSystem  # noqa: E402
from model.input_contract import coal_hhv_kj_kg  # noqa: E402
from model.material import MaterialService  # noqa: E402
from model.model_case_adapter import build_gasifier_inputs_from_model_case  # noqa: E402
from model.p0_round_20260930 import ELEMENT_KG_PER_MOL, atoms_from_gas  # noqa: E402
from model.p1_5_round_20261030 import run_case, silence_solver_logs  # noqa: E402
from model.revision_evidence_common import load_frozen_split, source_commit  # noqa: E402

OUT_DIR = ROOT / "outputs" / "diagnostics" / "round-p2.0a-20261113"
CACHE_DIR = OUT_DIR / "cache"

P114D_CSV = ROOT / "outputs" / "diagnostics" / "p1.14d-20261107" / "windows_142.csv"
P114A_CSV = ROOT / "outputs" / "diagnostics" / "round-p1.14a-20261107" / "baseline_142.csv"
P112A_CSV = ROOT / "outputs" / "diagnostics" / "round-p1.12a-20261106" / "input_sensitivity.csv"
P115A_SLOPES = ROOT / "outputs" / "diagnostics" / "round-p1.15a-20261108" / "plant_slopes.csv"
P115A_CAND = ROOT / "outputs" / "diagnostics" / "round-p1.15a-20261108" / "candidates.csv"

# --- 冻结项 ---------------------------------------------------------------
DEV_L = 10.3
DZ_DOWN = 0.0515
L_WSR = 1.0
HL_NOM = 1.0
KAPPA = 1.0
F_GAS = 1.0
F_CO2_RATE = 1.0
HL_GRID = (0.5, 1.0, 2.0)

# --- 扰动 ---------------------------------------------------------------
FD_FRAC = 0.03
LN_P = math.log(1.0 + FD_FRAC)
LN_M = math.log(1.0 - FD_FRAC)
D_LN = LN_P - LN_M
PERTS = ("nom", "o2_p", "o2_m", "coal_p", "coal_m", "car_p", "car_m")
PERT_SCALE = {
    "nom": (1.0, 1.0, 1.0),
    "o2_p": (1.0 + FD_FRAC, 1.0, 1.0),
    "o2_m": (1.0 - FD_FRAC, 1.0, 1.0),
    "coal_p": (1.0, 1.0 + FD_FRAC, 1.0),
    "coal_m": (1.0, 1.0 - FD_FRAC, 1.0),
    "car_p": (1.0, 1.0, 1.0 + FD_FRAC),
    "car_m": (1.0, 1.0, 1.0 - FD_FRAC),
}
XC_CEIL_PP = 99.9
FAIL_WINDOW_STOP = 5

PHASES = ("phase1", "phase2", "phase3", "phase4")
PHASE_MAP = {"P1": "phase1", "P2": "phase2", "P3": "phase3", "P4": "phase4"}
OBS_ALL = (
    "X_C_pp", "CO_vol_pct", "H2_vol_pct", "CO2_vol_pct", "CH4_vol_pct",
    "dry_per_coal_mol_kg", "T_out_K", "T_wsr_K",
)
OBS_MAIN = ("CO_vol_pct", "CO2_vol_pct", "H2_vol_pct")
INPUTS = ("ln_O2", "ln_coal", "ln_carrier")
INPUT_PERTS = {"ln_O2": ("o2_p", "o2_m"), "ln_coal": ("coal_p", "coal_m"),
               "ln_carrier": ("car_p", "car_m")}

# 两个钩子候选；C0 为两钩子关闭的基准对照
CANDIDATES: list[dict] = [
    {"id": "C0", "kind": "C0", "label": "基准（两钩子关闭，κ=1）",
     "x0": None, "param_name": "none", "lo": 0.0, "hi": 0.0},
    {"id": "M1-0", "kind": "M1", "label": "后期衰减 X0=0", "x0": 0.0,
     "param_name": "n", "lo": 0.0, "hi": 8.0},
    {"id": "M1-80", "kind": "M1", "label": "后期衰减 X0=0.80", "x0": 0.80,
     "param_name": "n", "lo": 0.0, "hi": 8.0},
    {"id": "M1-90", "kind": "M1", "label": "后期衰减 X0=0.90", "x0": 0.90,
     "param_name": "n", "lo": 0.0, "hi": 8.0},
    {"id": "M2", "kind": "M2", "label": "渣层截留", "x0": None,
     "param_name": "f_s", "lo": 0.0, "hi": 0.5},
]
CAND_BY_ID = {c["id"]: c for c in CANDIDATES}
FIT_CANDS = tuple(c["id"] for c in CANDIDATES if c["kind"] != "C0")

# 第 4 节预声明：P1.15-A 的 8 个有符号组合（CO、CO2、H2 对氧量与煤量）
G2_COMBOS = (
    ("phase1", "CO2_vol_pct", "ln_coal"),
    ("phase2", "CO_vol_pct", "ln_coal"),
    ("phase2", "CO2_vol_pct", "ln_coal"),
    ("phase3", "CO_vol_pct", "ln_coal"),
    ("phase3", "CO2_vol_pct", "ln_coal"),
    ("phase4", "CO_vol_pct", "ln_coal"),
    ("phase4", "H2_vol_pct", "ln_O2"),
    ("phase4", "CO2_vol_pct", "ln_coal"),
)
G2_MIN = 6
G1_MIN_K = 50.0
G3_MIN_ECOAL = 0.3
G3_MIN_PHASES = 3

_CTX: dict = {}


# ---------------------------------------------------------------------------
# 输入与网格
# ---------------------------------------------------------------------------

def mesh_n() -> int:
    rl = DEV_L - L_WSR
    n_full = int(math.floor(rl / DZ_DOWN + 1e-12))
    rem = rl - float(n_full) * DZ_DOWN
    return 1 + n_full + (1 if rem > 1e-12 else 0)


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


def ref_window_ids() -> list[str]:
    """P1.12-A 的 9 个参照窗（L_wsr=1 且 perturb_tag=baseline）。"""
    p = pd.read_csv(P112A_CSV)
    ref = p[(p["L_wsr"] == 1.0) & (p["perturb_tag"] == "baseline")]
    return [str(w) for w in ref["window_id"]]


def phase_ref_windows() -> dict[str, list[str]]:
    p114d = _CTX["p114d"]
    out = {ph: [] for ph in PHASES}
    for wid in sorted(ref_window_ids()):
        out[_phase_of_row(p114d.loc[wid, "phase"])].append(wid)
    return out


def phase_windows() -> dict[str, list[str]]:
    p114d = _CTX["p114d"]
    out = {ph: [] for ph in PHASES}
    for wid in sorted(str(w) for w in p114d.index):
        out[_phase_of_row(p114d.loc[wid, "phase"])].append(wid)
    return out


def plant_xc_median(wids: list[str]) -> float:
    d = _CTX["p114d"]
    return float(np.median([float(d.loc[w, "xc_recon_phase_cal_measured_pct"]) for w in wids]))


def plant_obs_median(wids: list[str], col: str) -> float:
    d = _CTX["p114d"]
    return float(np.median([float(d.loc[w, col]) for w in wids]))


def plant_dry_per_coal(wid: str) -> float:
    row = _CTX["p114d"].loc[wid]
    dry = float(row["coal_dry_recon_phase_kg_s"])
    mol = float(row["syngas_dry_cal_nm3_h"]) / 22.414 / 3600.0 * 1000.0
    return mol / max(dry, 1e-12)


# ---------------------------------------------------------------------------
# 求解与快照
# ---------------------------------------------------------------------------

def hook_op(cand_id: str, param: float) -> dict:
    c = CAND_BY_ID[cand_id]
    if c["kind"] == "M1":
        return {"CharLateDecayX0": float(c["x0"]), "CharLateDecayExponent": float(param)}
    if c["kind"] == "M2":
        return {"SlagCaptureFraction": float(param)}
    return {}


def cache_key(wid: str, cand_id: str, param: float, hl: float,
              o2m: float, coalm: float, carm: float) -> str:
    """缓存的键必须含完整 window_id 与全部钩子取值（P1.15-A 的截断坑）。"""
    return json.dumps(
        {
            "window_id": str(wid),
            "candidate": str(cand_id),
            "param": repr(float(param)),
            "x0": None if CAND_BY_ID[cand_id]["x0"] is None
            else repr(float(CAND_BY_ID[cand_id]["x0"])),
            "hl": repr(float(hl)),
            "o2": repr(float(o2m)),
            "coal": repr(float(coalm)),
            "carrier": repr(float(carm)),
            "kappa": repr(float(KAPPA)),
            "f_gas": repr(float(F_GAS)),
            "f_co2_rate": repr(float(F_CO2_RATE)),
            "L_wsr": repr(float(L_WSR)),
            "dz_down": repr(float(DZ_DOWN)),
        },
        ensure_ascii=False, sort_keys=True,
    )


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{hashlib.sha1(key.encode('utf-8')).hexdigest()}.json"


def _load_cache(key: str) -> dict | None:
    p = _cache_path(key)
    if not p.exists():
        return None
    try:
        hit = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if hit.get("key") != key:
        return None
    return hit.get("snap")


def _store_cache(key: str, snap: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(key).write_text(
        json.dumps({"key": key, "snap": snap}, ensure_ascii=False), encoding="utf-8"
    )


def _closures(system: GasifierSystem) -> dict:
    """C/H/O/S 元素闭合与能量闭合；M2 计入被截留流股。"""
    out: dict = {}
    if not getattr(system, "results", None):
        return out
    cp = system.coal_props
    inlet_state = system.cells[0].inlet
    outlet = system.results[-1]
    coal = {
        e: float(system.W_dry) * (float(cp[k]) / 100.0) / ELEMENT_KG_PER_MOL[e]
        for e, k in zip(("C", "H", "O", "N", "S"), ("Cd", "Hd", "Od", "Nd", "Sd"))
    }
    moisture_mol_s = (float(system.tmp_W_liq_evap) / 18.015) * 1000.0
    coal["H"] += 2.0 * moisture_mol_s
    coal["O"] += moisture_mol_s
    inlet_gas = atoms_from_gas(inlet_state.gas_moles)
    outlet_gas = atoms_from_gas(outlet.gas_moles)
    cap = system.slag_capture
    c_cap_mol = 0.0 if not cap else float(cap["carbon_kg_s"]) / ELEMENT_KG_PER_MOL["C"]
    c_solid_mol = float(outlet.solid_mass) * float(outlet.carbon_fraction) / ELEMENT_KG_PER_MOL["C"]
    for e in ("C", "H", "O"):
        solid_out = (c_solid_mol + c_cap_mol) if e == "C" else 0.0
        total_in = coal[e] + inlet_gas[e]
        err = (outlet_gas[e] + solid_out) - total_in
        out[f"{e}_closure_mol_s"] = float(err)
        out[f"{e}_closure_fraction"] = float(err / total_in) if abs(total_in) > 1e-12 else math.nan
    # 能量闭合
    H_in = MaterialService.get_total_enthalpy(
        inlet_state, cp, include_formation_enthalpy=True
    )
    e_src = 0.0
    for cell in system.cells:
        for src in cell.sources:
            _g, _sm, e = src.get_sources(cell.idx, cell.z, cell.dz)
            e_src += float(e)
    q_loss = float(sum(float(d.get("hloss_W", 0.0) or 0.0) for d in system.cell_diagnostics))
    H_out = MaterialService.get_total_enthalpy(
        outlet, cp, T_solid_override=outlet.T_solid_or_gas,
        include_formation_enthalpy=False,
    )
    H_cap = 0.0 if not cap else float(cap["enthalpy_W"])
    err_e = (H_out + H_cap + q_loss) - (H_in + e_src)
    out["energy_closure_W"] = float(err_e)
    out["energy_closure_fraction"] = float(err_e / abs(H_in)) if abs(H_in) > 1e-12 else math.nan
    out["H_in_W"] = float(H_in)
    out["H_captured_W"] = float(H_cap)
    out["Q_loss_W"] = float(q_loss)
    return out


def _build_op(wid: str, cand_id: str, param: float, hl: float,
              o2m: float, coalm: float, carm: float) -> tuple[dict, float]:
    dry = float(_CTX["p114d"].loc[wid, "coal_dry_recon_phase_kg_s"]) * float(coalm)
    carrier_nm3 = float(_CTX["p114d"].loc[wid, "co2_carrier_measured_nm3_h"]) * float(carm)
    op = {
        "HeatLossModel": "total_percent",
        "HeatLossPercentTotal": float(hl),
        "GasificationRateFactor": float(F_GAS),
        "CharCO2RateFactor": float(F_CO2_RATE),
        "carrier_gas_flow": p18a._carrier_kg_s(carrier_nm3),
        "MeshFamily": "wsr_plus_pfr",
        "WSRLength_m": float(L_WSR),
        "PFRCellLength_m": float(DZ_DOWN),
        "OxygenMixingModel": "instant",
        "o2_flow_scale": float(KAPPA) * float(o2m),
    }
    op.update(hook_op(cand_id, param))
    return op, dry


def cell_carbon_residual_frac(wid: str, cand_id: str, param: float,
                              hl: float = HL_NOM) -> dict:
    """Σ逐格碳残差 / 入炉煤碳：元素闭合误差的来源（求解器停点，不是缺项）。"""
    _ensure_ctx()
    op, dry = _build_op(wid, cand_id, param, hl, 1.0, 1.0, 1.0)
    rec = run_case(_CTX["by_id"][wid], n_cells=mesh_n(), coal_flow_dry_kg_s=dry,
                   keep_system=True, mesh_family="wsr_plus_pfr", extra_op=op)
    system = rec.get("_system")
    if system is None or not system.results:
        return {}
    total = 0.0
    costs = []
    for cell, st in zip(system.cells, system.results):
        r = np.asarray(cell.residuals(st.to_array()), dtype=float)
        un = r * cell.residual_scale_vector()
        total += float(un[1] + un[2] + un[3]) * 0.012011 + float(un[9])
        costs.append(float(0.5 * np.sum(r * r)))
    c_in = float(system.W_dry) * float(system.coal_props.get("Cd", 0.0)) / 100.0
    return {
        "sum_cell_carbon_kg_s": float(total),
        "sum_cell_carbon_frac": float(total / max(c_in, 1e-12)),
        "max_cell_cost": float(max(costs)) if costs else float("nan"),
        "n_cells_cost_gt_1e4": int(sum(c > 1e-4 for c in costs)),
    }


def _snapshot(wid: str, cand_id: str, param: float, hl: float,
              o2m: float, coalm: float, carm: float, want_profile: bool) -> dict:
    row = _CTX["by_id"][wid]
    op, dry = _build_op(wid, cand_id, param, hl, o2m, coalm, carm)
    carrier_nm3 = float(_CTX["p114d"].loc[wid, "co2_carrier_measured_nm3_h"]) * float(carm)
    rec = run_case(
        row, n_cells=mesh_n(), coal_flow_dry_kg_s=dry,
        keep_system=True, mesh_family="wsr_plus_pfr", extra_op=op,
    )
    system = rec.get("_system")
    snap: dict = {
        "window_id": wid, "candidate": cand_id, "param": float(param),
        "hl": float(hl), "o2_mult": float(o2m), "coal_mult": float(coalm),
        "carrier_mult": float(carm),
        "phase": _phase_of_row(_CTX["p114d"].loc[wid, "phase"]),
        "execution_status": str(rec.get("execution_status")),
        "no_root_cell_index": rec.get("no_root_cell_index"),
        "max_residual_cost": _f(rec.get("max_residual_cost")),
        "solve_wall_time_s": _f(rec.get("solve_wall_time_s")),
        "n_cells": int(rec.get("N_cells") or 0),
        "coal_dry_kg_s": float(dry),
        "carrier_nm3_h": float(carrier_nm3),
        "ok": False,
    }
    if system is not None:
        snap["hloss_status"] = str(getattr(system, "hloss_status", "n/a"))
        snap["hloss_iters"] = int(getattr(system, "heat_loss_iterations", -1))
    if system is None or not system.results or rec.get("execution_status") != "completed":
        return snap
    cd = float(system.coal_props.get("Cd", 0.0)) / 100.0
    c_in = float(system.W_dry) * cd
    outlet = system.results[-1]
    gas = np.asarray(outlet.gas_moles, dtype=float)
    cap = system.slag_capture
    c_out = float(outlet.solid_mass) * float(outlet.carbon_fraction)
    c_cap = 0.0 if not cap else float(cap["carbon_kg_s"])
    c_wsr = float(system.results[0].solid_mass) * float(system.results[0].carbon_fraction)
    hhv_kj = float(coal_hhv_kj_kg(system.coal_props))
    snap.update({
        "ok": True,
        "X_C_frac": float(1.0 - (c_out + c_cap) / max(c_in, 1e-12)),
        "X_C_pp": float((1.0 - (c_out + c_cap) / max(c_in, 1e-12)) * 100.0),
        "X_C_run_case_pp": float(rec["X_C_implied"]) * 100.0,
        "T_out_K": float(outlet.T),
        "T_wsr_K": float(system.results[0].T),
        "Ts_wsr_K": float(system.results[0].T_solid_or_gas),
        "X_C_wsr_pp": float((1.0 - c_wsr / max(c_in, 1e-12)) * 100.0),
        "F_CO": float(gas[2]), "F_CO2": float(gas[3]), "F_H2": float(gas[5]),
        "F_H2O": float(gas[7]), "F_CH4": float(gas[1]), "F_H2S": float(gas[4]),
        "dry_total_mol_s": float(np.sum(gas[:7])),
        "F_O2_out": float(gas[0]),
        "capture_fraction": 0.0 if not cap else float(cap["fraction"]),
        "capture_carbon_frac_of_inlet": float(c_cap / max(c_in, 1e-12)),
        "capture_enthalpy_frac_hhv_pct": (
            0.0 if not cap
            else float(cap["enthalpy_W"] / max(dry * hhv_kj / 1000.0 * 1e6, 1e-12) * 100.0)
        ),
        "capture_solid_mass_kg_s": 0.0 if not cap else float(cap["solid_mass_kg_s"]),
        "capture_T_solid_K": float("nan") if not cap else float(cap["T_solid_K"]),
        "wsr_C_CO2_bind": bool(
            (system.cell_diagnostics[0].get("C_CO2_LimitedByCO2") if system.cell_diagnostics else False)
        ),
    })
    snap.update(_closures(system))
    if want_profile:
        prof = []
        for i, (cell, st) in enumerate(zip(system.cells, system.results)):
            c_local = float(st.solid_mass) * float(st.carbon_fraction)
            xc = 1.0 - c_local / max(c_in, 1e-12)
            prof.append({
                "i": int(i), "z_m": float(st.z), "dz_m": float(cell.dz),
                "X_C_frac": float(xc), "T_gas_K": float(st.T),
                "T_solid_K": float(st.T_solid_or_gas),
                "solid_mass_kg_s": float(st.solid_mass),
                "carbon_fraction": float(st.carbon_fraction),
                "is_combustion": bool(
                    system.cell_diagnostics[i].get("is_combustion_zone", False)
                ) if i < len(system.cell_diagnostics) else False,
            })
        snap["profile"] = prof
    rec.pop("_system", None)
    return snap


def _init_worker() -> None:
    silence_solver_logs()
    ok, rows_, _inv = load_frozen_split()
    if not ok:
        raise RuntimeError("frozen split was not consumed intact")
    by_id = {r["window_id"]: r for r in rows_}
    p114d = pd.read_csv(P114D_CSV).set_index("window_id")
    csv_ids = set(p114d.index.astype(str))
    if csv_ids != set(str(k) for k in by_id):
        raise RuntimeError("windows_142 ids do not match frozen split")
    _CTX.update({"by_id": by_id, "p114d": p114d})


def _ensure_ctx() -> None:
    if "p114d" not in _CTX:
        _init_worker()


def solve_snap(wid: str, cand_id: str, param: float, hl: float = HL_NOM,
               pert: str = "nom", want_profile: bool = False) -> dict:
    _ensure_ctx()
    o2m, coalm, carm = PERT_SCALE[pert]
    key = cache_key(wid, cand_id, param, hl, o2m, coalm, carm)
    hit = _load_cache(key)
    if hit is not None and (hit.get("profile") or not want_profile):
        return hit
    snap = _snapshot(wid, cand_id, param, hl, o2m, coalm, carm, want_profile)
    snap["pert"] = pert
    _store_cache(key, snap)
    return snap


def _job(spec: dict) -> dict:
    try:
        s = solve_snap(
            str(spec["window_id"]), str(spec["candidate"]), float(spec["param"]),
            float(spec.get("hl", HL_NOM)), str(spec.get("pert", "nom")),
            bool(spec.get("want_profile", False)),
        )
        s["spec_tag"] = spec.get("tag", "")
        return s
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return {
            "window_id": spec.get("window_id"), "candidate": spec.get("candidate"),
            "param": spec.get("param"), "hl": spec.get("hl", HL_NOM),
            "pert": spec.get("pert", "nom"), "ok": False, "spec_tag": spec.get("tag", ""),
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_specs(specs: list[dict], n_workers: int) -> list[dict]:
    if not specs:
        return []
    n_workers = max(1, min(int(n_workers), len(specs)))
    if n_workers == 1:
        return [_job(s) for s in specs]
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as ex:
        return list(ex.map(_job, specs, chunksize=1))


# ---------------------------------------------------------------------------
# ξ（激冷段变换）与观测量
# ---------------------------------------------------------------------------

def apply_xi(snap: dict, xi: float) -> dict | None:
    if not snap.get("ok"):
        return None
    xi = float(xi)
    f_co = float(snap["F_CO"]) - xi
    f_co2 = float(snap["F_CO2"]) + xi
    f_h2 = float(snap["F_H2"]) + xi
    f_h2o = float(snap["F_H2O"]) - xi
    dry = float(snap["dry_total_mol_s"]) + xi
    if not (math.isfinite(dry) and dry > 0.0):
        return None
    coal = float(snap["coal_dry_kg_s"])
    return {
        "X_C_pp": float(snap["X_C_pp"]),
        "CO_vol_pct": f_co / dry * 100.0,
        "H2_vol_pct": f_h2 / dry * 100.0,
        "CO2_vol_pct": f_co2 / dry * 100.0,
        "CH4_vol_pct": float(snap["F_CH4"]) / dry * 100.0,
        "dry_per_coal_mol_kg": dry / max(coal, 1e-12),
        "T_out_K": float(snap["T_out_K"]),
        "T_wsr_K": float(snap["T_wsr_K"]),
        "F_H2O_out": f_h2o,
        "dry_total_mol_s": dry,
        "xi_mol_s": xi,
        "CO_positive": bool(f_co > 0.0),
    }


def xi_median(store: dict[tuple[str, str, float, float], dict], wids: list[str],
              cand_id: str, param: float, hl: float, xi: float) -> float:
    vals = []
    for w in wids:
        y = apply_xi(store[(w, cand_id, param, hl)], xi)
        if y is None:
            return float("nan")
        vals.append(y["CO2_vol_pct"])
    return float(np.median(vals)) if vals else float("nan")


def solve_xi(store, wids, cand_id, param, hl, target) -> float:
    """单参数闭合：每阶段一个常数 ξ，使参照窗 CO2 中位 = 装置值。"""
    lo_snaps = [store[(w, cand_id, param, hl)] for w in wids]
    ok = [s for s in lo_snaps if s.get("ok")]
    if not ok:
        return float("nan")
    lo = -0.95 * min(float(s["F_CO2"]) for s in ok)
    hi = 0.95 * min(float(s["F_CO"]) for s in ok)
    f = lambda x: xi_median(store, wids, cand_id, param, hl, x) - target  # noqa: E731
    flo, fhi = f(lo), f(hi)
    if not (math.isfinite(flo) and math.isfinite(fhi)):
        return float("nan")
    if flo > 0.0 or fhi < 0.0:
        return float("nan")
    a, b = lo, hi
    for _ in range(200):
        mid = 0.5 * (a + b)
        fm = f(mid)
        if not math.isfinite(fm):
            return float("nan")
        if abs(fm) < 1e-10 or (b - a) < 1e-10 * max(abs(b), 1.0):
            return float(mid)
        if fm < 0.0:
            a = mid
        else:
            b = mid
    return float(0.5 * (a + b))


# ---------------------------------------------------------------------------
# 有限差分与阶段统计
# ---------------------------------------------------------------------------

def fd_stats(snaps: dict[str, dict], xi: float) -> dict:
    ys = {p: apply_xi(snaps.get(p) or {}, xi) for p in PERTS}
    out: dict = {"ok_nom": ys["nom"] is not None}
    if not out["ok_nom"]:
        return out
    for obs in OBS_ALL:
        y0 = ys["nom"].get(obs)
        for inp, (pp, pm) in INPUT_PERTS.items():
            yp = ys[pp].get(obs) if ys[pp] else None
            ym = ys[pm].get(obs) if ys[pm] else None
            if yp is None or ym is None or not all(
                math.isfinite(float(v)) for v in (y0, yp, ym)
            ):
                out[f"d2_{obs}__{inp}"] = float("nan")
                out[f"dplus_{obs}__{inp}"] = float("nan")
                out[f"dminus_{obs}__{inp}"] = float("nan")
                continue
            out[f"d2_{obs}__{inp}"] = (float(yp) - float(ym)) / D_LN * LN_P
            out[f"dplus_{obs}__{inp}"] = float(yp) - float(y0)
            out[f"dminus_{obs}__{inp}"] = float(y0) - float(ym)
    return out


def elasticities(xc_pp: float, per3_xc_pp: float) -> tuple[float, float]:
    if not (math.isfinite(xc_pp) and math.isfinite(per3_xc_pp)) or xc_pp <= 1e-9:
        return float("nan"), float("nan")
    r = (per3_xc_pp / xc_pp) / LN_P
    return 1.0 + r, r


# ---------------------------------------------------------------------------
# 第 2 节：比较点
# ---------------------------------------------------------------------------

def compare_point(cand_id: str, wids: list[str], n_workers: int,
                  hl: float = HL_NOM) -> dict:
    """一维求根：参照窗模型 X_C 中位 = 同批窗装置 X_C 中位（容差 0.2 pp）。"""
    c = CAND_BY_ID[cand_id]
    target = plant_xc_median(wids)
    rec: dict = {
        "candidate": cand_id, "kind": c["kind"], "phase": rec_phase(wids),
        "param_name": c["param_name"], "target_X_C_pp": target,
        "n_ref_windows": len(wids), "hl_tot_pct": float(hl),
        "reachable": False, "reason": "", "param": float("nan"), "at_bound": "",
        "X_C_pp_median": float("nan"), "dX_C_pp": float("nan"),
        "endpoint_X_C_pp": float("nan"), "n_solve_fail": 0,
    }
    if c["kind"] == "C0":
        store = {(w, cand_id, 0.0, hl): solve_snap(w, cand_id, 0.0, hl) for w in wids}
        xs = [s["X_C_pp"] for s in store.values() if s.get("ok")]
        rec.update({
            "param": 0.0, "reachable": True,
            "X_C_pp_median": float(np.median(xs)) if xs else float("nan"),
            "n_solve_fail": len(wids) - len(xs),
        })
        rec["dX_C_pp"] = rec["X_C_pp_median"] - target
        return rec, store

    def xc_median(p: float, store: dict) -> float:
        for w in wids:
            if (w, cand_id, p, hl) not in store:
                store[(w, cand_id, p, hl)] = solve_snap(w, cand_id, p, hl)
        xs = [store[(w, cand_id, p, hl)]["X_C_pp"] for w in wids
              if store[(w, cand_id, p, hl)].get("ok")]
        return float(np.median(xs)) if xs else float("nan")

    store: dict = {}
    lo, hi = float(c["lo"]), float(c["hi"])
    x_lo = xc_median(lo, store)
    x_hi = xc_median(hi, store)
    rec["X_C_at_lo"] = x_lo
    rec["X_C_at_hi"] = x_hi
    n_fail = sum(1 for w in wids if not store[(w, cand_id, lo, hl)].get("ok")) + sum(
        1 for w in wids if not store[(w, cand_id, hi, hl)].get("ok")
    )
    rec["n_solve_fail"] = n_fail
    if not (math.isfinite(x_lo) and math.isfinite(x_hi)):
        rec["reason"] = "端点求解失败"
        rec["endpoint_X_C_pp"] = x_lo if math.isfinite(x_lo) else x_hi
        return rec, store
    # 目标须落在 [x_hi, x_lo]（两个机理都使 X_C 随参数单调下降）；
    # 端点若已在 0.2 pp 容差内，按卡面「容差 0.2 个百分点」接受该端点。
    if target > x_lo:
        if abs(x_lo - target) <= 0.2:
            rec.update({"param": lo, "X_C_pp_median": x_lo, "dX_C_pp": x_lo - target,
                        "reachable": True, "at_bound": "lo",
                        "reason": "在上界（最完全转化端）于 0.2 pp 容差内达标"})
            return rec, store
        rec["reason"] = "端点仍高于装置值（参数下界已是最完全转化）"
        rec["endpoint_X_C_pp"] = x_lo
        rec["X_C_pp_median"] = x_lo
        return rec, store
    if target < x_hi:
        if abs(x_hi - target) <= 0.2:
            rec.update({"param": hi, "X_C_pp_median": x_hi, "dX_C_pp": x_hi - target,
                        "reachable": True, "at_bound": "hi",
                        "reason": "在上界 n=8 / f_s=0.5 处于 0.2 pp 容差内达标"})
            return rec, store
        rec["reason"] = "端点仍低于装置值（参数上界不足以回到装置值）"
        rec["endpoint_X_C_pp"] = x_hi
        rec["X_C_pp_median"] = x_hi
        rec["dX_C_pp"] = x_hi - target
        return rec, store
    a, b = lo, hi
    xa, xb = x_lo, x_hi
    best_p, best_x = None, float("nan")
    for _ in range(60):
        mid = 0.5 * (a + b)
        xm = xc_median(mid, store)
        if not math.isfinite(xm):
            rec["reason"] = "求根中点求解失败"
            break
        if abs(xm - target) < 0.02:
            best_p, best_x = mid, xm
            break
        best_p, best_x = mid, xm
        if xm > target:
            a, xa = mid, xm
        else:
            b, xb = mid, xm
        if (b - a) < 1e-6 * max(abs(hi), 1.0):
            best_p, best_x = 0.5 * (a + b), xc_median(0.5 * (a + b), store)
            break
    if best_p is None:
        rec["reason"] = rec["reason"] or "求根未收敛"
        return rec, store
    rec.update({
        "param": float(best_p), "X_C_pp_median": float(best_x),
        "dX_C_pp": float(best_x - target),
        "reachable": bool(abs(best_x - target) <= 0.2),
        "reason": "" if abs(best_x - target) <= 0.2 else "容差 0.2 pp 内未达",
    })
    return rec, store


def rec_phase(wids: list[str]) -> str:
    return _phase_of_row(_CTX["p114d"].loc[wids[0], "phase"]) if wids else ""


def _compare_job(spec: dict) -> dict:
    _ensure_ctx()
    rec, _store = compare_point(
        str(spec["candidate"]), list(spec["wids"]), 1, float(spec.get("hl", HL_NOM))
    )
    return rec


# ---------------------------------------------------------------------------
# 阶段入口
# ---------------------------------------------------------------------------

def stage_unit(n_workers: int) -> None:
    _ensure_ctx()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    refw = sorted(ref_window_ids())
    p112a = pd.read_csv(P112A_CSV)
    ref_rows = p112a[(p112a["L_wsr"] == 1.0) & (p112a["perturb_tag"] == "baseline")]
    ref_rows = ref_rows.set_index("window_id")
    rows: list[dict] = []

    # --- U1：缺省钩子与 P1.12-A 一致（≤1e-9 相对偏差）
    base = {}
    for w in refw:
        base[w] = solve_snap(w, "C0", 0.0, HL_NOM)
    max_rel = 0.0
    diffs = []
    for w in refw:
        s = base[w]
        r = ref_rows.loc[w]
        vals = {
            "X_C": (_f(s.get("X_C_run_case_pp")) / 100.0, _f(r["X_C_implied"])),
            "T_out_K": (_f(s.get("T_out_K")), _f(r["T_out_K"])),
        }
        dry = _f(s.get("dry_total_mol_s"))
        for tag, fld, ref_col in (("CO", "F_CO", "CO_vol_pct"),
                                  ("CO2", "F_CO2", "CO2_vol_pct"),
                                  ("H2", "F_H2", "H2_vol_pct")):
            vals[tag] = (_f(s.get(fld)) / dry * 100.0, _f(r[ref_col]))
        for tag, (a, b) in vals.items():
            rel = abs(a - b) / max(abs(b), 1e-12)
            max_rel = max(max_rel, rel)
            if rel > 1e-9:
                diffs.append({"window_id": w, "metric": tag, "model": a, "p1_12a": b,
                              "rel": rel})
    rows.append({
        "check": "U1", "item": "缺省钩子 9 窗 vs P1.12-A(L=1,baseline)",
        "n": len(refw), "value": max_rel, "threshold": 1e-9,
        "pass": bool(max_rel <= 1e-9), "detail": json.dumps(diffs, ensure_ascii=False),
    })

    # --- U2：n=0（任一 X0）与 f_s=0 与 U1 逐位一致
    fields = ("X_C_pp", "T_out_K", "F_CO", "F_CO2", "F_H2", "F_H2O", "F_CH4",
              "dry_total_mol_s", "T_wsr_K", "X_C_wsr_pp")
    variants = (("M1-0", 0.0), ("M1-80", 0.0), ("M1-90", 0.0), ("M2", 0.0))
    for cid, p in variants:
        n_bit = 0
        max_rel2 = 0.0
        for w in refw:
            s1 = base[w]
            s2 = solve_snap(w, cid, p, HL_NOM)
            same = all(
                repr(s1.get(f)) == repr(s2.get(f)) for f in fields
            )
            if same:
                n_bit += 1
            else:
                for f in fields:
                    a, b = _f(s1.get(f)), _f(s2.get(f))
                    max_rel2 = max(max_rel2, abs(a - b) / max(abs(b), 1e-12))
        rows.append({
            "check": "U2", "item": f"{cid} param={p:g} vs U1 逐位一致",
            "n": len(refw), "value": n_bit, "threshold": len(refw),
            "pass": bool(n_bit == len(refw)),
            "detail": f"bitwise_equal_windows={n_bit}/{len(refw)} max_rel={max_rel2:.3e}",
        })

    # --- U3：元素与能量闭合（开启钩子）
    #   卡面阈值 1e-8（相对）。实测 C0 自身在这 9 窗上就有 1e-11 ~ 1e-5 的差异
    #   （取决于 least_squares 的停点），故同时报「相对同窗 C0 的增量」与求解精度底 1e-4。
    hook_sets = (("M1-0", 2.0), ("M1-80", 3.0), ("M1-90", 4.0), ("M2", 0.30))
    for cid, p in hook_sets:
        acc = {el: {"abs_max": 0.0, "inc_max": 0.0, "c0_max": 0.0} for el in ("C", "H", "O")}
        e_abs = 0.0
        e_inc = 0.0
        for w in refw:
            s = solve_snap(w, cid, p, HL_NOM)
            b = base[w]
            for el in ("C", "H", "O"):
                v = _f(s.get(f"{el}_closure_fraction"))
                vb = _f(b.get(f"{el}_closure_fraction"))
                acc[el]["abs_max"] = max(acc[el]["abs_max"], abs(v))
                acc[el]["inc_max"] = max(acc[el]["inc_max"], abs(v - vb))
                acc[el]["c0_max"] = max(acc[el]["c0_max"], abs(vb))
                rows.append({
                    "check": "U3", "item": f"{cid}@{p:g} {el} 元素闭合（相对）",
                    "n": 1, "value": v, "threshold": 1e-8,
                    "pass": bool(abs(v) <= 1e-8),
                    "pass_solver_floor": bool(abs(v) <= 1e-4),
                    "detail": (f"window={w} C0={vb:.6e} "
                               f"increment={v - vb:.3e} "
                               f"|incr|<=1e-8:{bool(abs(v - vb) <= 1e-8)}"),
                })
            rows.append({
                "check": "U3", "item": f"{cid}@{p:g} 能量闭合（相对，含截留焓）",
                "n": 1, "value": _f(s.get("energy_closure_fraction")), "threshold": 1e-8,
                "pass": bool(abs(_f(s.get("energy_closure_fraction"))) <= 1e-8),
                "pass_solver_floor": bool(
                    abs(_f(s.get("energy_closure_fraction"))) <= 1e-6),
                "detail": (
                    f"window={w} err_W={_f(s.get('energy_closure_W')):.4e} "
                    f"H_cap_W={_f(s.get('H_captured_W')):.4e} "
                    f"C0_rel={_f(b.get('energy_closure_fraction')):.3e}"
                ),
            })
            e_abs = max(e_abs, abs(_f(s.get("energy_closure_fraction"))))
            e_inc = max(e_inc, abs(_f(s.get("energy_closure_fraction"))
                                   - _f(b.get("energy_closure_fraction"))))
        for el in ("C", "H", "O"):
            rows.append({
                "check": "U3", "item": f"{cid}@{p:g} {el} 闭合汇总（9 参照窗）",
                "n": len(refw), "value": acc[el]["abs_max"], "threshold": 1e-8,
                "pass": bool(acc[el]["abs_max"] <= 1e-8),
                "pass_solver_floor": bool(acc[el]["inc_max"] <= 1e-4
                                          and acc[el]["abs_max"] <= 1e-4),
                "detail": (f"max|closure|={acc[el]['abs_max']:.3e} "
                           f"max|increment_vs_C0|={acc[el]['inc_max']:.3e} "
                           f"C0 自身 max|closure|={acc[el]['c0_max']:.3e}"),
            })
        rows.append({
            "check": "U3", "item": f"{cid}@{p:g} 能量闭合汇总", "n": len(refw),
            "value": e_abs, "threshold": 1e-8, "pass": bool(e_abs <= 1e-8),
            "pass_solver_floor": bool(e_inc <= 1e-6),
            "detail": f"max|rel|={e_abs:.3e} max|increment_vs_C0|={e_inc:.3e}",
        })
        # 元素闭合误差的来源：Σ逐格碳残差（求解器停点），前 3 个参照窗
        for w in refw[:3]:
            diag = cell_carbon_residual_frac(w, cid, p, HL_NOM)
            s = solve_snap(w, cid, p, HL_NOM)
            rows.append({
                "check": "U3", "item": f"{cid}@{p:g} 碳闭合误差来源（Σ逐格碳残差）",
                "n": 182, "value": _f(diag.get("sum_cell_carbon_frac")), "threshold": "",
                "pass": "",
                "detail": (f"window={w} 闭合={_f(s.get('C_closure_fraction')):.3e} "
                           f"Σ逐格碳残差/入炉碳={_f(diag.get('sum_cell_carbon_frac')):.3e} "
                           f"max_cell_cost={_f(diag.get('max_cell_cost')):.3e} "
                           f"n_cell_gt_1e-4={diag.get('n_cells_cost_gt_1e4')}"),
            })

    # --- U4：ϑ 在 X0 处连续、0<ϑ≤1；n 与 f_s 递增使 X_C 单调下降
    for x0 in (0.0, 0.80, 0.90):
        for n in (0.0, 0.5, 1.0, 2.0, 3.0, 8.0):
            ok_range = True
            for k in range(0, 1000):
                X = k / 1000.0
                v = 1.0 if X <= x0 else ((1.0 - X) / (1.0 - x0)) ** n
                if not (0.0 < v <= 1.0 + 1e-15):
                    ok_range = False
            jump = abs(((1.0 - min(x0 + 1e-12, 1.0)) / (1.0 - x0)) ** n - 1.0)
            rows.append({
                "check": "U4", "item": f"ϑ 连续性/值域 X0={x0:g} n={n:g}",
                "n": 1000, "value": jump, "threshold": 1e-9,
                "pass": bool(ok_range and jump <= 1e-9),
                "detail": (f"值域在 X∈[0,0.999] 上检查（0<ϑ≤1）；"
                           f"|ϑ(X0+1e-12)-ϑ(X0)|={jump:.3e}，真值为 0，跳变随 ε 线性趋零；"
                           f"ϑ(X=1)=0 是极限（该处已无碳）；range_ok={ok_range}"),
            })
    probe_w = phase_ref_windows()["phase2"][0]
    ns = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
    xc_n = [solve_snap(probe_w, "M1-80", n, HL_NOM)["X_C_pp"] for n in ns]
    mono_n = all(b <= a + 1e-12 for a, b in zip(xc_n, xc_n[1:]))
    rows.append({
        "check": "U4", "item": "n 递增 → X_C 单调下降（M1-80，阶段 2 参照窗）",
        "n": len(ns), "value": float(np.max(np.diff(xc_n))) if len(xc_n) > 1 else 0.0,
        "threshold": 0.0, "pass": bool(mono_n),
        "detail": f"window={probe_w} X_C(n)={[round(v, 4) for v in xc_n]}",
    })
    fss = (0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
    xc_f = [solve_snap(probe_w, "M2", f, HL_NOM)["X_C_pp"] for f in fss]
    mono_f = all(b <= a + 1e-12 for a, b in zip(xc_f, xc_f[1:]))
    rows.append({
        "check": "U4", "item": "f_s 递增 → X_C 单调下降（M2，阶段 2 参照窗）",
        "n": len(fss), "value": float(np.max(np.diff(xc_f))) if len(xc_f) > 1 else 0.0,
        "threshold": 0.0, "pass": bool(mono_f),
        "detail": f"window={probe_w} X_C(f_s)={[round(v, 4) for v in xc_f]}",
    })

    # --- U5：JAX 批量路径对非缺省钩子报 NotImplementedError
    wid = refw[0]
    built = build_gasifier_inputs_from_model_case(
        _CTX["by_id"][wid],
        coal_flow_dry_kg_s=float(_CTX["p114d"].loc[wid, "coal_dry_recon_phase_kg_s"]),
    )
    for cid, p in (("M1-80", 3.0), ("M2", 0.3)):
        op = dict(built["op_conds"])
        op.update({
            "MeshFamily": "wsr_plus_pfr", "ParticleSizeModel": "fortran_constant",
            "SolverStrictRoot": True, "AllowTruncatedMesh": False,
            "WSRLength_m": L_WSR, "PFRCellLength_m": DZ_DOWN,
            "OxygenMixingModel": "instant", "HeatLossModel": "hhv_percent",
        })
        op.update(hook_op(cid, p))
        exc_name, msg = "", ""
        try:
            system = GasifierSystem(built["geometry"], built["coal_props"], op)
            system.solve(N_cells=mesh_n(), solver_method="jax_jit", jax_warmup=False)
        except Exception as exc:  # noqa: BLE001
            exc_name = type(exc).__name__
            msg = str(exc)[:180]
        rows.append({
            "check": "U5", "item": f"JAX 批量路径 {cid}@{p:g} → NotImplementedError",
            "n": 1, "value": exc_name, "threshold": "NotImplementedError",
            "pass": bool(exc_name == "NotImplementedError"), "detail": msg,
        })

    pd.DataFrame(rows).to_csv(OUT_DIR / "unit_checks.csv", index=False)
    n_fail = sum(1 for r in rows if r["check"] in ("U1", "U2", "U5") and not r["pass"])
    print(f"unit checks written: {len(rows)} rows, hard-fail(U1/U2/U5)={n_fail}", flush=True)


def stage_compare(n_workers: int) -> None:
    _ensure_ctx()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pref = phase_ref_windows()
    specs = [
        {"candidate": cid, "wids": pref[ph], "hl": HL_NOM, "tag": f"{cid}|{ph}"}
        for cid in CAND_BY_ID for ph in PHASES
    ]
    specs.sort(key=lambda s: (s["candidate"], PHASES.index(rec_phase(s["wids"]))))
    n_use = max(1, min(n_workers, len(specs)))
    print(f"compare: {len(specs)} root-finds on {n_use} workers", flush=True)
    if n_use == 1:
        recs = [_compare_job(s) for s in specs]
    else:
        with ProcessPoolExecutor(max_workers=n_use, initializer=_init_worker) as ex:
            recs = list(ex.map(_compare_job, specs, chunksize=1))
    df = pd.DataFrame(recs)
    df.to_csv(OUT_DIR / "comparison_points.csv", index=False)
    print(df[["phase", "candidate", "reachable", "param", "X_C_pp_median",
              "target_X_C_pp", "n_solve_fail", "reason"]].to_string(), flush=True)


def stage_perturb(n_workers: int) -> None:
    """阶段 3：比较点上 142 窗基准 + 6 扰动。"""
    _ensure_ctx()
    cp = pd.read_csv(OUT_DIR / "comparison_points.csv")
    pw = phase_windows()
    specs = []
    for _, r in cp.iterrows():
        cid = str(r["candidate"])
        if CAND_BY_ID[cid]["kind"] == "C0":
            param, hl, usable = 0.0, HL_NOM, True
        else:
            usable = bool(r["reachable"])
            param, hl = float(r["param"]), float(r["hl_tot_pct"])
        if not usable:
            print(f"skip {cid} {r['phase']}: unreachable", flush=True)
            continue
        for w in pw[str(r["phase"])]:
            for pert in PERTS:
                specs.append({"window_id": w, "candidate": cid, "param": param,
                              "hl": hl, "pert": pert,
                              "tag": f"{cid}|{r['phase']}"})
    print(f"perturb: {len(specs)} solves on {n_workers} workers", flush=True)
    out = run_specs(specs, n_workers)
    payload = []
    for s in out:
        s.pop("profile", None)
        payload.append(s)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "perturb_snaps.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    n_ok = sum(1 for s in payload if s.get("ok"))
    print(f"perturb done: {n_ok}/{len(payload)} ok", flush=True)


def stage_hl(n_workers: int) -> None:
    """G5：HL_tot = 0.5 与 2 时只在参照窗上重做第 2、3 节。"""
    _ensure_ctx()
    pref = phase_ref_windows()
    specs = [
        {"candidate": cid, "wids": pref[ph], "hl": hl, "tag": f"{cid}|{ph}"}
        for hl in HL_GRID for cid in CAND_BY_ID for ph in PHASES
    ]
    n_use = max(1, min(n_workers, len(specs)))
    print(f"hl: {len(specs)} root-finds on {n_use} workers", flush=True)
    if n_use == 1:
        recs = [_compare_job(s) for s in specs]
    else:
        with ProcessPoolExecutor(max_workers=n_use, initializer=_init_worker) as ex:
            recs = list(ex.map(_compare_job, specs, chunksize=1))
    cp = pd.DataFrame(recs)
    cp.to_csv(OUT_DIR / "hl_comparison_points.csv", index=False)
    # 参照窗扰动
    specs = []
    for _, r in cp.iterrows():
        cid = str(r["candidate"])
        if CAND_BY_ID[cid]["kind"] == "C0":
            param = 0.0
        elif bool(r["reachable"]):
            param = float(r["param"])
        else:
            print(f"skip hl={r['hl_tot_pct']} {cid} {r['phase']}: unreachable", flush=True)
            continue
        for w in pref[str(r["phase"])]:
            for pert in PERTS:
                specs.append({"window_id": w, "candidate": cid, "param": param,
                              "hl": float(r["hl_tot_pct"]), "pert": pert,
                              "tag": f"{cid}|{r['phase']}|hl{r['hl_tot_pct']:g}"})
    print(f"hl perturb: {len(specs)} solves", flush=True)
    out = run_specs(specs, n_workers)
    for s in out:
        s.pop("profile", None)
    (OUT_DIR / "hl_snaps.json").write_text(
        json.dumps(out, ensure_ascii=False), encoding="utf-8"
    )
    print(f"hl done: {sum(1 for s in out if s.get('ok'))}/{len(out)} ok", flush=True)


def stage_profiles(n_workers: int) -> None:
    _ensure_ctx()
    cp = pd.read_csv(OUT_DIR / "comparison_points.csv")
    pref = phase_ref_windows()
    specs = []
    for _, r in cp.iterrows():
        cid = str(r["candidate"])
        param = 0.0 if CAND_BY_ID[cid]["kind"] == "C0" else float(r["param"])
        if CAND_BY_ID[cid]["kind"] != "C0" and not bool(r["reachable"]):
            continue
        w = pref[str(r["phase"])][0]
        specs.append({"window_id": w, "candidate": cid, "param": param,
                      "hl": float(r["hl_tot_pct"]), "pert": "nom",
                      "want_profile": True})
    out = run_specs(specs, n_workers)
    keep = [s for s in out if s.get("profile")]
    (OUT_DIR / "profiles.json").write_text(json.dumps(keep, ensure_ascii=False),
                                           encoding="utf-8")
    print(f"profiles: {len(keep)} window-candidate pairs", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=("unit", "compare", "perturb", "hl", "profiles", "read", "all"))
    ap.add_argument("--n-workers", type=int, default=13)
    ap.add_argument("--source-commit", default=None)
    args = ap.parse_args()
    silence_solver_logs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage in ("unit",):
        stage_unit(args.n_workers)
    elif args.stage == "compare":
        stage_compare(args.n_workers)
    elif args.stage == "perturb":
        stage_perturb(args.n_workers)
    elif args.stage == "hl":
        stage_hl(args.n_workers)
    elif args.stage == "profiles":
        stage_profiles(args.n_workers)
    elif args.stage == "read":
        import p2_0a_read
        p2_0a_read.build_all(OUT_DIR, args.source_commit)
    else:
        stage_unit(args.n_workers)
        stage_compare(args.n_workers)
        stage_perturb(args.n_workers)
        stage_hl(args.n_workers)
        stage_profiles(args.n_workers)
        import p2_0a_read
        p2_0a_read.build_all(OUT_DIR, args.source_commit)


if __name__ == "__main__":
    main()
