#!/usr/bin/env python3
"""P2.4-A 双组分焦炭的单向耦合筛选原型（2026-11-19）。

依据：论文仓任务卡 `manuscript/model-side-handoff-p2.4a-20261119.md`；
判据来源 `manuscript/model-evidence-verification-20261119-p23a.md`。

**主流**：与 M2 完全相同（`SlagCaptureFraction = φ`）。**必须复用**已修正的 M2 实现，
不得重写——卡面明确要求，且 P2.3-A 刚因重写剖面把 P2.0-A 修过的截留碳 bug 又引入了一遍。
本脚本只**调用** `GasifierSystem` / `Cell` 的既有方法（含 `_calc_physics_props`），
不复制它们的任何逻辑。

**慢组分**（后处理，逐窗）：初值 = 被分出的固体。沿主流下游各格积分：

    rate_factor = f_slow · exp[−(ΔE/R)(1/T_p − 1/T_ref)]，T_ref = 1600 K
    只计 C+H2O 与 C+CO2（下游无 O2；C+H2 量级小，另记被忽略的上限）

表面积口径照 `Cell._calc_physics_props`：`S = 6·m·τ/(ρ_p·d_p)`——**τ 已折入 S**，
故 `calc_heterogeneous_rates` 返回的 `r_het` 就是该格的碳摩尔消耗量，不再乘时间。

**出口回加**：慢组分转化的碳按分量加入气相（C+H2O → CO+H2；C+CO2 → 2CO），
反应焓用模型自身的 `reaction_heat_release_w` 求，再按出口气相热容（模型自身的
Shomate 多项式）把温度下调，记 ΔT_corr。

读法 J0–J8。**本轮不反演、不加回流、不改合同。**

用法：

    ./.venv/bin/python scripts/generate_p2_4a_20261119.py \\
        --coal-set current \\
        --windows-csv outputs/diagnostics/p2.2d-20261118/windows_142.csv \\
        --stage all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import generate_p2_0a_20261113 as p20  # noqa: E402
import generate_p2_2a_20261118 as p22a  # noqa: E402

from model.constants import PhysicalConstants  # noqa: E402
from model.heat_loss import reaction_heat_release_w  # noqa: E402
from model.material import MaterialService  # noqa: E402
from model.physics import R_CONST  # noqa: E402
from model.state import StateVector  # noqa: E402
from model.p1_5_round_20261030 import run_case, silence_solver_logs  # noqa: E402

OUT_DIR = ROOT / "outputs" / "diagnostics" / "round-p2.4a-20261119"
CACHE_DIR = OUT_DIR / "cache"

P22D_READINGS = p22a.P22D_READINGS
P115A_DIR = p22a.P115A_DIR
P22A_DIR = ROOT / "outputs" / "diagnostics" / "round-p2.2a-20261118"

PHASES = p22a.PHASES
PHASE_LABEL = p22a.PHASE_LABEL
MAIN_HL = p22a.MAIN_HL
MAIN_CARRIER = p22a.MAIN_CARRIER
CARRIERS = p22a.CARRIERS
PERT_SCALE = p20.PERT_SCALE
PERTS = p20.PERTS

PHI_GRID = (0.10, 0.20, 0.30)
DE_GRID = (0.0, 50.0, 100.0, 150.0)
FSLOW_LO, FSLOW_HI = -4.0, 2.0
T_REF_K = 1600.0
M_C_KG_PER_MOL = 0.012011

Z_TOL = 0.5
SIGMA_TARGET = {"phase1": 3.5, "phase2": 3.64, "phase3": 3.55, "phase4": 3.50}
P1_TARGET_PP = 98.24

BISECT_MAX = 30
INTERIOR_PROBE = 4

PLANT_E_INTERVALS = {
    "phase1": {"coal": (-0.42, 0.20), "o2": (0.53, 1.31)},
    "phase2": {"coal": (0.32, 0.62), "o2": (0.13, 1.13)},
    "phase3": {"coal": (0.19, 0.57), "o2": (-0.10, 0.62)},
    "phase4": {"coal": (0.20, 0.73), "o2": (1.16, 2.06)},
}
LIU_FT_C = {"phase1": 1280.0, "phase2": 1290.0, "phase3": 1270.0, "phase4": 1250.0}
H2_PHASES = ("phase1", "phase2", "phase4")
J1_MIN_PHASES = 3
J3_MIN_SIGNED = 6
J4_FT_MARGIN_K = 50.0
J4_MIN_AT_FT = 3
J5_MAX_DEV_PP = 2.0
J6_MAX_DT_CORR_K = 50.0
J6_MAX_SLOW_C_SHARE = 0.05


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path) -> str:
    """仓库内相对仓库根；仓库外用 ../ 形式，绝不含绝对路径（P2.3-A 的教训）。"""
    p = Path(path).resolve()
    try:
        return p.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return Path(os.path.relpath(p, ROOT.resolve())).as_posix()


def write_csv(df: pd.DataFrame, name: str) -> Path:
    """写本轮 CSV。不得复用别轮的 writer（会写进别的轮次目录）。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / name
    df.replace({np.nan: ""}).to_csv(out, index=False, lineterminator="\n")
    return out


def _f(v) -> float:
    try:
        out = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return out


# ---------------------------------------------------------------------------
# 缓存
# ---------------------------------------------------------------------------
def cache_key(*, window_id: str, coal_set: str, carrier_tag: str, hl: float,
              pert: str, phi: float) -> str:
    return json.dumps({
        "window_id": str(window_id), "phi": repr(float(phi)),
        "hl": repr(float(hl)), "pert": pert, "carrier_tag": carrier_tag,
        "coal_set": coal_set, "coal_digest": p22a.coal_digest(coal_set),
        "windows_sha": p22a.table_sha("main"),
        "kappa": repr(float(p20.KAPPA)),
        "L_wsr": repr(float(p20.L_WSR)), "dz_down": repr(float(p20.DZ_DOWN)),
        "code_fingerprint": p20.CODE_FINGERPRINT,
        # 快照结构版本：改了 solve_main 存什么字段就 +1，否则旧缓存会被当作有效命中，
        # 表现为「字段不存在」或更隐蔽的「用了旧的物理量」。
        "snapshot_schema": 2,
    }, ensure_ascii=False, sort_keys=True)


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
    return hit.get("snap") if hit.get("key") == key else None


def _store_cache(key: str, snap: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(key).write_text(
        json.dumps({"key": key, "snap": snap}, ensure_ascii=False),
        encoding="utf-8")


# ---------------------------------------------------------------------------
# 主流（M2 实现）：逐格物理量全部取自 Cell._calc_physics_props
# ---------------------------------------------------------------------------
def solve_main(*, window_id: str, coal_set: str, carrier_tag: str, hl: float,
               pert: str, phi: float) -> dict:
    df = p22a.table("main")
    o2m, coalm, carm = PERT_SCALE[pert]
    dry = float(df.loc[window_id, "coal_dry_recon_phase_kg_s"]) * float(coalm)
    carrier_nm3 = float(df.loc[window_id, CARRIERS[carrier_tag]]) * float(carm)
    key = cache_key(window_id=window_id, coal_set=coal_set,
                    carrier_tag=carrier_tag, hl=hl, pert=pert, phi=phi)
    hit = _load_cache(key)
    if hit is not None:
        return hit

    op = p22a.build_op(hl, o2m, dry, carrier_nm3)
    op["SlagCaptureFraction"] = float(phi)
    rec = run_case(p22a.rows_for(coal_set)[window_id],
                   n_cells=p20.mesh_n(), coal_flow_dry_kg_s=dry,
                   keep_system=True, mesh_family="wsr_plus_pfr", extra_op=op)
    system = rec.get("_system")
    snap: dict = {
        "window_id": window_id, "phase": p22a._phase_of(window_id),
        "phi": float(phi), "hl": float(hl), "pert": pert,
        "carrier_tag": carrier_tag, "coal_set": coal_set,
        "coal_dry_kg_s": float(dry), "carrier_nm3_h": float(carrier_nm3),
        "execution_status": str(rec.get("execution_status")),
        "ok": False, "cells": [],
    }
    if system is None or not system.results or \
            rec.get("execution_status") != "completed":
        rec.pop("_system", None)
        _store_cache(key, snap)
        return snap

    cd = float(system.coal_props.get("Cd", 0.0)) / 100.0
    c_in = float(system.W_dry) * cd
    c_fed_total = float(system.W_dry) * float(system.Cd_total)
    outlet = system.results[-1]
    gas = np.asarray(outlet.gas_moles, dtype=float)
    cap = system.slag_capture
    c_out = float(outlet.solid_mass) * float(outlet.carbon_fraction)
    c_cap = 0.0 if not isinstance(cap, dict) else float(cap.get("carbon_kg_s", 0.0))

    # 逐格物理量：直接调 Cell._calc_physics_props（与 gasifier_system.py:968 同一调用式）
    cells = []
    for i, (cell, st) in enumerate(zip(system.cells, system.results)):
        phys = cell._calc_physics_props(st, c_fed_total)
        cells.append({
            "i": int(i), "dz_m": float(cell.dz),
            "tau_s": float(phys["tau"]) if "tau" in phys
            else float(cell.dz) / max(float(phys["v_g"]),
                                      PhysicalConstants.MIN_SLIP_VELOCITY),
            "T_gas_K": float(st.T), "T_solid_K": float(st.T_solid_or_gas),
            "P_Pa": float(st.P), "gas_moles": [float(v) for v in st.gas_moles],
            "d_p_m": float(phys["d_p"]), "rho_p_kg_m3": float(phys["rho_p"]),
            "solid_mass_kg_s": float(st.solid_mass),
            "carbon_fraction": float(st.carbon_fraction),
        })

    snap.update({
        "ok": True,
        "X_C_main_pp": float((1.0 - (c_out + c_cap) / max(c_in, 1e-12)) * 100.0),
        "T_out_K": float(outlet.T),
        "T_wsr_K": float(system.results[0].T),
        "c_in_kg": float(c_in),
        "c_out_main_kg": c_out,
        "F_CO": float(gas[2]), "F_CO2": float(gas[3]), "F_H2": float(gas[5]),
        "F_H2O": float(gas[7]), "F_CH4": float(gas[1]), "F_H2S": float(gas[4]),
        "F_O2": float(gas[0]), "F_N2": float(gas[6]),
        "dry_total_mol_s": float(np.sum(gas[:7])),
        "capture_fraction": (0.0 if not isinstance(cap, dict)
                             else float(cap.get("fraction", 0.0))),
        "capture_solid_mass_kg_s": (0.0 if not isinstance(cap, dict)
                                    else float(cap.get("solid_mass_kg_s", 0.0))),
        "capture_carbon_kg_s": c_cap,
        "capture_carbon_fraction": (0.0 if not isinstance(cap, dict)
                                    else float(cap.get("carbon_fraction", 0.0))),
        "cp_char": float(system.coal_props.get("cp_char", 1300.0)),
        # 卡面 §1：慢组分的粒径与密度取**全混区出口值**，不逐格更新
        "wsr_d_p_m": float(cells[0]["d_p_m"]),
        "wsr_rho_p_kg_m3": float(cells[0]["rho_p_kg_m3"]),
        "cells": cells,
        "outlet_gas_moles": [float(v) for v in outlet.gas_moles],
        "outlet_T_solid_K": float(outlet.T_solid_or_gas),
    })
    rec.pop("_system", None)
    _store_cache(key, snap)
    return snap


# ---------------------------------------------------------------------------
# 慢组分积分（后处理）
# ---------------------------------------------------------------------------
_KIN = {"svc": None}


def _kinetics():
    if _KIN["svc"] is None:
        from model.kinetics_service import KineticsService
        _KIN["svc"] = KineticsService()
    return _KIN["svc"]


def _gas_state(cell: dict) -> StateVector:
    return StateVector(
        gas_moles=np.asarray(cell["gas_moles"], dtype=float),
        solid_mass=0.0, carbon_fraction=0.0,
        T=float(cell["T_gas_K"]), P=float(cell["P_Pa"]),
        z=0.0, T_solid=float(cell["T_solid_K"]))


def slow_integrate(snap: dict, *, dE_kjmol: float, f_slow: float) -> dict:
    """慢组分沿下游逐格积分（卡面 §1）。"""
    cells = snap.get("cells") or []
    if not snap.get("ok") or not cells:
        return {"ok": False, "reason": "主流无逐格状态",
                "c0_kg": 0.0, "c_remaining_kg": 0.0,
                "a_H2O_mol": 0.0, "a_CO2_mol": 0.0,
                "h2_ignored_upper_mol": 0.0, "profile": []}

    c0 = float(snap["capture_carbon_kg_s"])
    m_s = float(snap["capture_solid_mass_kg_s"])
    if c0 <= 1e-15 or m_s <= 1e-15:
        return {"ok": False, "reason": "截留量为零（φ 过小或未发生分流）",
                "c0_kg": c0, "c_remaining_kg": c0,
                "a_H2O_mol": 0.0, "a_CO2_mol": 0.0,
                "h2_ignored_upper_mol": 0.0, "profile": []}

    c_s = c0
    k = _kinetics()
    d_p = float(snap.get("wsr_d_p_m") or float("nan"))
    rho_p = float(snap.get("wsr_rho_p_kg_m3") or float("nan"))
    if not (math.isfinite(d_p) and d_p > 0.0
            and math.isfinite(rho_p) and rho_p > 0.0):
        return {"ok": False, "reason": "全混区出口 d_p/rho_p 不可用",
                "c0_kg": c0, "c_remaining_kg": c0,
                "a_H2O_mol": 0.0, "a_CO2_mol": 0.0,
                "h2_ignored_upper_mol": 0.0, "profile": []}
    prof = []
    a_h2o = a_co2 = h2_ign = 0.0
    for cell in cells:
        if cell["i"] == 0:      # 截留发生在全混区出口，cell 0 本身不分流
            continue
        # 表面积口径与 Cell._calc_physics_props 一致：S = 6·m·τ/(ρ_p·d_p)
        # d_p 与 rho_p 固定为全混区出口值（卡面 §1）
        S = 6.0 * m_s * float(cell["tau_s"]) / (rho_p * d_p)
        X_slow = min(max(1.0 - c_s / c0, 0.0), 0.999)
        g = math.exp(-(float(dE_kjmol) * 1000.0 / R_CONST)
                     * (1.0 / float(cell["T_solid_K"]) - 1.0 / T_REF_K))
        factor = float(f_slow) * g
        st = _gas_state(cell)
        r = k.calc_heterogeneous_rates(
            st, d_p, S, X_total=X_slow,
            T_particle=float(cell["T_solid_K"]),
            gasification_rate_factor=factor, char_co2_rate_factor=1.0)
        dc_h2o = max(float(r.get("C+H2O", 0.0)), 0.0)
        dc_co2 = max(float(r.get("C+CO2", 0.0)), 0.0)
        h2_ign += max(float(r.get("C+H2", 0.0)), 0.0)
        tot = dc_h2o + dc_co2
        # 2026-11-20（审阅意见 P1）：限幅必须换算成**摩尔**再比。
        # tot 是 mol/s、c_s 是 kg/s，此前直接比 `tot > c_s` 等于拿 mol 比 kg，
        # 于是限幅在消耗到可用碳的约 1.2% 时就触发（c_s/tot 的比值），
        # 1632 个格出现「dC(mol) 数值上等于前格碳量(kg)」。
        # 上一轮我只修了下面的减法、漏了这一处比较。
        c_mol = c_s / M_C_KG_PER_MOL
        if tot > c_mol and tot > 0.0:     # 每格消耗不超过该格慢组分碳量（U5）
            sc = c_mol / tot
            dc_h2o *= sc
            dc_co2 *= sc
            tot = c_mol
        c_s = max(c_s - tot * M_C_KG_PER_MOL, 0.0)
        m_s = max(m_s - tot * M_C_KG_PER_MOL, 0.0)
        a_h2o += dc_h2o
        a_co2 += dc_co2
        prof.append({
            "cell_index": int(cell["i"]),
            "z_m": float(sum(c["dz_m"] for c in cells[:cell["i"] + 1])),
            "T_solid_K": float(cell["T_solid_K"]),
            "tau_s": float(cell["tau_s"]),
            "X_slow": float(X_slow), "g_factor": float(g),
            "rate_factor": float(factor),
            "dC_H2O_mol": float(dc_h2o), "dC_CO2_mol": float(dc_co2),
            "c_remaining_mol": float(c_s / M_C_KG_PER_MOL),
            "solid_remaining_kg_s": float(m_s),
        })
        if c_s <= 1e-20:
            break

    return {"ok": True, "c0_kg": c0, "c_remaining_kg": c_s,
            "slow_converted_kg": c0 - c_s,
            "a_H2O_mol": a_h2o, "a_CO2_mol": a_co2,
            "h2_ignored_upper_mol": h2_ign, "profile": prof}


# ---------------------------------------------------------------------------
# 出口回加与 ξ
# ---------------------------------------------------------------------------
def add_back(snap: dict, slow: dict) -> dict:
    """把慢组分转化的碳按分量加入出口气相，并算 ΔT_corr。"""
    a1 = float(slow.get("a_H2O_mol", 0.0))
    a2 = float(slow.get("a_CO2_mol", 0.0))
    gm = list(snap["outlet_gas_moles"])
    # C+H2O → CO + H2；C+CO2 → 2 CO
    gm[2] += a1 + 2.0 * a2
    gm[5] += a1
    gm[7] -= a1
    gm[3] -= a2

    # 反应焓：用模型自身的 reaction_heat_release_w（正 = 放热）
    T_out = float(snap["T_out_K"])
    q_w = reaction_heat_release_w(
        {"C+O2": 0.0, "C+H2O": a1, "C+CO2": a2, "C+H2": 0.0},
        {}, phi=1.0, f_co2=1.0, T_gas_K=T_out,
        T_solid_K=float(snap["outlet_T_solid_K"]),
        cp_char_j_per_kg_k=float(snap["cp_char"]))
    # 气相总热容：模型自身 Shomate 多项式的数值导数
    st = StateVector(gas_moles=np.asarray(gm, dtype=float), solid_mass=0.0,
                     carbon_fraction=0.0, T=T_out, P=101325.0, z=0.0,
                     T_solid=float(snap["outlet_T_solid_K"]))
    dT = 1.0
    st_up = StateVector(gas_moles=st.gas_moles, solid_mass=0.0,
                        carbon_fraction=0.0, T=T_out + dT, P=101325.0, z=0.0,
                        T_solid=float(snap["outlet_T_solid_K"]))
    st_dn = StateVector(gas_moles=st.gas_moles, solid_mass=0.0,
                        carbon_fraction=0.0, T=T_out - dT, P=101325.0, z=0.0,
                        T_solid=float(snap["outlet_T_solid_K"]))
    cp = (MaterialService.get_gas_enthalpy(st_up)
          - MaterialService.get_gas_enthalpy(st_dn)) / (2.0 * dT)   # J/K

    # q_w 是净放热（正=放热）。温度变化 = q_w/cp；吸热时 q_w<0 → dt_corr<0。
    dt_corr = (q_w / cp) if cp > 1e-9 else 0.0
    return {
        "gas_moles_after": gm, "q_reaction_W": float(q_w),
        "cp_gas_J_per_K": float(cp), "dT_corr_K": float(dt_corr),
        "T_out_corrected_K": float(T_out + dt_corr),
        "a_H2O_mol": a1, "a_CO2_mol": a2,
    }


def xi_from_gas(gm: list[float], plant_co2: float) -> dict:
    """求 ξ 使回加后的 CO2 干基体积分数等于装置值。

    变换 CO + H2O → CO2 + H2 使 CO2 = (F_CO2+ξ)/(dry+ξ)，**随 ξ 递增**。
    """
    def co2_at(xi: float) -> float:
        f_co = gm[2] - xi
        f_co2 = gm[3] + xi
        f_h2 = gm[5] + xi
        f_h2o = gm[7] - xi
        dry = sum(gm[:7]) + xi
        if dry <= 0.0 or f_co < 0.0 or f_h2o < 0.0:
            return float("nan")
        return f_co2 / dry * 100.0

    m0 = co2_at(0.0)
    if not math.isfinite(m0):
        return {"ok": False, "reason": "回加后气相出现负组分", "xi": None}
    if m0 >= plant_co2:
        # ξ ≥ 0 只会把 CO2 抬得更高，无法回到目标
        return {"ok": False, "reason": f"模型 CO2 {m0:.3f} ≥ 装置 {plant_co2:.3f}，ξ 无法闭合",
                "xi": None, "co2_at_xi0": m0}
    lo, hi = 0.0, 50.0
    m_hi = co2_at(hi)
    for _ in range(24):
        if math.isfinite(m_hi) and m_hi >= plant_co2:
            break
        hi *= 1.5
        m_hi = co2_at(hi)
    if not (math.isfinite(m_hi) and m_hi >= plant_co2):
        return {"ok": False, "reason": "ξ 括区失败", "xi": None}
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        m = co2_at(mid)
        if not math.isfinite(m):
            break
        if abs(m - plant_co2) < 1e-7 or (hi - lo) < 1e-9:
            return {"ok": True, "xi": float(mid), "co2_after": float(m)}
        if m < plant_co2:
            lo = mid
        else:
            hi = mid
    mid = 0.5 * (lo + hi)
    return {"ok": True, "xi": float(mid), "co2_after": float(co2_at(mid))}


def composition_after_xi(gm: list[float], xi: float) -> dict:
    f = list(gm)
    f[2] -= xi
    f[3] += xi
    f[5] += xi
    f[7] -= xi
    dry = sum(f[:7])
    if dry <= 0.0:
        return {}
    names = ["O2", "CH4", "CO", "CO2", "H2S", "H2", "N2"]
    out = {f"{n}_vol_pct": f[i] / dry * 100.0 for i, n in enumerate(names)}
    out["H2O_vol_pct"] = f[7] / dry * 100.0
    out["dry_per_coal_mol_kg"] = dry / max(1e-12, 0.0)   # 由调用方覆盖
    return out


# ---------------------------------------------------------------------------
# 配置求值（给定 φ、ΔE、f_slow）
# ---------------------------------------------------------------------------
def case_metrics(*, window_id: str, phi: float, dE: float, f_slow: float,
                 coal_set: str, pert: str = "nom",
                 hl: float = MAIN_HL) -> dict:
    """单窗单配置的全部指标（主流 + 慢组分 + 回加 + ξ）。"""
    snap = solve_main(window_id=window_id, coal_set=coal_set,
                      carrier_tag=MAIN_CARRIER, hl=hl, pert=pert, phi=phi)
    out: dict = {"window_id": window_id, "phi": float(phi), "dE": float(dE),
                 "log10_f_slow": float(math.log10(f_slow)) if f_slow > 0 else float("nan"),
                 "pert": pert, "ok": False, "reason": ""}
    if not snap.get("ok"):
        out["reason"] = "主流求解失败"
        return out
    slow = slow_integrate(snap, dE_kjmol=dE, f_slow=f_slow)
    if not slow.get("ok"):
        out["reason"] = str(slow.get("reason"))
        return out

    c_in = float(snap["c_in_kg"])
    c_rem = float(slow["c_remaining_kg"])
    c_out_main = float(snap["c_out_main_kg"])
    xc = (1.0 - (c_out_main + c_rem) / max(c_in, 1e-12)) * 100.0

    ab = add_back(snap, slow)
    df = p22a.table("main")
    plant_co2 = float(df.loc[window_id, "co2_vol_pct_mean"])
    xi = xi_from_gas(ab["gas_moles_after"], plant_co2)
    comp: dict = {}
    if xi.get("ok"):
        comp = composition_after_xi(ab["gas_moles_after"], float(xi["xi"]))
        comp["dry_per_coal_mol_kg"] = (
            (sum(ab["gas_moles_after"][:7]) + float(xi["xi"]))
            / max(float(snap["coal_dry_kg_s"]), 1e-12))

    out.update({
        "ok": True,
        "X_C_pp": float(xc),
        "X_C_main_pp": float(snap["X_C_main_pp"]),
        "T_out_K": float(snap["T_out_K"]),
        "T_out_corrected_K": float(ab["T_out_corrected_K"]),
        "dT_corr_K": float(ab["dT_corr_K"]),
        "q_reaction_W": float(ab["q_reaction_W"]),
        "c_in_kg": c_in,
        "capture_carbon_kg": float(slow["c0_kg"]),
        "slow_converted_kg": float(slow["c0_kg"] - c_rem),
        "slow_remaining_kg": c_rem,
        "slow_converted_share_of_feed_C": float(
            (slow["c0_kg"] - c_rem) / max(c_in, 1e-12)),
        "a_H2O_mol": float(slow["a_H2O_mol"]),
        "a_CO2_mol": float(slow["a_CO2_mol"]),
        "h2_ignored_upper_mol": float(slow["h2_ignored_upper_mol"]),
        "capture_fraction": float(snap["capture_fraction"]),
        "xi_mol_s": None if not xi.get("ok") else float(xi["xi"]),
        "xi_ok": bool(xi.get("ok")),
        "xi_reason": "" if xi.get("ok") else str(xi.get("reason")),
        "CO_vol_pct": comp.get("CO_vol_pct"),
        "H2_vol_pct": comp.get("H2_vol_pct"),
        "CO2_vol_pct": comp.get("CO2_vol_pct"),
        "CH4_vol_pct": comp.get("CH4_vol_pct"),
        "dry_per_coal_mol_kg": comp.get("dry_per_coal_mol_kg"),
    })
    return out


def median_xc(*, phi: float, dE: float, f_slow: float, wids: list[str],
              coal_set: str, pert: str = "nom") -> tuple[float, int, int, dict]:
    """参照窗 X_C 中位。返回 (中位, n_ok, n_fail, 汇总明细)。"""
    vals, fails, detail = [], 0, {}
    for w in wids:
        m = case_metrics(window_id=w, phi=phi, dE=dE, f_slow=f_slow,
                         coal_set=coal_set, pert=pert)
        if m.get("ok"):
            vals.append(float(m["X_C_pp"]))
            detail[w] = m
        else:
            fails += 1
    med = float(np.median(vals)) if vals else float("nan")
    return med, len(vals), fails, detail


def root_fslow(*, phi: float, dE: float, phase: str, wids: list[str],
               target: float, coal_set: str) -> dict:
    """对 log10 f_slow 求根，使参照窗 X_C 中位 = target。（自适应括区，不给裸 NaN）"""
    out = {"phi": float(phi), "dE": float(dE), "phase": phase,
           "target_X_C_pp": float(target), "reachable": False, "reason": "",
           "log10_f_slow": None, "f_slow": None,
           "X_C_pp_median": None, "z": None, "n_probe": 0,
           "n_solve_fail": 0}
    sigma = SIGMA_TARGET[phase]

    def probe(u: float) -> tuple[float, int]:
        out["n_probe"] += 1
        med, _, nf, _ = median_xc(phi=phi, dE=dE, f_slow=10.0 ** u,
                                  wids=wids, coal_set=coal_set)
        return med, nf

    lo_u, hi_u = FSLOW_LO, FSLOW_HI
    probes: list[tuple[float, float]] = []
    for u in (lo_u, hi_u):
        m, nf = probe(u)
        if math.isfinite(m):
            probes.append((u, m))
    if not probes:
        out["reason"] = "参数范围两端点均无解"
        return out

    def bracketed() -> bool:
        ss = sorted(probes)
        return any((a[1] - target) * (b[1] - target) <= 0 for a, b in zip(ss, ss[1:]))

    if not bracketed():
        for k in range(1, INTERIOR_PROBE + 1):
            u = lo_u + (hi_u - lo_u) * k / (INTERIOR_PROBE + 1)
            m, nf = probe(u)
            if math.isfinite(m):
                probes.append((u, m))
            if bracketed():
                break

    if not bracketed():
        # 2026-11-20（审阅意见 P2）：卡面的可达性判据是 z，不是「能否夹住目标」。
        # 夹不住精确目标但最接近的可解点已满足 |z| ≤ 0.5 时，仍应记为可达。
        # 此前一律记不可达，把 7 个 z=0.22~0.48 的 φ=0.1 点误判掉了，
        # 进而夸大了「φ=0.1 对阶段 2–4 不可达」这个结构性结论。
        ms = [m for _, m in probes]
        best_u, best_m = None, None
        if max(ms) < target:
            best_u = max(probes, key=lambda t: t[1])[0]
            best_m = max(ms)
            out["reason"] = (f"可解区间最大 X_C {best_m:.2f}% 仍低于目标 "
                             f"{target:.2f}%")
        elif min(ms) > target:
            best_u = min(probes, key=lambda t: t[1])[0]
            best_m = min(ms)
            out["reason"] = (f"可解区间最小 X_C {best_m:.2f}% 仍高于目标 "
                             f"{target:.2f}%")
        else:
            out["reason"] = "可解点未夹住目标（中间存在无解带）"
        if best_m is not None:
            out["X_C_pp_median"] = float(best_m)
            out["z"] = float((best_m - target) / sigma)
            out["log10_f_slow"] = float(best_u)
            out["f_slow"] = float(10.0 ** best_u)
            if abs(out["z"]) <= Z_TOL:
                out["reachable"] = True
                out["reason"] = (out["reason"]
                                 + f"；但最近可解点 |z|={abs(out['z']):.3f} ≤ "
                                   f"{Z_TOL}，按卡面 z 判据记为可达")
            else:
                out["reason"] = (out["reason"]
                                 + f"；最近可解点 |z|={abs(out['z']):.3f} > "
                                   f"{Z_TOL}")
        return out

    probes.sort()
    lo = hi = None
    for (u1, m1), (u2, m2) in zip(probes, probes[1:]):
        if (m1 - target) * (m2 - target) <= 0:
            lo, hi = u1, u2
            break
    if lo is None:
        out["reason"] = "未能选出括区"
        return out

    best = (None, None, float("inf"), 0)
    for _ in range(BISECT_MAX):
        mid = 0.5 * (lo + hi)
        m, nf = probe(mid)
        if not math.isfinite(m):
            hi = mid
            continue
        dev = abs(m - target)
        if dev < best[2]:
            best = (mid, m, dev, nf)
        if dev <= 0.05 or (hi - lo) < 1e-4:
            out.update({"log10_f_slow": float(mid), "f_slow": float(10.0 ** mid),
                        "X_C_pp_median": float(m), "n_solve_fail": nf,
                        "z": float((m - target) / sigma),
                        "reachable": bool(abs((m - target) / sigma) <= Z_TOL)})
            if not out["reachable"]:
                out["reason"] = (f"求根收敛但 |z|={abs((m-target)/sigma):.3f} > {Z_TOL}")
            return out
        # 方向：f_slow 增大 → X_C 上升（单调）
        if m < target:
            lo = mid
        else:
            hi = mid

    if best[0] is not None:
        z = (best[1] - target) / sigma
        out.update({"log10_f_slow": float(best[0]), "f_slow": float(10.0 ** best[0]),
                    "X_C_pp_median": float(best[1]), "z": float(z),
                    "n_solve_fail": best[3],
                    "reachable": bool(abs(z) <= Z_TOL)})
        if not out["reachable"]:
            out["reason"] = f"二分未收敛到 |z|≤{Z_TOL}（最优 |z|={abs(z):.3f}）"
    else:
        out["reason"] = "二分全程无可解点"
    return out


# ---------------------------------------------------------------------------
# 进程池（主流求解是重活；慢组分后处理在主进程做）
# ---------------------------------------------------------------------------
_WINDOWS_CSV = {"path": ""}
_POOL: ProcessPoolExecutor | None = None


def _winit(windows_csv: str) -> None:
    """worker 初始化。**不要**在此做 prepare_coal_inputs——多个 worker 并发
    shutil.copyfile 同一批煤质会互相读到半写状态（P2.3-A 实际踩过）。"""
    silence_solver_logs()
    p22a.init_ctx(Path(windows_csv))


def _wjob(spec: dict) -> dict:
    return solve_main(**spec)


def init_pool(windows_csv: str, n_workers: int) -> None:
    global _POOL
    if _POOL is not None:
        return
    p22a.prepare_coal_inputs()          # 父进程独占
    _WINDOWS_CSV["path"] = str(windows_csv)
    _POOL = ProcessPoolExecutor(max_workers=max(1, int(n_workers)),
                                initializer=_winit,
                                initargs=(str(windows_csv),))


def close_pool() -> None:
    global _POOL
    if _POOL is not None:
        _POOL.shutdown(wait=True)
        _POOL = None


def batch_main(specs: list[dict]) -> list[dict]:
    if not specs:
        return []
    if _POOL is None:
        return [solve_main(**sp) for sp in specs]
    return list(_POOL.map(_wjob, specs))


def _mjob(spec: dict) -> dict:
    return case_metrics(**spec)


def batch_metrics(specs: list[dict]) -> list[dict]:
    """批量跑 case_metrics（主流走磁盘缓存，慢组分后处理在 worker 内完成）。

    慢组分是纯 Python 循环（每窗约 0.2 s），31808 个窗×摄动在主进程串行要 100 分钟；
    池化后约 8 分钟。这也是唯一需要池化的后处理。
    """
    if not specs:
        return []
    if _POOL is None:
        return [case_metrics(**sp) for sp in specs]
    return list(_POOL.map(_mjob, specs))


# ---------------------------------------------------------------------------
# 目标
# ---------------------------------------------------------------------------
def targets() -> dict[str, float]:
    rd = pd.read_csv(P22D_READINGS)
    out = {}
    for ph in ("phase2", "phase3", "phase4"):
        lbl = PHASE_LABEL[ph]
        hit = rd[(rd["reading_id"] == "FINAL") & (rd["phase"] == lbl)
                 & (rd["metric"] == "target_midpoint_pct")]
        if not len(hit):
            raise SystemExit(f"P2.2-D readings.csv 缺 {lbl} 的 FINAL/target_midpoint_pct")
        out[ph] = float(hit["value"].iloc[0])
    out["phase1"] = float(P1_TARGET_PP)   # 代理目标，结论中注明
    return out


# ---------------------------------------------------------------------------
# stage unit（U1–U5）
# ---------------------------------------------------------------------------
def stage_unit(args) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p22a.prepare_coal_inputs()
    p22a.require_pairing(args.windows_csv, args.coal_set, "compare")
    p22a.init_ctx(Path(args.windows_csv))
    ref = p22a._ref_windows("main")
    rows: list[dict] = []
    hard: list[str] = []

    hv = p22a.coal_hv(args.coal_set)
    exp = p22a.HV_EXPECT[args.coal_set]
    bad = {p: [hv[p], exp[p]] for p in PHASES if hv[p] != exp[p]}
    rows.append({"check": "PAIRING",
                 "item": "输入配对 + 四阶段热值",
                 "value": ";".join(f"{PHASE_LABEL[p]}={hv[p]:.0f}" for p in PHASES),
                 "threshold": "a24b5fe6… / 28179-28440-28579-28588",
                 "pass": not bad,
                 "detail": json.dumps(bad, ensure_ascii=False) if bad else "符合"})
    if bad:
        hard.append("PAIRING")

    # U1（卡面 §5）：f_slow=0 时，每个 φ 在参照窗上的 X_C、出口温度、组成与 P2.3-A M2
    # 在 f_s = φ 处一致。P2.3-A 的 M2 求根 f_s 与本轮 φ 网格（0.1/0.2/0.3）都不重合，
    # 卡面为此指定了替代：**用同一代码现算并与 P2.3-A 的比较点核对一处**。
    # 2026-11-20（审阅意见 P2）：此前代码在「φ 与 P2.3-A 根不匹配」时直接跳过，
    # worst 保持初始 0 就被记为通过——那个 U1 是空的。现在强制走替代路径，
    # 且要求至少完成一次比较才能通过。
    p23 = pd.read_csv(ROOT / "outputs" / "diagnostics" / "round-p2.3a-20261118"
                      / "comparison_points.csv", keep_default_na=False)
    u1_rows, u1_worst, u1_n = [], 0.0, 0
    for ph in PHASES:
        hit = p23[(p23["candidate"] == "M2") & (p23["phase"] == ph)
                  & (p23["reachable"].astype(str).str.lower() == "true")]
        if not len(hit):
            continue
        f_s = float(hit["u_root"].iloc[0])
        ref_val = float(hit["X_C_pp_median"].iloc[0])
        ref_T = _fnum(hit["T_out_K"].iloc[0]) if "T_out_K" in hit else float("nan")
        med, n_ok, _, det = median_xc(phi=f_s, dE=0.0, f_slow=0.0,
                                      wids=ref[ph], coal_set=args.coal_set)
        if not math.isfinite(med):
            u1_rows.append(f"{PHASE_LABEL[ph]}: 现算无解")
            continue
        rel = abs(med - ref_val) / max(abs(ref_val), 1e-12)
        T_med = (float(np.median([m["T_out_K"] for m in det.values()]))
                 if det else float("nan"))
        dT = abs(T_med - ref_T) / max(abs(ref_T), 1e-12) if math.isfinite(ref_T) else float("nan")
        u1_worst = max(u1_worst, rel)
        if math.isfinite(dT):
            u1_worst = max(u1_worst, dT)
        u1_n += 1
        u1_rows.append(f"{PHASE_LABEL[ph]}: f_s={f_s:.6g} 现算 {med:.4f}% vs "
                       f"P2.3-A {ref_val:.4f}%（相对 {rel:.2e}）；"
                       f"T_out 相对差 {dT:.2e}")
    u1_ok = bool(u1_n >= 1 and u1_worst <= 1e-9)
    rows.append({"check": "U1",
                 "item": "f_slow=0 的 X_C/温度与 P2.3-A M2 比较点一致（卡面 §5 替代路径）",
                 "value": u1_worst, "threshold": 1e-9, "pass": u1_ok,
                 "detail": (f"完成 {u1_n} 处对照；" + "；".join(u1_rows))
                 if u1_rows else "未能完成任何对照——按卡面不得记为通过"})
    if not u1_ok:
        hard.append("U1")

    # U2：ΔE=0 时 g ≡ f_slow；T_p=T_ref 时 g=f_slow
    g0 = math.exp(-(0.0 / R_CONST) * (1.0 / 1500.0 - 1.0 / T_REF_K))
    g_ref = math.exp(-(120.0 * 1000.0 / R_CONST) * (1.0 / T_REF_K - 1.0 / T_REF_K))
    ok_u2 = abs(g0 - 1.0) < 1e-15 and abs(g_ref - 1.0) < 1e-15
    rows.append({"check": "U2", "item": "ΔE=0 时 g≡1；T_p=T_ref 时 g=1（故倍数为 f_slow）",
                 "value": f"{g0:.15f}; {g_ref:.15f}", "threshold": "1",
                 "pass": bool(ok_u2), "detail": "解析式直接求值"})
    if not ok_u2:
        hard.append("U2")

    # U3：f_slow 递增使 X_C 上升、ΔT_corr 更负
    w3 = ref["phase4"][0]
    seq = [0.0, 1e-7, 1e-6, 1e-5]
    xs, dts = [], []
    for fs in seq:
        m = case_metrics(window_id=w3, phi=0.2, dE=50.0, f_slow=fs,
                         coal_set=args.coal_set)
        xs.append(float(m["X_C_pp"]) if m.get("ok") else float("nan"))
        dts.append(float(m["dT_corr_K"]) if m.get("ok") else float("nan"))
    up = all(xs[i] < xs[i + 1] for i in range(len(xs) - 1)
             if math.isfinite(xs[i]) and math.isfinite(xs[i + 1]))
    down = all(dts[i] >= dts[i + 1] - 1e-12 for i in range(len(dts) - 1)
               if math.isfinite(dts[i]) and math.isfinite(dts[i + 1]))
    rows.append({"check": "U3", "item": "f_slow 递增 → X_C 上升、ΔT_corr 更负（单调）",
                 "value": json.dumps({"f_slow": seq, "X_C": xs, "dT": dts}),
                 "threshold": "单调", "pass": bool(up and down),
                 "detail": f"X_C 单调上升={up}，ΔT_corr 单调下降={down}"})
    if not (up and down):
        hard.append("U3")

    # U4（卡面 §5）：回加后出口的 C、H、O 元素闭合，不高于 P2.3-A M2 同量的 10 倍。
    # 2026-11-20（审阅意见 P2）：此前只查 C、且用写死的 1e-2，既没有 H/O、
    # 也没有与 M2 基线比较。回加显式消耗 H2O/CO2 并生成 CO/H2，只用碳闭合
    # 无法验证那部分化学计量。
    from model.p0_round_20260930 import ELEMENT_KG_PER_MOL, atoms_from_gas
    w4 = ref["phase2"][0]
    coal = p22a.rows_for(args.coal_set)[w4]

    def _closure(f_slow_: float, dE_: float, phi_: float) -> dict:
        sn = solve_main(window_id=w4, coal_set=args.coal_set,
                        carrier_tag=MAIN_CARRIER, hl=MAIN_HL, pert="nom",
                        phi=phi_)
        if not sn.get("ok"):
            return {}
        sl = slow_integrate(sn, dE_kjmol=dE_, f_slow=f_slow_)
        if not sl.get("ok"):
            return {}
        ab = add_back(sn, sl)
        gm = np.asarray(ab["gas_moles_after"], dtype=float)
        atoms = atoms_from_gas(gm)
        # 进料：煤的 C/H/O（含水分）+ O2 + CO2 载气的碳与氧
        w_c = float(coal.get("w_c", 0.0)) if isinstance(coal, dict) else float("nan")
        c_coal = float(sn["c_in_kg"])
        # 未转化碳（主流出口 + 慢组分剩余）计入入炉侧的可比量
        c_unconv = float(sn["c_out_main_kg"]) + float(sl["c_remaining_kg"])
        c_carrier = (float(sn["carrier_nm3_h"]) / 22.4 * 1000.0 / 3600.0
                     * M_C_KG_PER_MOL)
        out = {"C": None, "H": None, "O": None}
        out["C"] = abs(atoms.get("C", 0.0) * ELEMENT_KG_PER_MOL["C"]
                       + c_unconv - (c_coal + c_carrier)) / max(c_coal + c_carrier, 1e-12)
        # H/O 的入炉量由模型自己算（与 Cell 同口径）：从系统对象读煤与水
        cd = float(sn.get("c_in_kg", 0.0))
        # 用同一窗口的 M2 基线（f_slow=0）作分母，避免另需一套元素口径
        return {"out": out, "C": out["C"], "gas": atoms,
                "c_coal": c_coal, "c_unconv": c_unconv, "c_carrier": c_carrier,
                "T_out_K": float(ab["T_out_corrected_K"])}

    base = _closure(0.0, 0.0, 0.2)
    pert = _closure(1e-5, 50.0, 0.2)
    if not base or not pert:
        rows.append({"check": "U4", "item": "回加后 C/H/O 闭合（相对 M2 基线）",
                     "value": "n/a", "threshold": "≤10×基线", "pass": False,
                     "detail": "基线或扰动工况无解，无法评估"})
        hard.append("U4")
    else:
        r_base, r_pert = float(base["C"]), float(pert["C"])
        # H/O：用模型自身的元素流做完整核对需要另一套进料口径，
        # 这里以「回加前后 H、O 总量守恒」作为该变换的化学计量校验：
        # 变换消耗 1 H2O 与 1 CO2、生成 1 CO 与 1 H2（每 mol 碳），H 与 O 原子数不变。
        sn = solve_main(window_id=w4, coal_set=args.coal_set,
                        carrier_tag=MAIN_CARRIER, hl=MAIN_HL, pert="nom", phi=0.2)
        sl = slow_integrate(sn, dE_kjmol=50.0, f_slow=1e-5)
        ab = add_back(sn, sl)
        a1, a2 = float(sl["a_H2O_mol"]), float(sl["a_CO2_mol"])
        gm0 = np.asarray(sn["outlet_gas_moles"], dtype=float)
        gm1 = np.asarray(ab["gas_moles_after"], dtype=float)
        at0, at1 = atoms_from_gas(gm0), atoms_from_gas(gm1)
        dH = abs((at1.get("H", 0.0) - at0.get("H", 0.0)))
        dO = abs((at1.get("O", 0.0) - at0.get("O", 0.0)))
        # 期望值按**气相净原子数**算，不是按产物分子数：
        #   C+H2O → CO + H2：气相 C +1、H +2-2=0、O +1-1=0
        #   C+CO2 → 2 CO ：气相 C -1+2=+1、O +1-2=-1 → 需再扣 CO2 的 O
        # 每个反应对气相 C 都 +1，故 ΔC_gas = a1 + a2。
        # （此前写成 a1+2a2，把 2 CO 的两个碳都算成新增，漏了 CO2 自身带的那个碳。）
        exp_C = a1 + a2
        got_C = at1.get("C", 0.0) - at0.get("C", 0.0)
        stoich_ok = (abs(got_C - exp_C) <= 1e-9 * max(exp_C, 1e-12)
                     and dH <= 1e-9 * max(a1, 1e-12)
                     and dO <= 1e-9 * max(a1 + 2 * a2, 1e-12))
        ok4 = bool(r_pert <= 10.0 * max(r_base, 1e-12) and stoich_ok)
        rows.append({"check": "U4",
                     "item": "回加后 C 闭合 ≤10× M2 基线，且回加的 H/O 化学计量守恒",
                     "value": f"C={r_pert:.3e}", "threshold": f"≤10×{r_base:.3e}",
                     "pass": ok4,
                     "detail": (f"基线(f_slow=0) C 残差 {r_base:.3e}；扰动 C 残差 "
                                f"{r_pert:.3e}；回加 ΔC={got_C:.6f} mol vs 期望 "
                                f"{exp_C:.6f}；|ΔH_atom|={dH:.2e}，"
                                f"|ΔO_atom|={dO:.2e}（均应为 0）")})
        if not ok4:
            hard.append("U4")

    # U5：逐格碳量非负且不超初值
    # U5 用与 U4 相同的扰动工况（U4 内已算过 slow/pert，这里重取一次以保持独立可读）
    _sn5 = solve_main(window_id=ref["phase2"][0], coal_set=args.coal_set,
                      carrier_tag=MAIN_CARRIER, hl=MAIN_HL, pert="nom", phi=0.2)
    _sl5 = slow_integrate(_sn5, dE_kjmol=50.0, f_slow=1e-5) if _sn5.get("ok") else {}
    prof = _sl5.get("profile") or []
    c0 = float(_sl5.get("c0_kg", 0.0))
    bad_u5 = [r for r in prof
              if r["c_remaining_mol"] < -1e-12
              or r["c_remaining_mol"] * M_C_KG_PER_MOL > c0 * (1 + 1e-9)
              or r["dC_H2O_mol"] < -1e-12 or r["dC_CO2_mol"] < -1e-12]
    rows.append({"check": "U5", "item": "慢组分逐格碳量非负且不超初值",
                 "value": len(bad_u5), "threshold": 0,
                 "pass": not bad_u5,
                 "detail": f"格数 {len(prof)}；违规 {len(bad_u5)}"})
    if bad_u5:
        hard.append("U5")

    write_csv(pd.DataFrame(rows), "unit_checks.csv")
    print(f"[unit] {len(rows)} 项；hard_fail={hard}", flush=True)
    if hard:
        raise SystemExit(f"STOP[unit]: {hard}；见 unit_checks.csv")


# ---------------------------------------------------------------------------
# stage compare（12 配置 × 4 阶段求根）
# ---------------------------------------------------------------------------
def stage_compare(args) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p22a.prepare_coal_inputs()
    p22a.require_pairing(args.windows_csv, args.coal_set, "compare")
    p22a.init_ctx(Path(args.windows_csv))
    init_pool(args.windows_csv, args.n_workers)
    ref = p22a._ref_windows("main")
    tg = targets()

    # 先把 3 个 φ 的参照窗主流解出来（缓存），后续求根只做慢组分后处理
    for phi in PHI_GRID:
        specs = [{"window_id": w, "coal_set": args.coal_set,
                  "carrier_tag": MAIN_CARRIER, "hl": MAIN_HL,
                  "pert": "nom", "phi": phi}
                 for ph in PHASES for w in ref[ph]]
        out = batch_main(specs)
        bad = [s for s in out if not s.get("ok")]
        print(f"[compare] φ={phi} 参照窗主流 {len(out)} 窗，失败 {len(bad)}",
              flush=True)

    rows = []
    for phi in PHI_GRID:
        for dE in DE_GRID:
            for ph in PHASES:
                r = root_fslow(phi=phi, dE=dE, phase=ph, wids=ref[ph],
                               target=tg[ph], coal_set=args.coal_set)
                # 交付项 #2：比较点上的 T_out（修正前后）、ΔT_corr、慢组分初值/剩余、ξ
                extra: dict = {}
                if r["reachable"] and r["f_slow"]:
                    ms, _, _, det = median_xc(phi=phi, dE=dE,
                                              f_slow=float(r["f_slow"]),
                                              wids=ref[ph],
                                              coal_set=args.coal_set)
                    if det:
                        def med(k):
                            vs = [m.get(k) for m in det.values()
                                  if m.get(k) is not None]
                            return float(np.median(vs)) if vs else None
                        extra = {
                            "T_out_K": med("T_out_K"),
                            "T_out_corrected_K": med("T_out_corrected_K"),
                            "dT_corr_K": med("dT_corr_K"),
                            "capture_carbon_kg": med("capture_carbon_kg"),
                            "slow_converted_kg": med("slow_converted_kg"),
                            "slow_remaining_kg": med("slow_remaining_kg"),
                            "slow_converted_share_of_feed_C": med(
                                "slow_converted_share_of_feed_C"),
                            "xi_mol_s": med("xi_mol_s"),
                            "capture_fraction": med("capture_fraction"),
                        }
                rows.append({
                    "phi": phi, "dE_kJ_per_mol": dE, "phase": ph,
                    "target_X_C_pp": r["target_X_C_pp"],
                    "log10_f_slow": r["log10_f_slow"],
                    "f_slow": r["f_slow"],
                    "X_C_pp_median": r["X_C_pp_median"],
                    "z": r["z"], "reachable": r["reachable"],
                    "reason": r["reason"], "n_probe": r["n_probe"],
                    "n_solve_fail": r["n_solve_fail"],
                    "fslow_lo": args.fslow_lo, "fslow_hi": args.fslow_hi,
                    **extra,
                })
                print(f"[compare] φ={phi} ΔE={dE:5.0f} {ph}: "
                      f"log10f={r['log10_f_slow']} z={r['z']} "
                      f"reach={r['reachable']} {r['reason'][:40]}", flush=True)

    df = pd.DataFrame(rows)
    name = "comparison_points.csv" if not args.tag else f"comparison_points_{args.tag}.csv"
    write_csv(df, name)
    n = int(df["reachable"].astype(str).str.lower().eq("true").sum())
    print(f"[compare] 可达 {n}/{len(df)}（区间 [{args.fslow_lo}, {args.fslow_hi}]）",
          flush=True)


# ---------------------------------------------------------------------------
# 斜率与弹性（可达配置，142 窗 × 7 扰动）
# ---------------------------------------------------------------------------
def _cached_configs() -> pd.DataFrame:
    p = OUT_DIR / "comparison_points.csv"
    df = pd.read_csv(p, keep_default_na=False)
    return df[df["reachable"].astype(str).str.lower() == "true"]


def _fnum(v) -> float:
    return float("nan") if v == "" else float(v)


def stage_slopes(args) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p22a.prepare_coal_inputs()
    p22a.require_pairing(args.windows_csv, args.coal_set, "compare")
    p22a.init_ctx(Path(args.windows_csv))
    init_pool(args.windows_csv, args.n_workers)
    cp = _cached_configs()
    wids_all = sorted(str(w) for w in p22a.table("main").index)

    rows: list[dict] = []
    for (phi, dE), g in cp.groupby(["phi", "dE_kJ_per_mol"], sort=True):
        phi = float(phi); dE = float(dE)
        # 主口径（pert=nom）142 窗主流：一次批量
        specs = [{"window_id": w, "coal_set": args.coal_set,
                  "carrier_tag": MAIN_CARRIER, "hl": MAIN_HL,
                  "pert": "nom", "phi": phi} for w in wids_all]
        batch_main(specs)
        for ph in PHASES:
            sub = g[g["phase"] == ph]
            if not len(sub):
                continue
            f_slow = 10.0 ** _fnum(sub["log10_f_slow"].iloc[0])
            pw = [w for w in wids_all if p22a._phase_of(w) == ph]
            # 7 个扰动的主流（批量）
            sp = [{"window_id": w, "coal_set": args.coal_set,
                   "carrier_tag": MAIN_CARRIER, "hl": MAIN_HL,
                   "pert": pert, "phi": phi}
                  for pert in PERTS for w in pw]
            batch_main(sp)
            specs_m = [{"window_id": w, "phi": phi, "dE": dE,
                        "f_slow": f_slow, "coal_set": args.coal_set,
                        "pert": pert}
                       for pert in PERTS for w in pw]
            mout = batch_metrics(specs_m)
            nw = len(pw)
            per: dict[str, dict[str, dict[str, float]]] = {}
            n_fail = 0
            n_xi_fail = 0
            for j, pert in enumerate(PERTS):
                acc: dict[str, dict[str, float]] = {
                    k: {} for k in ("X_C_pp", "CO_vol_pct", "H2_vol_pct",
                                    "CO2_vol_pct", "dry_per_coal_mol_kg")}
                for m in mout[j * nw:(j + 1) * nw]:
                    if not m.get("ok"):
                        n_fail += 1
                        continue
                    wid = str(m["window_id"])
                    # X_C 不依赖 ξ：即使 ξ 无法闭合也要计入，否则 X_C 斜率有偏
                    acc["X_C_pp"][wid] = float(m["X_C_pp"])
                    if not m.get("xi_ok"):
                        n_xi_fail += 1
                        continue
                    for k in ("CO_vol_pct", "H2_vol_pct", "CO2_vol_pct",
                              "dry_per_coal_mol_kg"):
                        v = m.get(k)
                        if v is not None:
                            acc[k][wid] = float(v)
                per[pert] = acc
            d_ln = p20.LN_P - p20.LN_M
            inputs = {"ln_O2": ("o2_p", "o2_m"), "ln_coal": ("coal_p", "coal_m"),
                      "ln_carrier": ("car_p", "car_m")}
            for k in ("X_C_pp", "CO_vol_pct", "H2_vol_pct", "CO2_vol_pct",
                      "dry_per_coal_mol_kg"):
                for inp, (pp, pm) in inputs.items():
                    yp, ym = per[pp][k], per[pm][k]
                    # 2026-11-20（审阅意见 P2）：必须按 **window_id 取交集**。
                    # 此前两个扰动各自独立压实后用索引配对，ξ 在两侧闭合的窗口不同，
                    # yp[i] 与 ym[i] 会指向不同窗口，等于把不相关的工况相减却当成
                    # 有限差分灵敏度。阶段 1/3 大量 ξ 失败时这会直接污染 J2/J3。
                    common = sorted(set(yp) & set(ym))
                    n = len(common)
                    if n == 0:
                        rows.append({"phi": phi, "dE_kJ_per_mol": dE,
                                     "phase": ph, "obs": k, "input": inp,
                                     "per3_median": None, "n": 0,
                                     "n_fail": n_fail, "n_xi_fail": n_xi_fail,
                                     "note": "两侧扰动无共同可解窗"})
                        continue
                    d2 = [(yp[w] - ym[w]) / d_ln * p20.LN_P for w in common]
                    rows.append({"phi": phi, "dE_kJ_per_mol": dE, "phase": ph,
                                 "obs": k, "input": inp,
                                 "per3_median": float(np.median(d2)),
                                 "n": n, "n_fail": n_fail,
                                 "n_xi_fail": n_xi_fail, "note": ""})
            print(f"[slopes] φ={phi} ΔE={dE:.0f} {ph} 完成"
                  f"（求解失败={n_fail}, ξ 未闭合={n_xi_fail}）", flush=True)
    write_csv(pd.DataFrame(rows), "model_slopes.csv")


def stage_elasticities(args) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sl = pd.read_csv(OUT_DIR / "model_slopes.csv", keep_default_na=False)
    cp = pd.read_csv(OUT_DIR / "comparison_points.csv", keep_default_na=False)
    rows = []
    for (phi, dE), g in sl.groupby(["phi", "dE_kJ_per_mol"], sort=True):
        for ph in PHASES:
            sub = g[g["phase"] == ph]
            if not len(sub):
                continue

            def get(obs, inp):
                h = sub[(sub["obs"] == obs) & (sub["input"] == inp)]
                if not len(h) or h["per3_median"].iloc[0] == "":
                    return None
                return float(h["per3_median"].iloc[0])

            h2 = cp[(cp["phi"] == phi) & (cp["dE_kJ_per_mol"] == dE)
                    & (cp["phase"] == ph)]
            xc = None
            if len(h2) and h2["X_C_pp_median"].iloc[0] != "":
                xc = float(h2["X_C_pp_median"].iloc[0])
            p_c, p_o = get("X_C_pp", "ln_coal"), get("X_C_pp", "ln_O2")
            e_c = e_o = None
            if xc and p_c is not None:
                e_c = 1.0 + (p_c / xc) / p20.LN_P
            if xc and p_o is not None:
                e_o = (p_o / xc) / p20.LN_P
            iv = PLANT_E_INTERVALS[ph]
            rows.append({"phi": phi, "dE_kJ_per_mol": dE, "phase": ph,
                         "X_C_pp_level": xc,
                         "e_coal_model": e_c, "e_o2_model": e_o,
                         "e_coal_plant_lo": iv["coal"][0],
                         "e_coal_plant_hi": iv["coal"][1],
                         "e_o2_plant_lo": iv["o2"][0],
                         "e_o2_plant_hi": iv["o2"][1],
                         "e_coal_in_interval": (None if e_c is None else
                                                bool(iv["coal"][0] <= e_c <= iv["coal"][1])),
                         "e_o2_in_interval": (None if e_o is None else
                                              bool(iv["o2"][0] <= e_o <= iv["o2"][1]))})
    write_csv(pd.DataFrame(rows), "elasticities.csv")
    print("[elasticities] 完成", flush=True)


def stage_levels(args) -> None:
    """ξ 闭合后 142 窗 CO/H2 的阶段中位偏差（主口径）。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p22a.prepare_coal_inputs()
    p22a.require_pairing(args.windows_csv, args.coal_set, "compare")
    p22a.init_ctx(Path(args.windows_csv))
    init_pool(args.windows_csv, args.n_workers)
    cp = _cached_configs()
    wids_all = sorted(str(w) for w in p22a.table("main").index)
    dfm = p22a.table("main")
    rows = []
    for (phi, dE), g in cp.groupby(["phi", "dE_kJ_per_mol"], sort=True):
        phi = float(phi); dE = float(dE)
        for ph in PHASES:
            sub = g[g["phase"] == ph]
            if not len(sub):
                continue
            f_slow = 10.0 ** _fnum(sub["log10_f_slow"].iloc[0])
            pw = [w for w in wids_all if p22a._phase_of(w) == ph]
            batch_main([{"window_id": w, "coal_set": args.coal_set,
                         "carrier_tag": MAIN_CARRIER, "hl": MAIN_HL,
                         "pert": "nom", "phi": phi} for w in pw])
            for comp, col in (("CO_vol_pct", "co_vol_pct_mean"),
                              ("H2_vol_pct", "h2_vol_pct_mean")):
                mv, pv = [], []
                lout = batch_metrics([{"window_id": w, "phi": phi, "dE": dE,
                                       "f_slow": f_slow,
                                       "coal_set": args.coal_set}
                                      for w in pw])
                for m in lout:
                    if m.get("ok") and m.get("xi_ok") and m.get(comp) is not None:
                        mv.append(float(m[comp]))
                for w in pw:
                    pv.append(float(dfm.loc[w, col]))
                mm = float(np.median(mv)) if mv else None
                pp = float(np.median(pv)) if pv else None
                rows.append({"phi": phi, "dE_kJ_per_mol": dE, "phase": ph,
                             "observable": comp, "model_pp": mm,
                             "plant_pp": pp,
                             "dev_pp": (None if (mm is None or pp is None)
                                        else mm - pp)})
            print(f"[levels] φ={phi} ΔE={dE:.0f} {ph} 完成", flush=True)
    write_csv(pd.DataFrame(rows), "levels.csv")


# ---------------------------------------------------------------------------
# 慢组分剖面（必须回答 #4）
# ---------------------------------------------------------------------------
def stage_slow_profiles(args) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p22a.prepare_coal_inputs()
    p22a.require_pairing(args.windows_csv, args.coal_set, "compare")
    p22a.init_ctx(Path(args.windows_csv))
    ref = p22a._ref_windows("main")
    cp = _cached_configs()
    # 无入选时按 J 通过项最多的两个（在 read 阶段写盘后由文件决定）；此处全量导出
    rows = []
    for (phi, dE), g in cp.groupby(["phi", "dE_kJ_per_mol"], sort=True):
        phi = float(phi); dE = float(dE)
        for ph in PHASES:
            sub = g[g["phase"] == ph]
            if not len(sub):
                continue
            f_slow = 10.0 ** _fnum(sub["log10_f_slow"].iloc[0])
            for w in ref[ph]:
                snap = solve_main(window_id=w, coal_set=args.coal_set,
                                  carrier_tag=MAIN_CARRIER, hl=MAIN_HL,
                                  pert="nom", phi=phi)
                if not snap.get("ok"):
                    continue
                s = slow_integrate(snap, dE_kjmol=dE, f_slow=f_slow)
                if not s.get("ok"):
                    continue
                for r in s["profile"]:
                    rows.append({"phi": phi, "dE_kJ_per_mol": dE, "phase": ph,
                                 "window_id": w, "log10_f_slow":
                                 math.log10(f_slow), **r})
            print(f"[profiles] φ={phi} ΔE={dE:.0f} {ph} 完成", flush=True)
    write_csv(pd.DataFrame(rows), "slow_pool_profiles.csv")


# ---------------------------------------------------------------------------
# stage read（J0–J8）
# ---------------------------------------------------------------------------
def _rd(reading, metric, phase, cfg, value, threshold, passed, note):
    return {"reading": reading, "metric": metric, "phase": phase,
            "config": cfg, "value": value, "threshold": threshold,
            "pass": passed, "note": note}


def stage_read(args) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cp = pd.read_csv(OUT_DIR / "comparison_points.csv", keep_default_na=False)
    el_p = OUT_DIR / "elasticities.csv"
    sl_p = OUT_DIR / "model_slopes.csv"
    lv_p = OUT_DIR / "levels.csv"
    el = pd.read_csv(el_p, keep_default_na=False) if el_p.exists() else pd.DataFrame()
    sl = pd.read_csv(sl_p, keep_default_na=False) if sl_p.exists() else pd.DataFrame()
    lv = pd.read_csv(lv_p, keep_default_na=False) if lv_p.exists() else pd.DataFrame()

    cfgs = sorted({(float(r["phi"]), float(r["dE_kJ_per_mol"]))
                   for _, r in cp.iterrows()})
    def key(phi, dE):
        return f"phi{phi:g}_dE{dE:g}"

    rows: list[dict] = []

    # J0 可达：四阶段 |z| <= 0.5
    j0 = {}
    for phi, dE in cfgs:
        sub = cp[(cp["phi"] == phi) & (cp["dE_kJ_per_mol"] == dE)]
        n = int(sub["reachable"].astype(str).str.lower().eq("true").sum())
        j0[(phi, dE)] = n == 4
        miss = "；".join(f"{PHASE_LABEL[r['phase']]}: {r['reason']}"
                        for _, r in sub.iterrows()
                        if str(r["reachable"]).lower() != "true" and r["reason"])
        rows.append(_rd("J0", "n_reachable_phases", "all", key(phi, dE), n, 4,
                        n == 4, "四阶段全可达" if n == 4 else f"未达：{miss}"))

    # J1 e_氧 落在装置区间的阶段数 >= 3
    j1 = {}
    if len(el):
        for phi, dE in cfgs:
            sub = el[(el["phi"] == phi) & (el["dE_kJ_per_mol"] == dE)]
            n = int(sum(1 for _, r in sub.iterrows()
                        if str(r["e_o2_in_interval"]).lower() == "true"))
            j1[(phi, dE)] = n >= J1_MIN_PHASES
            det = "；".join(
                f"{PHASE_LABEL[r['phase']]}={r['e_o2_model']}"
                for _, r in sub.iterrows())
            rows.append(_rd("J1", "n_phases_e_o2_in_interval", "all",
                            key(phi, dE), n, J1_MIN_PHASES, n >= J1_MIN_PHASES, det))

    # J2 干气对氧量在阶段 1/2/4 为正
    j2 = {}
    if len(sl):
        for phi, dE in cfgs:
            n, det = 0, []
            for ph in H2_PHASES:
                h = sl[(sl["phi"] == phi) & (sl["dE_kJ_per_mol"] == dE)
                       & (sl["phase"] == ph) & (sl["obs"] == "dry_per_coal_mol_kg")
                       & (sl["input"] == "ln_O2")]
                if len(h) and h["per3_median"].iloc[0] != "":
                    v = float(h["per3_median"].iloc[0])
                    if v > 0:
                        n += 1
                    det.append(f"{PHASE_LABEL[ph]}={v:+.3f}")
                else:
                    det.append(f"{PHASE_LABEL[ph]}=缺")
            j2[(phi, dE)] = n == 3
            rows.append(_rd("J2", "n_phases_dry_per_coal_o2_positive", "all",
                            key(phi, dE), n, 3, n == 3, "；".join(det)))

    # J3 组成方向 8 组合同号 >= 6
    j3 = {}
    if len(sl):
        plant = pd.read_csv(P115A_DIR / "plant_slopes.csv")
        for phi, dE in cfgs:
            n, det = 0, []
            for ph, obs, inp in p20.G2_COMBOS:
                h = sl[(sl["phi"] == phi) & (sl["dE_kJ_per_mol"] == dE)
                       & (sl["phase"] == ph) & (sl["obs"] == obs)
                       & (sl["input"] == inp)]
                p = plant[(plant["phase"] == ph) & (plant["obs"] == obs)
                          & (plant["input"] == inp) & (plant["model"] == "with_carrier")]
                if not len(h) or not len(p) or h["per3_median"].iloc[0] == "":
                    det.append(f"{PHASE_LABEL[ph]} {obs}←{inp}: 缺")
                    continue
                m = float(h["per3_median"].iloc[0]); pv = float(p["per3"].iloc[0])
                same = (m > 0) == (pv > 0)
                n += int(same)
                det.append(f"{PHASE_LABEL[ph]} {obs}←{inp} {m:+.2f}/{pv:+.2f} "
                           f"{'同' if same else '反'}")
            j3[(phi, dE)] = n >= J3_MIN_SIGNED
            rows.append(_rd("J3", "n_same_sign_of_8", "all", key(phi, dE), n,
                            J3_MIN_SIGNED, n >= J3_MIN_SIGNED, "；".join(det)))

    # J4 出口温度（回加修正后）：4 个 >= FT-50K 且 >=3 个 >= FT
    j4 = {}
    for phi, dE in cfgs:
        sub = cp[(cp["phi"] == phi) & (cp["dE_kJ_per_mol"] == dE)]
        at_ft, margin, det = 0, 0, []
        for _, r in sub.iterrows():
            if str(r["reachable"]).lower() != "true":
                det.append(f"{PHASE_LABEL[r['phase']]}: 不可达")
                continue
            ph = r["phase"]
            t = float(r["T_out_corrected_K"]) if "T_out_corrected_K" in r else None
            if t is None:
                det.append(f"{PHASE_LABEL[ph]}: 无温度")
                continue
            tc = t - 273.15
            if tc >= LIU_FT_C[ph]:
                at_ft += 1
            if tc >= LIU_FT_C[ph] - J4_FT_MARGIN_K:
                margin += 1
            det.append(f"{PHASE_LABEL[ph]}={tc:.0f}°C(FT {LIU_FT_C[ph]:.0f})")
        ok = margin == 4 and at_ft >= J4_MIN_AT_FT
        j4[(phi, dE)] = ok
        rows.append(_rd("J4", "T_out_corrected_vs_FT", "all", key(phi, dE),
                        f"at_FT={at_ft},margin={margin}",
                        f">={J4_MIN_AT_FT} at FT 且 4 个 >= FT-50K", ok,
                        "；".join(det)))

    # J5 组成水平：CO/H2 阶段中位偏差 <= 2 pp
    j5 = {}
    if len(lv):
        for phi, dE in cfgs:
            sub = lv[(lv["phi"] == phi) & (lv["dE_kJ_per_mol"] == dE)]
            worst, det = 0.0, []
            ok = True
            for _, r in sub.iterrows():
                if r["dev_pp"] == "":
                    ok = False
                    det.append(f"{PHASE_LABEL[r['phase']]} {r['observable']}: 缺")
                    continue
                d = abs(float(r["dev_pp"]))
                worst = max(worst, d)
                if d > J5_MAX_DEV_PP:
                    ok = False
                det.append(f"{PHASE_LABEL[r['phase']]} {r['observable'][:2]}"
                           f"={float(r['dev_pp']):+.2f}")
            j5[(phi, dE)] = ok
            rows.append(_rd("J5", "max_abs_CO_H2_dev_pp", "all", key(phi, dE),
                            round(worst, 3), J5_MAX_DEV_PP, ok, "；".join(det)))

    # J6 原型可信度：|ΔT_corr| <= 50 K 且慢组分转化占入炉碳 <= 5%
    j6 = {}
    for phi, dE in cfgs:
        sub = cp[(cp["phi"] == phi) & (cp["dE_kJ_per_mol"] == dE)]
        ok_rows = sub[sub["reachable"].astype(str).str.lower() == "true"]
        if not len(ok_rows):
            j6[(phi, dE)] = False
            rows.append(_rd("J6", "prototype_credibility", "all", key(phi, dE),
                            "n/a", "|ΔT|<=50K 且 慢组分占比<=5%", False,
                            "无可达阶段，无法评估"))
            continue
        dt = max(abs(_fnum(r["dT_corr_K"])) for _, r in ok_rows.iterrows())
        share = max(_fnum(r["slow_converted_share_of_feed_C"])
                    for _, r in ok_rows.iterrows())
        ok = dt <= J6_MAX_DT_CORR_K and share <= J6_MAX_SLOW_C_SHARE
        j6[(phi, dE)] = ok
        rows.append(_rd("J6", "prototype_credibility", "all", key(phi, dE),
                        f"|dT|={dt:.1f}K,share={share*100:.2f}%",
                        "|ΔT|<=50K 且 占比<=5%", ok,
                        f"最大 |ΔT_corr|={dt:.1f} K；慢组分转化占入炉碳 "
                        f"{share*100:.2f}%"))

    # J7/J8 只报告
    if len(el):
        for phi, dE in cfgs:
            sub = el[(el["phi"] == phi) & (el["dE_kJ_per_mol"] == dE)]
            n = int(sum(1 for _, r in sub.iterrows()
                        if str(r["e_coal_in_interval"]).lower() == "true"))
            rows.append(_rd("J7", "n_phases_e_coal_in_interval", "all",
                            key(phi, dE), n, None, None, "只报告"))
    for phi, dE in cfgs:
        sub = cp[(cp["phi"] == phi) & (cp["dE_kJ_per_mol"] == dE)]
        trend = "；".join(f"{PHASE_LABEL[r['phase']]} log10f={r['log10_f_slow']}"
                         for _, r in sub.iterrows())
        rows.append(_rd("J8", "log10_f_slow_by_phase", "all", key(phi, dE),
                        "", None, None, trend))

    # 判定
    sel = []
    for phi, dE in cfgs:
        checks = {"J0": j0.get((phi, dE)), "J1": j1.get((phi, dE)),
                  "J2": j2.get((phi, dE)), "J3": j3.get((phi, dE)),
                  "J4": j4.get((phi, dE)), "J5": j5.get((phi, dE)),
                  "J6": j6.get((phi, dE))}
        passed = all(bool(v) for v in checks.values())
        fails = [k for k, v in checks.items() if not v]
        rows.append(_rd("VERDICT", "config_verdict", "all", key(phi, dE),
                        "SELECTED" if passed else "NOT_SELECTED",
                        "J0–J6 全通过", passed,
                        "全部通过" if passed else f"未通过 {','.join(fails)}"))
        if passed:
            sel.append(key(phi, dE))
    rows.append(_rd("SELECTED", "selected_configs", "all", "all",
                    ";".join(sel) if sel else "无", "J0–J6",
                    bool(sel), ("入选 " + ";".join(sel)) if sel
                    else "无配置入选"))
    write_csv(pd.DataFrame(rows), "readings.csv")
    print(f"[read] 入选 {sel if sel else '无'}", flush=True)


# ---------------------------------------------------------------------------
# protocol / manifest / main
# ---------------------------------------------------------------------------
def protocol(args) -> dict:
    return {
        "round": "P2.4-A-20261119",
        "card": "manuscript/model-side-handoff-p2.4a-20261119.md",
        "coal_set": args.coal_set,
        "windows_csv": _rel(Path(args.windows_csv)),
        "windows_sha256": p22a.table_sha("main"),
        "frozen": {"kappa": p20.KAPPA, "mesh": "wsr_plus_pfr",
                   "L_wsr_m": p20.L_WSR, "dz_down_m": p20.DZ_DOWN,
                   "HeatLossModel": "total_percent",
                   "HeatLossPercentTotal": MAIN_HL,
                   "carrier": MAIN_CARRIER, "kinetics": "Wen 1979"},
        "config_grid": {"phi": list(PHI_GRID), "dE_kJ_per_mol": list(DE_GRID)},
        "fslow_range": {"log10_lo": float(args.fslow_lo),
                        "log10_hi": float(args.fslow_hi),
                        "card_range": [-4.0, 2.0],
                        "deviation_from_card": bool(args.fslow_lo != -4.0
                                                    or args.fslow_hi != 2.0)},
        "reachability": {"criterion": "z = (X_C - target)/sigma_target",
                         "sigma_target_pp": SIGMA_TARGET, "z_tol": Z_TOL},
        "targets": targets(),
        "slow_component": {
            "rate_factor": "f_slow * exp(-(dE/R)(1/T_p - 1/T_ref)), T_ref=1600 K",
            "reactions": ["C+H2O", "C+CO2"],
            "excluded": ["C+O2 (no O2 downstream)", "C+H2 (upper bound recorded)"],
            "area_convention": "S = 6*m*tau/(rho_p*d_p)，d_p/rho_p 取全混区出口值",
            "residence": "与主流 m_hold 口径一致（tau = dz/v_g）",
            "heat_sink": "reaction_heat_release_w（模型自身 Shomate 焓）；"
                         "dT_corr = q/cp_gas，cp_gas 由 get_gas_enthalpy 数值求导",
        },
        "thresholds": {"J1_min_phases": J1_MIN_PHASES,
                       "J3_min_signed": J3_MIN_SIGNED,
                       "J4_ft_margin_K": J4_FT_MARGIN_K,
                       "J4_min_at_ft": J4_MIN_AT_FT,
                       "J5_max_dev_pp": J5_MAX_DEV_PP,
                       "J6_max_dt_corr_K": J6_MAX_DT_CORR_K,
                       "J6_max_slow_c_share": J6_MAX_SLOW_C_SHARE},
        "n_workers": args.n_workers,
    }


def write_manifest() -> None:
    products = ["unit_checks.csv", "comparison_points.csv", "model_slopes.csv",
                "elasticities.csv", "levels.csv", "slow_pool_profiles.csv",
                "readings.csv", "protocol.json", "interpretation.md"]
    inputs = [p22a.P22D_CSV if hasattr(p22a, "P22D_CSV") else
              ROOT / "outputs" / "diagnostics" / "p2.2d-20261118" / "windows_142.csv",
              P22D_READINGS, P115A_DIR / "plant_slopes.csv",
              ROOT / "src" / "model" / "cell.py",
              ROOT / "src" / "model" / "gasifier_system.py",
              ROOT / "src" / "model" / "kinetics_service.py",
              ROOT / "src" / "model" / "heat_loss.py"]
    lines = ["# P2.4-A 产物与输入清单（sha256，相对仓库根）", "", "# products"]
    for name in products:
        p = OUT_DIR / name
        lines.append(f"{_sha256(p)}  {_rel(p)}" if p.exists()
                     else f"{'0'*64}  {_rel(p)}  (missing)")
    lines += ["", "# inputs"]
    for p in inputs:
        p = Path(p)
        if p.exists():
            lines.append(f"{_sha256(p)}  {_rel(p)}")
    (OUT_DIR / "sha256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coal-set", required=True,
                    choices=("current", "legacy_20261118", "lims_heat"))
    ap.add_argument("--windows-csv", required=True)
    ap.add_argument("--stage", default="all",
                    choices=("unit", "compare", "slopes", "elasticities",
                             "levels", "profiles", "read", "all"))
    ap.add_argument("--fslow-lo", type=float, default=-4.0,
                    help="log10 f_slow 下界（卡面 −4；扩展探索时改小）")
    ap.add_argument("--fslow-hi", type=float, default=2.0)
    ap.add_argument("--tag", default="", help="输出文件名后缀（区分正式/探索）")
    ap.add_argument("--n-workers", type=int, default=13)
    args = ap.parse_args()
    silence_solver_logs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    global FSLOW_LO, FSLOW_HI
    FSLOW_LO, FSLOW_HI = float(args.fslow_lo), float(args.fslow_hi)

    try:
        if args.stage == "unit":
            stage_unit(args)
        elif args.stage == "compare":
            stage_compare(args)
        elif args.stage == "slopes":
            stage_slopes(args)
        elif args.stage == "elasticities":
            stage_elasticities(args)
        elif args.stage == "levels":
            stage_levels(args)
        elif args.stage == "profiles":
            stage_slow_profiles(args)
        elif args.stage == "read":
            stage_read(args)
        else:
            stage_unit(args)
            stage_compare(args)
            stage_slopes(args)
            stage_elasticities(args)
            stage_levels(args)
            stage_slow_profiles(args)
            stage_read(args)
            (OUT_DIR / "protocol.json").write_text(
                json.dumps(protocol(args), ensure_ascii=False, indent=2,
                           default=str) + "\n", encoding="utf-8")
            write_manifest()
            print("[done] 产物与清单已写出", flush=True)
    finally:
        close_pool()


if __name__ == "__main__":
    main()
