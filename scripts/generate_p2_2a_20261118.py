#!/usr/bin/env python3
"""P2.2-A（第 2 版）：现有模型在测量不确定度尺度上的重新基线（2026-11-18 卡）。

冻结项与 P2.0-A 的 C0 相同：o2_flow_scale = 1、GasificationRateFactor = 1、
CharCO2RateFactor = 1、两个 P2.0-A 钩子关闭、MeshFamily = wsr_plus_pfr、
WSRLength_m = 1、下游步长 0.0515 m、HeatLossModel = total_percent、实测载气、
逐窗协调干煤量。**本轮不反演、不加钩子、不加回流、不改合同。**

第 0 节的配对是硬约束（煤质集合与窗口表必须同属一行，由 sha256 判定）：

    current / lims_heat  ↔  outputs/diagnostics/p2.2d-20261118/windows_142.csv
    legacy_20261118      ↔  outputs/diagnostics/p1.14d-20261107/windows_142.csv

用法::

    ./.venv/bin/python scripts/generate_p2_2a_20261118.py \
        --coal-set current \
        --windows-csv outputs/diagnostics/p2.2d-20261118/windows_142.csv \
        --stage unit|compare|sens|profiles|read|all

    # U1 必须用 legacy 煤质 + 旧窗口表另跑一次 unit
    ./.venv/bin/python scripts/generate_p2_2a_20261118.py \
        --coal-set legacy_20261118 \
        --windows-csv outputs/diagnostics/p1.14d-20261107/windows_142.csv \
        --stage unit
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import generate_p1_8a_20261104 as p18a  # noqa: E402
import generate_p1_15a_20261108 as p115a  # noqa: E402
import generate_p2_0a_20261113 as p20  # noqa: E402

from model.p1_5_round_20261030 import run_case, silence_solver_logs  # noqa: E402
from model.revision_evidence_common import (  # noqa: E402
    DATA_PREP_ROOT,
    coal_assumption_paths,
    load_frozen_split,
    model_case_rows,
    source_commit as _git_source_commit,
)

# ---------------------------------------------------------------------------
# 路径与常量
# ---------------------------------------------------------------------------

OUT_DIR = ROOT / "outputs" / "diagnostics" / "round-p2.2a-20261118"
CACHE_DIR = OUT_DIR / "cache"
COAL_INPUT_DIR = OUT_DIR / "coal_inputs"
LIMS_HEAT_DIR = OUT_DIR / "coal_lims_heat"

P22D_CSV = ROOT / "outputs" / "diagnostics" / "p2.2d-20261118" / "windows_142.csv"
P114D_CSV = ROOT / "outputs" / "diagnostics" / "p1.14d-20261107" / "windows_142.csv"
P20A_DIR = ROOT / "outputs" / "diagnostics" / "round-p2.0a-20261113"
P115A_DIR = ROOT / "outputs" / "diagnostics" / "round-p1.15a-20261108"

PAPER_ROOT = Path(os.environ.get("PAPER_ROOT", "/Users/liuzhen/AI-projects/ai-research"))
PAPER_P22D = (PAPER_ROOT / "papers" / "jax-gasifier-cascade-hybrid"
              / "revision-evidence-20261118" / "P2.2-D")
P22D_READINGS = PAPER_P22D / "readings.csv"

# 第 0 节：窗口表指纹（P2.2-D 已合并版本 / P1.14-D 已合并版本）
WINDOWS_SHA = {
    "p2.2d": "a24b5fe636752d8b7872ebacd569e40325244f90b851ae4ff1adaeee6e989119",
    "p1.14d": "25f3a7b4d403adc83ab888c893bc23e555af2121b1d4fa04cb53c93d57e0f991",
}

PAIRING = {
    "current": "p2.2d",
    "lims_heat": "p2.2d",
    "legacy_20261118": "p1.14d",
}

PHASES = ("phase1", "phase2", "phase3", "phase4")
PHASE_LABEL = {"phase1": "P1", "phase2": "P2", "phase3": "P3", "phase4": "P4"}

COAL_KEY_BY_PHASE = {
    "phase1": "v0_baseline_coal_assumption",
    "phase2": "huayi_p2_sy3sy2_coal_assumption",
    "phase3": "huayi_p3_sy3sy2_coal_assumption",
    "phase4": "huayi_p4_sy3sy2_coal_assumption",
}

# 第 0 节第 2 条 / 第 2 节第 2 条：四阶段热值（kJ/kg，干燥基低位）
HV_EXPECT = {
    "current": {"phase1": 28179.0, "phase2": 28440.0, "phase3": 28579.0, "phase4": 28588.0},
    "legacy_20261118": {"phase1": 27720.0, "phase2": 28094.0, "phase3": 27976.0,
                        "phase4": 28103.0},
    # 化验表值按配比：P2 取 legacy p3（70:30）文件、P3 取 legacy p2（50:50）文件
    "lims_heat": {"phase1": 27720.0, "phase2": 27976.0, "phase3": 28094.0, "phase4": 28103.0},
}

HL_GRID = (0.5, 1.0, 2.0)
CARRIERS = {"C-m": "co2_carrier_measured_nm3_h", "C-J": "co2_carrier_j_nm3_h"}
MAIN_HL = 1.0
MAIN_CARRIER = "C-m"

# 读法阈值（卡面第 3 节，跑完不改）
B1_Z_MAX = 1.5
B1_P1_MIN = 97.0
B2_MARGIN_K = 50.0
B3_MIN_SIGNED = 6
B6_CEIL_PP = 99.9
B6_TARGET_PP = 99.0

LIU_FT_C = {"phase1": 1280.0, "phase2": 1290.0, "phase3": 1270.0, "phase4": 1250.0}
LIU_T25_C = {"phase1": 1346.4, "phase2": 1327.5, "phase3": 1282.4, "phase4": 1277.0}

PERT_SCALE = p20.PERT_SCALE
PERTS = p20.PERTS
OBS_ALL = p20.OBS_ALL
INPUTS = p20.INPUTS
LN_P = p20.LN_P
CAND = "C0"

_CTX: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def _f(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rel(path: Path) -> str:
    """相对仓库根路径；不得出现绝对路径。"""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return f"<outside-repo>/{p.name}"


def _write_csv(df: pd.DataFrame, name: str) -> Path:
    out = OUT_DIR / name
    df.to_csv(out, index=False, lineterminator="\n")
    return out


def _phase_of(wid: str) -> str:
    return p20._phase_of_row(_CTX["main"].loc[wid, "phase"])


def _phase_windows(tbl: str = "main") -> dict[str, list[str]]:
    df = table(tbl)
    out: dict[str, list[str]] = {ph: [] for ph in PHASES}
    for wid in sorted(str(w) for w in df.index):
        out[p20._phase_of_row(df.loc[wid, "phase"])].append(wid)
    return out


def _ref_windows(tbl: str = "main") -> dict[str, list[str]]:
    df = table(tbl)
    out: dict[str, list[str]] = {ph: [] for ph in PHASES}
    for wid in sorted(p20.ref_window_ids()):
        out[p20._phase_of_row(df.loc[wid, "phase"])].append(wid)
    return out


# ---------------------------------------------------------------------------
# 煤质集合
# ---------------------------------------------------------------------------

def prepare_coal_inputs() -> None:
    """把四份 current、四份 legacy 复制进本轮目录，并按配比生成 lims_heat 四份。

    lims_heat 以 current 的煤质为底，**只**把 heating_value_dry_kj_kg.value 换成化验表值；
    其余各项仍取 current。复制而不是引用，是为了让 sha256.txt 里的路径全部落在仓库内、
    且本轮目录自带可复现的输入快照。
    """
    COAL_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    LIMS_HEAT_DIR.mkdir(parents=True, exist_ok=True)
    for coal_set in ("current", "legacy_20261118"):
        dest = COAL_INPUT_DIR / coal_set
        dest.mkdir(parents=True, exist_ok=True)
        for key, src in coal_assumption_paths(coal_set).items():
            dst = dest / src.name
            shutil.copyfile(src, dst)
            if _sha256(src) != _sha256(dst):
                raise RuntimeError(f"coal assumption copy mismatch: {src}")
    for phase, hv in HV_EXPECT["lims_heat"].items():
        key = COAL_KEY_BY_PHASE[phase]
        base = COAL_INPUT_DIR / "current" / coal_assumption_paths("current")[key].name
        data = yaml.safe_load(base.read_text(encoding="utf-8")) or {}
        data["heating_value_dry_kj_kg"] = {
            "value": float(hv),
            "unit": "kJ/kg",
            "basis": "adb",
            "label_zh": "干燥基低位发热量（化验表值，按配比）",
            "source": {
                "type": "lims_table_legacy_heat_only",
                "note_zh": ("P2.2-A 敏感性第 2 节用：只把热值换成化验表值（按配比），"
                            "其余各项仍取当前版 current 煤质"),
                "phase": PHASE_LABEL[phase],
            },
            "model_input_role": "primary",
        }
        data["p22a_provenance"] = {
            "round": "P2.2-A-20261118",
            "derivation": "current 煤质 + 化验表热值（按配比）；其余字段取自 current",
            "heating_value_source": "legacy_20261118 快照中的化验表值",
            "phase": PHASE_LABEL[phase],
        }
        (LIMS_HEAT_DIR / base.name).write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def coal_paths(coal_set: str) -> dict[str, Path]:
    if coal_set == "lims_heat":
        return {key: LIMS_HEAT_DIR / coal_assumption_paths("current")[key].name
                for key in COAL_KEY_BY_PHASE.values()}
    if coal_set not in ("current", "legacy_20261118"):
        raise KeyError(f"unknown coal_set {coal_set!r}")
    return {key: COAL_INPUT_DIR / coal_set / coal_assumption_paths(coal_set)[key].name
            for key in COAL_KEY_BY_PHASE.values()}


def coal_digest(coal_set: str) -> str:
    """所选煤质集合四份 YAML 内容的 sha256 摘要（进缓存键）。"""
    paths = coal_paths(coal_set)
    h = hashlib.sha256()
    for key in COAL_KEY_BY_PHASE.values():
        h.update(paths[key].read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def coal_hv(coal_set: str) -> dict[str, float]:
    paths = coal_paths(coal_set)
    return {
        phase: float((yaml.safe_load(paths[key].read_text(encoding="utf-8"))
                      or {})["heating_value_dry_kj_kg"]["value"])
        for phase, key in COAL_KEY_BY_PHASE.items()
    }


# ---------------------------------------------------------------------------
# 上下文：窗口表与 model case 行
# ---------------------------------------------------------------------------

def init_ctx(windows_csv: Path) -> None:
    if _CTX.get("windows_csv") == str(windows_csv):
        return
    silence_solver_logs()
    main = pd.read_csv(windows_csv).set_index("window_id")
    saved = {k: _CTX[k] for k in ("rows",) if k in _CTX}
    _CTX.clear()
    _CTX.update({
        "windows_csv": str(windows_csv),
        "windows_sha": _sha256(Path(windows_csv)),
        "main": main,
        "tables": {"main": main},
        "rows": saved.get("rows", {}),
    })


def require_pairing(windows_csv, coal_set: str, stage_name: str) -> None:
    """第 0 节配对守卫：任何阶段都不得把煤质集合与不匹配的窗口表混用。

    `unit` 阶段会把配对结果写成读数（失败即停），但其余阶段若不设防就会静默混用：
    煤质来自一套集合、协调煤量来自另一套煤质的窗口表，结果既不能用也无法察觉。
    2026-11-18 审查指出 `stage_sens` 正是如此（固定生成 current/lims_heat 两组，
    却按 `args.coal_set` 取 ξ），此处统一拦在求解之前。
    """
    want_table = PAIRING[coal_set]
    want_sha = WINDOWS_SHA[want_table]
    actual = _sha256(Path(windows_csv))
    if actual != want_sha:
        raise SystemExit(
            f"STOP[{stage_name}]: 煤质集合 {coal_set} 必须配 {want_table} 窗口表"
            f"（期望 sha256 {want_sha}），实际 {actual}（{windows_csv}）。"
            "第 0 节禁止新旧混用。")
    if stage_name == "sens" and want_table != "p2.2d":
        raise SystemExit(
            f"STOP[sens]: 敏感性阶段固定生成 current 与 lims_heat 两组煤质，"
            f"因此必须用 P2.2-D 窗口表并用主口径的 ξ；"
            f"当前 coal_set={coal_set} 对应 {want_table}。请用 --coal-set current。")


def table(tbl: str) -> pd.DataFrame:
    if tbl == "main":
        return _CTX["main"]
    if tbl not in _CTX["tables"]:
        _CTX["tables"][tbl] = pd.read_csv(P114D_CSV).set_index("window_id")
    return _CTX["tables"][tbl]


def table_sha(tbl: str) -> str:
    return _CTX["windows_sha"] if tbl == "main" else WINDOWS_SHA["p1.14d"]


def rows_for(coal_set: str) -> dict[str, dict]:
    if coal_set in _CTX["rows"]:
        return _CTX["rows"][coal_set]
    ok, _rows, primary = load_frozen_split(coal_set="legacy_20261118")
    if not ok:
        raise RuntimeError("frozen split was not consumed intact")
    if coal_set == "lims_heat":
        base_rows = model_case_rows(primary, coal_set="current")
        hv = HV_EXPECT["lims_heat"]
        for r in base_rows:
            r["coal_heating_value_dry_kj_kg_value"] = float(hv[f"phase{int(r['phase_number'])}"])
    else:
        base_rows = model_case_rows(primary, coal_set=coal_set)
    by_id = {r["window_id"]: r for r in base_rows}
    csv_ids = set(str(w) for w in _CTX["main"].index)
    if csv_ids != set(str(k) for k in by_id):
        raise RuntimeError("windows ids do not match frozen split")
    _CTX["rows"][coal_set] = by_id
    return by_id


# ---------------------------------------------------------------------------
# 缓存（键必须含煤质集合与窗口表指纹）
# ---------------------------------------------------------------------------

def cache_key(*, window_id: str, coal_set: str, tbl: str, carrier_tag: str,
              hl: float, pert: str, dry_coal: float, carrier_nm3: float) -> str:
    """P2.0-A 的 cache_key 不含煤质集合，三种煤质共用一个键会互相串味。

    这里在 P2.0-A 的字段之外再加：coal_set 名、所选集合四份 YAML 的 sha 摘要、
    所用窗口表的 sha256、载气口径与实际载气/干煤量（C-m 与 C-J 同窗同 id）。
    """
    o2m, coalm, carm = PERT_SCALE[pert]
    return json.dumps({
        "window_id": str(window_id), "candidate": CAND, "param": repr(0.0), "x0": None,
        "hl": repr(float(hl)), "o2": repr(float(o2m)), "coal": repr(float(coalm)),
        "carrier": repr(float(carm)),
        "dry_coal_kg_s": repr(float(dry_coal)),
        "carrier_nm3_h": repr(float(carrier_nm3)),
        "carrier_tag": carrier_tag,
        "coal_set": coal_set, "coal_digest": coal_digest(coal_set),
        "windows_table": tbl, "windows_sha": table_sha(tbl),
        "kappa": repr(float(p20.KAPPA)), "f_gas": repr(float(p20.F_GAS)),
        "f_co2_rate": repr(float(p20.F_CO2_RATE)),
        "L_wsr": repr(float(p20.L_WSR)), "dz_down": repr(float(p20.DZ_DOWN)),
        "code_fingerprint": p20.CODE_FINGERPRINT,
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
    if hit.get("key") != key:
        return None
    return hit.get("snap")


def _store_cache(key: str, snap: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(key).write_text(
        json.dumps({"key": key, "snap": snap}, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# 求解（求解层复用 P2.0-A：_build_op / _snapshot 参数化载气列与煤质集合）
# ---------------------------------------------------------------------------

def build_op(hl: float, o2m: float, dry_coal: float, carrier_nm3: float) -> dict:
    return {
        "HeatLossModel": "total_percent",
        "HeatLossPercentTotal": float(hl),
        "GasificationRateFactor": float(p20.F_GAS),
        "CharCO2RateFactor": float(p20.F_CO2_RATE),
        "carrier_gas_flow": p18a._carrier_kg_s(carrier_nm3),
        "MeshFamily": "wsr_plus_pfr",
        "WSRLength_m": float(p20.L_WSR),
        "PFRCellLength_m": float(p20.DZ_DOWN),
        "OxygenMixingModel": "instant",
        "o2_flow_scale": float(p20.KAPPA) * float(o2m),
    }


def _snapshot(*, window_id: str, coal_set: str, tbl: str, carrier_tag: str,
              hl: float, pert: str, want_profile: bool) -> dict:
    """复制自 P2.0-A ``_snapshot``，只把煤质集合、窗口表与载气列变成显式参数。"""
    o2m, coalm, carm = PERT_SCALE[pert]
    df = table(tbl)
    row = rows_for(coal_set)[window_id]
    dry = float(df.loc[window_id, "coal_dry_recon_phase_kg_s"]) * float(coalm)
    carrier_nm3 = float(df.loc[window_id, CARRIERS[carrier_tag]]) * float(carm)
    op = build_op(hl, o2m, dry, carrier_nm3)
    rec = run_case(row, n_cells=p20.mesh_n(), coal_flow_dry_kg_s=dry,
                   keep_system=True, mesh_family="wsr_plus_pfr", extra_op=op)
    system = rec.get("_system")
    snap: dict = {
        "window_id": window_id, "candidate": CAND, "param": 0.0,
        "coal_set": coal_set, "windows_table": tbl,
        "carrier_tag": carrier_tag, "carrier_column": CARRIERS[carrier_tag],
        "hl": float(hl), "o2_mult": float(o2m), "coal_mult": float(coalm),
        "carrier_mult": float(carm), "pert": pert,
        "phase": _phase_of(window_id),
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
    c_out = float(outlet.solid_mass) * float(outlet.carbon_fraction)
    c_wsr = float(system.results[0].solid_mass) * float(system.results[0].carbon_fraction)
    snap.update({
        "ok": True,
        "X_C_pp": float((1.0 - c_out / max(c_in, 1e-12)) * 100.0),
        "X_C_wsr_pp": float((1.0 - c_wsr / max(c_in, 1e-12)) * 100.0),
        "T_out_K": float(outlet.T),
        "T_wsr_K": float(system.results[0].T),
        "Ts_wsr_K": float(system.results[0].T_solid_or_gas),
        "F_CO": float(gas[2]), "F_CO2": float(gas[3]), "F_H2": float(gas[5]),
        "F_H2O": float(gas[7]), "F_CH4": float(gas[1]),
        "dry_total_mol_s": float(np.sum(gas[:7])),
        "coal_C_kg_s": float(c_in),
    })
    snap.update(p20._closures(system))
    if want_profile:
        prof = []
        for i, (cell, st) in enumerate(zip(system.cells, system.results)):
            c_local = float(st.solid_mass) * float(st.carbon_fraction)
            prof.append({
                "i": int(i), "z_m": float(st.z), "dz_m": float(cell.dz),
                "X_C_frac": float(1.0 - c_local / max(c_in, 1e-12)),
                "T_gas_K": float(st.T),
                "T_solid_K": float(st.T_solid_or_gas),
                "is_combustion": bool(
                    system.cell_diagnostics[i].get("is_combustion_zone", False)
                ) if i < len(system.cell_diagnostics) else False,
            })
        snap["profile"] = prof
    rec.pop("_system", None)
    return snap


def solve_snap(*, window_id: str, coal_set: str, tbl: str = "main",
               carrier_tag: str = MAIN_CARRIER, hl: float = MAIN_HL,
               pert: str = "nom", want_profile: bool = False) -> dict:
    df = table(tbl)
    o2m, coalm, carm = PERT_SCALE[pert]
    dry = float(df.loc[window_id, "coal_dry_recon_phase_kg_s"]) * float(coalm)
    carrier_nm3 = float(df.loc[window_id, CARRIERS[carrier_tag]]) * float(carm)
    key = cache_key(window_id=window_id, coal_set=coal_set, tbl=tbl,
                    carrier_tag=carrier_tag, hl=hl, pert=pert,
                    dry_coal=dry, carrier_nm3=carrier_nm3)
    hit = _load_cache(key)
    if hit is not None and (hit.get("profile") or not want_profile):
        return hit
    snap = _snapshot(window_id=window_id, coal_set=coal_set, tbl=tbl,
                     carrier_tag=carrier_tag, hl=hl, pert=pert,
                     want_profile=want_profile)
    snap["pert"] = pert
    _store_cache(key, snap)
    return snap


def _job(spec: dict) -> dict:
    try:
        s = solve_snap(
            window_id=str(spec["window_id"]), coal_set=str(spec["coal_set"]),
            tbl=str(spec.get("tbl", "main")),
            carrier_tag=str(spec.get("carrier_tag", MAIN_CARRIER)),
            hl=float(spec.get("hl", MAIN_HL)), pert=str(spec.get("pert", "nom")),
            want_profile=bool(spec.get("want_profile", False)),
        )
        s["spec_tag"] = spec.get("tag", "")
        return s
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return {
            "window_id": spec.get("window_id"), "coal_set": spec.get("coal_set"),
            "windows_table": spec.get("tbl", "main"),
            "carrier_tag": spec.get("carrier_tag", MAIN_CARRIER),
            "hl": spec.get("hl", MAIN_HL), "pert": spec.get("pert", "nom"),
            "ok": False, "spec_tag": spec.get("tag", ""),
            "phase": (_phase_of(str(spec["window_id"]))
                      if str(spec.get("window_id")) in _CTX["main"].index else ""),
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_specs(specs: list[dict], n_workers: int) -> list[dict]:
    if not specs:
        return []
    n_workers = max(1, min(int(n_workers), len(specs)))
    if n_workers == 1:
        return [_job(s) for s in specs]
    windows_csv = _CTX["windows_csv"]
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                             initargs=(windows_csv,)) as ex:
        return list(ex.map(_job, specs, chunksize=1))


def _init_worker(windows_csv: str) -> None:
    silence_solver_logs()
    init_ctx(Path(windows_csv))


# ---------------------------------------------------------------------------
# ξ（每阶段一个常数，主口径参照窗 CO2 中位 = 装置值）
# ---------------------------------------------------------------------------

def compute_xi(coal_set: str) -> dict[str, float]:
    ref = _ref_windows("main")
    df = _CTX["main"]
    out: dict[str, float] = {}
    for ph in PHASES:
        wids = ref[ph]
        store = {(w, CAND, 0.0, MAIN_HL): solve_snap(window_id=w, coal_set=coal_set,
                                                     hl=MAIN_HL, pert="nom")
                 for w in wids}
        target = float(np.median([float(df.loc[w, "co2_vol_pct_mean"]) for w in wids]))
        out[ph] = p20.solve_xi(store, wids, CAND, 0.0, MAIN_HL, target)
    return out


def load_or_compute_xi(coal_set: str) -> dict[str, float]:
    path = OUT_DIR / "xi.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("coal_set") == coal_set:
            return {ph: float(v) for ph, v in payload["xi_mol_s"].items()}
    xi = compute_xi(coal_set)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"coal_set": coal_set,
                                "xi_mol_s": {k: float(v) for k, v in xi.items()}},
                               ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return xi


# ---------------------------------------------------------------------------
# unit_checks / solve_failures 追加
# ---------------------------------------------------------------------------

def _append_rows(name: str, rows: list[dict], key_cols: tuple[str, ...]) -> None:
    path = OUT_DIR / name
    new = pd.DataFrame(rows)
    cols = list(key_cols)
    if path.exists():
        old = pd.read_csv(path)
        if set(cols).issubset(old.columns):
            drop = set(map(tuple, new[cols].astype(str).to_numpy()))
            keep = [tuple(x) not in drop for x in old[cols].astype(str).to_numpy()]
            old = old[keep]
            new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False, lineterminator="\n")


def _record_failures(snaps: list[dict]) -> None:
    rows = []
    for s in snaps:
        if s.get("ok"):
            continue
        rows.append({
            "window_id": s.get("window_id"), "coal_set": s.get("coal_set"),
            "windows_table": s.get("windows_table", ""),
            "carrier_tag": s.get("carrier_tag", ""),
            "hl_pct": s.get("hl"), "pert": s.get("pert"),
            "phase": s.get("phase", ""),
            "execution_status": s.get("execution_status", s.get("error", "")),
            "spec_tag": s.get("spec_tag", ""),
            "max_residual_cost": s.get("max_residual_cost", float("nan")),
            "note": s.get("error", ""),
        })
    if not rows:
        # 失败为零也要有表头，否则产物清单里会出现 missing
        if not (OUT_DIR / "solve_failures.csv").exists():
            pd.DataFrame(columns=["window_id", "coal_set", "windows_table", "carrier_tag",
                                   "hl_pct", "pert", "phase", "execution_status",
                                   "spec_tag", "max_residual_cost", "note"]).to_csv(
                OUT_DIR / "solve_failures.csv", index=False, lineterminator="\n")
        return
    _append_rows("solve_failures.csv", rows,
                 ("window_id", "coal_set", "windows_table", "carrier_tag", "hl_pct",
                  "pert", "spec_tag"))


# ---------------------------------------------------------------------------
# unit 阶段（第 0 节断言 + U1 + U2）
# ---------------------------------------------------------------------------

def _export_hv_provenance() -> list[dict]:
    rows = []
    for name in ("model_case_contract.yaml", "model_case_contract_huayi_p2.yaml",
                 "model_case_contract_huayi_p3.yaml", "model_case_contract_huayi_p4.yaml",
                 "model_case_contract_yulin_wgs2.yaml"):
        path = DATA_PREP_ROOT / "config" / name
        if not path.exists():
            rows.append({"file": name, "source_status": "MISSING", "assumption_ref": ""})
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        fields = data.get("variables") or data.get("fields") or data
        blk = fields.get("coal_heating_value_dry_kj_kg") if isinstance(fields, dict) else None
        blk = blk or {}
        rows.append({
            "file": name,
            "source_status": blk.get("source_status", ""),
            "assumption_ref": blk.get("assumption_ref", ""),
        })
    return rows


def stage_unit(args) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prepare_coal_inputs()
    init_ctx(Path(args.windows_csv))
    coal_set = args.coal_set
    rows: list[dict] = []
    hard_fail: list[str] = []

    # ---- 第 0 节第 1 条：煤质集合与窗口表配对（sha256）
    actual_sha = _CTX["windows_sha"]
    want_table = PAIRING[coal_set]
    want_sha = WINDOWS_SHA[want_table]
    pairing_ok = actual_sha == want_sha
    row_no = 1 if want_table == "p2.2d" else 2
    rows.append({
        "check": "PAIRING", "coal_set": coal_set,
        "item": f"煤质集合 ↔ 窗口表（第 0 节第 {row_no} 行）",
        "n": 1, "value": actual_sha, "threshold": want_sha, "pass": bool(pairing_ok),
        "detail": f"windows_csv={_rel(Path(args.windows_csv))} expected_table={want_table}",
    })
    if not pairing_ok:
        hard_fail.append("PAIRING")

    # ---- 第 0 节第 2 条：四阶段热值
    hv = coal_hv(coal_set)
    bad = {ph: [hv[ph], HV_EXPECT[coal_set][ph]] for ph in PHASES
           if hv[ph] != HV_EXPECT[coal_set][ph]}
    rows.append({
        "check": "COAL_HV", "coal_set": coal_set,
        "item": "四阶段热值等于所选集合的值", "n": len(PHASES),
        "value": ";".join(f"{PHASE_LABEL[p]}={hv[p]:.0f}" for p in PHASES),
        "threshold": ";".join(f"{PHASE_LABEL[p]}={HV_EXPECT[coal_set][p]:.0f}"
                              for p in PHASES),
        "pass": bool(not bad), "detail": json.dumps(bad, ensure_ascii=False) if bad else "",
    })
    if bad:
        hard_fail.append("COAL_HV")

    # ---- 第 0 节第 3 条：导出算例的热值溯源（合同侧，不是煤质 YAML 侧）
    prov = _export_hv_provenance()
    huayi = [r for r in prov if "yulin" not in r["file"]]
    ok_prov = all(r["source_status"] == "literature_derived" for r in huayi)
    rows.append({
        "check": "EXPORT_HV_PROVENANCE", "coal_set": coal_set,
        "item": "config/model_case_contract*.yaml → coal_heating_value_dry_kj_kg.source_status",
        "n": len(huayi),
        "value": ";".join(f"{r['file']}={r['source_status']}" for r in huayi),
        "threshold": "literature_derived", "pass": bool(ok_prov),
        "detail": "审计表读煤质 YAML、预览读合同文件，两者不是同一来源，故单独断言导出侧",
    })
    if not ok_prov:
        hard_fail.append("EXPORT_HV_PROVENANCE")
    yulin = [r for r in prov if "yulin" in r["file"]]
    rows.append({
        "check": "EXPORT_HV_PROVENANCE", "coal_set": coal_set,
        "item": "（参考，不参与断言）yulin 合同", "n": len(yulin),
        "value": ";".join(f"{r['file']}={r['source_status']}" for r in yulin),
        "threshold": "", "pass": "", "detail": "非 R52-C-75 数据集，仅记录",
    })

    # ---- U1：legacy 煤质 + 旧窗口表 vs P2.0-A C0
    if coal_set == "legacy_20261118" and _CTX["windows_sha"] == WINDOWS_SHA["p1.14d"]:
        rows.extend(_unit_u1())
    else:
        rows.append({
            "check": "U1", "coal_set": coal_set,
            "item": "不适用：U1 须 legacy 煤质 + p1.14d 旧窗口表", "n": 0,
            "value": "", "threshold": 1e-9, "pass": "",
            "detail": "按第 0 节配对表，U1 只在 legacy+旧窗口上定义",
        })

    # ---- U2：current 煤质 + 新窗口表，元素闭合量级 vs P2.0-A C0
    if coal_set == "current" and _CTX["windows_sha"] == WINDOWS_SHA["p2.2d"]:
        rows.extend(_unit_u2())
    else:
        rows.append({
            "check": "U2", "coal_set": coal_set,
            "item": "不适用：U2 须 current 煤质 + P2.2-D 新窗口表", "n": 0,
            "value": "", "threshold": 10.0, "pass": "",
            "detail": "按第 0 节配对表，U2 只在 current+新窗口上定义",
        })

    _append_rows("unit_checks.csv", rows, ("check", "coal_set", "item"))
    print(f"[unit] {len(rows)} rows written for coal_set={coal_set}", flush=True)
    if hard_fail:
        raise SystemExit(f"STOP: 第 0 节断言未通过 {hard_fail}；见 unit_checks.csv。")


def _p20a_c0_snaps() -> dict[str, dict]:
    out = {}
    for s in json.loads((P20A_DIR / "perturb_snaps.json").read_text(encoding="utf-8")):
        if s.get("candidate") == CAND and s.get("pert") == "nom":
            out[str(s["window_id"])] = s
    return out


def _unit_u1() -> list[dict]:
    """9 个参照窗 C0 与 P2.0-A 逐窗一致（相对 ≤ 1e-9），并核对 phase 中位。"""
    ref = _ref_windows("main")
    old_snaps = _p20a_c0_snaps()
    cp = pd.read_csv(P20A_DIR / "comparison_points.csv")
    rows: list[dict] = []
    max_rel = 0.0
    detail = []
    n_ref = 0
    for ph in PHASES:
        for w in ref[ph]:
            n_ref += 1
            s = solve_snap(window_id=w, coal_set="legacy_20261118", hl=MAIN_HL, pert="nom")
            o = old_snaps.get(w, {})
            a, b = _f(s.get("X_C_pp")), _f(o.get("X_C_pp"))
            rel = (abs(a - b) / max(abs(b), 1e-12)
                   if math.isfinite(a) and math.isfinite(b) else float("inf"))
            max_rel = max(max_rel, rel)
            if rel > 1e-9:
                detail.append({"window_id": w, "model": a, "p2_0a": b, "rel": rel})
    rows.append({
        "check": "U1", "coal_set": "legacy_20261118",
        "item": "9 参照窗 C0 逐窗 X_C_pp vs P2.0-A perturb_snaps.json",
        "n": n_ref, "value": max_rel, "threshold": 1e-9,
        "pass": bool(max_rel <= 1e-9),
        "detail": json.dumps(detail, ensure_ascii=False) if detail else "逐窗一致",
    })

    for ph in PHASES:
        hit = cp[(cp["candidate"] == CAND) & (cp["phase"] == ph)]
        if not len(hit):
            continue
        r = hit.iloc[0]
        xi = _f(r["xi_mol_s"])
        xs, ts, cos, co2s, h2s = [], [], [], [], []
        for w in ref[ph]:
            s = solve_snap(window_id=w, coal_set="legacy_20261118", hl=MAIN_HL, pert="nom")
            y = p20.apply_xi(s, xi)
            if y is None:
                continue
            xs.append(y["X_C_pp"]); ts.append(y["T_out_K"])
            cos.append(y["CO_vol_pct"]); co2s.append(y["CO2_vol_pct"])
            h2s.append(y["H2_vol_pct"])
        obs = {
            "X_C_pp_median": (float(np.median(xs)), _f(r["X_C_pp_median"])),
            "T_out_med_K": (float(np.median(ts)), _f(r["T_out_med_K"])),
            "CO_med_model": (float(np.median(cos)), _f(r["CO_med_model"])),
            "CO2_med_model": (float(np.median(co2s)), _f(r["CO2_med_model"])),
            "H2_med_model": (float(np.median(h2s)), _f(r["H2_med_model"])),
        }
        dmax = 0.0
        for _k, (a, b) in obs.items():
            dmax = max(dmax, abs(a - b) / max(abs(b), 1e-12))
        rows.append({
            "check": "U1", "coal_set": "legacy_20261118",
            "item": f"{ph} 参照窗中位 vs P2.0-A comparison_points.csv C0 行",
            "n": len(xs), "value": dmax, "threshold": 1e-9,
            "pass": bool(dmax <= 1e-9),
            "detail": json.dumps({k: {"ours": v[0], "p2_0a": v[1]} for k, v in obs.items()},
                                 ensure_ascii=False),
        })
    return rows


def _unit_u2() -> list[dict]:
    """current+新窗：9 参照窗 C/H/O 闭合残差量级 vs P2.0-A C0 同窗，不高于 10 倍。

    主口径（口径 A）：逐元素取 9 窗的**最大绝对残差**，与该元素在 P2.0-A 的 9 窗最大
    绝对残差相比 —— 这是「同量级」的实际判据。参考口径（口径 B，`U2-REF`）：逐窗逐一
    比值的最大值；求解器停在机器噪声底（1e-11）时该比值没有物理意义，故只作记录。
    """
    ref = _ref_windows("main")
    old_snaps = _p20a_c0_snaps()
    rows: list[dict] = []
    detail_rows = []
    n_ref = sum(len(v) for v in ref.values())
    for el in ("C", "H", "O"):
        ours, olds, ratios = [], [], []
        for ph in PHASES:
            for w in ref[ph]:
                s = solve_snap(window_id=w, coal_set="current", hl=MAIN_HL, pert="nom")
                a = abs(_f(s.get(f"{el}_closure_fraction")))
                b = abs(_f(old_snaps.get(w, {}).get(f"{el}_closure_fraction")))
                ratio = a / b if b > 0 else (float("inf") if a > 0 else 1.0)
                ours.append(a)
                olds.append(b)
                ratios.append(ratio)
                detail_rows.append({"element": el, "phase": ph, "window_id": w,
                                    "ours_abs": a, "p2_0a_abs": b, "ratio": ratio})
        omax, pmax = max(ours), max(olds)
        agg = omax / pmax if pmax > 0 else (float("inf") if omax > 0 else 1.0)
        rows.append({
            "check": "U2", "coal_set": "current",
            "item": f"{el} 闭合残差量级（9 窗最大绝对残差比，口径 A）",
            "n": n_ref, "value": agg, "threshold": 10.0, "pass": bool(agg <= 10.0),
            "detail": f"ours_max={omax:.3e} p2_0a_max={pmax:.3e}",
        })
        n_bad = sum(1 for x in ratios if x > 10.0)
        rows.append({
            "check": "U2-REF", "coal_set": "current",
            "item": f"{el} 逐窗比值（参考口径 B，不作判据）",
            "n": n_ref, "value": max(ratios), "threshold": 10.0,
            "pass": bool(n_bad == 0),
            "detail": (f"over_10x_windows={n_bad}；P2.0-A 部分窗的残差落在求解器噪声底"
                       f"（约 1e-11），逐窗比值无物理意义，故仅记录"),
        })
    _write_csv(pd.DataFrame(detail_rows), "unit_u2_closures.csv")
    return rows


# ---------------------------------------------------------------------------
# compare 阶段（主口径 142 窗 × 7 spec）
# ---------------------------------------------------------------------------

def stage_compare(args) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prepare_coal_inputs()
    require_pairing(args.windows_csv, args.coal_set, "compare")
    init_ctx(Path(args.windows_csv))
    coal_set = args.coal_set
    if _CTX["windows_sha"] != WINDOWS_SHA[PAIRING[coal_set]]:
        raise SystemExit("STOP: 煤质集合与窗口表配对不成立（先跑 unit 阶段）")
    pw = _phase_windows("main")
    specs = [{"window_id": w, "coal_set": coal_set, "hl": MAIN_HL,
              "carrier_tag": MAIN_CARRIER, "pert": pert, "tag": f"main|{pert}"}
             for ph in PHASES for w in pw[ph] for pert in PERTS]
    print(f"[compare] {len(specs)} solves on {args.n_workers} workers", flush=True)
    out = run_specs(specs, args.n_workers)
    _record_failures(out)

    df = _CTX["main"]
    nom = [s for s in out if s.get("pert") == "nom"]
    xi = compute_xi(coal_set)
    (OUT_DIR / "xi.json").write_text(
        json.dumps({"coal_set": coal_set,
                    "xi_mol_s": {k: float(v) for k, v in xi.items()}},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rows = []
    for s in nom:
        w = s["window_id"]
        ph = s["phase"]
        y = p20.apply_xi(s, xi[ph]) if s.get("ok") else None
        dry = _f(s.get("dry_total_mol_s"))
        rows.append({
            "window_id": w, "phase": ph, "split_role": str(df.loc[w, "split_role"]),
            "coal_set": coal_set,
            "X_C_pp": _f(s.get("X_C_pp")),
            "T_out_K": _f(s.get("T_out_K")), "T_wsr_K": _f(s.get("T_wsr_K")),
            "CO_vol_pct": _f(s["F_CO"]) / dry * 100.0 if s.get("ok") and dry else float("nan"),
            "H2_vol_pct": _f(s["F_H2"]) / dry * 100.0 if s.get("ok") and dry else float("nan"),
            "CO2_vol_pct": _f(s["F_CO2"]) / dry * 100.0 if s.get("ok") and dry else float("nan"),
            "CH4_vol_pct": _f(s["F_CH4"]) / dry * 100.0 if s.get("ok") and dry else float("nan"),
            "dry_total_mol_s": dry,
            "coal_dry_kg_s": _f(s.get("coal_dry_kg_s")),
            "carrier_nm3_h": _f(s.get("carrier_nm3_h")),
            "xi_mol_s": xi[ph],
            "CO_vol_pct_xi": y["CO_vol_pct"] if y else float("nan"),
            "H2_vol_pct_xi": y["H2_vol_pct"] if y else float("nan"),
            "CO2_vol_pct_xi": y["CO2_vol_pct"] if y else float("nan"),
            "CH4_vol_pct_xi": y["CH4_vol_pct"] if y else float("nan"),
            "dry_per_coal_mol_kg": y["dry_per_coal_mol_kg"] if y else float("nan"),
            "plant_X_C_pp": _f(df.loc[w, "xc_recon_phase_cal_measured_pct"]),
            "plant_CO_vol_pct": _f(df.loc[w, "co_vol_pct_mean"]),
            "plant_H2_vol_pct": _f(df.loc[w, "h2_vol_pct_mean"]),
            "plant_CO2_vol_pct": _f(df.loc[w, "co2_vol_pct_mean"]),
            "plant_CH4_vol_pct": _f(df.loc[w, "ch4_vol_pct_mean"]),
            "execution_status": str(s.get("execution_status", "unknown")),
            "n_cells": s.get("n_cells", 0),
            "max_residual_cost": _f(s.get("max_residual_cost")),
        })
    base = pd.DataFrame(rows).sort_values(["phase", "window_id"]).reset_index(drop=True)
    _write_csv(base, "baseline_142.csv")

    n_fail = base.groupby("phase")["execution_status"].apply(
        lambda s: int((s != "completed").sum())).to_dict()
    skipped = [ph for ph in PHASES if n_fail.get(ph, 0) > p20.FAIL_WINDOW_STOP]
    (OUT_DIR / "phase_status.json").write_text(
        json.dumps({"n_solve_fail_nom": {k: int(v) for k, v in n_fail.items()},
                    "phase_skipped": {ph: (ph in skipped) for ph in PHASES},
                    "threshold": p20.FAIL_WINDOW_STOP},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    slopes = _model_slopes(out, xi, skipped)
    _write_csv(slopes, "model_slopes.csv")
    _write_csv(_elasticities(slopes, base, skipped), "elasticities.csv")
    print(f"[compare] nom failures={n_fail} skipped={skipped}", flush=True)


def _index_snaps(snaps: list[dict]) -> dict[str, dict[str, dict]]:
    idx: dict[str, dict[str, dict]] = {ph: {} for ph in PHASES}
    for s in snaps:
        ph = s.get("phase")
        if ph not in idx:
            continue
        idx[ph].setdefault(str(s["window_id"]), {})[str(s.get("pert", "nom"))] = s
    return idx


def _model_slopes(snaps: list[dict], xi: dict[str, float], skipped: list[str]) -> pd.DataFrame:
    idx = _index_snaps(snaps)
    pw = _phase_windows("main")
    plant = pd.read_csv(P115A_DIR / "plant_slopes.csv")
    rows = []
    for ph in PHASES:
        wids = pw[ph]
        coll: dict[tuple[str, str], list[float]] = {}
        plus: dict[tuple[str, str], list[float]] = {}
        minus: dict[tuple[str, str], list[float]] = {}
        n_ok = 0
        n_fail = 0
        ceil_any = 0
        ceil_nom = 0
        for w in wids:
            sn = idx[ph].get(w, {})
            if not (sn.get("nom") or {}).get("ok"):
                n_fail += 1
                continue
            n_ok += 1
            xcs = [_f((sn.get(p) or {}).get("X_C_pp")) for p in PERTS]
            if any(math.isfinite(v) and v >= B6_CEIL_PP for v in xcs):
                ceil_any += 1
            if _f(sn["nom"].get("X_C_pp")) >= B6_CEIL_PP:
                ceil_nom += 1
            stats = p20.fd_stats(sn, xi[ph])
            for obs in OBS_ALL:
                for inp in INPUTS:
                    coll.setdefault((obs, inp), []).append(
                        stats.get(f"d2_{obs}__{inp}", float("nan")))
                    plus.setdefault((obs, inp), []).append(
                        stats.get(f"dplus_{obs}__{inp}", float("nan")))
                    minus.setdefault((obs, inp), []).append(
                        stats.get(f"dminus_{obs}__{inp}", float("nan")))
        for obs in OBS_ALL:
            for inp in INPUTS:
                vals = [v for v in coll.get((obs, inp), []) if math.isfinite(v)]
                pv = [v for v in plus.get((obs, inp), []) if math.isfinite(v)]
                mv = [v for v in minus.get((obs, inp), []) if math.isfinite(v)]
                hit = plant[(plant["phase"] == ph) & (plant["obs"] == obs)
                            & (plant["input"] == inp) & (plant["model"] == "with_carrier")]
                plant_per3 = _f(hit.iloc[0]["per3"]) if len(hit) else float("nan")
                med = float(np.median(vals)) if vals else float("nan")
                same = bool(math.isfinite(med) and math.isfinite(plant_per3)
                            and med != 0.0 and plant_per3 != 0.0
                            and np.sign(med) == np.sign(plant_per3))
                rows.append({
                    "phase": ph, "obs": obs, "input": inp,
                    "beta_ln_median": med / LN_P if math.isfinite(med) else float("nan"),
                    "per3_median": med,
                    "per3_min": float(np.min(vals)) if vals else float("nan"),
                    "per3_max": float(np.max(vals)) if vals else float("nan"),
                    "dplus_median": float(np.median(pv)) if pv else float("nan"),
                    "dminus_median": float(np.median(mv)) if mv else float("nan"),
                    "n_windows": len(wids), "n_ok": len(vals), "n_solve_fail": n_fail,
                    "phase_skipped": ph in skipped,
                    "plant_per3": plant_per3,
                    "plant_boot_lo": _f(hit.iloc[0]["boot_lo"]) if len(hit) else float("nan"),
                    "plant_boot_hi": _f(hit.iloc[0]["boot_hi"]) if len(hit) else float("nan"),
                    "plant_signed": bool(hit.iloc[0]["signed"]) if len(hit) else False,
                    "same_sign": same,
                    "xc_ceil_frac_nom": ceil_nom / n_ok if n_ok else float("nan"),
                    "xc_ceil_frac_any_pert": ceil_any / n_ok if n_ok else float("nan"),
                    "xi_mol_s": xi[ph],
                })
    return pd.DataFrame(rows)


def _elasticities(slopes: pd.DataFrame, base: pd.DataFrame,
                  skipped: list[str]) -> pd.DataFrame:
    p20a = pd.read_csv(P20A_DIR / "elasticities.csv")
    rows = []
    for ph in PHASES:
        sub = base[base["phase"] == ph]
        xc = float(np.median(sub["X_C_pp"].dropna())) if len(sub) else float("nan")
        sl = slopes[(slopes["phase"] == ph) & (slopes["obs"] == "X_C_pp")]
        p_coal = _f(sl[sl["input"] == "ln_coal"]["per3_median"].iloc[0]) if len(sl) else float("nan")
        p_o2 = _f(sl[sl["input"] == "ln_O2"]["per3_median"].iloc[0]) if len(sl) else float("nan")
        e_coal, _r = p20.elasticities(xc, p_coal)
        _c, e_o2 = p20.elasticities(xc, p_o2)
        hit = p20a[(p20a["candidate"] == CAND) & (p20a["phase"] == ph)]
        r = hit.iloc[0] if len(hit) else None
        rows.append({
            "phase": ph, "X_C_pp_level": xc,
            "per3_XC_coal_model": p_coal, "per3_XC_o2_model": p_o2,
            "e_coal_model": e_coal, "e_o2_model": e_o2,
            "e_coal_plant": _f(r["e_coal_plant"]) if r is not None else float("nan"),
            "e_coal_plant_lo": _f(r["e_coal_plant_lo"]) if r is not None else float("nan"),
            "e_coal_plant_hi": _f(r["e_coal_plant_hi"]) if r is not None else float("nan"),
            "e_o2_plant": _f(r["e_o2_plant"]) if r is not None else float("nan"),
            "e_o2_plant_lo": _f(r["e_o2_plant_lo"]) if r is not None else float("nan"),
            "e_o2_plant_hi": _f(r["e_o2_plant_hi"]) if r is not None else float("nan"),
            "phase_skipped": ph in skipped,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# sens 阶段（9 参照窗 × 12 组，只算基准 nom）
# ---------------------------------------------------------------------------

def stage_sens(args) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prepare_coal_inputs()
    require_pairing(args.windows_csv, args.coal_set, "sens")
    init_ctx(Path(args.windows_csv))
    ref = _ref_windows("main")
    specs = []
    for hv_set in ("current", "lims_heat"):
        for hl in HL_GRID:
            for ctag in CARRIERS:
                for ph in PHASES:
                    for w in ref[ph]:
                        specs.append({
                            "window_id": w, "coal_set": hv_set, "tbl": "main",
                            "carrier_tag": ctag, "hl": hl, "pert": "nom",
                            "tag": f"sens|{hv_set}|hl{hl:g}|{ctag}",
                        })
    print(f"[sens] {len(specs)} solves on {args.n_workers} workers", flush=True)
    out = run_specs(specs, args.n_workers)
    _record_failures(out)
    xi = load_or_compute_xi(args.coal_set)
    rows = []
    for hv_set in ("current", "lims_heat"):
        for hl in HL_GRID:
            for ctag in CARRIERS:
                for ph in PHASES:
                    ss = [s for s in out
                          if s.get("coal_set") == hv_set and s.get("carrier_tag") == ctag
                          and abs(_f(s.get("hl")) - hl) < 1e-9 and s.get("phase") == ph]
                    ok = [s for s in ss if s.get("ok")]
                    ys = [y for y in (p20.apply_xi(s, xi[ph]) for s in ok) if y]
                    rows.append({
                        "hv_variant": hv_set, "hl_pct": hl, "carrier": ctag, "phase": ph,
                        "hv_kj_kg": HV_EXPECT[hv_set][ph],
                        "n_ref_windows": len(ss), "n_ok": len(ok),
                        "n_solve_fail": len(ss) - len(ok), "xi_mol_s": xi[ph],
                        "X_C_pp_med": float(np.median([s["X_C_pp"] for s in ok])) if ok else float("nan"),
                        "T_out_K_med": float(np.median([s["T_out_K"] for s in ok])) if ok else float("nan"),
                        "CO_vol_pct_med": float(np.median([y["CO_vol_pct"] for y in ys])) if ys else float("nan"),
                        "H2_vol_pct_med": float(np.median([y["H2_vol_pct"] for y in ys])) if ys else float("nan"),
                        "CO2_vol_pct_med": float(np.median([y["CO2_vol_pct"] for y in ys])) if ys else float("nan"),
                        "CH4_vol_pct_med": float(np.median([y["CH4_vol_pct"] for y in ys])) if ys else float("nan"),
                        "is_main_basis": bool(hv_set == "current"
                                              and abs(hl - MAIN_HL) < 1e-9
                                              and ctag == MAIN_CARRIER),
                    })
    tab = pd.DataFrame(rows).sort_values(["hv_variant", "hl_pct", "carrier", "phase"])
    _write_csv(tab.reset_index(drop=True), "sensitivity_ref9.csv")
    print(f"[sens] written {len(tab)} rows", flush=True)


# ---------------------------------------------------------------------------
# profiles 阶段
# ---------------------------------------------------------------------------

def stage_profiles(args) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prepare_coal_inputs()
    require_pairing(args.windows_csv, args.coal_set, "profiles")
    init_ctx(Path(args.windows_csv))
    ref = _ref_windows("main")
    rows = []
    for ph in PHASES:
        for w in ref[ph]:
            s = solve_snap(window_id=w, coal_set=args.coal_set, hl=MAIN_HL,
                           pert="nom", want_profile=True)
            prof = s.get("profile")
            if not prof:
                rows.append({"phase": ph, "window_id": w, "z_target_m": float("nan"),
                             "X_C_pp": float("nan"), "T_gas_K": float("nan"),
                             "execution_status": s.get("execution_status", "no_profile")})
                continue
            wsr_end = prof[0]["z_m"] + prof[0]["dz_m"] / 2.0
            last = prof[-1]
            picks = [("wsr_out", wsr_end, prof[0]), ("z2", 2.0, _cell_at(prof, 2.0)),
                     ("z4", 4.0, _cell_at(prof, 4.0)), ("z6", 6.0, _cell_at(prof, 6.0)),
                     ("exit", last["z_m"] + last["dz_m"] / 2.0, last)]
            for tag, zt, cell in picks:
                rows.append({
                    "phase": ph, "window_id": w, "z_target_m": float(zt),
                    "z_cell_m": cell["z_m"],
                    "X_C_pp": float(cell["X_C_frac"]) * 100.0,
                    "T_gas_K": float(cell["T_gas_K"]),
                    "T_solid_K": float(cell["T_solid_K"]),
                    "execution_status": "completed",
                })
    tab = pd.DataFrame(rows)
    med_rows = []
    for ph in PHASES:
        sub = tab[(tab["phase"] == ph) & tab["X_C_pp"].notna()]
        for tag, zt in (("wsr_out", 1.0), ("z2", 2.0), ("z4", 4.0), ("z6", 6.0),
                        ("exit", 10.3)):
            block = sub[np.isclose(sub["z_target_m"], zt)]
            if not len(block):
                continue
            med_rows.append({
                "phase": ph, "window_id": "median_ref", "z_target_m": float(zt),
                "z_cell_m": float(block["z_cell_m"].median()),
                "X_C_pp": float(block["X_C_pp"].median()),
                "T_gas_K": float(block["T_gas_K"].median()),
                "T_solid_K": float(block["T_solid_K"].median()),
                "execution_status": "completed",
            })
    out = pd.concat([tab, pd.DataFrame(med_rows)], ignore_index=True)
    _write_csv(out, "profiles.csv")
    print(f"[profiles] written {len(out)} rows", flush=True)


def _cell_at(prof: list[dict], z: float) -> dict:
    best = None
    for cell in prof:
        zl = cell["z_m"] - cell["dz_m"] / 2.0
        zr = cell["z_m"] + cell["dz_m"] / 2.0
        if zl <= z <= zr:
            return cell
        if best is None or abs(cell["z_m"] - z) < abs(best["z_m"] - z):
            best = cell
    return best


# ---------------------------------------------------------------------------
# read 阶段（B1–B7）
# ---------------------------------------------------------------------------

def _skipped_phases() -> list[str]:
    """读 phase_status.json 里被判为跳过的阶段。缺失时返回空表（旧产物向后兼容）。"""
    p = OUT_DIR / "phase_status.json"
    if not p.exists():
        return []
    try:
        st = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [ph for ph, v in st.get("phase_skipped", {}).items() if v]


def _mask_skipped(rows: list[dict], summary: dict, skipped: list[str],
                  tag: str) -> tuple[list[dict], dict]:
    """把被跳过阶段的行改成不通过并写明原因，同时把对应 verdict 置 False。

    注意 B2 的 verdict 键是 `(phase, group)` 元组、B1 是 `phase` 字符串：
    这里只翻转**已存在的键**，不新增字符串键，否则下游的元组解包会崩。
    """
    out = []
    for r in rows:
        if r.get("phase") in skipped:
            r = dict(r)
            r["pass"] = False
            r["note"] = (f"[{tag}] 该阶段求解失败超过卡面阈值，已跳过判定；"
                         f"原读数 {r.get('value')}（{r.get('note', '')}）")
        out.append(r)
    verd = dict(summary.get("verdict", {}))
    for k in list(verd):
        ph = k[0] if isinstance(k, tuple) else k
        if ph in skipped:
            verd[k] = False
    new = dict(summary)
    new["verdict"] = verd
    new["skipped_phases"] = skipped
    return out, new


def _mask_skipped_b3(rows: list[dict], summary: dict,
                     skipped: list[str]) -> tuple[list[dict], dict]:
    """B3：被跳过阶段的组合不计同号，n_same 从掩码后的行重算。"""
    out = []
    for r in rows:
        if r.get("phase") in skipped:
            r = dict(r)
            r["pass"] = False
            r["note"] = (f"[B3] 该阶段求解失败超过卡面阈值，已跳过；"
                         f"原读数 同号={r.get('pass')}（{r.get('note', '')}）")
        out.append(r)
    # 从个体组合行重算同号数（排除被跳过阶段与汇总行本身）
    n_same = sum(1 for r in out
                 if r.get("reading") == "B3" and r.get("metric") != "n_same_sign"
                 and r.get("phase") not in skipped and r.get("pass") is True)
    for r in out:
        if r.get("reading") == "B3" and r.get("metric") == "n_same_sign":
            r["value"] = n_same
            r["pass"] = bool(n_same >= B3_MIN_SIGNED)
            r["note"] = (f"8 个有符号组合中同号 {n_same} 个（阈值 ≥{B3_MIN_SIGNED}）；"
                         f"已排除被跳过阶段：{','.join(skipped) if skipped else '无'}")
    new = dict(summary)
    new["n_same"] = n_same
    new["skipped_phases"] = skipped
    return out, new


def _b1() -> tuple[list[dict], dict]:
    df = pd.read_csv(OUT_DIR / "baseline_142.csv")
    readings = pd.read_csv(P22D_READINGS)
    fin = readings[readings["reading_id"] == "FINAL"]
    rows, summary, verdicts = [], {}, {}
    for ph in PHASES:
        sub = df[(df["phase"] == ph) & (df["execution_status"] == "completed")]
        med = float(np.median(sub["X_C_pp"])) if len(sub) else float("nan")
        summary[ph] = {"model_median_pct": med, "n_windows": int(len(sub))}
        if ph == "phase1":
            passed = bool(math.isfinite(med) and med >= B1_P1_MIN)
            p1 = fin[fin["phase"] == "P1"]
            rows.append({
                "reading": "B1", "metric": "z_vs_FINAL", "phase": ph,
                "value": float("nan"), "threshold": B1_P1_MIN, "pass": passed,
                "note": (f"模型 142 窗全窗中位 X_C={med:.4f}%；阶段 1 的 FINAL 行 verdict="
                         f"{p1['verdict'].iloc[0] if len(p1) else 'missing'}"
                         f"（6/6 组合不可行），不得当目标，故按卡面用"
                         f"「模型 ≥ {B1_P1_MIN:.0f}% 即通过」判定；"
                         f"FINAL 描述值={p1['value'].iloc[0] if len(p1) else 'n/a'}"),
            })
            verdicts[ph] = passed
            continue
        lbl = PHASE_LABEL[ph]
        tgt = fin[(fin["phase"] == lbl) & (fin["metric"] == "target_midpoint_pct")]
        unc = fin[(fin["phase"] == lbl) & (fin["metric"] == "target_uncertainty_pct")]
        if not len(tgt) or not len(unc):
            rows.append({"reading": "B1", "metric": "z_vs_FINAL", "phase": ph,
                         "value": float("nan"), "threshold": B1_Z_MAX, "pass": False,
                         "note": "P2.2-D readings.csv 缺 FINAL target 行"})
            verdicts[ph] = False
            continue
        t, s = float(tgt["value"].iloc[0]), float(unc["value"].iloc[0])
        z = (med - t) / s if s else float("nan")
        passed = bool(math.isfinite(z) and abs(z) <= B1_Z_MAX)
        summary[ph].update({"target_midpoint_pct": t, "target_uncertainty_pct": s, "z": z})
        rows.append({
            "reading": "B1", "metric": "z_vs_FINAL", "phase": ph,
            "value": z, "threshold": B1_Z_MAX, "pass": passed,
            "note": (f"模型阶段中位={med:.4f}%（142 窗全窗中位）；"
                     f"FINAL/target_midpoint_pct={t:.4f}% ± {s:.4f} pp"),
        })
        verdicts[ph] = passed
    return rows, {"phases": summary, "verdict": verdicts}


def _b2(sens: pd.DataFrame | None) -> tuple[list[dict], dict]:
    base = pd.read_csv(OUT_DIR / "baseline_142.csv")
    ref_ids = set(p20.ref_window_ids())
    rows, verdicts, detail = [], {}, {}
    for ph in PHASES:
        sub = base[(base["phase"] == ph) & base["window_id"].isin(ref_ids)
                   & (base["execution_status"] == "completed")]
        t_c = float(np.median(sub["T_out_K"])) - 273.15 if len(sub) else float("nan")
        ft, t25 = LIU_FT_C[ph], LIU_T25_C[ph]
        v = ("pass" if t_c >= ft else ("marginal" if t_c >= ft - B2_MARGIN_K else "fail"))
        verdicts[(ph, "main")] = v
        detail[ph] = {"T_out_C_median": t_c, "FT_C": ft, "T25_C": t25, "verdict": v}
        rows.append({
            "reading": "B2", "metric": "T_out_vs_FT", "phase": ph,
            "value": t_c, "threshold": ft, "pass": v != "fail",
            "note": (f"主口径参照窗中位 T_out={t_c:.2f} °C；FT={ft:.1f} °C，"
                     f"T25={t25:.1f} °C；判定={v}（≥FT 通过、FT−50 K ≤ T < FT 边缘）"),
        })
    if sens is not None:
        for _, r in sens.iterrows():
            ph = str(r["phase"])
            t_c = float(r["T_out_K_med"]) - 273.15
            ft = LIU_FT_C[ph]
            v = ("pass" if t_c >= ft else ("marginal" if t_c >= ft - B2_MARGIN_K else "fail"))
            verdicts[(ph, f"{r['hv_variant']}|hl{r['hl_pct']:g}|{r['carrier']}")] = v
            rows.append({
                "reading": "B2", "metric": "T_out_vs_FT_sens", "phase": ph,
                "value": t_c, "threshold": ft, "pass": v != "fail",
                "note": (f"{r['hv_variant']} 热值 {r['hv_kj_kg']:.0f} kJ/kg、"
                         f"热损 {r['hl_pct']:g}%、载气 {r['carrier']}："
                         f"T_out={t_c:.2f} °C，FT={ft:.1f} °C，判定={v}"),
            })
    return rows, {"detail": detail, "verdict": verdicts}


def _plant_slopes_recalc(tab: pd.DataFrame) -> pd.DataFrame:
    """用给定窗口表重算装置斜率（with_carrier），不改 p115a 的全局状态。"""
    saved = dict(p115a._CTX)
    p115a._CTX["p114d"] = tab.set_index("window_id")
    try:
        out = p115a.plant_slopes_table(tab)
    finally:
        p115a._CTX.clear()
        p115a._CTX.update(saved)
    return out[out["model"] == "with_carrier"]


def _b3() -> tuple[list[dict], dict]:
    slopes = pd.read_csv(OUT_DIR / "model_slopes.csv")
    plant = pd.read_csv(P115A_DIR / "plant_slopes.csv")
    rows, combos = [], []
    for ph, obs, inp in p20.G2_COMBOS:
        ms = slopes[(slopes["phase"] == ph) & (slopes["obs"] == obs)
                    & (slopes["input"] == inp)]["per3_median"]
        m = _f(ms.iloc[0]) if len(ms) else float("nan")
        pl = plant[(plant["phase"] == ph) & (plant["obs"] == obs)
                   & (plant["input"] == inp) & (plant["model"] == "with_carrier")]
        p = _f(pl.iloc[0]["per3"]) if len(pl) else float("nan")
        same = bool(math.isfinite(m) and math.isfinite(p) and m != 0.0 and p != 0.0
                    and np.sign(m) == np.sign(p))
        combos.append({"phase": ph, "obs": obs, "input": inp, "model_per3": m,
                       "plant_per3": p, "same_sign": same})
        rows.append({
            "reading": "B3", "metric": f"{obs}<->{inp}", "phase": ph,
            "value": m, "threshold": p, "pass": same,
            "note": f"模型阶段中位 per3={m:.4f}，装置 per3={p:.4f}，同号={same}",
        })
    n_same = sum(1 for c in combos if c["same_sign"])
    rows.append({
        "reading": "B3", "metric": "n_same_sign", "phase": "all",
        "value": n_same, "threshold": B3_MIN_SIGNED,
        "pass": bool(n_same >= B3_MIN_SIGNED),
        "note": f"8 个有符号组合中同号 {n_same} 个（阈值 ≥{B3_MIN_SIGNED}）",
    })

    # 卡面硬要求：阶段 2 上用新窗口表重算装置斜率核对（任一 beta_ln 偏差 > 1e-6 → 停）
    new_tab = pd.read_csv(P22D_CSV)
    old_tab = pd.read_csv(P114D_CSV)
    recalc_new = _plant_slopes_recalc(new_tab)
    recalc_old = _plant_slopes_recalc(old_tab)
    stored_p2 = plant[(plant["model"] == "with_carrier") & (plant["phase"] == "phase2")]
    detail, max_dev, max_dev_oldtab, outliers = [], 0.0, 0.0, []
    for _, r in recalc_new[recalc_new["phase"] == "phase2"].iterrows():
        hit = stored_p2[(stored_p2["obs"] == r["obs"]) & (stored_p2["input"] == r["input"])]
        old_hit = recalc_old[(recalc_old["phase"] == "phase2")
                             & (recalc_old["obs"] == r["obs"])
                             & (recalc_old["input"] == r["input"])]
        if not len(hit):
            continue
        a = _f(r["beta_ln"])
        b = _f(hit.iloc[0]["beta_ln"])
        c = _f(old_hit.iloc[0]["beta_ln"]) if len(old_hit) else float("nan")
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
        d = abs(a - b)
        max_dev = max(max_dev, d)
        max_dev_oldtab = max(max_dev_oldtab, abs(c - b))
        detail.append({"phase": "phase2", "obs": r["obs"], "input": r["input"],
                       "beta_recalc_new_table": a,
                       "beta_recalc_old_table": c,
                       "beta_stored_p115a": b,
                       "abs_dev_new_vs_stored": d,
                       "abs_dev_oldtab_vs_stored": abs(c - b)})
        if d > 1e-6:
            outliers.append({"obs": r["obs"], "input": r["input"], "recalc": a,
                             "stored": b, "abs_dev": d})
    _write_csv(pd.DataFrame(detail), "b3_check_detail.csv")
    rows.append({
        "reading": "B3-CHECK", "metric": "phase2_plant_slope_recalc_max_abs_dev",
        "phase": "phase2", "value": max_dev, "threshold": 1e-6,
        "pass": bool(max_dev <= 1e-6),
        "note": ("用新窗口表重算 phase2 装置斜率（with_carrier），与 "
                 "round-p1.15a-20261108/plant_slopes.csv 的 beta_ln 比；"
                 "同一函数用旧窗口表重算时最大偏差 "
                 f"{max_dev_oldtab:.3e}（用于确认调用本身没走偏）；"
                 "逐系数见 b3_check_detail.csv"),
    })
    return rows, {"combos": combos, "n_same": n_same, "recalc_max_dev": max_dev,
                  "recalc_max_dev_old_table": max_dev_oldtab,
                  "recalc_outliers": outliers, "detail": detail}



def _b4() -> tuple[list[dict], dict]:
    base = pd.read_csv(OUT_DIR / "baseline_142.csv")
    rows, out = [], {}
    for ph in PHASES:
        sub = base[(base["phase"] == ph) & (base["execution_status"] == "completed")]
        rec = {}
        for obs, col, pcol in (("CO", "CO_vol_pct_xi", "plant_CO_vol_pct"),
                               ("H2", "H2_vol_pct_xi", "plant_H2_vol_pct"),
                               ("CH4", "CH4_vol_pct_xi", "plant_CH4_vol_pct")):
            d = (sub[col] - sub[pcol]).dropna()
            med = float(np.median(d)) if len(d) else float("nan")
            rec[obs] = {"median_bias": med, "n": int(len(d))}
            rows.append({
                "reading": "B4", "metric": f"{obs}_bias_xi_vs_plant", "phase": ph,
                "value": med, "threshold": "", "pass": "",
                "note": f"ξ 闭合后 142 窗中位偏差（模型−装置），{obs}；只报告",
            })
        out[ph] = rec
    return rows, out


def _b5() -> tuple[list[dict], dict]:
    elas = pd.read_csv(OUT_DIR / "elasticities.csv")
    rows, out = [], {}
    for _, r in elas.iterrows():
        ph = str(r["phase"])
        rec = {}
        for tag, mcol, pcol, lo, hi in (
            ("e_coal", "e_coal_model", "e_coal_plant", "e_coal_plant_lo", "e_coal_plant_hi"),
            ("e_o2", "e_o2_model", "e_o2_plant", "e_o2_plant_lo", "e_o2_plant_hi"),
        ):
            m, p, l, h = _f(r[mcol]), _f(r[pcol]), _f(r[lo]), _f(r[hi])
            inside = bool(math.isfinite(m) and l <= m <= h)
            rec[tag] = {"model": m, "plant": p, "lo": l, "hi": h, "inside": inside}
            rows.append({
                "reading": "B5", "metric": tag, "phase": ph,
                "value": m, "threshold": f"[{l:.4f}, {h:.4f}]", "pass": "",
                "note": (f"模型 {tag}={m:.4f}；P2.0-A 装置 {tag}={p:.4f}，"
                         f"95% 区间 [{l:.4f}, {h:.4f}]；"
                         f"{'落在区间内' if inside else '落在区间外'}（只报告）"),
            })
        out[ph] = rec
    return rows, out


def _b6(args) -> tuple[list[dict], dict]:
    slopes = pd.read_csv(OUT_DIR / "model_slopes.csv")
    base = pd.read_csv(OUT_DIR / "baseline_142.csv")
    rows, out = [], {}
    ref = _ref_windows("main")
    for ph in PHASES:
        sl = slopes[(slopes["phase"] == ph) & (slopes["obs"] == "X_C_pp")
                    & (slopes["input"] == "ln_O2")]
        frac_nom = _f(sl["xc_ceil_frac_nom"].iloc[0]) if len(sl) else float("nan")
        frac_any = _f(sl["xc_ceil_frac_any_pert"].iloc[0]) if len(sl) else float("nan")
        sub = base[(base["phase"] == ph) & (base["execution_status"] == "completed")]
        med = float(np.median(sub["X_C_pp"])) if len(sub) else float("nan")
        rec = {"frac_ge_99.9_nom": frac_nom, "frac_ge_99.9_any_pert": frac_any,
               "median_X_C_pp": med, "o2_multiplier_for_99": float("nan")}
        rows.append({
            "reading": "B6", "metric": "frac_X_C_ge_99.9_nom", "phase": ph,
            "value": frac_nom, "threshold": "", "pass": "",
            "note": f"基准点 142 窗中 X_C ≥ 99.9% 的比例；阶段中位 X_C={med:.4f}%",
        })
        rows.append({
            "reading": "B6", "metric": "frac_X_C_ge_99.9_any_pert", "phase": ph,
            "value": frac_any, "threshold": "", "pass": "",
            "note": "基准 + 任一扰动点（6 个 ±3% 扰动）中 X_C ≥ 99.9% 的比例",
        })
        if math.isfinite(med) and med < B6_TARGET_PP:
            m = _solve_o2_multiplier(args.coal_set, ph, ref[ph], B6_TARGET_PP)
            rec["o2_multiplier_for_99"] = m
            rows.append({
                "reading": "B6", "metric": "o2_multiplier_for_99pct", "phase": ph,
                "value": m, "threshold": B6_TARGET_PP, "pass": "",
                "note": ("阶段中位 X_C < 99%，一维求根给出使参照窗中位 X_C=99% 的氧量倍数"
                         "（o2_flow_scale 倍数，其余冻结项不变）"),
            })
        else:
            rec["o2_multiplier_for_99"] = 1.0
            rows.append({
                "reading": "B6", "metric": "o2_multiplier_for_99pct", "phase": ph,
                "value": 1.0, "threshold": B6_TARGET_PP, "pass": "",
                "note": (f"阶段中位 X_C={med:.4f}% ≥ 99%，无需加氧；"
                         f"按「无需倍数」记 1.0（未做一维求根）"),
            })
        out[ph] = rec
    return rows, out


def _solve_o2_multiplier(coal_set: str, ph: str, wids: list[str], target: float) -> float:
    def med(m: float) -> float:
        vals = []
        for w in wids:
            s = _solve_with_o2(w, coal_set, m)
            if s.get("ok"):
                vals.append(float(s["X_C_pp"]))
        return float(np.median(vals)) if vals else float("nan")

    lo, hi = 1.0, 3.0
    f_lo, f_hi = med(lo) - target, med(hi) - target
    if not (math.isfinite(f_lo) and math.isfinite(f_hi)):
        return float("nan")
    if f_lo >= 0.0:
        return 1.0 if f_lo <= 0.2 else float("nan")
    for _ in range(4):
        if f_hi >= 0.0:
            break
        hi *= 1.5
        f_hi = med(hi) - target
    if f_hi < 0.0:
        return float("nan")
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        fm = med(mid) - target
        if not math.isfinite(fm):
            return float("nan")
        if abs(fm) < 1e-3 or (hi - lo) < 1e-4:
            return float(mid)
        if fm < 0.0:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def _solve_with_o2(w: str, coal_set: str, o2_mult: float) -> dict:
    """B6 一维求根用：给定绝对氧量倍数求解（缓存键含该倍数）。"""
    df = table("main")
    dry = float(df.loc[w, "coal_dry_recon_phase_kg_s"])
    carrier_nm3 = float(df.loc[w, CARRIERS[MAIN_CARRIER]])
    key = json.dumps({
        "window_id": str(w), "coal_set": coal_set, "coal_digest": coal_digest(coal_set),
        "windows_table": "main", "windows_sha": table_sha("main"),
        "carrier_tag": MAIN_CARRIER, "hl": repr(float(MAIN_HL)), "pert": "nom",
        "o2_mult_abs": repr(float(o2_mult)),
        "dry_coal_kg_s": repr(float(dry)), "carrier_nm3_h": repr(float(carrier_nm3)),
        "code_fingerprint": p20.CODE_FINGERPRINT, "candidate": CAND, "purpose": "B6",
    }, sort_keys=True)
    hit = _load_cache(key)
    if hit is not None:
        return hit
    row = rows_for(coal_set)[w]
    op = build_op(MAIN_HL, o2_mult, dry, carrier_nm3)
    rec = run_case(row, n_cells=p20.mesh_n(), coal_flow_dry_kg_s=dry,
                   keep_system=True, mesh_family="wsr_plus_pfr", extra_op=op)
    system = rec.get("_system")
    snap = {"window_id": w, "ok": False, "phase": _phase_of(w),
            "execution_status": str(rec.get("execution_status")), "o2_mult": float(o2_mult)}
    if system is not None and system.results and rec.get("execution_status") == "completed":
        outlet = system.results[-1]
        c_in = float(system.W_dry) * float(system.coal_props.get("Cd", 0.0)) / 100.0
        c_out = float(outlet.solid_mass) * float(outlet.carbon_fraction)
        snap.update({"ok": True, "X_C_pp": float((1.0 - c_out / max(c_in, 1e-12)) * 100.0),
                     "T_out_K": float(outlet.T)})
    rec.pop("_system", None)
    _store_cache(key, snap)
    return snap


def _b7() -> tuple[list[dict], dict]:
    sens = pd.read_csv(OUT_DIR / "sensitivity_ref9.csv")
    xi = load_or_compute_xi(_CTX.get("main_coal_set", "current"))
    rows, out = [], {}
    ref = _ref_windows("main")

    # ---- 7a：主口径（新窗口）刘文 vs 化验表热值，同热损 1%、同载气 C-m
    main_basis = sens[(sens["hl_pct"] == MAIN_HL) & (sens["carrier"] == MAIN_CARRIER)]
    out["hv_effect_main"] = {}
    for ph in PHASES:
        a = main_basis[(main_basis["phase"] == ph) & (main_basis["hv_variant"] == "current")]
        b = main_basis[(main_basis["phase"] == ph) & (main_basis["hv_variant"] == "lims_heat")]
        if not len(a) or not len(b):
            continue
        rec = {}
        for tag, col in (("dT_out_K", "T_out_K_med"), ("dX_C_pp", "X_C_pp_med"),
                         ("dCO_vol_pct", "CO_vol_pct_med"),
                         ("dH2_vol_pct", "H2_vol_pct_med")):
            la, lb = _f(a[col].iloc[0]), _f(b[col].iloc[0])
            rec[tag] = la - lb
            rows.append({
                "reading": "B7", "metric": tag + "_liu_minus_lims", "phase": ph,
                "value": la - lb, "threshold": "", "pass": "",
                "note": (f"主口径（新窗口、热损 1%、C-m）参照窗中位：刘文 {la:.4f} vs "
                         f"化验表 {lb:.4f}；用于核对论文「约 100–150 K」粗估"),
            })
        out["hv_effect_main"][ph] = rec

    # ---- 7b：U1（legacy+旧窗）与主口径之差，拆成三项
    states = (("A_legacy_old", "legacy_20261118", "old"),
              ("B_current_old", "current", "old"),
              ("C_lims_heat_old", "lims_heat", "old"),
              ("D_current_new_main", "current", "main"))
    vals: dict[str, dict[str, dict[str, float]]] = {}
    for tag, coal_set, tbl in states:
        vals[tag] = {}
        for ph in PHASES:
            xs, ts, cos, h2s = [], [], [], []
            for w in ref[ph]:
                s = solve_snap(window_id=w, coal_set=coal_set, tbl=tbl,
                               carrier_tag=MAIN_CARRIER, hl=MAIN_HL, pert="nom")
                if not s.get("ok"):
                    continue
                y = p20.apply_xi(s, xi[ph])
                if y is None:
                    continue
                xs.append(float(s["X_C_pp"])); ts.append(float(s["T_out_K"]))
                cos.append(float(y["CO_vol_pct"])); h2s.append(float(y["H2_vol_pct"]))
            vals[tag][ph] = {
                "X_C_pp": float(np.median(xs)) if xs else float("nan"),
                "T_out_K": float(np.median(ts)) if ts else float("nan"),
                "CO_vol_pct": float(np.median(cos)) if cos else float("nan"),
                "H2_vol_pct": float(np.median(h2s)) if h2s else float("nan"),
                "n": len(xs),
            }
    decomp = []
    for ph in PHASES:
        A, B, C, D = (vals[k][ph] for k in ("A_legacy_old", "B_current_old",
                                            "C_lims_heat_old", "D_current_new_main"))
        rec = {
            "phase": ph,
            "dX_C_pp_total": D["X_C_pp"] - A["X_C_pp"],
            "dT_out_K_total": D["T_out_K"] - A["T_out_K"],
            "dCO_vol_pct_total": D["CO_vol_pct"] - A["CO_vol_pct"],
            "dH2_vol_pct_total": D["H2_vol_pct"] - A["H2_vol_pct"],
            "dX_C_pp_coal_swap": B["X_C_pp"] - A["X_C_pp"],
            "dX_C_pp_hv_only": C["X_C_pp"] - B["X_C_pp"],
            "dX_C_pp_drycoal_const": D["X_C_pp"] - B["X_C_pp"],
            "dT_out_K_coal_swap": B["T_out_K"] - A["T_out_K"],
            "dT_out_K_hv_only": C["T_out_K"] - B["T_out_K"],
            "dT_out_K_drycoal_const": D["T_out_K"] - B["T_out_K"],
            "dCO_vol_pct_coal_swap": B["CO_vol_pct"] - A["CO_vol_pct"],
            "dCO_vol_pct_hv_only": C["CO_vol_pct"] - B["CO_vol_pct"],
            "dCO_vol_pct_drycoal_const": D["CO_vol_pct"] - B["CO_vol_pct"],
            "dH2_vol_pct_coal_swap": B["H2_vol_pct"] - A["H2_vol_pct"],
            "dH2_vol_pct_hv_only": C["H2_vol_pct"] - B["H2_vol_pct"],
            "dH2_vol_pct_drycoal_const": D["H2_vol_pct"] - B["H2_vol_pct"],
            "A_X_C_pp": A["X_C_pp"], "B_X_C_pp": B["X_C_pp"],
            "C_X_C_pp": C["X_C_pp"], "D_X_C_pp": D["X_C_pp"],
            "A_T_out_K": A["T_out_K"], "B_T_out_K": B["T_out_K"],
            "C_T_out_K": C["T_out_K"], "D_T_out_K": D["T_out_K"],
            "drycoal_stage_constant": (1.00875 if ph == "phase2"
                                       else (0.99132 if ph == "phase3" else 1.0)),
            "dT_out_K_swap_hv_part": B["T_out_K"] - C["T_out_K"],
            "dT_out_K_swap_composition_part": (B["T_out_K"] - A["T_out_K"])
                                              - (B["T_out_K"] - C["T_out_K"]),
            "dX_C_pp_swap_hv_part": B["X_C_pp"] - C["X_C_pp"],
            "dX_C_pp_swap_composition_part": (B["X_C_pp"] - A["X_C_pp"])
                                             - (B["X_C_pp"] - C["X_C_pp"]),
            "coal_composition_changed": bool(ph in ("phase2", "phase3")),
            "n_ref_windows": A["n"],
        }
        decomp.append(rec)
        for tag in ("dX_C_pp_total", "dT_out_K_total", "dCO_vol_pct_total",
                    "dH2_vol_pct_total", "dX_C_pp_coal_swap", "dX_C_pp_hv_only",
                    "dX_C_pp_drycoal_const", "dT_out_K_coal_swap", "dT_out_K_hv_only",
                    "dT_out_K_drycoal_const", "dCO_vol_pct_hv_only",
                    "dH2_vol_pct_hv_only", "dT_out_K_swap_hv_part",
                    "dT_out_K_swap_composition_part", "dX_C_pp_swap_hv_part",
                    "dX_C_pp_swap_composition_part"):
            rows.append({
                "reading": "B7", "metric": tag + "_U1_to_main", "phase": ph,
                "value": rec[tag], "threshold": "", "pass": "",
                "note": ("拆分口径：A=legacy+旧窗（U1）→ B=current+旧窗（煤质互换，含热值）→ "
                         "C=lims_heat+旧窗（只换热值）→ D=current+新窗（主口径）；"
                         "各差值按该步替换前后之差，参照窗中位，ξ 沿用主口径各阶段常数"
                         + ("；注意：本阶段煤质成分未变，A→B 与 B→C 是同一热值效应的两半"
                            if ph in ("phase1", "phase4") else "")),
            })
    _write_csv(pd.DataFrame(decomp), "b7_decomposition.csv")
    out["decomposition"] = decomp
    out["states"] = vals
    return rows, out


# ---------------------------------------------------------------------------
# protocol / interpretation / manifest
# ---------------------------------------------------------------------------

def _protocol(args, extra: dict) -> dict:
    return {
        "round": "P2.2-A-20261118",
        "card_version": "第 2 版（2026-11-18，P2.2-D 合并后修订）",
        "source_commit": args.source_commit or _git_source_commit(ROOT),
        "baseline_commit": "db56ebe",
        "coal_set": args.coal_set,
        "coal_set_files": {k: _rel(v) for k, v in coal_paths(args.coal_set).items()},
        "coal_set_digest": coal_digest(args.coal_set),
        "coal_heating_value_dry_kj_kg": coal_hv(args.coal_set),
        "windows_csv": _rel(Path(args.windows_csv)),
        "windows_sha256": _CTX["windows_sha"],
        "expected_windows_table": PAIRING[args.coal_set],
        "expected_windows_sha256": WINDOWS_SHA[PAIRING[args.coal_set]],
        "p22d_readings_ref": ("papers/jax-gasifier-cascade-hybrid/revision-evidence-20261118"
                              "/P2.2-D/readings.csv"),
        "p22d_readings_sha256": (_sha256(P22D_READINGS) if P22D_READINGS.exists() else "missing"),
        "frozen": {
            "o2_flow_scale": 1.0, "GasificationRateFactor": 1.0, "CharCO2RateFactor": 1.0,
            "hooks": "P2.0-A 两钩子关闭", "MeshFamily": "wsr_plus_pfr",
            "WSRLength_m": p20.L_WSR, "PFRCellLength_m": p20.DZ_DOWN,
            "n_cells": p20.mesh_n(), "HeatLossModel": "total_percent",
            "HeatLossPercentTotal": MAIN_HL, "carrier": "实测载气 C-m",
            "coal_flow": "逐窗协调干煤量", "kinetics": "Wen 1979 常数",
            "particle_size_model": "fortran_constant", "OxygenMixingModel": "instant",
        },
        "definitions": {
            "perturbation": "±3%（ln 中心差分，D_LN = ln1.03 − ln0.97）",
            "beta_ln": "per3 / LN_P，LN_P = ln(1.03)",
            "per3": "p20.fd_stats 的 d2（该输入 +3% 的观测量变化）",
            "e_coal": "1 + (per3_coal / X_C_level) / LN_P（p20.elasticities）",
            "e_o2": "(per3_O2 / X_C_level) / LN_P",
            "X_C_level": "该阶段 142 窗全窗中位 X_C",
            "xi": "每阶段一个常数，使主口径参照窗 CO2 中位 = 装置中位；扰动与敏感性沿用",
            "B1_model_level": "该阶段 142 窗全窗中位 X_C",
            "B1_P1_rule": f"阶段 1 的 FINAL verdict=descriptive_not_d6_validated，改判模型 ≥ {B1_P1_MIN}%",
            "B2_rule": "T_out ≥ FT 通过；FT−50 K ≤ T_out < FT 边缘；其余不通过",
            "B3_rule": f"8 个有符号组合中同号 ≥ {B3_MIN_SIGNED} 通过",
            "B3_check": "阶段 2 新窗口表重算装置斜率，任一 beta_ln 偏差 > 1e-6 停止",
            "B6_o2_multiplier": "阶段中位 X_C < 99% 时，参照窗中位 X_C=99% 的 o2_flow_scale 倍数",
            "B7_decomposition": "逐项替换 A→B→C，B→D；D 为主口径；T_out/X_C 与 ξ 无关，组成用主口径 ξ",
        },
        "thresholds": {
            "U1_rel": 1e-9, "U2_ratio": 10.0, "B1_z": B1_Z_MAX,
            "B1_P1_min_pct": B1_P1_MIN, "B2_margin_K": B2_MARGIN_K,
            "B3_min_signed": B3_MIN_SIGNED, "B6_ceil_pp": B6_CEIL_PP,
            "B6_target_pp": B6_TARGET_PP,
            "main_phase_fail_stop": p20.FAIL_WINDOW_STOP,
        },
        "liu_reference": {"FT_C": LIU_FT_C, "T25_C": LIU_T25_C},
        "stage": args.stage, "n_workers": int(args.n_workers),
        **extra,
    }


def _interpretation(unit: pd.DataFrame, x: dict, verdict: str, n_failures: int,
                    halted: str = "") -> str:
    L = ["# P2.2-A 现有模型在测量不确定度尺度上的重新基线", ""]
    L.append("口径：κ = 1、钩子关闭、Wen 1979 动力学、逐窗协调干煤量、"
             "实测载气 C-m、热损 1%。本轮不反演、不加钩子、不改合同。")
    L.append("")
    if halted:
        L.append("> **已触发卡面停止条件，总判定未给出。** " + halted)
        L.append("")
    L.append("## 第 0 节与复现检查")
    L.append("")
    for _, r in unit.iterrows():
        p = "通过" if r["pass"] is True else ("不通过" if r["pass"] is False else "仅记录")
        L.append(f"- `{r['check']}` · {r['item']}：{p}（值 {r['value']}，阈值 {r['threshold']}）")
    L.append("")
    L.append("## B1 碳转化率（142 窗全窗中位 vs FINAL 目标）")
    L.append("")
    L.append("| 阶段 | 模型中位 % | FINAL 目标 % | 1σ pp | z | 判定 |")
    L.append("|---|---:|---:|---:|---:|---|")
    for ph in PHASES:
        s = x["b1"]["phases"].get(ph, {})
        L.append(f"| {PHASE_LABEL[ph]} | {s.get('model_median_pct', float('nan')):.4f} | "
                 f"{s.get('target_midpoint_pct', 'n/a')} | "
                 f"{s.get('target_uncertainty_pct', 'n/a')} | "
                 f"{s.get('z', 'n/a')} | {'通过' if x['b1']['verdict'].get(ph) else '不通过'} |")
    L.append("")
    L.append("## B2 出口温度（主口径参照窗中位）")
    L.append("")
    L.append("| 阶段 | T_out °C | FT °C | T25 °C | 判定 |")
    L.append("|---|---:|---:|---:|---|")
    for ph in PHASES:
        d = x["b2"]["detail"].get(ph, {})
        L.append(f"| {PHASE_LABEL[ph]} | {d.get('T_out_C_median', float('nan')):.2f} | "
                 f"{d.get('FT_C')} | {d.get('T25_C')} | {d.get('verdict')} |")
    L.append("")
    L.append("## B3 响应方向（8 个有符号组合）")
    L.append("")
    L.append("| 阶段 | 观测量 | 输入 | 模型 per3 | 装置 per3 | 同号 |")
    L.append("|---|---|---|---:|---:|---|")
    for c in x["b3"]["combos"]:
        L.append(f"| {PHASE_LABEL[c['phase']]} | {c['obs']} | {c['input']} | "
                 f"{c['model_per3']:.4f} | {c['plant_per3']:.4f} | {c['same_sign']} |")
    L.append("")
    L.append(f"同号 {x['b3']['n_same']}/8。phase2 装置斜率核对：新窗口表最大 |Δbeta_ln| = "
             f"{x['b3']['recalc_max_dev']:.3e}（旧窗口表 {x['b3']['recalc_max_dev_old_table']:.3e}）。")
    L.append("")
    L.append("## B4 组成水平（ξ 闭合后 142 窗中位偏差，模型−装置）")
    L.append("")
    L.append("| 阶段 | CO pp | H2 pp | CH4 pp |")
    L.append("|---|---:|---:|---:|")
    for ph in PHASES:
        r = x["b4"].get(ph, {})
        L.append(f"| {PHASE_LABEL[ph]} | {r.get('CO', {}).get('median_bias', float('nan')):.3f} | "
                 f"{r.get('H2', {}).get('median_bias', float('nan')):.3f} | "
                 f"{r.get('CH4', {}).get('median_bias', float('nan')):.3f} |")
    L.append("")
    L.append("## B5 弹性（模型 vs P2.0-A 装置 95% 区间）")
    L.append("")
    L.append("| 阶段 | e_煤 模型 | 装置区间 | e_氧 模型 | 装置区间 |")
    L.append("|---|---:|---|---:|---|")
    for ph in PHASES:
        r = x["b5"].get(ph, {})
        c, o = r.get("e_coal", {}), r.get("e_o2", {})
        L.append(f"| {PHASE_LABEL[ph]} | {c.get('model', float('nan')):.3f} | "
                 f"[{c.get('lo', float('nan')):.3f}, {c.get('hi', float('nan')):.3f}] | "
                 f"{o.get('model', float('nan')):.3f} | "
                 f"[{o.get('lo', float('nan')):.3f}, {o.get('hi', float('nan')):.3f}] |")
    L.append("")
    L.append("## B6 封顶")
    L.append("")
    L.append("| 阶段 | X_C≥99.9% 基准 | 含扰动 | 阶段中位 X_C % | 达 99% 所需氧量倍数 |")
    L.append("|---|---:|---:|---:|---:|")
    for ph in PHASES:
        r = x["b6"].get(ph, {})
        L.append(f"| {PHASE_LABEL[ph]} | {r.get('frac_ge_99.9_nom', float('nan')):.3f} | "
                 f"{r.get('frac_ge_99.9_any_pert', float('nan')):.3f} | "
                 f"{r.get('median_X_C_pp', float('nan')):.3f} | "
                 f"{r.get('o2_multiplier_for_99', float('nan')):.3f} |")
    L.append("")
    L.append("## B7 热值影响与 U1→主口径拆分")
    L.append("")
    L.append("| 阶段 | ΔT_out（刘文−化验表）K | ΔX_C pp | ΔCO pp | ΔH2 pp |")
    L.append("|---|---:|---:|---:|---:|")
    for ph in PHASES:
        r = x["b7"].get("hv_effect_main", {}).get(ph, {})
        L.append(f"| {PHASE_LABEL[ph]} | {r.get('dT_out_K', float('nan')):.1f} | "
                 f"{r.get('dX_C_pp', float('nan')):.3f} | "
                 f"{r.get('dCO_vol_pct', float('nan')):.3f} | "
                 f"{r.get('dH2_vol_pct', float('nan')):.3f} |")
    L.append("")
    L.append("| 阶段 | ΔT_out 合计 | = 煤质互换（含热值） | + 干煤量常数 | 其中热值 | 其中组成余量 |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for r in x["b7"].get("decomposition", []):
        L.append(f"| {PHASE_LABEL[r['phase']]} | {r['dT_out_K_total']:.1f} | "
                 f"{r['dT_out_K_coal_swap']:.1f} | {r['dT_out_K_drycoal_const']:.1f} | "
                 f"{r['dT_out_K_swap_hv_part']:.1f} | "
                 f"{r['dT_out_K_swap_composition_part']:.1f} |")
    L.append("")
    L.append("说明：「煤质互换（含热值）」= B−A；「干煤量常数」= D−B，两者相加等于 D−A。"
             "「其中热值」= B−C（同一 current 组成下把刘文热值换成化验表值）；"
             "「其中组成余量」= (B−A)−(B−C)，含 legacy 化验表值 28094 与 current 化验表值 27976 "
             "之间约 10 K 的未单列残差，故真实组成项略大于此值。"
             "阶段 1、4 的煤质成分未变，A→B 与 B→C 是同一热值效应的两半。")
    L.append("")
    L.append("## 判定")
    L.append("")
    L.append(f"**{verdict}**")
    L.append("")
    L.append(f"求解失败记录见 `solve_failures.csv`，共 {n_failures} 条。")
    L.append("")
    return "\n".join(L)


def write_manifest() -> Path:
    script = Path(__file__).resolve()
    products = ["unit_checks.csv", "baseline_142.csv", "model_slopes.csv",
                "elasticities.csv", "sensitivity_ref9.csv", "profiles.csv",
                "readings.csv", "protocol.json", "interpretation.md", "xi.json",
                "solve_failures.csv", "b7_decomposition.csv", "b3_check_detail.csv",
                "phase_status.json", "unit_u2_closures.csv"]
    coal_files = []
    for s in ("current", "legacy_20261118"):
        coal_files += sorted((COAL_INPUT_DIR / s).glob("*.yaml"))
    coal_files += sorted(LIMS_HEAT_DIR.glob("*.yaml"))
    paths = [script, P22D_CSV, P114D_CSV] + coal_files + [OUT_DIR / p for p in products]
    lines = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            lines.append(f"{'0' * 64}  {_rel(p)}  (missing)")
        else:
            lines.append(f"{_sha256(p)}  {_rel(p)}")
    dest = OUT_DIR / "sha256.txt"
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest


def _b3_root_cause(b3: dict) -> dict:
    """B3 核对失败的成因诊断（三件事，都与冻结项无关）。

    1. 阶段常数在窗口表里是 4 位小数的**比值**，不是精确常数：coal_new/coal_old
       逐窗不同（相对离散 ~2e-6）。回归设计矩阵条件数 ~3e4，这一点离散就足以把
       beta_ln 推动 1e-3 量级（远超 1e-6）。
    2. `X_C_pp` 与 `dry_per_coal_mol_kg` 是**按煤量归一化**的比值型观测量：
       煤量乘常数 c 会使观测量除以 c，斜率同比缩放 —— 这不是「只改截距」。
    3. 装置侧 `xc_recon_phase_cal_measured_pct` 也随煤质集合改变（不是同一条曲线）。
    """
    new = pd.read_csv(P22D_CSV).set_index("window_id")
    old = pd.read_csv(P114D_CSV).set_index("window_id")
    out: dict = {"phases": {}, "phase2_slope_scale": {}, "phase2_rounding_only": {}}
    for ph in PHASES:
        ai = new[new["phase"] == PHASE_LABEL[ph]]
        bi = old.loc[ai.index]
        c = ai["coal_dry_recon_phase_kg_s"] / bi["coal_dry_recon_phase_kg_s"]
        out["phases"][ph] = {
            "n": int(len(ai)),
            "coal_ratio_mean": float(c.mean()),
            "coal_ratio_rel_std": float(c.std() / c.mean()),
            "coal_ratio_n_unique": int(c.nunique()),
            "w_c_old": float(bi["w_c"].iloc[0]), "w_c_new": float(ai["w_c"].iloc[0]),
            "xc_median_old": float(bi["xc_recon_phase_cal_measured_pct"].median()),
            "xc_median_new": float(ai["xc_recon_phase_cal_measured_pct"].median()),
        }

    # 2：phase2 上比值型观测量的斜率缩放（实测 vs 由煤量/煤质预测）
    det = pd.DataFrame(b3["detail"])
    c2 = out["phases"]["phase2"]["coal_ratio_mean"]
    wratio = out["phases"]["phase2"]["w_c_new"] / out["phases"]["phase2"]["w_c_old"]
    for obs, pred in (("dry_per_coal_mol_kg", 1.0 / c2),
                      ("X_C_pp", 1.0 / (c2 * wratio))):
        rows = det[det["obs"] == obs]
        if not len(rows):
            continue
        out["phase2_slope_scale"][obs] = {
            "predicted_multiplier": float(pred),
            "observed_multipliers": [float(r["beta_recalc_new_table"]
                                           / r["beta_stored_p115a"])
                                     for _, r in rows.iterrows()],
        }

    # 3：仅由 4 位小数舍入造成的斜率变化（不涉及观测量重算）
    a = new[new["phase"] == PHASE_LABEL["phase2"]]
    b = old[old["phase"] == PHASE_LABEL["phase2"]]
    syn = b.copy()
    syn["coal_dry_recon_phase_kg_s"] = syn["coal_dry_recon_phase_kg_s"] * c2

    def _slopes(tab: pd.DataFrame, col: str) -> np.ndarray:
        X = np.column_stack([
            np.ones(len(tab)),
            np.log(tab["o2_fy1103a_nm3_h_mean"].to_numpy(float)),
            np.log(tab["coal_dry_recon_phase_kg_s"].to_numpy(float)),
            np.log(tab["co2_carrier_measured_nm3_h"].to_numpy(float)),
        ])
        y = tab[col].to_numpy(float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return beta

    for col in ("co_vol_pct_mean", "h2_vol_pct_mean", "co2_vol_pct_mean",
                "ch4_vol_pct_mean", "xc_recon_phase_cal_measured_pct"):
        bo = _slopes(b, col)
        bn = _slopes(a, col)
        bs = _slopes(syn, col)
        out["phase2_rounding_only"][col] = {
            "max_abs_dev_exact_constant": float(np.max(np.abs(bs[1:] - bo[1:]))),
            "max_abs_dev_actual_table": float(np.max(np.abs(bn[1:] - bo[1:]))),
        }
    return out



def stage_read(args) -> None:
    prepare_coal_inputs()
    require_pairing(args.windows_csv, args.coal_set, "read")
    init_ctx(Path(args.windows_csv))
    _CTX["main_coal_set"] = args.coal_set
    _record_failures([])  # 零失败也要留下带表头的 solve_failures.csv
    unit = pd.read_csv(OUT_DIR / "unit_checks.csv")
    sens_path = OUT_DIR / "sensitivity_ref9.csv"
    sens = pd.read_csv(sens_path) if sens_path.exists() else None

    b1_rows, b1 = _b1()
    b2_rows, b2 = _b2(sens)
    b3_rows, b3 = _b3()
    b4_rows, b4 = _b4()
    b5_rows, b5 = _b5()
    b6_rows, b6 = _b6(args)
    b7_rows, b7 = _b7()

    # 2026-11-18 审查：某阶段求解失败超过卡面阈值时，`stage_compare` 已在
    # phase_status.json / model_slopes.csv 里标了 phase_skipped，但读法阶段此前
    # 不消费该标记 —— 于是被跳过的阶段照样拿 B1/B2/B3 判定，还会计入
    # verdict_if_check_ignored。求解失败不能静默地替我们挑选样本。
    skipped_phases = _skipped_phases()
    if skipped_phases:
        b1_rows, b1 = _mask_skipped(b1_rows, b1, skipped_phases, "B1")
        b2_rows, b2 = _mask_skipped(b2_rows, b2, skipped_phases, "B2")
        b3_rows, b3 = _mask_skipped_b3(b3_rows, b3, skipped_phases)
        all_rows_skipped = [{
            "reading": "SKIPPED", "metric": "phase_skipped", "phase": ph,
            "value": 1, "threshold": p20.FAIL_WINDOW_STOP, "pass": False,
            "note": (f"该阶段求解失败窗口数超过卡面阈值 "
                     f"{p20.FAIL_WINDOW_STOP}，按卡面停止条件跳过该阶段的判定；"
                     "B1/B2/B3 对该阶段不给通过，且不计入可判定的阶段数"),
        } for ph in skipped_phases]
    else:
        all_rows_skipped = []

    all_rows = (all_rows_skipped + b1_rows + b2_rows + b3_rows
                + b4_rows + b5_rows + b6_rows + b7_rows)
    b1_ok = all(b1["verdict"].get(ph) for ph in PHASES)
    b2_main = {ph: v for (ph, g), v in b2["verdict"].items() if g == "main"}
    b2_ok = all(v != "fail" for v in b2_main.values())
    b3_ok = bool(b3["n_same"] >= B3_MIN_SIGNED) and not skipped_phases
    halted = bool(b3["recalc_max_dev"] > 1e-6) or bool(skipped_phases)

    if halted:
        # 卡面停止条件：B3 装置斜率核对不通过 → 不给总判定，仍产出可核查的读数与协议
        reason = (f"B3 装置斜率核对不通过：phase2 用新窗口表重算的最大 |Δbeta_ln| = "
                  f"{b3['recalc_max_dev']:.6e} > 1e-6（同一函数用旧窗口表重算时为 "
                  f"{b3['recalc_max_dev_old_table']:.3e}）")
        all_rows.append({
            "reading": "HALT", "metric": "stop_condition", "phase": "phase2",
            "value": b3["recalc_max_dev"], "threshold": 1e-6, "pass": False,
            "note": reason + "；按卡面停止条件不给总判定，B1–B7 数值照常列出",
        })
        _write_csv(pd.DataFrame(all_rows), "readings.csv")
        extra = {
            "b1": b1,
            "b2": {"detail": b2["detail"], "verdict_main": b2_main,
                   "verdict_all": {f"{k[0]}|{k[1]}": v for k, v in b2["verdict"].items()}},
            "b3": b3, "b4": b4, "b5": b5, "b6": b6, "b7": b7,
            "halted": True, "halt_reason": reason,
            "b3_check_max_abs_dev": b3["recalc_max_dev"],
            "b3_check_max_abs_dev_old_table": b3["recalc_max_dev_old_table"],
            "b3_root_cause": _b3_root_cause(b3),
            "verdict_withheld": True,
            "verdict_if_check_ignored": (
                "在当前数据精度下现有模型与装置一致"
                if (b1_ok and b2_ok and b3_ok) else "不一致"),
            "n_solve_failures": 0,
        }
        fail_path = OUT_DIR / "solve_failures.csv"
        if fail_path.exists():
            extra["n_solve_failures"] = int(len(pd.read_csv(fail_path)))
        (OUT_DIR / "protocol.json").write_text(
            json.dumps(_protocol(args, extra), ensure_ascii=False, indent=2,
                       default=float) + "\n", encoding="utf-8")
        (OUT_DIR / "interpretation.md").write_text(
            _interpretation(unit, extra, "HALTED（B3 核对未通过）",
                            extra["n_solve_failures"], halted=reason),
            encoding="utf-8")
        write_manifest()
        print(f"[read] HALT: {reason}", flush=True)
        raise SystemExit(f"STOP: {reason}；产物已写出，未给总判定。")

    overall = b1_ok and b2_ok and b3_ok
    fails = []
    if not b1_ok:
        fails += [f"B1 {PHASE_LABEL[ph]}" for ph in PHASES if not b1["verdict"].get(ph)]
    if not b2_ok:
        fails += [f"B2 {PHASE_LABEL[ph]}" for ph, v in b2_main.items() if v == "fail"]
    if not b3_ok:
        fails.append(f"B3 同号 {b3['n_same']}/8 < {B3_MIN_SIGNED}")
    verdict = ("在当前数据精度下现有模型与装置一致" if overall
               else "不一致：" + "，".join(fails))

    n_b2_ok = sum(1 for v in b2["verdict"].values() if v != "fail")
    all_rows.append({
        "reading": "VERDICT", "metric": "overall", "phase": "all",
        "value": "consistent" if overall else "inconsistent", "threshold": "",
        "pass": bool(overall),
        "note": (f"B1 {sum(bool(v) for v in b1['verdict'].values())}/4 通过、"
                 f"B2 主口径 {sum(1 for v in b2_main.values() if v != 'fail')}/4 通过或边缘"
                 f"（含敏感性 {n_b2_ok}/{len(b2['verdict'])}）、"
                 f"B3 {b3['n_same']}/8 同号 → {verdict}"),
    })
    _write_csv(pd.DataFrame(all_rows), "readings.csv")

    fail_path = OUT_DIR / "solve_failures.csv"
    fail_df = pd.read_csv(fail_path) if fail_path.exists() else pd.DataFrame()
    extra = {
        "b1": b1, "b2": {"detail": b2["detail"],
                         "verdict_main": b2_main,
                         "verdict_all": {f"{k[0]}|{k[1]}": v
                                         for k, v in b2["verdict"].items()}},
        "b3": b3, "b4": b4, "b5": b5, "b6": b6, "b7": b7,
        "halted": False, "verdict": verdict, "verdict_pass": bool(overall),
        "b3_check_max_abs_dev": b3["recalc_max_dev"],
        "b3_check_max_abs_dev_old_table": b3["recalc_max_dev_old_table"],
        "n_solve_failures": int(len(fail_df)),
    }
    (OUT_DIR / "protocol.json").write_text(
        json.dumps(_protocol(args, extra), ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8")
    (OUT_DIR / "interpretation.md").write_text(
        _interpretation(unit, extra, verdict, len(fail_df)),
        encoding="utf-8")
    write_manifest()
    print(f"[read] verdict: {verdict}", flush=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="P2.2-A 重新基线（2026-11-18 第 2 版）")
    ap.add_argument("--coal-set", required=True,
                    choices=("current", "legacy_20261118", "lims_heat"))
    ap.add_argument("--windows-csv", required=True)
    ap.add_argument("--stage", default="all",
                    choices=("unit", "compare", "sens", "profiles", "read", "all"))
    ap.add_argument("--n-workers", type=int, default=13)
    ap.add_argument("--source-commit", default=None)
    args = ap.parse_args()

    silence_solver_logs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    init_ctx(Path(args.windows_csv))

    if args.stage == "unit":
        stage_unit(args)
    elif args.stage == "compare":
        stage_compare(args)
    elif args.stage == "sens":
        stage_sens(args)
    elif args.stage == "profiles":
        stage_profiles(args)
    elif args.stage == "read":
        stage_read(args)
    else:
        stage_unit(args)
        stage_compare(args)
        stage_sens(args)
        stage_profiles(args)
        stage_read(args)


if __name__ == "__main__":
    main()




