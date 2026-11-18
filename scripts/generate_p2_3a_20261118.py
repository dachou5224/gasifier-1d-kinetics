#!/usr/bin/env python3
"""P2.3-A：新基线上的焦炭转化受限机理比较（2026-11-18）。

依据：论文仓任务卡 `manuscript/model-side-handoff-p2.3a-20261118.md`；
判据来源：`manuscript/model-evidence-verification-20261118-p22a.md`。

**本轮不做多参数反演，不加回流，不改合同。**

冻结项与 P2.2-A 主口径一致（κ = 1、wsr_plus_pfr、L_wsr = 1 m、dz = 0.0515 m、
HeatLossModel = total_percent、HeatLossPercentTotal = 1、实测载气、Wen 1979 常数）。
除被测候选的钩子外，其余钩子全部关闭。

候选（第 1 节）：

    C0             现有模型（对照，不参与求根）
    F1             整体气化反应性倍数 f：GasificationRateFactor = f，log10 f ∈ [-4, 0]
    F2-50/100/150  参考温度反应性 f_ref + 活化能增量 ΔE：
                   k = k_Wen(T_p)·f_ref·exp[-(ΔE/R)(1/T_p - 1/T_ref)]
                   log10 f_ref ∈ [-4, 1]，ΔE = 50/100/150 kJ/mol，T_ref = 1600 K
                   新钩子 CharGasActivationIncrement_kJmol（缺省 0）、CharGasTref_K（缺省 1600）
    M1-0           后期反应性衰减 X0 = 0，CharLateDecayExponent = n ∈ [0, 8]
    M2             渣层截留，SlagCaptureFraction = f_s ∈ [0, 0.5]

求根（第 2 节，含 P2.2-A B6 的教训）：括区自适应搜索；端点/中间点无解时向可解方向
收缩并记录；找不到可解括区或端点达不到目标时记「不可达」并写原因，**不输出裸 NaN**。

读法（第 4 节）：H0–H7，跑完不改。

用法：

    ./.venv/bin/python scripts/generate_p2_3a_20261118.py \\
        --coal-set current \\
        --windows-csv outputs/diagnostics/p2.2d-20261118/windows_142.csv \\
        --stage all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

from model.input_contract import coal_hhv_kj_kg  # noqa: E402
from model.p1_5_round_20261030 import run_case, silence_solver_logs  # noqa: E402

OUT_DIR = ROOT / "outputs" / "diagnostics" / "round-p2.3a-20261118"
CACHE_DIR = OUT_DIR / "cache"

P22D_DIR = ROOT / "outputs" / "diagnostics" / "p2.2d-20261118"
# P2.2-D 的 readings.csv 只在论文仓（P2.2-A 同口径引用）
P22D_READINGS = p22a.P22D_READINGS
P115A_DIR = ROOT / "outputs" / "diagnostics" / "round-p1.15a-20261108"
P20A_DIR = ROOT / "outputs" / "diagnostics" / "round-p2.0a-20261113"
P22A_DIR = ROOT / "outputs" / "diagnostics" / "round-p2.2a-20261118"

PHASES = ("phase1", "phase2", "phase3", "phase4")
PHASE_LABEL = {"phase1": "P1", "phase2": "P2", "phase3": "P3", "phase4": "P4"}

MAIN_HL = 1.0
MAIN_CARRIER = "C-m"
CARRIERS = p22a.CARRIERS
PERT_SCALE = p20.PERT_SCALE
PERTS = p20.PERTS

# 第 4 节预声明阈值
H1_MIN_PHASES = 3
H3_MIN_SIGNED = 6
H4_FT_MARGIN_K = 50.0
H4_MIN_AT_FT = 3
H5_MAX_DEV_PP = 2.0

# 刘文表 1 的煤灰流动温度 FT（°C）
LIU_FT_C = {"phase1": 1280.0, "phase2": 1290.0, "phase3": 1270.0, "phase4": 1250.0}

# 阶段 1 无有效协调目标，卡面指定用刘文表 2（第 2 节）
P1_TARGET_PP = 98.24

# 求根
ROOT_TOL_PP = 0.2
BISECT_MAX = 30
INTERIOR_PROBE = 4

# 装置弹性 95% 区间（P2.0-A elasticities.csv，C0 行的 *_plant_lo/hi）
PLANT_E_INTERVALS = {
    "phase1": {"coal": (-0.42, 0.20), "o2": (0.53, 1.31)},
    "phase2": {"coal": (0.32, 0.62), "o2": (0.13, 1.13)},
    "phase3": {"coal": (0.19, 0.57), "o2": (-0.10, 0.62)},
    "phase4": {"coal": (0.20, 0.73), "o2": (1.16, 2.06)},
}

# H2：装置单位煤量干气量对氧量斜率有符号的阶段（P1.15-A，区间不含 0）
H2_PHASES = ("phase1", "phase2", "phase4")

R_GAS = 8.3144626
T_REF_K = 1600.0


# ---------------------------------------------------------------------------
# 候选定义
# ---------------------------------------------------------------------------
# direction = +1 表示「自由量增大 → X_C 增大」；-1 反之。
CANDIDATES: list[dict] = [
    {"id": "C0", "kind": "C0", "label": "现有模型（对照）",
     "u_lo": 0.0, "u_hi": 0.0, "direction": 0, "param_name": "none"},
    {"id": "F1", "kind": "F1", "label": "整体反应性倍数 f",
     "u_lo": -4.0, "u_hi": 0.0, "direction": +1, "param_name": "log10_f"},
    {"id": "F2-50", "kind": "F2", "label": "f_ref + ΔE=50 kJ/mol",
     "u_lo": -4.0, "u_hi": 1.0, "direction": +1, "param_name": "log10_f_ref",
     "delta_e": 50.0},
    {"id": "F2-100", "kind": "F2", "label": "f_ref + ΔE=100 kJ/mol",
     "u_lo": -4.0, "u_hi": 1.0, "direction": +1, "param_name": "log10_f_ref",
     "delta_e": 100.0},
    {"id": "F2-150", "kind": "F2", "label": "f_ref + ΔE=150 kJ/mol",
     "u_lo": -4.0, "u_hi": 1.0, "direction": +1, "param_name": "log10_f_ref",
     "delta_e": 150.0},
    {"id": "M1-0", "kind": "M1", "label": "后期衰减 X0=0",
     "u_lo": 0.0, "u_hi": 8.0, "direction": -1, "param_name": "n", "x0": 0.0},
    {"id": "M2", "kind": "M2", "label": "渣层截留 f_s",
     "u_lo": 0.0, "u_hi": 0.5, "direction": -1, "param_name": "f_s"},
]
CAND = {c["id"]: c for c in CANDIDATES}
FIT_CANDS = tuple(c["id"] for c in CANDIDATES if c["kind"] != "C0")


def hook_op(cand_id: str, u: float) -> dict:
    """候选自由量 → op_conds 钩子字典。F1/F2 的自由量是 log10。"""
    c = CAND[cand_id]
    kind = c["kind"]
    if kind == "C0":
        return {}
    if kind in ("F1", "F2"):
        op = {"GasificationRateFactor": float(10.0 ** float(u))}
        if kind == "F2":
            op["CharGasActivationIncrement_kJmol"] = float(c["delta_e"])
            op["CharGasTref_K"] = float(T_REF_K)
        return op
    if kind == "M1":
        return {"CharLateDecayX0": float(c["x0"]),
                "CharLateDecayExponent": float(u)}
    if kind == "M2":
        return {"SlagCaptureFraction": float(u)}
    raise ValueError(f"unknown candidate kind {kind!r}")


def param_report(cand_id: str, u: float | None) -> tuple[float | None, str]:
    """人类可读参数值（comparison_points.csv）。"""
    c = CAND[cand_id]
    if u is None:
        return None, "未求根"
    if c["kind"] == "C0":
        return 1.0, "（无自由量）"
    if c["kind"] in ("F1", "F2"):
        f = float(10.0 ** float(u))
        if c["kind"] == "F1":
            return f, f"f={f:.6g}"
        return f, f"f_ref={f:.6g}, ΔE={c['delta_e']:.0f} kJ/mol, T_ref={T_REF_K:.0f} K"
    return float(u), f"{c['param_name']}={u:.6g}"


# ---------------------------------------------------------------------------
# 缓存：键含完整钩子取值
# ---------------------------------------------------------------------------
def cache_key(*, window_id: str, coal_set: str, carrier_tag: str, hl: float,
              pert: str, dry_coal: float, carrier_nm3: float,
              cand_id: str, hooks: dict[str, float]) -> str:
    return json.dumps({
        "window_id": str(window_id),
        "candidate": str(cand_id),
        "hooks": {k: repr(float(v)) for k, v in sorted(hooks.items())},
        "hl": repr(float(hl)),
        "pert": pert,
        "dry_coal_kg_s": repr(float(dry_coal)),
        "carrier_nm3_h": repr(float(carrier_nm3)),
        "carrier_tag": carrier_tag,
        "coal_set": coal_set, "coal_digest": p22a.coal_digest(coal_set),
        "windows_sha": p22a.table_sha("main"),
        "kappa": repr(float(p20.KAPPA)),
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
        json.dumps({"key": key, "snap": snap}, ensure_ascii=False),
        encoding="utf-8")


# ---------------------------------------------------------------------------
# 单点求解
# ---------------------------------------------------------------------------
def solve(*, window_id: str, coal_set: str, carrier_tag: str, hl: float,
          pert: str, cand_id: str, u: float) -> dict:
    """单窗单点求解。

    X_C 的分子必须含被截留的碳（`c_cap`），否则 M2 会把「截留」记成「转化」——
    那正是 P2.0-A 已修过的 P1 错误。口径与 p20._snapshot 逐项一致。
    """
    df = p22a.table("main")
    o2m, coalm, carm = PERT_SCALE[pert]
    dry = float(df.loc[window_id, "coal_dry_recon_phase_kg_s"]) * float(coalm)
    carrier_nm3 = float(df.loc[window_id, CARRIERS[carrier_tag]]) * float(carm)
    hooks = hook_op(cand_id, u)
    key = cache_key(window_id=window_id, coal_set=coal_set, carrier_tag=carrier_tag,
                    hl=hl, pert=pert, dry_coal=dry, carrier_nm3=carrier_nm3,
                    cand_id=cand_id, hooks=hooks)
    hit = _load_cache(key)
    if hit is not None:
        return hit

    op = p22a.build_op(hl, o2m, dry, carrier_nm3)
    op.update(hooks)
    rec = run_case(p22a.rows_for(coal_set)[window_id],
                   n_cells=p20.mesh_n(), coal_flow_dry_kg_s=dry,
                   keep_system=True, mesh_family="wsr_plus_pfr", extra_op=op)
    system = rec.get("_system")
    snap: dict = {
        "window_id": window_id, "candidate": cand_id, "u": float(u),
        "hooks": hooks, "phase": p22a._phase_of(window_id),
        "carrier_tag": carrier_tag, "hl": float(hl), "pert": pert,
        "coal_dry_kg_s": float(dry), "carrier_nm3_h": float(carrier_nm3),
        "execution_status": str(rec.get("execution_status")),
        "no_root_cell_index": rec.get("no_root_cell_index"),
        "ok": False,
    }
    if system is None or not system.results or \
            rec.get("execution_status") != "completed":
        rec.pop("_system", None)
        _store_cache(key, snap)
        return snap

    cd = float(system.coal_props.get("Cd", 0.0)) / 100.0
    c_in = float(system.W_dry) * cd
    outlet = system.results[-1]
    gas = np.asarray(outlet.gas_moles, dtype=float)
    cap = system.slag_capture
    c_out = float(outlet.solid_mass) * float(outlet.carbon_fraction)
    c_cap = 0.0 if not isinstance(cap, dict) else float(cap.get("carbon_kg_s", 0.0))
    hhv_kj = float(coal_hhv_kj_kg(system.coal_props))
    snap.update({
        "ok": True,
        "X_C_frac": float(1.0 - (c_out + c_cap) / max(c_in, 1e-12)),
        "X_C_pp": float((1.0 - (c_out + c_cap) / max(c_in, 1e-12)) * 100.0),
        "T_out_K": float(outlet.T),
        "T_wsr_K": float(system.results[0].T),
        "F_CO": float(gas[2]), "F_CO2": float(gas[3]), "F_H2": float(gas[5]),
        "F_H2O": float(gas[7]), "F_CH4": float(gas[1]), "F_H2S": float(gas[4]),
        "dry_total_mol_s": float(np.sum(gas[:7])),
        "capture_fraction": 0.0 if not isinstance(cap, dict) else float(cap.get("fraction", 0.0)),
        "capture_carbon_kg_s": c_cap,
        "capture_carbon_frac_of_inlet": float(c_cap / max(c_in, 1e-12)),
        "capture_enthalpy_W": (0.0 if not isinstance(cap, dict)
                               else float(cap.get("enthalpy_W", 0.0))),
        "capture_enthalpy_frac_hhv_pct": (
            0.0 if not isinstance(cap, dict)
            else float(cap.get("enthalpy_W", 0.0)
                       / max(dry * hhv_kj / 1000.0 * 1e6, 1e-12) * 100.0)),
        "capture_solid_mass_kg_s": (0.0 if not isinstance(cap, dict)
                                    else float(cap.get("solid_mass_kg_s", 0.0))),
        "coal_dry_kg_s": float(dry),
    })
    rec.pop("_system", None)
    _store_cache(key, snap)
    return snap


def apply_xi(snap: dict, xi: float) -> dict | None:
    """与 p20.apply_xi 同一变换（ξ mol/s 把 CO+H2O 部分变换为 CO2+H2）。"""
    return p20.apply_xi(snap, xi)


# ---------------------------------------------------------------------------
# 池化求解（串行 5.3 s/次，9 窗一次探点串行要 48 s；池化后约 5 s）
# ---------------------------------------------------------------------------
_POOL: ProcessPoolExecutor | None = None
_WINDOWS_CSV = {"path": ""}


def _winit(windows_csv: str) -> None:
    """worker 初始化。

    **不要**在这里调 prepare_coal_inputs：它会 shutil.copyfile 四套煤质到本轮目录，
    13 个 worker 并发拷贝同一批文件会互相读到半写状态，触发
    「coal assumption copy mismatch」。该函数由父进程在创建池之前执行一次。
    """
    silence_solver_logs()
    p22a.init_ctx(Path(windows_csv))


def _wjob(spec: dict) -> dict:
    return solve(**spec)


def init_pool(windows_csv: str, n_workers: int) -> None:
    global _POOL
    if _POOL is not None:
        return
    # 父进程独占完成煤质拷贝（worker 并发拷贝会产生竞态）
    p22a.prepare_coal_inputs()
    _WINDOWS_CSV["path"] = str(windows_csv)
    _POOL = ProcessPoolExecutor(max_workers=max(1, int(n_workers)),
                                initializer=_winit,
                                initargs=(str(windows_csv),))


def close_pool() -> None:
    global _POOL
    if _POOL is not None:
        _POOL.shutdown(wait=True)
        _POOL = None


def batch_solve(specs: list[dict]) -> list[dict]:
    """并行求解一批 spec（每个 spec 是 solve 的关键字参数）。"""
    if not specs:
        return []
    if _POOL is None:
        return [solve(**sp) for sp in specs]
    return list(_POOL.map(_wjob, specs))


def _specs_for(cand_id: str, u_values: list[float], wids: list[str], *,
               coal_set: str, carrier_tag: str, hl: float,
               pert: str) -> list[dict]:
    out = []
    for u in u_values:
        for w in wids:
            out.append({"window_id": w, "coal_set": coal_set,
                        "carrier_tag": carrier_tag, "hl": hl, "pert": pert,
                        "cand_id": cand_id, "u": float(u)})
    return out


def median_xc_multi(cand_id: str, u_values: list[float], wids: list[str], *,
                    coal_set: str, carrier_tag: str = MAIN_CARRIER,
                    hl: float = MAIN_HL,
                    pert: str = "nom") -> dict[float, tuple[float, int, int]]:
    """一次批量求解多个 u 值的参照窗中位。返回 {u: (中位, n_ok, n_fail)}。"""
    specs = _specs_for(cand_id, u_values, wids, coal_set=coal_set,
                       carrier_tag=carrier_tag, hl=hl, pert=pert)
    snaps = batch_solve(specs)
    nw = len(wids)
    out: dict[float, tuple[float, int, int]] = {}
    for i, u in enumerate(u_values):
        chunk = snaps[i * nw:(i + 1) * nw]
        vals = [float(s["X_C_pp"]) for s in chunk if s.get("ok")]
        n_fail = len(chunk) - len(vals)
        out[float(u)] = (float(np.median(vals)) if vals else float("nan"),
                         len(vals), n_fail)
    return out


def pool_batch(specs: list[dict]) -> list[dict]:
    return batch_solve(specs)


def median_xc(cand_id: str, u: float, wids: list[str], *, coal_set: str,
              carrier_tag: str = MAIN_CARRIER, hl: float = MAIN_HL,
              pert: str = "nom") -> tuple[float, int, int]:
    vals, n_fail = [], 0
    for w in wids:
        s = solve(window_id=w, coal_set=coal_set, carrier_tag=carrier_tag,
                  hl=hl, pert=pert, cand_id=cand_id, u=u)
        if s.get("ok"):
            vals.append(float(s["X_C_pp"]))
        else:
            n_fail += 1
    return (float(np.median(vals)) if vals else float("nan")), len(vals), n_fail


# ---------------------------------------------------------------------------
# 求根（自适应括区，不可达必须给原因）
# ---------------------------------------------------------------------------
def solve_param(cand_id: str, ph: str, wids: list[str], target: float, *,
                coal_set: str, carrier_tag: str = MAIN_CARRIER,
                hl: float = MAIN_HL) -> dict:
    """一维求根：使参照窗 X_C 中位 = target（±0.2 pp）。

    括区自适应：先探两端；夹不住目标则向内扫；端点/中间点无解时向可解方向收缩。
    不可达一律给 reason，不输出裸 NaN。所有探点在每个阶段用批量求解（池化）。
    """
    c = CAND[cand_id]
    out: dict = {
        "candidate": cand_id, "phase": ph, "target_X_C_pp": float(target),
        "reachable": False, "reason": "", "u": None,
        "X_C_pp_median": None, "n_ref": len(wids), "n_solve_fail_at_root": 0,
        "n_probe": 0,
    }

    def probe_many(us: list[float]) -> dict[float, tuple[float, int]]:
        out["n_probe"] += len(us)
        res = median_xc_multi(cand_id, us, wids, coal_set=coal_set,
                              carrier_tag=carrier_tag, hl=hl)
        return {u: (m, nf) for u, (m, _n, nf) in res.items()}

    if c["kind"] == "C0":
        res = probe_many([0.0])
        med, n_fail = res[0.0]
        out.update({"reachable": bool(math.isfinite(med)), "u": 0.0,
                    "X_C_pp_median": float(med) if math.isfinite(med) else None,
                    "n_solve_fail_at_root": n_fail,
                    "reason": "" if math.isfinite(med) else "全部参照窗无解"})
        return out

    hi_u, lo_u = c["u_hi"], c["u_lo"]
    res = probe_many([hi_u, lo_u])
    solvable: list[tuple[float, float]] = []
    if math.isfinite(res[hi_u][0]):
        solvable.append((hi_u, res[hi_u][0]))
    if math.isfinite(res[lo_u][0]):
        solvable.append((lo_u, res[lo_u][0]))

    def bracketed() -> bool:
        ss = sorted(solvable)
        return any((a[1] - target) * (b[1] - target) <= 0.0
                   for a, b in zip(ss, ss[1:]))

    if not bracketed():
        us = [lo_u + (hi_u - lo_u) * k / (INTERIOR_PROBE + 1)
              for k in range(1, INTERIOR_PROBE + 1)]
        res2 = probe_many(us)
        for u in us:
            if math.isfinite(res2[u][0]):
                solvable.append((u, res2[u][0]))
            if bracketed():
                break

    if not solvable:
        out["reason"] = "参数范围内端点与内点均无解（已自适应扫描）"
        return out
    if not bracketed():
        ms = [m for _, m in solvable]
        if min(ms) > target:
            out["reason"] = (f"可解点最小 X_C 中位 {min(ms):.2f}% 仍高于目标 "
                             f"{target:.2f}%，该参数不足以把转化压到目标")
            out["X_C_pp_median"] = float(min(ms))
        elif max(ms) < target:
            out["reason"] = (f"可解点最大 X_C 中位 {max(ms):.2f}% 仍低于目标 "
                             f"{target:.2f}%，该参数不足以把转化提到目标")
            out["X_C_pp_median"] = float(max(ms))
        else:
            out["reason"] = "可解点之间未夹住目标（中间存在无解带）"
        return out

    solvable.sort()
    lo = hi = None
    for (u1, m1), (u2, m2) in zip(solvable, solvable[1:]):
        if (m1 - target) * (m2 - target) <= 0.0:
            lo, hi = u1, u2
            break
    if lo is None:
        out["reason"] = "未能选出括区（内部错误）"
        return out

    best_u, best_m, best_dev = None, None, float("inf")
    direction = c["direction"]
    for _ in range(BISECT_MAX):
        mid = 0.5 * (lo + hi)
        r = probe_many([mid])
        m, n_fail = r[mid]
        if not math.isfinite(m):
            # 中间点无解：向两端各试一点，取可解的一侧收缩
            alt = [lo + (hi - lo) * 0.25, lo + (hi - lo) * 0.75]
            ra = probe_many(alt)
            pick = next(((u, ra[u]) for u in alt if math.isfinite(ra[u][0])), None)
            if pick is None:
                hi = mid
                continue
            u_pick, (m_pick, _nf) = pick
            # 用可解点替换无解中点所在的一侧
            if u_pick < mid:
                hi = mid
            else:
                lo = mid
            continue
        dev = abs(m - target)
        if dev < best_dev:
            best_u, best_m, best_dev = mid, m, dev
        if dev <= ROOT_TOL_PP or (hi - lo) < 1e-4:
            out.update({"reachable": True, "u": float(mid),
                        "X_C_pp_median": float(m), "n_solve_fail_at_root": n_fail})
            return out
        if direction > 0:
            if m < target:
                lo = mid
            else:
                hi = mid
        else:
            if m < target:
                hi = mid
            else:
                lo = mid

    if best_u is not None and best_dev <= ROOT_TOL_PP:
        out.update({"reachable": True, "u": float(best_u),
                    "X_C_pp_median": float(best_m)})
    else:
        out["reason"] = (f"二分未在容差内收敛（最小偏差 {best_dev:.3f} pp > "
                         f"{ROOT_TOL_PP} pp）")
        out["X_C_pp_median"] = None if best_m is None else float(best_m)
    return out


# ---------------------------------------------------------------------------
# ξ（激冷段变换）
# ---------------------------------------------------------------------------
def solve_xi(cand_id: str, ph: str, wids: list[str], *, coal_set: str,
             u: float, hl: float = MAIN_HL) -> tuple[float, float, int]:
    """使参照窗 CO2 中位等于装置值的 ξ。返回 (xi, 闭合后 CO2 中位, n_fail)。"""
    df = p22a.table("main")
    target = float(np.median([float(df.loc[w, "co2_vol_pct_mean"]) for w in wids]))
    specs = [{"window_id": w, "coal_set": coal_set, "carrier_tag": MAIN_CARRIER,
              "hl": hl, "pert": "nom", "cand_id": cand_id, "u": float(u)}
             for w in wids]
    results = batch_solve(specs)
    snaps = [s for s in results if s.get("ok")]
    n_fail = len(results) - len(snaps)
    if not snaps:
        return float("nan"), float("nan"), n_fail

    def co2_at(xi: float) -> float:
        vals = []
        for s in snaps:
            a = apply_xi(s, xi)
            if a is not None:
                vals.append(a["CO2_vol_pct"])
        return float(np.median(vals)) if vals else float("nan")

    # 变换 CO + H2O → CO2 + H2 使 CO2 = (F_CO2+ξ)/(dry+ξ)，**随 ξ 单调递增**。
    # 因此「模型 CO2 低于装置值」正是需要增大 ξ 的情形，不是不可达。
    m0 = co2_at(0.0)
    if not math.isfinite(m0):
        return float("nan"), float("nan"), n_fail
    if m0 >= target:
        # ξ ≥ 0 只会把 CO2 抬得更高，已无法回到目标：记 0 并如实返回实际值
        return 0.0, m0, n_fail
    lo, hi = 0.0, 50.0
    m_hi = co2_at(hi)
    for _ in range(20):
        if math.isfinite(m_hi) and m_hi >= target:
            break
        hi *= 1.5
        m_hi = co2_at(hi)
    if not (math.isfinite(m_hi) and m_hi >= target):
        return float("nan"), float("nan"), n_fail
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        m = co2_at(mid)
        if not math.isfinite(m):
            break
        if abs(m - target) < 1e-6 or (hi - lo) < 1e-8:
            return float(mid), float(m), n_fail
        if m < target:
            lo = mid
        else:
            hi = mid
    mid = 0.5 * (lo + hi)
    return float(mid), float(co2_at(mid)), n_fail


# ---------------------------------------------------------------------------
# 阶段 unit
# ---------------------------------------------------------------------------
def stage_unit(args) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p22a.prepare_coal_inputs()
    p22a.require_pairing(args.windows_csv, args.coal_set, "compare")
    p22a.init_ctx(Path(args.windows_csv))
    rows: list[dict] = []
    hard_fail: list[str] = []

    hv = p22a.coal_hv(args.coal_set)
    exp = p22a.HV_EXPECT[args.coal_set]
    bad_hv = {p: [hv[p], exp[p]] for p in PHASES if hv[p] != exp[p]}
    rows.append({"check": "PAIRING", "item": "第 0 节：配对断言 + 四阶段热值",
                 "value": ";".join(f"{PHASE_LABEL[p]}={hv[p]:.0f}" for p in PHASES),
                 "threshold": "a24b5fe6… / 28179-28440-28579-28588", "pass": not bad_hv,
                 "detail": json.dumps(bad_hv, ensure_ascii=False) if bad_hv else "均符合"})
    if bad_hv:
        hard_fail.append("PAIRING")

    ref = p22a._ref_windows("main")
    w4 = ref["phase4"][0]
    base = pd.read_csv(P22A_DIR / "baseline_142.csv")

    # U1：钩子缺省 → 与 P2.2-A 主口径 C0 逐位一致
    worst, worst_w = 0.0, None
    for ph in PHASES:
        for w in ref[ph]:
            s = solve(window_id=w, coal_set=args.coal_set, carrier_tag=MAIN_CARRIER,
                      hl=MAIN_HL, pert="nom", cand_id="C0", u=0.0)
            hit = base[base["window_id"] == w]
            if not len(hit) or not s.get("ok"):
                continue
            dev = abs(float(s["X_C_pp"]) - float(hit["X_C_pp"].iloc[0])) \
                / max(abs(float(hit["X_C_pp"].iloc[0])), 1e-12)
            if dev > worst:
                worst, worst_w = dev, w
    rows.append({"check": "U1", "item": "全部钩子缺省时与 P2.2-A 主口径 C0 一致",
                 "value": worst, "threshold": 1e-9, "pass": worst <= 1e-9,
                 "detail": f"9 参照窗 × 4 阶段；最大相对偏差出现在 {worst_w}"})
    if worst > 1e-9:
        hard_fail.append("U1")

    # U2：F2 取 ΔE=0 与 F1 在 f=0.1 逐位一致
    a = solve(window_id=w4, coal_set=args.coal_set, carrier_tag=MAIN_CARRIER,
              hl=MAIN_HL, pert="nom", cand_id="F1", u=math.log10(0.1))
    dfm = p22a.table("main")
    dry4 = float(dfm.loc[w4, "coal_dry_recon_phase_kg_s"])
    car4 = float(dfm.loc[w4, CARRIERS[MAIN_CARRIER]])
    op = p22a.build_op(MAIN_HL, 1.0, dry4, car4)
    op.update({"GasificationRateFactor": 0.1,
               "CharGasActivationIncrement_kJmol": 0.0,
               "CharGasTref_K": T_REF_K})
    rec = run_case(p22a.rows_for(args.coal_set)[w4], n_cells=p20.mesh_n(),
                   coal_flow_dry_kg_s=dry4, keep_system=True,
                   mesh_family="wsr_plus_pfr", extra_op=op)
    system = rec.get("_system")
    b_xc = None
    if system is not None and system.results and \
            rec.get("execution_status") == "completed":
        cd = float(system.coal_props.get("Cd", 0.0)) / 100.0
        c_in = float(system.W_dry) * cd
        o = system.results[-1]
        cap = system.slag_capture
        c_cap = 0.0 if not isinstance(cap, dict) else float(cap.get("carbon_kg_s", 0.0))
        b_xc = float((1.0 - (float(o.solid_mass) * float(o.carbon_fraction) + c_cap)
                      / max(c_in, 1e-12)) * 100.0)
    rec.pop("_system", None)
    same = (a.get("ok") and b_xc is not None and float(a["X_C_pp"]) == b_xc)
    rows.append({"check": "U2", "item": "F2 取 ΔE=0 时与 F1 在 f=0.1 处逐位一致",
                 "value": (None if b_xc is None
                           else abs(float(a["X_C_pp"]) - b_xc)),
                 "threshold": 0.0, "pass": bool(same),
                 "detail": f"F1 X_C={a.get('X_C_pp')}；F2(ΔE=0) X_C={b_xc}"})
    if not same:
        hard_fail.append("U2")
    fac = math.exp(-(50.0 * 1000.0 / R_GAS) * (1.0 / T_REF_K - 1.0 / T_REF_K))
    rows.append({"check": "U2b", "item": "F2 乘子在 T_p=T_ref 处为 1（倍数即 f_ref）",
                 "value": fac, "threshold": 1.0, "pass": abs(fac - 1.0) < 1e-15,
                 "detail": "exp[-(ΔE/R)(1/T_ref-1/T_ref)] = 1"})

    # U3：单调性
    mono = []
    for cand_id, seq in (("F1", [0.0, -0.5, -1.0]),
                         ("M1-0", [0.0, 2.0, 4.0]),
                         ("M2", [0.0, 0.1, 0.2])):
        xs = []
        for u in seq:
            s = solve(window_id=w4, coal_set=args.coal_set,
                      carrier_tag=MAIN_CARRIER, hl=MAIN_HL, pert="nom",
                      cand_id=cand_id, u=u)
            xs.append(float(s["X_C_pp"]) if s.get("ok") else float("nan"))
        dec = all(xs[i] > xs[i + 1] for i in range(len(xs) - 1)
                  if math.isfinite(xs[i]) and math.isfinite(xs[i + 1]))
        mono.append({"candidate": cand_id, "u": seq, "X_C": xs, "dec": dec})
    u3 = all(r["dec"] for r in mono)
    rows.append({"check": "U3", "item": "F1 / M1-0 / M2 自由量增大使 X_C 单调下降",
                 "value": json.dumps([{"cand": r["candidate"],
                                       "X_C": [None if not math.isfinite(v) else round(v, 4)
                                               for v in r["X_C"]]} for r in mono],
                                     ensure_ascii=False),
                 "threshold": "严格单调下降", "pass": bool(u3),
                 "detail": "F1 自由量为 log10 f：0 / -0.5 / -1 → f = 1 / 0.316 / 0.1"})
    if not u3:
        hard_fail.append("U3")

    # U4：新钩子在 JAX 批量路径报错
    u4_type, u4_msg = "", ""
    try:
        op = p22a.build_op(MAIN_HL, 1.0, dry4, car4)
        op.pop("HeatLossPercentTotal", None)
        op["HeatLossModel"] = "hhv_percent"
        op["HeatLossPercentHHV"] = 1.0
        op["CharGasActivationIncrement_kJmol"] = 50.0
        run_case(p22a.rows_for(args.coal_set)[w4], n_cells=p20.mesh_n(),
                 coal_flow_dry_kg_s=dry4, keep_system=True,
                 mesh_family="wsr_plus_pfr", extra_op=op, solver_method="jax_jit")
        u4_ok = False
    except Exception as exc:  # noqa: BLE001
        u4_type = type(exc).__name__
        u4_msg = str(exc)
        u4_ok = u4_type == "ValueError" and "CharGasActivationIncrement_kJmol" in u4_msg
    rows.append({"check": "U4", "item": "新钩子在 JAX 批量路径报错（与现有守卫同型）",
                 "value": u4_type, "threshold": "ValueError",
                 "pass": bool(u4_ok), "detail": u4_msg[:240]})
    if not u4_ok:
        hard_fail.append("U4")

    write_csv(pd.DataFrame(rows), "unit_checks.csv")
    print(f"[unit] {len(rows)} 项；hard_fail={hard_fail}", flush=True)
    if hard_fail:
        raise SystemExit(f"STOP[unit]: 停止条件未通过 {hard_fail}；见 unit_checks.csv")


# ---------------------------------------------------------------------------
# 比较点目标
# ---------------------------------------------------------------------------
def targets() -> dict[str, float]:
    rd = pd.read_csv(P22D_READINGS)
    out: dict[str, float] = {}
    for ph in ("phase2", "phase3", "phase4"):
        lbl = PHASE_LABEL[ph]
        hit = rd[(rd["reading_id"] == "FINAL") & (rd["phase"] == lbl)
                 & (rd["metric"] == "target_midpoint_pct")]
        if not len(hit):
            raise SystemExit(f"P2.2-D readings.csv 缺 {lbl} 的 FINAL/target_midpoint_pct")
        out[ph] = float(hit["value"].iloc[0])
    out["phase1"] = float(P1_TARGET_PP)
    return out


def stage_compare(args) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p22a.prepare_coal_inputs()
    p22a.require_pairing(args.windows_csv, args.coal_set, "compare")
    p22a.init_ctx(Path(args.windows_csv))
    init_pool(args.windows_csv, args.n_workers)
    ref = p22a._ref_windows("main")
    tg = targets()

    rows: list[dict] = []
    for cand_id in ("C0",) + FIT_CANDS:
        for ph in PHASES:
            r = solve_param(cand_id, ph, ref[ph], tg[ph], coal_set=args.coal_set)
            pv, note = param_report(cand_id, r["u"])
            extra: dict = {}
            if CAND[cand_id]["kind"] == "M2" and r["u"] is not None:
                specs_m2 = [{"window_id": w, "coal_set": args.coal_set,
                             "carrier_tag": MAIN_CARRIER, "hl": MAIN_HL,
                             "pert": "nom", "cand_id": cand_id,
                             "u": float(r["u"])} for w in ref[ph]]
                snaps = batch_solve(specs_m2)
                ok = [s for s in snaps if s.get("ok")]
                if ok:
                    extra = {
                        "capture_fraction": float(np.median(
                            [s["capture_fraction"] for s in ok])),
                        "capture_carbon_kg_s": float(np.median(
                            [s["capture_carbon_kg_s"] for s in ok])),
                        "capture_carbon_frac_of_inlet": float(np.median(
                            [s["capture_carbon_frac_of_inlet"] for s in ok])),
                        "capture_enthalpy_W": float(np.median(
                            [s["capture_enthalpy_W"] for s in ok])),
                        "capture_enthalpy_frac_hhv_pct": float(np.median(
                            [s["capture_enthalpy_frac_hhv_pct"] for s in ok])),
                    }
            rows.append({
                "candidate": cand_id, "phase": ph, "kind": CAND[cand_id]["kind"],
                "param_name": CAND[cand_id]["param_name"],
                "u_root": r["u"], "param_value": pv, "param_note": note,
                "target_X_C_pp": r["target_X_C_pp"],
                "X_C_pp_median": r["X_C_pp_median"],
                "dX_C_pp": (None if r["X_C_pp_median"] is None
                            else float(r["X_C_pp_median"] - r["target_X_C_pp"])),
                "reachable": bool(r["reachable"]), "reason": r["reason"],
                "n_ref": r["n_ref"], "n_solve_fail_at_root": r["n_solve_fail_at_root"],
                "n_probe": r["n_probe"],
                "hl_pct": MAIN_HL, "carrier": MAIN_CARRIER,
                "T_out_K": None, "T_wsr_K": None, "xi_mol_s": None,
                "CO2_after_xi_vol_pct": None,
                **extra,
            })
        print(f"[compare] {cand_id} 完成", flush=True)

    # 各候选在各阶段的 T_out / T_wsr / ξ
    df = pd.DataFrame(rows)
    for i, row in df.iterrows():
        if not row["reachable"] or row["u_root"] is None and row["candidate"] != "C0":
            continue
        u = 0.0 if row["candidate"] == "C0" else float(row["u_root"])
        ph = row["phase"]
        wids = ref[ph]
        specs = [{"window_id": w, "coal_set": args.coal_set,
                  "carrier_tag": MAIN_CARRIER, "hl": MAIN_HL, "pert": "nom",
                  "cand_id": row["candidate"], "u": float(u)} for w in wids]
        snaps = batch_solve(specs)
        ok = [s for s in snaps if s.get("ok")]
        if not ok:
            continue
        xi, co2_after, _ = solve_xi(row["candidate"], ph, wids,
                                    coal_set=args.coal_set, u=u)
        df.at[i, "T_out_K"] = float(np.median([s["T_out_K"] for s in ok]))
        df.at[i, "T_wsr_K"] = float(np.median([s["T_wsr_K"] for s in ok]))
        df.at[i, "xi_mol_s"] = xi
        df.at[i, "CO2_after_xi_vol_pct"] = co2_after
        print(f"[compare] {row['candidate']} {ph} ξ={xi:.3f}", flush=True)

    write_csv(df, "comparison_points.csv")
    reach = df[df["reachable"]].groupby("candidate")["phase"].count().to_dict()
    print(f"[compare] 可达阶段数: {reach}", flush=True)


# ---------------------------------------------------------------------------
# 斜率 / 弹性（只对可达候选）
# ---------------------------------------------------------------------------
def _roots_df() -> pd.DataFrame:
    return pd.read_csv(OUT_DIR / "comparison_points.csv")


def _cand_u(cand_id: str, ph: str) -> float | None:
    df = _roots_df()
    hit = df[(df["candidate"] == cand_id) & (df["phase"] == ph) & (df["reachable"])]
    if not len(hit):
        return None
    v = hit["u_root"].iloc[0]
    return None if pd.isna(v) else float(v)


def reachable_cands() -> list[str]:
    df = _roots_df()
    return sorted(set(df[df["reachable"] & (df["candidate"] != "C0")]["candidate"]))


def _slopes_for(cand_id: str, *, coal_set: str, hl: float = MAIN_HL,
                xi_by_phase: dict[str, float] | None = None) -> pd.DataFrame:
    df = p22a.table("main")
    wids = sorted(str(w) for w in df.index)
    obs_keys = ("X_C_pp", "CO_vol_pct", "H2_vol_pct", "CO2_vol_pct",
                "dry_per_coal_mol_kg")
    inputs = {"ln_O2": ("o2_p", "o2_m"), "ln_coal": ("coal_p", "coal_m"),
              "ln_carrier": ("car_p", "car_m")}
    rows: list[dict] = []
    for ph in PHASES:
        u = _cand_u(cand_id, ph)
        if u is None:
            continue
        pw = [w for w in wids if p22a._phase_of(w) == ph]
        xi_ph = 0.0 if not xi_by_phase else float(xi_by_phase.get(ph, 0.0))
        specs = [{"window_id": w, "coal_set": coal_set,
                  "carrier_tag": MAIN_CARRIER, "hl": hl, "pert": pert,
                  "cand_id": cand_id, "u": u}
                 for pert in PERTS for w in pw]
        results = batch_solve(specs)
        nw = len(pw)
        per: dict[str, dict[str, list[float]]] = {}
        n_fail = 0
        for j, pert in enumerate(PERTS):
            chunk = results[j * nw:(j + 1) * nw]
            acc: dict[str, list[float]] = {k: [] for k in obs_keys}
            for s in chunk:
                if not s.get("ok"):
                    n_fail += 1
                    continue
                a = apply_xi(s, xi_ph) or {}
                for k in obs_keys:
                    if k in a and a[k] is not None:
                        acc[k].append(float(a[k]))
            per[pert] = acc
        d_ln = p20.LN_P - p20.LN_M
        for k in obs_keys:
            for inp, (pp, pm) in inputs.items():
                yp, ym = per[pp][k], per[pm][k]
                n = min(len(yp), len(ym))
                if n == 0:
                    rows.append({"candidate": cand_id, "phase": ph, "obs": k,
                                 "input": inp, "per3_median": None, "n": 0,
                                 "n_fail": n_fail, "note": "该扰动下无可解窗"})
                    continue
                d2 = [(yp[i] - ym[i]) / d_ln * p20.LN_P for i in range(n)]
                rows.append({"candidate": cand_id, "phase": ph, "obs": k,
                             "input": inp, "per3_median": float(np.median(d2)),
                             "n": n, "n_fail": n_fail, "note": ""})
        print(f"[slopes] {cand_id} {ph} 完成（n_fail={n_fail}）", flush=True)
    return pd.DataFrame(rows)


def stage_slopes(args) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p22a.prepare_coal_inputs()
    p22a.require_pairing(args.windows_csv, args.coal_set, "compare")
    p22a.init_ctx(Path(args.windows_csv))
    init_pool(args.windows_csv, args.n_workers)
    cands = reachable_cands()
    ref = p22a._ref_windows("main")
    frames = []
    for c in cands:
        xi_by_phase = {}
        for ph in PHASES:
            u = 0.0 if c == "C0" else _cand_u(c, ph)
            if u is None:
                continue
            xi, _, _ = solve_xi(c, ph, ref[ph], coal_set=args.coal_set, u=u)
            if math.isfinite(xi):
                xi_by_phase[ph] = xi
        frames.append(_slopes_for(c, coal_set=args.coal_set,
                                  xi_by_phase=xi_by_phase))
    cols = ["candidate", "phase", "obs", "input", "per3_median", "n", "n_fail", "note"]
    write_csv(pd.concat(frames, ignore_index=True) if frames
                    else pd.DataFrame(columns=cols), "model_slopes.csv")
    print(f"[slopes] 候选 {cands}", flush=True)


def _elasticities(slopes: pd.DataFrame, cp: pd.DataFrame) -> pd.DataFrame:
    """弹性定义与 P2.0-A 的 `elasticities()` 逐字一致。

    `e_coal = 1 + r`、`e_o2 = r`，其中 `r = (per3_XC / XC) / ln(1.03)`。
    那个 `+1` 不是笔误：P2.0-A 的 e_coal 是**已转化碳量**对煤量的弹性——煤量乘 (1+δ)、
    转化率不变时，转化碳量也乘 (1+δ)，故 e = 1（封顶时的正确值）。
    装置区间（PLANT_E_INTERVALS）来自同一个定义，少加这个 +1 会使全部 e_coal 平移 1，
    与装置不可比。e_o2 无 +1。
    """
    rows = []
    for cand_id in sorted(set(slopes["candidate"])):
        for ph in PHASES:
            sub = slopes[(slopes["candidate"] == cand_id) & (slopes["phase"] == ph)]

            def get(obs: str, inp: str):
                h = sub[(sub["obs"] == obs) & (sub["input"] == inp)]
                if not len(h) or pd.isna(h["per3_median"].iloc[0]):
                    return None
                return float(h["per3_median"].iloc[0])

            h = cp[(cp["candidate"] == cand_id) & (cp["phase"] == ph)]
            xc_med = (None if not len(h) or pd.isna(h["X_C_pp_median"].iloc[0])
                      else float(h["X_C_pp_median"].iloc[0]))
            p_c, p_o = get("X_C_pp", "ln_coal"), get("X_C_pp", "ln_O2")
            e_c = e_o = None
            note = ""
            if xc_med is None:
                note = "该阶段不可达，无比较点，不计算弹性"
            elif p_c is None or p_o is None:
                note = "缺斜率，不计算弹性"
            else:
                r_c = (p_c / xc_med) / p20.LN_P
                r_o = (p_o / xc_med) / p20.LN_P
                e_c, e_o = 1.0 + r_c, r_o
            iv = PLANT_E_INTERVALS[ph]
            rows.append({
                "candidate": cand_id, "phase": ph, "X_C_pp_level": xc_med,
                "e_coal_model": e_c, "e_o2_model": e_o,
                "e_coal_plant_lo": iv["coal"][0], "e_coal_plant_hi": iv["coal"][1],
                "e_o2_plant_lo": iv["o2"][0], "e_o2_plant_hi": iv["o2"][1],
                "e_coal_in_interval": (None if e_c is None
                                       else bool(iv["coal"][0] <= e_c <= iv["coal"][1])),
                "e_o2_in_interval": (None if e_o is None
                                     else bool(iv["o2"][0] <= e_o <= iv["o2"][1])),
                "note": note,
            })
    return pd.DataFrame(rows)


def stage_elasticities(args) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slopes = pd.read_csv(OUT_DIR / "model_slopes.csv")
    cp = pd.read_csv(OUT_DIR / "comparison_points.csv")
    df = _elasticities(slopes, cp)
    # 不可达阶段的数值缺失写空字符串，不写 NaN（卡面：不得输出裸 NaN）
    write_csv(df, "elasticities.csv")
    print("[elasticities] 完成", flush=True)


# ---------------------------------------------------------------------------
# 组成水平（levels）
# ---------------------------------------------------------------------------
def stage_levels(args) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p22a.prepare_coal_inputs()
    p22a.require_pairing(args.windows_csv, args.coal_set, "compare")
    p22a.init_ctx(Path(args.windows_csv))
    init_pool(args.windows_csv, args.n_workers)
    ref = p22a._ref_windows("main")
    rows: list[dict] = []
    for cand_id in ("C0",) + tuple(reachable_cands()):
        for ph in PHASES:
            u = 0.0 if cand_id == "C0" else _cand_u(cand_id, ph)
            if u is None:
                continue
            wids = ref[ph]
            xi, _, n_fail = solve_xi(cand_id, ph, wids, coal_set=args.coal_set, u=u)
            if not math.isfinite(xi):
                rows.append({"candidate": cand_id, "phase": ph,
                             "note": "ξ 无法闭合（模型 CO2 低于装置值）",
                             "n_fail": n_fail})
                continue
            dfm = p22a.table("main")
            for comp, col in (("CO_vol_pct", "co_vol_pct_mean"),
                              ("H2_vol_pct", "h2_vol_pct_mean"),
                              ("CO2_vol_pct", "co2_vol_pct_mean"),
                              ("CH4_vol_pct", "ch4_vol_pct_mean")):
                mvals, pvals = [], []
                specs_lv = [{"window_id": w, "coal_set": args.coal_set,
                             "carrier_tag": MAIN_CARRIER, "hl": MAIN_HL,
                             "pert": "nom", "cand_id": cand_id, "u": float(u)}
                            for w in wids]
                for s in batch_solve(specs_lv):
                    a = apply_xi(s, xi) if s.get("ok") else None
                    if a is not None and a.get(comp) is not None:
                        mvals.append(float(a[comp]))
                for w in wids:
                    pvals.append(float(dfm.loc[w, col]))
                m = float(np.median(mvals)) if mvals else None
                p = float(np.median(pvals)) if pvals else None
                rows.append({
                    "candidate": cand_id, "phase": ph, "observable": comp,
                    "model_pp": m, "plant_pp": p,
                    "dev_pp": None if (m is None or p is None) else m - p,
                    "xi_mol_s": xi, "n_fail": n_fail,
                })
    write_csv(pd.DataFrame(rows), "levels.csv")
    print(f"[levels] {len(rows)} 行", flush=True)


# ---------------------------------------------------------------------------
# 稳健性（热损 0.5 / 2%）
# ---------------------------------------------------------------------------
def stage_robustness(args) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p22a.prepare_coal_inputs()
    p22a.require_pairing(args.windows_csv, args.coal_set, "compare")
    p22a.init_ctx(Path(args.windows_csv))
    init_pool(args.windows_csv, args.n_workers)
    ref = p22a._ref_windows("main")
    tg = targets()
    cands = reachable_cands()
    # 卡面第 5 节：只对入选候选；无入选时对「H 通过项最多」的候选（最多两个）。
    # 通过项从 readings.csv 的 H0–H5 逐条计；若有多个并列第一则最多取两个。
    rd = pd.read_csv(OUT_DIR / "readings.csv", keep_default_na=False)
    n_pass: dict[str, int] = {}
    for _, r in rd[rd["reading"].isin(["H0", "H1", "H2", "H3", "H4", "H5"])].iterrows():
        c = str(r["phase"])
        if str(r["pass"]).strip().lower() == "true":
            n_pass[c] = n_pass.get(c, 0) + 1
    ranked = sorted((c for c in cands), key=lambda c: -n_pass.get(c, 0))
    top = n_pass.get(ranked[0], 0) if ranked else 0
    picks = [c for c in ranked if n_pass.get(c, 0) == top and top > 0][:2] if top else []
    if not picks:
        picks = []
    print(f"[robust] H 通过项计数 {n_pass}；选取 {picks}"
          f"（无候选有任何 H 通过项时不跑稳健性）", flush=True)
    if not picks:
        write_csv(pd.DataFrame([{"candidate": "", "phase": "", "hl_pct": "",
                                 "note": "无候选通过任何 H 读法，按卡面第 5 节不跑稳健性"}]), 
                  "robustness.csv")
        return
    rows: list[dict] = []
    for cand_id in picks:
        for hl in (0.5, 2.0):
            for ph in PHASES:
                r = solve_param(cand_id, ph, ref[ph], tg[ph],
                                coal_set=args.coal_set, hl=hl)
                snaps = []
                if r["u"] is not None:
                    specs_rb = [{"window_id": w, "coal_set": args.coal_set,
                                 "carrier_tag": MAIN_CARRIER, "hl": hl,
                                 "pert": "nom", "cand_id": cand_id,
                                 "u": float(r["u"])} for w in ref[ph]]
                    snaps = batch_solve(specs_rb)
                ok = [s for s in snaps if s.get("ok")]
                t_out = float(np.median([s["T_out_K"] for s in ok])) if ok else None
                # 参照窗 e_氧（同主口径定义）
                xi_ph = {}
                for _ph in PHASES:
                    _u = _cand_u(cand_id, _ph)
                    if _u is None:
                        continue
                    _xi, _, _ = solve_xi(cand_id, _ph, ref[_ph],
                                         coal_set=args.coal_set, u=_u, hl=hl)
                    if math.isfinite(_xi):
                        xi_ph[_ph] = _xi
                slopes_hl = _slopes_for(cand_id, coal_set=args.coal_set, hl=hl,
                                        xi_by_phase=xi_ph)
                sub = slopes_hl[(slopes_hl["phase"] == ph) & (slopes_hl["obs"] == "X_C_pp")
                                & (slopes_hl["input"] == "ln_O2")]
                per3 = None
                if len(sub) and not pd.isna(sub["per3_median"].iloc[0]):
                    per3 = float(sub["per3_median"].iloc[0])
                e_o = None
                if per3 is not None and r["X_C_pp_median"]:
                    e_o = (per3 / float(r["X_C_pp_median"])) / p20.LN_P
                iv = PLANT_E_INTERVALS[ph]
                rows.append({
                    "candidate": cand_id, "phase": ph, "hl_pct": hl,
                    "u_root": r["u"], "reachable": bool(r["reachable"]),
                    "X_C_pp_median": r["X_C_pp_median"],
                    "dX_C_pp": (None if r["X_C_pp_median"] is None
                                else float(r["X_C_pp_median"] - r["target_X_C_pp"])),
                    "T_out_K": t_out,
                    "e_o2_model": e_o,
                    "e_o2_in_interval": (None if e_o is None
                                         else bool(iv["o2"][0] <= e_o <= iv["o2"][1])),
                    "H4_ok": (None if t_out is None else bool(
                        t_out >= LIU_FT_C[ph] - H4_FT_MARGIN_K)),
                    "reason": r["reason"],
                })
                print(f"[robust] {cand_id} hl={hl} {ph} 完成", flush=True)
    write_csv(pd.DataFrame(rows), "robustness.csv")


# ---------------------------------------------------------------------------
# 剖面
# ---------------------------------------------------------------------------
def stage_profiles(args) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p22a.prepare_coal_inputs()
    p22a.require_pairing(args.windows_csv, args.coal_set, "compare")
    p22a.init_ctx(Path(args.windows_csv))
    ref = p22a._ref_windows("main")
    rows: list[dict] = []
    for cand_id in ("C0",) + tuple(reachable_cands()):
        for ph in PHASES:
            u = 0.0 if cand_id == "C0" else _cand_u(cand_id, ph)
            if u is None:
                continue
            for w in ref[ph]:
                dfm = p22a.table("main")
                dry = float(dfm.loc[w, "coal_dry_recon_phase_kg_s"])
                car = float(dfm.loc[w, CARRIERS[MAIN_CARRIER]])
                op = p22a.build_op(MAIN_HL, 1.0, dry, car)
                op.update(hook_op(cand_id, u))
                rec = run_case(p22a.rows_for(args.coal_set)[w],
                               n_cells=p20.mesh_n(), coal_flow_dry_kg_s=dry,
                               keep_system=True, mesh_family="wsr_plus_pfr",
                               extra_op=op)
                system = rec.get("_system")
                if system is None or not system.results:
                    rows.append({"candidate": cand_id, "phase": ph, "window_id": w,
                                 "cell_index": None, "z_m": None, "X_C_pp": None,
                                 "T_gas_K": None,
                                 "execution_status": rec.get("execution_status")})
                    rec.pop("_system", None)
                    continue
                cd = float(system.coal_props.get("Cd", 0.0)) / 100.0
                c_in = float(system.W_dry) * cd
                z = 0.0
                for i, cell in enumerate(system.results):
                    z += float(getattr(cell, "dz", 0.0) or 0.0)
                    c_out = float(cell.solid_mass) * float(cell.carbon_fraction)
                    rows.append({
                        "candidate": cand_id, "phase": ph, "window_id": w,
                        "cell_index": i, "z_m": z,
                        "X_C_pp": float((1.0 - c_out / max(c_in, 1e-12)) * 100.0),
                        "T_gas_K": float(cell.T),
                        "execution_status": rec.get("execution_status"),
                    })
                rec.pop("_system", None)
    write_csv(pd.DataFrame(rows), "profiles.csv")
    print(f"[profiles] {len(rows)} 行", flush=True)


# ---------------------------------------------------------------------------
# 阶段 read：H0–H7 与判定
# ---------------------------------------------------------------------------
def _reading(reading: str, metric: str, phase: str, value, threshold,
             passed, note: str) -> dict:
    return {"reading": reading, "metric": metric, "phase": phase,
            "value": value, "threshold": threshold, "pass": passed, "note": note}


def _h0(cp: pd.DataFrame) -> tuple[list[dict], dict]:
    rows, out = [], {}
    for cand_id in FIT_CANDS:
        n = int(cp[(cp["candidate"] == cand_id) & (cp["reachable"])]["phase"].nunique())
        miss = cp[(cp["candidate"] == cand_id) & (~cp["reachable"])]
        detail = "；".join(f"{PHASE_LABEL[r['phase']]}: {r['reason']}"
                          for _, r in miss.iterrows() if r["reason"])
        rows.append(_reading("H0", "n_reachable_phases", cand_id, n, 4, n == 4,
                             "四阶段全可达" if n == 4 else f"未达/不可达：{detail}"))
        out[cand_id] = {"n_reachable": n, "ok": n == 4}
    return rows, out


def _h1(el: pd.DataFrame) -> tuple[list[dict], dict]:
    rows, out = [], {}
    for cand_id in sorted(set(el["candidate"])):
        sub = el[el["candidate"] == cand_id]
        n = int(sum(1 for _, r in sub.iterrows() if r["e_o2_in_interval"] is True))
        ok = n >= H1_MIN_PHASES
        detail = "；".join(
            f"{PHASE_LABEL[r['phase']]} e_氧={r['e_o2_model']:.3f}"
            if r["e_o2_model"] is not None and not pd.isna(r["e_o2_model"])
            else f"{PHASE_LABEL[r['phase']]} 不可达"
            for _, r in sub.iterrows())
        rows.append(_reading("H1", "n_phases_e_o2_in_interval", cand_id, n,
                             H1_MIN_PHASES, ok, detail))
        out[cand_id] = {"n_phases": n, "ok": ok}
    return rows, out


def _h2(sl: pd.DataFrame) -> tuple[list[dict], dict]:
    rows, out = [], {}
    for cand_id in sorted(set(sl["candidate"])):
        same, detail = 0, []
        for ph in H2_PHASES:
            sub = sl[(sl["candidate"] == cand_id) & (sl["phase"] == ph)
                     & (sl["obs"] == "dry_per_coal_mol_kg")
                     & (sl["input"] == "ln_O2")]
            if not len(sub) or pd.isna(sub["per3_median"].iloc[0]):
                detail.append(f"{PHASE_LABEL[ph]}: 无斜率")
                continue
            v = float(sub["per3_median"].iloc[0])
            if v > 0:
                same += 1
            detail.append(f"{PHASE_LABEL[ph]}={v:+.3f}（装置为正）")
        ok = same == 3
        rows.append(_reading("H2", "n_phases_dry_per_coal_o2_positive", cand_id,
                             same, 3, ok, "；".join(detail)))
        out[cand_id] = {"n_same": same, "ok": ok}
    return rows, out


def _h3(sl: pd.DataFrame) -> tuple[list[dict], dict]:
    plant = pd.read_csv(P115A_DIR / "plant_slopes.csv")
    rows, out = [], {}
    for cand_id in sorted(set(sl["candidate"])):
        n, detail = 0, []
        for ph, obs, inp in p20.G2_COMBOS:
            sub = sl[(sl["candidate"] == cand_id) & (sl["phase"] == ph)
                     & (sl["obs"] == obs) & (sl["input"] == inp)]
            psub = plant[(plant["phase"] == ph) & (plant["obs"] == obs)
                         & (plant["input"] == inp) & (plant["model"] == "with_carrier")]
            if not len(sub) or not len(psub) or pd.isna(sub["per3_median"].iloc[0]):
                detail.append(f"{PHASE_LABEL[ph]} {obs}←{inp}: 缺")
                continue
            m = float(sub["per3_median"].iloc[0])
            p = float(psub["per3"].iloc[0])
            same = (m > 0) == (p > 0)
            n += int(same)
            detail.append(f"{PHASE_LABEL[ph]} {obs}←{inp}: {m:+.3f}/{p:+.3f} "
                          f"{'同号' if same else '反号'}")
        ok = n >= H3_MIN_SIGNED
        rows.append(_reading("H3", "n_same_sign_of_8", cand_id, n,
                             H3_MIN_SIGNED, ok, "；".join(detail)))
        out[cand_id] = {"n_same": n, "ok": ok}
    return rows, out


def _h4(cp: pd.DataFrame) -> tuple[list[dict], dict]:
    rows, out = [], {}
    for cand_id in sorted(set(cp["candidate"])):
        sub = cp[cp["candidate"] == cand_id]
        at_ft, margin_ok, detail = 0, 0, []
        for _, r in sub.iterrows():
            ph = r["phase"]
            if r["T_out_K"] is None or pd.isna(r["T_out_K"]):
                detail.append(f"{PHASE_LABEL[ph]}: 无温度")
                continue
            t_c = float(r["T_out_K"]) - 273.15
            ft = LIU_FT_C[ph]
            if t_c >= ft:
                at_ft += 1
            if t_c >= ft - H4_FT_MARGIN_K:
                margin_ok += 1
            detail.append(f"{PHASE_LABEL[ph]}={t_c:.0f}°C vs FT {ft:.0f}（差 {t_c-ft:+.0f} K）")
        ok = margin_ok == 4 and at_ft >= H4_MIN_AT_FT
        rows.append(_reading("H4", "T_out_vs_FT", cand_id,
                             f"at_FT={at_ft},margin_ok={margin_ok}",
                             f"≥{H4_MIN_AT_FT} at FT 且 4 个 ≥ FT-50K", ok,
                             "；".join(detail)))
        out[cand_id] = {"at_ft": at_ft, "margin_ok": margin_ok, "ok": ok}
    return rows, out


def _h5(lv: pd.DataFrame) -> tuple[list[dict], dict]:
    rows, out = [], {}
    for cand_id in sorted(set(lv["candidate"])):
        sub = lv[(lv["candidate"] == cand_id)
                 & lv["observable"].isin(["CO_vol_pct", "H2_vol_pct"])]
        worst, ok, detail = 0.0, True, []
        for _, r in sub.iterrows():
            if r["dev_pp"] is None or pd.isna(r["dev_pp"]):
                ok = False
                detail.append(f"{PHASE_LABEL[r['phase']]} {r['observable']}: 缺")
                continue
            d = abs(float(r["dev_pp"]))
            worst = max(worst, d)
            if d > H5_MAX_DEV_PP:
                ok = False
            detail.append(f"{PHASE_LABEL[r['phase']]} {r['observable']}={float(r['dev_pp']):+.2f} pp")
        rows.append(_reading("H5", "max_abs_CO_H2_dev_pp", cand_id,
                             round(worst, 3) if worst else None, H5_MAX_DEV_PP,
                             ok, "；".join(detail)))
        out[cand_id] = {"max_dev": worst, "ok": ok}
    return rows, out


def _h6_h7(el: pd.DataFrame, cp: pd.DataFrame) -> list[dict]:
    rows = []
    for cand_id in sorted(set(el["candidate"])):
        sub = el[el["candidate"] == cand_id]
        n = int(sum(1 for _, r in sub.iterrows() if r["e_coal_in_interval"] is True))
        rows.append(_reading("H6", "n_phases_e_coal_in_interval", cand_id, n,
                             None, None, "只报告，不进判定"))
    for cand_id in FIT_CANDS:
        pts = cp[(cp["candidate"] == cand_id) & (cp["reachable"])]
        trend = "；".join(
            f"{PHASE_LABEL[r['phase']]} 神优3={['100','70','50','30'][i]}% → "
            f"{r['param_note']}"
            for i, (_, r) in enumerate(pts.iterrows()))
        rows.append(_reading("H7", "param_vs_blend", cand_id, len(pts), None, None,
                             trend or "无可达阶段"))
    return rows


def stage_read(args) -> None:
    p22a.prepare_coal_inputs()
    p22a.require_pairing(args.windows_csv, args.coal_set, "read")
    p22a.init_ctx(Path(args.windows_csv))
    cp = pd.read_csv(OUT_DIR / "comparison_points.csv")
    sl_path = OUT_DIR / "model_slopes.csv"
    el_path = OUT_DIR / "elasticities.csv"
    lv_path = OUT_DIR / "levels.csv"
    sl = pd.read_csv(sl_path) if sl_path.exists() else pd.DataFrame(
        columns=["candidate", "phase", "obs", "input", "per3_median"])
    el = pd.read_csv(el_path) if el_path.exists() else pd.DataFrame(
        columns=["candidate", "phase", "e_coal_model", "e_o2_model"])
    lv = pd.read_csv(lv_path) if lv_path.exists() else pd.DataFrame(
        columns=["candidate", "phase", "observable", "dev_pp"])

    rows: list[dict] = []
    h0r, h0 = _h0(cp)
    h1r, h1 = _h1(el) if len(el) else ([], {})
    h2r, h2 = _h2(sl) if len(sl) else ([], {})
    h3r, h3 = _h3(sl) if len(sl) else ([], {})
    h4r, h4 = _h4(cp)
    h5r, h5 = _h5(lv) if len(lv) else ([], {})
    rows += h0r + h1r + h2r + h3r + h4r + h5r + _h6_h7(el, cp)

    ok_all = []
    for cand_id in FIT_CANDS:
        checks = {
            "H0": h0.get(cand_id, {}).get("ok", False),
            "H1": h1.get(cand_id, {}).get("ok", False),
            "H2": h2.get(cand_id, {}).get("ok", False),
            "H3": h3.get(cand_id, {}).get("ok", False),
            "H4": h4.get(cand_id, {}).get("ok", False),
            "H5": h5.get(cand_id, {}).get("ok", False),
        }
        passed = all(checks.values())
        fails = [k for k, v in checks.items() if not v]
        rows.append(_reading("VERDICT", "candidate_verdict", cand_id,
                             "SELECTED" if passed else "NOT_SELECTED", "H0–H5 全通过",
                             passed,
                             "全部通过" if passed else f"未通过：{','.join(fails)}"))
        if passed:
            ok_all.append(cand_id)
    for cand_id in FIT_CANDS:
        n_in = (h1.get(cand_id, {}).get("n_phases", 0)
                + (0 if cand_id not in h1 else 0))
    rows.append(_reading("SELECTED", "selected_candidates", "all",
                         ";".join(ok_all) if ok_all else "无", "H0–H5",
                         bool(ok_all),
                         ("入选：" + ";".join(ok_all)) if ok_all
                         else "无候选入选，逐候选列出未通过项"))

    write_csv(pd.DataFrame(rows), "readings.csv")
    print(f"[read] 入选={ok_all if ok_all else '无'}", flush=True)


# ---------------------------------------------------------------------------
# 清单
# ---------------------------------------------------------------------------
def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(df: pd.DataFrame, name: str) -> Path:
    """写本轮 CSV。

    **不得使用 p22a._write_csv**：那个函数写的是 p22a.OUT_DIR（P2.2-A 目录），
    用它会把本轮的产物写进已合并的 P2.2-A 证据目录，并可能覆盖其中的同名文件
    （2026-11-18 实际发生过一次：unit_checks.csv 被覆盖，已从 git 恢复）。
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / name
    # 卡面：不得输出裸 NaN。缺失值写空字符串，原因另写在 note/reason 列。
    df = df.replace({np.nan: ""})
    df.to_csv(out, index=False, lineterminator="\n")
    return out


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def protocol(args) -> dict:
    return {
        "round": "P2.3-A-20261118",
        "card": "manuscript/model-side-handoff-p2.3a-20261118.md",
        "repo_commit": p20.source_commit(),
        "coal_set": args.coal_set,
        "windows_csv": _rel(Path(args.windows_csv)),
        "windows_sha256": p22a.table_sha("main"),
        "frozen": {
            "kappa": p20.KAPPA, "mesh": "wsr_plus_pfr",
            "L_wsr_m": p20.L_WSR, "dz_down_m": p20.DZ_DOWN,
            "HeatLossModel": "total_percent", "HeatLossPercentTotal": MAIN_HL,
            "carrier": MAIN_CARRIER, "kinetics": "Wen 1979",
        },
        "candidates": [{k: v for k, v in c.items()} for c in CANDIDATES],
        "root_finding": {
            "variable": "candidate free parameter (F1/F2 in log10)",
            "tolerance_pp": ROOT_TOL_PP, "bisect_max": BISECT_MAX,
            "interior_probe": INTERIOR_PROBE,
            "bracketing": "端点先探；夹不住目标则向内自适应扫描；无解向可解方向收缩",
            "no_bare_nan": "不可达一律写 reason，value 列留空或给可解界值",
        },
        "targets": targets(),
        "plant_elasticity_intervals": PLANT_E_INTERVALS,
        "liu_FT_C": LIU_FT_C,
        "thresholds": {
            "H0_reachable_phases": 4, "H1_min_phases_e_o2_in_interval": H1_MIN_PHASES,
            "H2_phases": list(H2_PHASES), "H3_min_signed": H3_MIN_SIGNED,
            "H4_ft_margin_K": H4_FT_MARGIN_K, "H4_min_at_ft": H4_MIN_AT_FT,
            "H5_max_dev_pp": H5_MAX_DEV_PP,
        },
        "n_workers": args.n_workers,
    }


def write_manifest() -> None:
    script = Path(__file__).resolve()
    products = ["unit_checks.csv", "comparison_points.csv", "model_slopes.csv",
                "elasticities.csv", "levels.csv", "robustness.csv",
                "profiles.csv", "readings.csv", "protocol.json",
                "interpretation.md"]
    inputs = [p22a.P22D_CSV if hasattr(p22a, "P22D_CSV") else
              ROOT / "outputs" / "diagnostics" / "p2.2d-20261118" / "windows_142.csv",
              P22D_READINGS,
              P115A_DIR / "plant_slopes.csv",
              P20A_DIR / "elasticities.csv",
              P22A_DIR / "baseline_142.csv",
              ROOT / "src" / "model" / "cell.py",
              ROOT / "src" / "model" / "gasifier_system.py"]
    lines = ["# P2.3-A 产物与输入清单（sha256，相对仓库根）", "", "# products"]
    for name in products:
        p = OUT_DIR / name
        if p.exists():
            lines.append(f"{_sha(p)}  {_rel(p)}")
        else:
            lines.append(f"{'0'*64}  {_rel(p)}  (missing)")
    lines += ["", "# inputs"]
    for p in inputs:
        p = Path(p)
        if p.exists():
            lines.append(f"{_sha(p)}  {_rel(p)}")
    (OUT_DIR / "sha256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coal-set", required=True,
                    choices=("current", "legacy_20261118", "lims_heat"))
    ap.add_argument("--windows-csv", required=True)
    ap.add_argument("--stage", default="all",
                    choices=("unit", "compare", "slopes", "elasticities",
                             "levels", "robustness", "profiles", "read", "all"))
    ap.add_argument("--n-workers", type=int, default=13)
    ap.add_argument("--source-commit", default=None)
    args = ap.parse_args()
    silence_solver_logs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
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
    elif args.stage == "robustness":
        stage_robustness(args)
    elif args.stage == "profiles":
        stage_profiles(args)
    elif args.stage == "read":
        stage_read(args)
    else:
        stage_unit(args)
        stage_compare(args)
        stage_slopes(args)
        stage_elasticities(args)
        stage_levels(args)
        stage_profiles(args)
        # read 必须在 robustness 之前：稳健性要读 readings.csv 的 H 通过项来选候选
        stage_read(args)
        stage_robustness(args)
        (OUT_DIR / "protocol.json").write_text(
            json.dumps(protocol(args), ensure_ascii=False, indent=2,
                       default=str) + "\n", encoding="utf-8")
        write_manifest()
        print("[done] 产物与清单已写出", flush=True)


if __name__ == "__main__":
    main()
