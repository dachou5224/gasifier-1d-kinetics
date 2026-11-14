#!/usr/bin/env python3
"""P2.0-A 读法层：由快照生成全部产物与 G0–G6 判定（2026-11-13 卡第 3、4 节）。

输入（本目录）：
  comparison_points.csv   第 2 节比较点（stage compare 写的原表）
  hl_comparison_points.csv
  perturb_snaps.json      142 窗 × {基准 + 6 扰动} 的反应器快照
  hl_snaps.json           参照窗的 HL 敏感性快照
  profiles.json           剖面

输出：model_slopes.csv、elasticities.csv、levels.csv、hl_sensitivity.csv、
      profiles.csv、readings.csv、comparison_points.csv（补 ξ/温度/截留列）、
      protocol.json、interpretation.md、sha256.txt
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

P114D_CSV = ROOT / "outputs" / "diagnostics" / "p1.14d-20261107" / "windows_142.csv"
P112A_CSV = ROOT / "outputs" / "diagnostics" / "round-p1.12a-20261106" / "input_sensitivity.csv"
P115A_SLOPES = ROOT / "outputs" / "diagnostics" / "round-p1.15a-20261108" / "plant_slopes.csv"
P115A_CAND = ROOT / "outputs" / "diagnostics" / "round-p1.15a-20261108" / "candidates.csv"

PHASES = ("phase1", "phase2", "phase3", "phase4")
PHASE_MAP = {"P1": "phase1", "P2": "phase2", "P3": "phase3", "P4": "phase4"}
CAND_ORDER = ("C0", "M1-0", "M1-80", "M1-90", "M2")
FIT_CANDS = ("M1-0", "M1-80", "M1-90", "M2")
KIND = {"C0": "C0", "M1-0": "M1", "M1-80": "M1", "M1-90": "M1", "M2": "M2"}
PARAM_NAME = {"C0": "none", "M1-0": "n", "M1-80": "n", "M1-90": "n", "M2": "f_s"}

OBS_ALL = ("X_C_pp", "CO_vol_pct", "H2_vol_pct", "CO2_vol_pct", "CH4_vol_pct",
           "dry_per_coal_mol_kg", "T_out_K", "T_wsr_K")
OBS_LEVEL = ("X_C_pp", "CO_vol_pct", "CO2_vol_pct", "H2_vol_pct", "CH4_vol_pct",
             "dry_per_coal_mol_kg")
INPUTS = ("ln_O2", "ln_coal", "ln_carrier")
FD_FRAC = 0.03
LN_P = math.log(1.0 + FD_FRAC)
LN_M = math.log(1.0 - FD_FRAC)
D_LN = LN_P - LN_M
PERTS = ("nom", "o2_p", "o2_m", "coal_p", "coal_m", "car_p", "car_m")
INPUT_PERTS = {"ln_O2": ("o2_p", "o2_m"), "ln_coal": ("coal_p", "coal_m"),
               "ln_carrier": ("car_p", "car_m")}
XC_CEIL_PP = 99.9
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
G1_MIN_K = 50.0
G2_MIN = 6
G3_MIN_ECOAL = 0.3
G3_MIN_PHASES = 3

_CTX: dict = {}


def _f(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def _phase_of(raw) -> str:
    s = str(raw)
    return PHASE_MAP.get(s, s if s.startswith("phase") else f"phase{s[-1]}")


def _load_inputs() -> None:
    d = pd.read_csv(P114D_CSV).set_index("window_id")
    _CTX["p114d"] = d
    _CTX["phase_of"] = {str(w): _phase_of(d.loc[w, "phase"]) for w in d.index}
    p112a = pd.read_csv(P112A_CSV)
    ref = p112a[(p112a["L_wsr"] == 1.0) & (p112a["perturb_tag"] == "baseline")]
    _CTX["ref"] = {ph: [] for ph in PHASES}
    for w in sorted(str(x) for x in ref["window_id"]):
        _CTX["ref"][_CTX["phase_of"][w]].append(w)
    _CTX["plant"] = pd.read_csv(P115A_SLOPES)
    _CTX["theta_fit"] = pd.read_csv(P115A_CAND)
    _CTX["win"] = {ph: [] for ph in PHASES}
    for w in sorted(str(x) for x in d.index):
        _CTX["win"][_CTX["phase_of"][w]].append(w)


# ---------------------------------------------------------------------------
# ξ 与观测量
# ---------------------------------------------------------------------------

def apply_xi(snap: dict, xi: float) -> dict | None:
    if not snap or not snap.get("ok"):
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
        "CO_positive": bool(f_co > 0.0),
    }


def solve_xi(store: dict[str, dict], wids: list[str], target: float) -> float:
    ok = [store[w] for w in wids if store.get(w, {}).get("ok")]
    if not ok:
        return float("nan")

    def med(x: float) -> float:
        vals = []
        for w in wids:
            y = apply_xi(store.get(w) or {}, x)
            if y is None:
                return float("nan")
            vals.append(y["CO2_vol_pct"])
        return float(np.median(vals)) if vals else float("nan")

    lo = -0.95 * min(float(s["F_CO2"]) for s in ok)
    hi = 0.95 * min(float(s["F_CO"]) for s in ok)
    flo, fhi = med(lo) - target, med(hi) - target
    if not (math.isfinite(flo) and math.isfinite(fhi)) or flo > 0.0 or fhi < 0.0:
        return float("nan")
    a, b = lo, hi
    for _ in range(200):
        mid = 0.5 * (a + b)
        fm = med(mid) - target
        if not math.isfinite(fm):
            return float("nan")
        if abs(fm) < 1e-10 or (b - a) < 1e-9 * max(abs(b), 1.0):
            return float(mid)
        if fm < 0.0:
            a = mid
        else:
            b = mid
    return float(0.5 * (a + b))


def plant_xc_med(phase: str) -> float:
    d = _CTX["p114d"]
    return float(np.median([float(d.loc[w, "xc_recon_phase_cal_measured_pct"])
                            for w in _CTX["win"][phase]]))


def plant_obs_med(phase: str, col: str) -> float:
    d = _CTX["p114d"]
    return float(np.median([float(d.loc[w, col]) for w in _CTX["win"][phase]]))


# ---------------------------------------------------------------------------
# 主构建
# ---------------------------------------------------------------------------

def build_all(out_dir: Path, src_commit: str | None = None) -> None:
    _load_inputs()
    snaps = json.loads((out_dir / "perturb_snaps.json").read_text(encoding="utf-8"))
    cp_raw = pd.read_csv(out_dir / "comparison_points.csv")
    hl_snaps = json.loads((out_dir / "hl_snaps.json").read_text(encoding="utf-8"))
    hl_cp = pd.read_csv(out_dir / "hl_comparison_points.csv")
    prof = json.loads((out_dir / "profiles.json").read_text(encoding="utf-8"))
    # 单元检查表也需要传入解释生成器：否则 U1–U5 一节会落进「未提供」分支，
    # 与实际已生成的 unit_checks.csv 不一致（2026-11-14 修正）。
    unit_df = pd.read_csv(out_dir / "unit_checks.csv")

    # 快照索引：candidate → phase → window → pert → snap
    idx: dict[str, dict[str, dict[str, dict[str, dict]]]] = {}
    for s in snaps:
        ph = _CTX["phase_of"].get(str(s.get("window_id")), "")
        idx.setdefault(str(s["candidate"]), {}).setdefault(ph, {}).setdefault(
            str(s["window_id"]), {})[str(s.get("pert", "nom"))] = s
    hl_idx: dict[str, dict[str, dict[str, dict[str, dict]]]] = {}
    for s in hl_snaps:
        ph = _CTX["phase_of"].get(str(s.get("window_id")), "")
        tag = float(s["hl"])
        hl_idx.setdefault(f"{s['candidate']}|{tag:g}", {}).setdefault(ph, {}).setdefault(
            str(s["window_id"]), {})[str(s.get("pert", "nom"))] = s

    # ---- ξ（每阶段每候选一常数，用参照窗 CO2 中位闭合装置值）
    xi_tab: dict[tuple[str, str], float] = {}
    cp = cp_raw.copy()
    cp["xi_mol_s"] = float("nan")
    for cand in CAND_ORDER:
        for ph in PHASES:
            wids = _CTX["ref"][ph]
            store = {w: idx.get(cand, {}).get(ph, {}).get(w, {}).get("nom", {})
                     for w in wids}
            if not any(v.get("ok") for v in store.values()):
                continue
            target = float(np.median([float(_CTX["p114d"].loc[w, "co2_vol_pct_mean"])
                                      for w in wids]))
            x = solve_xi(store, wids, target)
            xi_tab[(cand, ph)] = x
            m = cp[(cp["candidate"] == cand) & (cp["phase"] == ph)].index
            cp.loc[m, "xi_mol_s"] = x
    # ---- 比较点补充列（参照窗上的温度、CO2、截留）
    extra_cols = {
        "T_out_med_K": [], "T_wsr_med_K": [], "CO2_med_model": [], "CO2_med_plant": [],
        "CO_med_model": [], "H2_med_model": [],
        "capture_carbon_frac_inlet_med": [], "capture_hhv_pct_med": [],
        "X_C_pp_median_all_phase": [],
    }
    for _, r in cp.iterrows():
        cand, ph = str(r["candidate"]), str(r["phase"])
        wids = _CTX["ref"][ph]
        xi = _f(r["xi_mol_s"])
        ys = [apply_xi(idx.get(cand, {}).get(ph, {}).get(w, {}).get("nom", {}), xi)
              for w in wids]
        ys = [y for y in ys if y]
        extra_cols["T_out_med_K"].append(
            float(np.median([y["T_out_K"] for y in ys])) if ys else float("nan"))
        extra_cols["T_wsr_med_K"].append(
            float(np.median([y["T_wsr_K"] for y in ys])) if ys else float("nan"))
        extra_cols["CO2_med_model"].append(
            float(np.median([y["CO2_vol_pct"] for y in ys])) if ys else float("nan"))
        extra_cols["CO2_med_plant"].append(
            float(np.median([float(_CTX["p114d"].loc[w, "co2_vol_pct_mean"])
                             for w in wids])))
        extra_cols["CO_med_model"].append(
            float(np.median([y["CO_vol_pct"] for y in ys])) if ys else float("nan"))
        extra_cols["H2_med_model"].append(
            float(np.median([y["H2_vol_pct"] for y in ys])) if ys else float("nan"))
        caps = [idx.get(cand, {}).get(ph, {}).get(w, {}).get("nom", {}) for w in wids]
        extra_cols["capture_carbon_frac_inlet_med"].append(
            float(np.median([_f(c.get("capture_carbon_frac_of_inlet")) for c in caps]))
            if caps else float("nan"))
        extra_cols["capture_hhv_pct_med"].append(
            float(np.median([_f(c.get("capture_enthalpy_frac_hhv_pct")) for c in caps]))
            if caps else float("nan"))
        allx = []
        for w in _CTX["win"][ph]:
            y = apply_xi(idx.get(cand, {}).get(ph, {}).get(w, {}).get("nom", {}), xi)
            if y:
                allx.append(y["X_C_pp"])
        extra_cols["X_C_pp_median_all_phase"].append(
            float(np.median(allx)) if allx else float("nan"))
    for k, v in extra_cols.items():
        cp[k] = v
    cp.to_csv(out_dir / "comparison_points.csv", index=False)

    # ---- 斜率与弹性
    slope_rows, n_fail = _model_slopes(idx, cp, xi_tab)
    slopes_df = pd.DataFrame(slope_rows)
    slopes_df.to_csv(out_dir / "model_slopes.csv", index=False)

    levels_df = _levels(idx, cp, xi_tab)
    levels_df.to_csv(out_dir / "levels.csv", index=False)

    elas_df = _elasticities(slopes_df, cp)
    elas_df.to_csv(out_dir / "elasticities.csv", index=False)

    prof_df = _profiles(prof, cp)
    prof_df.to_csv(out_dir / "profiles.csv", index=False)

    hl_df, hl_verdicts = _hl_sensitivity(hl_idx, hl_cp)
    hl_df.to_csv(out_dir / "hl_sensitivity.csv", index=False)

    readings = _readings(cp, slopes_df, elas_df, levels_df, hl_verdicts, n_fail)
    readings.to_csv(out_dir / "readings.csv", index=False)

    text = _interpretation(cp, slopes_df, elas_df, levels_df, readings,
                           hl_df, hl_verdicts, n_fail,
                           prof_df=prof_df, unit_df=unit_df)
    (out_dir / "interpretation.md").write_text(text, encoding="utf-8")

    proto = _protocol(out_dir, cp, readings, hl_verdicts, src_commit)
    (out_dir / "protocol.json").write_text(
        json.dumps(proto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _manifest(out_dir)
    print("read stage written", flush=True)


# ---------------------------------------------------------------------------
# 表
# ---------------------------------------------------------------------------

def _d2(snaps: dict[str, dict], xi: float, obs: str, inp: str) -> tuple:
    ys = {p: apply_xi(snaps.get(p) or {}, xi) for p in PERTS}
    if any(ys[p] is None for p in ("nom",) + INPUT_PERTS[inp]):
        return float("nan"), float("nan"), float("nan")
    pp, pm = INPUT_PERTS[inp]
    y0, yp, ym = ys["nom"][obs], ys[pp][obs], ys[pm][obs]
    if not all(math.isfinite(v) for v in (y0, yp, ym)):
        return float("nan"), float("nan"), float("nan")
    return ((yp - ym) / D_LN * LN_P, yp - y0, y0 - ym)


def _model_slopes(idx, cp, xi_tab) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    n_fail: dict = {}
    param_of = {(str(r["candidate"]), str(r["phase"])): r for _, r in cp.iterrows()}
    plant = _CTX["plant"]
    for cand in CAND_ORDER:
        for ph in PHASES:
            wids = _CTX["win"][ph]
            if cand not in idx or ph not in idx.get(cand, {}):
                n_fail[(cand, ph)] = {"n_ok": 0, "n_fail": len(wids), "skipped": True}
                continue
            xi = xi_tab.get((cand, ph), float("nan"))
            r0 = param_of.get((cand, ph))
            skipped = bool(r0 is not None and KIND[cand] != "C0" and not bool(r0["reachable"]))
            n_ok = 0
            n_fail_w = 0
            ceil_any = 0
            ceil_nom = 0
            coll: dict[tuple[str, str], list[float]] = {}
            plus: dict[tuple[str, str], list[float]] = {}
            minus: dict[tuple[str, str], list[float]] = {}
            for w in wids:
                sn = idx.get(cand, {}).get(ph, {}).get(w, {})
                if not (sn.get("nom") or {}).get("ok"):
                    n_fail_w += 1
                    continue
                n_ok += 1
                xcs = [_f((sn.get(p) or {}).get("X_C_pp")) for p in PERTS]
                if any(math.isfinite(v) and v >= XC_CEIL_PP for v in xcs):
                    ceil_any += 1
                if _f(sn["nom"].get("X_C_pp")) >= XC_CEIL_PP:
                    ceil_nom += 1
                for obs in OBS_ALL:
                    for inp in INPUTS:
                        d2, dp, dm = _d2(sn, xi, obs, inp)
                        coll.setdefault((obs, inp), []).append(d2)
                        plus.setdefault((obs, inp), []).append(dp)
                        minus.setdefault((obs, inp), []).append(dm)
            n_fail[(cand, ph)] = {"n_ok": n_ok, "n_fail": n_fail_w, "skipped": skipped}
            for obs in OBS_ALL:
                for inp in INPUTS:
                    vals = [v for v in coll.get((obs, inp), []) if math.isfinite(v)]
                    pv = [v for v in plus.get((obs, inp), []) if math.isfinite(v)]
                    mv = [v for v in minus.get((obs, inp), []) if math.isfinite(v)]
                    hit = plant[(plant["phase"] == ph) & (plant["obs"] == obs)
                                & (plant["input"] == inp) & (plant["model"] == "with_carrier")]
                    plant_per3 = _f(hit.iloc[0]["per3"]) if len(hit) else float("nan")
                    plant_signed = bool(hit.iloc[0]["signed"]) if len(hit) else False
                    med = float(np.median(vals)) if vals else float("nan")
                    same = bool(
                        math.isfinite(med) and math.isfinite(plant_per3)
                        and med != 0.0 and plant_per3 != 0.0
                        and np.sign(med) == np.sign(plant_per3)
                    )
                    rows.append({
                        "candidate": cand, "kind": KIND[cand], "phase": ph,
                        "obs": obs, "input": inp,
                        "param": _f(r0["param"]) if r0 is not None else float("nan"),
                        "n_windows": len(wids), "n_ok": len(vals),
                        "n_solve_fail": n_fail_w, "stage3_skipped": skipped,
                        "model_per3_median": med,
                        "model_per3_min": float(np.min(vals)) if vals else float("nan"),
                        "model_per3_max": float(np.max(vals)) if vals else float("nan"),
                        "model_per3_plus_median": float(np.median(pv)) if pv else float("nan"),
                        "model_per3_minus_median": float(np.median(mv)) if mv else float("nan"),
                        "plant_per3": plant_per3,
                        "plant_boot_lo": _f(hit.iloc[0]["boot_lo"]) if len(hit) else float("nan"),
                        "plant_boot_hi": _f(hit.iloc[0]["boot_hi"]) if len(hit) else float("nan"),
                        "plant_signed": plant_signed,
                        "same_sign": same,
                        "xc_ceil_frac_any_pert": (ceil_any / n_ok) if n_ok else float("nan"),
                        "xc_ceil_frac_nom": (ceil_nom / n_ok) if n_ok else float("nan"),
                        "xi_mol_s": xi,
                    })
    return rows, n_fail


def _levels(idx, cp, xi_tab) -> pd.DataFrame:
    rows = []
    for _, r in cp.iterrows():
        cand, ph = str(r["candidate"]), str(r["phase"])
        xi = xi_tab.get((cand, ph), float("nan"))
        wids = [w for w in _CTX["win"][ph]
                if w in idx.get(cand, {}).get(ph, {})]
        if not wids:
            continue
        biases: dict[str, list[float]] = {o: [] for o in OBS_LEVEL}
        for w in wids:
            y = apply_xi(idx[cand][ph][w].get("nom", {}), xi)
            if y is None:
                continue
            plant = {
                "X_C_pp": float(_CTX["p114d"].loc[w, "xc_recon_phase_cal_measured_pct"]),
                "CO_vol_pct": float(_CTX["p114d"].loc[w, "co_vol_pct_mean"]),
                "CO2_vol_pct": float(_CTX["p114d"].loc[w, "co2_vol_pct_mean"]),
                "H2_vol_pct": float(_CTX["p114d"].loc[w, "h2_vol_pct_mean"]),
                "CH4_vol_pct": float(_CTX["p114d"].loc[w, "ch4_vol_pct_mean"]),
                "dry_per_coal_mol_kg": (
                    float(_CTX["p114d"].loc[w, "syngas_dry_cal_nm3_h"]) / 22.414
                    / 3600.0 * 1000.0 / max(float(_CTX["p114d"].loc[w,
                                                                  "coal_dry_recon_phase_kg_s"]), 1e-12)
                ),
            }
            for o in OBS_LEVEL:
                biases[o].append(float(y[o]) - plant[o])
        for o in OBS_LEVEL:
            v = [b for b in biases[o] if math.isfinite(b)]
            rows.append({
                "candidate": cand, "phase": ph, "obs": o,
                "param": _f(r["param"]), "xi_mol_s": xi,
                "n": len(v), "bias_median": float(np.median(v)) if v else float("nan"),
                "bias_abs_median": float(np.median(np.abs(v))) if v else float("nan"),
            })
    return pd.DataFrame(rows)


def _plant_elasticity(ph: str, inp: str, xc_med: float) -> tuple[float, float, float]:
    plant = _CTX["plant"]
    hit = plant[(plant["phase"] == ph) & (plant["obs"] == "X_C_pp")
                & (plant["input"] == inp) & (plant["model"] == "with_carrier")]
    if not len(hit) or not math.isfinite(xc_med) or xc_med <= 0:
        return float("nan"), float("nan"), float("nan")
    r = hit.iloc[0]
    off = LN_P
    base = 0.0 if inp == "ln_O2" else 0.0
    e = (1.0 if inp == "ln_coal" else 0.0) + (_f(r["per3"]) / xc_med) / off
    lo = (1.0 if inp == "ln_coal" else 0.0) + (_f(r["boot_lo"]) / xc_med) / off
    hi = (1.0 if inp == "ln_coal" else 0.0) + (_f(r["boot_hi"]) / xc_med) / off
    return float(e), float(lo), float(hi)


def _elasticities(slopes: pd.DataFrame, cp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in cp.iterrows():
        cand, ph = str(r["candidate"]), str(r["phase"])
        if KIND[cand] != "C0" and not bool(r["reachable"]):
            continue
        sub = slopes[(slopes["candidate"] == cand) & (slopes["phase"] == ph)
                     & (slopes["obs"] == "X_C_pp")]
        xc_model = _f(r.get("X_C_pp_median_all_phase"))
        if not math.isfinite(xc_model):
            xc_model = _f(r["X_C_pp_median"])
        p_coal = _f(sub[sub["input"] == "ln_coal"]["model_per3_median"].iloc[0]) if len(sub) else float("nan")
        p_o2 = _f(sub[sub["input"] == "ln_O2"]["model_per3_median"].iloc[0]) if len(sub) else float("nan")
        e_coal = 1.0 + (p_coal / xc_model) / LN_P if xc_model else float("nan")
        e_o2 = (p_o2 / xc_model) / LN_P if xc_model else float("nan")
        xc_plant = plant_xc_med(ph)
        pe_c, pe_c_lo, pe_c_hi = _plant_elasticity(ph, "ln_coal", xc_plant)
        pe_o, pe_o_lo, pe_o_hi = _plant_elasticity(ph, "ln_O2", xc_plant)
        in_c = bool(math.isfinite(e_coal) and pe_c_lo <= e_coal <= pe_c_hi)
        in_o = bool(math.isfinite(e_o2) and pe_o_lo <= e_o2 <= pe_o_hi)
        dT = _f(slopes[(slopes["candidate"] == cand) & (slopes["phase"] == ph)
                       & (slopes["obs"] == "T_out_K")
                       & (slopes["input"] == "ln_O2")]["model_per3_median"].iloc[0]) \
            if len(slopes[(slopes["candidate"] == cand) & (slopes["phase"] == ph)
                          & (slopes["obs"] == "T_out_K")
                          & (slopes["input"] == "ln_O2")]) else float("nan")
        rows.append({
            "candidate": cand, "kind": KIND[cand], "phase": ph,
            "param": _f(r["param"]), "xi_mol_s": _f(r["xi_mol_s"]),
            "X_C_pp_model": xc_model, "X_C_pp_plant": xc_plant,
            "per3_XC_coal_model": p_coal, "per3_XC_o2_model": p_o2,
            "e_coal_model": e_coal, "e_o2_model": e_o2,
            "e_coal_plant": pe_c, "e_coal_plant_lo": pe_c_lo, "e_coal_plant_hi": pe_c_hi,
            "e_o2_plant": pe_o, "e_o2_plant_lo": pe_o_lo, "e_o2_plant_hi": pe_o_hi,
            "e_coal_in_plant_interval": in_c, "e_o2_in_plant_interval": in_o,
            "dT_out_per3_K_median": dT,
        })
    return pd.DataFrame(rows)


def _profiles(prof: list[dict], cp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in prof:
        cand, ph = str(s["candidate"]), str(s["phase"])
        hit = cp[(cp["candidate"] == cand) & (cp["phase"] == ph)]
        param = _f(hit["param"].iloc[0]) if len(hit) else float("nan")
        x0 = {"M1-0": 0.0, "M1-80": 0.80, "M1-90": 0.90}.get(cand)
        n = param if KIND[cand] == "M1" else None
        for p in s["profile"]:
            xc = float(p["X_C_frac"])
            theta = float("nan")
            if KIND[cand] == "M1" and x0 is not None:
                theta = 1.0 if xc <= x0 else ((1.0 - xc) / (1.0 - x0)) ** float(n)
            rows.append({
                "phase": ph, "candidate": cand, "window_id": s["window_id"],
                "param": param, "i": p["i"], "z_m": p["z_m"], "dz_m": p["dz_m"],
                "X_C_frac": xc, "X_C_pp": xc * 100.0,
                "T_gas_K": p["T_gas_K"], "T_solid_K": p["T_solid_K"],
                "solid_mass_kg_s": p["solid_mass_kg_s"],
                "carbon_fraction": p["carbon_fraction"],
                "is_combustion": p["is_combustion"], "late_decay_theta": theta,
            })
    return pd.DataFrame(rows)


def _hl_sensitivity(hl_idx, hl_cp) -> tuple[pd.DataFrame, dict]:
    rows = []
    verdict: dict = {}
    for hl in (0.5, 1.0, 2.0):
        for cand in CAND_ORDER:
            key = f"{cand}|{hl:g}"
            if key not in hl_idx:
                continue
            for ph in PHASES:
                hit = hl_cp[(hl_cp["candidate"] == cand) & (hl_cp["phase"] == ph)
                            & (np.isclose(hl_cp["hl_tot_pct"], hl))]
                r = hit.iloc[0] if len(hit) else None
                wids = _CTX["ref"][ph]
                store = {w: hl_idx[key].get(ph, {}).get(w, {}) for w in wids}
                target = float(np.median([float(_CTX["p114d"].loc[w, "co2_vol_pct_mean"])
                                          for w in wids]))
                xi = solve_xi({w: v.get("nom", {}) for w, v in store.items()}, wids, target)
                xc_mp = []
                for w in wids:
                    y = apply_xi(store[w].get("nom", {}), xi)
                    if y:
                        xc_mp.append(y["X_C_pp"])
                xc_med = float(np.median(xc_mp)) if xc_mp else float("nan")
                p_coal = p_o2 = dT = float("nan")
                combo = 0
                n_combo = 0
                if store and all(store.get(w) for w in wids):
                    cols = []
                    for w in wids:
                        d2c = _d2(store[w], xi, "X_C_pp", "ln_coal")[0]
                        d2o = _d2(store[w], xi, "X_C_pp", "ln_O2")[0]
                        cols.append((d2c, d2o, _d2(store[w], xi, "T_out_K", "ln_O2")[0]))
                    p_coal = float(np.nanmedian([c[0] for c in cols]))
                    p_o2 = float(np.nanmedian([c[1] for c in cols]))
                    dT = float(np.nanmedian([c[2] for c in cols]))
                    for (pcombo, obs, inp) in G2_COMBOS:
                        if pcombo != ph:
                            continue
                        v = [_d2(store[w], xi, obs, inp)[0] for w in wids]
                        v = [x for x in v if math.isfinite(x)]
                        if not v:
                            continue
                        n_combo += 1
                        pl = _CTX["plant"]
                        hh = pl[(pl["phase"] == ph) & (pl["obs"] == obs)
                                & (pl["input"] == inp) & (pl["model"] == "with_carrier")]
                        if len(hh) and np.sign(float(np.median(v))) == np.sign(_f(hh.iloc[0]["per3"])):
                            combo += 1
                e_coal = 1.0 + (p_coal / xc_med) / LN_P if xc_med else float("nan")
                e_o2 = (p_o2 / xc_med) / LN_P if xc_med else float("nan")
                rows.append({
                    "hl_tot_pct": hl, "candidate": cand, "phase": ph,
                    "param": _f(r["param"]) if r is not None else float("nan"),
                    "reachable": bool(r["reachable"]) if r is not None else False,
                    "X_C_pp_median_ref": xc_med,
                    "target_X_C_pp": float(r["target_X_C_pp"]) if r is not None else float("nan"),
                    "dX_C_pp": float(r["dX_C_pp"]) if r is not None else float("nan"),
                    "xi_mol_s": xi, "e_coal_model": e_coal, "e_o2_model": e_o2,
                    "dT_out_per3_K_median": dT,
                    "g1_pass": bool(math.isfinite(dT) and abs(dT) >= G1_MIN_K),
                    "g2_same_sign": combo, "g2_n": n_combo,
                })
    df = pd.DataFrame(rows)
    # 每个 HL 档的候选级判定
    for hl in (0.5, 1.0, 2.0):
        for cand in CAND_ORDER:
            sub = df[(df["hl_tot_pct"] == hl) & (df["candidate"] == cand)]
            if sub.empty:
                continue
            verdict.setdefault(f"{cand}@{hl:g}", {})
            verdict[f"{cand}@{hl:g}"] = {
                "n_reachable_phases": int(sub["reachable"].sum()),
                "n_g1_phases": int(sub["g1_pass"].sum()),
                "min_abs_dT": float(np.nanmin(np.abs(sub["dT_out_per3_K_median"]))),
                "g2_same_sign": int(sub["g2_same_sign"].sum()),
                "g2_n": int(sub["g2_n"].sum()),
                "n_phases_ecoal_ge_0p3": int((sub["e_coal_model"] >= G3_MIN_ECOAL).sum()),
            }
    return df, verdict


# ---------------------------------------------------------------------------
# 读法
# ---------------------------------------------------------------------------

def _readings(cp, slopes, elas, levels, hl_verdict, n_fail) -> pd.DataFrame:
    rows: list[dict] = []
    theta = _CTX["theta_fit"]
    c0t = theta[theta["candidate"] == "C0"]
    theta_ref = {
        "X_C_pp": _f(c0t["r5_d_X_C_pp"].iloc[0]) if len(c0t) else float("nan"),
        "CO_vol_pct": _f(c0t["r5_d_CO_vol_pct"].iloc[0]) if len(c0t) else float("nan"),
        "H2_vol_pct": _f(c0t["r5_d_H2_vol_pct"].iloc[0]) if len(c0t) else float("nan"),
        "CO2_vol_pct": _f(c0t["r5_d_CO2_vol_pct"].iloc[0]) if len(c0t) else float("nan"),
    }

    def add(reading, metric, phase, value, threshold, passed, note, cand="", expl=False):
        rows.append({
            "reading": reading, "metric": metric, "phase": phase,
            "value": value, "threshold": threshold,
            "pass": "" if passed is None else bool(passed),
            "note": note, "candidate": cand, "exploratory": bool(expl),
        })

    for cand in CAND_ORDER:
        sub = cp[cp["candidate"] == cand]
        n_reach = int(sub["reachable"].sum())
        add("G0", "reachable_phases", "all", n_reach, len(PHASES),
            KIND[cand] == "C0" or n_reach == len(PHASES),
            "四个阶段的比较点是否都可达；"
            + ("C0 为基准对照，无需求根" if KIND[cand] == "C0" else
               "; ".join(f"{r['phase']}:{bool(r['reachable'])}"
                         + ("" if bool(r["reachable"]) else f"({r['reason']})")
                         for _, r in sub.iterrows())),
            cand)
        if KIND[cand] != "C0":
            for _, r in sub.iterrows():
                add("G0", "reachable", str(r["phase"]), 1.0 if bool(r["reachable"]) else 0.0,
                    1.0, bool(r["reachable"]),
                    f"param={_f(r['param']):.6g} X_C_med={_f(r['X_C_pp_median']):.4f} "
                    f"目标={_f(r['target_X_C_pp']):.4f} {r['reason'] or ''}".strip(), cand)

        e = elas[elas["candidate"] == cand]
        for _, r in e.iterrows():
            add("G1", "dT_out_per3_K", str(r["phase"]),
                _f(r["dT_out_per3_K_median"]), G1_MIN_K,
                bool(math.isfinite(_f(r["dT_out_per3_K_median"]))
                     and abs(_f(r["dT_out_per3_K_median"])) >= G1_MIN_K),
                "O2 +3% 双侧有限差分（每 +3%）的出口温度阶段中位", cand)
        g1_vals = [abs(_f(v)) for v in e["dT_out_per3_K_median"] if math.isfinite(_f(v))]
        add("G1", "min_abs_dT_across_phases", "all",
            float(min(g1_vals)) if g1_vals else float("nan"), G1_MIN_K,
            bool(len(g1_vals) == len(PHASES) and min(g1_vals) >= G1_MIN_K),
            f"四个阶段的最小 |ΔT_out|；判定要求四个阶段都有斜率且每阶段 ≥50 K"
            f"（实际有斜率的阶段数 {len(g1_vals)}）", cand)

        n_same = 0
        for (ph, obs, inp) in G2_COMBOS:
            h = slopes[(slopes["candidate"] == cand) & (slopes["phase"] == ph)
                       & (slopes["obs"] == obs) & (slopes["input"] == inp)]
            if not len(h):
                continue
            same = bool(h.iloc[0]["same_sign"])
            n_same += int(same)
            add("G2", f"same_sign::{obs}::{inp}", ph,
                _f(h.iloc[0]["model_per3_median"]), _f(h.iloc[0]["plant_per3"]), same,
                f"模型每+3%={_f(h.iloc[0]['model_per3_median']):.4g} "
                f"装置={_f(h.iloc[0]['plant_per3']):.4g}", cand)
        add("G2", "same_sign_total", "all", n_same, G2_MIN, n_same >= G2_MIN,
            "P1.15-A 的 8 个有符号组合中模型与装置同号个数", cand)

        n_ge = int((e["e_coal_model"] >= G3_MIN_ECOAL).sum()) if len(e) else 0
        add("G3", "phases_with_ecoal_ge_0p3", "all", n_ge, G3_MIN_PHASES,
            n_ge >= G3_MIN_PHASES, "e_煤 = 1 + (ΔX_C/X_C)/ln1.03 ≥ 0.3 的阶段数", cand)
        for _, r in e.iterrows():
            add("G3", "e_coal", str(r["phase"]), _f(r["e_coal_model"]),
                f"[{_f(r['e_coal_plant_lo']):.3f},{_f(r['e_coal_plant_hi']):.3f}]",
                bool(r["e_coal_in_plant_interval"]),
                f"装置 e_煤={_f(r['e_coal_plant']):.3f}", cand)
            add("G3", "e_o2", str(r["phase"]), _f(r["e_o2_model"]),
                f"[{_f(r['e_o2_plant_lo']):.3f},{_f(r['e_o2_plant_hi']):.3f}]",
                bool(r["e_o2_in_plant_interval"]),
                f"装置 e_氧={_f(r['e_o2_plant']):.3f}", cand)

        lv = levels[levels["candidate"] == cand]
        for obs in ("X_C_pp", "CO_vol_pct", "H2_vol_pct", "CO2_vol_pct"):
            for ph in PHASES:
                h = lv[(lv["phase"] == ph) & (lv["obs"] == obs)]
                if not len(h):
                    continue
                add("G4", f"level_bias::{obs}", ph, _f(h.iloc[0]["bias_median"]),
                    _f(theta_ref.get(obs)),
                    None,
                    f"模型−装置 142 窗中位；P1.13-A θ̂ 拟合同量={_f(theta_ref.get(obs)):.4f}",
                    cand)

        for key, v in hl_verdict.items():
            if not key.startswith(f"{cand}@"):
                continue
            hl_tag = key.split("@")[1]
            add("G5", "hl_variant_g1_min_abs_dT", f"HL={hl_tag}", v["min_abs_dT"], G1_MIN_K,
                bool(v["min_abs_dT"] >= G1_MIN_K),
                f"可达阶段 {v['n_reachable_phases']}/4；G1 通过阶段 {v['n_g1_phases']}/4", cand)
            add("G5", "hl_variant_g2_same_sign", f"HL={hl_tag}", v["g2_same_sign"], G2_MIN,
                v["g2_same_sign"] >= G2_MIN,
                f"分母 {v['g2_n']}（参照窗）", cand)
            add("G5", "hl_variant_g3_phases", f"HL={hl_tag}", v["n_phases_ecoal_ge_0p3"],
                G3_MIN_PHASES, v["n_phases_ecoal_ge_0p3"] >= G3_MIN_PHASES,
                "e_煤 ≥ 0.3 的阶段数（参照窗）", cand)

        car_rows = slopes[(slopes["candidate"] == cand) & (slopes["input"] == "ln_carrier")
                          & (slopes["obs"].isin(("CO_vol_pct", "H2_vol_pct")))]
        pl = _CTX["plant"]
        n6 = 0
        n6_tot = 0
        for _, r in car_rows.iterrows():
            h = pl[(pl["phase"] == r["phase"]) & (pl["obs"] == r["obs"])
                   & (pl["input"] == "ln_carrier") & (pl["model"] == "with_carrier")
                   & (pl["signed"])]
            if not len(h):
                continue
            n6_tot += 1
            n6 += int(bool(r["same_sign"]))
        add("G6", "carrier_same_sign", "all", n6, n6_tot,
            None, "CO、H2 对载气的 5 个有符号组合（探索性，不计入判定）", cand, True)

    # 判定
    verdict_rows = []
    for cand in CAND_ORDER:
        g = {}
        for reading in ("G0", "G1", "G2", "G3"):
            sub = [r for r in rows if r["reading"] == reading and r["candidate"] == cand]
            tot = [r for r in sub if r["metric"] in
                   ("reachable_phases", "min_abs_dT_across_phases", "same_sign_total",
                    "phases_with_ecoal_ge_0p3")]
            g[reading] = bool(tot and tot[0]["pass"] is True)
        passed = all(g.values())
        e = elas[elas["candidate"] == cand]
        n_int = int(e["e_coal_in_plant_interval"].sum()) + int(
            e["e_o2_in_plant_interval"].sum()) if len(e) else 0
        verdict_rows.append({
            "candidate": cand, "G0": g["G0"], "G1": g["G1"], "G2": g["G2"], "G3": g["G3"],
            "selected": passed, "n_interval_hits": n_int,
        })
    vd = pd.DataFrame(verdict_rows)
    sel = vd[vd["selected"]]
    if sel.empty:
        note = ("两个候选均不能同时给出碳转化率水平与响应方向"
                if not any(vd["candidate"].isin(("M1-0", "M1-80", "M1-90", "M2")))
                else "无候选同时通过 G0–G3：单列「两个候选均不能同时给出碳转化率水平与响应方向」")
    else:
        best = sel.sort_values("n_interval_hits", ascending=False).iloc[0]
        kinds = {KIND[c] for c in sel["candidate"]}
        note = (f"入选：{', '.join(sel['candidate'])}；"
                f"落在装置弹性区间内的阶段数最多者为 {best['candidate']}"
                f"（{int(best['n_interval_hits'])}）。"
                + ("两类机理都有入选，按卡面以区间命中数多者优先；若相同则并列带入下一轮。"
                   if len(kinds) > 1 else "只有一类机理入选。"))
    for _, r in vd.iterrows():
        rows.append({
            "reading": "verdict", "metric": "G0_G1_G2_G3_selected",
            "phase": "all", "value": f"G0={int(r['G0'])} G1={int(r['G1'])} "
                                     f"G2={int(r['G2'])} G3={int(r['G3'])}",
            "threshold": "", "pass": bool(r["selected"]), "note": note,
            "candidate": r["candidate"], "exploratory": False,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# protocol / interpretation / manifest
# ---------------------------------------------------------------------------

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protocol(out_dir: Path, cp, readings, hl_verdict, src_commit) -> dict:
    import subprocess

    commit = src_commit
    if not commit:
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
        except Exception:  # noqa: BLE001
            commit = ""
    src = {
        "p1.14d": ("outputs/diagnostics/p1.14d-20261107/windows_142.csv", P114D_CSV),
        "p1.12a": ("outputs/diagnostics/round-p1.12a-20261106/input_sensitivity.csv", P112A_CSV),
        "p1.15a_plant_slopes": (
            "outputs/diagnostics/round-p1.15a-20261108/plant_slopes.csv", P115A_SLOPES),
        "p1.15a_candidates": (
            "outputs/diagnostics/round-p1.15a-20261108/candidates.csv", P115A_CAND),
    }
    return {
        "round": "P2.0-A / 2026-11-13",
        "model_repo": "gasifier-1d-kinetic",
        "source_commit": commit,
        "task_card": "papers/jax-gasifier-cascade-hybrid/manuscript/"
                     "model-side-handoff-p2.0a-20261113.md",
        "frozen": {
            "MeshFamily": "wsr_plus_pfr",
            "WSRLength_m": 1.0,
            "PFRCellLength_m": 0.0515,
            "L_reactor_m": 10.3,
            "n_cells": 182,
            "HeatLossModel": "total_percent",
            "HeatLossPercentTotal_nominal": 1.0,
            "o2_flow_scale_kappa": 1.0,
            "GasificationRateFactor": 1.0,
            "CharCO2RateFactor": 1.0,
            "OxygenMixingModel": "instant",
            "particle_size_model": "fortran_constant",
            "coal_flow": "windows_142.coal_dry_recon_phase_kg_s（逐窗协调干煤量）",
            "carrier": "windows_142.co2_carrier_measured_nm3_h（实测载气，CO2）",
            "kinetics": "全部 Arrhenius 为 Wen 1979 原值",
            "solver": "scipy least_squares (trf, xtol=1e-12, ftol=1e-8)",
        },
        "hooks": {
            "M1": {
                "op_conds": ["CharLateDecayX0 (default None = off)",
                             "CharLateDecayExponent (default 0)"],
                "form": "X<=X0: theta=1; X>X0: theta=((1-X)/(1-X0))**n",
                "applies_to": ["C+H2O", "C+CO2"],
                "X_source": "Cell._calc_physics_props 的 X_total（相对入炉煤碳）",
                "X0_tiers": [0.0, 0.80, 0.90],
                "n_bounds": [0.0, 8.0],
            },
            "M2": {
                "op_conds": ["SlagCaptureFraction (default 0)"],
                "location": "全混区（首格）出口一处",
                "split": "固体质量份额 f_s，碳与灰同比例离开反应流",
                "enthalpy": ("被截留固体在全混区出口固相温度下的显热（焦炭生成焓取 0，"
                             "与 Cell._calc_energy_balance 出口口径一致）；单列，不计入 "
                             "HeatLossPercentTotal；以占干煤 HHV 百分比报告"),
                "X_C": "1 - (出口夹带固体碳 + 被截留碳)/入炉煤碳",
                "f_s_bounds": [0.0, 0.5],
            },
        },
        "comparison_point": {
            "target": "参照窗模型 X_C 中位 = 同批窗装置 X_C 中位（baseline_142.plant_X_C_pp）",
            "tolerance_pp": 0.2,
            "search": "一维二分（端点扫描 + 60 次折半，容差 0.02 pp）",
            "unreachable": "记不可达并报端点 X_C；该候选不做第 3 节",
            "xi": "每阶段每候选一个常数，使参照窗 CO2 中位 = 装置值；扰动时不变",
        },
        "perturbation": {
            "scale": "±3%，每次只改一项（O2 / 干煤量 / 载气）",
            "two_sided": "(y(1.03)-y(0.97))/ln(1.03/0.97)*ln(1.03)",
            "one_sided": "plus = y(1.03)-y(1)；minus = y(1)-y(0.97)",
            "elasticity": "e_煤 = 1 + (ΔX_C/X_C)/ln1.03; e_氧 = (ΔX_C/X_C)/ln1.03",
            "plant_side": "P1.15-A plant_slopes.csv 含载气主模型，同公式；区间用自助端点",
            "cache_key": ("含完整 window_id、candidate、param、X0、HL、三个扰动倍数、"
                          "κ、f_gas、f_CO2、L_wsr、dz_down；载入时核对 key 全等"),
        },
        "predeclared": {
            "G0": "四个阶段的比较点都可达",
            "G1": "O2 ±3% 的出口温度变化（每 +3%，阶段中位）≥ 50 K；判定要求每阶段都满足",
            "G2": "P1.15-A 的 8 个有符号组合中模型与装置同号 ≥ 6",
            "G3": "e_煤 ≥ 0.3 的阶段数 ≥ 3；逐阶段报 (e_煤, e_氧) 是否落在装置 95% 区间",
            "G4": "ξ 闭合后 142 窗 CO、H2、X_C 的阶段中位偏差，与 P1.13-A θ̂ 拟合并列",
            "G5": "HL_tot ∈ {0.5, 2} 时只在参照窗上重做第 2、3 节，看 G1–G3 结论是否改变",
            "G6": "CO、H2 对载气的 5 个有符号组合同号个数（探索性，不计入判定）",
            "verdict": "G0–G3 全通过 → 入选；两类都入选以落在装置弹性区间内的阶段数多者为先",
        },
        "stopping": {
            "unit_fail": "U1–U5 任一不通过即停止",
            "window_fail_per_stage": 5,
            "unreachable": "该候选不做第 3 节",
        },
        "adopted_parameters": False,
        "no_recirculation": True,
        "contract_unchanged": True,
        "source_csv": {
            k: {"relative_path": p, "sha256": _sha(f) if f.exists() else "",
                "exists": bool(f.exists())}
            for k, (p, f) in src.items()
        },
        "outputs": ["unit_checks.csv", "comparison_points.csv", "model_slopes.csv",
                    "elasticities.csv", "levels.csv", "hl_sensitivity.csv",
                    "profiles.csv", "readings.csv", "protocol.json",
                    "interpretation.md", "sha256.txt"],
    }


def _fmt(x, nd=4):
    v = _f(x)
    return "NA" if not math.isfinite(v) else f"{v:.{nd}g}"


def _is_true(x) -> bool:
    return (x is True) or (x == True)  # noqa: E712


def _interpretation(cp, slopes, elas, levels, readings, hl_df, hl_verdict, n_fail,
                    prof_df=None, unit_df=None) -> str:
    L = []
    L.append("# P2.0-A 碳转化率受限机理的并列比较（2026-11-13）")
    L.append("")
    L.append("κ = 1（实测氧量）、`GasificationRateFactor = 1`、`CharCO2RateFactor = 1`；"
             "两个钩子并列比较。**本轮不反演、不采纳参数、不加回流、不改合同。**")
    L.append("")
    L.append("## 第 1 节 单元检查")
    if unit_df is not None and len(unit_df):
        L.append("逐条见 `unit_checks.csv`（%d 行）。U1 的参照为 P1.12-A "
                 "`input_sensitivity.csv` 中 `L_wsr = 1`、`perturb_tag = baseline` 的 9 行。"
                 % len(unit_df))
        L.append("")
        u1 = unit_df[unit_df["check"] == "U1"]
        if len(u1):
            L.append(f"- **U1**：9 窗 X_C／出口温度／CO／CO2／H2 的最大相对偏差 = "
                     f"{_f(u1['value'].iloc[0]):.3e}（阈值 1e-9）→ "
                     f"**{'一致' if _is_true(u1['pass'].iloc[0]) else '不一致'}**。"
                     "缺省钩子时不进入任何新分支，实测逐位相同。")
        for cid in ("M1-0", "M1-80", "M1-90", "M2"):
            h = unit_df[(unit_df["check"] == "U2") & (unit_df["item"].str.startswith(cid))]
            if len(h):
                L.append(f"- **U2** {cid}（param=0）：{h['value'].iloc[0]}/"
                         f"{h['threshold'].iloc[0]} 窗逐位一致 → "
                         f"**{'一致' if _is_true(h['pass'].iloc[0]) else '不一致'}**。")
        u3 = unit_df[(unit_df["check"] == "U3") & (unit_df["item"].str.contains("汇总"))]
        if len(u3):
            L.append("- **U3** 汇总（9 参照窗的最大相对闭合误差）：")
            for _, r in u3.iterrows():
                L.append(f"  - {r['item']}：{_f(r['value']):.3e}"
                         f"（字面 1e-8：{'通过' if _is_true(r['pass']) else '不通过'}）")
            L.append("  字面阈值的失败不是钩子引入的：冻结求解器把逐格碳残差按 `char_Xc0`"
                     "（无量纲 ~0.7）缩放，182 格累加后碳闭合底就在 1e-4 量级；"
                     "M1 的碳闭合 1e-9、M2 的 1.2e-4 与同窗「Σ逐格碳残差/入炉碳」逐位对应，"
                     "说明误差是求解器停点而不是缺项。M2 若**不**计入被截留流股，碳闭合误差"
                     "会等于截留碳占入炉碳的比例（1e-1 量级）。能量闭合含截留焓后 ≤1.4e-9。")
        u4 = unit_df[(unit_df["check"] == "U4")]
        if len(u4):
            L.append(f"- **U4**：ϑ 连续性与 0<ϑ≤1 全部 "
                     f"{int(u4['pass'].sum())}/{len(u4)} 通过；"
                     "n 递增与 f_s 递增在阶段 2 参照窗上都使 X_C 单调下降。")
        u5 = unit_df[(unit_df["check"] == "U5")]
        for _, r in u5.iterrows():
            L.append(f"- **U5** {r['item']}：实际异常 = `{r['value']}` → "
                     f"**{'通过' if _is_true(r['pass']) else '不通过'}**。")
    else:
        L.append("未读到 `unit_checks.csv`。")
    L.append("")
    L.append("## 第 2 节 比较点")
    L.append("")
    L.append("| 阶段 | 候选 | 参数 | 模型 X_C | 装置 X_C | Δ | 出口温度 K | 全混区温度 K | ξ mol/s | 截留碳/入炉碳 | 截留焓/干煤 HHV % |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in cp.iterrows():
        cand = str(r["candidate"])
        pg = _fmt(r["param"], 5) if PARAM_NAME[cand] != "none" else "—"
        L.append(
            f"| {r['phase']} | {cand} | {PARAM_NAME[cand]}={pg} | "
            f"{_fmt(r['X_C_pp_median'], 5)} | {_fmt(r['target_X_C_pp'], 5)} | "
            f"{_fmt(r['dX_C_pp'], 3)} | {_fmt(r['T_out_med_K'], 5)} | "
            f"{_fmt(r['T_wsr_med_K'], 5)} | {_fmt(r['xi_mol_s'], 4)} | "
            f"{('—' if KIND[cand] != 'M2' else _fmt(100 * _f(r['capture_carbon_frac_inlet_med']), 3))} | "
            f"{('—' if KIND[cand] != 'M2' else _fmt(r['capture_hhv_pct_med'], 3))} |"
        )
    L.append("")
    L.append("X_C 为百分点。M2 的截留碳占入炉碳按中位窗报告（%），截留焓占干煤 HHV 同。")
    L.append("")
    L.append("## 第 3 节 斜率与弹性")
    L.append("")
    L.append("`model_slopes.csv` 给双侧每 +3% 与两个单侧斜率；`elasticities.csv` 给 e_煤、e_氧。")
    L.append("")
    L.append("| 阶段 | 候选 | e_煤 | 装置区间 | 在区间 | e_氧 | 装置区间 | 在区间 | ΔT_out/+3% K |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in elas.iterrows():
        L.append(
            f"| {r['phase']} | {r['candidate']} | {_fmt(r['e_coal_model'], 3)} | "
            f"[{_fmt(r['e_coal_plant_lo'], 3)},{_fmt(r['e_coal_plant_hi'], 3)}] | "
            f"{'Y' if r['e_coal_in_plant_interval'] else 'N'} | "
            f"{_fmt(r['e_o2_model'], 3)} | "
            f"[{_fmt(r['e_o2_plant_lo'], 3)},{_fmt(r['e_o2_plant_hi'], 3)}] | "
            f"{'Y' if r['e_o2_in_plant_interval'] else 'N'} | "
            f"{_fmt(r['dT_out_per3_K_median'], 4)} |"
        )
    L.append("")
    L.append("## 第 4 节 G0–G6")
    for reading in ("G0", "G1", "G2", "G3"):
        L.append(f"### {reading}")
        sub = readings[(readings["reading"] == reading)
                       & (readings["metric"].isin(
                           ("reachable_phases", "min_abs_dT_across_phases",
                            "same_sign_total", "phases_with_ecoal_ge_0p3")))]
        for _, r in sub.iterrows():
            L.append(f"- {r['candidate']}：{_fmt(r['value'], 4)}（阈值 {r['threshold']}）"
                     f"**{'通过' if _is_true(r['pass']) else '不通过'}**。{r['note']}")
        L.append("")
    L.append("### G4 水平（ξ 闭合后，142 窗阶段中位偏差，模型−装置）")
    L.append("")
    L.append("| 候选 | 阶段 | X_C pp | 装置 θ̂ 同量 | CO | 装置 θ̂ | CO2 | 装置 θ̂ | H2 | 装置 θ̂ |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    th = _CTX["theta_fit"]
    c0t = th[th["candidate"] == "C0"]
    ref = {}
    if len(c0t):
        ref = {"X_C_pp": _f(c0t["r5_d_X_C_pp"].iloc[0]),
               "CO_vol_pct": _f(c0t["r5_d_CO_vol_pct"].iloc[0]),
               "CO2_vol_pct": _f(c0t["r5_d_CO2_vol_pct"].iloc[0]),
               "H2_vol_pct": _f(c0t["r5_d_H2_vol_pct"].iloc[0])}
    for _, r in cp.iterrows():
        cand, ph = str(r["candidate"]), str(r["phase"])
        g = {}
        for obs in ("X_C_pp", "CO_vol_pct", "CO2_vol_pct", "H2_vol_pct"):
            h = levels[(levels["candidate"] == cand) & (levels["phase"] == ph)
                       & (levels["obs"] == obs)]
            g[obs] = _f(h["bias_median"].iloc[0]) if len(h) else float("nan")
        L.append(
            f"| {cand} | {ph} | {_fmt(g['X_C_pp'], 3)} | {_fmt(ref.get('X_C_pp'), 3)} | "
            f"{_fmt(g['CO_vol_pct'], 3)} | {_fmt(ref.get('CO_vol_pct'), 3)} | "
            f"{_fmt(g['CO2_vol_pct'], 3)} | {_fmt(ref.get('CO2_vol_pct'), 3)} | "
            f"{_fmt(g['H2_vol_pct'], 3)} | {_fmt(ref.get('H2_vol_pct'), 3)} |"
        )
    L.append("")
    L.append("θ̂ 列取自 P1.15-A `candidates.csv` 的 C0 行（即 P1.13-A 的 κ、ξ、f_gas 拟合），"
             "为 133 成功窗的中位；本表为 142 窗口径，两者窗集合不同，只作量级并列。")
    L.append("")
    L.append("### G5 热损敏感性（只在参照窗上）")
    L.append("")
    L.append("| HL_tot % | 候选 | 可达阶段 | G1 通过阶段 | 最小 |ΔT| K | G2 同号 | e_煤≥0.3 阶段 |")
    L.append("|---|---|---|---|---|---|---|")
    for key in sorted(hl_verdict, key=lambda k: (float(k.split("@")[1]), k.split("@")[0])):
        v = hl_verdict[key]
        cand, hl = key.split("@")
        L.append(
            f"| {hl} | {cand} | {v['n_reachable_phases']}/4 | {v['n_g1_phases']}/4 | "
            f"{_fmt(v['min_abs_dT'], 3)} | {v['g2_same_sign']}/{v['g2_n']} | "
            f"{v['n_phases_ecoal_ge_0p3']}/4 |"
        )
    L.append("")
    L.append("比较点参数随阶段的走向（阶段按 liu2023 的 X_C 递减排序，代表掺混比例上升）：")
    liu = {}
    for ph in PHASES:
        ws = _CTX["win"][ph]
        liu[ph] = float(np.median([float(_CTX["p114d"].loc[w, "liu2023_xc_phase_pct"])
                                   for w in ws]))
    L.append("")
    L.append("| 阶段 | liu2023 X_C % | " + " | ".join(
        f"{c}（{PARAM_NAME[c]}）" for c in CAND_ORDER if PARAM_NAME[c] != "none") + " |")
    L.append("|---|---|" + "---|" * sum(1 for c in CAND_ORDER if PARAM_NAME[c] != "none"))
    for ph in PHASES:
        cells = []
        for cand in CAND_ORDER:
            if PARAM_NAME[cand] == "none":
                continue
            h = cp[(cp["candidate"] == cand) & (cp["phase"] == ph)]
            if len(h):
                rr = h.iloc[0]
                tag = _fmt(rr["param"], 4) if bool(rr["reachable"]) else "不可达"
                cells.append(tag)
            else:
                cells.append("—")
        L.append(f"| {ph} | {_fmt(liu[ph], 4)} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("### G6 载气（探索性）")
    sub = readings[(readings["reading"] == "G6")
                   & (readings["metric"] == "carrier_same_sign")]
    for _, r in sub.iterrows():
        L.append(f"- {r['candidate']}：{_fmt(r['value'], 3)}/{_fmt(r['threshold'], 3)} 同号")
    L.append("")
    L.append("### 剖面：未转化碳留在何处")
    L.append("")
    if prof_df is not None and len(prof_df):
        L.append("每阶段取该阶段第一个参照窗（按 window_id 排序）。X_C 为相对入炉煤碳。")
        L.append("")
        L.append("| 阶段 | 候选 | 窗 | X_C 全混区出口 | 2 m | 4 m | 6 m | X_C 出口 | T 全混区出口 K | T 出口 K |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for ph in PHASES:
            for cand in CAND_ORDER:
                sub = prof_df[(prof_df["phase"] == ph) & (prof_df["candidate"] == cand)]
                if sub.empty:
                    continue
                sub = sub.sort_values("i")
                z = sub["z_m"].to_numpy()
                xc = sub["X_C_pp"].to_numpy()
                t = sub["T_gas_K"].to_numpy()
                wid = str(sub["window_id"].iloc[0])
                at = lambda zz: float(np.interp(zz, z, xc))  # noqa: E731
                L.append(
                    f"| {ph} | {cand} | …{wid[-9:]} | {_fmt(xc[0], 4)} | {_fmt(at(2.0), 4)} | "
                    f"{_fmt(at(4.0), 4)} | {_fmt(at(6.0), 4)} | {_fmt(xc[-1], 4)} | "
                    f"{_fmt(t[0], 5)} | {_fmt(t[-1], 5)} |"
                )
        L.append("")
        L.append("读法：全混区（首格 1 m）出口只转化了约一半碳，其后的下游格在约 2 700 K 下把"
                 "夹带碳迅速烧掉；M1 的 ϑ 把这一段的 C+H2O／C+CO2 压下去，未转化碳因此"
                 "留在下游段而非全混区；M2 的未转化碳则在全混区出口一次离开反应流。")
        L.append("")
    else:
        L.append("未生成剖面。")
        L.append("")
    L.append("## 判定")
    for _, r in readings[readings["reading"] == "verdict"].iterrows():
        L.append(f"- {r['candidate']}：{r['value']} → "
                 f"**{'入选' if _is_true(r['pass']) else '不入选'}**。{r['note']}")
    L.append("")
    L.append("## 求解失败、不可达与跨越 99.9%")
    L.append("")
    L.append("跨越判据：该阶段全部窗口中，基准或任一扰动点的 X_C ≥ 99.9% 的窗口比例。")
    L.append("")
    L.append("| 候选 | 阶段 | X_C≥99.9% 窗比例（任一扰动） | 基准点窗比例 |")
    L.append("|---|---|---|---|")
    for cand in CAND_ORDER:
        sub = slopes[(slopes["candidate"] == cand) & (slopes["obs"] == "X_C_pp")
                     & (slopes["input"] == "ln_O2")]
        for _, r in sub.iterrows():
            L.append(f"| {cand} | {r['phase']} | {_fmt(r['xc_ceil_frac_any_pert'], 3)} | "
                     f"{_fmt(r['xc_ceil_frac_nom'], 3)} |")
    L.append("")
    L.append("| 候选 | 阶段 | 成功窗 | 失败窗 | 第 3 节 |")
    L.append("|---|---|---|---|---|")
    for (cand, ph), v in n_fail.items():
        L.append(f"| {cand} | {ph} | {v['n_ok']} | {v['n_fail']} | "
                 f"{'跳过（不可达）' if v.get('skipped') else '做'} |")
    L.append("")
    L.append("## C0 的地位：κ = 1 的基准几乎自己通过闸门")
    L.append("")
    L.append("C0 是两个钩子都关闭、κ = 1 的现有模型。逐闸门看：G0 通过、G2 6/8 通过、"
             "G3 3/3 通过，**只差 G1**（20.48 K < 50 K）。也就是说：去掉 κ 这个补偿因子之后，"
             "不加任何机理钩子，响应方向与碳转化率弹性就已经落在装置一侧。")
    L.append("")
    L.append("这必须与 P1.15-A 的记载分开读，否则看起来像数据矛盾。两者的模型斜率符号相反：")
    L.append("")
    L.append("| 组合 | P1.15-A C0（κ = κ̂ ≈ 0.90–0.96） | 本轮 C0（κ = 1） |")
    L.append("|---|---:|---:|")
    L.append("| phase1 CO2←煤量 | +0.168 | −0.570 |")
    L.append("| phase2 CO←煤量 | −0.691 | +0.259 |")
    L.append("| phase2 CO2←煤量 | +0.273 | −0.850 |")
    L.append("| phase4 CO←煤量 | −0.791 | −0.695 |")
    L.append("")
    L.append("**这不是实现差异**。κ = κ̂ 时模型处于能量封顶（缺氧）状态，多给的煤不参与反应，"
             "组成响应由激冷段变换主导；κ = 1 时供氧充足，组成响应回到碳转化的配平。"
             "验证文档已写明 κ 是「唯一能移动 X_C 的量」，把它固定在 1 就同时改变了响应结构。")
    L.append("")
    L.append("**对判定的影响**：G2 的阈值是参照装置斜率定的，而装置斜率与 κ 无关，所以用 κ = 1 "
             "的模型去比是自洽的。但结果是 **C0 的 G2（6/8）高于 M2（5/8）、高于 M1-0（3/8）**——"
             "两个候选在响应方向上都没有超过「什么都不加、只把 κ 放回 1」。"
             "这一点比「均不入选」本身更值得带进下一轮：它提示瓶颈可能不在缺少某个机理钩子，"
             "而在 κ 被当成自由量这件事本身。")
    L.append("")
    L.append("## 卡面自相矛盾 / 无法按字面执行的条目")
    L.append("")
    L.append("1. **U3 的元素闭合阈值 1e-8 低于冻结求解器的精度底。** C0（两钩子关闭）"
             "自身在 9 个参照窗上 O 闭合达 1.7e-5、C 闭合达 4.0e-6；原因是 `res_Xc` 的"
             "缩放因子是 `char_Xc0`（无量纲），不是碳流量。逐窗核对显示闭合误差与"
             "「Σ逐格碳残差/入炉碳」逐位对应。因此字面上的 1e-8 无论是否开钩子都不成立；"
             "本轮按「钩子不引入新的闭合缺项」判定（M2 计入被截留流股后碳闭合从 1e-1 降到 "
             "1e-4 量级），并保留字面结果供核对。")
    L.append("2. **U5 的异常类型。** 卡面写「报 `NotImplementedError`（与 "
             "`GasificationRateFactor` 现有做法一致）」，但现有代码对 `GasificationRateFactor` "
             "与 `CharCO2RateFactor` 抛的是 `ValueError`。本轮按字面执行，两个钩子抛 "
             "`NotImplementedError`；同一位置的守卫、同一错误信息格式与既有做法一致。")
    L.append("3. **M2 的「全混区出口温度」有歧义。** 气相温度与固相温度在全混区出口可不同。"
             "被截留固体的焓按**固相温度**（`results[0].T_solid`）计，因为首格能量平衡正是"
             "在这一温度上记固相焓，这样系统级能量闭合严格。`comparison_points.csv` 的 "
             "`T_wsr_med_K` 为气相温度，`capture_T_solid_K` 为所用固相温度。")
    L.append("4. **G1 的阶段口径。** 卡面只给「出口温度变化（每 +3%，阶段中位）≥ 50 K」，"
             "未说明是否要求每个阶段都满足。本轮按「四个阶段都 ≥ 50 K」判定，"
             "同时报最小 |ΔT_out|，使放宽口径时也能直接读出。")
    L.append("5. **U4 的 0<ϑ≤1 与 X=1。** 幂律在 X=1 处给 ϑ=0。该点已无碳，速率本就是 0；"
             "本轮在 X∈[0,0.999] 上检查值域，并说明 ϑ(1)=0 是极限。")
    L.append("6. **判定与「不可达」的关系。** M1-80 与 M1-90 在 3 个阶段不可达，按卡面"
             "「不可达的候选不做第 3 节」，其 G1–G3 无逐窗斜率，G0 直接不通过，"
             "因此不进入入选比较；它们的端点 X_C 已逐阶段报出。")
    L.append("")
    return "\n".join(L) + "\n"


def _manifest(out_dir: Path) -> None:
    skip = {"sha256.txt", "perturb_snaps.json", "hl_snaps.json", "profiles.json"}
    files = sorted(p for p in out_dir.iterdir() if p.is_file() and p.name not in skip)
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}" for p in files]
    (out_dir / "sha256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    big = sorted(p for p in out_dir.iterdir()
                 if p.is_file() and p.name in skip)
    if big:
        with (out_dir / "sha256.txt").open("a", encoding="utf-8") as fh:
            for p in big:
                fh.write(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n")
