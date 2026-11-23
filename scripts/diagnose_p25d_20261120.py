#!/usr/bin/env python3
"""P2.5-D：装置组成偏还原的成因诊断（离线分析，不重跑求解器）。

依据：论文仓 `papers/jax-gasifier-cascade-hybrid/manuscript/data-side-handoff-p2.5d-20261120.md`
（基线提交 `5f2cdd4`）。

输入（只读）：
- `outputs/diagnostics/round-p2.2a-20261118/baseline_142.csv`（P2.2-A，142 窗）
- `../ai-research/papers/jax-gasifier-cascade-hybrid/revision-evidence-20261118/P2.2-D/reconciled_windows.csv`
  （取 `combination == "S1-C-m"`）
- 可选：`../gasifier-data-prep/outputs/diagnostics/p1.6d-20261031/window_quench_tags.csv`
  （仅用于 R0 可辨识性表的「现有数据」一栏；缺失时退回 P1.6-D 判读文本的区间）

输出：`outputs/diagnostics/round-p2.5d-20261120/`

本脚本不做任何求解器调用，不写任何已合并产物。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "diagnostics" / "round-p2.5d-20261120"

BASELINE = ROOT / "outputs" / "diagnostics" / "round-p2.2a-20261118" / "baseline_142.csv"
P22D_WINDOWS = (ROOT.parent / "ai-research" / "papers" / "jax-gasifier-cascade-hybrid"
                / "revision-evidence-20261118" / "P2.2-D" / "reconciled_windows.csv")
P16D_QUENCH = (ROOT.parent / "gasifier-data-prep" / "outputs" / "diagnostics"
               / "p1.6d-20261031" / "window_quench_tags.csv")
CANONICAL_DIR = ROOT.parent / "gasifier-data-prep" / "outputs" / "canonical"
# canonical 的湿基/干基通道文件名按阶段不同：P1 是单品煤 huayi_sy3_phase1，
# P2–P4 是混煤 huayi_sy3sy2_phaseN。不要用单一 glob 猜。
CANONICAL_WET = {
    "P1": CANONICAL_DIR / "huayi_sy3_phase1_wgs1.parquet",
    "P2": CANONICAL_DIR / "huayi_sy3sy2_phase2_wgs1.parquet",
    "P3": CANONICAL_DIR / "huayi_sy3sy2_phase3_wgs1.parquet",
    "P4": CANONICAL_DIR / "huayi_sy3sy2_phase4_wgs1.parquet",
}
WET_F = "outlet_syngas_wet_washtower_nm3_h"
WET_T = "outlet_syngas_wet_washtower_temperature_c"
WET_P = "outlet_syngas_wet_washtower_pressure_mpa"
DRY_F = "outlet_syngas_dry_nm3_h"

sys.path.insert(0, str(ROOT / "src"))
from model.kinetics import calculate_wgs_equilibrium  # noqa: E402

# --- 预声明常数（卡片 §2.3）-------------------------------------------------
SIGMA_CO = 0.5      # vol%
SIGMA_CO2 = 0.3     # vol%
SIGMA_H2 = 0.5      # vol%
SIGMA_VEC = np.array([SIGMA_CO, SIGMA_CO2, SIGMA_H2])
WEIGHTS = 1.0 / SIGMA_VEC**2
SIGMA_COMP = np.diag(SIGMA_VEC**2)
CHI2_THRESHOLD = 5.99          # dof = 2 的 95% 分位（卡片写死 5.99）
DRY_SCALE = 0.04               # dry ±4% 同步缩放
Y_H2O_SCENARIOS = (0.05, 0.10, 0.15, 0.20)
U2B_TOL = 1e-9
U2_TOL = 1e-9

# U1 已知差异（卡片 §5，阶段 1–4，绝对值百分比）
U1_KNOWN_PCT = {1: 4.68, 2: 1.80, 3: 4.35, 4: 6.37}

PHASE_ORDER = ("phase1", "phase2", "phase3", "phase4")
PHASE_NUM = {"phase1": 1, "phase2": 2, "phase3": 3, "phase4": 4}
PHASE_LABEL = {"phase1": "P1", "phase2": "P2", "phase3": "P3", "phase4": "P4"}


# --- 通用写出工具 -----------------------------------------------------------
def fmt(value, digits: int = 12) -> str:
    """数值格式化：空值/NaN 一律写空串，绝不写字面 NaN。"""
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if not math.isfinite(v):
            return ""
        return f"{v:.{digits}g}"
    return str(value)


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n",
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: fmt(row.get(k)) for k in fields})


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel_path(path: Path) -> str:
    """仓库根相对路径；仓库外文件用 `../` 相对路径，绝不写绝对路径。"""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(os.path.relpath(path.resolve(), ROOT)).as_posix()


def cv(series) -> float:
    arr = np.asarray(series, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    m = arr.mean()
    if m == 0:
        return float("nan")
    return float(arr.std(ddof=1) / m)


# --- 输入 -------------------------------------------------------------------
def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(BASELINE)
    df["phase_num"] = df["phase"].map(PHASE_NUM)
    df = df.sort_values(["phase_num"], kind="stable").reset_index(drop=True)
    if len(df) != 142:
        raise SystemExit(f"baseline 行数异常: {len(df)} != 142")
    if set(df["execution_status"].unique()) != {"completed"}:
        raise SystemExit("baseline 存在非 completed 窗，停止")
    if df.isna().any().any():
        raise SystemExit("baseline 存在缺失值，停止")

    rec = pd.read_csv(P22D_WINDOWS)
    rec = rec[rec["combination"] == "S1-C-m"].copy()
    if len(rec) != 142:
        raise SystemExit(f"P2.2-D S1-C-m 行数异常: {len(rec)}")
    return df, rec


# --- R1：逐窗 ξ 反解 --------------------------------------------------------
def invert_xi(row) -> tuple[float, str]:
    """卡片 §2.1 的口径：ξ = dry·(plant_CO2 − model_CO2)/(100 − plant_CO2)。

    `model_CO2` 取变换前模型组成（`CO2_vol_pct`）；变换后的列不满足该反解。
    """
    denom = 100.0 - float(row["plant_CO2_vol_pct"])
    if not math.isfinite(denom) or denom <= 1.0:
        return float("nan"), "inversion_denominator_below_1pp"
    dry = float(row["dry_total_mol_s"])
    if not math.isfinite(dry) or dry <= 0:
        return float("nan"), "non_positive_dry_flow"
    xi = dry * (float(row["plant_CO2_vol_pct"]) - float(row["CO2_vol_pct"])) / denom
    if not math.isfinite(xi):
        return float("nan"), "non_finite_result"
    return xi, ""


def build_xi_table(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    rows, csv_rows = [], []
    for _, row in df.iterrows():
        xi, reason = invert_xi(row)
        ok = not reason
        dry_val = float(row["dry_total_mol_s"])
        rows.append({"window_id": row["window_id"], "phase": row["phase"], "xi": xi,
                     "xi_over_dry": xi / dry_val, "dry": dry_val,
                     "T_out": float(row["T_out_K"])})
        csv_rows.append({
            "window_id": row["window_id"],
            "phase": row["phase"],
            "xi_mol_s": xi,
            "xi_over_dry": xi / float(row["dry_total_mol_s"]) if ok else float("nan"),
            "dry_total": float(row["dry_total_mol_s"]),
            "T_out_K": float(row["T_out_K"]),
            "T_wsr_K": float(row["T_wsr_K"]),
            "model_CO2": float(row["CO2_vol_pct"]),
            "plant_CO2": float(row["plant_CO2_vol_pct"]),
            "inversion_ok": ok,
            "reason": reason,
        })
    return pd.DataFrame(rows), csv_rows


def build_xi_spread(df: pd.DataFrame, xi_df: pd.DataFrame) -> tuple[list[dict], dict]:
    rows, summary = [], {}
    for phase in PHASE_ORDER:
        sub = xi_df[xi_df["phase"] == phase]
        g = df[df["phase"] == phase]
        xi = sub["xi"].to_numpy(dtype=float)
        xi_ok = xi[np.isfinite(xi)]
        info = {
            "n": int(len(sub)),
            "n_ok": int(xi_ok.size),
            "xi_median": float(np.median(xi_ok)) if xi_ok.size else float("nan"),
            "xi_q1": float(np.percentile(xi_ok, 25)) if xi_ok.size else float("nan"),
            "xi_q3": float(np.percentile(xi_ok, 75)) if xi_ok.size else float("nan"),
            "xi_cv": cv(xi_ok),
            "dry_cv": cv(g["dry_total_mol_s"]),
            "model_co2_cv": cv(g["CO2_vol_pct"]),
            "t_out_cv": cv(g["T_out_K"]),
            "xi_phase_constant": float(g["xi_mol_s"].iloc[0]),
        }
        info["max_input_cv"] = float(np.nanmax(
            [info["dry_cv"], info["model_co2_cv"], info["t_out_cv"]]))
        info["xi_cv_over_max_input_cv"] = (info["xi_cv"] / info["max_input_cv"]
                                           if info["max_input_cv"] else float("nan"))
        summary[phase] = info
        rows.append({"phase": phase, **info})
    return rows, summary


# --- R2：单 ξ 共同位移一致性 -----------------------------------------------
def _shifted_components(dry: float, f: np.ndarray, xi: float) -> np.ndarray:
    total = dry + xi
    return np.array([(f[0] - xi) / total, (f[1] + xi) / total, (f[2] + xi) / total]) * 100.0


def _d_shifted_dxi(dry: float, f: np.ndarray, xi: float) -> np.ndarray:
    total = dry + xi
    return np.array([-100.0 * (dry + f[0]), 100.0 * (dry - f[1]), 100.0 * (dry - f[2])]) / total**2


def fit_single_xi(dry: float, f: np.ndarray, plant: np.ndarray
                  ) -> tuple[float, float, np.ndarray, str]:
    """只拟合 ξ 一个参数；`dry` 固定；Σ = diag(0.5², 0.3², 0.5²)。

    返回 (ξ̂, χ², 标准化残差, 求解路径)。求解路径用于审计：
    `brentq_dchi2` 为预声明路径，`bounded_fallback` 为导数不变号时的显式退路。
    """

    def chi2(xi: float) -> float:
        r = plant - _shifted_components(dry, f, xi)
        return float(np.sum(WEIGHTS * r * r))

    def dchi2(xi: float) -> float:
        r = plant - _shifted_components(dry, f, xi)
        return float(-2.0 * np.sum(WEIGHTS * r * _d_shifted_dxi(dry, f, xi)))

    grid = np.linspace(-0.9 * dry, 3000.0, 1201)
    values = np.array([chi2(x) for x in grid])
    idx = int(np.argmin(values))
    lo = grid[max(idx - 2, 0)]
    hi = grid[min(idx + 2, grid.size - 1)]
    if dchi2(lo) * dchi2(hi) < 0.0:
        xi_hat = brentq(dchi2, lo, hi, xtol=1e-14, rtol=1e-15, maxiter=200)
        path = "brentq_dchi2"
    else:  # 最优落在边界或导数无变号（本数据未出现，留作显式记录）
        xi_hat = float(minimize_scalar(chi2, bounds=(lo, hi), method="bounded",
                                       options={"xatol": 1e-13}).x)
        path = "bounded_fallback"
    resid = plant - _shifted_components(dry, f, xi_hat)
    return float(xi_hat), chi2(xi_hat), resid / SIGMA_VEC, path


def build_triplet(df: pd.DataFrame) -> tuple[list[dict], dict]:
    rank_sigma = int(np.linalg.matrix_rank(SIGMA_COMP))
    eig = np.linalg.eigvalsh(SIGMA_COMP)
    meta = {
        "rank_Sigma": rank_sigma,
        "eigenvalues_Sigma": [float(x) for x in eig],
        "dof": rank_sigma - 1,
        "chi2_threshold": CHI2_THRESHOLD,
        "sigma_vol_pct": {"CO": SIGMA_CO, "CO2": SIGMA_CO2, "H2": SIGMA_H2},
    }
    if rank_sigma != 3:
        # 卡片 §2.3：Σ_comp 不满秩 → 停止并报告，不发明投影
        raise SystemExit(f"Σ_comp 不满秩 (rank={rank_sigma})：按卡片停止，不执行投影")
    if meta["dof"] != 2:
        raise SystemExit(f"dof={meta['dof']} != 2：与卡片预声明不符，停止")

    win_rows, per_phase = [], {}
    for phase in PHASE_ORDER:
        sub = df[df["phase"] == phase]
        stats = []
        for _, row in sub.iterrows():
            dry = float(row["dry_total_mol_s"])
            f = np.array([row["CO_vol_pct"], row["CO2_vol_pct"], row["H2_vol_pct"]],
                         dtype=float) * dry / 100.0
            plant = np.array([row["plant_CO_vol_pct"], row["plant_CO2_vol_pct"],
                              row["plant_H2_vol_pct"]], dtype=float)
            xi_hat, chi2, z, path = fit_single_xi(dry, f, plant)

            # dry ±4%：dry 与各 F_i 同步缩放（组成不变）→ ξ̂ 同比例缩放
            sens = {}
            for tag, scale in (("m4", 1.0 - DRY_SCALE), ("p4", 1.0 + DRY_SCALE)):
                xi_s, chi2_s, _, _ = fit_single_xi(dry * scale, f * scale, plant)
                sens[tag] = (xi_s, chi2_s, xi_s / (dry * scale))

            stats.append({
                "window_id": row["window_id"], "phase": phase, "xi_hat": xi_hat,
                "xi_hat_over_dry": xi_hat / dry, "chi2": chi2,
                "compatible": bool(chi2 <= CHI2_THRESHOLD),
                "z_CO": float(z[0]), "z_CO2": float(z[1]), "z_H2": float(z[2]),
                "solver_path": path,
                "xi_hat_m4": sens["m4"][0], "chi2_m4": sens["m4"][1],
                "ov_m4": sens["m4"][2],
                "xi_hat_p4": sens["p4"][0], "chi2_p4": sens["p4"][1],
                "ov_p4": sens["p4"][2],
            })
            win_rows.append({
                "row_kind": "window", "phase": phase, "window_id": row["window_id"],
                "n": 1, "xi_hat": xi_hat, "xi_hat_over_dry": xi_hat / dry,
                "chi2": chi2, "compatible": bool(chi2 <= CHI2_THRESHOLD),
                "z_CO": float(z[0]), "z_CO2": float(z[1]), "z_H2": float(z[2]),
                "xi_hat_dry_m4pct": sens["m4"][0], "xi_hat_dry_p4pct": sens["p4"][0],
                "xi_over_dry_dry_m4pct": sens["m4"][2],
                "xi_over_dry_dry_p4pct": sens["p4"][2],
                "chi2_dry_m4pct": sens["m4"][1], "chi2_dry_p4pct": sens["p4"][1],
                "rank_Sigma": meta["rank_Sigma"], "dof": meta["dof"],
                "chi2_threshold": CHI2_THRESHOLD,
            })
        sdf = pd.DataFrame(stats)
        info = {
            "n": int(len(sdf)),
            "n_compatible": int(sdf["compatible"].sum()),
            "chi2_median": float(sdf["chi2"].median()),
            "xi_hat_median": float(sdf["xi_hat"].median()),
            "xi_hat_over_dry_median": float(sdf["xi_hat_over_dry"].median()),
            "xi_hat_m4_median": float(sdf["xi_hat_m4"].median()),
            "xi_hat_p4_median": float(sdf["xi_hat_p4"].median()),
            "chi2_m4_median": float(sdf["chi2_m4"].median()),
            "chi2_p4_median": float(sdf["chi2_p4"].median()),
            "z_median_abs_CO": float(sdf["z_CO"].abs().median()),
            "z_median_abs_CO2": float(sdf["z_CO2"].abs().median()),
            "z_median_abs_H2": float(sdf["z_H2"].abs().median()),
            "z_signed_median_CO": float(sdf["z_CO"].median()),
            "z_signed_median_CO2": float(sdf["z_CO2"].median()),
            "z_signed_median_H2": float(sdf["z_H2"].median()),
            "max_ov_dev_m4": float(np.max(np.abs(sdf["ov_m4"] - sdf["xi_hat_over_dry"]))),
            "max_ov_dev_p4": float(np.max(np.abs(sdf["ov_p4"] - sdf["xi_hat_over_dry"]))),
            "max_chi2_dev_m4": float(np.max(np.abs(sdf["chi2_m4"] - sdf["chi2"]))),
            "max_chi2_dev_p4": float(np.max(np.abs(sdf["chi2_p4"] - sdf["chi2"]))),
            "n_brentq": int((sdf["solver_path"] == "brentq_dchi2").sum()),
            "n_bounded_fallback": int((sdf["solver_path"] == "bounded_fallback").sum()),
        }
        per_phase[phase] = info
        win_rows.append({
            "row_kind": "phase_summary", "phase": phase, "window_id": "",
            "n": info["n"], "n_compatible": info["n_compatible"],
            "xi_hat_median": info["xi_hat_median"],
            "xi_hat_over_dry_median": info["xi_hat_over_dry_median"],
            "chi2_median": info["chi2_median"],
            "xi_hat_dry_m4pct_median": info["xi_hat_m4_median"],
            "xi_hat_dry_p4pct_median": info["xi_hat_p4_median"],
            "chi2_dry_m4pct_median": info["chi2_m4_median"],
            "chi2_dry_p4pct_median": info["chi2_p4_median"],
            "z_median_abs_CO": info["z_median_abs_CO"],
            "z_median_abs_CO2": info["z_median_abs_CO2"],
            "z_median_abs_H2": info["z_median_abs_H2"],
            "rank_Sigma": meta["rank_Sigma"], "dof": meta["dof"],
            "chi2_threshold": CHI2_THRESHOLD,
            "reason": "phase_summary_row_not_a_window_record",
        })
    return win_rows, {"meta": meta, "per_phase": per_phase}


# --- R3：温度相关 -----------------------------------------------------------
def build_correlations(df: pd.DataFrame, xi_df: pd.DataFrame,
                       spread: dict) -> list[dict]:
    rows = []
    for scope, phase in ([(PHASE_LABEL[p], p) for p in PHASE_ORDER]
                         + [("ALL", None)]):
        sub = xi_df if phase is None else xi_df[xi_df["phase"] == phase]
        n_total = int(len(sub))
        invalid = bool(phase is not None and spread.get(phase, {}).get("xi_cv", 0.0) > 0.5)
        for key in ("xi", "xi_over_dry"):
            x = sub[key].to_numpy(dtype=float)
            y = sub["T_out"].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() >= 3:
                res = spearmanr(x[mask], y[mask])
                rho, p = float(res.statistic), float(res.pvalue)
            else:
                rho, p = float("nan"), float("nan")
            n = int(mask.sum())
            n_missing = int(n_total - n)
            note = ""
            if not math.isfinite(rho):
                verdict = "not_testable"
            elif rho > 0.3 and p < 0.05:
                verdict = "consistent_with_HA_direction"
            elif rho < -0.3:
                verdict = "opposite_to_HA_direction"
            else:
                verdict = "no_directional_signal"
            if invalid:
                verdict = "invalid_xi_cv_above_50pct"
                note = "该阶段逐窗 ξ 的 CV > 50%，按卡片 §6 相关检验作废"
            rows.append({
                "scope": scope, "phase": phase if phase else "ALL",
                "metric": f"spearman_rho({key},T_out_K)",
                "variable": key, "n": n, "n_missing": n_missing,
                "rho": rho, "p_value": p, "direction_verdict": verdict, "note": note,
            })
    return rows


# --- R4：T_eq 对出口水情景的敏感性 -----------------------------------------
def solve_teq(Q: float) -> tuple[float, str]:
    if not math.isfinite(Q) or Q <= 0:
        return float("nan"), "non_positive_Q"
    lo, hi = 400.0, 6000.0
    k_lo = float(calculate_wgs_equilibrium(lo))
    k_hi = float(calculate_wgs_equilibrium(hi))
    if not (k_hi <= Q <= k_lo):
        return float("nan"), "Q_outside_K_range_400_6000K"
    t = brentq(lambda T: float(calculate_wgs_equilibrium(T)) - Q, lo, hi,
               xtol=1e-8, rtol=1e-14, maxiter=200)
    return float(t), ""


def build_teq(df: pd.DataFrame) -> tuple[list[dict], dict]:
    rows, summary = [], {}
    for phase in PHASE_ORDER:
        sub = df[df["phase"] == phase]
        per_scenario: dict[float, dict] = {}
        for y in Y_H2O_SCENARIOS:
            q_list, t_list = [], []
            q_alt_list, t_alt_list = [], []
            for _, row in sub.iterrows():
                dry = float(row["dry_total_mol_s"])
                xi = float(row["xi_mol_s"])            # 卡片：用 *_xi 列（阶段常数 ξ）
                dry_xi = dry + xi
                n_co2 = float(row["CO2_vol_pct_xi"]) / 100.0 * dry_xi
                n_h2 = float(row["H2_vol_pct_xi"]) / 100.0 * dry_xi
                n_co = float(row["CO_vol_pct_xi"]) / 100.0 * dry_xi
                f_h2o = y / (1.0 - y) * dry             # 卡片 §2.4 字面口径
                f_h2o_alt = y / (1.0 - y) * dry_xi      # 变换后干气口径（备查）
                q = n_co2 * n_h2 / (n_co * f_h2o)
                q_alt = n_co2 * n_h2 / (n_co * f_h2o_alt)
                t, _ = solve_teq(q)
                t_alt, _ = solve_teq(q_alt)
                q_list.append(q)
                t_list.append(t)
                q_alt_list.append(q_alt)
                t_alt_list.append(t_alt)
            q_arr = np.array(q_list)
            t_arr = np.array(t_list)
            t_alt_arr = np.array(t_alt_list)
            per_scenario[y] = {
                "Q_median": float(np.median(q_arr)),
                "T_eq_median": float(np.nanmedian(t_arr)),
                "T_eq_q1": float(np.nanpercentile(t_arr, 25)),
                "T_eq_q3": float(np.nanpercentile(t_arr, 75)),
                "Q_alt_median": float(np.median(q_alt_list)),
                "T_eq_alt_median": float(np.nanmedian(t_alt_arr)),
                "n": int(np.isfinite(t_arr).sum()),
            }
        ref = per_scenario[0.10]["T_eq_median"]
        ref_alt = per_scenario[0.10]["T_eq_alt_median"]
        drifts = {y: per_scenario[y]["T_eq_median"] - ref for y in Y_H2O_SCENARIOS}
        drifts_alt = {y: per_scenario[y]["T_eq_alt_median"] - ref_alt for y in Y_H2O_SCENARIOS}
        max_drift = float(np.nanmax([abs(v) for v in drifts.values()]))
        if max_drift <= 50.0:
            usability = "usable_as_descriptive_proxy"
        elif max_drift > 150.0:
            usability = "no_information"
        else:
            usability = "marginal"
        summary[phase] = {"max_abs_drift_K": max_drift, "usability": usability,
                          "scenarios": per_scenario, "drifts": drifts,
                          "drifts_alt": drifts_alt, "T_eq_ref": ref}
        for y in Y_H2O_SCENARIOS:
            s = per_scenario[y]
            rows.append({
                "phase": phase, "y_H2O": y,
                "F_H2O_out_mol_s": y / (1.0 - y) * float(sub["dry_total_mol_s"].median()),
                "water_basis": "F_H2O=(y/(1-y))*dry_total (card literal)",
                "Q": s["Q_median"],
                "T_eq_K": s["T_eq_median"],
                "T_eq_C": s["T_eq_median"] - 273.15,
                "T_eq_q1_K": s["T_eq_q1"], "T_eq_q3_K": s["T_eq_q3"],
                "drift_vs_10pct_K": drifts[y],
                "Q_alt_post_shift_dry": s["Q_alt_median"],
                "T_eq_alt_K": s["T_eq_alt_median"],
                "drift_alt_vs_10pct_K": drifts_alt[y],
                "n_windows": s["n"],
                "max_abs_drift_K": max_drift,
                "teq_usability": usability,
                "reason": "",
            })
    return rows, summary


# --- R0：可辨识性表 ---------------------------------------------------------
def audit_wet_basis() -> dict:
    """审计洗涤塔出口湿基通道：逐阶段覆盖率，以及「水分率是否只是饱和假设回算」。

    关键：数据仓 config/canonical_flows.yaml 的 wet_to_dry 明写
    dry = wet*(1 - Psat(T)/P)（src/features/wet_to_dry_syngas.py:29-38 落实）。
    因此用 canonical 的 1-dry/wet 复原出的水分率按构造恒等于 Psat/P，
    它只能反映输入假设，不能独立检验「是否饱和」。这里量化该恒等偏差。

    状态传播（第三轮审查 P2）：未核查**不得**被报成通过。
    - 缺文件 → status="missing"，max_dev 为 None（不是 0.0）
    - 文件在但无有效样本 → status="no_samples"，max_dev 为 None
    - 部分阶段缺 → 顶层 status="partial"；全缺 → "not_audited"；全有 → "complete"
    max_dev 只在至少一个阶段真正算出偏差时才非 None。
    """
    per_phase, devs = {}, []
    for phase, path in CANONICAL_WET.items():
        if not path.exists():
            per_phase[phase] = {"status": "missing", "audited": False,
                                "n_minutes": 0, "n_complete": 0, "coverage": None,
                                "max_abs_dev_vs_psat_over_p": None,
                                "note": "canonical 文件缺失，未核查"}
            continue
        df = pd.read_parquet(path, columns=[WET_F, WET_T, WET_P, DRY_F])
        ok = df[WET_F].notna() & df[WET_T].notna() & df[WET_P].notna() & df[DRY_F].notna()
        n, n_ok = int(len(df)), int(ok.sum())
        if n_ok == 0:
            per_phase[phase] = {"status": "no_samples", "audited": False,
                                "n_minutes": n, "n_complete": 0, "coverage": None,
                                "max_abs_dev_vs_psat_over_p": None,
                                "note": "文件在但无有效样本，未核查"}
            continue
        y_flow = 1.0 - df.loc[ok, DRY_F] / df.loc[ok, WET_F]
        psat_arr = np.array([_psat_mpa(t) for t in df.loc[ok, WET_T]])
        # 与数据仓一致：y = (Psat/P).clip(0, 0.99)
        psat_frac = np.clip(psat_arr / df.loc[ok, WET_P].to_numpy(dtype=float), 0.0, 0.99)
        dev = float(np.nanmax(np.abs(y_flow.to_numpy(dtype=float) - psat_frac)))
        devs.append(dev)
        per_phase[phase] = {"status": "audited", "audited": True,
                            "n_minutes": n, "n_complete": n_ok,
                            "coverage": (n_ok / n) if n else None,
                            "max_abs_dev_vs_psat_over_p": dev}

    n_audited = sum(1 for v in per_phase.values() if v.get("audited"))
    if n_audited == 0:
        overall = "not_audited"
    elif n_audited == len(per_phase):
        overall = "complete"
    else:
        overall = "partial"
    max_dev = float(np.max(devs)) if devs else None
    audited_phases = [p for p, v in per_phase.items() if v.get("audited")]
    missing_phases = [p for p, v in per_phase.items() if not v.get("audited")]
    return {"per_phase": per_phase, "max_dev": max_dev, "status": overall,
            "n_audited_phases": n_audited, "audited_phases": audited_phases,
            "missing_phases": missing_phases}


def _psat_mpa(t_c: float) -> float:
    """水饱和蒸气压（MPa）。

    必须与数据仓 src/features/wet_to_dry_syngas.py 完全一致：那里是**查表线性插值**，
    不是 Antoine 式。若这里换一个解析式，回算水分率与 Psat/P 的偏差会变成
    「两套公式之差」，把「恒等式」误读成「独立核验」，正是本轮要避免的错误。
    """
    ts = np.array([100.0, 150.0, 180.0, 200.0, 204.0, 210.0, 220.0, 250.0])
    ps = np.array([0.1014, 0.476, 1.024, 1.554, 1.687, 1.914, 2.320, 3.997])
    t = float(np.clip(float(t_c), ts.min(), ts.max()))
    return float(np.interp(t, ts, ps))


def _fmt_cov(a: dict) -> str:
    out = []
    for p in ("P1", "P2", "P3", "P4"):
        v = a["per_phase"].get(p, {})
        st = v.get("status")
        if st == "missing":
            out.append(f"{p} 缺文件（未核查）")
        elif st == "no_samples":
            out.append(f"{p} 无有效样本（未核查）")
        elif v.get("coverage") is None:
            out.append(f"{p} 未核查")
        else:
            out.append(f"{p} {v['n_complete']}/{v['n_minutes']}（{v['coverage'] * 100:.1f}%）")
    return "；".join(out)


def _cov_claim(a: dict) -> str:
    """覆盖率断言必须由实际 audited 阶段生成，不能写死「近全覆盖」。"""
    st = a.get("status")
    if st == "complete":
        return "四阶段分钟级近全覆盖"
    if st == "partial":
        return (f"仅 {a['n_audited_phases']}/4 阶段已核查"
                f"（{'、'.join(a['audited_phases'])}）；"
                f"{'、'.join(a['missing_phases'])} 未核查")
    return "**未核查**（无可用的 canonical 湿基输入）"


def _dev_claim(a: dict) -> str:
    """恒等偏差断言同理：无核查时不得写数值，更不能写 0。"""
    if a.get("max_dev") is None:
        return "未核查（无可用样本），偏差未知"
    return f"{a['max_dev']:.2e}"


def build_identifiability(rec: pd.DataFrame, wet_audit: dict | None = None) -> tuple[list[dict], dict]:
    nw = rec.groupby("phase")["nw_after"].median()
    nw_txt = " / ".join(f"{nw[p]:.2f}" for p in ("P1", "P2", "P3", "P4"))

    wet_txt = "0.4525-0.4646 (9 窗)"
    wet_src = ("P1.6-D window_quench_tags.csv: FR_1015/FR1_1015, "
               "y_sat/y_flow 1.027-1.029")
    wet_sha = None
    if P16D_QUENCH.exists():
        q = pd.read_csv(P16D_QUENCH)
        if {"y_h2o_from_flows"}.issubset(q.columns) and len(q):
            wet_txt = (f"{q['y_h2o_from_flows'].min():.4f}-{q['y_h2o_from_flows'].max():.4f}"
                       f" ({len(q)} 窗)")
            wet_sha = sha256_of(P16D_QUENCH)

    wet_audit = wet_audit if wet_audit is not None else audit_wet_basis()
    cov_txt = _fmt_cov(wet_audit)
    dev_txt = _dev_claim(wet_audit)
    claim_txt = _cov_claim(wet_audit)

    rows = [
        {
            "quantity": "plant_inlet_water_total_mol_s",
            "required_for": "H-A_vs_H-B",
            "available_in_project": "partial",
            "source_or_gap": ("coal moisture fixed at 2% of dry coal (P2.2-D constants); "
                              "burner steam = not_applicable (user-confirmed 2026-10-31); "
                              "no quench-water flow tag in the Huayi mapping "
                              "(the 激冷水入水流量 entry belongs to the Yulin tag file)"),
            "note": ("煤水分是假设常数不是逐窗测量；Huayi 侧无激冷水流量位号；"
                     "三者相加不等于反应器入口水"),
        },
        {
            "quantity": "washtower_wet_flow_channel",
            "required_for": "H-A_vs_H-B",
            "available_in_project": "yes" if wet_audit["status"] == "complete" else "not_audited",
            "source_or_gap": (f"直接测量：{WET_F} + {WET_T} + {WET_P}"
                              f"（canonical parquet）。覆盖：{cov_txt}"),
            "note": (f"{claim_txt}。这是**湿流量通道**，是真测量。"
                     "审查前本卡把「有该通道」误当成「有独立水分率测量」，此处更正："
                     "通道可用，但由它得不到独立的 H2O 分率（见下一行）"),
        },
        {
            "quantity": "plant_outlet_h2o_fraction_wet_basis",
            "required_for": "H-A_vs_H-B",
            "available_in_project": "derived",
            "source_or_gap": (f"**推算值，非测量**：数据仓 config/canonical_flows.yaml 的 "
                              f"wet_to_dry 规定 dry = wet*(1 − Psat(T)/P)"
                              f"（src/features/wet_to_dry_syngas.py:29-38 落实）。"
                              f"故 1 − dry/wet ≡ Psat(T)/P；实测恒等偏差 "
                              f"|1−dry/wet − Psat/P|max = {dev_txt}"),
            "note": ("因此该水分率**只是输入的饱和假设本身**，不能用来检验「是否饱和」，"
                     "也不能充当独立的湿基 H2O 测量。中位约 0.46 这一点成立的唯一含义是"
                     "「canonical 按 ~0.46 的饱和假设生成」。此前「98% 饱和」的说法"
                     "是拿 Psat 与 Psat 相比，为循环论证，已撤回"),
        },
        {
            "quantity": "p16d_flow_difference_h2o_estimate",
            "required_for": "H-A_vs_H-B",
            "available_in_project": "conditional",
            "source_or_gap": f"{wet_src}；y_H2O = {wet_txt}",
            "note": ("来自另一口径的原始 FR1 湿/干流量差，**不是** canonical 的饱和假设回算；"
                     "但 P1.6-D 原判读明确保留「同一标准状态与流量计标定尚未独立审计」的前提，"
                     "故只作条件性估计，不能当已核实的独立测量，也不能由 9 窗推出全时段"),
        },
        {
            "quantity": "plant_reactor_outlet_h2o_fraction",
            "required_for": "H-A_vs_H-B",
            "available_in_project": "no",
            "source_or_gap": ("online analyzer is dry-basis (contract composition_basis: "
                              "outlet_*_dry_vol_pct); C/H/O 协调的碳平衡无水项"),
            "note": ("这是 H-A/H-B 不可分的直接原因。准确表述是"
                     "「**湿流量通道有，缺的是所需位置**」——反应器出口/激冷入口；"
                     "**不能**写成「已有独立 H2O 测量」：洗涤塔出口那份水分率是饱和假设推算的"),
        },
        {
            "quantity": "plant_wgs_quotient_Q",
            "required_for": "H-A_vs_H-B",
            "available_in_project": "no",
            "source_or_gap": "Q=CO2*H2/(CO*H2O) 的分母 H2O 不可辨识",
            "note": ("干基三组分可算比值但缺 H2O；用洗涤塔出口水会得到与 205C 冻结状态"
                     "不符的表观温度"),
        },
        {
            "quantity": "net_reacted_water_nw_after",
            "required_for": "H-B",
            "available_in_project": "partial",
            "source_or_gap": f"P2.2-D S1-C-m reconciled_windows.csv; 阶段中位 {nw_txt}",
            "note": ("H 平衡反推的净耗水，P4 为负；不是输入水残差，也不能当出口残余水"
                     "（原 R2/R4 已按卡片撤回）"),
        },
        {
            "quantity": "quench_and_waste_heat_temperature_residence",
            "required_for": "H-A",
            "available_in_project": "partial",
            "source_or_gap": ("炉膛温度 3 层×2（TI_1025/1026/1027，水冷壁渣层偏置、仅趋势）、"
                              "洗涤塔出口温度与压力（全时段）；无激冷段入口温度、无停留时间"),
            "note": "只有点位温度，没有停留时间与沿程温度剖面；下游若确有变换需要能定位温度窗口的剖面",
        },
        {
            "quantity": "plant_dry_gas_flow",
            "required_for": "R2",
            "available_in_project": "yes",
            "source_or_gap": ("canonical 表全时段（minutes）；P1.6-D 提取了 9 个代表窗，"
                              "FR1_1015 与 canonical 干基流量有两个口径"),
            "note": "R2 按卡片固定模型 dry、不把装置 dry 引入拟合；装置 dry 未参与本卡任何判据",
        },
        {
            "quantity": "reactor_outlet_temperature_measured",
            "required_for": "R3",
            "available_in_project": "no",
            "source_or_gap": ("Huayi 当前映射无合成气出口温度位号；只有炉膛温度"
                              "（渣层偏置、仅趋势）与洗涤塔出口温度；原始表未映射位号未核实"),
            "note": "R3 按卡片用模型 T_out；本卡没有可与 T_out 对打的独立出口温度",
        },
        {
            "quantity": "literature_wet_gas_composition",
            "required_for": "cross_check",
            "available_in_project": "no",
            "source_or_gap": "project 现有刘文对照表（contract_vs_liu2023_by_phase.csv）只有干基组成",
            "note": "未见湿基数据，无法交叉核对 H2O",
        },
        {
            "quantity": "analyzer_basis_dry",
            "required_for": "R0_root_cause",
            "available_in_project": "yes",
            "source_or_gap": ("contract composition_basis: measurement_basis=online_analyzer, "
                              "applies_to outlet_*_dry_vol_pct"),
            "note": "三个观测量 CO/CO2/H2 全是干基，水项按构造缺席",
        },
    ]
    meta = {"y_h2o_wet_reference": wet_txt, "wet_reference_sha256": wet_sha,
            "wet_flow_channel_coverage": cov_txt,
            "wet_fraction_derivation_identity_max_dev": wet_audit["max_dev"],
            "wet_basis_position": "wash-tower outlet (~204.5 C / 3.65 MPa)",
            "nw_after_medians": {p: float(nw[p]) for p in ("P1", "P2", "P3", "P4")}}
    return rows, meta


# --- U1/U2/U2b --------------------------------------------------------------
def build_unit_checks(df: pd.DataFrame, xi_df: pd.DataFrame,
                      spread: dict, triplet: dict) -> tuple[list[dict], dict]:
    rows, meta = [], {}

    # U1：逐窗中位 ξ 与 P2.2-A 阶段常数的差（诊断项，不是停止条件）
    u1_max_dev = 0.0
    for phase in PHASE_ORDER:
        sub = xi_df[xi_df["phase"] == phase]
        const = float(df[df["phase"] == phase]["xi_mol_s"].iloc[0])
        med = float(np.nanmedian(sub["xi"]))
        rel = (med - const) / const * 100.0
        expected = U1_KNOWN_PCT[PHASE_NUM[phase]]
        dev = abs(abs(rel) - expected)
        u1_max_dev = max(u1_max_dev, dev)
        rows.append({
            "check": "U1", "phase": phase, "n": int(len(sub)),
            "metric": "window_xi_median_vs_phase_constant_pct",
            "value": rel, "reference": -expected,
            "abs_deviation_from_card_pct_points": dev,
            "tolerance": 0.05, "passed": bool(dev <= 0.05),
            "status": "diagnostic_only_not_stop_condition",
            "note": ("来源：P2.2-A 在参照窗中位上拟合阶段常数，"
                     "逐窗中位不必等于中位的逐窗反解"),
        })
    meta["u1_max_dev_pct_points"] = u1_max_dev

    # U2：3 窗独立重算（从变换定义出发：由 (F_CO2+ξ)/(dry+ξ)*100 = plant_CO2 解 ξ）
    picks = [0, len(df) // 2, len(df) - 1]
    u2_max = 0.0
    for i in picks:
        row = df.iloc[i]
        dry = float(row["dry_total_mol_s"])
        f_co2 = float(row["CO2_vol_pct"]) * dry / 100.0
        p = float(row["plant_CO2_vol_pct"])
        xi_indep = (p * dry - 100.0 * f_co2) / (100.0 - p)
        xi_script = float(xi_df.iloc[i]["xi"])
        diff = abs(xi_indep - xi_script)
        u2_max = max(u2_max, diff)
        rows.append({
            "check": "U2", "phase": row["phase"], "n": 1,
            "metric": f"independent_recompute_abs_diff[{row['window_id']}]",
            "value": diff, "reference": U2_TOL, "tolerance": U2_TOL,
            "passed": bool(diff <= U2_TOL), "status": "pass_if_below_tolerance",
            "note": "独立推导：F_CO2 由组成×dry 反算，再由 (F_CO2+ξ)/(dry+ξ)*100=plant_CO2 解 ξ",
        })
    meta["u2_max_abs_diff"] = u2_max

    # U2b：dry 与 F_i 同步缩放 (1±4%)，ξ̂/dry 与 χ² 必须不变
    u2b_max_ov = max(p["max_ov_dev_m4"] for p in triplet["per_phase"].values())
    u2b_max_ov = max(u2b_max_ov, max(p["max_ov_dev_p4"] for p in triplet["per_phase"].values()))
    u2b_max_chi = max(p["max_chi2_dev_m4"] for p in triplet["per_phase"].values())
    u2b_max_chi = max(u2b_max_chi, max(p["max_chi2_dev_p4"] for p in triplet["per_phase"].values()))
    for name, val in (("xi_hat_over_dry", u2b_max_ov), ("chi2", u2b_max_chi)):
        rows.append({
            "check": "U2b", "phase": "ALL", "n": 284,
            "metric": f"max_abs_dev_{name}_under_dry_scaling_pm4pct",
            "value": val, "reference": U2B_TOL, "tolerance": U2B_TOL,
            "passed": bool(val <= U2B_TOL), "status": "pass_if_below_tolerance",
            "note": "同步缩放 dry 与各 F_i（组成不变）；只报 ξ̂ 绝对量的按比例变化，不做跨量纲比较",
        })
    meta["u2b_max_dev_ov"] = u2b_max_ov
    meta["u2b_max_dev_chi2"] = u2b_max_chi

    # 反解/可解率与 CV 检查（§6 停止条件）
    for phase in PHASE_ORDER:
        sub = xi_df[xi_df["phase"] == phase]
        n_ok = int(np.isfinite(sub["xi"]).sum())
        rate = n_ok / len(sub)
        rows.append({
            "check": "R1_solvable_rate", "phase": phase, "n": int(len(sub)),
            "metric": "finite_inversion_fraction", "value": rate,
            "reference": 0.5, "tolerance": 0.5, "passed": bool(rate >= 0.5),
            "status": "stop_phase_correlation_if_below_0.5",
            "note": "",
        })
        rows.append({
            "check": "R1_xi_cv", "phase": phase, "n": int(len(sub)),
            "metric": "xi_cv", "value": spread[phase]["xi_cv"],
            "reference": 0.5, "tolerance": 0.5,
            "passed": bool(spread[phase]["xi_cv"] <= 0.5),
            "status": "invalidate_phase_correlation_if_above_0.5",
            "note": "",
        })
    meta["solvable_rate"] = {p: float(np.isfinite(xi_df[xi_df['phase'] == p]['xi']).mean())
                             for p in PHASE_ORDER}
    return rows, meta


# --- readings ---------------------------------------------------------------
def build_readings(spread, triplet, corr_rows, teq, r0_rows, unit_meta,
                   wet_audit=None) -> list[dict]:
    rows = []
    wet_audit = wet_audit or {}
    dev_txt = _dev_claim(wet_audit)

    n_yes = sum(1 for r in r0_rows if r["available_in_project"] == "yes")
    n_partial = sum(1 for r in r0_rows if r["available_in_project"] == "partial")
    n_no = sum(1 for r in r0_rows if r["available_in_project"] == "no")
    n_der = sum(1 for r in r0_rows if r["available_in_project"] == "derived")
    n_cond = sum(1 for r in r0_rows if r["available_in_project"] == "conditional")
    rows.append({"reading": "R0", "metric": "quantities_listed", "phase": "ALL",
                 "value": len(r0_rows), "note": "R0 可辨识性表行数"})
    rows.append({"reading": "R0", "metric": "available_yes_partial_derived_cond_no",
                 "phase": "ALL", "value": f"{n_yes}/{n_partial}/{n_der}/{n_cond}/{n_no}",
                 "note": ("yes / partial / derived / conditional / no。判别 H-A/H-B 的两个量里，"
                          "入炉水只有 2% 假设（partial），反应器出口 H₂O 没有（no）。"
                          f"湿流量通道审计状态 = {wet_audit['status']}，但由其得到的水分率是 derived"
                          f"（恒等于 Psat/P 的饱和假设回算，偏差 {dev_txt}），"
                          "P1.6-D 的流量差估计是 conditional")})
    rows.append({
        "reading": "R0", "metric": "decidable_HA_vs_HB", "phase": "ALL", "value": "no",
        "note": ("反应器出口水与装置 Q 都不可辨识。洗涤塔出口有湿流量通道，"
                 "但水分率是饱和假设回算、不是独立测量，故不能替代反应器出口水"),
    })

    for phase in PHASE_ORDER:
        s = spread[phase]
        p = PHASE_LABEL[phase]
        rows += [
            {"reading": "R1", "metric": "xi_median_mol_s", "phase": p,
             "value": s["xi_median"], "note": ""},
            {"reading": "R1", "metric": "xi_q1_mol_s", "phase": p, "value": s["xi_q1"], "note": ""},
            {"reading": "R1", "metric": "xi_q3_mol_s", "phase": p, "value": s["xi_q3"], "note": ""},
            {"reading": "R1", "metric": "xi_cv", "phase": p, "value": s["xi_cv"], "note": ""},
            {"reading": "R1", "metric": "dry_cv", "phase": p, "value": s["dry_cv"], "note": ""},
            {"reading": "R1", "metric": "model_co2_cv", "phase": p,
             "value": s["model_co2_cv"], "note": ""},
            {"reading": "R1", "metric": "t_out_cv", "phase": p, "value": s["t_out_cv"], "note": ""},
            {"reading": "R1", "metric": "xi_cv_over_max_input_cv", "phase": p,
             "value": s["xi_cv_over_max_input_cv"],
             "note": ("<1 才谈得上「ξ 比输入更稳」；仅作描述，不是判别依据")},
        ]

    for phase in PHASE_ORDER:
        t = triplet["per_phase"][phase]
        p = PHASE_LABEL[phase]
        rows += [
            {"reading": "R2", "metric": "n_compatible_of_n", "phase": p,
             "value": f"{t['n_compatible']}/{t['n']}", "note": "χ² ≤ 5.99"},
            {"reading": "R2", "metric": "chi2_median", "phase": p,
             "value": t["chi2_median"], "note": "dof=2, 阈值 5.99"},
            {"reading": "R2", "metric": "xi_hat_median_mol_s", "phase": p,
             "value": t["xi_hat_median"], "note": ""},
            {"reading": "R2", "metric": "xi_hat_over_dry_median", "phase": p,
             "value": t["xi_hat_over_dry_median"], "note": ""},
            {"reading": "R2", "metric": "z_median_abs_CO_CO2_H2", "phase": p,
             "value": f"{t['z_median_abs_CO']:.2f}/{t['z_median_abs_CO2']:.2f}/"
                      f"{t['z_median_abs_H2']:.2f}",
             "note": "标准化残差绝对值中位；看是哪个量在拖"},
            {"reading": "R2", "metric": "xi_hat_at_dry_m4_p4pct_median", "phase": p,
             "value": f"{t['xi_hat_m4_median']:.6g}/{t['xi_hat_p4_median']:.6g}",
             "note": "同步缩放：绝对量按比例变化，xi_hat/dry 与 chi2 不变（见 U2b）"},
        ]
    rows.append({"reading": "R2", "metric": "rank_Sigma_comp", "phase": "ALL",
                 "value": triplet["meta"]["rank_Sigma"],
                 "note": f"dof={triplet['meta']['dof']}, 阈值 {CHI2_THRESHOLD}"})

    for r in corr_rows:
        if r["variable"] not in ("xi", "xi_over_dry"):
            continue
        rows.append({
            "reading": "R3", "metric": f"rho_{r['variable']}_vs_T_out", "phase": r["scope"],
            "value": r["rho"],
            "note": f"p={r['p_value']:.3g}; n={r['n']}; missing={r['n_missing']}; "
                    f"{r['direction_verdict']}",
        })

    for phase in PHASE_ORDER:
        t = teq[phase]
        p = PHASE_LABEL[phase]
        rows.append({
            "reading": "R4", "metric": "max_abs_T_eq_drift_K", "phase": p,
            "value": t["max_abs_drift_K"],
            "note": (f"y_H2O 5/10/15/20%；漂移 " +
                     " / ".join(f"{t['drifts'][y]:+.1f}" for y in Y_H2O_SCENARIOS) +
                     f"；{t['usability']}；只决定 T_eq 可否引用，不支撑 H-A"),
        })

    rows.append({"reading": "U1", "metric": "max_abs_dev_from_card_pct_points", "phase": "ALL",
                 "value": unit_meta["u1_max_dev_pct_points"],
                 "note": "诊断项，不触发停止；来源=参照窗中位拟合 vs 逐窗中位反解"})
    rows.append({"reading": "U2", "metric": "max_abs_diff_3_windows", "phase": "ALL",
                 "value": unit_meta["u2_max_abs_diff"], "note": "容差 1e-9"})
    rows.append({"reading": "U2b", "metric": "max_abs_dev_xi_over_dry_and_chi2", "phase": "ALL",
                 "value": f"{unit_meta['u2b_max_dev_ov']:.3g}/"
                          f"{unit_meta['u2b_max_dev_chi2']:.3g}",
                 "note": "dry 与 F_i 同步 ±4%；容差 1e-9"})

    # 判定
    comp_txt = "；".join(
        f"{PHASE_LABEL[p]} {triplet['per_phase'][p]['n_compatible']}/"
        f"{triplet['per_phase'][p]['n']}" for p in PHASE_ORDER)
    rows.append({
        "reading": "VERDICT", "metric": "outcome_of_three", "phase": "ALL", "value": 3,
        "note": ("分不开：R0 两个关键量不可辨识；R3 在 P1/P3 与 H-A 方向相反，P2 无信号。"
                 "第 2 种（与 H-A 相容）未获支持；R3 受模型内 CO2 与 T_out 机械耦合污染，"
                 "不能作为 H-A 的反证"),
    })
    rows.append({
        "reading": "VERDICT", "metric": "R2_compatible_windows", "phase": "ALL",
        "value": comp_txt, "note": ("P2 全部相容；P4 全部超阈值（中位 χ² > 5.99）；"
                                    "P1/P3 约一半超阈值"),
    })
    rows.append({
        "reading": "VERDICT", "metric": "outcome_1_scope", "phase": "P4",
        "value": "holds_at_phase_level",
        "note": "P4 阶段中位 χ² 超阈值、全部窗超阈值：单一 WGS 移动解释不了，缺口不止一个成分",
    })
    rows.append({
        "reading": "VERDICT", "metric": "outcome_2", "phase": "ALL", "value": "not_supported",
        "note": ("R3 方向在 P1/P3 与 H-A 相反（ρ<-0.3 且 p<0.05），按卡片预声明判据"
                 "字面不满足「方向一致」。但 ξ 按定义是模型与装置 CO2 的残差，"
                 "ρ(ξ,模型 CO2)=-0.70/-0.33/-0.80/-0.68 而 ρ(ξ,装置 CO2)=0.21/0.08/0.25/0.89；"
                 "P1–P3 内模型 CO2 与 T_out 强同向（0.95/0.97/0.95），机械地把 ξ 压成负。"
                 "故第 2 种记为**未获支持**，不能记为被排除：该检验近似在测模型内部协方差"),
    })
    rows.append({
        "reading": "VERDICT", "metric": "outcome_2_predeclared_literal", "phase": "ALL",
        "value": "excluded",
        "note": ("按卡片原始字面判据的历史结果（已由 outcome_2 取代）："
                 "「R3 方向在 P1/P3 与 H-A 相反，不满足方向一致 → 排除」。"
                 "保留仅作预声明记录，不得作为当前结论引用"),
    })
    rows.append({
        "reading": "VERDICT", "metric": "aggregation_rule_predeclared", "phase": "ALL",
        "value": "no",
        "note": ("卡片只给逐窗阈值与阶段汇总列，未规定「阶段相容」如何由窗计数聚合；"
                 "本卡按「中位 χ² 与超阈值窗数并报」处理，不自行定义通过/不通过"),
    })
    return rows


# --- protocol / interpretation ---------------------------------------------
def build_protocol(df, rec, spread, triplet, teq, r0_meta, unit_meta,
                   corr_rows, diag, wet_audit=None) -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         text=True).strip()
    except Exception:  # pragma: no cover
        commit = "unknown"
    inputs = []
    for p in (BASELINE, P22D_WINDOWS):
        inputs.append({"path": rel_path(p), "sha256": sha256_of(p)})
    if P16D_QUENCH.exists():
        inputs.append({"path": rel_path(P16D_QUENCH), "sha256": sha256_of(P16D_QUENCH),
                       "role": "R0 only (read-only reference, outside this repo)"})
    # R0 的覆盖率与恒等偏差直接由这四份 parquet 算出，必须进输入清单，
    # 否则 canonical 更新后无法从交付物判断本次核查的是哪个版本。
    canonical_inputs = []
    for phase, path in CANONICAL_WET.items():
        entry = {"path": rel_path(path), "phase": phase}
        if path.exists():
            entry["sha256"] = sha256_of(path)
            entry["role"] = "R0 wet-basis audit (coverage + saturation identity)"
            st = (wet_audit or {}).get("per_phase", {}).get(phase, {})
            entry.update({
                "n_minutes": st.get("n_minutes"),
                "n_complete": st.get("n_complete"),
                "coverage": st.get("coverage"),
                "max_abs_dev_vs_psat_over_p": st.get("max_abs_dev_vs_psat_over_p"),
            })
        else:
            # 缺文件时明确标为「未核查」，不能用固定文字宣称四阶段全覆盖
            entry["sha256"] = None
            entry["role"] = "R0 wet-basis audit -- MISSING, NOT AUDITED"
            entry["audited"] = False
        canonical_inputs.append(entry)
        inputs.append(entry)
    psat_src = (CANONICAL_DIR.parents[1] / "src" / "features" / "wet_to_dry_syngas.py")
    if psat_src.exists():
        inputs.append({"path": rel_path(psat_src), "sha256": sha256_of(psat_src),
                       "role": ("R0 only: the Psat lookup table replicated in _psat_mpa must match "
                                "this file; a different formula would turn the tautology into a "
                                "false independent check")})
    return {
        "round": "P2.5-D / 2026-11-20",
        "model_repo": "gasifier-1d-kinetic",
        "source_commit": commit,
        "baseline_commit_expected": "5f2cdd4",
        "task_card": ("papers/jax-gasifier-cascade-hybrid/manuscript/"
                      "data-side-handoff-p2.5d-20261120.md"),
        "solver_calls": 0,
        "solver_note": "本卡不重跑求解器；全部量由两个 CSV 离线导出",
        "inputs": inputs,
        "wet_basis_audit": {
            "note": ("R0 wet FLOW channel is a real measurement; the water FRACTION is derived, "
                     "because dry = wet*(1-Psat(T)/P) by construction, so 1-dry/wet == Psat/P."),
            "status": (wet_audit or {}).get("status"),
            "n_audited_phases": (wet_audit or {}).get("n_audited_phases"),
            "missing_phases": (wet_audit or {}).get("missing_phases"),
            "max_abs_dev_identity": (wet_audit or {}).get("max_dev"),
            "per_phase": (wet_audit or {}).get("per_phase", {}),
            "psat_table_source": (rel_path(CANONICAL_DIR.parents[1] / "src" / "features"
                                           / "wet_to_dry_syngas.py")),
        },
        "windows": {"n": int(len(df)),
                    "by_phase": {p: int((df['phase'] == p).sum()) for p in PHASE_ORDER},
                    "execution_status": "completed (all)"},
        "definitions": {
            "xi_window": ("R1 逐窗反解：xi = dry*(plant_CO2 - model_CO2)/(100 - plant_CO2)，"
                          "model_CO2 = baseline 的变换前 CO2_vol_pct（不是 *_xi 列）"),
            "xi_phase_constant": "P2.2-A baseline 的 xi_mol_s（每阶段一个唯一值）",
            "F_i": "F_i = 组成 vol% × dry_total / 100 (mol/s)",
            "shifted_model": ("CO=(F_CO-d)/(dry+d)*100, CO2=(F_CO2+d)/(dry+d)*100, "
                              "H2=(F_H2+d)/(dry+d)*100，d 即 ξ"),
            "R2_estimator": ("只拟合 ξ 一个参数；模型 dry 固定；Σ_comp=diag(0.5²,0.3²,0.5²)；"
                             "χ²=yᵀΣ⁻¹y；dof=rank(Σ)-1=2；阈值 5.99"),
            "R2_solver": ("dχ²/dξ 在扫描区间内变号后用 brentq 求根（xtol=1e-14），"
                          "否则退回 bounded 最小化并显式记录"),
            "compatible": "chi2 <= 5.99（逐窗）",
            "R3": "Spearman ρ(ξ, T_out_K) 与 ρ(ξ/dry, T_out_K)，双侧 p",
            "R3_verdict": ("ρ>0.3 且 p<0.05→consistent_with_HA_direction；ρ<-0.3→"
                           "opposite_to_HA_direction；其余 no_directional_signal"),
            "R4_Q": "Q = n_CO2*n_H2/(n_CO*n_H2O)，n_i 由 *_xi 列 ×(dry+ξ) 换算（阶段常数 ξ）",
            "R4_H2O": ("主口径 F_H2O=(y/(1-y))*dry_total（卡片 §2.4 字面）；"
                       "备查口径 F_H2O=(y/(1-y))*(dry+ξ)，两口径都写入 CSV"),
            "R4_T_eq": "解 K_WGS(T)=Q，K 用 model.kinetics.calculate_wgs_equilibrium（模型自身 Gibbs）",
            "R4_drift": "drift(y) = median(T_eq|y) − median(T_eq|y=10%)，单位 K",
            "R4_usability": "max|drift| ≤50 K usable；>150 K no_information；其间 marginal",
            "CV": "样本标准差(ddof=1)/均值",
            "U1": "诊断项，不是停止条件",
            "U2": ("3 窗独立重算：由 (F_CO2+ξ)/(dry+ξ)*100 = plant_CO2 解 ξ，"
                   "与卡片公式互为独立推导"),
            "U2b": "dry 与各 F_i 同步乘 (1±4%)，ξ̂/dry 与 χ² 应不变（≤1e-9）",
            "blank_cells": ("结构性空缺：triplet_span.csv 的 phase_summary 行逐窗字段、"
                            "unit_checks.csv 中非 U1 行的 U1 偏差列。这些不是缺失值；"
                            "真缺失一律留空并在 reason 列写明原因，CSV 中不出现字面 NaN"),
        },
        "R2_checks": {
            "rank_Sigma": triplet["meta"]["rank_Sigma"],
            "eigenvalues_Sigma": triplet["meta"]["eigenvalues_Sigma"],
            "dof": triplet["meta"]["dof"],
            "chi2_threshold": CHI2_THRESHOLD,
            "full_rank_required": True,
            "note": "若 rank<3 应停止报告；本次满秩，未启用任何投影分支",
        },
        "R2_by_phase": triplet["per_phase"],
        "R1_spread": spread,
        "R3_summary": {f"{r['scope']}|{r['variable']}":
                       {"rho": r["rho"], "p": r["p_value"], "n": r["n"],
                        "n_missing": r["n_missing"],
                        "verdict": r["direction_verdict"]}
                       for r in corr_rows},
        "R4_by_phase": {p: {"max_abs_drift_K": teq[p]["max_abs_drift_K"],
                            "usability": teq[p]["usability"],
                            "drifts_K": {str(y): teq[p]["drifts"][y] for y in Y_H2O_SCENARIOS},
                            "drifts_alt_basis_K": {str(y): teq[p]["drifts_alt"][y]
                                                   for y in Y_H2O_SCENARIOS},
                            "T_eq_reference_K": teq[p]["T_eq_ref"]}
                        for p in PHASE_ORDER},
        "R0": r0_meta,
        "diagnostic_relations": diag,
        "unit_checks": unit_meta,
        "ambiguities": [
            ("R4 的水基准：卡片写 F_H2O=(y/(1-y))*dry_total，但变换后干气应为 dry+ξ。"
             "主口径按字面用 dry_total，备查口径同时写入 CSV（两口径 T_eq 差约 10-25 K，"
             "不改变 ≤50/>150 K 的分类）"),
            ("R4 的 ξ 口径：卡片说「用 *_xi 列的对应流量」，即阶段常数 ξ；"
             "R1 的逐窗 ξ 是另一个量，本卡未在 R4 中使用"),
            ("R2 的阶段聚合规则未预声明：卡片只给逐窗阈值与阶段汇总列；"
             "本卡同时报中位 χ² 与超阈值窗数，不自行定义阶段通过/不通过"),
            ("R3 的实现细节未规定：本卡用双侧 p、逐窗 ξ（R1）与模型 T_out；"
             "未使用实测出口温度（见 R0 行 reactor_outlet_temperature_measured）"),
            ("R1 的 inversion_ok 未定义：本卡定义为「分母 >1 pp 且 dry>0 且结果有限」，"
             "不把负 ξ 判为失败（负值是结果，不是数值失败）"),
            ("洗涤塔出口湿基测量（P1.6-D，9 窗）的分母基准是洗涤塔出口干气，"
             "与 baseline 的模型 dry 不是同一位置；本卡未把它并入任何判据"),
        ],
        "known_limitations": [
            "不重跑求解器，ξ 的因果归属不在本卡范围内",
            "不改 WGS 动力学：这是范围选择，不是「已证明无关」",
            "R4 任何结果都不构成对 H-A 的支持",
        ],
    }


def diagnostic_relations(df: pd.DataFrame, xi_df: pd.DataFrame) -> dict:
    """辅助诊断（不是卡片预声明的判据）：ξ 的窗口间变化跟随谁。"""
    out = {}
    merged = df[["window_id", "phase", "CO2_vol_pct", "plant_CO2_vol_pct",
                 "T_out_K"]].merge(
        xi_df[["window_id", "xi"]], on="window_id", how="left")
    for phase in PHASE_ORDER:
        g = merged[merged["phase"] == phase]
        out[phase] = {
            "n": int(len(g)),
            "rho_xi_modelCO2": float(spearmanr(g["xi"], g["CO2_vol_pct"]).statistic),
            "rho_xi_plantCO2": float(spearmanr(g["xi"], g["plant_CO2_vol_pct"]).statistic),
            "rho_modelCO2_Tout": float(spearmanr(g["CO2_vol_pct"], g["T_out_K"]).statistic),
            "label": "diagnostic_only_not_a_predeclared_reading",
        }
    return out


def build_interpretation(df, xi_df, diag, spread, triplet, corr_rows, teq,
                         r0_rows, unit_meta, readings, wet_audit=None) -> str:
    wet_audit = wet_audit or {}
    claim_txt = _cov_claim(wet_audit)
    lines = []
    lines.append("# P2.5-D 装置组成偏还原的成因诊断 — 判读")
    lines.append("")
    lines.append("Status: `EXECUTED — offline analysis only`（无求解器调用）")
    lines.append("")
    lines.append("依据：论文仓任务卡 `data-side-handoff-p2.5d-20261120.md`；"
                 "输入 P2.2-A `baseline_142.csv`（142 窗，全部 completed）与 "
                 "P2.2-D `reconciled_windows.csv`（`combination=S1-C-m`）。")
    lines.append("")
    lines.append("## 结论摘要")
    lines.append("")
    lines.append("1. **在现有数据下，H-A 与 H-B 分不开。** 判别所需的两项——装置侧实际入炉水量、"
                 "反应器出口湿气 H₂O 分率——都没有在**所需位置**上的独立测量。R0 表逐项给出："
                 f"{sum(1 for r in r0_rows if r['available_in_project']=='yes')} 项 yes、"
                 f"{sum(1 for r in r0_rows if r['available_in_project']=='partial')} 项 partial、"
                 f"{sum(1 for r in r0_rows if r['available_in_project']=='derived')} 项 derived、"
                 f"{sum(1 for r in r0_rows if r['available_in_project']=='conditional')} 项 conditional、"
                 f"{sum(1 for r in r0_rows if r['available_in_project']=='no')} 项 no。"
                 f"洗涤塔出口确实有**湿流量通道**（{claim_txt}），"
                 "但由它得到的水分率是 `dry = wet*(1−Psat/P)` 的**回算**，恒等于饱和假设本身，"
                 "不是独立测量；缺口是「**缺所需位置**」，不是「缺湿流量通道」，"
                 "更不能写成「已有独立 H₂O 测量」。")
    lines.append("2. **单 ξ 共同位移在 P2 与全部窗相容，在 P1/P3 约一半窗超阈值，在 P4 全部超阈值。**"
                 "P2 只能说「与单一移动相容」，不是「证明够用」；P4 的阶段中位 χ² 已超阈值，"
                 "单一移动解释不了该阶段的缺口。")
    lines.append("3. **R3 给不出与 H-A 一致的方向（P1/P3 反而相反）。** ρ(ξ, T_out) 在 P1（−0.65）、P3（−0.74）显著为负，"
                 "P2 无信号（−0.24, p=0.12），仅 P4 为正（+0.67）。"
                 "按卡片预声明判据，这不是「与 H-A 方向一致」。"
                 "**但第 2 种结果记为「未获支持」，不是「被排除」**：ξ 按定义是模型与装置 CO₂ 的残差，"
                 "它跟模型 CO₂（ρ=−0.70/−0.33/−0.80/−0.68）而非装置 CO₂（0.21/0.08/0.25/0.89）；"
                 "P1–P3 内模型 CO₂ 与 T_out 强同向（0.95/0.97/0.95），机械地把 ξ 压成负相关。"
                 "该检验近似在测模型内部协方差，因此不构成对 H-A 的反证。（原字面判据的结果见 "
                 "VERDICT/outcome_2_predeclared_literal，仅作历史记录。）")
    lines.append("4. **R4 的 T_eq 不可引用。** 出口水情景从 5% 扫到 20%，T_eq 漂移远超 150 K，"
                 "按预声明判据记为「无信息」。R4 不参与 H-A 判定。")
    lines.append("5. 因此本卡属于第 3 种结果：**如实报「分不开」**，最终产出是所需观测清单（§R0）"
                 "与 P4 的阶段级不相容。")
    lines.append("")
    lines.append("## R0 可辨识性（本卡主要产出）")
    lines.append("")
    lines.append("| quantity | required_for | available | source_or_gap | note |")
    lines.append("|---|---|---|---|---|")
    for r in r0_rows:
        lines.append(f"| `{r['quantity']}` | {r['required_for']} | "
                     f"{r['available_in_project']} | {r['source_or_gap']} | {r['note']} |")
    lines.append("")
    lines.append("要点：三个观测量（CO/CO₂/H₂）都是**干基**，碳平衡没有水项，"
                 "所以装置侧 `Q_plant` 与反应器出口残余 H₂O 在现有数据下**都不可辨识**。"
                 f"洗涤塔出口确有**湿流量通道**（`outlet_syngas_wet_washtower_*`，{claim_txt}；"
                 "逐阶段覆盖率见上表与 protocol），这是真测量；"
                 "**但由它回算的水分率不是独立测量**：数据仓规定 `dry = wet*(1−Psat(T)/P)`，"
                 "故 `1 − dry/wet ≡ Psat(T)/P`，实测恒等偏差 "
                 f"{_dev_claim(wet_audit or {})}，即饱和假设本身被原样返回。"
                 "P1.6-D 的 9 窗 `y_H2O` = 0.4525–0.4646 来自**另一口径**的原始 FR_1015/FR1_1015 "
                 "湿/干流量差，残差 2.7–2.9% 是一次真实的交叉检验；但其原判读已保留"
                 "「未做流量计标定、未核实是否同为标准状态」的前提，故只能算**条件性估计**。"
                 "测点在**洗涤塔出口**（约 204.5 °C / 3.65 MPa），水由循环洗涤水支配，"
                 "不是反应器出口或激冷入口的水，**不能**用来反推 `Q_plant`。"
                 "所以卡片「出口湿气 H₂O 分率：没有」应更精确地写成"
                 "「缺的是**所需位置上的独立湿基 H₂O 测量**；洗涤塔出口有湿流量通道，"
                 "但那里的水分率是饱和假设的回算」——**不是**「已有湿基测量、只是位置不对」。")
    lines.append("")
    lines.append("另需注意：`y_H2O ≈ 0.46` 这个数（无论是 canonical 回算的中位还是 P1.6-D 的条件性估计）"
                 "比 R4 扫描的 5–20% 高一个量级。R4 是显式假设扫描而不是测量，两者不能混为一谈。")
    lines.append("")
    lines.append("## R1 逐窗 ξ 分布")
    lines.append("")
    lines.append("| phase | n | ξ 中位 mol/s | Q1 | Q3 | CV(ξ) | CV(dry) | CV(model CO₂) | CV(T_out) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for p in PHASE_ORDER:
        s = spread[p]
        lines.append(f"| {PHASE_LABEL[p]} | {s['n']} | {s['xi_median']:.2f} | {s['xi_q1']:.2f} | "
                     f"{s['xi_q3']:.2f} | {s['xi_cv']:.3f} | {s['dry_cv']:.3f} | "
                     f"{s['model_co2_cv']:.3f} | {s['t_out_cv']:.3f} |")
    lines.append("")
    lines.append("CV(ξ) **并不小于全部输入量的 CV**：它小于 `model_CO2` 的 CV（0.04–0.18），"
                 "但大于 `dry`（0.006–0.038）与 `T_out`（0.012–0.039）。"
                 "因此卡片 §2.1 的「ξ 的 CV 明显小于各输入量 → 更像阶段系统偏差」这一条"
                 "**没有被一致地满足**，不能据此把 ξ 说成纯阶段系统偏差。")
    lines.append("")
    lines.append("## R2 单 ξ 共同位移一致性")
    lines.append("")
    lines.append(f"`Σ_comp = diag(0.5², 0.3², 0.5²)`，`rank(Σ_comp) = {triplet['meta']['rank_Sigma']}`，"
                 f"`dof = {triplet['meta']['dof']}`，阈值 {CHI2_THRESHOLD}。")
    lines.append("")
    lines.append("| phase | n | 相容窗 | χ² 中位 | ξ̂ 中位 mol/s | ξ̂/dry 中位 | "
                 "中位\\|z\\| CO/CO₂/H₂ |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for p in PHASE_ORDER:
        t = triplet["per_phase"][p]
        lines.append(f"| {PHASE_LABEL[p]} | {t['n']} | {t['n_compatible']}/{t['n']} | "
                     f"{t['chi2_median']:.2f} | {t['xi_hat_median']:.2f} | "
                     f"{t['xi_hat_over_dry_median']:.4f} | "
                     f"{t['z_median_abs_CO']:.2f} / {t['z_median_abs_CO2']:.2f} / "
                     f"{t['z_median_abs_H2']:.2f} |")
    lines.append("")
    worst = max(PHASE_ORDER, key=lambda p: triplet["per_phase"][p]["chi2_median"])
    t = triplet["per_phase"][worst]
    zs = {"CO": t["z_median_abs_CO"], "CO2": t["z_median_abs_CO2"], "H2": t["z_median_abs_H2"]}
    worst_z = max(zs, key=zs.get)
    z_label = {"CO": "CO", "CO2": "CO₂", "H2": "H₂"}[worst_z]
    lines.append(f"残差最大的量：{z_label}（{zs[worst_z]:.2f}，在 {PHASE_LABEL[worst]}）。"
                 "CO₂ 的 σ 最小（0.3 vol%），同样的绝对偏差在 χ² 里权重最大（P3 的 CO₂ "
                 f"|z| 中位也有 {triplet['per_phase']['phase3']['z_median_abs_CO2']:.2f}）；"
                 "P4 三个分量同时放大（1.37 / 2.83 / 2.98），是该阶段整窗不相容的原因。")
    lines.append("")
    lines.append("`dry` ±4% 同步缩放：ξ̂ 的绝对量按同一比例变化"
                 f"（P1 {triplet['per_phase']['phase1']['xi_hat_m4_median']:.3f} / "
                 f"{triplet['per_phase']['phase1']['xi_hat_p4_median']:.3f} mol/s），"
                 "而 ξ̂/dry 与 χ² 不变（U2b 见下）。"
                 "这只检验绝对位移的尺度，不是组成/状态误差的影响。")
    lines.append("")
    lines.append("**措辞**：P2（44/44）只能说「与单一移动相容」，**不能**写成「证明够用」。"
                 f"P1 有 {triplet['per_phase']['phase1']['n'] - triplet['per_phase']['phase1']['n_compatible']}"
                 f"/{triplet['per_phase']['phase1']['n']} 窗超阈值，"
                 f"P3 有 {triplet['per_phase']['phase3']['n'] - triplet['per_phase']['phase3']['n_compatible']}"
                 f"/{triplet['per_phase']['phase3']['n']} 窗超阈值，"
                 f"P4 全部 {triplet['per_phase']['phase4']['n']} 窗超阈值"
                 f"（中位 χ² = {triplet['per_phase']['phase4']['chi2_median']:.2f}），"
                 "P4 的阶段级结论是「不相容」。")
    lines.append("")
    lines.append("## R3 温度相关")
    lines.append("")
    lines.append("| scope | variable | ρ | p | n | 判读 |")
    lines.append("|---|---|---:|---:|---:|---|")
    for r in corr_rows:
        lines.append(f"| {r['scope']} | {r['variable']} | {r['rho']:.3f} | "
                     f"{r['p_value']:.3g} | {r['n']} | {r['direction_verdict']} |")
    lines.append("")
    lines.append("H-A 的方向性预测是 **ρ > 0**（出口越热 → K_WGS 越小 → 需要更大的正向前移）。"
                 "实测 P1、P3 显著为负，全样本也为负（−0.40）。"
                 "归一的 ξ/dry 与 T_out 相关在 P1 减弱到 −0.35（p=0.03），P3 仍为 −0.65，"
                 "说明负号不是单纯由 dry 的窗口差异造成。"
                 "按预声明判据，这是「与 H-A 方向相反」，不是「不矛盾」。")
    lines.append("")
    lines.append("一个与判据无关的观察（诊断，不是新判据）：逐窗反解的 ξ 主要跟随**模型** CO₂ 的"
                 "窗口间起伏（ρ(ξ, model_CO₂) = " +
                 " / ".join(f"{diag[p]['rho_xi_modelCO2']:.2f}" for p in PHASE_ORDER) +
                 "），而不是装置 CO₂（ρ = " +
                 " / ".join(f"{diag[p]['rho_xi_plantCO2']:.2f}" for p in PHASE_ORDER) +
                 "，只有 P4 例外）；模型 CO₂ 的 CV（0.04–0.18）也远大于 T_out（0.012–0.039）。"
                 "在 P1–P3 内，模型 CO₂ 与 T_out 同向（ρ = " +
                 " / ".join(f"{diag[p]['rho_modelCO2_Tout']:.2f}" for p in PHASE_ORDER[:3]) +
                 "），于是 ξ 被压成负相关；P4 反过来（ρ = "
                 f"{diag['phase4']['rho_modelCO2_Tout']:.2f}），ξ 与 T_out 才成正相关。"
                 "也就是说，R3 的符号主要由阶段内模型自身的 CO₂–T_out 关系决定，"
                 "不是由装置侧决定——这正是卡片说这个检验很弱的原因。"
                 "本卡不据此把负号读成机理证据，只把它如实记为「与 H-A 方向相反」。")
    lines.append("")
    t4 = triplet["per_phase"]["phase4"]
    s4 = spread["phase4"]
    lines.append(f"P4 的方向虽然一致（ρ = +0.67），但同一阶段的单一移动拟合最差："
                 f"ξ̂ 中位 {t4['xi_hat_median']:.1f} mol/s，比逐窗反解中位 "
                 f"{s4['xi_median']:.1f} 低约 "
                 f"{100 * (s4['xi_median'] - t4['xi_hat_median']) / s4['xi_median']:.0f}%，"
                 f"中位 χ² = {t4['chi2_median']:.1f}，{t4['n']}/{t4['n']} 窗超阈值。"
                 "方向一致出现在单移动假设被拒绝的阶段，不能当作对 H-A 的支持。")
    lines.append("")
    lines.append("## R4 T_eq 对出口水情景的敏感性")
    lines.append("")
    lines.append("| phase | y_H2O | Q | T_eq (K) | 相对 10% 漂移 (K) | 备查口径漂移 (K) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for p in PHASE_ORDER:
        for y in Y_H2O_SCENARIOS:
            s = teq[p]["scenarios"][y]
            lines.append(f"| {PHASE_LABEL[p]} | {y:.0%} | {s['Q_median']:.4f} | "
                         f"{s['T_eq_median']:.1f} | {teq[p]['drifts'][y]:+.1f} | "
                         f"{teq[p]['drifts_alt'][y]:+.1f} |")
    lines.append("")
    lines.append(                 "四个阶段的 `max|drift|` 分别为 " +
                 " / ".join(f"{teq[p]['max_abs_drift_K']:.0f} K" for p in PHASE_ORDER) +
                 "，全部远超 150 K，按预声明判据记为 **「无信息，本卡不得引用」**。"
                 "`T_eq` 因此不能被引用为描述性代理量。"
                 "需要强调：R4 只决定 T_eq 能不能被引用，**任何结果都不构成对 H-A 的支持**。")
    lines.append("")
    lines.append("## 单元检查")
    lines.append("")
    lines.append(f"- **U1**：逐窗中位 ξ 与 P2.2-A 阶段常数的差，最大偏离卡面已知值 "
                 f"{unit_meta['u1_max_dev_pct_points']:.4f} 个百分点（容差 0.05）。"
                 "U1 是诊断项，不触发停止。")
    lines.append(f"- **U2**：3 窗独立重算，最大绝对差 {unit_meta['u2_max_abs_diff']:.3g}（容差 1e-9）。")
    lines.append(f"- **U2b**：`dry` 与各 `F_i` 同步 ±4%，`ξ̂/dry` 最大偏差 "
                 f"{unit_meta['u2b_max_dev_ov']:.3g}、χ² 最大偏差 "
                 f"{unit_meta['u2b_max_dev_chi2']:.3g}（容差 1e-9），通过。")
    lines.append("- **U3**：`correlations.csv` 每个检验都带 `n` 与 `n_missing`；142 窗全部可解，"
                 "无缺失。")
    lines.append("")
    lines.append("## 判定")
    lines.append("")
    lines.append("**第 3 种：分不开。** 理由：")
    lines.append("")
    lines.append("1. R0 显示判别 H-A/H-B 的两个关键量（入炉水量、反应器出口 H₂O）在现有数据下不可辨识；"
                 "洗涤塔出口的湿流量通道存在且覆盖近全时段，但由此回算的水分率恒等于饱和假设，"
                 "不是独立测量——缺口在位置，不在通道。")
    lines.append("2. R3 给不出与 H-A 一致的方向（P1/P3 反而相反），第 2 种结果因此**未获支持**；"
                 "但 ξ 与模型 CO₂ 的机械耦合使该检验不构成反证，故**不能**记为「被排除」。")
    lines.append("3. R2 是混合的：P2 全部相容，P1/P3 约一半超阈值，P4 全部超阈值。"
                 "P4 的阶段级结论是第 1 种（不相容，缺口不止一个成分），但不能推广到四阶段。")
    lines.append("")
    lines.append("**不采用第 1 种作为总判定**：卡片没有预声明「阶段是否相容」如何由逐窗计数聚合，"
                 "P1/P3 的中位 χ² 仍在阈值内；把 P4 的结果推广到全部阶段会超出数据。")
    lines.append("")
    lines.append("## 能确定 / 不能确定")
    lines.append("")
    lines.append("**能确定：**")
    lines.append("")
    lines.append("- 三观测量为干基，现有数据下 `Q_plant` 与反应器出口 H₂O 不可辨识（R0）。")
    lines.append("- 单一 WGS 移动在 P2 与全部窗相容；在 P4 与全部 17 窗不相容。")
    lines.append("- `ξ̂/dry` 与 χ² 对 `dry` 的同步缩放不变（U2b），R2 的判据不依赖流量规模。")
    lines.append("- T_eq 对出口水情景极不稳健，不可引用（R4）。")
    lines.append("")
    lines.append("**不能确定：**")
    lines.append("")
    lines.append("- 缺口是 H-A（缺下游过程）还是 H-B（水基准错），以及 ξ 的物理归属。")
    lines.append("- 沿程动力学是否相关（本卡按范围选择不改 WGS 动力学，未检验）。")
    lines.append("- P1/P3 那部分超阈值窗是结构差异还是分析仪口径问题（需要独立观测）。")
    lines.append("")
    lines.append("## 所需观测（若要与本卡之外的证据合流）")
    lines.append("")
    lines.append("1. 反应器出口或激冷入口的**湿基** H₂O 测量——现有湿基通道在洗涤塔出口，"
                 "位置不对。**注意：仅把取压/取样点前移不能闭合这一项**——现有水分率是 "
                 "`dry = wet*(1−Psat/P)` 的回算，若前移后仍用同一换算，循环依赖照旧。"
                 "需要的是**独立**的 H₂O 测量（如在线湿度/露点、或独立的湿基流量与干基流量组合），"
                 "或对湿/干两只流量计做独立标定并确认同为标准状态。")
    lines.append("2. 独立的入炉水测量：逐窗煤水分、激冷水流量、蒸汽流量（当前只有 2% 假设与"
                 "「无烧嘴蒸汽」的定性确认）。")
    lines.append("3. 激冷段/废锅的温度与停留时间剖面，用于判断下游是否真有变换窗口。")
    lines.append("4. 刘文或其他文献的湿气数据作交叉核对（现有对照表只有干基）。")
    lines.append("5. 若要检验沿程动力学，需另立 WGS 速率倍数敏感性实验（本卡不做）。")
    lines.append("")
    lines.append("## 歧义（未自行决定，交由统筹方）")
    lines.append("")
    for a in [
        "R4 水基准：卡片写 `F_H2O=(y/(1-y))*dry_total`，但变换后干气应为 `dry+ξ`；"
        "主口径按字面，备查口径同列，两者分类结论相同。",
        "R4 的 ξ 口径：卡片说「用 `*_xi` 列」即阶段常数 ξ；R1 的逐窗 ξ 未用于 R4。",
        "R2 的阶段聚合规则未预声明；本卡并报中位 χ² 与超阈值窗数，不定义阶段通过/不通过。",
        "R3 未规定单/双侧 p 与用模型还是实测 T_out；本卡用双侧 p 与模型 T_out。",
        "R1 的 `inversion_ok` 未定义；本卡定义为数值可解（分母 >1 pp、dry>0、结果有限），"
        "负 ξ 不算失败。",
        "P1.6-D 的湿基测量分母是洗涤塔出口干气，与 baseline 的模型 dry 不同位置；"
        "本卡未把它并入任何判据。",
    ]:
        lines.append(f"- {a}")
    lines.append("")
    return "\n".join(lines)


# --- 清单 -------------------------------------------------------------------

def assert_no_false_audit_claim(wet_audit: dict) -> None:
    """回归护栏：审计不完整时，产物里不得出现「全覆盖」或零偏差通过的字样。

    这是第三轮审查 P2 的直接防线——未核查绝不能被报成通过。
    """
    if wet_audit.get("status") == "complete":
        return
    interp = (OUT_DIR / "interpretation.md").read_text(encoding="utf-8")
    ident = (OUT_DIR / "identifiability.csv").read_text(encoding="utf-8")
    readings = (OUT_DIR / "readings.csv").read_text(encoding="utf-8")
    blob = interp + ident + readings
    banned = ["四阶段分钟级近全覆盖", "四阶段覆盖接近 100%", "覆盖接近 100%"]
    hits = [b for b in banned if b in blob]
    if hits:
        raise SystemExit(f"审计状态={wet_audit.get('status')} 却仍宣称全覆盖: {hits}")
    if wet_audit.get("max_dev") is None:
        for tok in ("0.00e+00", "0.0e+00"):
            if tok in ident or tok in interp:
                raise SystemExit(f"未核查却输出了零偏差字样: {tok}")

def write_manifest(products: list[str], inputs: list[Path]) -> None:
    lines = ["# P2.5-D 产物与输入清单（sha256，相对仓库根）",
             "# self 行：sha256.txt 自身的哈希按「去掉最后一行 self 行后的内容」计算，",
             "# 校验：sed '$d' sha256.txt | shasum -a 256 应与 self 行一致",
             "", "# products"]
    for name in products:
        if name == "sha256.txt":
            continue
        p = OUT_DIR / name
        lines.append(f"{sha256_of(p)}  {rel_path(p)}")
    lines += ["", "# inputs"]
    for p in inputs:
        if p.exists():
            lines.append(f"{sha256_of(p)}  {rel_path(p)}")
        else:
            # 缺失输入必须显式出现，不能静默跳过（否则自校验看不出它缺席）
            lines.append(f"MISSING  {rel_path(p)}")
    lines += ["", "# self"]
    text_wo_self = "\n".join(lines) + "\n"
    self_hash = hashlib.sha256(text_wo_self.encode("utf-8")).hexdigest()
    lines.append(f"{self_hash}  {rel_path(OUT_DIR / 'sha256.txt')}")
    (OUT_DIR / "sha256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_outputs(products: list[str], inputs: list[Path]) -> dict:
    bad_paths, bad_nan, bad_crlf, hash_mismatch = [], [], [], []
    for p in sorted(OUT_DIR.iterdir()):
        if not p.is_file():
            continue
        raw = p.read_bytes()
        if b"\r\n" in raw:
            bad_crlf.append(p.name)
        text = raw.decode("utf-8")
        if "/Users/" in text or "/home/" in text:
            bad_paths.append(p.name)
        if p.suffix == ".csv" and any(tok in {"nan", "NaN", "NAN"}
                                      for tok in text.replace(",", " ").split()):
            bad_nan.append(p.name)

    # 清单自校验（含 self 行）
    manifest = (OUT_DIR / "sha256.txt").read_text(encoding="utf-8")
    ml = manifest.rstrip("\n").split("\n")
    listed = {}
    for line in ml:
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            listed[parts[-1]] = parts[0]
    for name in products:
        key = rel_path(OUT_DIR / name)
        if key == rel_path(OUT_DIR / "sha256.txt"):
            continue
        if listed.get(key) != sha256_of(OUT_DIR / name):
            hash_mismatch.append(name)
    body = "\n".join(ml[:-1]) + "\n"
    self_key = rel_path(OUT_DIR / "sha256.txt")
    if listed.get(self_key) != hashlib.sha256(body.encode("utf-8")).hexdigest():
        hash_mismatch.append("sha256.txt(self)")
    for p in inputs:
        if p.exists() and listed.get(rel_path(p)) != sha256_of(p):
            hash_mismatch.append(rel_path(p))

    return {"files": sorted(p.name for p in OUT_DIR.iterdir() if p.is_file()),
            "absolute_path_hits": bad_paths, "literal_nan_hits": bad_nan,
            "crlf_hits": bad_crlf, "manifest_hash_mismatch": hash_mismatch}


# --- main -------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, rec = load_inputs()

    xi_df, xi_rows = build_xi_table(df)
    spread_rows, spread = build_xi_spread(df, xi_df)

    triplet_rows, triplet = build_triplet(df)
    corr_rows = build_correlations(df, xi_df, spread)
    teq_rows, teq = build_teq(df)
    wet_audit = audit_wet_basis()
    r0_rows, r0_meta = build_identifiability(rec, wet_audit=wet_audit)

    unit_rows, unit_meta = build_unit_checks(df, xi_df, spread, triplet)
    readings = build_readings(spread, triplet, corr_rows, teq, r0_rows, unit_meta,
                              wet_audit=wet_audit)

    products = ["identifiability.csv", "unit_checks.csv", "xi_window.csv",
                "xi_spread.csv", "triplet_span.csv", "correlations.csv",
                "teq_sensitivity.csv", "readings.csv", "protocol.json",
                "interpretation.md", "sha256.txt"]
    inputs = [BASELINE, P22D_WINDOWS, P16D_QUENCH,
              *CANONICAL_WET.values(),
              CANONICAL_DIR.parents[1] / "src" / "features" / "wet_to_dry_syngas.py"]

    write_csv(OUT_DIR / "identifiability.csv",
              ["quantity", "required_for", "available_in_project",
               "source_or_gap", "note"], r0_rows)
    write_csv(OUT_DIR / "xi_window.csv",
              ["window_id", "phase", "xi_mol_s", "xi_over_dry", "dry_total", "T_out_K",
               "T_wsr_K", "model_CO2", "plant_CO2", "inversion_ok", "reason"], xi_rows)
    write_csv(OUT_DIR / "xi_spread.csv",
              ["phase", "n", "n_ok", "xi_median", "xi_q1", "xi_q3", "xi_cv",
               "dry_cv", "model_co2_cv", "t_out_cv", "max_input_cv",
               "xi_cv_over_max_input_cv", "xi_phase_constant"], spread_rows)
    write_csv(OUT_DIR / "triplet_span.csv",
              ["row_kind", "phase", "window_id", "n", "n_compatible", "xi_hat",
               "xi_hat_over_dry", "chi2", "compatible", "z_CO", "z_CO2", "z_H2",
               "xi_hat_dry_m4pct", "xi_hat_dry_p4pct", "xi_over_dry_dry_m4pct",
               "xi_over_dry_dry_p4pct", "chi2_dry_m4pct", "chi2_dry_p4pct",
               "xi_hat_median", "xi_hat_over_dry_median", "chi2_median",
               "xi_hat_dry_m4pct_median", "xi_hat_dry_p4pct_median",
               "chi2_dry_m4pct_median", "chi2_dry_p4pct_median",
               "z_median_abs_CO", "z_median_abs_CO2", "z_median_abs_H2",
               "rank_Sigma", "dof", "chi2_threshold", "reason"], triplet_rows)
    write_csv(OUT_DIR / "correlations.csv",
              ["scope", "phase", "metric", "variable", "n", "n_missing", "rho",
               "p_value", "direction_verdict", "note"], corr_rows)
    write_csv(OUT_DIR / "teq_sensitivity.csv",
              ["phase", "y_H2O", "F_H2O_out_mol_s", "water_basis", "Q", "T_eq_K",
               "T_eq_C", "T_eq_q1_K", "T_eq_q3_K", "drift_vs_10pct_K",
               "Q_alt_post_shift_dry", "T_eq_alt_K", "drift_alt_vs_10pct_K",
               "n_windows", "max_abs_drift_K", "teq_usability", "reason"], teq_rows)
    write_csv(OUT_DIR / "unit_checks.csv",
              ["check", "phase", "n", "metric", "value", "reference", "tolerance",
               "abs_deviation_from_card_pct_points", "passed", "status", "note"],
              unit_rows)
    write_csv(OUT_DIR / "readings.csv",
              ["reading", "metric", "phase", "value", "note"], readings)

    diag = diagnostic_relations(df, xi_df)
    interp = build_interpretation(df, xi_df, diag, spread, triplet, corr_rows,
                                  teq, r0_rows, unit_meta, readings, wet_audit=wet_audit)
    (OUT_DIR / "interpretation.md").write_text(interp, encoding="utf-8")

    protocol = build_protocol(df, rec, spread, triplet, teq, r0_meta, unit_meta,
                              corr_rows, diag, wet_audit=wet_audit)
    (OUT_DIR / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    write_manifest(products, inputs)
    assert_no_false_audit_claim(wet_audit)
    verify = verify_outputs(products, inputs)

    if verify["absolute_path_hits"]:
        raise SystemExit(f"产物含绝对路径: {verify['absolute_path_hits']}")
    if verify["literal_nan_hits"]:
        raise SystemExit(f"产物含字面 NaN: {verify['literal_nan_hits']}")
    if verify["crlf_hits"]:
        raise SystemExit(f"产物含 CRLF: {verify['crlf_hits']}")
    if verify["manifest_hash_mismatch"]:
        raise SystemExit(f"清单哈希不一致: {verify['manifest_hash_mismatch']}")
    if verify["files"] != sorted(products):
        raise SystemExit(f"产物文件集不符: {verify['files']}")
    print("P2.5-D 完成:", len(verify["files"]), "个文件；清单自校验通过")


if __name__ == "__main__":
    main()
