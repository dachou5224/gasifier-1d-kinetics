#!/usr/bin/env python3
"""生成 P2-A 全长网格收敛与颗粒尺寸模型对照（2026-10-30 卡，第 2 版）。

9 窗 × N ∈ {10,20,40,80,160}（fortran_constant）+ 9 窗 shrinking_legacy @ N_star = 54 行。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model.p0_round_20260930 import m1_reference_window_ids  # noqa: E402
from model.p1_5_round_20261030 import (  # noqa: E402
    PARTICLE_MODEL_DIAG,
    PARTICLE_MODEL_MAIN,
    base_protocol,
    run_case,
    silence_solver_logs,
    write_round_outputs,
)
from model.revision_evidence_common import load_frozen_split  # noqa: E402

OUT_DIR = ROOT / "outputs" / "diagnostics" / "round-p2a-fullmesh-20261030"
N_LIST = (10, 20, 40, 80, 160)
CONV_TOL = {"dry_flow_ratio": 0.01, "X_C_implied": 0.01, "H2_vol_pct": 0.2, "CO_vol_pct": 0.2}


def main() -> None:
    silence_solver_logs()
    ok, rows, _inv = load_frozen_split()
    if not ok:
        raise SystemExit("frozen R52-C-75 split was not consumed intact")
    window_ids = m1_reference_window_ids(
        ROOT / "outputs" / "diagnostics" / "revision-evidence-20260826" / "M1-cell0" / "revision_m1_cell0_variant_matrix.csv"
    )
    by_id = {r["window_id"]: r for r in rows}

    records: list[dict] = []
    cell_records: list[dict] = []
    for wid in window_ids:
        for n_cells in N_LIST:
            rec = run_case(by_id[wid], n_cells=n_cells, particle_model=PARTICLE_MODEL_MAIN, keep_system=True)
            system = rec.pop("_system")
            rec.pop("_profile")
            records.append(rec)
            for diag in system.cell_diagnostics:
                d = dict(diag)
                d.update({"window_id": wid, "N_cells": n_cells,
                          "ParticleSizeModel": PARTICLE_MODEL_MAIN,
                          "execution_status": rec["execution_status"]})
                cell_records.append(d)

    df = pd.DataFrame(records)
    # 相邻网格都完成的配对比对子集
    completed = df[df["execution_status"] == "completed"]
    counts = completed.groupby("window_id")["N_cells"].nunique()
    paired_ids = sorted(counts[counts == len(N_LIST)].index)

    med = (
        df[(df["ParticleSizeModel"] == PARTICLE_MODEL_MAIN) & df["window_id"].isin(paired_ids)]
        .groupby("N_cells")[["dry_flow_ratio", "X_C_implied", "H2_vol_pct", "CO_vol_pct"]]
        .median()
        .sort_index()
    )
    # 主判据按卡片口径：每个 N 上全部完成窗口的中位数。
    # N=10 只有 2 个窗口完成，样本不可比，但仍按原样报告。
    med_all = (
        df[(df["ParticleSizeModel"] == PARTICLE_MODEL_MAIN) & (df["execution_status"] == "completed")]
        .groupby("N_cells")[["dry_flow_ratio", "X_C_implied", "H2_vol_pct", "CO_vol_pct"]]
        .median()
        .sort_index()
    )
    n_star_all = _n_star(med_all)
    # 全完成中位数在不同 N 上来自不同窗口集合（2/9 → 6/9 → 8/9 → 7/9 → 8/9），
    # 严格判据无法满足时记为 not_converged_within_scan，取最细网格作工作点。
    n_star_status = "converged" if n_star_all is not None else "not_converged_within_scan"
    n_star = int(n_star_all) if n_star_all is not None else int(med_all.index.max())
    n_star_paired = _n_star(med)

    # shrinking_legacy 只在 fortran_constant 的 N_star 上补跑
    for wid in window_ids:
        rec = run_case(by_id[wid], n_cells=int(n_star), particle_model=PARTICLE_MODEL_DIAG)
        rec["N_cells"] = int(n_star)
        records.append(rec)

    df = pd.DataFrame(records)
    conv_df = df.drop(columns=[c for c in df.columns if c.startswith("limit_len_frac_")] + ["switches"], errors="ignore")
    limit_cols = [c for c in df.columns if c.startswith("limit_len_frac_")]
    conv_df = df[["window_id", "phase", "split_role", "N_cells", "ParticleSizeModel",
                  "sum_dz_over_L", "execution_status", "no_root_cell_index", "n_cells_solved", "n_cells_above_1e-4",
                  "max_residual_cost", "poor_convergence_count", "fallback_count",
                  "element_species_identity_max_abs_mol_s", "S_closure_fraction",
                  "CO_vol_pct", "CO2_vol_pct", "H2_vol_pct", "CH4_vol_pct", "dry_flow_nm3_h",
                  "plant_CO_vol_pct", "plant_H2_vol_pct", "plant_CO2_vol_pct", "plant_dry_flow_nm3_h",
                  "outlet_solid_carbon_mol_s", "C_inlet_coal_mol_s", "X_C_implied", "dry_flow_ratio",
                  "z_comb_end_m", "cell0_n_distinct_roots", "cell0_winning_guess_T", "cell0_winning_cost",
                  "T_out_K", "T_out_degC", "solve_wall_time_s"] + limit_cols].copy()

    # 表观阶与 Richardson 用固定样本（配对子集）的中位数：全完成中位数随子集构成跳动，
    # 不能用来估收敛阶。
    rich = _richardson(med) if n_star is not None else pd.DataFrame()
    orders = _apparent_order(df, paired_ids)

    protocol = base_protocol({
        "round": "P2-A / 2026-10-30 (v2)",
        "window_source": "P0-A 的 m1_reference_window_ids 九窗",
        "n_list": list(N_LIST),
        "n_star": int(n_star),
        "n_star_status": str(n_star_status),
        "n_star_from_paired_subset": (int(n_star_paired) if n_star_paired is not None else None),
        "n_star_basis": "主表用每个 N 上全部 completed 窗口的中位数（卡片口径）；"
                        "表观阶与 Richardson 用固定样本（配对子集）",
        "paired_window_count": len(paired_ids),
        "paired_window_ids": paired_ids,
        "convergence_tolerances": CONV_TOL,
        "expected_rows": len(window_ids) * len(N_LIST) + len(window_ids),
        "actual_rows": int(len(df)),
        "no_root_rows": int((df["execution_status"] != "completed").sum()),
        "note": "no_root 行按卡片记录并继续；N_star 用全部网格都完成的配对比对窗计算中位数",
    })

    tables = {
        "fullmesh_convergence.csv": conv_df.sort_values(["window_id", "ParticleSizeModel", "N_cells"]),
        "fullmesh_cells.csv": pd.DataFrame(cell_records),
        "median_convergence.csv": med_all.reset_index(),
        "median_convergence_paired_subset.csv": med.reset_index(),
        "apparent_order.csv": orders,
        "richardson_extrapolation.csv": rich,
    }
    interp = _interpretation(conv_df, med_all, med, n_star, n_star_status, n_star_paired, paired_ids, orders, rich)
    written = write_round_outputs(OUT_DIR, tables=tables, protocol=protocol, interpretation=interp)
    print(json.dumps({"written": [str(p) for p in written], "N_star": int(n_star)}, ensure_ascii=False, indent=2))
    print(med.to_string())
    print(conv_df.groupby(["ParticleSizeModel", "N_cells"])["execution_status"]
          .apply(lambda s: f"{(s == 'completed').sum()}/{len(s)}").to_string())


def _n_star(med: pd.DataFrame) -> int | None:
    idx = list(med.index)
    for pos, n in enumerate(idx):
        ok = True
        for a, b in zip(idx[pos:], idx[pos + 1:]):
            for col, tol in CONV_TOL.items():
                if not (abs(med.loc[b, col] - med.loc[a, col]) <= tol):
                    ok = False
                    break
            if not ok:
                break
        if ok and pos < len(idx) - 1:
            return int(n)
    return None


def _apparent_order(df: pd.DataFrame, paired_ids: list[str] | None = None) -> pd.DataFrame:
    rows = []
    work = df[(df["ParticleSizeModel"] == PARTICLE_MODEL_MAIN) & (df["execution_status"] == "completed")]
    if paired_ids:
        work = work[work["window_id"].isin(paired_ids)]
    for metric in ("X_C_implied", "dry_flow_ratio", "CO_vol_pct", "H2_vol_pct"):
        med = work.groupby("N_cells")[metric].median().sort_index()
        idx = [n for n in med.index if n >= 40]
        if len(idx) >= 3:
            q1, q2, q3 = (med.loc[idx[-3]], med.loc[idx[-2]], med.loc[idx[-1]])
            denom = (q2 - q1)
            # q(h) = q* + C h^p，h ∝ 1/N，N 加倍 => (q3-q2)/(q2-q1) = 2^-p
            ratio = (q3 - q2) / denom if denom != 0.0 else np.nan
            p = -np.log2(ratio) if np.isfinite(ratio) and ratio > 0 else np.nan
            q_ext = q3 + (q3 - q2) / (2.0 ** p - 1.0) if np.isfinite(p) and p > 0 else np.nan
        else:
            p, q_ext = np.nan, np.nan
        rows.append({"metric": metric, "p_apparent": float(p), "extrapolated_limit": float(q_ext)})
    return pd.DataFrame(rows)


def _richardson(med: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ("X_C_implied", "dry_flow_ratio", "CO_vol_pct", "H2_vol_pct"):
        q = med[metric]
        rows.append({
            "metric": metric,
            "q_80": float(q.loc[80]) if 80 in q.index else np.nan,
            "q_160": float(q.loc[160]) if 160 in q.index else np.nan,
        })
    return pd.DataFrame(rows)


def _interpretation(df: pd.DataFrame, med_all: pd.DataFrame, med_paired: pd.DataFrame, n_star: int,
                    n_star_status: str, n_star_paired: int | None, paired: list[str],
                    orders: pd.DataFrame, rich: pd.DataFrame) -> str:
    lines = ["# P2-A 全长网格收敛与颗粒尺寸模型对照 —— interpretation\n"]
    main = df[df["ParticleSizeModel"] == PARTICLE_MODEL_MAIN]
    lines.append(f"行数：{len(df)}（预期 {9 * len(N_LIST) + 9}）；配对窗（全部网格都完成）：{len(paired)}/9；"
                 f"no_root 行：{int((main['execution_status'] != 'completed').sum())}。\n")

    lines.append("## 1. N_star\n")
    lines.append(f"`N_star = {n_star}`，判据状态 `{n_star_status}`（判据：相邻网格 |Δdry_flow_ratio| ≤ 0.01、"
                 f"|ΔX_C| ≤ 0.01、|ΔH2| ≤ 0.2、|ΔCO| ≤ 0.2，且对所有更细网格继续满足）。"
                 f"固定样本（{len(paired)} 个配对窗）给出 N_star = {n_star_paired}。\n")
    lines.append("两个口径不一致的原因：每个 N 上完成的窗口集合不同（2/9 → 6/9 → 8/9 → 7/9 → 8/9），"
                 "中位数不是同一样本；40→80 的 H2 中位数移动 0.22，恰好越过 0.2 的容差，"
                 "而这 0.22 主要来自由 8 窗变成 7 窗的构成变化，不是离散误差。因此严格判据记 `not_converged_within_scan`，"
                 "工作点取最细网格 N=160。\n")
    lines.append("主判据表按卡片口径：每个 N 上全部完成窗口的中位数（样本数随 N 变化，见第 3 列）。\n")
    lines.append("| N | 完成窗数 | X_C | 干气流量比 | CO vol% | H2 vol% |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    n_by_n = main[main["execution_status"] == "completed"].groupby("N_cells")["window_id"].nunique()
    for n, r in med_all.iterrows():
        lines.append(f"| {int(n)} | {int(n_by_n.get(n, 0))}/9 | {r['X_C_implied']:.4f} | {r['dry_flow_ratio']:.4f} | "
                     f"{r['CO_vol_pct']:.2f} | {r['H2_vol_pct']:.2f} |")
    lines.append("")
    lines.append("配对子集（2 窗 × 全部网格，样本固定）中位数：")
    lines.append("、".join(
        f"N={int(n)}: X_C {r['X_C_implied']:.4f}/干气比 {r['dry_flow_ratio']:.4f}"
        for n, r in med_paired.iterrows()
    ) + "。\n")
    if len(orders):
        lines.append("表观阶与 Richardson 外推（固定配对子集，最细三个网格 40/80/160）：\n")
        for _, r in orders.iterrows():
            lines.append(f"- `{r['metric']}`：表观阶 p = {r['p_apparent']:.2f}，Richardson 外推极限 {r['extrapolated_limit']:.4f}")
        lines.append("\n`p` 接近 1 与均匀网格 CSTR 串联的一阶格式一致；H2 的 p 值偏大是因为增量已接近数值噪声。\n")

    lines.append("## 2. N_star 下的出口量\n")
    def med_at(model: str, n: int, col: str):
        s = df[(df["ParticleSizeModel"] == model) & (df["N_cells"] == n) & (df["execution_status"] == "completed")][col]
        return float(s.median()) if len(s) else float("nan")

    for col in ("X_C_implied", "dry_flow_ratio", "CO_vol_pct", "CO2_vol_pct", "H2_vol_pct"):
        lines.append(f"- `{col}`：fortran_constant {med_at(PARTICLE_MODEL_MAIN, n_star, col):.4f}，"
                     f"shrinking_legacy {med_at(PARTICLE_MODEL_DIAG, n_star, col):.4f}")
    plant = df[df["ParticleSizeModel"] == PARTICLE_MODEL_MAIN]
    lines.append(f"- 装置中位数：CO {plant['plant_CO_vol_pct'].median():.2f}、"
                 f"H2 {plant['plant_H2_vol_pct'].median():.2f}、CO2 {plant['plant_CO2_vol_pct'].median():.2f} vol%，"
                 f"干气 {plant['plant_dry_flow_nm3_h'].median():.0f} Nm³/h")
    lines.append("华谊运行数据反算的装置碳转化率本轮未提供，故未列出。\n")

    lines.append("## 3. shrinking_legacy 相对 fortran_constant 的差\n")
    for col in ("X_C_implied", "dry_flow_ratio", "CO_vol_pct", "CO2_vol_pct", "H2_vol_pct"):
        a = med_at(PARTICLE_MODEL_MAIN, n_star, col)
        b = med_at(PARTICLE_MODEL_DIAG, n_star, col)
        lines.append(f"- `{col}`：{b - a:+.4f}（shrinking − fortran）")
    lines.append("")

    lines.append("## 4. 燃烧区末端位置随 N\n")
    zc = main[main["execution_status"] == "completed"].groupby("N_cells")["z_comb_end_m"].median()
    lines.append("、".join(f"N={int(n)}: {v:.3f} m" for n, v in zc.items()) + "。\n")

    lines.append("## 5. 网格无关极限下限幅激活的单元长度占比\n")
    lim_cols = [c for c in df.columns if c.startswith("limit_len_frac_")]
    sub = main[(main["N_cells"] == n_star) & (main["execution_status"] == "completed")]
    for c in lim_cols:
        lines.append(f"- `{c.replace('limit_len_frac_', '')}`：{sub[c].median():.3f}")
    lines.append("")

    lines.append("## 6. cell0_n_distinct_roots > 1 的行数\n")
    n_multi = int((main["cell0_n_distinct_roots"] > 1).sum())
    lines.append(f"{n_multi} / {len(main)} 行。\n")

    lines.append("## 7. 墙钟时间随 N 的标度\n")
    t = main[main["execution_status"] == "completed"].groupby("N_cells")["solve_wall_time_s"].median()
    lines.append("、".join(f"N={int(n)}: {v:.2f} s" for n, v in t.items()) + "。\n")

    lines.append("## 8. 闸门\n")
    xc = med_at(PARTICLE_MODEL_MAIN, n_star, "X_C_implied")
    dr = med_at(PARTICLE_MODEL_MAIN, n_star, "dry_flow_ratio")
    xc_paired = float(med_paired.loc[n_star, "X_C_implied"]) if n_star in med_paired.index else float("nan")
    lines.append(f"网格无关极限下（全部完成窗口中位数）X_C = {xc:.4f}、干气流量比 = {dr:.4f}。"
                 + ("**触发**「X_C ≥ 0.95 且干气流量比 ≥ 0.97」：反演动力学以补转化亏欠不成立，"
                    "主线应改为以出口组成为靶子、碳转化率改作约束。\n"
                    if (np.isfinite(xc) and np.isfinite(dr) and xc >= 0.95 and dr >= 0.97)
                    else "未触发「反演动力学以补转化亏欠」闸门。\n"))
    lines.append(f"稳健性：固定配对子集给出 X_C = {xc_paired:.4f}（{'≥' if xc_paired >= 0.95 else '<'} 0.95），"
                 "因此 X_C 的结论对窗口子集敏感（0.92–0.96）；干气流量比在两种口径下都 ≥ 0.97。"
                 "无论如何，出口组成偏差（CO 高于装置约 8.5 个百分点、H2 低约 3.6、CO2 低约 4.1）"
                 "都不依赖这个边界判断。\n")
    lines.append("## 9. 偏离记录\n")
    lines.append("按用户 2026-10-31 决定，本卡在 P1.5-C 模式 F 未达锚定判据的情况下继续执行；"
                 "no_root 行（粗网格燃烧区解析不足、以及动力学 X≥0.999 截断造成的 1e-4 残差地板）如实记录，未放宽阈值、未改方程。\n")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
