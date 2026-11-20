#!/usr/bin/env python3
"""P2.4-A 判定的证据固化（2026-11-20）。

用途：为「是否值得做完全耦合双组分」这一判定提供可复核的证据，而不是只留在
对话记录里。三项检查：

    A  C0（现有基线）的组成缺口：CO2 / CO / H2 相对装置值的偏差，ξ 之前与之后各一份
    B  ξ 与慢组分的 H2O 预算冲突：闭合 CO2 所需的 ξ 与慢组分 C+H2O 的耗汽之和
       是否超过下游可用 H2O
    C  ξ 的遮蔽效应：ξ 把 CO2 缺口吸收掉多少、又对 CO / H2 的偏差做了什么

数据来源（只读）：

    A、C  P2.2-A `baseline_142.csv`，它已同时存 ξ 前（`*_vol_pct`）与 ξ 后
          （`*_vol_pct_xi`）两套组成以及 `xi_mol_s`，因此无需重跑求解器
    B     需要主流出口的 F_H2O。用 P2.4-A 的 `solve_main`（φ=0.2）在 9 个参照窗上取；
          主流求解走 P2.4-A 的磁盘缓存，重复运行几乎零成本

产出：`outputs/diagnostics/round-p2.4a-20261119/followup/`
    `c0_gap.csv`、`xi_budget.csv`、`xi_masking.csv`、`readings.csv`、
    `protocol.json`、`sha256.txt`

用法：

    ./.venv/bin/python scripts/diagnose_p24a_followup.py \\
        --coal-set current \\
        --windows-csv outputs/diagnostics/p2.2d-20261118/windows_142.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import generate_p2_4a_20261119 as p24a  # noqa: E402

OUT_DIR = p24a.OUT_DIR / "followup"
P22A_BASELINE = ROOT / "outputs" / "diagnostics" / "round-p2.2a-20261118" / "baseline_142.csv"
P24A_DIR = p24a.OUT_DIR

PHASES = p24a.PHASES
PHASE_LABEL = p24a.PHASE_LABEL
M_C_KG_PER_MOL = p24a.M_C_KG_PER_MOL


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path) -> str:
    p = Path(path).resolve()
    try:
        return p.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return Path(os.path.relpath(p, ROOT.resolve())).as_posix()


def write_csv(df: pd.DataFrame, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / name
    df.replace({np.nan: ""}).to_csv(out, index=False, lineterminator="\n")
    return out


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _med(xs: list[float]) -> float:
    vals = [x for x in xs if math.isfinite(x)]
    return float(np.median(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# A + C：C0 的组成缺口与 ξ 的遮蔽（读 P2.2-A baseline，无需重跑）
# ---------------------------------------------------------------------------
def check_c0_gap() -> tuple[list[dict], list[dict]]:
    b = pd.read_csv(P22A_BASELINE, keep_default_na=False)
    b = b[b["execution_status"].astype(str) == "completed"]
    gap_rows: list[dict] = []
    mask_rows: list[dict] = []
    for ph in PHASES:
        s = b[b["phase"] == ph]
        if not len(s):
            continue
        for obs, mcol, pcol, xcol in (
            ("CO2", "CO2_vol_pct", "plant_CO2_vol_pct", "CO2_vol_pct_xi"),
            ("CO", "CO_vol_pct", "plant_CO_vol_pct", "CO_vol_pct_xi"),
            ("H2", "H2_vol_pct", "plant_H2_vol_pct", "H2_vol_pct_xi"),
        ):
            pre = _med([_f(v) for v in s[mcol]])
            post = _med([_f(v) for v in s[xcol]])
            plant = _med([_f(v) for v in s[pcol]])
            gap_rows.append({
                "phase": ph, "observable": obs,
                "model_before_xi_pp": pre, "plant_pp": plant,
                "gap_before_xi_pp": pre - plant,
                "model_after_xi_pp": post,
                "gap_after_xi_pp": post - plant,
                "n_windows": int(len(s)),
            })
        # C：ξ 吸收了 CO2 缺口的多少比例
        pre = _med([_f(v) for v in s["CO2_vol_pct"]])
        post = _med([_f(v) for v in s["CO2_vol_pct_xi"]])
        plant = _med([_f(v) for v in s["plant_CO2_vol_pct"]])
        abs_before = abs(pre - plant)
        abs_after = abs(post - plant)
        xi_med = _med([_f(v) for v in s["xi_mol_s"]])
        mask_rows.append({
            "phase": ph,
            "CO2_abs_gap_before_pp": abs_before,
            "CO2_abs_gap_after_pp": abs_after,
            "fraction_of_gap_absorbed_by_xi": (
                1.0 - abs_after / abs_before if abs_before > 1e-12 else float("nan")),
            "xi_median_mol_s": xi_med,
            "n_windows": int(len(s)),
        })
    return gap_rows, mask_rows


# ---------------------------------------------------------------------------
# B：ξ 的水需求（两个层面必须分开）
# ---------------------------------------------------------------------------
def check_xi_budget(coal_set: str) -> tuple[list[dict], list[dict]]:
    """对每个 (φ, ΔE, phase)：ξ 与慢组分的水需求，分两个层面报告。

    **层面 1（实现限制）**：P2.4-A 的 `xi_from_gas` 在 H2O 或 CO 变负时返回失败，
    所以 ξ 实际可用到 `min(F_H2O_after_slow, F_CO)`。需要超过它的窗即「需激冷补水」——
    这是 P1.13-A 的 P4 口径，是**标记项**，不是可行性判据。

    **层面 2（物理供给）**：P1.13-A 第 23 行明确写 ξ 的「水由激冷水提供，
    **不受出口 H2O 限制**」。因此「ξ > 出口 H2O」**不是物理不可行**，只说明需要补水；
    本项目没有激冷水供给量的上界数据，不能据此判不可行。

    2026-11-20 审查更正（三条）：
    (a) `p24a.add_back()` 已经把 `a_H2O_mol` 从 `gm[7]` 里扣掉了，所以 `gm[7]` 是
        **慢组分反应之后的剩余水**。此前又把同一份 `a_H2O_mol` 加到需求上，
        等于双重计算，把 phase 3 从 8/11 错报成 0/11。
    (b) 因此正确做法是：**ξ 单独**与剩余水比（层面 1），或**总需求**与扣减前的水比。
        这里两个都给。
    (c) `xi <= F_H2O` 不能当作物理限制（见层面 2）。
    """
    cp = pd.read_csv(P24A_DIR / "comparison_points.csv", keep_default_na=False)
    cp = cp[cp["reachable"].astype(str).str.lower() == "true"]
    ref = p24a.p22a._ref_windows("main")
    rows: list[dict] = []
    window_rows: list[dict] = []
    for (phi, dE), g in cp.groupby(["phi", "dE_kJ_per_mol"], sort=True):
        phi = float(phi); dE = float(dE)
        for ph in PHASES:
            sub = g[g["phase"] == ph]
            if not len(sub):
                continue
            f = 10.0 ** float(sub["log10_f_slow"].iloc[0])
            recs = []
            for w in ref[ph]:
                sn = p24a.solve_main(window_id=w, coal_set=coal_set,
                                     carrier_tag=p24a.MAIN_CARRIER,
                                     hl=p24a.MAIN_HL, pert="nom", phi=phi)
                if not sn.get("ok"):
                    continue
                sl = p24a.slow_integrate(sn, dE_kjmol=dE, f_slow=f)
                if not sl.get("ok"):
                    continue
                a1 = float(sl["a_H2O_mol"])
                a2 = float(sl["a_CO2_mol"])
                h2o_before = float(sn["F_H2O"])          # 慢组分之前
                ab = p24a.add_back(sn, sl)
                gm = np.asarray(ab["gas_moles_after"], dtype=float)
                # gm[7] 已扣掉 a1 → 这是慢组分之后的剩余水
                h2o_after = float(gm[7])
                dry = float(sum(gm[:7]))
                plant_co2 = _f(p24a.p22a.table("main").loc[w, "co2_vol_pct_mean"])
                k = plant_co2 / 100.0
                xi = (k * dry - gm[3]) / (1.0 - k) if abs(1.0 - k) > 1e-12 else float("nan")
                recs.append(dict(a1=a1, a2=a2, h2o_before=h2o_before,
                                 h2o_after=h2o_after, f_co=float(gm[2]),
                                 xi=float(xi)))
            if not recs:
                continue

            # 2026-11-20 第二次审查（P2）：必须**逐窗**算余量再汇总。
            # 此前先取 median(xi) 与 median(remaining) 再相减取 max(...,0)——
            # 中位不可加。反例：xi=[1,10,10]、剩余=[2,2,20]，
            # 中位相减给 10-2=8，而逐窗补水=[0,8,0] 的中位是 0。
            per_window = []
            for k, r in enumerate(recs):
                surplus = r["h2o_after"] - r["xi"]
                quench = max(-surplus, 0.0)
                co_margin = r["f_co"] - r["xi"]
                before_margin = r["h2o_before"] - (r["xi"] + r["a1"])
                per_window.append({
                    "phi": phi, "dE_kJ_per_mol": dE, "phase": ph,
                    "window_index": k,
                    "slow_a_H2O_mol": r["a1"],
                    "slow_a_CO2_mol": r["a2"],
                    "xi_needed_mol": r["xi"],
                    "h2o_at_outlet_before_slow_mol": r["h2o_before"],
                    "h2o_remaining_after_slow_mol": r["h2o_after"],
                    "F_CO_available_mol": r["f_co"],
                    "h2o_surplus_mol": surplus,
                    "quench_water_mol": quench,
                    "window_needs_quench": bool(quench > 1e-9),
                    "co_margin_mol": co_margin,
                    "window_co_ok": bool(co_margin >= -1e-9),
                    "total_demand_vs_before_margin_mol": before_margin,
                })
            window_rows.extend(per_window)

            n = len(per_window)
            n_need = sum(1 for w in per_window if w["window_needs_quench"])
            qs = [w["quench_water_mol"] for w in per_window]
            xi_s = [w["xi_needed_mol"] for w in per_window]
            sur = [w["h2o_surplus_mol"] for w in per_window]
            co_ok_n = sum(1 for w in per_window if w["window_co_ok"])
            before_ok_n = sum(
                1 for w in per_window
                if w["total_demand_vs_before_margin_mol"] >= -1e-9)
            rows.append({
                "phi": phi, "dE_kJ_per_mol": dE, "phase": ph,
                "n_windows": n,
                "n_windows_need_quench": n_need,
                "all_windows_need_quench": bool(n_need == n),
                "any_window_needs_quench": bool(n_need > 0),
                "median_window_quench_mol": _med(qs),
                "max_window_quench_mol": float(max(qs)) if qs else 0.0,
                "median_xi_mol": _med(xi_s),
                "median_window_surplus_mol": _med(sur),
                "min_window_surplus_mol": float(min(sur)) if sur else float("nan"),
                "n_windows_co_ok": co_ok_n,
                "n_windows_total_demand_ok": before_ok_n,
                "config_level_rule": (
                    "all_windows = 全部窗均不需补水；any_window = 任一窗需补水；"
                    "median_window = 中位窗需补水。三者分列，不合并成单一'可行'"),
                "all_windows_ok_no_quench": bool(n_need == 0),
                "median_window_ok_no_quench": bool(_med(qs) <= 1e-9),
                "physically_infeasible": None,
            })
    return rows, window_rows


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coal-set", required=True,
                    choices=("current", "legacy_20261118", "lims_heat"))
    ap.add_argument("--windows-csv", required=True)
    args = ap.parse_args()
    p24a.silence_solver_logs()
    p24a.p22a.prepare_coal_inputs()
    p24a.p22a.require_pairing(args.windows_csv, args.coal_set, "compare")
    p24a.p22a.init_ctx(Path(args.windows_csv))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gap, mask = check_c0_gap()
    budget, budget_windows = check_xi_budget(args.coal_set)

    write_csv(pd.DataFrame(gap), "c0_gap.csv")
    write_csv(pd.DataFrame(mask), "xi_masking.csv")
    write_csv(pd.DataFrame(budget), "xi_budget.csv")
    write_csv(pd.DataFrame(budget_windows), "xi_budget_windows.csv")

    # 汇总读数
    readings: list[dict] = []
    for r in gap:
        readings.append({
            "reading": "A", "metric": f"gap_before_xi_{r['observable']}",
            "phase": r["phase"], "value": r["gap_before_xi_pp"], "note": "",
        })
    for r in mask:
        readings.append({
            "reading": "C", "metric": "co2_gap_absorbed_by_xi",
            "phase": r["phase"], "value": r["fraction_of_gap_absorbed_by_xi"],
            "note": (f"CO2 绝对缺口 {r['CO2_abs_gap_before_pp']:.2f} → "
                     f"{r['CO2_abs_gap_after_pp']:.2f} pp；"
                     f"ξ 中位 {r['xi_median_mol_s']:.1f} mol/s"),
        })
    # 逐窗口径汇总（2026-11-20 第二次审查）：配置级定义显式分三种，不合并
    for ph in PHASES:
        sub = [r for r in budget if r["phase"] == ph]
        if not sub:
            continue
        readings.append({
            "reading": "B1", "metric": "configs_all_windows_no_quench", "phase": ph,
            "value": sum(1 for r in sub if r["all_windows_ok_no_quench"]),
            "note": (f"{sum(1 for r in sub if r['all_windows_ok_no_quench'])}/{len(sub)} "
                     f"个配置**全部窗**都不需补水（最严口径）"),
        })
        readings.append({
            "reading": "B2", "metric": "configs_any_window_needs_quench", "phase": ph,
            "value": sum(1 for r in sub if r["any_window_needs_quench"]),
            "note": (f"{sum(1 for r in sub if r['any_window_needs_quench'])}/{len(sub)} "
                     f"个配置**至少一窗**需补水（最松口径）"),
        })
        readings.append({
            "reading": "B3", "metric": "configs_median_window_needs_quench", "phase": ph,
            "value": sum(1 for r in sub if not r["median_window_ok_no_quench"]),
            "note": (f"{sum(1 for r in sub if not r['median_window_ok_no_quench'])}"
                     f"/{len(sub)} 个配置的**中位窗**需补水"),
        })
        worst = max(sub, key=lambda r: r["max_window_quench_mol"])
        readings.append({
            "reading": "B4", "metric": "max_single_window_quench_mol", "phase": ph,
            "value": worst["max_window_quench_mol"],
            "note": (f"最大单窗补水量出现在 φ={worst['phi']} "
                     f"ΔE={worst['dE_kJ_per_mol']:.0f}："
                     f"{worst['max_window_quench_mol']:.2f} mol/s"
                     f"（该配置 {worst['n_windows_need_quench']}/"
                     f"{worst['n_windows']} 窗需补水）"),
        })
    write_csv(pd.DataFrame(readings), "readings.csv")

    protocol = {
        "purpose": "P2.4-A 判定的证据固化：为「是否做完全耦合双组分」提供可复核依据",
        "sources": {
            "A_C": _rel(P22A_BASELINE),
            "B": _rel(P24A_DIR / "comparison_points.csv"),
        },
        "checks": {
            "A": "C0（现有基线）的 CO2/CO/H2 缺口，ξ 前与 ξ 后各一份",
            "B": "ξ 需求 + 慢组分耗汽 vs 下游可用 H2O（ξ 的解析式与两个上界）",
            "C": "ξ 把 CO2 缺口吸收了多大比例",
        },
        "xi_formula": "CO2 = (F_CO2+xi)/(dry+xi)；解 xi = (k*dry - F_CO2)/(1-k)，k=plant/100",
        "xi_bounds": (
            "IMPLEMENTATION limit: P2.4-A's xi_from_gas fails when H2O or CO goes "
            "negative, so usable xi is bounded by the remaining H2O and by F_CO. "
            "PHYSICAL supply: P1.13-A line 23 states quench water supplies xi and is "
            "NOT limited by outlet H2O, so xi > outlet H2O is a quench-water "
            "requirement (P1.13-A P4 marker), not an infeasibility. No upper bound on "
            "quench supply is available in this project, so physically_infeasible is "
            "left null rather than asserted."),
        "no_rerun_for_A_C": True,
        "coal_set": args.coal_set,
        "windows_csv": _rel(Path(args.windows_csv)),
    }
    (OUT_DIR / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# P2.4-A followup 证据清单（sha256，相对仓库根）", ""]
    for name in ("c0_gap.csv", "xi_masking.csv", "xi_budget.csv",
                 "xi_budget_windows.csv", "readings.csv", "protocol.json"):
        p = OUT_DIR / name
        lines.append(f"{_sha(p)}  {_rel(p)}")
    lines += ["", "# inputs"]
    for p in (P22A_BASELINE, P24A_DIR / "comparison_points.csv",
              P24A_DIR / "slow_pool_profiles.csv"):
        lines.append(f"{_sha(p)}  {_rel(p)}")
    (OUT_DIR / "sha256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[A] C0 缺口（ξ 前）：")
    for r in gap:
        print(f"    {PHASE_LABEL[r['phase']]} {r['observable']:3s} "
              f"{r['gap_before_xi_pp']:+7.2f} pp")
    print(f"[C] ξ 吸收的 CO2 缺口比例：")
    for r in mask:
        print(f"    {PHASE_LABEL[r['phase']]}: {r['fraction_of_gap_absorbed_by_xi']*100:5.1f}%"
              f"（{r['CO2_abs_gap_before_pp']:.2f} → {r['CO2_abs_gap_after_pp']:.2f} pp）")
    print("[B] 逐窗口径（配置级三定义分列）：")
    for ph in PHASES:
        sub = [r for r in budget if r["phase"] == ph]
        if not sub:
            continue
        n_all = sum(1 for r in sub if r["all_windows_ok_no_quench"])
        n_any = sum(1 for r in sub if r["any_window_needs_quench"])
        n_med = sum(1 for r in sub if not r["median_window_ok_no_quench"])
        mx = max(r["max_window_quench_mol"] for r in sub)
        print(f"    {PHASE_LABEL[ph]}: 全部窗不需补水 {n_all}/{len(sub)}；"
              f"任一窗需补水 {n_any}/{len(sub)}；中位窗需补水 {n_med}/{len(sub)}；"
              f"最大单窗补水 {mx:.1f} mol/s")
    print(f"    清单已写：{_rel(OUT_DIR / 'sha256.txt')}")


if __name__ == "__main__":
    main()
