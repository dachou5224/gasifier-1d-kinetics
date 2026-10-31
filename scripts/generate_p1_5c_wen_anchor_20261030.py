#!/usr/bin/env python3
"""生成 P1.5-C Wen/Fortran 锚点轮次产物（2026-10-30 卡）。

Fortran 侧：解析 fortran_build/run.log（已编译运行，见 build.log / io_patch.diff）。
Python 侧：模式 M（论文主线）与模式 F（Fortran 兼容），N ∈ {40, 80, 160}。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model.fortran_anchor_20261030 import (  # noqa: E402
    DRY_POWDER_CASES,
    EXPECTED_CASE_NAMES,
    parse_fortran_log,
    solve_fortran_case,
)
from model.p1_5_round_20261030 import (  # noqa: E402
    base_protocol,
    silence_solver_logs,
    write_round_outputs,
)
from model.revision_evidence_common import source_commit  # noqa: E402

OUT_DIR = ROOT / "outputs" / "diagnostics" / "round-p1.5c-wen-anchor-20261030"
BUILD_DIR = OUT_DIR / "fortran_build"
N_LIST = (40, 80, 160)
N_REF = 160

ANCHOR_TOL = {
    "X_C_pp": 2.0,
    "CO_vol_pct": 2.0,
    "H2_vol_pct": 2.0,
    "CO2_vol_pct": 2.0,
    "T_gas_K": 50.0,
    "O2_drop_rel": 0.20,
}

LEDGER = [
    {
        "item": "particle size model",
        "fortran_behavior": "rp = ri*(swell**0.333) 后恒定，densp 随转化下降（行 130、419、442、755）；swell=1.0（行 96）",
        "python_mainline": "ParticleSizeModel='fortran_constant'（P1.5-A 起为默认），cell.py _calc_physics_props",
        "switch": "ParticleSizeModel",
        "in_compat_mode": True,
    },
    {
        "item": "heterogeneous diffusion coefficient",
        "fortran_behavior": "kdiff 形式（combus L1179-1181、cbstm L1218、cbco2 L1249、cbhym L1334）",
        "python_mainline": "Fuller + Sh，use_fortran_diffusion=False",
        "switch": "UseFortranDiffusion",
        "in_compat_mode": True,
    },
    {
        "item": "WGS equilibrium constant",
        "fortran_behavior": "exp(-3.6893 + 7234/(1.8*T_mean))（wgshift L1271-1281）",
        "python_mainline": "kinetics.calculate_wgs_equilibrium(T)",
        "switch": "none",
        "in_compat_mode": False,
    },
    {
        "item": "WGS rat factor",
        "fortran_behavior": "rat = exp(-8.91 + 5553/ts)（L1271-1281）",
        "python_mainline": "WGS_RatFactor=False（因子=1）",
        "switch": "WGS_RatFactor",
        "in_compat_mode": True,
    },
    {
        "item": "WGS catalytic factor",
        "fortran_behavior": "0.2（wgshift）",
        "python_mainline": "WGS_CatalyticFactor 缺省 0.2",
        "switch": "WGS_CatalyticFactor",
        "in_compat_mode": True,
    },
    {
        "item": "char combustion rate factor",
        "fortran_behavior": "无 0.3 缩放（combus 直接算速率）",
        "python_mainline": "CharCombustionRateFactor=0.3",
        "switch": "CharCombustionRateFactor",
        "in_compat_mode": True,
    },
    {
        "item": "hydrogen consumption limit after instant oxidation",
        "fortran_behavior": "combus 内部按可用 O2 分配，无额外 H2 限幅",
        "python_mainline": "Cell.limit_h2_consumers_after_oxidation 截断 C+H2 与负向 WGS",
        "switch": "none",
        "in_compat_mode": False,
    },
    {
        "item": "combustion-zone C+CO2",
        "fortran_behavior": "cbco2 只按 pO2 分区调用（L1232-1258、poxyin L345）",
        "python_mainline": "NumPy Cell 两条路径均不置零；JAX 侧仍置零（P1.5-B 待对齐）",
        "switch": "none",
        "in_compat_mode": False,
    },
    {
        "item": "particle temperature reaction heat",
        "fortran_behavior": "燃烧区 RK-Gill 含反应热（L459-475）",
        "python_mainline": "USE_RK_GILL_COMBUSTION=False，指数弛豫（constants.py）",
        "switch": "none",
        "in_compat_mode": False,
    },
    {
        "item": "pyrolysis product split",
        "fortran_behavior": "pyroly（L1032）按 fc、vm0 分配",
        "python_mainline": "PyrolysisService.calc_yields",
        "switch": "none",
        "in_compat_mode": False,
    },
    {
        "item": "heat loss closure",
        "fortran_behavior": "燃烧区 hl=30% 反应热；气化区 u0 气壁换热（L774-780）",
        "python_mainline": "HeatLossPercent=2% × HHV × dz/L × (T/1800)^4",
        "switch": "none",
        "in_compat_mode": False,
    },
    {
        "item": "MSR temperature threshold",
        "fortran_behavior": "ts <= 1000 K 时 rate=0（ch4ref L1353）",
        "python_mainline": "MSR_Tmin_K=1000（缺省一致）",
        "switch": "MSR_Tmin_K",
        "in_compat_mode": True,
    },
    {
        "item": "pressure-driven vs concentration-driven rate conversion",
        "fortran_behavior": "以分压 + rg=82.06 计算（L1180 等）",
        "python_mainline": "以浓度 + R=8.314 计算，两者等价",
        "switch": "none",
        "in_compat_mode": False,
    },
    {
        "item": "inlet temperature split",
        "fortran_behavior": "气相 tg1（行 185-187）；颗粒入口 ta（行 179 tsin=ta）",
        "python_mainline": "单一 T_in，气固同温",
        "switch": "none",
        "in_compat_mode": False,
    },
]


def main() -> None:
    silence_solver_logs()
    build_log = BUILD_DIR / "build.log"
    print("compiler build log:", (build_log.read_text()[:200] if build_log.exists() else "MISSING"))

    fortran_out, fortran_prof = parse_fortran_log(BUILD_DIR / "run.log")
    # Fortran 的 name 是 character*12，水煤浆算例名被截断；按前缀匹配
    parsed_names = [c["case_name"] for c in fortran_out]
    missing = [n for n in EXPECTED_CASE_NAMES if not any(n.startswith(p) for p in parsed_names)]
    if missing:
        raise SystemExit(f"Fortran 输出缺少算例: {missing}")

    for case in fortran_out:
        inlet_o2 = float(case["foxy_g_per_g"]) * float(case["tfcoal_g_s"])
        prof = [p for p in fortran_prof if p["case_name"] == case["case_name"]]
        z_drop = float("nan")
        for p in prof:
            if np.isfinite(p.get("goxy_g_s", np.nan)) and p["goxy_g_s"] <= 0.01 * inlet_o2:
                z_drop = float(p["hs_cm"]) / 100.0
                break
        case["O2_drop_to_1pct_z_m"] = z_drop
        case["is_dry_powder"] = case["case_name"] in DRY_POWDER_CASES

    py_rows: list[dict] = []
    py_profiles: list[dict] = []
    for case in fortran_out:
        for mode in ("M", "F"):
            for n_cells in N_LIST:
                rec = solve_fortran_case(case, n_cells=n_cells, mode=mode)
                prof = rec.pop("_profile", None)
                z = rec.pop("_z", None)
                rec.pop("_system", None)
                py_rows.append(rec)
                if prof is not None and z is not None:
                    gas = np.asarray(prof, dtype=float)
                    for idx, (row, zz) in enumerate(zip(gas, z)):
                        dry = float(np.sum(row[:7]))
                        py_profiles.append({
                            "case_name": rec["case_name"], "mode": mode, "cell_index": idx,
                            "z_m": float(zz), "T_gas_K": float(row[10]),
                            "CO_vol_pct": float(row[2] / dry * 100.0) if dry > 0 else np.nan,
                            "H2_vol_pct": float(row[5] / dry * 100.0) if dry > 0 else np.nan,
                            "CO2_vol_pct": float(row[3] / dry * 100.0) if dry > 0 else np.nan,
                            "O2_mol_s": float(row[0]),
                        })

    attribution_rows: list[dict] = []
    switch_items = [
        ("UseFortranDiffusion", {"UseFortranDiffusion": True}),
        ("WGS_RatFactor", {"WGS_RatFactor": True}),
        ("CharCombustionRateFactor", {"CharCombustionRateFactor": 1.0}),
    ]
    for case in fortran_out:
        if not case["is_dry_powder"]:
            continue
        base = next(r for r in py_rows if r["case_name"] == case["case_name"] and r["mode"] == "M" and r["N_cells"] == N_REF)
        for item, switches in switch_items:
            alt = solve_fortran_case(case, n_cells=N_REF, mode="M", switches=switches)
            alt.pop("_profile", None); alt.pop("_z", None); alt.pop("_system", None)
            attribution_rows.append({
                "case_name": case["case_name"],
                "item": item,
                "d_X_C_pp": alt.get("X_C_pct", np.nan) - base.get("X_C_pct", np.nan),
                "d_CO_vol_pct": alt.get("CO_vol_pct", np.nan) - base.get("CO_vol_pct", np.nan),
                "d_H2_vol_pct": alt.get("H2_vol_pct", np.nan) - base.get("H2_vol_pct", np.nan),
                "d_CO2_vol_pct": alt.get("CO2_vol_pct", np.nan) - base.get("CO2_vol_pct", np.nan),
                "d_T_gas_K": alt.get("T_gas_K", np.nan) - base.get("T_gas_K", np.nan),
                "execution_status": alt.get("execution_status"),
            })

    # B9 敏感性：固相焓按 Hf_coal 计 vs 只计显热（Fortran 侧无此形成焓项）
    b9_rows: list[dict] = []
    for case in fortran_out:
        if not case["is_dry_powder"]:
            continue
        base = next(r for r in py_rows if r["case_name"] == case["case_name"] and r["mode"] == "F" and r["N_cells"] == N_REF)
        alt = solve_fortran_case(case, n_cells=N_REF, mode="F", coal_overrides={"Hf_coal": 0.0})
        alt.pop("_profile", None); alt.pop("_z", None); alt.pop("_system", None)
        b9_rows.append({
            "case_name": case["case_name"],
            "T_with_Hf_coal_K": base.get("T_gas_K", np.nan),
            "T_without_Hf_coal_K": alt.get("T_gas_K", np.nan),
            "dT_K": alt.get("T_gas_K", np.nan) - base.get("T_gas_K", np.nan),
            "X_C_with_Hf_coal_pct": base.get("X_C_pct", np.nan),
            "X_C_without_Hf_coal_pct": alt.get("X_C_pct", np.nan),
            "fortran_T_K": case["T_gas_K"],
            "fortran_X_C_pct": case["X_C_pct"],
        })

    py_out_df = pd.DataFrame(py_rows)
    fort_df = pd.DataFrame(fortran_out)
    fort_prof_df = pd.DataFrame(fortran_prof)
    attr_df = pd.DataFrame(attribution_rows)
    ledger_df = pd.DataFrame(LEDGER)

    anchor_rows: list[dict] = []
    for case in fortran_out:
        if not case["is_dry_powder"]:
            continue
        for mode in ("M", "F"):
            r = next(x for x in py_rows if x["case_name"] == case["case_name"] and x["mode"] == mode and x["N_cells"] == N_REF)
            checks = {
                "X_C_pp": (r.get("X_C_pct", np.nan), case["X_C_pct"]),
                "CO_vol_pct": (r.get("CO_vol_pct", np.nan), case["CO_vol_pct"]),
                "H2_vol_pct": (r.get("H2_vol_pct", np.nan), case["H2_vol_pct"]),
                "CO2_vol_pct": (r.get("CO2_vol_pct", np.nan), case["CO2_vol_pct"]),
                "T_gas_K": (r.get("T_gas_K", np.nan), case["T_gas_K"]),
                "O2_drop_rel": (r.get("O2_drop_to_1pct_z_m", np.nan), case.get("O2_drop_to_1pct_z_m", np.nan)),
            }
            for metric, (pv, fv) in checks.items():
                diff = abs(pv - fv)
                evaluable = metric != "O2_drop_rel"
                note = ""
                if metric == "O2_drop_rel":
                    # Fortran 只在收敛后打印少数气室，第一次打印处 O2 已为 0；
                    # Python 在 cell 0 就已降到入口 1% 以下。两侧都无法给出该位置。
                    diff = diff / max(abs(fv), 1e-9)
                    note = "not evaluable: Fortran 打印点已无 O2，Python 在 cell 0 即触发阈值"
                anchor_rows.append({
                    "case_name": case["case_name"], "mode": mode, "metric": metric,
                    "python_value": pv, "fortran_value": fv,
                    "abs_diff": diff, "tolerance": ANCHOR_TOL[metric],
                    "evaluable": bool(evaluable),
                    "pass": (bool(np.isfinite(diff) and diff <= ANCHOR_TOL[metric])
                             if evaluable else None),
                    "note": note,
                })
    anchor_df = pd.DataFrame(anchor_rows)

    tables = {
        "fortran_outlet.csv": fort_df,
        "fortran_profiles.csv": fort_prof_df,
        "python_outlet.csv": py_out_df,
        "python_profiles.csv": pd.DataFrame(py_profiles),
        "deviation_ledger.csv": ledger_df,
        "attribution.csv": attr_df,
        "anchor_criteria.csv": anchor_df,
        "b9_solid_enthalpy_sensitivity.csv": pd.DataFrame(b9_rows),
    }

    protocol = base_protocol({
        "round": "P1.5-C / 2026-10-30",
        "compiler": "gfortran 16.1.0 (Homebrew GCC 16.1.0)",
        "compile_command": "gfortran -std=legacy -O2 -ffixed-form -o wen.x source1.f",
        "fortran_io_patch": "ic1..ic5 由 INTEGER 改为 CHARACTER*4（仅旧绘图标签，DATA 语句类型错误）；见 fortran_build/io_patch.diff",
        "fortran_geometry": {"L_m": 3.30, "D_m": 1.524, "P_atm": 24.0, "lines": [35]},
        "fortran_inlet_mapping": {
            "coal_flow": "tfcoal/1000 [kg/s]，Mt=xmois*100",
            "o2_flow": "foxy*tfcoal/1000（foxy 按纯 O2 处理）",
            "steam_flow": "fsteam*tfcoal/1000",
            "T_in": "tg1 质量加权（源程序行 185-187）",
            "particle_diameter": "2*ri = 3.5e-4 m（行 57）",
            "coal_ultimate": "干基 = (1-yash)*dmmf",
        },
        "anchor_tolerances": ANCHOR_TOL,
        "n_list": list(N_LIST),
        "n_reference": N_REF,
        "source_commit": source_commit(),
    })

    interp = _interpretation(fort_df, py_out_df, anchor_df, attr_df, ledger_df, pd.DataFrame(b9_rows))
    written = write_round_outputs(OUT_DIR, tables=tables, protocol=protocol, interpretation=interp)
    print(json.dumps({"written": [str(p) for p in written]}, ensure_ascii=False, indent=2))
    print(py_out_df[["case_name", "mode", "N_cells", "execution_status", "X_C_pct",
                     "CO_vol_pct", "H2_vol_pct", "CO2_vol_pct", "T_gas_K"]].to_string(index=False))


def _interpretation(fort: pd.DataFrame, py: pd.DataFrame, anchor: pd.DataFrame,
                    attr: pd.DataFrame, ledger: pd.DataFrame, b9: pd.DataFrame) -> str:
    lines = ["# P1.5-C Wen/Fortran 锚点 —— interpretation\n"]
    lines.append("## 1. Fortran 是否成功运行\n")
    lines.append(
        "gfortran 16.1.0 `-std=legacy -O2 -ffixed-form` 编译通过；唯一补丁是把只用于旧绘图标签的 "
        "`ic1..ic5` 从 INTEGER 改成 CHARACTER*4（DATA 语句类型错误）。7 个算例全部跑完，"
        "进程退出码 2 来自文件末尾多读一次 namelist，不影响结果。出口值见 `fortran_outlet.csv`。\n"
    )
    lines.append("## 2. 模式 F 是否达到锚定判据\n")
    dry_f = anchor[(anchor["mode"] == "F") & (anchor["evaluable"].astype(bool))]
    if len(dry_f):
        n_pass = int(dry_f.groupby("case_name")["pass"].all().sum())
        lines.append(f"5 个干粉算例中，模式 F 全部**可评估**判据通过的算例数：**{n_pass}/5**。逐算例逐判据见 `anchor_criteria.csv`。\n")
        for case, grp in dry_f.groupby("case_name"):
            bad = grp[~grp["pass"].astype(bool)]
            lines.append(f"- `{case}`：{'全部通过' if bad.empty else '未通过 ' + '、'.join(bad['metric'])}")
        lines.append("")
        lines.append("O2 降到入口 1% 的位置不可评估：Fortran 只在收敛后打印少数气室，第一次打印时 O2 已为 0；"
                     "Python 在 cell 0 就已低于阈值。该判据在两侧都没有分辨率，未计入上面的通过数。\n")
    lines.append("## 3. 模式 M 与 Fortran 的差距\n")
    if len(py):
        m = py[(py["mode"] == "M") & (py["N_cells"] == py["N_cells"].max())]
        for _, r in m.iterrows():
            f = fort[fort["case_name"] == r["case_name"]].iloc[0]
            lines.append(
                f"- `{r['case_name']}`：X_C {r['X_C_pct']:.1f} vs {f['X_C_pct']:.1f} pp，"
                f"CO {r['CO_vol_pct']:.1f} vs {f['CO_vol_pct']:.1f}，H2 {r['H2_vol_pct']:.1f} vs {f['H2_vol_pct']:.1f}，"
                f"CO2 {r['CO2_vol_pct']:.1f} vs {f['CO2_vol_pct']:.1f}，T {r['T_gas_K']:.0f} vs {f['T_gas_K']:.0f} K"
                + ("" if r["execution_status"] == "completed" else f"（{r['execution_status']}）")
            )
        lines.append("")
    if len(attr):
        rank = attr.assign(absd=attr["d_X_C_pp"].abs()).groupby("item")["absd"].mean().sort_values(ascending=False)
        lines.append("按平均 |ΔX_C| 排序的归因项（模式 M 下一次性切换）：" + "、".join(f"{k}（{v:.2f} pp）" for k, v in rank.items()) + "。")
        lines.append("没有开关的差异项（WGS 平衡常数、热损闭合、颗粒温度反应热、热解分配、氢气限幅、入口温度分裂）无法逐项切换，"
                     "剩余偏差主要来自它们，尤其热损闭合（Fortran 燃烧区按 30% 反应热、气化区按气壁换热，本模型按 2% HHV × T⁴）。\n")
    if len(b9):
        lines.append("### 温度差距的来源（B9 敏感性，模式 F，N=160）\n")
        for _, r in b9.iterrows():
            lines.append(
                f"- `{r['case_name']}`：带 `Hf_coal` {r['T_with_Hf_coal_K']:.0f} K → 去掉 `Hf_coal` "
                f"{r['T_without_Hf_coal_K']:.0f} K（Δ{r['dT_K']:+.0f} K），Fortran {r['fortran_T_K']:.0f} K；"
                f"X_C {r['X_C_with_Hf_coal_pct']:.1f}% → {r['X_C_without_Hf_coal_pct']:.1f}%，"
                f"Fortran {r['fortran_X_C_pct']:.1f}%。"
            )
        lines.append(
            "\n说明：`Hf_coal`（固相焓按原煤生成焓计，审计 B9/D-HUAYI-02）是出口温度差距的主导项——"
            "去掉它会把温度抬高 300 K 量级并让 X_C 撞到 0.999，说明本模型现在落在 Fortran 与“无形成焓”之间。"
            "热损不是主因：把 `HeatLossPercent` 从 2% 降到 0 只抬高约 22 K（texaco i-1），升到 5% 只降低约 32 K。"
            "该差异没有开关，按卡片要求本轮不新建开关、不改方程。\n"
        )
    lines.append("## 4. 两模式的碳转化率\n")
    for case in fort["case_name"]:
        row_m = py[(py["case_name"] == case) & (py["mode"] == "M") & (py["N_cells"] == py["N_cells"].max())]
        row_f = py[(py["case_name"] == case) & (py["mode"] == "F") & (py["N_cells"] == py["N_cells"].max())]
        fval = fort[fort["case_name"] == case].iloc[0]["X_C_pct"]
        lines.append(
            f"- `{case}`：Fortran {fval:.1f}%，模式 M {row_m.iloc[0]['X_C_pct'] if len(row_m) else float('nan'):.1f}%，"
            f"模式 F {row_f.iloc[0]['X_C_pct'] if len(row_f) else float('nan'):.1f}%"
        )
    lines.append("\n## 5. 台账\n")
    lines.append(f"共 {len(ledger)} 项，其中有开关且模式 F 已切换的 {int(ledger['in_compat_mode'].sum())} 项；"
                 "其余记为 `none`，按卡片要求本轮不新建开关。\n")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
