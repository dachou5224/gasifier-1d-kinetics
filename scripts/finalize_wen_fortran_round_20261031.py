#!/usr/bin/env python3
"""汇总 Wen Fortran 对照轮的 protocol / interpretation / sha256。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "outputs" / "diagnostics" / "round-wen-fortran-texaco-20261031"


def main() -> None:
    fort = pd.read_csv(OUT / "fortran_outlet.csv")
    pyf = pd.read_csv(OUT / "python_vs_fortran.csv")

    fest = fort[["case_name", "T_gas_K", "CO_vol_pct", "H2_vol_pct", "CO2_vol_pct",
                 "X_C_pct", "paper_CO", "paper_H2", "paper_CO2", "paper_X_C_pct"]].copy()
    fest["dCO_vs_paper"] = fest["CO_vol_pct"] - fest["paper_CO"]
    fest["dX_C_vs_paper"] = fest["X_C_pct"] - fest["paper_X_C_pct"]
    fest.to_csv(OUT / "fortran_vs_paper.csv", index=False)

    protocol = {
        "round": "Wen Fortran reference for Texaco I-1 / I-3 / I-6 / 2026-10-31",
        "trigger": "test_validation_config Texaco_I-3 and Texaco_I-6 exit at 3400 K / 3479 K",
        "fortran_source": "reference_fortran/Source1_副本.for",
        "fortran_original_input": "reference_fortran/input_副本.txt (7 cases only)",
        "cases_added_to_input": ["texaco i-3", "texaco i-6"],
        "added_case_basis": (
            "operating conditions from data/validation_cases_final.json; coal from the Fortran "
            "i-1 namelist, which matches H-coal_residue_Illinois_No6 in "
            "data/archive/validation_cases_OriginalPaper.json after dmmf normalisation"
        ),
        "coal_parameters_used": {
            "yoxy": 0.0156956, "yh": 0.0743163, "yc": 0.8802616,
            "ys": 0.0210464, "yn": 0.0084423, "yash": 0.159, "xmois": 0.002,
        },
        "coal_consistency_check": (
            "C=74.05/84.10=0.8805 vs yc=0.88026; O=1.32/84.10=0.0157 vs yoxy=0.015696; "
            "H=6.25/84.10=0.0743 vs yh=0.074316; S=1.77/84.10=0.02105 vs ys=0.021046; "
            "N=0.71/84.10=0.00844 vs yn=0.008442"
        ),
        "only_source_patch": "ic1..ic5 INTEGER -> CHARACTER*4 (legacy plotting labels only)",
        "compiler": subprocess.run(["gfortran", "--version"], capture_output=True, text=True).stdout.splitlines()[0],
        "fortran_geometry": {"height_cm": 330, "dt_cm": 152.4, "P_atm": 24.0, "source_line": 32},
        "paper_geometry": {"length_m": 6.096, "diameter_m": 1.524, "source": "paper metadata 20 ft x 5 ft"},
        "json_geometry": {"length_m": 6.0, "diameter_m": 2.0, "source": "data/validation_cases_final.json"},
        "json_coal_problem": {
            "coal_field": "empty object {}",
            "fallback": "validation_loader.normalize_final_case_to_legacy -> CASE_TO_COAL_MAP -> Paper_Base_Coal",
            "paper_base_coal": {"Cd": 77.56, "Hd": 4.67, "Od": 9.44, "Sd": 0.4, "Nd": 0.82, "Ad": 7.11, "Mt": 4.53},
            "wen_coal": {"C": 74.05, "H": 6.25, "O": 1.32, "S": 1.77, "N": 0.71, "Ash": 15.53},
        },
        "frozen": "no equation, tolerance or parameter changed; the only edit is the Fortran compile patch",
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    f = fest.set_index("case_name")
    L: list[str] = []
    L.append("# Wen Fortran 参考值对照（Texaco I-1 / I-3 / I-6，2026-10-31）\n")
    L.append("## 1. 用仓库自带 Fortran 跑出的参考值\n")
    L.append("`reference_fortran/input_副本.txt` 只有 7 个算例，不含 I-3 与 I-6。"
             "用 I-1 的煤质参数（经与原始论文 JSON 的 H-coal residue Illinois No.6 核对一致）"
             "补齐两个 namelist 条目后编译运行，全部 9 个算例跑完。\n")
    L.append("| 算例 | T_gas (K) | CO | H2 | CO2 | X_C (%) | 论文实验 CO | 论文实验 X_C |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name in ("texaco i-1", "texaco i-3", "texaco i-6"):
        if name not in f.index:
            continue
        r = f.loc[name]
        L.append(f"| {name} | {r['T_gas_K']:.1f} | {r['CO_vol_pct']:.2f} | {r['H2_vol_pct']:.2f} | "
                 f"{r['CO2_vol_pct']:.2f} | {r['X_C_pct']:.2f} | {r['paper_CO']:.2f} | {r['paper_X_C_pct']:.2f} |")
    L.append("")
    L.append("**Texaco_I-3 = 1350 K，Texaco_I-6 = 1345 K。3400 K 不可能是物理结果。**\n")
    L.append("## 2. Fortran 本身也不等于论文实验值\n")
    for name in ("texaco i-1", "texaco i-3", "texaco i-6"):
        if name not in f.index:
            continue
        r = f.loc[name]
        L.append(f"- `{name}`：碳转化率 {r['X_C_pct']:.1f}% 对实验 {r['paper_X_C_pct']:.1f}%"
                 f"（{r['dX_C_vs_paper']:+.1f} pp），CO {r['CO_vol_pct']:.2f} 对 {r['paper_CO']:.2f}"
                 f"（{r['dCO_vs_paper']:+.2f} pp）。")
    L.append("\n即 `validation_cases_final.json` 的 `expected_results` 是**实验值**，不是 Fortran 输出。"
             "任何「复现 Fortran」的实现都不可能同时满足这些 expected。\n")
    L.append("## 3. 3400 K 的来源：不是 H2 限幅，是输入数据错\n")
    L.append("`data/validation_cases_final.json` 里这些算例的 `coal` 是空对象 `{}`，"
             "`CASE_TO_COAL_MAP` 把它们全部回落到 `Paper_Base_Coal`：\n")
    L.append("| | C | H | O | S | N | Ash |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    L.append("| `Paper_Base_Coal`（JSON 实际用） | 77.56 | 4.67 | **9.44** | 0.40 | 0.82 | **7.11** |")
    L.append("| H-coal residue Illinois No.6（Wen 原文） | 74.05 | 6.25 | **1.32** | 1.77 | 0.71 | **15.53** |")
    L.append("\n氧含量差 7 倍、灰分差一半，这是两个不同的燃料。"
             "反应器几何也对不上：Fortran 自身是 3.3 m × 1.524 m，论文写 20 ft × 5 ft（6.096 m × 1.524 m），"
             "而 validation JSON 用 6.0 m × **2.0 m**。\n")
    sub = pyf[pyf["setting"] == "F_wen_coal_fortran_geom"]
    L.append("## 4. 改正输入后，现行限幅是收敛的必要条件\n")
    L.append("正确煤质 + Fortran 自身几何（3.3 m × 1.524 m）+ 模式 F 开关：\n")
    L.append("| 限幅 | 算例 | T (K) | ΔT vs Fortran | CO | max cost | 收敛 |")
    L.append("|---|---|---:|---:|---:|---:|---|")
    for cap in ("capped", "uncapped"):
        for name in ("Texaco_I-1", "Texaco_I-3", "Texaco_I-6"):
            r = sub[(sub["case_name"] == name) & (sub["wgs_cap"] == cap)].iloc[0]
            L.append(f"| {cap} | {name} | {r['T_K']:.1f} | {r['dT_vs_fortran_K']:+.1f} | "
                     f"{r['CO_vol_pct']:.2f} | {r['max_residual_cost']:.1e} | "
                     f"{'✓' if r['converged_1e-4'] else '✗'} |")
    L.append("")
    L.append("保留负向 WGS 截断时三个算例全部收敛（cost ≤ 1e-6），温度与 Fortran 相差 −46 ~ +120 K；"
             "去掉截断则三个算例全部发散到 1.77–1.88 kK。"
             "**所以这个截断不是缺陷，是正确的实现。**\n")
    L.append("## 5. 对前一份审计的更正\n")
    L.append("`round-wgs-cap-audit-20261031/interpretation.md` 第 2 节的结论——"
             "「没有单一子改动能让三个算例收敛，系统处于高度非光滑区域」——**作废**。"
             "那份审计是在错误的煤（`Paper_Base_Coal`）上做的变体对比，煤质错误本身足以解释全部发散。"
             "第 3 节（P2-A 依赖该截断）仍然成立：那一轮用的是正确的华谊煤。\n")
    L.append("## 6. 结论与待办\n")
    L.append("1. 三个算例的 3400 K 是输入数据错误导致的未收敛，不是模型缺陷，也不是 `H2` 限幅的问题。")
    L.append("2. 修复点应落在数据侧：给 `validation_cases_final.json` 的 Texaco 条目填 `coal_type`，"
             "并把几何改成 Fortran/论文口径；不是改方程。")
    L.append("3. `expected_results` 是实验值，校验测试若以它们为判据，需要同时接受「Fortran 也差 6–16 pp」这一事实。")
    L.append("4. 不建议据本轮改动任何物理代码。\n")
    (OUT / "interpretation.md").write_text("\n".join(L), encoding="utf-8")

    from model.revision_evidence_common import write_sha256_manifest

    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "sha256.txt")
    write_sha256_manifest(files, OUT / "sha256.txt")
    print("written:", len(files), "files")


if __name__ == "__main__":
    main()
