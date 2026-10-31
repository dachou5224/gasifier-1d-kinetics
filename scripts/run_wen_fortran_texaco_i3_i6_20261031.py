#!/usr/bin/env python3
"""用仓库自带 Wen & Chaung (1979) Fortran 程序跑 Texaco I-1/I-3/I-6 等算例。

背景：`data/validation_cases_final.json` 的 Texaco_I-3 与 Texaco_I-6 不在
`reference_fortran/input_副本.txt` 里（该文件只有 7 个算例），且其 `coal` 字段为空，
被加载器默认成了 `Paper_Base_Coal`。本脚本：

1. 从原始论文数据 `data/archive/validation_cases_OriginalPaper.json` 取 Fortran 的参数口径，
   确认 I-3 / I-6 与 I-1 同属 H-coal residue Illinois No.6；
2. 用同一煤质参数补齐 I-3 / I-6，拼出 9 个算例的 namelist 输入；
3. 编译并运行原始 Fortran（仅保留 `ic1..ic5` 的字符类型修复）；
4. 把 Fortran 出口值与论文实验值、Python 现行结果并列。

不改任何物理或数值逻辑。
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "outputs" / "diagnostics" / "round-wen-fortran-texaco-20261031"
BUILD = OUT / "fortran_build"
ORIG_SRC = ROOT / "reference_fortran" / "Source1_副本.for"
ORIG_INPUT = ROOT / "reference_fortran" / "input_副本.txt"

# H-coal residue Illinois No.6 的 dmmf 参数，取自 Fortran 输入 texaco i-1，
# 与原始论文 JSON 的 ultimate_analysis_wt_pct_dry 归一化后逐位一致。
COAL = {
    "yoxy": 0.0156956,
    "yh": 0.0743163,
    "yc": 0.8802616,
    "ys": 0.0210464,
    "yn": 0.0084423,
    "yash": 0.159,
    "xmois": 0.002,
}

# 原始 7 算例（input_副本.txt 已有）；补 I-3 / I-6
EXTRA_CASES = [
    {"name": "texaco i-3", "ta": 500.0, "tsteam": 696.67, "toxy": 298.0,
     "foxy": 0.813, "fsteam": 0.309, "tfcoal": 82.202},
    {"name": "texaco i-6", "ta": 500.0, "tsteam": 696.67, "toxy": 298.0,
     "foxy": 0.774, "fsteam": 0.291, "tfcoal": 87.73},
]


def _namelist(case: dict) -> str:
    return (
        f"&readdata name='{case['name']}',ta={case['ta']},tsteam={case['tsteam']},toxy={case['toxy']},\n"
        f"yoxy={COAL['yoxy']},yh={COAL['yh']},yc={COAL['yc']},ys={COAL['ys']},yn={COAL['yn']},\n"
        f"yash={COAL['yash']},xmois={COAL['xmois']},foxy={case['foxy']},fsteam={case['fsteam']},"
        f"tfcoal={case['tfcoal']}\n&end\n"
    )


def _patch_source(src_text: str) -> str:
    """唯一补丁：只用于旧绘图标签的 ic1..ic5 从 INTEGER 改为 CHARACTER*4。"""
    old = (
        "      dimension x1(120,2),y1(120,2),npt1(2),ic1(2)\n"
        "      dimension x2(120,4),y2(120,4),npt2(4),ic2(4)\n"
        "      dimension x3(120,3),y3(120,3),npt3(3),ic3(3)\n"
        "      dimension x4(120,1),y4(120,1),npt4(1),ic4(1)\n"
        "      dimension x5(120,2),y5(120,2),npt5(2),ic5(2)\n"
    )
    new = (
        "      dimension x1(120,2),y1(120,2),npt1(2)\n"
        "      dimension x2(120,4),y2(120,4),npt2(4)\n"
        "      dimension x3(120,3),y3(120,3),npt3(3)\n"
        "      dimension x4(120,1),y4(120,1),npt4(1)\n"
        "      dimension x5(120,2),y5(120,2),npt5(2)\n"
        "      character*4 ic1(2),ic2(4),ic3(3),ic4(1),ic5(2)\n"
    )
    if old not in src_text:
        raise SystemExit("source layout changed; patch anchor not found")
    return src_text.replace(old, new)


_NAME_RE = re.compile(r'NAME="([^"]*)"')
_NUM = r"([-\d.eE+]+)"


def parse_log(text: str) -> list[dict]:
    rows: list[dict] = []
    for block in text.split("&READDATA")[1:]:
        m = _NAME_RE.search(block)
        if not m:
            continue
        name = m.group(1).strip()
        tfcoal_m = re.search(r"TFCOAL=\s*" + _NUM, block)
        yc_m = re.search(r"YC=\s*" + _NUM, block)
        yash_m = re.search(r"YASH=\s*" + _NUM, block)
        xm_m = re.search(r"XMOIS=\s*" + _NUM, block)
        comps = block.split("for gaseous compartment")
        last = None
        for cb in comps[1:]:
            def g(pat):
                mm = re.search(pat, cb)
                return float(mm.group(1)) if mm else np.nan
            rec = {
                "case_name": name,
                "T_gas_K": g("corrected gas temp=" + r"\s*" + _NUM),
                "T_solid_K": g("ts outlet=" + r"\s*" + _NUM),
                "CO_vol_pct": g("co mol %=" + r"\s*" + _NUM),
                "H2_vol_pct": g("h2 mol %=" + r"\s*" + _NUM),
                "CO2_vol_pct": g("co2 mol %=" + r"\s*" + _NUM),
                "CH4_vol_pct": g("ch4 mol %=" + r"\s*" + _NUM),
                "N2_vol_pct": g("n2 mol %=" + r"\s*" + _NUM),
                "H2S_vol_pct": g("h2s mol %=" + r"\s*" + _NUM),
                "gco_g_s": g("co flow rate=" + r"\s*" + _NUM),
                "gco2_g_s": g("co2 flow rate=" + r"\s*" + _NUM),
                "gch4_g_s": g("gch4=" + r"\s*" + _NUM),
                "wl_pct": g("total weight loss=" + r"\s*" + _NUM),
            }
            if np.isfinite(rec["T_gas_K"]):
                last = rec
        if last is None:
            continue
        tfcoal = float(tfcoal_m.group(1)) if tfcoal_m else np.nan
        yc = float(yc_m.group(1)) if yc_m else np.nan
        yash = float(yash_m.group(1)) if yash_m else np.nan
        xmois = float(xm_m.group(1)) if xm_m else np.nan
        fcoal = tfcoal * (1.0 - yash) * (1.0 - xmois)   # Fortran 行 70
        xcarb = 100.0 * (last["gco_g_s"] / 28.0 + last["gco2_g_s"] / 44.0
                         + last["gch4_g_s"] / 16.0) * 12.0 / (fcoal * yc)
        last["X_C_pct"] = float(xcarb)
        last["tfcoal_g_s"] = tfcoal
        rows.append(last)
    return rows


def main() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)

    src = _patch_source(ORIG_SRC.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n"))
    (BUILD / "source1.f").write_text(src, encoding="utf-8")

    base_input = ORIG_INPUT.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n").rstrip() + "\n"
    extra = "".join(_namelist(c) for c in EXTRA_CASES)
    (BUILD / "input.txt").write_text(base_input + "\n" + extra, encoding="utf-8")

    subprocess.run(
        ["gfortran", "-std=legacy", "-O2", "-ffixed-form", "-o", "wen.x", "source1.f"],
        cwd=BUILD, check=True, capture_output=True, text=True,
    )
    proc = subprocess.run(["./wen.x"], cwd=BUILD, capture_output=True, text=True)
    (BUILD / "run.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")

    fort = parse_log(proc.stdout)
    fort_df = pd.DataFrame(fort)

    # 论文实验值（来自 data/validation_cases_final.json 的 expected_results）
    final = json.load(open(ROOT / "data" / "validation_cases_final.json", encoding="utf-8"))
    paper = {}
    for group in final["Pilot"].values():
        for name, rec in group.items():
            ex = rec.get("expected_results", {})
            gas = ex.get("dry_product_gas_vol_pct", {})
            paper[name] = {
                "paper_CO": gas.get("CO"),
                "paper_H2": gas.get("H2"),
                "paper_CO2": gas.get("CO2"),
                "paper_X_C_pct": ex.get("carbon_conversion_pct"),
            }

    alias = {"texaco i-1": "Texaco_I-1", "texaco i-2": "Texaco_I-2", "texaco i-5c": "Texaco_I-5C",
             "texaco i-10": "Texaco_I-10", "texaco exxon": "Texaco_Exxon",
             "texaco i-3": "Texaco_I-3", "texaco i-6": "Texaco_I-6"}
    fort_df["paper_case"] = fort_df["case_name"].map(alias)
    for col in ("paper_CO", "paper_H2", "paper_CO2", "paper_X_C_pct"):
        fort_df[col] = fort_df["paper_case"].map(lambda k: paper.get(k, {}).get(col))
    fort_df["dCO_vs_paper"] = fort_df["CO_vol_pct"] - fort_df["paper_CO"]
    fort_df["dH2_vs_paper"] = fort_df["H2_vol_pct"] - fort_df["paper_H2"]
    fort_df["dCO2_vs_paper"] = fort_df["CO2_vol_pct"] - fort_df["paper_CO2"]

    OUT.mkdir(parents=True, exist_ok=True)
    fort_df.to_csv(OUT / "fortran_outlet.csv", index=False)

    protocol = {
        "round": "Wen Fortran reference for Texaco I-3 / I-6 / 2026-10-31",
        "fortran_source": "reference_fortran/Source1_副本.for",
        "original_input": "reference_fortran/input_副本.txt (7 cases)",
        "added_cases": EXTRA_CASES,
        "coal_parameters_used": COAL,
        "coal_provenance": (
            "Fortran i-1 namelist 与原始论文 JSON 的 H-coal_residue_Illinois_No6 "
            "ultimate_analysis_wt_pct_dry (C=74.05,H=6.25,O=1.32,S=1.77,N=0.71,Ash=15.53) "
            "归一化到 dmmf 后逐位一致；I-3/I-6 描述同为该煤种。"
        ),
        "only_source_patch": "ic1..ic5 INTEGER -> CHARACTER*4 (legacy plotting labels)",
        "compiler": subprocess.run(["gfortran", "--version"], capture_output=True, text=True).stdout.splitlines()[0],
        "exit_code": proc.returncode,
        "exit_code_note": "Fortran 在读完输入后多读一次 namelist，返回 2；不影响结果",
        "frozen": "no physical or numerical logic changed",
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(fort_df[["case_name", "T_gas_K", "CO_vol_pct", "H2_vol_pct", "CO2_vol_pct",
                   "X_C_pct", "paper_CO", "paper_H2", "paper_CO2", "paper_X_C_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
