"""
对齐 VALIDATION_CASES.expected 的验证脚本（不依赖 pytest）。

对每个 case：
1) 使用 VALIDATION_CASES.inputs 构造 GasifierSystem（尽量与 gasifier_kinetic_ui.py 的口径一致）
2) 分别调用：
   - baseline：solver_method='minimize'（与 UI 默认 solve(...) 一致）
   - jax_path：solver_method='jax_newton'
3) 计算 UI 用的 KPI：
   - T_out_C = T_g - 273.15
   - 干基 yCO/yH2/yCO2：F_dry = sum(gas[:7])（气体顺序 [O2,CH4,CO,CO2,H2S,H2,N2,H2O]）
4) 与 expected 对比并写成 Markdown 表（清晰表格，含 wall time 与 profile 最大误差）。

用法示例：
  cd gasifier-1d-kinetic
  python3 scripts/validate_jax_solver_expected_against_ui.py --N_cells 40 --out docs/expected_jax_newton_report.md
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Dict, Tuple

import numpy as np


def _ensure_import_path() -> None:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src_path = os.path.join(project_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def _compute_kpis(profile: np.ndarray) -> Dict[str, float]:
    """
    profile: (N_cells, 11) -> 最后一行是出口
    返回与 gasifier_kinetic_ui.py 一致的 KPI：
      yCO,yH2,yCO2（干基 mol%）
      T_out_C
    """
    last = profile[-1]
    gas = last[:8].astype(float)
    F_dry = float(np.sum(gas[:7]) + 1e-12)  # dry basis excludes H2O（UI: sum(gas[:7])）

    y_co = float(gas[2] / F_dry * 100.0)
    y_h2 = float(gas[5] / F_dry * 100.0)
    y_co2 = float(gas[3] / F_dry * 100.0)
    t_out_c = float(last[10] - 273.15)
    return {"T_out_C": t_out_c, "yCO": y_co, "yH2": y_h2, "yCO2": y_co2}


def _max_profile_diffs(base: np.ndarray, jax: np.ndarray) -> Dict[str, float]:
    d = jax - base
    return {
        "max|ΔT|": float(np.max(np.abs(d[:, 10]))),
        "max|ΔCO|": float(np.max(np.abs(d[:, 2]))),
        "max|ΔCO2|": float(np.max(np.abs(d[:, 3]))),
        "max|ΔH2|": float(np.max(np.abs(d[:, 5]))),
        "max|ΔCH4|": float(np.max(np.abs(d[:, 1]))),
    }


def _fmt(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def _fmt_e(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}e}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N_cells", type=int, default=40)
    parser.add_argument("--out", type=str, default="docs/expected_jax_newton_report.md")
    parser.add_argument("--log-level", type=str, default="ERROR")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.ERROR))
    _ensure_import_path()

    from model.chemistry import COAL_DATABASE, VALIDATION_CASES
    from model.gasifier_system import GasifierSystem

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    case_names = list(VALIDATION_CASES.keys())
    results: Dict[str, Dict] = {}

    print(f"=== expected 对齐验证：N_cells={args.N_cells}, cases={case_names} ===")

    for idx, case_name in enumerate(case_names):
        case = VALIDATION_CASES[case_name]
        case_inputs = case["inputs"]
        expected = case.get("expected", {})

        print(f"\n[Case {idx+1}/{len(case_names)}] {case_name}")

        coal_key = case_inputs.get("coal") or case_inputs.get("coal_type")
        coal_props = COAL_DATABASE[coal_key].copy()

        # geometry：与 UI 默认一致（若 validation case 没写则取 UI 默认 6m/2m）
        geometry = {
            "L": float(case_inputs.get("L_reactor", case_inputs.get("L_m", 6.0))),
            "D": float(case_inputs.get("D_reactor", case_inputs.get("D_m", 2.0))),
        }

        # UI 里用 FeedRate / 3600 得到 coal_flow
        coal_flow_kg_s = float(case_inputs["FeedRate"]) / 3600.0
        o2_flow_kg_s = coal_flow_kg_s * float(case_inputs.get("Ratio_OC", 1.05))
        steam_flow_kg_s = coal_flow_kg_s * float(case_inputs.get("Ratio_SC", 0.0))

        op_conds = {
            "coal_flow": coal_flow_kg_s,
            "o2_flow": o2_flow_kg_s,
            "steam_flow": steam_flow_kg_s,
            "P": float(case_inputs["P"]),
            "T_in": float(case_inputs["TIN"]),
            "SlurryConcentration": float(case_inputs.get("SlurryConcentration", 60.0)),
            "HeatLossPercent": float(case_inputs.get("HeatLossPercent", 1.0)),
            "AdaptiveFirstCellLength": True,
            "Combustion_CO2_Fraction": float(case_inputs.get("Combustion_CO2_Fraction", 0.15)),
            "WGS_CatalyticFactor": float(case_inputs.get("WGS_CatalyticFactor", 1.5)),
        }

        # baseline：minimize（UI 默认）
        sys_b = GasifierSystem(geometry, coal_props, op_conds)
        t0 = time.perf_counter()
        base_profile, z_b = sys_b.solve(N_cells=args.N_cells, solver_method="minimize")
        t_base = time.perf_counter() - t0
        base_kpis = _compute_kpis(base_profile)

        # jax：jax_newton
        sys_j = GasifierSystem(geometry, coal_props, op_conds)
        t1 = time.perf_counter()
        jax_profile, z_j = sys_j.solve(N_cells=args.N_cells, solver_method="jax_newton", jax_warmup=True)
        t_jax = time.perf_counter() - t1
        jax_kpis = _compute_kpis(jax_profile)

        max_diffs = _max_profile_diffs(base_profile, jax_profile)

        # 与 expected 的差值
        exp_T = expected.get("TOUT_C")
        exp_yCO = expected.get("YCO")
        exp_yH2 = expected.get("YH2")
        exp_yCO2 = expected.get("YCO2")

        diff = {
            "dT_baseline": None if exp_T is None else (base_kpis["T_out_C"] - float(exp_T)),
            "dT_jax": None if exp_T is None else (jax_kpis["T_out_C"] - float(exp_T)),
            "dyCO_baseline": None if exp_yCO is None else (base_kpis["yCO"] - float(exp_yCO)),
            "dyCO_jax": None if exp_yCO is None else (jax_kpis["yCO"] - float(exp_yCO)),
            "dyH2_baseline": None if exp_yH2 is None else (base_kpis["yH2"] - float(exp_yH2)),
            "dyH2_jax": None if exp_yH2 is None else (jax_kpis["yH2"] - float(exp_yH2)),
            "dyCO2_baseline": None if exp_yCO2 is None else (base_kpis["yCO2"] - float(exp_yCO2)),
            "dyCO2_jax": None if exp_yCO2 is None else (jax_kpis["yCO2"] - float(exp_yCO2)),
        }

        results[case_name] = {
            "base_time_s": t_base,
            "jax_time_s": t_jax,
            "base_kpis": base_kpis,
            "jax_kpis": jax_kpis,
            "max_diffs": max_diffs,
            "expected": expected,
            "diff_to_expected": diff,
            "z_equal": bool(np.allclose(z_b, z_j)),
        }

        print(f"  expected T(°C)={exp_T if exp_T is not None else '-'}")
        print(
            f"  baseline(min) T={base_kpis['T_out_C']:.1f}C yCO={base_kpis['yCO']:.1f}% yH2={base_kpis['yH2']:.1f}% yCO2={base_kpis['yCO2']:.2f}%"
        )
        print(
            f"  jax_newton   T={jax_kpis['T_out_C']:.1f}C yCO={jax_kpis['yCO']:.1f}% yH2={jax_kpis['yH2']:.1f}% yCO2={jax_kpis['yCO2']:.2f}%"
        )
        print(
            f"  max|ΔT|(K)={max_diffs['max|ΔT|']:.2f}, max|ΔCO|(mol/s)={max_diffs['max|ΔCO|']:.2e}"
        )

    # 生成 Markdown
    lines = []
    lines.append("# gasifier-1d-kinetic：`jax_newton` vs expected（按 UI KPI 口径）")
    lines.append("")
    lines.append(f"- N_cells: {args.N_cells}")
    lines.append("- baseline: `minimize`（UI 默认）")
    lines.append("- jax solver: `jax_newton`")
    lines.append("")

    lines.append("## 出口 KPI 与 expected 对比（含 wall time）")
    lines.append(
        "| Case | exp T(°C) | baseline T | jax T | exp yCO | baseline yCO | jax yCO | exp yH2 | baseline yH2 | jax yH2 | exp yCO2 | baseline yCO2 | jax yCO2 | baseline time(s) | jax time(s) | max|ΔT|(K) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for case_name in case_names:
        r = results[case_name]
        exp = r["expected"]
        exp_T = exp.get("TOUT_C")
        exp_yCO = exp.get("YCO")
        exp_yH2 = exp.get("YH2")
        exp_yCO2 = exp.get("YCO2")
        max_d = r["max_diffs"]
        lines.append(
            "| "
            + " | ".join(
                [
                    case_name,
                    "-" if exp_T is None else _fmt(float(exp_T), 1),
                    _fmt(r["base_kpis"]["T_out_C"], 1),
                    _fmt(r["jax_kpis"]["T_out_C"], 1),
                    "-" if exp_yCO is None else _fmt(float(exp_yCO), 2),
                    _fmt(r["base_kpis"]["yCO"], 2),
                    _fmt(r["jax_kpis"]["yCO"], 2),
                    "-" if exp_yH2 is None else _fmt(float(exp_yH2), 2),
                    _fmt(r["base_kpis"]["yH2"], 2),
                    _fmt(r["jax_kpis"]["yH2"], 2),
                    "-" if exp_yCO2 is None else _fmt(float(exp_yCO2), 2),
                    _fmt(r["base_kpis"]["yCO2"], 2),
                    _fmt(r["jax_kpis"]["yCO2"], 2),
                    _fmt(r["base_time_s"], 2),
                    _fmt(r["jax_time_s"], 2),
                    _fmt(max_d["max|ΔT|"], 2),
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## expected 差值（正数=高估；负数=低估）")
    lines.append(
        "| Case | dT_baseline | dT_jax | dyCO_baseline | dyCO_jax | dyH2_baseline | dyH2_jax | dyCO2_baseline | dyCO2_jax |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for case_name in case_names:
        r = results[case_name]
        diff = r["diff_to_expected"]
        def fmt_or_dash(v):
            return "-" if v is None else _fmt(v, 2)

        lines.append(
            "| "
            + " | ".join(
                [
                    case_name,
                    fmt_or_dash(diff["dT_baseline"]),
                    fmt_or_dash(diff["dT_jax"]),
                    fmt_or_dash(diff["dyCO_baseline"]),
                    fmt_or_dash(diff["dyCO_jax"]),
                    fmt_or_dash(diff["dyH2_baseline"]),
                    fmt_or_dash(diff["dyH2_jax"]),
                    fmt_or_dash(diff["dyCO2_baseline"]),
                    fmt_or_dash(diff["dyCO2_jax"]),
                ]
            )
            + " |"
        )

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n=== 报告已生成：{args.out} ===")


if __name__ == "__main__":
    main()

