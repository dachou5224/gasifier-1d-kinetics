"""
遍历 `model.chemistry.VALIDATION_CASES`，对比 baseline(newton) vs jax_newton：

- 对比出口数据：`T_out_C`，以及干基 `yCO/yH2/yCO2`（与 `gasifier_kinetic_ui.py` 一致）
- 对比 profile：`max|ΔT|`、`max|ΔCO|`、`max|ΔCO2|`、`max|ΔH2|`、`max|ΔCH4|`
- 对比运行时间：baseline 与 jax_newton 的 wall time
- 生成 Markdown 报告：便于粘贴到文档/PR

用法示例：
  cd gasifier-1d-kinetic
  python3 scripts/validate_jax_solver_all_cases.py --N_cells 20 --out docs/validation_jax_newton_report.md
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


def _ensure_import_path() -> None:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src_path = os.path.join(project_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def _dry_y(gas: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    与 gasifier_kinetic_ui.py 对齐：
    F_dry = sum(gas[:7])，gas 顺序：[O2, CH4, CO, CO2, H2S, H2, N2, H2O]
    """
    F_dry = float(np.sum(gas[:7]) + 1e-12)
    return F_dry, gas


def _compute_kpis_from_profile(profile: np.ndarray) -> Dict[str, float]:
    """
    profile: (N_cells, 11) 列含：
      O2, CH4, CO, CO2, H2S, H2, N2, H2O, Ws, Xc, T_g(K)
    """
    last = profile[-1]
    gas = last[:8].astype(float)
    F_dry, _ = _dry_y(gas)

    y_co = float(gas[2] / F_dry * 100.0)
    y_h2 = float(gas[5] / F_dry * 100.0)
    y_co2 = float(gas[3] / F_dry * 100.0)
    t_out_c = float(last[10] - 273.15)

    return {
        "T_out_C": t_out_c,
        "yCO": y_co,
        "yH2": y_h2,
        "yCO2": y_co2,
    }


def _max_profile_diffs(base: np.ndarray, jax: np.ndarray) -> Dict[str, float]:
    d = jax - base
    return {
        "max|ΔT|": float(np.max(np.abs(d[:, 10]))),
        "max|ΔCO|": float(np.max(np.abs(d[:, 2]))),
        "max|ΔCO2|": float(np.max(np.abs(d[:, 3]))),
        "max|ΔH2|": float(np.max(np.abs(d[:, 5]))),
        "max|ΔCH4|": float(np.max(np.abs(d[:, 1]))),
    }


@dataclass
class CaseResult:
    case_name: str
    N_cells: int
    baseline_time_s: float
    jax_time_s: float
    baseline_kpis: Dict[str, float]
    jax_kpis: Dict[str, float]
    diff_kpis: Dict[str, float]
    max_profile_diffs: Dict[str, float]
    expected_kpis: Dict[str, float]


def _build_system_from_case(case_name: str, case_inputs: Dict[str, float], coal_database: Dict) -> "GasifierSystem":
    from model.gasifier_system import GasifierSystem

    coal_key = case_inputs.get("coal") or case_inputs.get("coal_type")
    coal_props = coal_database[coal_key].copy()

    # validation_cases.py 的约定：默认 L=6m, D=2m（与 UI 默认一致）
    geometry = {
        "L": float(case_inputs.get("L_reactor", 6.0)),
        "D": float(case_inputs.get("D_reactor", 2.0)),
    }

    feed_rate_kg_h = float(case_inputs["FeedRate"])
    coal_flow_kg_s = feed_rate_kg_h / 3600.0
    o2_flow_kg_s = coal_flow_kg_s * float(case_inputs.get("Ratio_OC", 1.0))
    steam_flow_kg_s = coal_flow_kg_s * float(case_inputs.get("Ratio_SC", case_inputs.get("Ratio_SC", 0.0)))

    op_conds = {
        "coal_flow": coal_flow_kg_s,
        "o2_flow": o2_flow_kg_s,
        # 本质：Ratio_SC = steam fraction mass basis（dry feed 情况会用到）
        "steam_flow": steam_flow_kg_s,
        "P": float(case_inputs["P"]),
        "T_in": float(case_inputs["TIN"]),
        "HeatLossPercent": float(case_inputs.get("HeatLossPercent", 1.0)),
        "SlurryConcentration": float(case_inputs.get("SlurryConcentration", 60.0)),
        "Combustion_CO2_Fraction": float(case_inputs.get("Combustion_CO2_Fraction", 0.15)),
        "WGS_CatalyticFactor": float(case_inputs.get("WGS_CatalyticFactor", 1.5)),
        "AdaptiveFirstCellLength": True,
    }

    # 可选字段：与 cell.py / kinetics_service.py 对齐
    if "P_O2_Combustion_atm" in case_inputs:
        op_conds["P_O2_Combustion_atm"] = float(case_inputs["P_O2_Combustion_atm"])

    return GasifierSystem(geometry, coal_props, op_conds)


def _run_case(
    case_name: str,
    case_inputs: Dict[str, float],
    expected: Dict[str, float],
    coal_database: Dict,
    *,
    N_cells: int,
    first_jax: bool,
) -> CaseResult:
    from model.chemistry import COAL_DATABASE

    from model.gasifier_system import GasifierSystem

    # baseline
    sys_b = _build_system_from_case(case_name, case_inputs, coal_database)
    t0 = time.perf_counter()
    base_profile, _z = sys_b.solve(
        N_cells=N_cells,
        solver_method="newton",
        use_jax_jacobian=False,
        jax_warmup=False,
    )
    t_base = time.perf_counter() - t0
    base_kpis = _compute_kpis_from_profile(base_profile)

    # jax path
    sys_j = _build_system_from_case(case_name, case_inputs, coal_database)
    t1 = time.perf_counter()
    jax_profile, _z2 = sys_j.solve(
        N_cells=N_cells,
        solver_method="jax_newton",
        use_jax_jacobian=False,
        jax_warmup=first_jax,
    )
    t_jax = time.perf_counter() - t1
    jax_kpis = _compute_kpis_from_profile(jax_profile)

    diff_kpis = {
        "dT_C": jax_kpis["T_out_C"] - base_kpis["T_out_C"],
        "dyCO": jax_kpis["yCO"] - base_kpis["yCO"],
        "dyH2": jax_kpis["yH2"] - base_kpis["yH2"],
        "dyCO2": jax_kpis["yCO2"] - base_kpis["yCO2"],
    }

    max_diffs = _max_profile_diffs(base_profile, jax_profile)

    return CaseResult(
        case_name=case_name,
        N_cells=N_cells,
        baseline_time_s=float(t_base),
        jax_time_s=float(t_jax),
        baseline_kpis=base_kpis,
        jax_kpis=jax_kpis,
        diff_kpis=diff_kpis,
        max_profile_diffs=max_diffs,
        expected_kpis=expected,
    )


def _fmt(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def _fmt_e(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}e}"


def _to_md_table(rows: Dict[str, str]) -> str:
    # 辅助：把多列拼成一行（不建议在复杂场景使用）
    return " | ".join(rows.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N_cells", type=int, default=20)
    parser.add_argument("--out", type=str, default="docs/validation_jax_newton_report.md")
    parser.add_argument("--log-level", type=str, default="ERROR")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.ERROR))
    _ensure_import_path()

    from model.chemistry import COAL_DATABASE, VALIDATION_CASES

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    case_names = list(VALIDATION_CASES.keys())
    results: Dict[str, CaseResult] = {}

    print(f"=== 开始验证：N_cells={args.N_cells}，总案例数={len(case_names)} ===")

    for idx, case_name in enumerate(case_names):
        case = VALIDATION_CASES[case_name]
        expected = case.get("expected", {})
        case_inputs = case["inputs"]
        print(f"\n[Case {idx+1}/{len(case_names)}] {case_name}")
        first_jax = idx == 0
        r = _run_case(
            case_name,
            case_inputs,
            expected,
            COAL_DATABASE,
            N_cells=args.N_cells,
            first_jax=first_jax,
        )
        results[case_name] = r

        print(
            f"  baseline(newton): {r.baseline_time_s:.2f}s | "
            f"T={r.baseline_kpis['T_out_C']:.1f}C yCO={r.baseline_kpis['yCO']:.1f}% yH2={r.baseline_kpis['yH2']:.1f}% yCO2={r.baseline_kpis['yCO2']:.2f}%"
        )
        print(
            f"  jax_newton      : {r.jax_time_s:.2f}s | "
            f"T={r.jax_kpis['T_out_C']:.1f}C yCO={r.jax_kpis['yCO']:.1f}% yH2={r.jax_kpis['yH2']:.1f}% yCO2={r.jax_kpis['yCO2']:.2f}%"
        )

    # Markdown report
    lines = []
    lines.append("# gasifier-1d-kinetic：验证 `jax_newton` vs baseline(newton)")
    lines.append("")
    lines.append(f"- N_cells: {args.N_cells}")
    lines.append("- baseline solver: `newton`")
    lines.append("- jax solver: `jax_newton`")
    lines.append("- jax_warmup: 仅对第一个案例启用（用于摊销编译开销）")
    lines.append("")

    # Summary table: baseline vs jax + time + profile diffs
    lines.append("## 汇总对比（出口 KPI + 运行时间 + profile 最大误差）")
    lines.append(
        "| Case | baseline T(°C) | baseline yCO(%) | baseline yH2(%) | baseline yCO2(%) | baseline time(s) | jax_newton T(°C) | jax_newton yCO(%) | jax_newton yH2(%) | jax_newton yCO2(%) | jax time(s) | max|ΔT|(K) | max|ΔCO|(mol/s) | max|ΔCO2|(mol/s) | max|ΔH2|(mol/s) | max|ΔCH4|(mol/s) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for case_name in case_names:
        r = results[case_name]
        md = r.max_profile_diffs
        lines.append(
            "| "
            + " | ".join(
                [
                    case_name,
                    _fmt(r.baseline_kpis["T_out_C"], 1),
                    _fmt(r.baseline_kpis["yCO"], 2),
                    _fmt(r.baseline_kpis["yH2"], 2),
                    _fmt(r.baseline_kpis["yCO2"], 2),
                    _fmt(r.baseline_time_s, 2),
                    _fmt(r.jax_kpis["T_out_C"], 1),
                    _fmt(r.jax_kpis["yCO"], 2),
                    _fmt(r.jax_kpis["yH2"], 2),
                    _fmt(r.jax_kpis["yCO2"], 2),
                    _fmt(r.jax_time_s, 2),
                    _fmt(md["max|ΔT|"], 3),
                    _fmt_e(md["max|ΔCO|"], 2),
                    _fmt_e(md["max|ΔCO2|"], 2),
                    _fmt_e(md["max|ΔH2|"], 2),
                    _fmt_e(md["max|ΔCH4|"], 2),
                ]
            )
            + " |"
        )

    lines.append("")

    # Expected comparison table
    lines.append("## 与预期（VALIDATION_CASES.expected）对齐情况")
    lines.append(
        "| Case | expected T(°C) | baseline ΔT(K) | jax_newton ΔT(K) | expected yCO(%) | baseline ΔyCO(%) | jax_newton ΔyCO(%) | expected yH2(%) | baseline ΔyH2(%) | jax_newton ΔyH2(%) | expected yCO2(%) | baseline ΔyCO2(%) | jax_newton ΔyCO2(%) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for case_name in case_names:
        r = results[case_name]
        exp = r.expected_kpis
        exp_T = exp.get("TOUT_C")
        exp_yCO = exp.get("YCO")
        exp_yH2 = exp.get("YH2")
        exp_yCO2 = exp.get("YCO2")

        def maybe_delta(actual: float, expected_val) -> str:
            if expected_val is None:
                return "-"
            return _fmt(actual - float(expected_val), 3)

        baseline_delta_T = "-" if exp_T is None else _fmt(r.baseline_kpis["T_out_C"] - float(exp_T), 3)
        jax_delta_T = "-" if exp_T is None else _fmt(r.jax_kpis["T_out_C"] - float(exp_T), 3)

        lines.append(
            "| "
            + " | ".join(
                [
                    case_name,
                    "-" if exp_T is None else _fmt(float(exp_T), 1),
                    baseline_delta_T,
                    jax_delta_T,
                    "-" if exp_yCO is None else _fmt(float(exp_yCO), 2),
                    maybe_delta(r.baseline_kpis["yCO"], exp_yCO),
                    maybe_delta(r.jax_kpis["yCO"], exp_yCO),
                    "-" if exp_yH2 is None else _fmt(float(exp_yH2), 2),
                    maybe_delta(r.baseline_kpis["yH2"], exp_yH2),
                    maybe_delta(r.jax_kpis["yH2"], exp_yH2),
                    "-" if exp_yCO2 is None else _fmt(float(exp_yCO2), 2),
                    maybe_delta(r.baseline_kpis["yCO2"], exp_yCO2),
                    maybe_delta(r.jax_kpis["yCO2"], exp_yCO2),
                ]
            )
            + " |"
        )

    md_text = "\n".join(lines) + "\n"
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md_text)

    print(f"\n=== 报告已生成：{args.out} ===")
    print("提示：你可以打开该 md 文件查看完整表格。")


if __name__ == "__main__":
    main()

