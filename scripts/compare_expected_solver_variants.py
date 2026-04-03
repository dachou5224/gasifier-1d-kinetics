"""
对比 validation cases 的三种求解路径与 expected KPI 的差值：

1) baseline：solver_method='minimize'（UI 默认）
2) trf_centered_fd：solver_method='minimize' 且 jacobian_mode='centered_fd'
3) newton_fd：solver_method='newton_fd'

输出：
- 终端打印每个 case 的 expected KPI 与三种路径的输出 KPI
- 终端打印 dT/dyCO/dyH2/dyCO2（正数表示高估）
- 生成 Markdown 表：docs/expected_solver_variants_report.md
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Dict

import numpy as np


def _ensure_import_path() -> None:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src_path = os.path.join(project_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def _compute_kpis(profile: np.ndarray) -> Dict[str, float]:
    last = profile[-1]
    gas = last[:8].astype(float)  # [O2,CH4,CO,CO2,H2S,H2,N2,H2O]
    F_dry = float(np.sum(gas[:7]) + 1e-12)  # 干基：排除 H2O（UI 口径）
    yCO = float(gas[2] / F_dry * 100.0)
    yH2 = float(gas[5] / F_dry * 100.0)
    yCO2 = float(gas[3] / F_dry * 100.0)
    T_out_C = float(last[10] - 273.15)
    return {"T_out_C": T_out_C, "yCO": yCO, "yH2": yH2, "yCO2": yCO2}


def _build_system(case_name: str):
    from model.chemistry import COAL_DATABASE, VALIDATION_CASES
    from model.gasifier_system import GasifierSystem

    case = VALIDATION_CASES[case_name]
    case_inputs = case["inputs"]
    expected = case.get("expected", {})

    coal_key = case_inputs.get("coal") or case_inputs.get("coal_type")
    coal_props = COAL_DATABASE[coal_key].copy()

    geometry = {
        "L": float(case_inputs.get("L_reactor", case_inputs.get("L_m", 6.0))),
        "D": float(case_inputs.get("D_reactor", case_inputs.get("D_m", 2.0))),
    }

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

    return GasifierSystem(geometry, coal_props, op_conds), expected


def _d_to_expected(actual: Dict[str, float], expected: Dict[str, float]) -> Dict[str, float]:
    # expected uses: TOUT_C, YCO, YH2, YCO2
    out = {}
    if "TOUT_C" in expected:
        out["dT_C"] = actual["T_out_C"] - float(expected["TOUT_C"])
    if "YCO" in expected:
        out["dyCO"] = actual["yCO"] - float(expected["YCO"])
    if "YH2" in expected:
        out["dyH2"] = actual["yH2"] - float(expected["YH2"])
    if "YCO2" in expected:
        out["dyCO2"] = actual["yCO2"] - float(expected["YCO2"])
    return out


def _fmt(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N_cells", type=int, default=20)
    parser.add_argument("--out", type=str, default="docs/expected_solver_variants_report.md")
    parser.add_argument("--log-level", type=str, default="ERROR")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.ERROR))
    _ensure_import_path()

    # 只跑 4 个 cases（validation 中最核心的）
    case_names = ["Paper_Case_6", "Paper_Case_1", "Paper_Case_2", "LuNan_Texaco_Slurry"]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    rows = []
    lines = []
    lines.append("# expected 对比：minimize vs minimize(centered_fd) vs newton_fd")
    lines.append("")
    lines.append(f"- N_cells: {args.N_cells}")
    lines.append("")
    lines.append("## 汇总表（终端打印与 MD 报告一致）")
    lines.append(
        "| Case | solver | time(s) | T_out_C | yCO(%) | yH2(%) | yCO2(%) | dT_C | dyCO | dyH2 | dyCO2 |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for case_name in case_names:
        sys0, expected = _build_system(case_name)

        # 1) baseline minimize
        t0 = time.perf_counter()
        prof_b, _z = sys0.solve(N_cells=args.N_cells, solver_method="minimize", jacobian_mode="scipy", jax_warmup=False)
        tb = time.perf_counter() - t0
        k_b = _compute_kpis(prof_b)
        d_b = _d_to_expected(k_b, expected)

        # 2) trf + centered_fd Jacobian
        sys1, _ = _build_system(case_name)
        t1 = time.perf_counter()
        prof_jt, _z2 = sys1.solve(N_cells=args.N_cells, solver_method="minimize", jacobian_mode="centered_fd", jax_warmup=True)
        tj = time.perf_counter() - t1
        k_jt = _compute_kpis(prof_jt)
        d_jt = _d_to_expected(k_jt, expected)

        # 3) newton_fd
        sys2, _ = _build_system(case_name)
        t2 = time.perf_counter()
        prof_n, _z3 = sys2.solve(N_cells=args.N_cells, solver_method="newton_fd", jacobian_mode="centered_fd", jax_warmup=True)
        tn = time.perf_counter() - t2
        k_n = _compute_kpis(prof_n)
        d_n = _d_to_expected(k_n, expected)

        print(f"\n=== {case_name} ===")
        print(f"expected: T={expected.get('TOUT_C','-')}C yCO={expected.get('YCO','-')} yH2={expected.get('YH2','-')} yCO2={expected.get('YCO2','-')}")

        def show(label, time_s, k, d):
            print(
                f"{label}: time={time_s:.2f}s "
                f"T={k['T_out_C']:.1f}C yCO={k['yCO']:.2f}% yH2={k['yH2']:.2f}% yCO2={k['yCO2']:.2f}% "
                f"dT={d.get('dT_C', 0):+.2f} dyCO={d.get('dyCO', 0):+.2f} dyH2={d.get('dyH2', 0):+.2f} dyCO2={d.get('dyCO2', 0):+.2f}"
            )

        show("minimize", tb, k_b, d_b)
        show("minimize(centered_fd)", tj, k_jt, d_jt)
        show("newton_fd", tn, k_n, d_n)

        rows.append((case_name, "minimize", tb, k_b, d_b))
        rows.append((case_name, "minimize+centered_fd", tj, k_jt, d_jt))
        rows.append((case_name, "newton_fd", tn, k_n, d_n))

    for case_name, solver, t_s, k, d in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    case_name,
                    solver,
                    _fmt(t_s, 2),
                    _fmt(k["T_out_C"], 1),
                    _fmt(k["yCO"], 2),
                    _fmt(k["yH2"], 2),
                    _fmt(k["yCO2"], 2),
                    _fmt(d.get("dT_C", 0.0), 2),
                    _fmt(d.get("dyCO", 0.0), 2),
                    _fmt(d.get("dyH2", 0.0), 2),
                    _fmt(d.get("dyCO2", 0.0), 2),
                ]
            )
            + " |"
        )

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n=== 报告生成：{args.out} ===")


if __name__ == "__main__":
    main()
