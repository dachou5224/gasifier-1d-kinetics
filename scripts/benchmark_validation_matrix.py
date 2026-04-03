#!/usr/bin/env python3
"""
跑 VALIDATION_CASES 的参数矩阵，并输出 JSON + Markdown 汇总。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_validation_newton_fd_vs_expected import _build_system, _kpis, VALIDATION_CASES


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_jacobian_mode_list(s: str) -> list[str]:
    vals = []
    for x in s.split(","):
        t = x.strip()
        if t:
            vals.append(t)
    return vals


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--solver-method", type=str, default="newton_fd")
    p.add_argument("--n-cells-list", type=str, default="20,40")
    p.add_argument("--stretch-list", type=str, default="1.0,1.02,1.06")
    p.add_argument("--jacobian-mode-list", type=str, default="scipy,centered_fd")
    p.add_argument("--fixed-ignition-length", action="store_true", default=True)
    p.add_argument("--json-out", type=str, default="docs/validation_matrix_report.json")
    p.add_argument("--md-out", type=str, default="docs/validation_matrix_report.md")
    args = p.parse_args()

    n_cells_list = _parse_int_list(args.n_cells_list)
    stretch_list = _parse_float_list(args.stretch_list)
    jacobian_list = _parse_jacobian_mode_list(args.jacobian_mode_list)
    case_names = sorted(VALIDATION_CASES.keys())

    rows = []
    for n_cells in n_cells_list:
        for stretch in stretch_list:
            for jacobian_mode in jacobian_list:
                combo = {
                    "solver_method": args.solver_method,
                    "n_cells": n_cells,
                    "stretch": stretch,
                    "jacobian_mode": jacobian_mode,
                    "cases": [],
                }
                abs_dt_sum = 0.0
                total_time = 0.0
                total_fallback = 0
                total_poor = 0
                for name in case_names:
                    exp = VALIDATION_CASES[name].get("expected") or {}
                    sys_g, _mesh = _build_system(
                        name,
                        n_cells=n_cells,
                        fixed_ignition_length=args.fixed_ignition_length,
                    )
                    sys_g.op_conds["IgnitionZoneStretchRatio"] = stretch
                    t0 = time.perf_counter()
                    prof, _ = sys_g.solve(
                        N_cells=n_cells,
                        solver_method=args.solver_method,
                        jacobian_mode=jacobian_mode,
                        jax_warmup=True,
                    )
                    elapsed = time.perf_counter() - t0
                    k = _kpis(prof)
                    dT = None
                    if exp.get("TOUT_C") is not None:
                        dT = k["T_out_C"] - float(exp["TOUT_C"])
                        abs_dt_sum += abs(dT)
                    total_time += elapsed
                    total_fallback += int(sys_g.solve_stats.get("fallback_count", 0))
                    total_poor += int(sys_g.solve_stats.get("poor_convergence_count", 0))
                    combo["cases"].append(
                        {
                            "case": name,
                            "time_s": elapsed,
                            "dT": dT,
                            "sim": k,
                        }
                    )
                combo["abs_dT_sum"] = abs_dt_sum
                combo["time_sum_s"] = total_time
                combo["fallback_count"] = total_fallback
                combo["poor_convergence_count"] = total_poor
                rows.append(combo)

    rows_sorted = sorted(rows, key=lambda x: (x["abs_dT_sum"], x["time_sum_s"]))

    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(
            {
                "solver_method": args.solver_method,
                "n_cells_list": n_cells_list,
                "stretch_list": stretch_list,
                "jacobian_mode_list": jacobian_list,
                "rows": rows_sorted,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    md_out = Path(args.md_out)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Validation Matrix Report\n")
    lines.append(
        f"- solver_method: `{args.solver_method}`\n"
        f"- n_cells: `{n_cells_list}`\n"
        f"- stretch: `{stretch_list}`\n"
        f"- jacobian_mode: `{jacobian_list}`\n"
    )
    lines.append("| rank | n_cells | stretch | jacobian_mode | abs_dT_sum | time_sum(s) | fallback | poor_conv |\n")
    lines.append("|---:|---:|---:|---|---:|---:|---:|---:|\n")
    for i, r in enumerate(rows_sorted, 1):
        lines.append(
            f"| {i} | {r['n_cells']} | {r['stretch']:.2f} | {r['jacobian_mode']} | "
            f"{r['abs_dT_sum']:.2f} | {r['time_sum_s']:.2f} | {r['fallback_count']} | {r['poor_convergence_count']} |\n"
        )
    md_out.write_text("".join(lines), encoding="utf-8")

    print(f"JSON written: {json_out}")
    print(f"MD written: {md_out}")
    if rows_sorted:
        best = rows_sorted[0]
        print(
            "BEST:",
            f"n_cells={best['n_cells']}, stretch={best['stretch']:.2f}, "
            f"jacobian_mode={best['jacobian_mode']}, abs_dT_sum={best['abs_dT_sum']:.2f}, "
            f"time_sum={best['time_sum_s']:.2f}",
        )


if __name__ == "__main__":
    main()
