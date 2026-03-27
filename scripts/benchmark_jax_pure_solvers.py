#!/usr/bin/env python3
"""
固化基准：同一工况下对比 solver_method=newton / jax_pure / jax_newton 的壁钟时间与出口 KPI。

示例：
  PYTHONPATH=src python3 scripts/benchmark_jax_pure_solvers.py --case Paper_Case_1 --n-cells 10 20
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np

logging.basicConfig(level=logging.ERROR)

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model.chemistry import COAL_DATABASE, VALIDATION_CASES
from model.gasifier_system import GasifierSystem


def _kpis(profile: np.ndarray) -> dict:
    last = profile[-1]
    gas = last[:8].astype(float)
    f_dry = float(np.sum(gas[:7]) + 1e-12)
    return {
        "T_out_C": float(last[10] - 273.15),
        "yCO": float(gas[2] / f_dry * 100.0),
        "yH2": float(gas[5] / f_dry * 100.0),
        "yCO2": float(gas[3] / f_dry * 100.0),
    }


def _build(case_name: str) -> GasifierSystem:
    case = VALIDATION_CASES[case_name]
    ci = case["inputs"]
    coal_props = COAL_DATABASE[ci["coal"]].copy()
    geometry = {
        "L": float(ci.get("L_reactor", ci.get("L_m", 6.0))),
        "D": float(ci.get("D_reactor", ci.get("D_m", 2.0))),
    }
    coal_flow_kg_s = float(ci["FeedRate"]) / 3600.0
    op_conds = {
        "coal_flow": coal_flow_kg_s,
        "o2_flow": coal_flow_kg_s * float(ci.get("Ratio_OC", 1.05)),
        "steam_flow": coal_flow_kg_s * float(ci.get("Ratio_SC", 0.0)),
        "P": float(ci["P"]),
        "T_in": float(ci["TIN"]),
        "SlurryConcentration": float(ci.get("SlurryConcentration", 60.0)),
        "HeatLossPercent": float(ci.get("HeatLossPercent", 1.0)),
    }
    return GasifierSystem(geometry, coal_props, op_conds)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="Paper_Case_1")
    p.add_argument("--n-cells", type=int, nargs="+", default=[10, 20])
    args = p.parse_args()

    expected = VALIDATION_CASES[args.case].get("expected", {})
    print(f"case {args.case} expected {expected}")

    for n in args.n_cells:
        print(f"\n=== N_cells={n} ===")
        for method in ("newton", "jax_pure", "jax_newton"):
            sys_g = _build(args.case)
            t0 = time.perf_counter()
            prof, _z = sys_g.solve(N_cells=n, solver_method=method, jax_warmup=True)
            elapsed = time.perf_counter() - t0
            k = _kpis(prof)
            print(f"{method:10} time(s) {elapsed:.4f}  kpi {k}")


if __name__ == "__main__":
    main()
