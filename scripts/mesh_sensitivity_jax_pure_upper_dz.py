#!/usr/bin/env python3
"""
网格敏感性：只研究上部（起燃/点火区）网格步长 IgnitionZoneDz 对出口结果影响。

- 固定 solver_method='jax_pure'
- 固定 Paper_Case_1 工况其它参数
- 仅改变 IgnitionZoneDz（MeshConfig.ignition_zone_res）
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.ERROR)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

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


def _build_paper_case_1_system(ignition_zone_dz: float) -> GasifierSystem:
    case = VALIDATION_CASES["Paper_Case_1"]
    ci = case["inputs"]

    coal_props = COAL_DATABASE[ci["coal"]].copy()
    geometry = {
        "L": float(ci.get("L_reactor", ci.get("L_m", 6.0))),
        "D": float(ci.get("D_reactor", ci.get("D_m", 2.0))),
    }

    coal_flow_kg_s = float(ci["FeedRate"]) / 3600.0
    op_conds = {
        "coal_flow": coal_flow_kg_s,
        "o2_flow": coal_flow_kg_s * float(ci.get("Ratio_OC", 1.06)),
        "steam_flow": coal_flow_kg_s * float(ci.get("Ratio_SC", 0.35)),
        "P": float(ci["P"]),
        "T_in": float(ci["TIN"]),
        "HeatLossPercent": float(ci.get("HeatLossPercent", 4.5)),
        "SlurryConcentration": float(ci.get("SlurryConcentration", 100.0)),
        "Combustion_CO2_Fraction": float(ci.get("Combustion_CO2_Fraction", 0.15)),
        "WGS_CatalyticFactor": float(ci.get("WGS_CatalyticFactor", 1.5)),
        # 这里是敏感性参数：点火区网格步长
        "IgnitionZoneDz": float(ignition_zone_dz),
    }
    return GasifierSystem(geometry, coal_props, op_conds)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-cells", type=int, default=40)
    p.add_argument(
        "--dz-list",
        type=float,
        nargs="+",
        default=[0.12, 0.08, 0.06, 0.05],
        help="IgnitionZoneDz 列表（m），按从粗到细建议放后验收",
    )
    args = p.parse_args()

    dz_list = [float(x) for x in args.dz_list]
    print(f"Paper_Case_1, solver_method=jax_pure, N_cells={args.n_cells}")
    print(f"IgnitionZoneDz list: {dz_list}\n")

    baseline_dz = dz_list[len(dz_list) // 2]  # 经验：把中间值当 baseline
    kpi_base = None
    base_key = None

    rows = []
    for dz in dz_list:
        sys_g = _build_paper_case_1_system(dz)
        t0 = time.perf_counter()
        prof, _z = sys_g.solve(
            N_cells=args.n_cells,
            solver_method="jax_pure",
            jax_warmup=True,
        )
        elapsed = time.perf_counter() - t0
        k = _kpis(prof)

        rows.append((dz, elapsed, k))
        print(f"=== IgnitionZoneDz={dz:.3f} m ===")
        print(
            f"  time(s)={elapsed:.3f}  "
            f"T_out_C={k['T_out_C']:.3f}  yCO={k['yCO']:.3f}  "
            f"yH2={k['yH2']:.3f}  yCO2={k['yCO2']:.3f}"
        )

        if abs(dz - baseline_dz) < 1e-12:
            kpi_base = k
            base_key = dz
        print()

    if kpi_base is None:
        # fallback：取第一个
        base_key = rows[0][0]
        kpi_base = rows[0][2]

    print(f"Baseline IgnitionZoneDz={base_key:.3f} m\n")
    for dz, elapsed, k in rows:
        dT = k["T_out_C"] - kpi_base["T_out_C"]
        dyCO = k["yCO"] - kpi_base["yCO"]
        dyH2 = k["yH2"] - kpi_base["yH2"]
        dyCO2 = k["yCO2"] - kpi_base["yCO2"]
        print(
            f"IgnitionZoneDz={dz:.3f} m: "
            f"dT={dT:+.3f}, dyCO={dyCO:+.3f}, dyH2={dyH2:+.3f}, dyCO2={dyCO2:+.3f}"
        )


if __name__ == "__main__":
    main()

