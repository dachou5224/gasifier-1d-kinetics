#!/usr/bin/env python3
"""
网格敏感性（严格版）：只研究上部点火区“内部细网格”精细度，
但保持点火区总物理长度不变。

实现方式：
- 固定 target_ignition_total_length = (cell0_length) + (fine_cells_count)*dz_ref
- 然后在相同 target 下改变 ignition_zone_res（IgnitionZoneDz），
  让内部用更细/更粗的单元数重新划分。
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
from model.constants import PhysicalConstants
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


def _gasifier_setup_paper_case_1() -> tuple[dict, dict, dict]:
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
    }
    return coal_props, geometry, op_conds


def _compute_cell0_length(op_conds: dict, n_cells: int) -> float:
    dz_cell0 = op_conds.get("FirstCellLength")
    if dz_cell0 is None and op_conds.get("AdaptiveFirstCellLength", False):
        m_coal_g_s = op_conds["coal_flow"] * 1000.0
        m_ref = 77.0
        dz_base = PhysicalConstants.FIRST_CELL_LENGTH
        dz_cell0 = dz_base * (m_coal_g_s / m_ref) ** 0.4
        dz_cell0 = max(0.03, min(0.5, float(dz_cell0)))
    if dz_cell0 is None:
        dz_cell0 = PhysicalConstants.FIRST_CELL_LENGTH
    return float(dz_cell0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-cells", type=int, default=40)
    p.add_argument(
        "--dz-list",
        type=float,
        nargs="+",
        default=[0.16, 0.14, 0.12, 0.10, 0.08, 0.06, 0.05],
        help="IgnitionZoneDz 列表（m），内部单元长度/精细度",
    )
    args = p.parse_args()

    coal_props, geometry, op_base = _gasifier_setup_paper_case_1()

    dz_ref = float(PhysicalConstants.IGNITION_ZONE_DZ)  # 默认 0.08（与你上一次 baseline 一致）
    cell0_len = _compute_cell0_length(op_base, args.n_cells)
    fine_end_default = min(20, args.n_cells)
    fine_cells_count = max(fine_end_default - 1, 0)
    target_total = cell0_len + fine_cells_count * dz_ref

    print(
        f"Paper_Case_1, solver_method=newton_fd, N_cells={args.n_cells}\n"
        f"固定点火区总物理长度：IgnitionZoneTotalLength={target_total:.4f} m\n"
        f"引用 dz_ref={dz_ref:.4f} m，cell0_len={cell0_len:.4f} m，fine_cells_count={fine_cells_count}\n"
        f"IgnitionZoneDz list: {args.dz_list}\n"
    )

    baseline = None
    rows = []

    for dz in [float(x) for x in args.dz_list]:
        op_conds = dict(op_base)
        op_conds["IgnitionZoneDz"] = dz
        op_conds["IgnitionZoneTotalLength"] = target_total

        sys_g = GasifierSystem(geometry, coal_props, op_conds)
        t0 = time.perf_counter()
        prof, _z = sys_g.solve(
            N_cells=args.n_cells,
            solver_method="newton_fd",
            jacobian_mode="centered_fd",
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
        print()

        if abs(dz - dz_ref) < 1e-12:
            baseline = k

    if baseline is None:
        baseline = rows[0][2]
        dz_ref_used = rows[0][0]
    else:
        dz_ref_used = dz_ref

    print(f"Baseline（dz_ref={dz_ref_used:.3f} m）")
    for dz, _elapsed, k in rows:
        dT = k["T_out_C"] - baseline["T_out_C"]
        dyCO = k["yCO"] - baseline["yCO"]
        dyH2 = k["yH2"] - baseline["yH2"]
        dyCO2 = k["yCO2"] - baseline["yCO2"]
        print(
            f"IgnitionZoneDz={dz:.3f} m: "
            f"dT={dT:+.3f}, dyCO={dyCO:+.3f}, dyH2={dyH2:+.3f}, dyCO2={dyCO2:+.3f}"
        )


if __name__ == "__main__":
    main()
