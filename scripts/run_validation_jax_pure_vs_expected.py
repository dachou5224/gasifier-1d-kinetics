#!/usr/bin/env python3
"""
使用当前 solver_method=jax_pure 配置，对 VALIDATION_CASES 中全部工况求解，
并将出口干基 KPI 与 case['expected'] 对比（不与 newton 对比）。

示例：
  PYTHONPATH=src python3 scripts/run_validation_jax_pure_vs_expected.py --n-cells 20
"""
from __future__ import annotations

import argparse
import json
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


def _cell0_length_m(op_conds: dict) -> float:
    """与 GasifierSystem.solve 中 FirstCellLength 逻辑一致。"""
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


def _build_system(
    case_name: str,
    *,
    n_cells: int,
    fixed_ignition_length: bool,
) -> tuple[GasifierSystem, dict]:
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
    for k in (
        "Combustion_CO2_Fraction",
        "WGS_CatalyticFactor",
        "L_evap_m",
        "AdaptiveFirstCellLength",
        "FirstCellLength",
        "IgnitionZoneDz",
        "IgnitionZoneFineStartZ",
        "IgnitionZoneStretchRatio",
        "n2_flow",
    ):
        if k in ci:
            op_conds[k] = ci[k]

    mesh_meta: dict = {}
    if fixed_ignition_length:
        dz_ref = float(op_conds.get("IgnitionZoneDz", PhysicalConstants.IGNITION_ZONE_DZ))
        L0 = _cell0_length_m(op_conds)
        fine_end = min(20, n_cells)
        n_fine = max(fine_end - 1, 0)
        target_total = L0 + n_fine * dz_ref
        op_conds["IgnitionZoneTotalLength"] = target_total
        mesh_meta = {
            "IgnitionZoneTotalLength_m": target_total,
            "cell0_len_m": L0,
            "dz_ref_m": dz_ref,
            "n_fine_slots": n_fine,
        }

    return GasifierSystem(geometry, coal_props, op_conds), mesh_meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-cells", type=int, default=20, help="轴向离散单元数")
    p.add_argument(
        "--ignition-zone-dz",
        type=float,
        default=None,
        help="覆盖 IgnitionZoneDz（m）",
    )
    p.add_argument(
        "--ignition-zone-stretch-ratio",
        type=float,
        default=None,
        help="覆盖 IgnitionZoneStretchRatio（>1 前细后粗，=1 等分）",
    )
    p.add_argument(
        "--fixed-ignition-length",
        action="store_true",
        help="启用与网格升级一致的固定点火区总长度（IgnitionZoneTotalLength，内部等分）",
    )
    p.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="可选：将结果写入 JSON 文件",
    )
    args = p.parse_args()

    case_names = sorted(VALIDATION_CASES.keys())
    print(
        f"solver_method=jax_pure, N_cells={args.n_cells}, "
        f"fixed_ignition_length={args.fixed_ignition_length}, cases={len(case_names)}"
        f"{', ignition_zone_dz='+str(args.ignition_zone_dz) if args.ignition_zone_dz is not None else ''}"
        f"{', stretch='+str(args.ignition_zone_stretch_ratio) if args.ignition_zone_stretch_ratio is not None else ''}\n"
    )
    json_rows = []

    for name in case_names:
        exp = VALIDATION_CASES[name].get("expected") or {}
        sys_g, mesh_meta = _build_system(
            name, n_cells=args.n_cells, fixed_ignition_length=args.fixed_ignition_length
        )
        if args.ignition_zone_dz is not None:
            sys_g.op_conds["IgnitionZoneDz"] = float(args.ignition_zone_dz)
        if args.ignition_zone_stretch_ratio is not None:
            sys_g.op_conds["IgnitionZoneStretchRatio"] = float(args.ignition_zone_stretch_ratio)
        t0 = time.perf_counter()
        prof, _z = sys_g.solve(
            N_cells=args.n_cells,
            solver_method="jax_pure",
            jax_warmup=True,
        )
        elapsed = time.perf_counter() - t0
        k = _kpis(prof)

        print(f"=== {name}  time(s)={elapsed:.3f} ===")
        if mesh_meta:
            print(f"  mesh  {mesh_meta}")
        print(f"  sim   T_out_C={k['T_out_C']:.4f}  yCO={k['yCO']:.4f}  yH2={k['yH2']:.4f}  yCO2={k['yCO2']:.4f}")
        if not exp:
            print("  expected: (none)")
            print()
            continue
        te = exp.get("TOUT_C")
        yco_e = exp.get("YCO")
        yh2_e = exp.get("YH2")
        yco2_e = exp.get("YCO2")
        if te is not None:
            print(f"  exp   T_out_C={te}  yCO={yco_e}  yH2={yh2_e}" + (f"  YCO2={yco2_e}" if yco2_e is not None else ""))
        parts = []
        if te is not None:
            parts.append(f"dT={k['T_out_C'] - float(te):.4f}")
        if yco_e is not None:
            parts.append(f"dyCO={k['yCO'] - float(yco_e):.4f}")
        if yh2_e is not None:
            parts.append(f"dyH2={k['yH2'] - float(yh2_e):.4f}")
        if yco2_e is not None:
            parts.append(f"dyCO2={k['yCO2'] - float(yco2_e):.4f}")
        print(f"  diff vs expected: " + "  ".join(parts))
        print()
        json_rows.append(
            {
                "case": name,
                "time_s": elapsed,
                "mesh": mesh_meta,
                "sim": k,
                "expected": {
                    "T_out_C": te,
                    "yCO": yco_e,
                    "yH2": yh2_e,
                    "yCO2": yco2_e,
                },
                "diff": {
                    "dT": (k["T_out_C"] - float(te)) if te is not None else None,
                    "dyCO": (k["yCO"] - float(yco_e)) if yco_e is not None else None,
                    "dyH2": (k["yH2"] - float(yh2_e)) if yh2_e is not None else None,
                    "dyCO2": (k["yCO2"] - float(yco2_e)) if yco2_e is not None else None,
                },
            }
        )

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "solver_method": "jax_pure",
            "n_cells": args.n_cells,
            "fixed_ignition_length": args.fixed_ignition_length,
            "ignition_zone_dz": args.ignition_zone_dz,
            "ignition_zone_stretch_ratio": args.ignition_zone_stretch_ratio,
            "rows": json_rows,
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON written: {out}")


if __name__ == "__main__":
    main()
