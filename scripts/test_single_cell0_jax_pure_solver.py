"""
单 Cell0 求解对比（不使用 pytest）：
baseline：solver_method='minimize'
jax_newton：solver_method='jax_newton'
jax_pure：solver_method='jax_pure'（host float64 残差 + 中心差分 Jacobian，见 jax_solver.newton_solve_cell_pure_jax_ad）

目的：
- 快速验证 jax_pure 是否能正常工作
- 观察输出状态向量（以及 cell0 residual 的缩放后残差）
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict

import numpy as np


def _build_system(case_name: str) -> "GasifierSystem":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(os.path.join(project_root, "src"))

    from model.chemistry import COAL_DATABASE, VALIDATION_CASES
    from model.gasifier_system import GasifierSystem

    case = VALIDATION_CASES[case_name]
    inputs: Dict[str, float] = case["inputs"]

    coal_key = inputs.get("coal") or inputs.get("coal_type")
    coal_props = COAL_DATABASE[coal_key].copy()

    feed_rate_kg_h = inputs["FeedRate"]
    coal_flow_kg_s = feed_rate_kg_h / 3600.0
    o2_flow_kg_s = coal_flow_kg_s * inputs.get("Ratio_OC", 1.0)
    steam_flow_kg_s = coal_flow_kg_s * inputs.get("Ratio_SC", 0.0)

    op_conds = {
        "coal_flow": coal_flow_kg_s,
        "o2_flow": o2_flow_kg_s,
        "steam_flow": steam_flow_kg_s,
        "P": inputs["P"],
        "T_in": inputs["TIN"],
        "HeatLossPercent": inputs.get("HeatLossPercent", 1.0),
        "epsilon": inputs.get("Voidage", 1.0),
        # 将主要离散/动力学分配参数透传（若不存在则保留默认逻辑）
        "Combustion_CO2_Fraction": inputs.get("Combustion_CO2_Fraction", 1.0),
        "WGS_CatalyticFactor": inputs.get("WGS_CatalyticFactor", None),
        "P_O2_Combustion_atm": inputs.get("P_O2_Combustion_atm", 0.05),
    }

    if case_name == "Paper_Case_6":
        # 对齐仓库对比脚本的关键覆盖
        op_conds["SlurryConcentration"] = inputs.get("SlurryConcentration", 62.0)
        op_conds["steam_flow"] = 0.0

    geometry = {"L": inputs.get("L_reactor", 6.0), "D": inputs.get("D_reactor", 2.0)}
    return GasifierSystem(geometry, coal_props, op_conds)


def _print_state_and_residuals(system, x: np.ndarray, label: str) -> None:
    species = ["O2", "CH4", "CO", "CO2", "H2S", "H2", "N2", "H2O"]
    print(f"\n[{label}] 出口状态（Cell0）")
    for i, sp in enumerate(species):
        print(f"  {sp:4}: {x[i]:12.6f} mol/s")
    print(f"  Ws  : {x[8]:12.6f} kg/s")
    print(f"  Xc  : {x[9]:12.6f} (0-1)")
    print(f"  T_g : {x[10]:12.3f} K")

    cell0 = system.cells[0]
    res = cell0.residuals(x)
    print(
        f"\n[{label}] cell0 residuals（缩放后，索引 0-7: 气体, 8:Ws, 9:Xc, 10:E）"
    )
    print(f"  max|res|: {np.max(np.abs(res)):.6e}")
    for i in range(8):
        print(f"  res[{i:02d}] {species[i]:4}: {res[i]: .6e}")
    print(f"  res[08] Ws : {res[8]: .6e}")
    print(f"  res[09] Xc : {res[9]: .6e}")
    print(f"  res[10] E  : {res[10]: .6e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default="Paper_Case_1")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    print(f"=== 单 Cell0 求解对比（含 jax_pure）：case={args.case} ===")

    system_min = _build_system(args.case)
    results_min, _ = system_min.solve(N_cells=1, solver_method="minimize", use_jax_jacobian=False)
    x_min = np.asarray(results_min[0], dtype=np.float64)
    _print_state_and_residuals(system_min, x_min, "baseline=minimize")

    system_jax_newton = _build_system(args.case)
    results_jax_newton, _ = system_jax_newton.solve(N_cells=1, solver_method="jax_newton", jax_warmup=True)
    x_jax_newton = np.asarray(results_jax_newton[0], dtype=np.float64)
    _print_state_and_residuals(system_jax_newton, x_jax_newton, "jax_newton")

    system_jax_pure = _build_system(args.case)
    results_jax_pure, _ = system_jax_pure.solve(N_cells=1, solver_method="jax_pure", jax_warmup=True)
    x_jax_pure = np.asarray(results_jax_pure[0], dtype=np.float64)
    _print_state_and_residuals(system_jax_pure, x_jax_pure, "jax_pure")

    print("\n=== 差值分析（jax_pure - baseline）===")
    print(f"  dT_g : {x_jax_pure[10] - x_min[10]: .6f} K")
    print(f"  dWs  : {x_jax_pure[8] - x_min[8]: .6e} kg/s")
    print(f"  dXc  : {x_jax_pure[9] - x_min[9]: .6e}")
    for i, sp in enumerate(["O2", "CH4", "CO", "CO2", "H2S", "H2", "N2", "H2O"]):
        print(f"  d{sp:4}: {x_jax_pure[i] - x_min[i]: .6e} mol/s")


if __name__ == "__main__":
    main()

