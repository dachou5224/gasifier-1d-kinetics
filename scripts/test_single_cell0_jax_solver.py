"""
单 Cell0 求解器对比脚本（不使用 pytest）。

目的：
1) 用 `solver_method='minimize'` 得到基线单元解；
2) 用 `solver_method='jax_newton'` 得到 JAX Newton（当前实现为 NumPy residual + 中心差分 Jacobian）的单元解；
3) 打印出口状态向量与 cell0 residuals（缩放后的残差向量），并做差值分析。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict

import numpy as np


def _build_system(case_name: str) -> "GasifierSystem":
    # 兼容：脚本从项目根目录运行时也能找到 src
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
        # 传递与 cell 计算相关的调参（若不存在则使用默认值）
        "Combustion_CO2_Fraction": inputs.get("Combustion_CO2_Fraction", 1.0),
        "WGS_CatalyticFactor": inputs.get("WGS_CatalyticFactor", None),
        "P_O2_Combustion_atm": inputs.get("P_O2_Combustion_atm", 0.05),
    }

    # Paper_Case_6：保持与仓库对比脚本一致的关键覆盖
    if case_name == "Paper_Case_6":
        op_conds["SlurryConcentration"] = inputs.get("SlurryConcentration", 62.0)
        op_conds["steam_flow"] = 0.0

    geometry = {
        "L": inputs.get("L_reactor", 6.0),
        "D": inputs.get("D_reactor", 2.0),
    }

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
    print(f"\n[{label}] cell0 residuals（缩放后，索引 0-7: 气体, 8:Ws, 9:Xc, 10:E）")
    print(f"  max|res|: {np.max(np.abs(res)):.6e}")
    for i in range(8):
        print(f"  res[{i:02d}] {species[i]:4}: {res[i]: .6e}")
    print(f"  res[08] Ws : {res[8]: .6e}")
    print(f"  res[09] Xc : {res[9]: .6e}")
    print(f"  res[10] E  : {res[10]: .6e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default="Paper_Case_6")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    print(f"=== 单 Cell0 求解对比：case={args.case} ===")

    system_min = _build_system(args.case)
    results_min, _ = system_min.solve(N_cells=1, solver_method="minimize", use_jax_jacobian=False)
    x_min = np.asarray(results_min[0], dtype=np.float64)
    _print_state_and_residuals(system_min, x_min, "baseline=minimize")

    system_jax = _build_system(args.case)
    results_jax, _ = system_jax.solve(N_cells=1, solver_method="jax_newton", jax_warmup=True)
    x_jax = np.asarray(results_jax[0], dtype=np.float64)
    _print_state_and_residuals(system_jax, x_jax, "jax_newton")

    print("\n=== 差值分析（jax_newton - baseline）===")
    print(f"  dT_g   : {x_jax[10] - x_min[10]: .6f} K")
    print(f"  dWs    : {x_jax[8] - x_min[8]: .6e} kg/s")
    print(f"  dXc    : {x_jax[9] - x_min[9]: .6e}")
    species = ["O2", "CH4", "CO", "CO2", "H2S", "H2", "N2", "H2O"]
    for i, sp in enumerate(species):
        print(f"  d{sp:4}: {x_jax[i] - x_min[i]: .6e} mol/s")

    # quick ignition heuristic
    print("\n=== 起燃判据（T>1200K）===")
    print(f"  baseline: T={x_min[10]:.2f}K -> {'YES' if x_min[10] > 1200 else 'NO'}")
    print(f"  jax_newton: T={x_jax[10]:.2f}K -> {'YES' if x_jax[10] > 1200 else 'NO'}")


if __name__ == "__main__":
    main()

