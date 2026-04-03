"""
整段反应器（多 Cell）求解对比脚本（不使用 pytest）。

对比：
1) baseline：`solver_method='minimize'`（现有基线）
2) fd_path：`solver_method='newton_fd'`

打印：
- 运行耗时
- 出口（最后一个 cell）状态对比：T、各气体 mol/s、Ws、Xc
- profile 对比：各 cell 的 max|Δ|（T、CO/CO2/H2 等），以及 ignition 判据
- 若差异较大，打印若干采样点（z/T/CO%）
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, Tuple

import numpy as np


def _build_system(case_name: str) -> "GasifierSystem":
    # 兼容从项目根目录/任意工作目录运行
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
        # 传递一些 cell/residual 相关调参（若未设置则用默认值由模型处理）
        "Combustion_CO2_Fraction": inputs.get("Combustion_CO2_Fraction", 1.0),
        "WGS_CatalyticFactor": inputs.get("WGS_CatalyticFactor", None),
        "P_O2_Combustion_atm": inputs.get("P_O2_Combustion_atm", 0.05),
        "SlurryConcentration": inputs.get("SlurryConcentration", 100.0),
    }

    # Paper_Case_6：与历史基准一致的关键覆盖
    if case_name == "Paper_Case_6":
        op_conds["steam_flow"] = 0.0
        # 与现有仓库对比脚本保持一致：Paper_Case_6 设为 62% 固含，并禁用外加 steam
        op_conds["SlurryConcentration"] = 62.0

    geometry = {
        "L": inputs.get("L_reactor", 6.0),
        "D": inputs.get("D_reactor", 2.0),
    }

    return GasifierSystem(geometry, coal_props, op_conds)


def _y_of(profile: np.ndarray, idx: int) -> np.ndarray:
    """profile: (N,11)，idx: 气体组分索引 0-7，返回 y_i（%）按 mol fraction 计算。"""
    gas = profile[:, :8]
    # 干基：与 gasifier_kinetic_ui.py 保持一致（忽略 H2O，使用 gas[:7]）
    denom = gas[:, :7].sum(axis=1)
    denom = np.where(denom <= 0, 1e-12, denom)
    return (gas[:, idx] / denom) * 100.0


def _print_exit(profile: np.ndarray, z_last: float, label: str) -> None:
    species = ["O2", "CH4", "CO", "CO2", "H2S", "H2", "N2", "H2O"]
    x = profile[-1]
    print(f"\n[{label}] 出口（z={z_last:.4f} m）")
    for i, sp in enumerate(species):
        print(f"  {sp:4}: {x[i]:12.6f} mol/s")
    print(f"  Ws  : {x[8]:12.6f} kg/s")
    print(f"  Xc  : {x[9]:12.6f}")
    print(f"  T_g : {x[10]:12.3f} K")


def _print_profile_diff(base: np.ndarray, jax: np.ndarray, z: np.ndarray) -> None:
    species = ["O2", "CH4", "CO", "CO2", "H2S", "H2", "N2", "H2O"]
    d = jax - base
    print("\n=== Profile 误差（jax - baseline）===")
    print(f"  max|ΔT|   : {np.max(np.abs(d[:,10])):.6f} K")
    for sp, idx in [("CO", 2), ("CO2", 3), ("H2", 5), ("CH4", 1)]:
        y_base = _y_of(base, idx)
        y_jax = _y_of(jax, idx)
        print(f"  max|Δy_{sp}|: {np.max(np.abs(y_jax - y_base)):.6f} %")

    # ignition 判据
    ign_th = 1200.0
    base_ign = np.where(base[:, 10] > ign_th)[0]
    jax_ign = np.where(jax[:, 10] > ign_th)[0]
    base_first = int(base_ign[0]) if base_ign.size else None
    jax_first = int(jax_ign[0]) if jax_ign.size else None
    print(f"  ignition_first_cell (T>{ign_th}K): baseline={base_first}, newton_fd={jax_first}")

    # 采样点展示：每 N/5 左右一个点，避免输出过多
    n = base.shape[0]
    sample_idx = sorted(set(np.linspace(0, n - 1, num=min(6, n), dtype=int).tolist()))
    print("\n=== 采样点（baseline vs newton_fd）===")
    for i in sample_idx:
        print(
            f"  cell{i:02d} z={z[i]:.4f}m | "
            f"T {base[i,10]:.1f}->{jax[i,10]:.1f} | "
            f"y_CO {y_base[i]:.2f}%->{y_jax[i]:.2f}% | "
            f"CO {base[i,2]:.2f}->{jax[i,2]:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default="Paper_Case_6")
    parser.add_argument("--N_cells", type=int, default=20)
    parser.add_argument("--baseline", type=str, default="minimize", choices=["minimize", "newton"])
    parser.add_argument("--jax_warmup", action="store_true")
    parser.add_argument("--log-level", type=str, default="ERROR")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.ERROR))

    print(f"=== 整段反应器求解对比：case={args.case}, N_cells={args.N_cells}, baseline={args.baseline} ===")

    sys_b = _build_system(args.case)
    t0 = __import__("time").time()
    base_profile, z_b = sys_b.solve(N_cells=args.N_cells, solver_method=args.baseline, jacobian_mode="scipy", jax_warmup=False)
    t_base = __import__("time").time() - t0

    sys_j = _build_system(args.case)
    t1 = __import__("time").time()
    jax_profile, z_j = sys_j.solve(
        N_cells=args.N_cells,
        solver_method="newton_fd",
        jacobian_mode="centered_fd",
        jax_warmup=args.jax_warmup,
    )
    t_jax = __import__("time").time() - t1

    print(f"\n[Timing] baseline={t_base:.2f}s, newton_fd={t_jax:.2f}s")
    if not np.allclose(z_b, z_j):
        print("[WARN] z_positions 不一致（可能是 grid 策略/浮点误差导致）。")

    _print_exit(base_profile, float(z_b[-1]), f"baseline={args.baseline}")
    _print_exit(jax_profile, float(z_j[-1]), "newton_fd")

    # 最后一个 cell 残差质量（用于判断求解器收敛质量）
    # 目前 GasifierSystem 可能不暴露 cells 列表，因此这里做容错跳过。
    if hasattr(sys_b, "cells") and hasattr(sys_j, "cells") and getattr(sys_b, "cells") and getattr(sys_j, "cells"):
        cell_b = sys_b.cells[-1]
        cell_j = sys_j.cells[-1]
        res_b = cell_b.residuals(base_profile[-1])
        res_j = cell_j.residuals(jax_profile[-1])
        print(f"\n[Final cell residuals quality]")
        print(f"  baseline={args.baseline}: max|res|={np.max(np.abs(res_b)):.6e}")
        print(f"  newton_fd             : max|res|={np.max(np.abs(res_j)):.6e}")
    else:
        print(f"\n[Final cell residuals quality] 跳过：GasifierSystem 未提供 cells 列表")

    _print_profile_diff(base_profile, jax_profile, z_b)


if __name__ == "__main__":
    main()
