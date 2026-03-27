"""
JAX 求解路径与基线数值对齐（Paper_Case_6，N_cells=20）。
"""
import json
import logging
import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from model.chemistry import COAL_DATABASE
from model.gasifier_system import GasifierSystem

logging.basicConfig(level=logging.ERROR)


def _load_case(case_name: str):
    json_path = os.path.join(os.path.dirname(__file__), "../../data/validation_cases.json")
    with open(json_path, "r") as f:
        config = json.load(f)
    if "cases" in config and case_name in config["cases"]:
        return config["cases"][case_name]
    from model.chemistry import VALIDATION_CASES
    return VALIDATION_CASES[case_name]


def _build_system(case_name: str) -> GasifierSystem:
    case_data = _load_case(case_name)
    inputs = case_data["inputs"]
    coal_key = case_data.get("coal_type", inputs.get("coal"))
    coal_props = COAL_DATABASE[coal_key].copy()
    feed_rate = inputs.get("FeedRate_kg_h", inputs.get("FeedRate"))
    coal_flow_kg_s = feed_rate / 3600.0
    o2_flow_kg_s = coal_flow_kg_s * inputs.get("Ratio_OC", 1.0)
    steam_flow_kg_s = coal_flow_kg_s * inputs.get("SteamRatio_SC", inputs.get("Ratio_SC", 0.0))
    op_conds = {
        "coal_flow": coal_flow_kg_s,
        "o2_flow": o2_flow_kg_s,
        "steam_flow": steam_flow_kg_s,
        "P": inputs.get("P_Pa", inputs.get("P")),
        "T_in": inputs.get("T_in_K", inputs.get("TIN")),
        "HeatLossPercent": inputs.get("HeatLossPercent", 1.0),
        "epsilon": inputs.get("Voidage", 1.0),
    }
    if case_name == "Paper_Case_6":
        op_conds["SlurryConcentration"] = inputs.get("SlurryConcentration", 62.0)
        op_conds["steam_flow"] = 0.0
    geometry = {
        "L": inputs.get("L_reactor", 6.0),
        "D": inputs.get("D_reactor", 2.0),
    }
    return GasifierSystem(geometry, coal_props, op_conds)


def test_jax_jacobian_matches_baseline_minimize():
    sys_b = _build_system("Paper_Case_6")
    res_base, z_b = sys_b.solve(N_cells=20, solver_method="minimize", use_jax_jacobian=False)
    sys_j = _build_system("Paper_Case_6")
    res_j, z_j = sys_j.solve(N_cells=20, solver_method="minimize", use_jax_jacobian=True)
    assert z_b.shape == z_j.shape
    assert np.max(np.abs(res_base[:, 10] - res_j[:, 10])) < 5.0
    assert np.max(np.abs(res_base[:, :8] - res_j[:, :8])) < 0.15


def test_jax_pure_reasonable_vs_minimize():
    """jax_pure（host FD Newton）应与 minimize 全炉温度/流量同量级。"""
    sys_m = _build_system("Paper_Case_6")
    res_m, _ = sys_m.solve(N_cells=20, solver_method="minimize", use_jax_jacobian=False)
    sys_p = _build_system("Paper_Case_6")
    res_p, _ = sys_p.solve(N_cells=20, solver_method="jax_pure", jax_warmup=True)
    assert np.max(np.abs(res_m[:, 10] - res_p[:, 10])) < 20.0
    assert np.max(np.abs(res_m[:, :8] - res_p[:, :8])) < 0.6


def test_jax_newton_reasonable_vs_minimize():
    sys_m = _build_system("Paper_Case_6")
    res_m, _ = sys_m.solve(N_cells=20, solver_method="minimize", use_jax_jacobian=False)
    sys_x = _build_system("Paper_Case_6")
    res_x, _ = sys_x.solve(N_cells=20, solver_method="jax_newton", jax_warmup=True)
    assert np.max(np.abs(res_m[:, 10] - res_x[:, 10])) < 15.0
    assert np.max(np.abs(res_m[:, :8] - res_x[:, :8])) < 0.5


def test_jax_kinetics_wgs_matches_numpy():
    from model.kinetics import calculate_wgs_equilibrium
    jnp = pytest.importorskip("jax.numpy")

    from model.jax_kinetics import wgs_equilibrium_k

    T = 1500.0
    k_np = calculate_wgs_equilibrium(T)
    k_jax = float(wgs_equilibrium_k(jnp.array(T)))
    assert abs(k_np - k_jax) < 1e-7


def test_jax_cell_residuals_jacfwd_matches_fd():
    """
    验证 `jax.jacfwd`（custom_jvp 提供 JVP）与基于 `cell.residuals` 的中心差分 Jacobian 一致性。
    只跑 N_cells=1 保持测试开销可控。
    """
    import jax
    import jax.numpy as jnp

    from model.jax_cell import make_cell_residuals_jax

    sys = _build_system("Paper_Case_6")
    # 先用原 solve 得到一个收敛的 cell 状态（避免在未收敛区域比较导数误差）。
    res, _ = sys.solve(N_cells=1, solver_method="minimize", use_jax_jacobian=False)
    cell = sys.cells[0]

    x0 = np.asarray(res[0], dtype=np.float64)
    residuals_jax = make_cell_residuals_jax(cell, out_size=11, fd_epsilon=1e-6)

    # 与 jax_cell 的 float32 回调约束对齐：输入与输出都用 float32
    x0_f32 = x0.astype(np.float32)
    x0_jax = jnp.asarray(x0_f32, dtype=jnp.float32)
    J_jax = np.asarray(jax.jacfwd(residuals_jax)(x0_jax), dtype=np.float32)

    # 同步有限差分（与 custom_jvp 的扰动策略一致）
    fd_eps = 1e-6
    F0 = np.asarray(cell.residuals(x0_f32.astype(np.float64)), dtype=np.float32)
    n = x0_f32.shape[0]
    m = F0.shape[0]
    J_fd = np.zeros((m, n), dtype=np.float32)
    for i in range(n):
        step = np.float32(fd_eps * max(abs(float(x0_f32[i])), 1.0))
        x_plus = x0_f32.copy()
        x_minus = x0_f32.copy()
        x_plus[i] += step
        x_minus[i] -= step
        Fp = np.asarray(cell.residuals(x_plus.astype(np.float64)), dtype=np.float32)
        Fm = np.asarray(cell.residuals(x_minus.astype(np.float64)), dtype=np.float32)
        J_fd[:, i] = (Fp - Fm) / (np.float32(2.0) * step)

    # 允许一定误差（导数本就对扰动步长敏感）
    max_abs_diff = float(np.max(np.abs(J_jax - J_fd)))
    assert max_abs_diff < 5e-2
