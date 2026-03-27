"""
残差 Jacobian 适配：在 cell.residuals 仍为 NumPy 时，为 SciPy / Newton 提供高质量 Jacobian。

完整 JAX 自动微分（jacfwd 机器精度）需要残差为 jnp 可 trace；见 jax_kinetics 与后续 jax_cell。
此处采用中心差分，显著优于原 NewtonSolver 前向差分（每列一次扰动）的数值稳定性。
"""
from __future__ import annotations

import numpy as np


def finite_difference_jacobian_forward(
    residuals_fn,
    x: np.ndarray,
    n_vars: int | None = None,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """前向有限差分 Jacobian (m x n)，与经典 NewtonSolver 一致。"""
    x = np.asarray(x, dtype=np.float64).copy()
    n = x.shape[0] if n_vars is None else int(n_vars)
    F0 = np.asarray(residuals_fn(x), dtype=np.float64)
    m = F0.shape[0]
    J = np.zeros((m, n), dtype=np.float64)
    for j in range(n):
        step = epsilon * max(abs(x[j]), 1.0)
        x_p = x.copy()
        x_p[j] += step
        Fp = np.asarray(residuals_fn(x_p), dtype=np.float64)
        J[:, j] = (Fp - F0) / step
    return J


def finite_difference_jacobian_centered(
    residuals_fn,
    x: np.ndarray,
    n_vars: int | None = None,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """中心差分 Jacobian；每列 2 次残差评估，精度优于前向差分。"""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0] if n_vars is None else int(n_vars)
    m = np.asarray(residuals_fn(x), dtype=np.float64).shape[0]
    J = np.zeros((m, n), dtype=np.float64)
    for j in range(n):
        step = epsilon * max(abs(x[j]), 1.0)
        x_p = x.copy()
        x_m = x.copy()
        x_p[j] += 0.5 * step
        x_m[j] -= 0.5 * step
        Fp = np.asarray(residuals_fn(x_p), dtype=np.float64)
        Fm = np.asarray(residuals_fn(x_m), dtype=np.float64)
        J[:, j] = (Fp - Fm) / step
    return J


def make_jacobian_fn(residuals_fn, n_vars: int = 11, centered: bool = True):
    """
    返回 SciPy least_squares 可用的 jac(x) -> (m, n) ndarray。
    centered=True 时使用中心差分（推荐配合 use_jax_jacobian）。
    """

    def jac(x):
        x = np.asarray(x, dtype=np.float64)
        if centered:
            return finite_difference_jacobian_centered(residuals_fn, x, n_vars=n_vars)
        return finite_difference_jacobian_forward(residuals_fn, x, n_vars=n_vars)

    return jac
