"""
JAX Cell 残差工厂（阶段 3：可用于 `jax.jacfwd` 的残差）。

说明：
- `cell.residuals` 目前是 NumPy 计算，包含大量 Python/NumPy 逻辑，无法直接被 JAX trace。
- 本实现用 `jax.pure_callback` 把 NumPy 残差“封装”为 JAX 可调用的函数；
- 并用 `custom_jvp` 为其提供 JVP（通过有限差分在 host 端计算），从而让 `jax.jacfwd` 可工作。

这仍然是“可微接口迁移”，不是完全纯 jnp 迁移（阶段 3 的最终极致版本需要把 cell/kinetics/thermo 全改成 jnp + lax）。
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def make_cell_residuals_jax(
    cell,
    *,
    out_size: int = 11,
    fd_epsilon: float = 1e-6,
):
    """
    为某个 `Cell` 实例构造一个 JAX 残差函数 residuals_jax(x)->(out_size,)

    返回的 residuals_jax 支持：
    - 直接前向调用：`residuals_jax(x)`（JAX tensor 输出）
    - 自动求导：`jax.jacfwd(residuals_jax)(x)`（依赖 custom_jvp 的有限差分 JVP）
    """
    import jax
    import jax.numpy as jnp

    # 兼容默认 `jax_enable_x64=False`：不要在 pure_callback 的结果 shape 中声明 float64
    out_dtype = jnp.float32
    out_shape = jax.ShapeDtypeStruct((out_size,), out_dtype)

    def residuals_host(x_np: np.ndarray) -> np.ndarray:
        # cell.residuals 输入为一维 NumPy 数组，输出为残差向量（NumPy）。
        y = cell.residuals(np.asarray(x_np, dtype=np.float64))
        return np.asarray(y, dtype=np.float32)

    @jax.custom_jvp
    def residuals_jax(x):
        x = jnp.asarray(x, dtype=out_dtype)
        return jax.pure_callback(residuals_host, out_shape, x)

    @residuals_jax.defjvp
    def residuals_jax_jvp(primals, tangents):
        (x,) = primals
        (x_dot,) = tangents

        y = residuals_jax(x)

        # 方向导数：数值 JVP
        # 使用与 finite difference Jacobian 一致的相对扰动步长：
        #   dx_i = fd_epsilon * max(|x_i|, 1) * x_dot_i
        # 并用 dx 的最大绝对值作为标量分母（在 jacfwd 的 e_i 情况下等价于 2*dx_i）。
        scale = jnp.maximum(jnp.abs(x), 1.0)
        dx = fd_epsilon * scale * x_dot
        x_plus = x + dx
        x_minus = x - dx

        y_plus = jax.pure_callback(residuals_host, out_shape, x_plus)
        y_minus = jax.pure_callback(residuals_host, out_shape, x_minus)

        den = 2.0 * jnp.max(jnp.abs(dx))
        den_safe = jnp.where(den == 0.0, 1.0, den)  # 切掉 0/0（x_dot 全 0 时 numerator 也为 0）
        jvp = (y_plus - y_minus) / den_safe
        return y, jvp

    return residuals_jax
