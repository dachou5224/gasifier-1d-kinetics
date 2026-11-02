"""
Small internal helpers for implicit-function-theorem gradients on JAX solves.

This module is intentionally not wired into the reactor solver yet.  It provides
the narrow reusable surface needed by tests and future single-cell experiments:
one scalar parameter, one residual system, and one fixed-branch solve function.
"""
from __future__ import annotations


def _root_guard_mask(residual, root_tol):
    """与 residual_cost 同一缩放：‖F‖∞ / max(1, ‖F‖₂)。返回「是否为根」的布尔数组。

    trace 安全：不做 Python 标量转换，可直接用于 ``jnp.where``。
    """
    import jax.numpy as jnp

    r = jnp.asarray(residual)
    norm = jnp.max(jnp.abs(r)) / jnp.maximum(1.0, jnp.linalg.norm(r))
    return jnp.isfinite(norm) & (norm <= root_tol)


def make_single_parameter_ift_solve(residual_fn, solve_fn, *, root_tol: float = 1e-6):
    """
    Return a custom-VJP solve(theta) for a square implicit system F(x, theta)=0.

    Args:
        residual_fn: callable ``residual_fn(x, theta) -> (n,)``.
        solve_fn: callable ``solve_fn(theta) -> (n,)``. The solve is assumed to
            stay on a fixed branch and return a sufficiently converged state.

    Backward pass:
        F_x.T @ lambda = x_bar
        theta_bar = -lambda.T @ F_theta

    This helper supports one scalar differentiable parameter. It deliberately
    avoids reactor-level scan, multistart branch selection, or public solver API
    changes.
    """
    import jax
    import jax.numpy as jnp

    @jax.custom_vjp
    def solve_ift(theta):
        return solve_fn(theta)

    def solve_ift_fwd(theta):
        x_sol = solve_fn(theta)
        return x_sol, (x_sol, theta)

    def solve_ift_bwd(saved, x_bar):
        x_sol, theta = saved
        # 根守卫（P1.5-B B7）：前向解不是根时返回 NaN，不给无意义的梯度。
        # 用 jnp.where 而不是 Python 分支，保证在 scan / vmap / jit 下仍可 trace。
        is_root = _root_guard_mask(residual_fn(x_sol, theta), root_tol)
        F_x = jax.jacfwd(lambda x: residual_fn(x, theta))(x_sol)
        F_theta = jax.jacfwd(lambda t: residual_fn(x_sol, t))(theta)
        lam = jnp.linalg.solve(F_x.T, x_bar)
        theta_bar = -jnp.vdot(lam, F_theta)
        return (jnp.where(is_root, theta_bar, jnp.nan),)

    solve_ift.defvjp(solve_ift_fwd, solve_ift_bwd)
    return solve_ift


def make_single_parameter_ift_solve_with_args(residual_fn, solve_fn, *, root_tol: float = 1e-6):
    """
    Return custom-VJP ``solve(theta, *args)`` for F(x, theta, *args)=0.

    This is the scan-friendly variant of ``make_single_parameter_ift_solve``:
    cotangents for ``*args`` are propagated with the same IFT linear solve, so
    JAX control-flow primitives such as ``lax.scan`` can chain gradients through
    cell-to-cell carry values.
    """
    import jax
    import jax.numpy as jnp

    @jax.custom_vjp
    def solve_ift(theta, *args):
        return solve_fn(theta, *args)

    def solve_ift_fwd(theta, *args):
        x_sol = solve_fn(theta, *args)
        return x_sol, (x_sol, theta, args)

    def solve_ift_bwd(saved, x_bar):
        x_sol, theta, args = saved
        is_root = _root_guard_mask(residual_fn(x_sol, theta, *args), root_tol)
        F_x = jax.jacfwd(lambda x: residual_fn(x, theta, *args))(x_sol)
        lam = jnp.linalg.solve(F_x.T, x_bar)
        _, pullback = jax.vjp(lambda t, *a: residual_fn(x_sol, t, *a), theta, *args)
        grads = pullback(-lam)
        return tuple(jnp.where(is_root, g, jnp.nan) for g in grads)

    solve_ift.defvjp(solve_ift_fwd, solve_ift_bwd)
    return solve_ift


__all__ = ["make_single_parameter_ift_solve", "make_single_parameter_ift_solve_with_args"]
