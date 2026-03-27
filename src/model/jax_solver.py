"""
JAX 辅助的 Newton 外层：在 NumPy 残差上配合中心差分 Jacobian 与可选 vmap 多初值。

完整 lax.scan + 可微残差见后续 jax_cell；此处 focus 工程可用性与 fallback。
"""
from __future__ import annotations

import logging
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .jax_residual_adapter import (
    finite_difference_jacobian_centered,
    finite_difference_jacobian_forward,
)
from .solver import SolverResult

logger = logging.getLogger(__name__)

# 默认与 GasifierSystem 中 jax_pure 分支一致
_JAX_PURE_N_VARS = 11


def _cell_residual_fn_f64(cell):
    """cell.residuals 统一在 float64 上求值（与 NewtonSolver / FD 一致）。"""

    def residual_fn(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return np.asarray(cell.residuals(x), dtype=np.float64)

    return residual_fn


def newton_solve_cell_pure_jax_ad(
    cell,
    x0: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    n_iter: int = 60,
    damper: float = 0.8,
    tol_residual: float = 1e-8,
    fd_epsilon: float = 1e-6,
    reg: float = 1e-8,
):
    """
    solver_method=`jax_pure`：在 **host NumPy** 上对 `cell.residuals` 做中心差分 Jacobian +
    阻尼 Newton（与 `newton_solve_cell_numpy` / `jax_newton` 内核一致）。

    说明：此前用 `make_cell_residuals_jax` + `jax.jacfwd` 会在每个变量上触发多次 `pure_callback`，
    壁钟时间往往比「同一 Python 循环里连续算残差」更慢；因此生产路径改为 host FD + `lstsq`。

    自动微分 / 可微残差接口仍保留在 `jax_cell.make_cell_residuals_jax`（测试与后续 jnp 迁移用）。
    """
    residual_fn = _cell_residual_fn_f64(cell)
    sol = newton_solve_cell_numpy(
        residual_fn,
        x0,
        lower,
        upper,
        n_iter=n_iter,
        damper=damper,
        reg=reg,
        tol_residual=tol_residual,
        epsilon=fd_epsilon,
        jacobian_centered=False,
    )
    msg = sol.message.replace("(jax_solver numpy)", "(jax_pure host FD)")
    return SolverResult(
        sol.x,
        sol.success,
        msg,
        sol.nit,
        sol.nfev,
        cost=sol.cost,
        njev=sol.njev,
    )


def newton_solve_cell_pure_jax_packed_callback(
    cell,
    x0: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    n_iter: int = 60,
    damper: float = 0.8,
    tol_residual: float = 1e-8,
    fd_epsilon: float = 1e-6,
    reg: float = 1e-8,
    n_vars: int = _JAX_PURE_N_VARS,
):
    """
    备选路径：每步 **一次** `pure_callback`，在 host 上同时返回 F 与 J（打包为一维向量），
    再在 JAX 上解 `(J+reg I)Δ=-F`。用于对照/未来 `jit` 化外层；默认生产用 `newton_solve_cell_pure_jax_ad`。
    """
    import jax
    import jax.numpy as jnp

    n = int(n_vars)

    def host_fj(x_flat: np.ndarray) -> np.ndarray:
        x_np = np.asarray(x_flat, dtype=np.float64)
        F = np.asarray(cell.residuals(x_np), dtype=np.float64)
        J = finite_difference_jacobian_centered(
            _cell_residual_fn_f64(cell),
            x_np,
            n_vars=n,
            epsilon=fd_epsilon,
        )
        return np.concatenate([F.ravel(), J.ravel()])

    out_len = n + n * n
    out_shape = jax.ShapeDtypeStruct((out_len,), jnp.float64)
    lower_np = np.asarray(lower, dtype=np.float64)
    upper_np = np.asarray(upper, dtype=np.float64)
    lower_j = jnp.asarray(lower_np, dtype=jnp.float64)
    upper_j = jnp.asarray(upper_np, dtype=jnp.float64)
    x = jnp.asarray(np.asarray(x0, dtype=np.float64), dtype=jnp.float64)

    nfev = 0
    for k in range(n_iter):
        x = jnp.clip(x, lower_j, upper_j)
        packed = jax.pure_callback(host_fj, out_shape, x)
        F = packed[:n]
        J = packed[n:].reshape(n, n)
        nfev += 1 + 2 * n
        max_res = float(jnp.max(jnp.abs(F)))
        cost = float(0.5 * jnp.sum(F * F))
        if max_res < tol_residual:
            x_np = np.asarray(jax.device_get(x), dtype=np.float64)
            return SolverResult(
                x_np,
                True,
                f"Converged in {k} iterations (jax_pure packed callback)",
                k,
                nfev,
                cost=cost,
                njev=k,
            )
        A = J + reg * jnp.eye(n, dtype=J.dtype)
        delta = jnp.linalg.solve(A, -F)
        x_new = jnp.clip(x + damper * delta, lower_j, upper_j)
        step = float(jnp.linalg.norm(x_new - x))
        if step < 1e-14:
            x_np = np.asarray(jax.device_get(x_new), dtype=np.float64)
            F2 = np.asarray(cell.residuals(x_np), dtype=np.float64)
            max_res2 = float(np.max(np.abs(F2)))
            success = max_res2 < tol_residual
            return SolverResult(
                x_np,
                success,
                (
                    f"Stagnation at iter {k} (jax_pure packed callback)"
                    if success
                    else f"Stagnation at iter {k} with residual max|F|={max_res2:.3e} (jax_pure packed callback)"
                ),
                k,
                nfev + 1,
                cost=float(0.5 * np.sum(F2**2)),
                njev=k + 1,
            )
        x = x_new

    x = jnp.clip(x, lower_j, upper_j)
    x_np = np.asarray(jax.device_get(x), dtype=np.float64)
    F_final = np.asarray(cell.residuals(x_np), dtype=np.float64)
    cost = float(0.5 * np.sum(F_final**2))
    return SolverResult(
        x_np,
        False,
        "Max iterations reached (jax_pure packed callback)",
        n_iter,
        nfev,
        cost=cost,
        njev=n_iter,
    )


def newton_solve_multistart_cell_pure_jax_ad(
    cell,
    x0_list: Sequence[np.ndarray],
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    n_iter: int = 60,
    damper: float = 0.8,
    tol_residual: float = 1e-8,
    fd_epsilon: float = 1e-6,
    reg: float = 1e-8,
    ignited_T_threshold: float = 1200.0,
    t_guesses: Optional[Sequence[float]] = None,
):
    """
    对同一个 cell 做多初值 jax_pure（host FD Newton），并选最优。
    """
    # 复用已有选择策略：success 优先 + 已起燃优先 + cost 最小
    candidates: List[Tuple[SolverResult, float]] = []
    for i, x0 in enumerate(x0_list):
        tg = float(t_guesses[i]) if t_guesses is not None and i < len(t_guesses) else float(x0[10])
        sol = newton_solve_cell_pure_jax_ad(
            cell,
            x0,
            lower,
            upper,
            n_iter=n_iter,
            damper=damper,
            tol_residual=tol_residual,
            fd_epsilon=fd_epsilon,
            reg=reg,
        )
        candidates.append((sol, tg))
    best = _select_best_multistart(candidates, ignited_T_threshold=ignited_T_threshold)
    return best, [sol for (sol, _tg) in candidates]


def newton_solve_cell_numpy(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    n_iter: int = 100,
    damper: float = 0.8,
    reg: float = 1e-10,
    tol_residual: float = 1e-9,
    epsilon: float = 1e-8,
    jacobian_centered: bool = True,
) -> SolverResult:
    """
    阻尼 Newton（NumPy），Jacobian 为中心差分；线性步用 lstsq 与 NewtonSolver 一致。
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    n = x.shape[0]
    nfev = 0
    for k in range(n_iter):
        x = np.clip(x, lower, upper)
        F = np.asarray(residual_fn(x), dtype=np.float64)
        nfev += 1
        max_res = float(np.max(np.abs(F)))
        cost = float(0.5 * np.sum(F**2))
        if max_res < tol_residual:
            return SolverResult(
                x,
                True,
                f"Converged in {k} iterations (jax_solver numpy)",
                k,
                nfev,
                cost=cost,
                njev=k,
            )
        if jacobian_centered:
            J = finite_difference_jacobian_centered(
                residual_fn, x, n_vars=n, epsilon=epsilon
            )
        else:
            J = finite_difference_jacobian_forward(
                residual_fn, x, n_vars=n, epsilon=epsilon
            )
        try:
            # 对齐 NewtonSolver：解 J * delta = -F（用 lstsq 处理奇异/病态）
            # reg 主要用于极端病态时的后备策略
            delta, _, _, _ = np.linalg.lstsq(J, -F, rcond=None)
        except np.linalg.LinAlgError:
            if reg > 0.0:
                # 后备：用正则化法方程做一次稳定求解
                JTJ = J.T @ J
                rhs = -J.T @ F
                delta, _, _, _ = np.linalg.lstsq(JTJ + reg * np.eye(n), rhs, rcond=None)
            else:
                return SolverResult(
                    x,
                    False,
                    f"Singular Jacobian at iter {k}",
                    k,
                    nfev,
                    cost=cost,
                    njev=k,
                )
        nfev += 2 * n  # centered: 2 evals per column, counted approximately in njev as 1
        x_new = np.clip(x + damper * delta, lower, upper)
        if np.linalg.norm(x_new - x) < 1e-14:
            F_new = np.asarray(residual_fn(x_new), dtype=np.float64)
            nfev += 1
            max_res_new = float(np.max(np.abs(F_new)))
            cost_new = float(0.5 * np.sum(F_new**2))
            if max_res_new < tol_residual:
                return SolverResult(
                    x_new,
                    True,
                    f"Stagnation at iter {k}",
                    k,
                    nfev,
                    cost=cost_new,
                    njev=k + 1,
                )
            return SolverResult(
                x_new,
                False,
                f"Stagnation at iter {k} with residual max|F|={max_res_new:.3e}",
                k,
                nfev,
                cost=cost_new,
                njev=k + 1,
            )
        x = x_new

    F_final = np.asarray(residual_fn(x), dtype=np.float64)
    nfev += 1
    cost = float(0.5 * np.sum(F_final**2))
    return SolverResult(
        x,
        False,
        "Max iterations reached (jax_solver numpy)",
        n_iter,
        nfev,
        cost=cost,
        njev=n_iter,
    )


def _select_best_multistart(
    candidates: Sequence[Tuple[SolverResult, float]],
    ignited_T_threshold: float = 1200.0,
) -> Optional[SolverResult]:
    """与 gasifier_system Cell0 逻辑对齐：优先已起燃，其次 cost，略高温度优先。"""
    best: Optional[SolverResult] = None
    best_cost = 1e30
    best_success = False
    for sol, _t_guess in candidates:
        if sol is None:
            continue
        is_ignited = float(sol.x[10]) > ignited_T_threshold
        if best is None:
            best = sol
            best_cost = sol.cost
            best_success = bool(sol.success)
            continue
        best_ignited = float(best.x[10]) > ignited_T_threshold
        sol_success = bool(sol.success)
        # 1) 总体优先 success=true
        if sol_success and not best_success:
            best = sol
            best_cost = sol.cost
            best_success = sol_success
            continue
        # 2) 同 success 状态下：优先已起燃，再比 cost
        if is_ignited and not best_ignited:
            best = sol
            best_cost = sol.cost
        elif is_ignited == best_ignited:
            if sol.cost < best_cost:
                best = sol
                best_cost = sol.cost
            elif abs(sol.cost - best_cost) < best_cost * 0.1 and sol.x[10] > best.x[10]:
                best = sol
                best_cost = sol.cost
    return best


def newton_solve_multistart_numpy(
    x0_list: Sequence[np.ndarray],
    residual_fn: Callable[[np.ndarray], np.ndarray],
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    n_iter: int = 100,
    damper: float = 0.8,
    reg: float = 1e-10,
    tol_residual: float = 1e-9,
    epsilon: float = 1e-8,
    jacobian_centered: bool = True,
    ignited_T_threshold: float = 1200.0,
    t_guesses: Optional[Sequence[float]] = None,
) -> Tuple[Optional[SolverResult], List[SolverResult]]:
    """
    对多个初值顺序求解并选最优。返回 (best, all_results)。
    """
    results: List[SolverResult] = []
    meta: List[Tuple[SolverResult, float]] = []
    for i, x0 in enumerate(x0_list):
        tg = float(t_guesses[i]) if t_guesses is not None and i < len(t_guesses) else float(x0[10])
        sol = newton_solve_cell_numpy(
            residual_fn,
            x0,
            lower,
            upper,
            n_iter=n_iter,
            damper=damper,
            reg=reg,
            tol_residual=tol_residual,
            epsilon=epsilon,
            jacobian_centered=jacobian_centered,
        )
        results.append(sol)
        meta.append((sol, tg))
    best = _select_best_multistart(meta, ignited_T_threshold=ignited_T_threshold)
    return best, results


def newton_solve_cell_lax_scan(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    n_iter: int = 100,
    damper: float = 0.8,
    reg: float = 1e-10,
    tol_residual: float = 1e-9,
    epsilon: float = 1e-8,
) -> SolverResult:
    """
    lax.scan 包装的阻尼 Newton：每步在 host 上算 F 与中心差分 J（与纯 NumPy 路径一致）。
    用于验证 XLA 外层 + pure_callback 组合；默认求解仍用 newton_solve_cell_numpy。
    """
    import jax
    import jax.numpy as jnp
    from jax import lax

    lower_np = np.asarray(lower, dtype=np.float64)
    upper_np = np.asarray(upper, dtype=np.float64)
    x0 = np.asarray(x0, dtype=np.float64)
    n = x0.shape[0]

    def host_step(x_flat: np.ndarray) -> np.ndarray:
        x_np = np.asarray(x_flat, dtype=np.float64)
        F = np.asarray(residual_fn(x_np), dtype=np.float64)
        J = finite_difference_jacobian_centered(
            residual_fn, x_np, n_vars=n, epsilon=epsilon
        )
        # 对齐 NewtonSolver：解 J * delta = -F
        delta, _, _, _ = np.linalg.lstsq(J, -F, rcond=None)
        x_new = np.clip(x_np + damper * delta, lower_np, upper_np)
        mx = float(np.max(np.abs(F)))
        return np.concatenate([x_new, np.array([mx], dtype=np.float64)])

    def body(carry, _):
        out = jax.pure_callback(
            host_step,
            jax.ShapeDtypeStruct((n + 1,), jnp.float64),
            carry,
        )
        return out[:n], out[n]

    x0_j = jnp.asarray(x0, dtype=jnp.float64)
    x_final, _mx_stack = lax.scan(body, x0_j, None, length=n_iter)
    x_out = np.asarray(jax.device_get(x_final), dtype=np.float64)
    F = np.asarray(residual_fn(x_out), dtype=np.float64)
    mx = float(np.max(np.abs(F)))
    cost = float(0.5 * np.sum(F**2))
    success = mx < tol_residual
    return SolverResult(
        x_out,
        success,
        "lax_scan newton (jax_solver)",
        n_iter,
        n_iter * (1 + 2 * n),
        cost=cost,
        njev=n_iter,
    )


def warmup_jax() -> None:
    """触发 JAX 导入与一次轻量计算，摊销首次编译。"""
    try:
        import jax
        import jax.numpy as jnp

        jax.numpy.ones((2, 2))
        _ = jnp.sum(jnp.arange(4.0).reshape(2, 2))
    except Exception as e:
        logger.debug("JAX warmup skipped: %s", e)
