"""
JAX 求解器实现 (Phase 4: 统一 9 组分 12 维状态 + 硫平衡修复版)。
"""
from __future__ import annotations

import logging
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from functools import partial

try:
    import jax

    # 全炉工业工况摩尔流量与焓量级大，float32 易溢出为 NaN；必须在首次 import jax.numpy 前开启
    jax.config.update("jax_enable_x64", True)

    import jax.numpy as jnp
    from jax import jit, jacfwd, lax, vmap

    JAX_AVAILABLE = True
except Exception:
    jax = None
    jnp = np
    JAX_AVAILABLE = False

    def jit(fn=None, **_kwargs):
        if fn is None:
            return lambda real_fn: real_fn
        return fn

    def jacfwd(_fn):
        raise RuntimeError("JAX is unavailable")

    class _UnavailableLax:
        def __getattr__(self, _name):
            raise RuntimeError("JAX is unavailable")

    lax = _UnavailableLax()

    def vmap(_fn):
        raise RuntimeError("JAX is unavailable")

from .jax_residual_adapter import (
    finite_difference_jacobian_centered,
    finite_difference_jacobian_forward,
)
from .solver import SolverResult

if JAX_AVAILABLE:
    from .jax_residuals import (
        cell_residuals_jax_flat,
        _calc_particle_temperature_jax,
        PARTICLE_DENSITY,
        EF_PARTICLE_TRANSIENT,
        STEFAN_BOLTZMANN,
        CONDUT_COEFF,
        TS_TRANSIENT_NC,
    )

logger = logging.getLogger(__name__)

_JAX_PURE_N_VARS = 11


def _cell_residual_fn_f64(cell):
    """cell.residuals 统一在 float64 上求值。"""

    def residual_fn(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return np.asarray(cell.residuals(x), dtype=np.float64)

    return residual_fn


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
    """阻尼 Newton（NumPy），作为当前 `newton_fd` 路径的核心实现。"""
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
            J = finite_difference_jacobian_centered(residual_fn, x, n_vars=n, epsilon=epsilon)
        else:
            J = finite_difference_jacobian_forward(residual_fn, x, n_vars=n, epsilon=epsilon)
        try:
            delta, _, _, _ = np.linalg.lstsq(J, -F, rcond=None)
        except np.linalg.LinAlgError:
            if reg > 0.0:
                JTJ = J.T @ J
                rhs = -J.T @ F
                delta, _, _, _ = np.linalg.lstsq(JTJ + reg * np.eye(n), rhs, rcond=None)
            else:
                return SolverResult(x, False, f"Singular Jacobian at iter {k}", k, nfev, cost=cost, njev=k)
        nfev += 2 * n
        x_new = np.clip(x + damper * delta, lower, upper)
        if np.linalg.norm(x_new - x) < 1e-14:
            F_new = np.asarray(residual_fn(x_new), dtype=np.float64)
            nfev += 1
            max_res_new = float(np.max(np.abs(F_new)))
            cost_new = float(0.5 * np.sum(F_new**2))
            if max_res_new < tol_residual:
                return SolverResult(x_new, True, f"Stagnation at iter {k}", k, nfev, cost=cost_new, njev=k + 1)
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
    return SolverResult(x, False, "Max iterations reached (jax_solver numpy)", n_iter, nfev, cost=cost, njev=n_iter)


def _select_best_multistart(
    candidates: Sequence[Tuple[SolverResult, float]],
    ignited_T_threshold: float = 1200.0,
) -> Optional[SolverResult]:
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
        if sol_success and not best_success:
            best = sol
            best_cost = sol.cost
            best_success = sol_success
            continue
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
    msg = sol.message.replace("(jax_solver numpy)", "(newton_fd host FD)")
    return SolverResult(sol.x, sol.success, msg, sol.nit, sol.nfev, cost=sol.cost, njev=sol.njev)


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

def _newton_loop_12(res_fn, x0, n_iter, damper, tol, reg):
    jac_fn = jacfwd(res_fn)
    lower = jnp.concatenate([jnp.zeros(11), jnp.array([300.0])])
    upper = jnp.concatenate([jnp.ones(9)*1e6, jnp.array([1e6, 1.0, 4000.0])])
    
    def body_fn(carry, _):
        x, _, converged = carry
        F = res_fn(x); J = jac_fn(x)
        J_reg = J + reg * jnp.eye(12)
        delta = jnp.linalg.solve(J_reg, -F)
        x_new = jnp.clip(x + damper * delta, lower, upper)
        max_res = jnp.max(jnp.abs(F))
        return (x_new, max_res, max_res < tol), max_res

    initial_carry = (x0, 1e10, False)
    (x_final, final_res, converged), _ = lax.scan(body_fn, initial_carry, None, length=n_iter)
    return x_final, converged, final_res

def _solve_vmap_12(res_fn, x0_list, n_iter, damper, tol, reg):
    def solve_one(x):
        return _newton_loop_12(res_fn, x, n_iter, damper, tol, reg)

    x_sols, converged, res_norms = vmap(solve_one)(x0_list)
    T_vals = x_sols[:, 11]
    # 气化炉物理：优先「已点燃」支路（T>1200）中残差最小者；避免冷态假收敛因残差更小胜出
    ignited = T_vals > 1200.0
    score = jnp.where(ignited, res_norms, 1e6 + res_norms)
    best_idx = jnp.argmin(score)
    return x_sols[best_idx], converged[best_idx], res_norms[best_idx]


def _solve_cell0_preferring_seed(res_fn, cell0_guesses, n_iter, damper, tol, reg):
    """
    首格优先使用 minimize 派生 seed（若调用方已将其放在 guesses[0]）；
    仅当该 seed 未收敛时，才回退到多起点扫描。
    """
    seed_sol, seed_conv, seed_res = _newton_loop_12(res_fn, cell0_guesses[0], n_iter, damper, tol, reg)
    multi_sol, multi_conv, multi_res = _solve_vmap_12(res_fn, cell0_guesses, n_iter, damper, tol, reg)
    use_seed = seed_conv | (seed_res <= multi_res + 1e-12)
    return (
        jnp.where(use_seed, seed_sol, multi_sol),
        jnp.where(use_seed, seed_conv, multi_conv),
        jnp.where(use_seed, seed_res, multi_res),
    )

@jit
def reactor_solve_v4(
    inlet_12,
    dz_list,
    g_src_9_list,
    s_src_list,
    e_src_list,
    mesh_z,
    A,
    cell0_guesses,
    P,
    C_fed,
    coal_flow,
    d_p0,
    eps,
    hl_pct,
    L,
    char_comb,
    wgs_cat,
    c_co2,
    p_o2_c,
    hl_norm,
    heat_loss_ref_temp,
    hf,
    cp,
    hhv,
    ts_in_0,
    f_ash,
    ref_f,
    ref_e,
    xc0,
    f_s_coal,
):
    def scan_body(carry, inputs):
        prev_x, prev_ts = carry
        dz, g9, s, e, z, idx = inputs
        def res_fn(x):
            return cell_residuals_jax_flat(
                x,
                prev_x,
                g9,
                s,
                e,
                dz,
                A,
                P,
                C_fed,
                prev_ts,
                coal_flow,
                d_p0,
                eps,
                hl_pct,
                L,
                char_comb,
                wgs_cat,
                c_co2,
                p_o2_c,
                hl_norm,
                heat_loss_ref_temp,
                hf,
                cp,
                hhv,
                f_ash,
                ref_f,
                ref_e,
                xc0,
                f_s_coal,
            )
        # reg：略增大以减轻病态 J 导致 jnp.linalg.solve 出 NaN（工业大流量工况）
        x_sol, conv, _ = lax.cond(
            idx == 0,
            lambda _: _solve_cell0_preferring_seed(res_fn, cell0_guesses, 80, 0.8, 1e-8, 1e-8),
            lambda _: _newton_loop_12(res_fn, prev_x, 80, 0.8, 1e-8, 1e-8),
            None,
        )
        # 下一格入口颗粒温度：与残差内 ``_calc_particle_temperature_jax`` 同一模型，避免 march 与残差不一致
        F = jnp.maximum(jnp.sum(x_sol[:9]), 1e-9)
        v = (F * 8.314 * x_sol[11]) / (P * A * eps + 1e-9)
        tau = dz / jnp.maximum(v, 0.1)
        Xt = jnp.clip(1.0 - (x_sol[9] * x_sol[10] / (C_fed + 1e-12)), 0.0, 1.0)
        dp = d_p0 * (f_ash + (1.0 - f_ash) * (1.0 - Xt)) ** (1 / 3.0)
        _, Ts_out = _calc_particle_temperature_jax(
            x_sol[11],
            prev_ts,
            tau,
            dp,
            PARTICLE_DENSITY,
            cp,
            EF_PARTICLE_TRANSIENT,
            STEFAN_BOLTZMANN,
            CONDUT_COEFF,
            nc=TS_TRANSIENT_NC,
        )
        return (x_sol, Ts_out), x_sol

    return lax.scan(scan_body, (inlet_12, ts_in_0), (dz_list, g_src_9_list, s_src_list, e_src_list, mesh_z, jnp.arange(dz_list.shape[0])))[1]

def jax_solve_cell_bridge(cell, x0, lower, upper, x0_list=None, **kwargs):
    if x0_list:
        best, _ = newton_solve_multistart_cell_pure_jax_ad(cell, x0_list, lower, upper, **kwargs)
        if best is not None:
            return best
    return newton_solve_cell_pure_jax_ad(cell, x0, lower, upper, **kwargs)

def warmup_jax() -> None:
    try:
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        jax.numpy.ones((2, 2))
        _ = jnp.sum(jnp.arange(4.0, dtype=jnp.float64).reshape(2, 2))
    except Exception as e:
        logger.debug("JAX warmup skipped: %s", e)
