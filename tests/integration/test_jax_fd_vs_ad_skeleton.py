"""
FD-vs-AD skeleton for the pure JAX model path.

This intentionally validates JAX against JAX only: a centered finite difference
of a JAX scalar function is compared with ``jax.grad`` for the same function.
It does not assert NumPy mainline parity and does not introduce IFT/custom VJP.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))


def test_heat_loss_percent_energy_residual_fd_matches_ad():
    """Compare d(res_E)/d(HeatLossPercent) by centered FD and jax.grad."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    jnp = pytest.importorskip("jax.numpy")

    from model.jax_residuals import cell_residuals_jax_flat

    # Fixed one-cell, single-branch JAX fixture. Species order is the JAX 9-species
    # contract: O2, CH4, CO, CO2, H2O, H2, N2, H2S, COS, then Ws, Xc, T.
    inlet_12 = jnp.asarray(
        [12.0, 0.8, 0.2, 0.1, 2.0, 0.1, 20.0, 0.0, 0.0, 1.2, 0.72, 900.0],
        dtype=jnp.float64,
    )
    x_12 = jnp.asarray(
        [6.0, 0.1, 1.4, 1.8, 3.0, 0.9, 20.0, 0.0, 0.0, 1.05, 0.68, 1600.0],
        dtype=jnp.float64,
    )
    g_src_9 = jnp.asarray([0.0, 0.3, 0.5, 0.2, 0.4, 0.1, 0.0, 0.0, 0.0], dtype=jnp.float64)

    def energy_residual(heat_loss_percent):
        residual = cell_residuals_jax_flat(
            x_12,
            inlet_12,
            g_src_9,
            0.0,       # s_src
            1.0e5,     # e_src
            0.05,      # dz
            1.0,       # A
            2.0e6,     # P
            0.86,      # C_fed_total
            900.0,     # Ts_in
            1.0,       # coal_flow_kg_s
            100e-6,    # d_p0
            1.0,       # epsilon
            heat_loss_percent,
            6.0,       # L_reactor
            0.3,       # CharCombFac
            0.2,       # WGS_CatFac
            1.0,       # Comb_CO2_Frac
            0.05,      # P_O2_Comb_atm
            6.0,       # L_heatloss_norm
            1800.0,    # HeatLossRefTemp
            -2.0e6,    # Hf_coal
            1300.0,    # cp_char
            30.0,      # HHV_mj
            0.08,      # f_ash
            35.0,      # ref_flow
            5.0e5,     # ref_energy
            0.72,      # char_Xc0_global
            0.0,       # f_s_coal
        )
        return residual[-1]

    p0 = jnp.asarray(2.0, dtype=jnp.float64)
    step = jnp.asarray(1.0e-3, dtype=jnp.float64)
    fd_grad = (energy_residual(p0 + step) - energy_residual(p0 - step)) / (2.0 * step)
    ad_grad = jax.grad(energy_residual)(p0)

    np.testing.assert_allclose(np.asarray(ad_grad), np.asarray(fd_grad), rtol=1e-5, atol=1e-7)


def test_heat_loss_percent_fixed_newton_temperature_fd_matches_ad():
    """
    Solver-level gradient skeleton: compare d(T_after_one_newton_step)/d(HeatLossPercent)
    by centered FD and AD through the existing pure-JAX fixed-iteration Newton path.

    This deliberately uses one fixed damped Newton update instead of a converged solve:
    it exercises the solver Jacobian/update path while avoiding branch changes and the
    compile cost of differentiating through many scan iterations.
    """
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    jnp = pytest.importorskip("jax.numpy")

    from model.jax_residuals import cell_residuals_jax_flat
    from model.jax_solver import _newton_loop_12

    inlet_12 = jnp.asarray(
        [12.0, 0.8, 0.2, 0.1, 2.0, 0.1, 20.0, 0.0, 0.0, 1.2, 0.72, 900.0],
        dtype=jnp.float64,
    )
    x0 = jnp.asarray(
        [6.0, 0.1, 1.4, 1.8, 3.0, 0.9, 20.0, 0.0, 0.0, 1.05, 0.68, 1600.0],
        dtype=jnp.float64,
    )
    g_src_9 = jnp.asarray([0.0, 0.3, 0.5, 0.2, 0.4, 0.1, 0.0, 0.0, 0.0], dtype=jnp.float64)

    def residuals_for(heat_loss_percent):
        def residuals(x):
            return cell_residuals_jax_flat(
                x,
                inlet_12,
                g_src_9,
                0.0,       # s_src
                1.0e5,     # e_src
                0.05,      # dz
                1.0,       # A
                2.0e6,     # P
                0.86,      # C_fed_total
                900.0,     # Ts_in
                1.0,       # coal_flow_kg_s
                100e-6,    # d_p0
                1.0,       # epsilon
                heat_loss_percent,
                6.0,       # L_reactor
                0.3,       # CharCombFac
                0.2,       # WGS_CatFac
                1.0,       # Comb_CO2_Frac
                0.05,      # P_O2_Comb_atm
                6.0,       # L_heatloss_norm
                1800.0,    # HeatLossRefTemp
                -2.0e6,    # Hf_coal
                1300.0,    # cp_char
                30.0,      # HHV_mj
                0.08,      # f_ash
                35.0,      # ref_flow
                5.0e5,     # ref_energy
                0.72,      # char_Xc0_global
                0.0,       # f_s_coal
            )

        return residuals

    def one_step_temperature(heat_loss_percent):
        x_sol, _, _ = _newton_loop_12(
            residuals_for(heat_loss_percent),
            x0,
            n_iter=1,
            damper=0.25,
            tol=1e-8,
            reg=1e-6,
        )
        return x_sol[11]

    p0 = jnp.asarray(2.0, dtype=jnp.float64)
    step = jnp.asarray(1.0e-3, dtype=jnp.float64)
    fd_grad = (one_step_temperature(p0 + step) - one_step_temperature(p0 - step)) / (2.0 * step)
    ad_grad = jax.grad(one_step_temperature)(p0)

    np.testing.assert_allclose(np.asarray(ad_grad), np.asarray(fd_grad), rtol=1e-6, atol=1e-7)


def test_heat_loss_percent_single_cell_ift_temperature_fd_matches_ad():
    """
    Minimal single-cell IFT/custom-VJP proof of concept.

    Formulation:
        F(x, theta) = cell_residuals_jax_flat(x, ..., HeatLossPercent=theta)
        x = 12-vector [O2, CH4, CO, CO2, H2O, H2, N2, H2S, COS, Ws, Xc, T]
        theta = HeatLossPercent

    Forward returns the converged-ish fixed-branch single-cell solve x(theta).
    Backward applies the implicit-function theorem:
        F_x.T @ lambda = x_bar
        theta_bar = -lambda.T @ F_theta
    """
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    jnp = pytest.importorskip("jax.numpy")

    from model.jax_residuals import cell_residuals_jax_flat
    from model.jax_ift import make_single_parameter_ift_solve
    from model.jax_solver import _newton_loop_12

    inlet_12 = jnp.asarray(
        [12.0, 0.8, 0.2, 0.1, 2.0, 0.1, 20.0, 0.0, 0.0, 1.2, 0.72, 900.0],
        dtype=jnp.float64,
    )
    x0 = jnp.asarray(
        [6.0, 0.1, 1.4, 1.8, 3.0, 0.9, 20.0, 0.0, 0.0, 1.05, 0.68, 1600.0],
        dtype=jnp.float64,
    )
    g_src_9 = jnp.asarray([0.0, 0.3, 0.5, 0.2, 0.4, 0.1, 0.0, 0.0, 0.0], dtype=jnp.float64)

    def residual(x, heat_loss_percent):
        return cell_residuals_jax_flat(
            x,
            inlet_12,
            g_src_9,
            0.0,       # s_src
            1.0e5,     # e_src
            0.05,      # dz
            1.0,       # A
            2.0e6,     # P
            0.86,      # C_fed_total
            900.0,     # Ts_in
            1.0,       # coal_flow_kg_s
            100e-6,    # d_p0
            1.0,       # epsilon
            heat_loss_percent,
            6.0,       # L_reactor
            0.3,       # CharCombFac
            0.2,       # WGS_CatFac
            1.0,       # Comb_CO2_Frac
            0.05,      # P_O2_Comb_atm
            6.0,       # L_heatloss_norm
            1800.0,    # HeatLossRefTemp
            -2.0e6,    # Hf_coal
            1300.0,    # cp_char
            30.0,      # HHV_mj
            0.08,      # f_ash
            35.0,      # ref_flow
            5.0e5,     # ref_energy
            0.72,      # char_Xc0_global
            0.0,       # f_s_coal
        )

    def solve_raw(heat_loss_percent):
        def residual_x(x):
            return residual(x, heat_loss_percent)

        x_sol, _, _ = _newton_loop_12(
            residual_x,
            x0,
            n_iter=12,
            damper=0.8,
            tol=1e-10,
            reg=1e-8,
        )
        return x_sol

    solve_ift = make_single_parameter_ift_solve(residual, solve_raw)

    def temperature_ift(heat_loss_percent):
        return solve_ift(heat_loss_percent)[11]

    def temperature_fd_target(heat_loss_percent):
        return solve_raw(heat_loss_percent)[11]

    p0 = jnp.asarray(2.0, dtype=jnp.float64)
    step = jnp.asarray(1.0e-3, dtype=jnp.float64)
    x_at_p0 = solve_raw(p0)
    residual_norm = jnp.max(jnp.abs(residual(x_at_p0, p0)))
    fd_grad = (temperature_fd_target(p0 + step) - temperature_fd_target(p0 - step)) / (2.0 * step)
    ift_grad = jax.grad(temperature_ift)(p0)

    assert float(residual_norm) < 1.0e-7
    np.testing.assert_allclose(np.asarray(ift_grad), np.asarray(fd_grad), rtol=1e-5, atol=1e-7)


def test_heat_loss_percent_two_cell_scan_ift_temperature_fd_matches_ad():
    """
    Two-cell scan POC: per-cell IFT custom VJP propagates through lax.scan carry.

    The second cell residual depends on the first cell outlet through ``prev_x``
    and ``prev_ts``. The helper returns cotangents for those carry values, so the
    scan transpose can chain the final-temperature cotangent back through both
    implicit cell solves.
    """
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    jnp = pytest.importorskip("jax.numpy")

    from model.jax_residuals import cell_residuals_jax_flat
    from model.jax_ift import make_single_parameter_ift_solve_with_args
    from model.jax_solver import _newton_loop_12

    inlet_12 = jnp.asarray(
        [12.0, 0.8, 0.2, 0.1, 2.0, 0.1, 20.0, 0.0, 0.0, 1.2, 0.72, 900.0],
        dtype=jnp.float64,
    )
    g_srcs = jnp.asarray(
        [
            [0.0, 0.3, 0.5, 0.2, 0.4, 0.1, 0.0, 0.0, 0.0],
            [0.0, 0.05, 0.1, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=jnp.float64,
    )

    def residual_cell(x, heat_loss_percent, prev_x, prev_ts, g_src):
        return cell_residuals_jax_flat(
            x,
            prev_x,
            g_src,
            0.0,       # s_src
            1.0e5,     # e_src
            0.05,      # dz
            1.0,       # A
            2.0e6,     # P
            0.86,      # C_fed_total
            prev_ts,
            1.0,       # coal_flow_kg_s
            100e-6,    # d_p0
            1.0,       # epsilon
            heat_loss_percent,
            6.0,       # L_reactor
            0.3,       # CharCombFac
            0.2,       # WGS_CatFac
            1.0,       # Comb_CO2_Frac
            0.05,      # P_O2_Comb_atm
            6.0,       # L_heatloss_norm
            1800.0,    # HeatLossRefTemp
            -2.0e6,    # Hf_coal
            1300.0,    # cp_char
            30.0,      # HHV_mj
            0.08,      # f_ash
            35.0,      # ref_flow
            5.0e5,     # ref_energy
            0.72,      # char_Xc0_global
            0.0,       # f_s_coal
        )

    def solve_cell_raw(heat_loss_percent, prev_x, prev_ts, g_src):
        def residual_x(x):
            return residual_cell(x, heat_loss_percent, prev_x, prev_ts, g_src)

        x_sol, _, _ = _newton_loop_12(
            residual_x,
            prev_x,
            n_iter=10,
            damper=0.8,
            tol=1e-10,
            reg=1e-8,
        )
        return x_sol

    solve_cell_ift = make_single_parameter_ift_solve_with_args(residual_cell, solve_cell_raw)

    def scan_solve(heat_loss_percent, cell_solve):
        def body(carry, g_src):
            prev_x, prev_ts = carry
            x_sol = cell_solve(heat_loss_percent, prev_x, prev_ts, g_src)
            return (x_sol, x_sol[11]), x_sol

        _, xs = jax.lax.scan(body, (inlet_12, inlet_12[11]), g_srcs)
        return xs

    def final_temperature_ift(heat_loss_percent):
        return scan_solve(heat_loss_percent, solve_cell_ift)[-1, 11]

    def final_temperature_fd_target(heat_loss_percent):
        return scan_solve(heat_loss_percent, solve_cell_raw)[-1, 11]

    p0 = jnp.asarray(2.0, dtype=jnp.float64)
    step = jnp.asarray(1.0e-3, dtype=jnp.float64)
    xs_at_p0 = scan_solve(p0, solve_cell_raw)
    carry = (inlet_12, inlet_12[11])
    residual_norms = []
    for i in range(2):
        prev_x, prev_ts = carry
        residual_norms.append(jnp.max(jnp.abs(residual_cell(xs_at_p0[i], p0, prev_x, prev_ts, g_srcs[i]))))
        carry = (xs_at_p0[i], xs_at_p0[i, 11])

    fd_grad = (final_temperature_fd_target(p0 + step) - final_temperature_fd_target(p0 - step)) / (2.0 * step)
    ift_grad = jax.grad(final_temperature_ift)(p0)

    assert float(jnp.max(jnp.asarray(residual_norms))) < 5.0e-7
    np.testing.assert_allclose(np.asarray(ift_grad), np.asarray(fd_grad), rtol=1e-5, atol=2e-6)
