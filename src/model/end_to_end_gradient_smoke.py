"""
End-to-end gradient smoke helpers for the current fixed-branch JAX/IFT path.

This module is diagnostic-only.  It does not calibrate parameters, change solver
defaults, or connect to the production reactor solve.  The fixture mirrors the
two-cell scan IFT regression so the smoke can answer one narrow question:
does an IFT/JAX gradient move a scalar objective downhill?
"""
from __future__ import annotations

import time
from typing import Mapping, Sequence

END_TO_END_GRADIENT_SMOKE_COLUMNS: tuple[str, ...] = (
    "step",
    "parameter_name",
    "parameter_value",
    "loss",
    "gradient",
    "step_size",
    "next_parameter_value",
    "next_loss",
    "solve_success",
    "fallback_count",
    "poor_convergence_count",
    "T_out_K",
    "T_target_K",
    "residual_max_abs",
    "direction_check_loss",
    "direction_check_passed",
    "runtime_s",
)


def run_heat_loss_two_cell_ift_smoke(
    *,
    theta0: float = 2.0,
    learning_rate: float = 0.01,
    n_steps: int = 8,
    target_delta_k: float = -3.0,
    theta_min: float = 0.0,
    theta_max: float = 20.0,
) -> list[dict[str, object]]:
    """
    Run a tiny gradient descent smoke on ``HeatLossPercent -> two-cell T_out``.

    The target is defined relative to the initial fixed-branch outlet
    temperature: ``T_target = T_out(theta0) + target_delta_k``.  With the current
    fixture, increasing heat loss lowers outlet temperature, so a negative
    gradient direction should reduce the squared-temperature loss.
    """
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    from .jax_ift import make_single_parameter_ift_solve_with_args
    from .jax_residuals import cell_residuals_jax_flat
    from .jax_solver import _newton_loop_12

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
            0.0,
            1.0e5,
            0.05,
            1.0,
            2.0e6,
            0.86,
            prev_ts,
            1.0,
            100e-6,
            1.0,
            heat_loss_percent,
            6.0,
            0.3,
            0.2,
            1.0,
            0.05,
            6.0,
            1800.0,
            -2.0e6,
            1300.0,
            30.0,
            0.08,
            35.0,
            5.0e5,
            0.72,
            0.0,
        )

    def solve_cell_raw(heat_loss_percent, prev_x, prev_ts, g_src):
        def residual_x(x):
            return residual_cell(x, heat_loss_percent, prev_x, prev_ts, g_src)

        x_sol, _converged, _res_norm = _newton_loop_12(
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

    def final_temperature(heat_loss_percent):
        return scan_solve(heat_loss_percent, solve_cell_ift)[-1, 11]

    initial_theta = jnp.asarray(theta0, dtype=jnp.float64)
    target_temperature = final_temperature(initial_theta) + jnp.asarray(target_delta_k, dtype=jnp.float64)

    def loss_fn(heat_loss_percent):
        t_out = final_temperature(heat_loss_percent)
        return 0.5 * (t_out - target_temperature) ** 2

    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    eval_loss = jax.jit(loss_fn)

    def residual_quality(heat_loss_percent):
        xs = scan_solve(heat_loss_percent, solve_cell_raw)
        carry = (inlet_12, inlet_12[11])
        residual_norms = []
        for i in range(2):
            prev_x, prev_ts = carry
            residual_norms.append(
                jnp.max(jnp.abs(residual_cell(xs[i], heat_loss_percent, prev_x, prev_ts, g_srcs[i])))
            )
            carry = (xs[i], xs[i, 11])
        max_residual = jnp.max(jnp.asarray(residual_norms))
        return xs[-1, 11], max_residual

    rows: list[dict[str, object]] = []
    theta = float(theta0)
    for step in range(int(n_steps)):
        start = time.perf_counter()
        theta_j = jnp.asarray(theta, dtype=jnp.float64)
        loss_value, grad_value = loss_and_grad(theta_j)
        gradient = float(grad_value)
        step_size = -float(learning_rate) * gradient
        next_theta = min(max(theta + step_size, float(theta_min)), float(theta_max))
        next_loss = float(eval_loss(jnp.asarray(next_theta, dtype=jnp.float64)))
        t_out, residual_max = residual_quality(theta_j)
        residual_max_f = float(residual_max)
        solve_success = residual_max_f < 5.0e-7
        direction_loss = float(eval_loss(jnp.asarray(theta + step_size, dtype=jnp.float64)))
        runtime_s = time.perf_counter() - start
        rows.append(
            {
                "step": step,
                "parameter_name": "HeatLossPercent",
                "parameter_value": theta,
                "loss": float(loss_value),
                "gradient": gradient,
                "step_size": step_size,
                "next_parameter_value": next_theta,
                "next_loss": next_loss,
                "solve_success": bool(solve_success),
                "fallback_count": 0,
                "poor_convergence_count": 0 if solve_success else 1,
                "T_out_K": float(t_out),
                "T_target_K": float(target_temperature),
                "residual_max_abs": residual_max_f,
                "direction_check_loss": direction_loss,
                "direction_check_passed": bool(direction_loss < float(loss_value)),
                "runtime_s": runtime_s,
            }
        )
        theta = next_theta
    return [{column: row.get(column) for column in END_TO_END_GRADIENT_SMOKE_COLUMNS} for row in rows]


def gradient_smoke_markdown(rows: Sequence[Mapping[str, object]]) -> str:
    """Return a compact Markdown summary for the end-to-end gradient smoke."""
    if not rows:
        return "# End-to-end gradient smoke\n\nNo rows generated.\n"
    initial_loss = float(rows[0]["loss"])
    final_loss = float(rows[-1]["next_loss"])
    monotonic = all(float(row["next_loss"]) <= float(row["loss"]) + 1.0e-12 for row in rows)
    lines = [
        "# End-to-end gradient smoke",
        "",
        "This is end-to-end gradient smoke evidence, not calibration accuracy evidence.",
        "",
        f"- Parameter: {rows[0]['parameter_name']}",
        f"- Steps: {len(rows)}",
        f"- Initial loss: {initial_loss:.8g}",
        f"- Final loss: {final_loss:.8g}",
        f"- Overall decrease: {final_loss < initial_loss}",
        f"- Per-step decrease: {monotonic}",
        "",
        "| step | theta | loss | grad | step_size | next_loss | T_out_K | residual_max |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {step} | {theta:.6g} | {loss:.6g} | {grad:.6g} | {step_size:.6g} | {next_loss:.6g} | {temp:.3f} | {res:.3e} |".format(
                step=int(row["step"]),
                theta=float(row["parameter_value"]),
                loss=float(row["loss"]),
                grad=float(row["gradient"]),
                step_size=float(row["step_size"]),
                next_loss=float(row["next_loss"]),
                temp=float(row["T_out_K"]),
                res=float(row["residual_max_abs"]),
            )
        )
    lines.extend(
        [
            "",
            "Evidence boundary: fixed-branch two-cell JAX/IFT smoke only. "
            "Do not interpret as hybrid calibration, full-reactor training, or industrial validation.",
        ]
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "END_TO_END_GRADIENT_SMOKE_COLUMNS",
    "gradient_smoke_markdown",
    "run_heat_loss_two_cell_ift_smoke",
]
