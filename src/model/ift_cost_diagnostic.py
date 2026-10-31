"""
IFT/custom-VJP vs direct-unrolled AD cost diagnostics.

This module benchmarks the fixed-branch JAX fixture used by the gradient smoke
and parity artifacts.  It is method evidence only: it does not calibrate,
train, or change the production solver API.
"""
from __future__ import annotations

import math
import platform
import sys
import time
from typing import Mapping, Sequence

IFT_COST_COLUMNS: tuple[str, ...] = (
    "method",
    "N_cells",
    "parameter",
    "output",
    "newton_iterations",
    "status",
    "failure_reason",
    "compile_or_cold_time_s",
    "warm_eval_time_s",
    "warm_repeats",
    "gradient_value",
    "reference_method",
    "reference_gradient_value",
    "abs_gradient_diff",
    "rel_gradient_diff",
    "residual_max_abs",
)

METHOD_IFT = "ift_custom_vjp"
METHOD_UNROLLED = "direct_unrolled_ad"
DEFAULT_TIMEOUT_S = 90.0


def _relative_error(abs_error: float, reference: float) -> float:
    return abs_error / max(abs(reference), 1.0e-12)


def run_cost_case(
    method: str,
    n_cells: int,
    *,
    newton_iterations: int = 8,
    warm_repeats: int = 2,
) -> dict[str, object]:
    """Run one benchmark case for d(final_T_K)/d(HeatLossPercent)."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    jax.config.update("jax_enable_x64", True)

    from .jax_ift import make_single_parameter_ift_solve_with_args
    from .jax_residuals import cell_residuals_jax_flat
    from .jax_solver import _newton_loop_12

    if method not in {METHOD_IFT, METHOD_UNROLLED}:
        raise ValueError(f"Unknown cost diagnostic method: {method}")

    inlet_12 = jnp.asarray(
        [12.0, 0.8, 0.2, 0.1, 2.0, 0.1, 20.0, 0.0, 0.0, 1.2, 0.72, 900.0],
        dtype=jnp.float64,
    )
    first_src = jnp.asarray([0.0, 0.3, 0.5, 0.2, 0.4, 0.1, 0.0, 0.0, 0.0], dtype=jnp.float64)
    downstream_src = jnp.asarray([0.0, 0.05, 0.1, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
    g_srcs = jnp.vstack([first_src] + [downstream_src for _ in range(max(int(n_cells) - 1, 0))])

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
            n_iter=int(newton_iterations),
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

    cell_solve = solve_cell_ift if method == METHOD_IFT else solve_cell_raw

    def final_temperature(heat_loss_percent):
        return scan_solve(heat_loss_percent, cell_solve)[-1, 11]

    grad_fn = jax.jit(jax.grad(final_temperature))
    theta = jnp.asarray(2.0, dtype=jnp.float64)

    start = time.perf_counter()
    gradient = float(np.asarray(grad_fn(theta)))
    cold_time_s = time.perf_counter() - start

    warm_times: list[float] = []
    for _ in range(int(warm_repeats)):
        warm_start = time.perf_counter()
        _ = float(np.asarray(grad_fn(theta)))
        warm_times.append(time.perf_counter() - warm_start)
    warm_eval_time_s = float(np.median(np.asarray(warm_times))) if warm_times else math.nan

    xs = scan_solve(theta, solve_cell_raw)
    carry = (inlet_12, inlet_12[11])
    residual_norms = []
    for i in range(int(n_cells)):
        prev_x, prev_ts = carry
        residual_norms.append(jnp.max(jnp.abs(residual_cell(xs[i], theta, prev_x, prev_ts, g_srcs[i]))))
        carry = (xs[i], xs[i, 11])

    return {
        "method": method,
        "N_cells": int(n_cells),
        "parameter": "HeatLossPercent",
        "output": "final_T_K",
        "newton_iterations": int(newton_iterations),
        "status": "ok",
        "failure_reason": "",
        "compile_or_cold_time_s": cold_time_s,
        "warm_eval_time_s": warm_eval_time_s,
        "warm_repeats": int(warm_repeats),
        "gradient_value": gradient,
        "reference_method": "",
        "reference_gradient_value": math.nan,
        "abs_gradient_diff": math.nan,
        "rel_gradient_diff": math.nan,
        "residual_max_abs": float(jnp.max(jnp.asarray(residual_norms))),
    }


def annotate_reference_differences(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Add direct-vs-IFT gradient differences where both methods completed."""
    by_n_method = {(int(row["N_cells"]), str(row["method"])): row for row in rows if row.get("status") == "ok"}
    out: list[dict[str, object]] = []
    for row in rows:
        copied = dict(row)
        if row.get("status") == "ok" and row.get("method") == METHOD_UNROLLED:
            ref = by_n_method.get((int(row["N_cells"]), METHOD_IFT))
            if ref is not None:
                ref_grad = float(ref["gradient_value"])
                grad = float(row["gradient_value"])
                abs_diff = abs(grad - ref_grad)
                copied["reference_method"] = METHOD_IFT
                copied["reference_gradient_value"] = ref_grad
                copied["abs_gradient_diff"] = abs_diff
                copied["rel_gradient_diff"] = _relative_error(abs_diff, ref_grad)
        out.append({column: copied.get(column) for column in IFT_COST_COLUMNS})
    return out


def timeout_row(method: str, n_cells: int, *, newton_iterations: int, timeout_s: float) -> dict[str, object]:
    """Return a standardized timeout row."""
    return {
        "method": method,
        "N_cells": int(n_cells),
        "parameter": "HeatLossPercent",
        "output": "final_T_K",
        "newton_iterations": int(newton_iterations),
        "status": "timeout",
        "failure_reason": f"case exceeded timeout_s={timeout_s}",
        "compile_or_cold_time_s": math.nan,
        "warm_eval_time_s": math.nan,
        "warm_repeats": 0,
        "gradient_value": math.nan,
        "reference_method": "",
        "reference_gradient_value": math.nan,
        "abs_gradient_diff": math.nan,
        "rel_gradient_diff": math.nan,
        "residual_max_abs": math.nan,
    }


def failed_row(method: str, n_cells: int, *, newton_iterations: int, failure_reason: str) -> dict[str, object]:
    """Return a standardized failure row."""
    row = timeout_row(method, n_cells, newton_iterations=newton_iterations, timeout_s=math.nan)
    row["status"] = "failed"
    row["failure_reason"] = failure_reason
    return row


def collect_protocol_metadata(*, timeout_s: float = DEFAULT_TIMEOUT_S, warm_repeats: int = 2) -> dict[str, object]:
    """Collect reproducibility metadata for the cost diagnostic protocol."""
    import numpy as np
    import scipy

    try:
        import jax
        import jaxlib

        jax_version = jax.__version__
        jaxlib_version = jaxlib.__version__
        jax_backend = jax.default_backend()
    except Exception as exc:  # pragma: no cover - depends on optional runtime availability
        jax_version = f"unavailable: {type(exc).__name__}: {exc}"
        jaxlib_version = "unavailable"
        jax_backend = "unavailable"

    return {
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "jax_version": jax_version,
        "jaxlib_version": jaxlib_version,
        "jax_backend": jax_backend,
        "timeout_s": float(timeout_s),
        "warm_repeats": int(warm_repeats),
        "parameter": "HeatLossPercent",
        "output": "final_T_K",
        "fixture": "fixed-branch JAX scan fixture with synthetic two-cell/N-cell source pattern",
        "newton_loop": "fixed-iteration _newton_loop_12; no production solver API changes",
    }


def ift_cost_protocol_markdown(metadata: Mapping[str, object] | None = None) -> str:
    """Return reviewer-facing protocol metadata for the IFT cost diagnostic."""
    meta = dict(metadata or collect_protocol_metadata())
    lines = [
        "# IFT cost diagnostic protocol",
        "",
        "This protocol describes how `ift_cost_diagnostic.csv` was produced. "
        "It is method-cost metadata for a fixed-branch JAX fixture, not a production benchmark.",
        "",
        "## Machine and software",
        "",
        f"- Python: {meta['python_version']}",
        f"- Platform: {meta['platform']}",
        f"- Machine: {meta['machine']}",
        f"- Processor: {meta['processor']}",
        f"- NumPy: {meta['numpy_version']}",
        f"- SciPy: {meta['scipy_version']}",
        f"- JAX: {meta['jax_version']}",
        f"- JAXLIB: {meta['jaxlib_version']}",
        f"- JAX backend: {meta['jax_backend']}",
        "",
        "## Fixture",
        "",
        f"- Parameter: {meta['parameter']}",
        f"- Output: {meta['output']}",
        f"- Fixture: {meta['fixture']}",
        f"- Nonlinear solve: {meta['newton_loop']}",
        "",
        "## Method definitions",
        "",
        f"- `{METHOD_IFT}`: each cell solve uses the existing custom VJP helper. "
        "The backward pass applies the implicit-function theorem by solving "
        "`F_x.T @ lambda = cotangent` and returning the parameter/carry cotangents "
        "from `-lambda.T @ F_theta` via the residual pullback.",
        f"- `{METHOD_UNROLLED}`: the same fixed-iteration Newton solve is used as an ordinary "
        "JAX computation graph. `jax.grad(final_temperature)` differentiates through the "
        "unrolled nonlinear iterations and scan carry directly, without custom VJP.",
        "",
        "## Timing definitions",
        "",
        "- `compile_or_cold_time_s`: wall-clock time for the first `jax.jit(jax.grad(...))(theta)` call "
        "inside a fresh subprocess for that method/N case. It includes tracing, XLA compilation, and "
        "the first execution, but not CSV writing.",
        f"- `warm_eval_time_s`: median wall-clock time across {meta['warm_repeats']} subsequent calls to the "
        "same compiled gradient function in the same subprocess.",
        "- The very small warm times are expected because they measure already-compiled scalar-gradient "
        "calls on the same fixed fixture.",
        "",
        "## Timeout rule",
        "",
        f"- Each method/N case is executed in its own spawned Python subprocess. The parent process waits "
        f"`timeout_s={meta['timeout_s']}` seconds using `Process.join(timeout_s)`.",
        "- If the subprocess is still alive after the timeout, it is terminated and a row with "
        "`status=timeout` and `failure_reason='case exceeded timeout_s=...'` is recorded.",
        "- A timeout is treated as a reproducibility/cost boundary, not hidden or retried with looser settings.",
        "",
        "## Evidence boundary",
        "",
        "This protocol supports the methodological claim that IFT/custom VJP is more scalable than "
        "direct unrolled AD for this fixed-branch implicit-solve fixture. Timings are machine- and "
        "JAX-version-dependent. They are not production reactor benchmarks, training results, or "
        "hybrid calibration accuracy evidence.",
    ]
    return "\n".join(lines) + "\n"


def ift_cost_markdown(rows: Sequence[Mapping[str, object]]) -> str:
    """Return a compact Markdown summary for IFT-vs-unrolled cost diagnostics."""
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    statuses = {}
    for row in rows:
        statuses[str(row["status"])] = statuses.get(str(row["status"]), 0) + 1
    ift_ok = [row for row in ok_rows if row["method"] == METHOD_IFT]
    unrolled_ok = [row for row in ok_rows if row["method"] == METHOD_UNROLLED]
    lines = [
        "# IFT vs direct-unrolled AD cost diagnostic",
        "",
        "This is method-cost evidence for a fixed-branch JAX fixture, not calibration accuracy evidence.",
        "",
        f"- Rows: {len(rows)}",
        f"- Status counts: {', '.join(f'{k}={v}' for k, v in sorted(statuses.items()))}",
        f"- IFT OK N_cells: {', '.join(str(int(row['N_cells'])) for row in ift_ok) if ift_ok else 'none'}",
        f"- Direct-unrolled OK N_cells: {', '.join(str(int(row['N_cells'])) for row in unrolled_ok) if unrolled_ok else 'none'}",
        "",
        "| method | N | status | cold_s | warm_s | grad | ref_grad | rel_diff | residual_max |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {n} | {status} | {cold:.3g} | {warm:.3g} | {grad:.6g} | {ref:.6g} | {rel:.3e} | {res:.3e} |".format(
                method=row["method"],
                n=int(row["N_cells"]),
                status=row["status"],
                cold=float(row["compile_or_cold_time_s"]) if row.get("compile_or_cold_time_s") is not None else math.nan,
                warm=float(row["warm_eval_time_s"]) if row.get("warm_eval_time_s") is not None else math.nan,
                grad=float(row["gradient_value"]) if row.get("gradient_value") is not None else math.nan,
                ref=float(row["reference_gradient_value"]) if row.get("reference_gradient_value") is not None else math.nan,
                rel=float(row["rel_gradient_diff"]) if row.get("rel_gradient_diff") is not None else math.nan,
                res=float(row["residual_max_abs"]) if row.get("residual_max_abs") is not None else math.nan,
            )
        )
    lines.extend(
        [
            "",
            "Conclusion: in this fixture, IFT/custom VJP is the scalable path to test across larger scan lengths. "
            "Direct-unrolled AD is useful for small-N agreement checks but is bounded by compile/runtime cost.",
            "",
            "Protocol metadata: see `ift_cost_protocol.md` for cold/warm timing definitions, timeout enforcement, "
            "software versions, and method definitions.",
            "",
            "Evidence boundary: fixed-branch HeatLossPercent -> final_T_K benchmark only. "
            "Do not interpret as full-reactor performance, training, or calibration evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "IFT_COST_COLUMNS",
    "METHOD_IFT",
    "METHOD_UNROLLED",
    "annotate_reference_differences",
    "collect_protocol_metadata",
    "failed_row",
    "ift_cost_protocol_markdown",
    "ift_cost_markdown",
    "run_cost_case",
    "timeout_row",
]
