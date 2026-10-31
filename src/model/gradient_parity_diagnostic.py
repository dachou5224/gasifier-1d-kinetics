"""
FD-vs-IFT/JAX gradient parity diagnostics for the fixed-branch two-cell fixture.

This module produces paper-friendly parity rows for the current reusable
``jax_ift`` helper without changing the public solver API.  It deliberately
uses a tiny fixed-branch JAX fixture instead of the full reactor path, so the
artifact is gradient-consistency evidence, not calibration accuracy evidence.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

GRADIENT_PARITY_COLUMNS: tuple[str, ...] = (
    "parameter",
    "parameter_key",
    "parameter_base_value",
    "parameter_step",
    "output",
    "output_base_value",
    "fd_grad",
    "ift_or_ad_grad",
    "abs_error",
    "rel_error",
    "residual_max_abs",
    "status",
    "notes",
)

SUPPORTED_PARAMETERS: tuple[str, ...] = (
    "HeatLossPercent",
    "oxygen_flow",
    "carrier_gas_flow",
    "coal_mass_flow",
)

BLOCKED_PARAMETERS: Mapping[str, str] = {}

DEFAULT_OUTPUTS: tuple[str, ...] = (
    "final_T_K",
    "final_CO_mol_s",
    "final_H2_mol_s",
    "final_CO2_mol_s",
    "final_dry_flow_mol_s",
)


def _relative_error(abs_error: float, reference: float) -> float:
    denom = max(abs(reference), 1.0e-12)
    return abs_error / denom


def build_gradient_parity_rows(
    *,
    parameters: Sequence[str] = SUPPORTED_PARAMETERS + tuple(BLOCKED_PARAMETERS),
    outputs: Sequence[str] = DEFAULT_OUTPUTS,
    step_fraction: float = 1.0e-3,
) -> list[dict[str, object]]:
    """Compute centered-FD vs IFT/JAX gradient parity rows."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    jax.config.update("jax_enable_x64", True)

    from .jax_ift import make_single_parameter_ift_solve_with_args
    from .jax_residuals import cell_residuals_jax_flat
    from .jax_solver import _newton_loop_12

    base_inlet = jnp.asarray(
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
    e_srcs = jnp.asarray([1.0e5, 1.0e5], dtype=jnp.float64)
    s_srcs = jnp.asarray([0.0, 0.0], dtype=jnp.float64)
    base_c_fed_total = jnp.asarray(0.86, dtype=jnp.float64)
    base_coal_flow = jnp.asarray(1.0, dtype=jnp.float64)

    parameter_meta = {
        "HeatLossPercent": {
            "key": "jax_fixture.HeatLossPercent",
            "base": 2.0,
            "kind": "heat_loss",
            "notes": "direct residual parameter in cell_residuals_jax_flat",
        },
        "oxygen_flow": {
            "key": "jax_fixture.inlet_12[O2]",
            "base": float(base_inlet[0]),
            "kind": "inlet_o2",
            "notes": "fixture inlet O2 molar flow; analogous to fixed-physics oxygen flow",
        },
        "carrier_gas_flow": {
            "key": "jax_fixture.inlet_12[CO2]",
            "base": float(base_inlet[3]),
            "kind": "inlet_co2_carrier",
            "notes": "fixture inlet CO2 molar flow; analogous to CO2 carrier gas flow",
        },
        "coal_mass_flow": {
            "key": "jax_fixture.dry_coal_feed_package_scale",
            "base": float(base_coal_flow),
            "kind": "coal_feed_package",
            "notes": (
                "diagnostic dry transported coal-feed package scale: scales inlet solid mass, "
                "coal-flow heat-loss basis, C_fed_total, and the fixture source package; "
                "does not apply LIMS moisture or alter ModelCase adapter semantics"
            ),
        },
    }

    output_indices = {
        "final_T_K": 11,
        "final_CO_mol_s": 2,
        "final_H2_mol_s": 5,
        "final_CO2_mol_s": 3,
    }

    def inlet_for(parameter_name, theta):
        kind = parameter_meta[parameter_name]["kind"]
        if kind == "inlet_o2":
            return base_inlet.at[0].set(theta)
        if kind == "inlet_co2_carrier":
            return base_inlet.at[3].set(theta)
        if kind == "coal_feed_package":
            scale = theta / base_coal_flow
            return base_inlet.at[9].set(base_inlet[9] * scale)
        return base_inlet

    def heat_loss_for(parameter_name, theta):
        if parameter_meta[parameter_name]["kind"] == "heat_loss":
            return theta
        return jnp.asarray(2.0, dtype=jnp.float64)

    def coal_scale_for(parameter_name, theta):
        if parameter_meta[parameter_name]["kind"] == "coal_feed_package":
            return theta / base_coal_flow
        return jnp.asarray(1.0, dtype=jnp.float64)

    def output_vector(xs):
        final = xs[-1]
        dry_flow = final[0] + final[1] + final[2] + final[3] + final[5] + final[6] + final[7] + final[8]
        values = {
            "final_T_K": final[output_indices["final_T_K"]],
            "final_CO_mol_s": final[output_indices["final_CO_mol_s"]],
            "final_H2_mol_s": final[output_indices["final_H2_mol_s"]],
            "final_CO2_mol_s": final[output_indices["final_CO2_mol_s"]],
            "final_dry_flow_mol_s": dry_flow,
        }
        return jnp.asarray([values[name] for name in outputs], dtype=jnp.float64)

    def build_for_parameter(parameter_name):
        def residual_cell(x, theta, prev_x, prev_ts, g_src, s_src, e_src):
            coal_scale = coal_scale_for(parameter_name, theta)
            return cell_residuals_jax_flat(
                x,
                prev_x,
                g_src * coal_scale,
                s_src * coal_scale,
                e_src * coal_scale,
                0.05,
                1.0,
                2.0e6,
                base_c_fed_total * coal_scale,
                prev_ts,
                base_coal_flow * coal_scale,
                100e-6,
                1.0,
                heat_loss_for(parameter_name, theta),
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

        def solve_cell_raw(theta, prev_x, prev_ts, g_src, s_src, e_src):
            def residual_x(x):
                return residual_cell(x, theta, prev_x, prev_ts, g_src, s_src, e_src)

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

        def scan_solve(theta, cell_solve):
            initial = inlet_for(parameter_name, theta)

            def body(carry, g_src):
                prev_x, prev_ts = carry
                g, s, e = g_src
                x_sol = cell_solve(theta, prev_x, prev_ts, g, s, e)
                return (x_sol, x_sol[11]), x_sol

            _, xs = jax.lax.scan(body, (initial, initial[11]), (g_srcs, s_srcs, e_srcs))
            return xs

        def values_raw(theta):
            return output_vector(scan_solve(theta, solve_cell_raw))

        def values_ift(theta):
            return output_vector(scan_solve(theta, solve_cell_ift))

        def residual_quality(theta):
            xs = scan_solve(theta, solve_cell_raw)
            carry = (inlet_for(parameter_name, theta), inlet_for(parameter_name, theta)[11])
            residual_norms = []
            for i in range(2):
                prev_x, prev_ts = carry
                residual_norms.append(jnp.max(jnp.abs(residual_cell(xs[i], theta, prev_x, prev_ts, g_srcs[i], s_srcs[i], e_srcs[i]))))
                carry = (xs[i], xs[i, 11])
            return jnp.max(jnp.asarray(residual_norms))

        return values_raw, values_ift, residual_quality

    rows: list[dict[str, object]] = []
    for parameter_name in parameters:
        if parameter_name in BLOCKED_PARAMETERS:
            for output_name in outputs:
                rows.append(
                    {
                        "parameter": parameter_name,
                        "parameter_key": "",
                        "parameter_base_value": math.nan,
                        "parameter_step": math.nan,
                        "output": output_name,
                        "output_base_value": math.nan,
                        "fd_grad": math.nan,
                        "ift_or_ad_grad": math.nan,
                        "abs_error": math.nan,
                        "rel_error": math.nan,
                        "residual_max_abs": math.nan,
                        "status": "blocked",
                        "notes": BLOCKED_PARAMETERS[parameter_name],
                    }
                )
            continue
        if parameter_name not in parameter_meta:
            raise ValueError(f"Unknown gradient parity parameter: {parameter_name}")

        meta = parameter_meta[parameter_name]
        theta0 = jnp.asarray(meta["base"], dtype=jnp.float64)
        step = jnp.asarray(max(abs(float(theta0)) * step_fraction, 1.0e-6), dtype=jnp.float64)
        values_raw, values_ift, residual_quality = build_for_parameter(parameter_name)
        base_values = values_raw(theta0)
        fd_grads = (values_raw(theta0 + step) - values_raw(theta0 - step)) / (2.0 * step)
        ift_grads = jax.jacrev(values_ift)(theta0)
        residual_max = float(residual_quality(theta0))
        for idx, output_name in enumerate(outputs):
            fd_grad = float(np.asarray(fd_grads[idx]))
            ift_grad = float(np.asarray(ift_grads[idx]))
            abs_error = abs(ift_grad - fd_grad)
            rel_error = _relative_error(abs_error, fd_grad)
            status = "ok" if residual_max < 5.0e-7 and math.isfinite(abs_error) else "failed"
            rows.append(
                {
                    "parameter": parameter_name,
                    "parameter_key": meta["key"],
                    "parameter_base_value": float(theta0),
                    "parameter_step": float(step),
                    "output": output_name,
                    "output_base_value": float(np.asarray(base_values[idx])),
                    "fd_grad": fd_grad,
                    "ift_or_ad_grad": ift_grad,
                    "abs_error": abs_error,
                    "rel_error": rel_error,
                    "residual_max_abs": residual_max,
                    "status": status,
                    "notes": meta["notes"],
                }
            )
    return [{column: row.get(column) for column in GRADIENT_PARITY_COLUMNS} for row in rows]


def gradient_parity_markdown(rows: Sequence[Mapping[str, object]]) -> str:
    """Return a compact Markdown summary for parity matrix artifacts."""
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    blocked_rows = [row for row in rows if row.get("status") == "blocked"]
    max_abs = max((float(row["abs_error"]) for row in ok_rows), default=math.nan)
    max_rel = max((float(row["rel_error"]) for row in ok_rows), default=math.nan)
    supported = sorted({str(row["parameter"]) for row in ok_rows})
    blocked = sorted({str(row["parameter"]) for row in blocked_rows})
    outputs = sorted({str(row["output"]) for row in rows})
    lines = [
        "# FD-vs-IFT/JAX gradient parity diagnostic",
        "",
        "This is fixed-branch gradient-consistency evidence, not hybrid calibration accuracy evidence.",
        "",
        f"- Rows: {len(rows)}",
        f"- OK rows: {len(ok_rows)}",
        f"- Blocked rows: {len(blocked_rows)}",
        f"- Supported parameters: {', '.join(supported) if supported else 'none'}",
        f"- Blocked parameters: {', '.join(blocked) if blocked else 'none'}",
        f"- Outputs: {', '.join(outputs)}",
        f"- Max abs error among OK rows: {max_abs:.6g}",
        f"- Max rel error among OK rows: {max_rel:.6g}",
        "",
        "| parameter | output | fd_grad | ift_or_ad_grad | abs_error | rel_error | status |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        fd = float(row["fd_grad"]) if row.get("status") == "ok" else math.nan
        ad = float(row["ift_or_ad_grad"]) if row.get("status") == "ok" else math.nan
        abs_err = float(row["abs_error"]) if row.get("status") == "ok" else math.nan
        rel_err = float(row["rel_error"]) if row.get("status") == "ok" else math.nan
        lines.append(
            "| {parameter} | {output} | {fd:.6g} | {ad:.6g} | {abs_err:.3e} | {rel_err:.3e} | {status} |".format(
                parameter=row["parameter"],
                output=row["output"],
                fd=fd,
                ad=ad,
                abs_err=abs_err,
                rel_err=rel_err,
                status=row["status"],
            )
        )
    if blocked_rows:
        lines.extend(["", "Blocked parameter notes:"])
        for parameter in blocked:
            note = next(str(row["notes"]) for row in blocked_rows if row["parameter"] == parameter)
            lines.append(f"- {parameter}: {note}")
    lines.extend(
        [
            "",
            "Evidence boundary: fixed-branch two-cell JAX/IFT fixture. "
            "Do not interpret as full-reactor gradient parity, training, or calibrated accuracy.",
        ]
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "BLOCKED_PARAMETERS",
    "DEFAULT_OUTPUTS",
    "GRADIENT_PARITY_COLUMNS",
    "SUPPORTED_PARAMETERS",
    "build_gradient_parity_rows",
    "gradient_parity_markdown",
]
