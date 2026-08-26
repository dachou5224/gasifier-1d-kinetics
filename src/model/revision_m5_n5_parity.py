"""M5 N=5 Huayi FD-vs-IFT gradient parity on the selected JAX path."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .gasifier_system import GasifierSystem
from .grid_service import AdaptiveMeshGenerator, MeshConfig
from .input_contract import ash_mass_fraction_dry, coal_flow_kg_s_for_heat_loss, heat_loss_norm_length_m, heat_loss_ref_temp_k, resolve_f_s_coal
from .material import SPECIES_NAMES
from .model_case_adapter import build_gasifier_inputs_from_model_case
from .physics import get_enthalpy_molar
from .revision_evidence_common import (
    DATASET_ID,
    EVIDENCE_BOUNDARY,
    apply_solver_options,
    load_frozen_split,
    source_commit,
    write_sha256_manifest,
)
from .revision_m1_cell0 import declared_variants, iter_windows, select_representative_windows
from .species_bridge import mainline_gas8_to_jax9

REL_ERROR_PASS = 1.0e-4
ABS_ERROR_PASS = 1.0e-4
NEAR_ZERO_GRAD = 1.0e-8
ABS_ERROR_NEAR_ZERO = 1.0e-6
STEP_FRACTIONS = (1.0e-2, 1.0e-3, 1.0e-4)
PRIMARY_STEP = 1.0e-3
OUTPUTS = ("CO_mol_s", "H2_mol_s", "CO2_mol_s")
PARAMETERS = ("WGS_CatalyticFactor", "oxygen_flow", "coal_mass_flow")


def _n5_options(configuration: str) -> dict[str, Any]:
    variant = declared_variants()[configuration]
    options = dict(variant.get("solver_options") or {})
    first = float(options.get("first_cell_length", 0.4))
    options["first_cell_length"] = first
    options["ignition_zone_res"] = (10.3 - first) / 4.0
    options["JaxSeedCell0FromMinimize"] = False
    return options


def select_n5_cases(runtime: pd.DataFrame, rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    selected = select_representative_windows(runtime, rows)
    picks = selected.sort_values(["phase", "window_id"]).groupby("phase", as_index=False).first()
    return picks


def _bundle(row: Mapping[str, Any], configuration: str):
    built = build_gasifier_inputs_from_model_case(row)
    options = _n5_options(configuration)
    op = apply_solver_options(built["op_conds"], options)
    op["JaxSeedCell0FromMinimize"] = False
    system = GasifierSystem(built["geometry"], built["coal_props"], op)
    L, D = system.geometry["L"], system.geometry["D"]
    A = np.pi * (D / 2.0) ** 2
    mesh = MeshConfig(
        total_length=L,
        n_cells=5,
        ignition_zone_length=op.get("FirstCellLength", 0.4),
        ignition_zone_res=op.get("IgnitionZoneRes", (L - 0.4) / 4.0),
        min_grid_size=0.001,
    )
    dz_list, z_positions = AdaptiveMeshGenerator(mesh).generate()
    system.z_positions = z_positions
    inlet = system._initialize_inlet()
    N = len(dz_list)
    g_src_9 = np.zeros((N, 9))
    s_src = np.zeros(N)
    e_src = np.zeros(N)
    f_h2o = (system.tmp_W_liq_evap / 18.015) * 1000.0
    l_evap = op.get("L_evap_m", 1e-6)
    for i in range(N):
        z, dz = z_positions[i], dz_list[i]
        z_s, z_e = z - dz / 2.0, z + dz / 2.0
        frac = (min(l_evap, z_e) - max(0.0, z_s)) / l_evap if z_e > 0 and z_s < l_evap else 0.0
        g_src_9[i, 4] += f_h2o * frac
        e_src[i] += f_h2o * frac * -285830.0
        if i == 0:
            v = system.tmp_F_vol
            g_src_9[i, :] = g_src_9[i, :] + mainline_gas8_to_jax9(v)
            s_src[i] = -system.tmp_W_vol
            e_src[i] += sum(v[j] * get_enthalpy_molar(sp, inlet.T) for j, sp in enumerate(SPECIES_NAMES) if v[j] > 0)
    hhv = system.coal_props.get("HHV_d", 30.0)
    hhv_mj = hhv / 1000.0 if hhv > 1000.0 else hhv
    f_ash = ash_mass_fraction_dry(system.coal_props) / (system.Cd_total + ash_mass_fraction_dry(system.coal_props) + 1e-9)
    ref_f = max(inlet.total_gas_moles, 1.0)
    ref_e = max(ref_f * 35.0 * 200.0, 5.0e5)
    g9_in = mainline_gas8_to_jax9(inlet.gas_moles)
    inlet_12 = np.concatenate([g9_in, [inlet.solid_mass, inlet.carbon_fraction, inlet.T]])
    guesses = np.array([
        np.concatenate([g9_in, [inlet.solid_mass, inlet.carbon_fraction, t]])
        for t in (3000.0, 2000.0, 1500.0, 1000.0, 400.0)
    ])
    return {
        "system": system,
        "A": A,
        "dz_list": dz_list,
        "z_positions": z_positions,
        "g_src_9": g_src_9,
        "s_src": s_src,
        "e_src": e_src,
        "inlet_12": inlet_12,
        "guesses": guesses,
        "P": float(inlet.P),
        "C_fed": float(inlet.solid_mass * inlet.carbon_fraction),
        "coal_flow": float(coal_flow_kg_s_for_heat_loss(op)),
        "d_p0": float(op.get("particle_diameter", 100e-6)),
        "eps": float(op.get("epsilon", 1.0)),
        "hl_pct": float(op.get("HeatLossPercent", 2.0)),
        "L": float(L),
        "char_comb": float(op.get("CharCombustionRateFactor", 0.3)),
        "wgs_cat": float(op.get("WGS_CatalyticFactor", 0.2)),
        "c_co2": float(op.get("Combustion_CO2_Fraction", 1.0)),
        "p_o2_c": float(op.get("P_O2_Combustion_atm", 0.05)),
        "hl_norm": float(heat_loss_norm_length_m({**op, "L_heatloss_norm": float(np.sum(dz_list))}, float(L), float(np.sum(dz_list)))),
        "hl_ref": float(heat_loss_ref_temp_k(op)),
        "hf": float(system.coal_props.get("Hf_coal", 0.0)),
        "cp": float(system.coal_props.get("cp_char", 1300.0)),
        "hhv_mj": float(hhv_mj),
        "ts_in_0": float(inlet.T),
        "f_ash": float(f_ash),
        "ref_f": float(ref_f),
        "ref_e": float(ref_e),
        "xc0": float(system.char_Xc0),
        "f_s_coal": float(resolve_f_s_coal(system.coal_props, op)),
        "base_o2": float(inlet_12[0]),
        "base_coal": float(inlet.solid_mass),
    }


def _solve_jax(bundle: Mapping[str, Any], *, wgs=None, o2_scale=1.0, coal_scale=1.0):
    import jax.numpy as jnp
    from .jax_solver import reactor_solve_v4

    inlet = np.array(bundle["inlet_12"], dtype=float)
    inlet[0] *= float(o2_scale)
    inlet[9] *= float(coal_scale)
    guesses = np.array(bundle["guesses"], dtype=float)
    guesses[:, 0] *= float(o2_scale)
    guesses[:, 9] *= float(coal_scale)
    g_src = np.array(bundle["g_src_9"], dtype=float) * float(coal_scale)
    s_src = np.array(bundle["s_src"], dtype=float) * float(coal_scale)
    e_src = np.array(bundle["e_src"], dtype=float) * float(coal_scale)
    profile = reactor_solve_v4(
        jnp.asarray(inlet, dtype=jnp.float64),
        jnp.asarray(bundle["dz_list"], dtype=jnp.float64),
        jnp.asarray(g_src, dtype=jnp.float64),
        jnp.asarray(s_src, dtype=jnp.float64),
        jnp.asarray(e_src, dtype=jnp.float64),
        jnp.asarray(bundle["z_positions"], dtype=jnp.float64),
        float(bundle["A"]),
        jnp.asarray(guesses, dtype=jnp.float64),
        float(bundle["P"]),
        float(bundle["C_fed"] * coal_scale),
        float(bundle["coal_flow"] * coal_scale),
        float(bundle["d_p0"]),
        float(bundle["eps"]),
        float(bundle["hl_pct"]),
        float(bundle["L"]),
        float(bundle["char_comb"]),
        float(bundle["wgs_cat"] if wgs is None else wgs),
        float(bundle["c_co2"]),
        float(bundle["p_o2_c"]),
        float(bundle["hl_norm"]),
        float(bundle["hl_ref"]),
        float(bundle["hf"]),
        float(bundle["cp"]),
        float(bundle["hhv_mj"]),
        float(bundle["ts_in_0"]),
        float(bundle["f_ash"]),
        float(bundle["ref_f"]),
        float(bundle["ref_e"]),
        float(bundle["xc0"]),
        float(bundle["f_s_coal"]),
    )
    return np.asarray(profile)


def _outputs(profile: np.ndarray) -> dict[str, float]:
    last = np.asarray(profile[-1], dtype=float)
    return {"CO_mol_s": float(last[2]), "H2_mol_s": float(last[5]), "CO2_mol_s": float(last[3])}


def _residual_quality(profile: np.ndarray, bundle: Mapping[str, Any], *, wgs=None, o2_scale=1.0, coal_scale=1.0) -> float:
    import jax.numpy as jnp
    from .jax_residuals import cell_residuals_jax_flat

    x_prev = np.array(bundle["inlet_12"], dtype=float)
    x_prev[0] *= o2_scale
    x_prev[9] *= coal_scale
    ts = bundle["ts_in_0"]
    g_src = np.array(bundle["g_src_9"]) * coal_scale
    max_abs = 0.0
    for i, x in enumerate(np.asarray(profile)):
        r = np.asarray(
            cell_residuals_jax_flat(
                jnp.asarray(x),
                jnp.asarray(x_prev),
                jnp.asarray(g_src[i]),
                float(bundle["s_src"][i] * coal_scale),
                float(bundle["e_src"][i] * coal_scale),
                float(bundle["dz_list"][i]),
                float(bundle["A"]),
                float(bundle["P"]),
                float(bundle["C_fed"] * coal_scale),
                float(ts),
                float(bundle["coal_flow"] * coal_scale),
                float(bundle["d_p0"]),
                float(bundle["eps"]),
                float(bundle["hl_pct"]),
                float(bundle["L"]),
                float(bundle["char_comb"]),
                float(bundle["wgs_cat"] if wgs is None else wgs),
                float(bundle["c_co2"]),
                float(bundle["p_o2_c"]),
                float(bundle["hl_norm"]),
                float(bundle["hl_ref"]),
                float(bundle["hf"]),
                float(bundle["cp"]),
                float(bundle["hhv_mj"]),
                float(bundle["f_ash"]),
                float(bundle["ref_f"]),
                float(bundle["ref_e"]),
                float(bundle["xc0"]),
                float(bundle["f_s_coal"]),
            )
        )
        max_abs = max(max_abs, float(np.max(np.abs(r))))
        x_prev = x
        ts = float(x[11])
    return max_abs


def _ift_grad(bundle: Mapping[str, Any], parameter: str, output: str) -> float:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from .jax_solver import reactor_solve_v4

    idx = {"CO_mol_s": 2, "H2_mol_s": 5, "CO2_mol_s": 3}[output]
    dz = jnp.asarray(bundle["dz_list"], dtype=jnp.float64)
    zpos = jnp.asarray(bundle["z_positions"], dtype=jnp.float64)
    g_src0 = jnp.asarray(bundle["g_src_9"], dtype=jnp.float64)
    s_src0 = jnp.asarray(bundle["s_src"], dtype=jnp.float64)
    e_src0 = jnp.asarray(bundle["e_src"], dtype=jnp.float64)
    inlet0 = jnp.asarray(bundle["inlet_12"], dtype=jnp.float64)
    guesses0 = jnp.asarray(bundle["guesses"], dtype=jnp.float64)

    def output_fn(theta):
        wgs = bundle["wgs_cat"]
        o2_scale = 1.0
        coal_scale = 1.0
        if parameter == "WGS_CatalyticFactor":
            wgs = theta
        elif parameter == "oxygen_flow":
            o2_scale = theta / bundle["base_o2"]
        else:
            coal_scale = theta / bundle["base_coal"]
        inlet = inlet0.at[0].set(inlet0[0] * o2_scale)
        inlet = inlet.at[9].set(inlet0[9] * coal_scale)
        guesses = guesses0.at[:, 0].set(guesses0[:, 0] * o2_scale)
        guesses = guesses.at[:, 9].set(guesses0[:, 9] * coal_scale)
        profile = reactor_solve_v4(
            inlet,
            dz,
            g_src0 * coal_scale,
            s_src0 * coal_scale,
            e_src0 * coal_scale,
            zpos,
            float(bundle["A"]),
            guesses,
            float(bundle["P"]),
            float(bundle["C_fed"]) * coal_scale,
            float(bundle["coal_flow"]) * coal_scale,
            float(bundle["d_p0"]),
            float(bundle["eps"]),
            float(bundle["hl_pct"]),
            float(bundle["L"]),
            float(bundle["char_comb"]),
            wgs,
            float(bundle["c_co2"]),
            float(bundle["p_o2_c"]),
            float(bundle["hl_norm"]),
            float(bundle["hl_ref"]),
            float(bundle["hf"]),
            float(bundle["cp"]),
            float(bundle["hhv_mj"]),
            float(bundle["ts_in_0"]),
            float(bundle["f_ash"]),
            float(bundle["ref_f"]),
            float(bundle["ref_e"]),
            float(bundle["xc0"]),
            float(bundle["f_s_coal"]),
        )
        return profile[-1, idx]

    if parameter == "WGS_CatalyticFactor":
        theta0 = bundle["wgs_cat"]
    elif parameter == "oxygen_flow":
        theta0 = bundle["base_o2"]
    else:
        theta0 = bundle["base_coal"]
    return float(jax.jacfwd(output_fn)(jnp.asarray(theta0, dtype=jnp.float64)))


def _status(fd: float, ift: float) -> str:
    abs_err = abs(fd - ift)
    scale = max(abs(fd), abs(ift), 0.0)
    if not math.isfinite(fd) or not math.isfinite(ift):
        return "fail_nonfinite"
    if scale < NEAR_ZERO_GRAD:
        return "pass" if abs_err < ABS_ERROR_NEAR_ZERO else "fail_abs_near_zero"
    rel = abs_err / max(scale, 1e-12)
    if rel <= REL_ERROR_PASS or abs_err <= ABS_ERROR_PASS:
        return "pass"
    return "fail"


def run_m5(*, output_dir: Path, runtime_path: Path, configuration: str) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_commit_hash = source_commit()
    frozen_consumed, rows, _ = load_frozen_split()
    if not frozen_consumed:
        raise RuntimeError("frozen split not consumed")
    cases = select_n5_cases(pd.read_csv(runtime_path), rows)
    subset = iter_windows(rows, cases["window_id"])
    parity_rows = []
    step_rows = []
    contract_rows = []
    for row in subset:
        bundle = _bundle(row, configuration)
        contract_rows.append(
            {
                "dataset_id": DATASET_ID,
                "window_id": row["window_id"],
                "phase": row["phase_number"],
                "N_cells": 5,
                "configuration": configuration,
                "jax_seed_from_minimize": False,
                "note": "selected-path N=5 JAX/IFT consistency, not a full-solver differentiability proof",
                "source_commit": source_commit_hash,
            }
        )
        base = _solve_jax(bundle)
        quality = _residual_quality(base, bundle)
        for parameter in PARAMETERS:
            if parameter == "WGS_CatalyticFactor":
                base_theta = bundle["wgs_cat"]
            elif parameter == "oxygen_flow":
                base_theta = bundle["base_o2"]
            else:
                base_theta = bundle["base_coal"]
            for step_frac in STEP_FRACTIONS:
                step = abs(base_theta) * step_frac
                plus = _outputs(
                    _solve_jax(
                        bundle,
                        wgs=base_theta + step if parameter == "WGS_CatalyticFactor" else None,
                        o2_scale=(base_theta + step) / bundle["base_o2"] if parameter == "oxygen_flow" else 1.0,
                        coal_scale=(base_theta + step) / bundle["base_coal"] if parameter == "coal_mass_flow" else 1.0,
                    )
                )
                minus = _outputs(
                    _solve_jax(
                        bundle,
                        wgs=base_theta - step if parameter == "WGS_CatalyticFactor" else None,
                        o2_scale=(base_theta - step) / bundle["base_o2"] if parameter == "oxygen_flow" else 1.0,
                        coal_scale=(base_theta - step) / bundle["base_coal"] if parameter == "coal_mass_flow" else 1.0,
                    )
                )
                for output in OUTPUTS:
                    fd = (plus[output] - minus[output]) / (2.0 * step)
                    step_rows.append(
                        {
                            "dataset_id": DATASET_ID,
                            "window_id": row["window_id"],
                            "parameter": parameter,
                            "output": output,
                            "step_fraction": step_frac,
                            "fd_grad": fd,
                            "source_commit": source_commit_hash,
                        }
                    )
                    if math.isclose(step_frac, PRIMARY_STEP):
                        try:
                            ift = _ift_grad(bundle, parameter, output)
                        except Exception as exc:
                            ift = math.nan
                            status = f"error:{type(exc).__name__}"
                        else:
                            status = _status(fd, ift)
                        parity_rows.append(
                            {
                                "dataset_id": DATASET_ID,
                                "window_id": row["window_id"],
                                "phase": row["phase_number"],
                                "parameter": parameter,
                                "output": output,
                                "parameter_base_value": base_theta,
                                "parameter_step": step,
                                "output_base_value": _outputs(base)[output],
                                "fd_grad": fd,
                                "ift_or_ad_grad": ift,
                                "abs_error": abs(fd - ift) if math.isfinite(fd) and math.isfinite(ift) else math.nan,
                                "rel_error": abs(fd - ift) / max(abs(fd), abs(ift), 1e-12) if math.isfinite(fd) and math.isfinite(ift) else math.nan,
                                "residual_max_abs": quality,
                                "status": status,
                                "source_commit": source_commit_hash,
                            }
                        )
    parity = pd.DataFrame(parity_rows)
    steps = pd.DataFrame(step_rows)
    contract = pd.DataFrame(contract_rows)
    parity.to_csv(output_dir / "revision_m5_n5_gradient_parity.csv", index=False)
    steps.to_csv(output_dir / "revision_m5_fd_step_stability.csv", index=False)
    contract.to_csv(output_dir / "revision_m5_case_contract.csv", index=False)
    n_fail = int((parity["status"] != "pass").sum()) if len(parity) else 0
    md = output_dir / "revision_m5_summary.md"
    md.write_text(
        f"""# Revision M5 N=5 Gradient Parity

- Predeclared thresholds: rel_error <= {REL_ERROR_PASS} or abs_error <= {ABS_ERROR_PASS}; near-zero |grad| < {NEAR_ZERO_GRAD} uses abs_error <= {ABS_ERROR_NEAR_ZERO}
- Primary FD step fraction: {PRIMARY_STEP}
- Failed pairs: {n_fail}/{len(parity)}
- Interpretation: selected-path gradient consistency on the N=5 JAX/IFT path, not a full-solver differentiability proof.

Evidence boundary: {EVIDENCE_BOUNDARY}
""",
        encoding="utf-8",
    )
    manifest = write_sha256_manifest(
        [
            output_dir / "revision_m5_n5_gradient_parity.csv",
            output_dir / "revision_m5_fd_step_stability.csv",
            output_dir / "revision_m5_case_contract.csv",
            md,
        ],
        output_dir / "revision_m5_manifest.sha256",
    )
    return {"summary": md, "manifest": manifest}
