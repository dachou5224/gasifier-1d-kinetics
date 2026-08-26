"""M4 scaled sensitivity / local estimability diagnostic."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .alpha_candidate_identifiability import candidate_alpha_registry, inject_alpha
from .gasifier_system import GasifierSystem
from .model_case_adapter import build_gasifier_inputs_from_model_case
from .revision_evidence_common import (
    DATASET_ID,
    EVIDENCE_BOUNDARY,
    apply_solver_options,
    load_frozen_split,
    source_commit,
    write_sha256_manifest,
)
from .revision_m1_cell0 import declared_variants, iter_windows, select_representative_windows

TARGET_SCALES = {
    "CO_volpct_dry": 10.0,
    "H2_volpct_dry": 10.0,
    "CO2_volpct_dry": 10.0,
    "H2_CO_ratio": 0.1,
}
RANK_TOL_REL = 1.0e-3
PROFILE_GRID = (0.05, 0.10, 0.20, 0.50, 1.00, 1.50, 2.00)


def _kpis(profile: np.ndarray) -> dict[str, float]:
    last = np.asarray(profile[-1], dtype=float)
    gas = last[:8]
    dry = float(np.sum(gas[:7])) + 1e-12
    co = float(gas[2] / dry * 100.0)
    h2 = float(gas[5] / dry * 100.0)
    return {
        "CO_volpct_dry": co,
        "H2_volpct_dry": h2,
        "CO2_volpct_dry": float(gas[3] / dry * 100.0),
        "H2_CO_ratio": h2 / max(co, 1e-12),
    }


def _solve(row: Mapping[str, Any], op_conds: Mapping[str, Any], n_cells: int) -> dict[str, float]:
    built = build_gasifier_inputs_from_model_case(row)
    system = GasifierSystem(built["geometry"], built["coal_props"], dict(op_conds))
    profile, _z = system.solve(N_cells=int(n_cells), solver_method="minimize", jacobian_mode="scipy", jax_warmup=False)
    return _kpis(profile)


def scaled_sensitivity_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    configuration: str,
    source_commit_hash: str,
) -> pd.DataFrame:
    variant = declared_variants()[configuration]
    registry = {item["alpha_name"]: item for item in candidate_alpha_registry()}
    out: list[dict[str, Any]] = []
    for row in rows:
        built = build_gasifier_inputs_from_model_case(row)
        base_op = apply_solver_options(built["op_conds"], variant.get("solver_options") or {})
        for name, spec in registry.items():
            record = {
                "dataset_id": DATASET_ID,
                "window_id": row["window_id"],
                "phase": row["phase_number"],
                "split_role": row["split_role"],
                "alpha_name": name,
                "feasibility_status": spec["feasibility_status"],
                "source_commit": source_commit_hash,
            }
            if not str(spec["feasibility_status"]).startswith("feasible"):
                record.update({"status": "unavailable", "evidence_field": "insufficiently_supported"})
                out.append(record)
                continue
            minus = float(spec["baseline_alpha"]) - float(spec["perturbation"])
            plus = float(spec["baseline_alpha"]) + float(spec["perturbation"])
            minus_k = _solve(row, inject_alpha(base_op, name, minus), variant["n_cells"])
            plus_k = _solve(row, inject_alpha(base_op, name, plus), variant["n_cells"])
            for target, scale in TARGET_SCALES.items():
                grad = (plus_k[target] - minus_k[target]) / (plus - minus)
                scaled = grad * (float(spec["perturbation"]) / scale)
                out.append(
                    {
                        **record,
                        "target": target,
                        "fd_gradient": float(grad),
                        "scaled_sensitivity": float(scaled),
                        "target_scale": scale,
                        "parameter_scale": float(spec["perturbation"]),
                        "status": "completed",
                        "evidence_field": "pending_svd",
                    }
                )
    return pd.DataFrame(out)


def singular_spectrum(sensitivity: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feasible = sensitivity[sensitivity["status"] == "completed"].copy()
    alphas = [name for name in ("alpha_wgs", "alpha_char_oxidation", "alpha_volatile_oxidation") if name in set(feasible["alpha_name"])]
    targets = list(TARGET_SCALES)
    pooled = []
    for target in targets:
        row_vals = []
        for alpha in alphas:
            sub = feasible[(feasible["target"] == target) & (feasible["alpha_name"] == alpha)]
            row_vals.append(float(sub["scaled_sensitivity"].mean()) if len(sub) else math.nan)
        pooled.append(row_vals)
    matrix = np.asarray(pooled, dtype=float)
    finite = np.isfinite(matrix)
    matrix = np.where(finite, matrix, 0.0)
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    sigma_max = float(s[0]) if s.size else math.nan
    numerical_rank = int(np.sum(s > RANK_TOL_REL * sigma_max)) if math.isfinite(sigma_max) and sigma_max > 0 else 0
    cond = float(s[0] / s[-1]) if s.size and s[-1] > 0 else math.inf
    spectrum = pd.DataFrame(
        [
            {
                "dataset_id": DATASET_ID,
                "component": i,
                "singular_value": float(val),
                "relative_to_max": float(val / sigma_max) if sigma_max else math.nan,
                "rank_tolerance": RANK_TOL_REL,
                "above_tolerance": bool(val > RANK_TOL_REL * sigma_max) if sigma_max else False,
            }
            for i, val in enumerate(s)
        ]
    )
    loadings = []
    for j, alpha in enumerate(alphas):
        mean_abs = float(np.mean(np.abs(feasible.loc[feasible["alpha_name"] == alpha, "scaled_sensitivity"])))
        channel = "locally_estimable_under_selected_measurements" if mean_abs > 1e-3 and numerical_rank >= 1 else "insufficiently_supported"
        loadings.append(
            {
                "dataset_id": DATASET_ID,
                "alpha_name": alpha,
                "mean_abs_scaled_sensitivity": mean_abs,
                "right_singular_vector_1": float(vt[0, j]) if vt.size else math.nan,
                "numerical_rank": numerical_rank,
                "condition_number": cond,
                "evidence_field": channel,
                "identifiable_flag_forbidden": False,
            }
        )
    loadings.append(
        {
            "dataset_id": DATASET_ID,
            "alpha_name": "alpha_char_gasification",
            "mean_abs_scaled_sensitivity": math.nan,
            "right_singular_vector_1": math.nan,
            "numerical_rank": numerical_rank,
            "condition_number": cond,
            "evidence_field": "insufficiently_supported",
            "identifiable_flag_forbidden": False,
        }
    )
    return spectrum, pd.DataFrame(loadings)


def profile_objectives(rows: Sequence[Mapping[str, Any]], *, configuration: str, source_commit_hash: str) -> pd.DataFrame:
    variant = declared_variants()[configuration]
    out = []
    for alpha_name in ("alpha_wgs", "alpha_char_oxidation"):
        for row in rows[: min(4, len(rows))]:
            built = build_gasifier_inputs_from_model_case(row)
            base_op = apply_solver_options(built["op_conds"], variant.get("solver_options") or {})
            for value in PROFILE_GRID:
                kpis = _solve(row, inject_alpha(base_op, alpha_name, value), variant["n_cells"])
                loss = (
                    ((kpis["CO_volpct_dry"] - float(row["co_vol_pct"])) / 10.0) ** 2
                    + ((kpis["H2_volpct_dry"] - float(row["h2_vol_pct"])) / 10.0) ** 2
                    + ((kpis["CO2_volpct_dry"] - float(row["co2_vol_pct"])) / 10.0) ** 2
                )
                out.append(
                    {
                        "dataset_id": DATASET_ID,
                        "window_id": row["window_id"],
                        "alpha_name": alpha_name,
                        "alpha_value": value,
                        "profile_objective": float(loss),
                        "note": "diagnostic profile, not a likelihood confidence interval",
                        "source_commit": source_commit_hash,
                    }
                )
    return pd.DataFrame(out)


def run_m4(*, output_dir: Path, runtime_path: Path, configuration: str) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_commit_hash = source_commit()
    frozen_consumed, rows, _ = load_frozen_split()
    if not frozen_consumed:
        raise RuntimeError("frozen split not consumed")
    selected = select_representative_windows(pd.read_csv(runtime_path), rows)
    subset = iter_windows(rows, selected["window_id"])
    sensitivity = scaled_sensitivity_rows(subset, configuration=configuration, source_commit_hash=source_commit_hash)
    spectrum, summary = singular_spectrum(sensitivity)
    profiles = profile_objectives(subset, configuration=configuration, source_commit_hash=source_commit_hash)
    sensitivity.to_csv(output_dir / "revision_m4_scaled_sensitivity_matrix.csv", index=False)
    spectrum.to_csv(output_dir / "revision_m4_singular_spectrum.csv", index=False)
    summary.to_csv(output_dir / "revision_m4_estimability_summary.csv", index=False)
    profiles.to_csv(output_dir / "revision_m4_profile_objectives.csv", index=False)
    md = output_dir / "revision_m4_summary.md"
    md.write_text(
        f"""# Revision M4 Local Estimability

- Scaling: target scales {TARGET_SCALES}; parameter scale = declared perturbation
- Rank tolerance: relative singular value > {RANK_TOL_REL}
- Configuration: `{configuration}`
- Cosine similarity is not used as an identifiability decision.

Channel evidence:
{summary[['alpha_name', 'evidence_field']].to_string(index=False)}

Evidence boundary: {EVIDENCE_BOUNDARY}
""",
        encoding="utf-8",
    )
    manifest = write_sha256_manifest(
        [
            output_dir / "revision_m4_scaled_sensitivity_matrix.csv",
            output_dir / "revision_m4_singular_spectrum.csv",
            output_dir / "revision_m4_estimability_summary.csv",
            output_dir / "revision_m4_profile_objectives.csv",
            md,
        ],
        output_dir / "revision_m4_manifest.sha256",
    )
    return {"summary": md, "manifest": manifest}
