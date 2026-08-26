"""
Candidate alpha identifiability screen before multi-alpha training.

This diagnostic perturbs feasible reaction-family scalar paths one at a time
under the frozen Huayi split.  It does not train a multi-alpha model.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .gasifier_system import GasifierSystem
from .model_case_adapter import build_gasifier_inputs_from_model_case
from .model_case_diagnostic_baseline import POOR_CONVERGENCE_COST_THRESHOLD, cell_residual_quality

IDENTIFIABILITY_EVIDENCE_BOUNDARY = (
    "candidate alpha identifiability screen only; not multi-alpha training, "
    "learned multiplier interpretation proof, weak-label coal-quality "
    "modeling, production control, or final temporal-generalization evidence"
)

TARGETS: tuple[str, ...] = ("CO_volpct_dry", "H2_volpct_dry", "CO2_volpct_dry", "H2_CO_ratio")

SENSITIVITY_COLUMNS: tuple[str, ...] = (
    "alpha_name",
    "mapped_parameters",
    "feasibility_status",
    "bound_low",
    "bound_high",
    "alpha_center",
    "alpha_minus",
    "alpha_plus",
    "case_id",
    "phase",
    "split",
    "target",
    "target_value",
    "prediction_minus",
    "prediction_plus",
    "fd_gradient",
    "abs_gradient",
    "gradient_sign",
    "N_cells",
    "status_minus",
    "status_plus",
    "fallback_count_minus",
    "fallback_count_plus",
    "poor_convergence_count_minus",
    "poor_convergence_count_plus",
    "poor_convergence_cell_index_minus",
    "poor_convergence_cell_index_plus",
    "cell0_cost_minus",
    "cell0_cost_plus",
    "near_lower_bound",
    "near_upper_bound",
    "evidence_boundary",
)

SIMILARITY_COLUMNS: tuple[str, ...] = (
    "alpha_i",
    "alpha_j",
    "cosine_similarity",
    "pearson_correlation",
    "shared_vector_length",
    "mean_abs_gradient_i",
    "mean_abs_gradient_j",
    "confounding_flag",
    "evidence_boundary",
)

RECOMMENDATION_COLUMNS: tuple[str, ...] = (
    "alpha_name",
    "mapped_parameters",
    "feasibility_status",
    "bounds",
    "baseline_alpha",
    "mean_abs_gradient",
    "max_abs_gradient",
    "min_split_sign_consistency",
    "max_abs_cosine_to_other",
    "solver_failure_count",
    "poor_convergence_count_sum",
    "bound_range_sanity",
    "recommendation",
    "rationale",
    "evidence_boundary",
)


def candidate_alpha_registry() -> list[dict[str, Any]]:
    return [
        {
            "alpha_name": "alpha_wgs",
            "mapped_parameters": "op_conds.WGS_CatalyticFactor",
            "feasibility_status": "feasible_existing_op_conds",
            "bound_low": 0.02,
            "bound_high": 2.0,
            "baseline_alpha": 0.3,
            "perturbation": 0.03,
            "group": "water-gas-shift net family",
        },
        {
            "alpha_name": "alpha_char_oxidation",
            "mapped_parameters": "op_conds.CharCombustionRateFactor",
            "feasibility_status": "feasible_existing_op_conds",
            "bound_low": 0.05,
            "bound_high": 1.0,
            "baseline_alpha": 0.3,
            "perturbation": 0.03,
            "group": "char oxidation C+O2",
        },
        {
            "alpha_name": "alpha_volatile_oxidation",
            "mapped_parameters": "op_conds.CO_OxidationFactor + op_conds.H2_OxidationFactor",
            "feasibility_status": "feasible_existing_op_conds_shared_scalar",
            "bound_low": 0.2,
            "bound_high": 1.2,
            "baseline_alpha": 1.0,
            "perturbation": 0.1,
            "group": "combustion-zone volatile CO/H2 oxidation",
        },
        {
            "alpha_name": "alpha_char_gasification",
            "mapped_parameters": "no clean grouped op_conds path",
            "feasibility_status": "design_only_no_clean_grouped_path",
            "bound_low": 0.1,
            "bound_high": 5.0,
            "baseline_alpha": math.nan,
            "perturbation": math.nan,
            "group": "char gasification C+CO2 and C+H2O",
        },
    ]


def feasible_candidates() -> list[dict[str, Any]]:
    return [row for row in candidate_alpha_registry() if row["feasibility_status"].startswith("feasible")]


def inject_alpha(op_conds: Mapping[str, Any], alpha_name: str, value: float) -> dict[str, Any]:
    out = dict(op_conds)
    value = float(value)
    if alpha_name == "alpha_wgs":
        out["WGS_CatalyticFactor"] = value
    elif alpha_name == "alpha_char_oxidation":
        out["CharCombustionRateFactor"] = value
    elif alpha_name == "alpha_volatile_oxidation":
        out["CO_OxidationFactor"] = value
        out["H2_OxidationFactor"] = value
    else:
        raise ValueError(f"No clean injection path for {alpha_name}")
    return out


def _dry_syngas_kpis(profile: np.ndarray) -> dict[str, float]:
    last = np.asarray(profile[-1], dtype=float)
    gas = last[:8]
    dry = float(np.sum(gas[:7]))
    denom = dry + 1.0e-12
    co = float(gas[2] / denom * 100.0)
    h2 = float(gas[5] / denom * 100.0)
    return {
        "CO_volpct_dry": co,
        "H2_volpct_dry": h2,
        "CO2_volpct_dry": float(gas[3] / denom * 100.0),
        "H2_CO_ratio": h2 / max(co, 1.0e-12),
    }


def _solve_with_alpha(model_case_row: Mapping[str, Any], alpha_name: str, alpha_value: float, n_cells: int) -> dict[str, Any]:
    built = build_gasifier_inputs_from_model_case(model_case_row)
    op_conds = inject_alpha(built["op_conds"], alpha_name, alpha_value)
    start = time.perf_counter()
    system = GasifierSystem(built["geometry"], built["coal_props"], op_conds)
    profile, _z = system.solve(N_cells=int(n_cells), solver_method="minimize", jacobian_mode="scipy", jax_warmup=False)
    runtime_s = time.perf_counter() - start
    residual_quality = cell_residual_quality(system, profile)
    poor_cells = [item["cell_index"] for item in residual_quality if item["cost"] > POOR_CONVERGENCE_COST_THRESHOLD]
    cell0 = residual_quality[0] if residual_quality else {}
    targets = built["targets"]
    kpis = _dry_syngas_kpis(profile)
    out = {
        "case_id": built["case_id"],
        "phase": model_case_row.get("phase"),
        "dataset_id": model_case_row.get("dataset_id"),
        "split": model_case_row.get("_split", model_case_row.get("split")),
        "split_role": model_case_row.get("_split_role", model_case_row.get("split_role")),
        "runtime_s": runtime_s,
        "fallback_count": int(system.solve_stats.get("fallback_count", 0)),
        "poor_convergence_count": int(system.solve_stats.get("poor_convergence_count", 0)),
        "poor_convergence_cell_index": ";".join(str(i) for i in poor_cells),
        "cell0_cost": float(cell0.get("cost", math.nan)),
        "solve_status": "ok",
        "target_CO_volpct_dry": float(targets["outlet_co_dry_vol_pct"]),
        "target_H2_volpct_dry": float(targets["outlet_h2_dry_vol_pct"]),
        "target_CO2_volpct_dry": float(targets["outlet_co2_dry_vol_pct"]),
    }
    out["target_H2_CO_ratio"] = out["target_H2_volpct_dry"] / max(out["target_CO_volpct_dry"], 1.0e-12)
    out.update(kpis)
    return out


SolverFn = Callable[[Mapping[str, Any], str, float, int], Mapping[str, Any]]


def _safe_solve(solver_fn: SolverFn, row: Mapping[str, Any], alpha_name: str, alpha_value: float, n_cells: int) -> dict[str, Any]:
    try:
        return dict(solver_fn(row, alpha_name, alpha_value, n_cells))
    except Exception as exc:  # pragma: no cover - diagnostic failure path
        return {
            "case_id": row.get("_model_case_id", row.get("case_id", "unknown")),
            "phase": row.get("phase"),
            "split": row.get("_split", row.get("split")),
            "solve_status": f"failed:{type(exc).__name__}:{exc}",
            "fallback_count": math.nan,
            "poor_convergence_count": math.nan,
            "poor_convergence_cell_index": "",
            "cell0_cost": math.nan,
        }


def _target_values(result: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    return {
        "CO_volpct_dry": (float(result["target_CO_volpct_dry"]), float(result["CO_volpct_dry"])),
        "H2_volpct_dry": (float(result["target_H2_volpct_dry"]), float(result["H2_volpct_dry"])),
        "CO2_volpct_dry": (float(result["target_CO2_volpct_dry"]), float(result["CO2_volpct_dry"])),
        "H2_CO_ratio": (float(result["target_H2_CO_ratio"]), float(result["H2_CO_ratio"])),
    }


def alpha_perturbation_points(candidate: Mapping[str, Any]) -> tuple[float, float]:
    center = float(candidate["baseline_alpha"])
    step = float(candidate["perturbation"])
    low = float(candidate["bound_low"])
    high = float(candidate["bound_high"])
    return max(low, center - step), min(high, center + step)


def alpha_candidate_sensitivity_rows(
    model_case_rows_with_split: Sequence[Mapping[str, Any]],
    *,
    n_cells: int = 3,
    solver_fn: SolverFn = _solve_with_alpha,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in feasible_candidates():
        alpha_minus, alpha_plus = alpha_perturbation_points(candidate)
        denom = alpha_plus - alpha_minus
        near_lower = math.isclose(alpha_minus, float(candidate["bound_low"]), rel_tol=0.0, abs_tol=1.0e-12)
        near_upper = math.isclose(alpha_plus, float(candidate["bound_high"]), rel_tol=0.0, abs_tol=1.0e-12)
        for model_case in model_case_rows_with_split:
            minus = _safe_solve(solver_fn, model_case, candidate["alpha_name"], alpha_minus, n_cells)
            plus = _safe_solve(solver_fn, model_case, candidate["alpha_name"], alpha_plus, n_cells)
            if not str(minus.get("solve_status", "ok")).startswith("ok") or not str(plus.get("solve_status", "ok")).startswith("ok"):
                for target in TARGETS:
                    rows.append(
                        {
                            "alpha_name": candidate["alpha_name"],
                            "mapped_parameters": candidate["mapped_parameters"],
                            "feasibility_status": candidate["feasibility_status"],
                            "bound_low": candidate["bound_low"],
                            "bound_high": candidate["bound_high"],
                            "alpha_center": candidate["baseline_alpha"],
                            "alpha_minus": alpha_minus,
                            "alpha_plus": alpha_plus,
                            "case_id": model_case.get("_model_case_id", model_case.get("case_id")),
                            "phase": model_case.get("phase"),
                            "split": model_case.get("_split", model_case.get("split")),
                            "target": target,
                            "target_value": math.nan,
                            "prediction_minus": math.nan,
                            "prediction_plus": math.nan,
                            "fd_gradient": math.nan,
                            "abs_gradient": math.nan,
                            "gradient_sign": "failed",
                            "N_cells": int(n_cells),
                            "status_minus": minus.get("solve_status", "failed"),
                            "status_plus": plus.get("solve_status", "failed"),
                            "fallback_count_minus": minus.get("fallback_count", math.nan),
                            "fallback_count_plus": plus.get("fallback_count", math.nan),
                            "poor_convergence_count_minus": minus.get("poor_convergence_count", math.nan),
                            "poor_convergence_count_plus": plus.get("poor_convergence_count", math.nan),
                            "poor_convergence_cell_index_minus": minus.get("poor_convergence_cell_index", ""),
                            "poor_convergence_cell_index_plus": plus.get("poor_convergence_cell_index", ""),
                            "cell0_cost_minus": minus.get("cell0_cost", math.nan),
                            "cell0_cost_plus": plus.get("cell0_cost", math.nan),
                            "near_lower_bound": near_lower,
                            "near_upper_bound": near_upper,
                            "evidence_boundary": IDENTIFIABILITY_EVIDENCE_BOUNDARY,
                        }
                    )
                continue
            minus_values = _target_values(minus)
            plus_values = _target_values(plus)
            for target in TARGETS:
                target_value, prediction_minus = minus_values[target]
                prediction_plus = plus_values[target][1]
                gradient = (prediction_plus - prediction_minus) / denom
                rows.append(
                    {
                        "alpha_name": candidate["alpha_name"],
                        "mapped_parameters": candidate["mapped_parameters"],
                        "feasibility_status": candidate["feasibility_status"],
                        "bound_low": candidate["bound_low"],
                        "bound_high": candidate["bound_high"],
                        "alpha_center": candidate["baseline_alpha"],
                        "alpha_minus": alpha_minus,
                        "alpha_plus": alpha_plus,
                        "case_id": minus["case_id"],
                        "phase": minus.get("phase"),
                        "split": minus.get("split"),
                        "target": target,
                        "target_value": target_value,
                        "prediction_minus": prediction_minus,
                        "prediction_plus": prediction_plus,
                        "fd_gradient": gradient,
                        "abs_gradient": abs(gradient),
                        "gradient_sign": "positive" if gradient > 0 else "negative" if gradient < 0 else "zero",
                        "N_cells": int(n_cells),
                        "status_minus": minus.get("solve_status", "ok"),
                        "status_plus": plus.get("solve_status", "ok"),
                        "fallback_count_minus": minus.get("fallback_count", 0),
                        "fallback_count_plus": plus.get("fallback_count", 0),
                        "poor_convergence_count_minus": minus.get("poor_convergence_count", 0),
                        "poor_convergence_count_plus": plus.get("poor_convergence_count", 0),
                        "poor_convergence_cell_index_minus": minus.get("poor_convergence_cell_index", ""),
                        "poor_convergence_cell_index_plus": plus.get("poor_convergence_cell_index", ""),
                        "cell0_cost_minus": minus.get("cell0_cost", math.nan),
                        "cell0_cost_plus": plus.get("cell0_cost", math.nan),
                        "near_lower_bound": near_lower,
                        "near_upper_bound": near_upper,
                        "evidence_boundary": IDENTIFIABILITY_EVIDENCE_BOUNDARY,
                    }
                )
    return [{column: row.get(column) for column in SENSITIVITY_COLUMNS} for row in rows]


def _gradient_vector(df: pd.DataFrame, alpha_name: str) -> np.ndarray:
    sub = df[df["alpha_name"] == alpha_name].sort_values(["case_id", "target"])
    return sub["fd_gradient"].astype(float).to_numpy()


def alpha_candidate_similarity_rows(sensitivity_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame(sensitivity_rows)
    rows: list[dict[str, Any]] = []
    names = sorted(df["alpha_name"].dropna().unique())
    for left in names:
        v_left = _gradient_vector(df, left)
        for right in names:
            v_right = _gradient_vector(df, right)
            if len(v_left) != len(v_right) or len(v_left) == 0:
                cosine = math.nan
                corr = math.nan
            else:
                denom = float(np.linalg.norm(v_left) * np.linalg.norm(v_right))
                cosine = float(np.dot(v_left, v_right) / denom) if denom > 0 else math.nan
                corr = float(np.corrcoef(v_left, v_right)[0, 1]) if len(v_left) > 1 and np.std(v_left) > 0 and np.std(v_right) > 0 else math.nan
            flag = "self" if left == right else "likely_confounded" if math.isfinite(cosine) and abs(cosine) >= 0.85 else "moderate_overlap" if math.isfinite(cosine) and abs(cosine) >= 0.6 else "separable_or_weak"
            rows.append(
                {
                    "alpha_i": left,
                    "alpha_j": right,
                    "cosine_similarity": cosine,
                    "pearson_correlation": corr,
                    "shared_vector_length": int(min(len(v_left), len(v_right))),
                    "mean_abs_gradient_i": float(np.nanmean(np.abs(v_left))) if len(v_left) else math.nan,
                    "mean_abs_gradient_j": float(np.nanmean(np.abs(v_right))) if len(v_right) else math.nan,
                    "confounding_flag": flag,
                    "evidence_boundary": IDENTIFIABILITY_EVIDENCE_BOUNDARY,
                }
            )
    return [{column: row.get(column) for column in SIMILARITY_COLUMNS} for row in rows]


def _min_split_sign_consistency(group: pd.DataFrame) -> float:
    values: list[float] = []
    for _split, split_group in group.groupby("split"):
        signs = split_group["gradient_sign"]
        nonzero = signs[signs != "zero"]
        if nonzero.empty:
            values.append(0.0)
            continue
        counts = nonzero.value_counts()
        values.append(float(counts.max() / len(nonzero)))
    return min(values) if values else 0.0


def alpha_candidate_recommendation_rows(
    sensitivity_rows: Sequence[Mapping[str, Any]],
    similarity_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    sens = pd.DataFrame(sensitivity_rows)
    sim = pd.DataFrame(similarity_rows)
    rows: list[dict[str, Any]] = []
    for candidate in candidate_alpha_registry():
        alpha_name = candidate["alpha_name"]
        if candidate["feasibility_status"].startswith("design_only"):
            rows.append(
                {
                    "alpha_name": alpha_name,
                    "mapped_parameters": candidate["mapped_parameters"],
                    "feasibility_status": candidate["feasibility_status"],
                    "bounds": f"[{candidate['bound_low']},{candidate['bound_high']}]",
                    "baseline_alpha": candidate["baseline_alpha"],
                    "mean_abs_gradient": math.nan,
                    "max_abs_gradient": math.nan,
                    "min_split_sign_consistency": math.nan,
                    "max_abs_cosine_to_other": math.nan,
                    "solver_failure_count": math.nan,
                    "poor_convergence_count_sum": math.nan,
                    "bound_range_sanity": "design_only_not_perturbed",
                    "recommendation": "design-only",
                    "rationale": "No clean grouped op_conds path exists without changing solver semantics; keep out of multi-alpha training until an explicit grouped interface is designed.",
                    "evidence_boundary": IDENTIFIABILITY_EVIDENCE_BOUNDARY,
                }
            )
            continue
        group = sens[sens["alpha_name"] == alpha_name]
        finite = group[np.isfinite(group["fd_gradient"].astype(float))]
        mean_abs = float(finite["abs_gradient"].mean()) if not finite.empty else math.nan
        max_abs = float(finite["abs_gradient"].max()) if not finite.empty else math.nan
        sign_consistency = _min_split_sign_consistency(finite) if not finite.empty else 0.0
        sim_group = sim[(sim["alpha_i"] == alpha_name) & (sim["alpha_j"] != alpha_name)]
        max_cos = float(sim_group["cosine_similarity"].abs().max()) if not sim_group.empty else math.nan
        failures = int((~group["status_minus"].astype(str).str.startswith("ok")).sum() + (~group["status_plus"].astype(str).str.startswith("ok")).sum())
        poor = int(pd.to_numeric(group["poor_convergence_count_minus"], errors="coerce").fillna(0).sum() + pd.to_numeric(group["poor_convergence_count_plus"], errors="coerce").fillna(0).sum())
        near_bound = bool(group["near_lower_bound"].any() or group["near_upper_bound"].any())
        if failures:
            recommendation = "exclude"
            rationale = "Local perturbation caused solver failures; do not include before path-specific diagnosis."
        elif mean_abs < 1.0e-3:
            recommendation = "exclude"
            rationale = "Outlet sensitivity is target-insensitive at the screened perturbation scale."
        elif math.isfinite(max_cos) and max_cos >= 0.85:
            recommendation = "merge"
            rationale = "Sensitivity vector is nearly collinear with another feasible alpha; use merged or staged ablation rather than simultaneous training."
        elif sign_consistency < 0.75:
            recommendation = "include-with-caution"
            rationale = "Sensitivity is non-negligible but split/target signs are mixed, so interpretability is weak and compensation risk is high."
        else:
            recommendation = "include"
            rationale = "Sensitivity is non-negligible, solver path is clean, and split-level signs are comparatively stable."
        rows.append(
            {
                "alpha_name": alpha_name,
                "mapped_parameters": candidate["mapped_parameters"],
                "feasibility_status": candidate["feasibility_status"],
                "bounds": f"[{candidate['bound_low']},{candidate['bound_high']}]",
                "baseline_alpha": candidate["baseline_alpha"],
                "mean_abs_gradient": mean_abs,
                "max_abs_gradient": max_abs,
                "min_split_sign_consistency": sign_consistency,
                "max_abs_cosine_to_other": max_cos,
                "solver_failure_count": failures,
                "poor_convergence_count_sum": poor,
                "bound_range_sanity": "near_bound" if near_bound else "interior_local_perturbation",
                "recommendation": recommendation,
                "rationale": rationale,
                "evidence_boundary": IDENTIFIABILITY_EVIDENCE_BOUNDARY,
            }
        )
    return [{column: row.get(column) for column in RECOMMENDATION_COLUMNS} for row in rows]


def identifiability_markdown(
    sensitivity_rows: Sequence[Mapping[str, Any]],
    similarity_rows: Sequence[Mapping[str, Any]],
    recommendation_rows: Sequence[Mapping[str, Any]],
) -> str:
    sens = pd.DataFrame(sensitivity_rows)
    sim = pd.DataFrame(similarity_rows)
    rec = pd.DataFrame(recommendation_rows)
    lines = [
        "# Huayi alpha candidate identifiability screen",
        "",
        "This fixed-design screen perturbs feasible alpha groups one at a time under the frozen split.",
        "",
        "## Per-alpha sensitivity headline",
        "",
        "| alpha | mean | max | min split sign consistency | failures | recommendation |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for _, row in rec.iterrows():
        lines.append(
            "| {alpha} | {mean} | {maxv} | {consistency} | {failures} | {rec} |".format(
                alpha=row["alpha_name"],
                mean="nan" if pd.isna(row["mean_abs_gradient"]) else f"{float(row['mean_abs_gradient']):.6g}",
                maxv="nan" if pd.isna(row["max_abs_gradient"]) else f"{float(row['max_abs_gradient']):.6g}",
                consistency="nan" if pd.isna(row["min_split_sign_consistency"]) else f"{float(row['min_split_sign_consistency']):.6g}",
                failures="nan" if pd.isna(row["solver_failure_count"]) else f"{int(row['solver_failure_count'])}",
                rec=row["recommendation"],
            )
        )
    lines.extend(["", "## Cross-alpha similarity", "", "| alpha_i | alpha_j | cosine | Pearson | flag |", "|---|---|---:|---:|---|"])
    for _, row in sim[sim["alpha_i"] < sim["alpha_j"]].iterrows():
        lines.append(
            f"| {row['alpha_i']} | {row['alpha_j']} | {float(row['cosine_similarity']):.6g} | {float(row['pearson_correlation']):.6g} | {row['confounding_flag']} |"
        )
    lines.extend(["", "## Split sign consistency", "", "| alpha | split | target | positive | negative | median gradient |", "|---|---|---|---:|---:|---:|"])
    for (alpha, split, target), group in sens.groupby(["alpha_name", "split", "target"]):
        lines.append(
            "| {alpha} | {split} | {target} | {pos} | {neg} | {median:.6g} |".format(
                alpha=alpha,
                split=split,
                target=target,
                pos=int((group["fd_gradient"] > 0).sum()),
                neg=int((group["fd_gradient"] < 0).sum()),
                median=float(group["fd_gradient"].median()),
            )
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            f"Evidence boundary: {IDENTIFIABILITY_EVIDENCE_BOUNDARY}.",
            "",
            "Recommendations are for the next diagnostic experiment only. They are not final manuscript claims.",
        ]
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "IDENTIFIABILITY_EVIDENCE_BOUNDARY",
    "RECOMMENDATION_COLUMNS",
    "SENSITIVITY_COLUMNS",
    "SIMILARITY_COLUMNS",
    "alpha_candidate_recommendation_rows",
    "alpha_candidate_sensitivity_rows",
    "alpha_candidate_similarity_rows",
    "candidate_alpha_registry",
    "identifiability_markdown",
    "inject_alpha",
]
