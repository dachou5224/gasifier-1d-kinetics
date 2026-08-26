"""M1 cell-0 residual-quality robustness for the 2026-08-26 revision."""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .revision_evidence_common import (
    DATASET_ID,
    EVIDENCE_BOUNDARY,
    POOR_CONVERGENCE_COST_THRESHOLD,
    SEED,
    TARGETS,
    execution_status_from_exception,
    iter_windows,
    load_frozen_split,
    metric_rows,
    objective_from_predictions,
    prediction_records,
    residual_quality_status,
    solve_model_case,
    source_commit,
    write_sha256_manifest,
    MODEL_ROOT,
)

ALPHA_GRID = (0.05, 0.20, 0.50, 1.00, 2.00)
ALPHA_LOW = 0.02
ALPHA_HIGH = 2.0
FEATURE_COLUMNS = ("o_c_ratio", "pressure_mpa", "syngas_flow_mean")
SLOPE_MAGNITUDE = 0.15
REACTOR_LENGTH_M = 10.3

VARIANT_MATRIX_COLUMNS = (
    "dataset_id",
    "window_id",
    "phase",
    "split_role",
    "configuration",
    "N_cells",
    "selection_reason",
    "o_c_ratio",
    "cell0_cost",
    "cell0_max_abs_scaled_residual",
    "cell0_max_abs_unscaled_residual",
    "cell0_dominant_scaled_name",
    "cell0_dominant_unscaled_name",
    "poor_convergence_count",
    "execution_status",
    "residual_quality_status",
    "wall_time_s",
    "source_commit",
)


def declared_variants(reactor_length_m: float = REACTOR_LENGTH_M) -> dict[str, dict[str, Any]]:
    first_cell = 0.4
    n_refined = 3
    return {
        "reference": {
            "n_cells": 1,
            "solver_options": {},
            "description": "current sequential minimize path, N_cells=1, documented cell-0 multistart",
        },
        "refined_first_cv": {
            "n_cells": n_refined,
            "solver_options": {
                "first_cell_length": first_cell,
                "ignition_zone_res": (float(reactor_length_m) - first_cell) / (n_refined - 1),
            },
            "description": "N_cells=3 with refined first CV; remaining cells fill conserved reactor length",
        },
        "alt_cell0_init": {
            "n_cells": 1,
            "solver_options": {
                "cell0_guess_mode": "oxidation_prestep",
                "cell0_guess_temperatures": [3500.0, 2800.0, 2200.0, 1800.0, 1200.0, 800.0, 400.0],
            },
            "description": "documented JAX-style oxidation prestep plus expanded temperature multistart",
        },
        "tighter_tol": {
            "n_cells": 1,
            "solver_options": {
                "solver_xtol": 1.0e-14,
                "solver_ftol": 1.0e-12,
                "solver_max_nfev": 5000,
            },
            "description": "tighter least_squares xtol/ftol and more function evaluations",
        },
    }


def select_representative_windows(
    runtime: pd.DataFrame,
    rows: Sequence[Mapping[str, Any]],
    *,
    min_count: int = 6,
) -> pd.DataFrame:
    meta = pd.DataFrame(
        [
            {
                "window_id": row["window_id"],
                "phase": int(row["phase_number"]),
                "split_role": row["split_role"],
                "o_c_ratio": float(row["o_c_ratio"]),
            }
            for row in rows
        ]
    )
    work = runtime.merge(meta, on="window_id", how="inner", suffixes=("", "_meta"))
    if "phase_meta" in work.columns:
        work["phase"] = work["phase"].fillna(work["phase_meta"])
    work = work.sort_values(["window_id"]).reset_index(drop=True)
    selected: dict[str, dict[str, Any]] = {}

    def _add(reason: str, record: Mapping[str, Any]) -> None:
        window_id = str(record["window_id"])
        if window_id not in selected:
            selected[window_id] = {
                "window_id": window_id,
                "phase": int(record["phase"]),
                "split_role": record["split_role"],
                "o_c_ratio": float(record["o_c_ratio"]),
                "cell0_cost": float(record["cell0_cost"]),
                "selection_reason": reason,
            }
        else:
            selected[window_id]["selection_reason"] += f";{reason}"

    _add("min_cell0_cost", work.loc[work["cell0_cost"].idxmin()])
    _add("max_cell0_cost", work.loc[work["cell0_cost"].idxmax()])
    for phase, group in work.groupby("phase"):
        target = float(group["o_c_ratio"].median())
        pick = group.iloc[(group["o_c_ratio"] - target).abs().to_numpy().argmin()]
        _add(f"phase{int(phase)}_median_oc", pick)
    _add("global_low_oc", work.loc[work["o_c_ratio"].idxmin()])
    _add("global_high_oc", work.loc[work["o_c_ratio"].idxmax()])
    median_oc = float(work["o_c_ratio"].median())
    _add("global_median_oc", work.iloc[(work["o_c_ratio"] - median_oc).abs().to_numpy().argmin()])
    leftover = work[~work["window_id"].isin(selected)].sort_values("window_id")
    for _, record in leftover.iterrows():
        if len(selected) >= min_count:
            break
        _add("fill_to_minimum", record)
    out = pd.DataFrame(selected.values()).sort_values(["phase", "window_id"]).reset_index(drop=True)
    if set(out["phase"]) < {1, 2, 3, 4}:
        raise RuntimeError("representative window set does not cover all four phases")
    if len(out) < min_count:
        raise RuntimeError("representative window set is smaller than the declared minimum")
    return out


def _safe_solve(row: Mapping[str, Any], variant: Mapping[str, Any], *, alpha_wgs: float = 1.0) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        outcome = solve_model_case(
            row,
            n_cells=int(variant["n_cells"]),
            alpha_wgs=float(alpha_wgs),
            solver_options=variant.get("solver_options") or {},
        )
        outcome["wall_time_s"] = time.perf_counter() - started
        return outcome
    except Exception as exc:
        return {
            "execution_status": execution_status_from_exception(exc),
            "residual_quality_status": "unknown",
            "N_cells": int(variant["n_cells"]),
            "alpha_wgs": float(alpha_wgs),
            "cell0_cost": math.nan,
            "cell0_max_abs_scaled_residual": math.nan,
            "cell0_max_abs_unscaled_residual": math.nan,
            "cell0_dominant_scaled_name": "",
            "cell0_dominant_unscaled_name": "",
            "poor_convergence_count": math.nan,
            "wall_time_s": time.perf_counter() - started,
            "pred_CO_vol_pct": math.nan,
            "pred_H2_vol_pct": math.nan,
            "pred_CO2_vol_pct": math.nan,
            "pred_H2_CO_ratio": math.nan,
            "pred_dry_flow_nm3_h": math.nan,
        }


def run_variant_screen(
    rows: Sequence[Mapping[str, Any]],
    selected: pd.DataFrame,
    *,
    source_commit_hash: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    variants = declared_variants()
    selected_rows = iter_windows(rows, selected["window_id"])
    reason = {row["window_id"]: row["selection_reason"] for row in selected.to_dict("records")}
    matrix_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    for row in selected_rows:
        for name, variant in variants.items():
            outcome = _safe_solve(row, variant)
            matrix_rows.append(
                {
                    "dataset_id": DATASET_ID,
                    "window_id": row["window_id"],
                    "phase": row["phase_number"],
                    "split_role": row["split_role"],
                    "configuration": name,
                    "N_cells": variant["n_cells"],
                    "selection_reason": reason[row["window_id"]],
                    "o_c_ratio": row["o_c_ratio"],
                    "cell0_cost": outcome.get("cell0_cost", math.nan),
                    "cell0_max_abs_scaled_residual": outcome.get("cell0_max_abs_scaled_residual", math.nan),
                    "cell0_max_abs_unscaled_residual": outcome.get("cell0_max_abs_unscaled_residual", math.nan),
                    "cell0_dominant_scaled_name": outcome.get("cell0_dominant_scaled_name", ""),
                    "cell0_dominant_unscaled_name": outcome.get("cell0_dominant_unscaled_name", ""),
                    "poor_convergence_count": outcome.get("poor_convergence_count", math.nan),
                    "execution_status": outcome.get("execution_status", "unknown"),
                    "residual_quality_status": outcome.get("residual_quality_status", "unknown"),
                    "wall_time_s": outcome.get("wall_time_s", math.nan),
                    "source_commit": source_commit_hash,
                }
            )
            names = outcome.get("residual_names") or []
            scaled = outcome.get("residual_scaled") or []
            unscaled = outcome.get("residual_unscaled") or []
            for index, name_i in enumerate(names):
                component_rows.append(
                    {
                        "dataset_id": DATASET_ID,
                        "window_id": row["window_id"],
                        "configuration": name,
                        "residual_name": name_i,
                        "residual_index": index,
                        "scaled_value": float(scaled[index]) if index < len(scaled) else math.nan,
                        "unscaled_value": float(unscaled[index]) if index < len(unscaled) else math.nan,
                        "is_dominant_scaled": name_i == outcome.get("cell0_dominant_scaled_name"),
                        "is_dominant_unscaled": name_i == outcome.get("cell0_dominant_unscaled_name"),
                        "execution_status": outcome.get("execution_status", "unknown"),
                        "source_commit": source_commit_hash,
                    }
                )
    return pd.DataFrame(matrix_rows), pd.DataFrame(component_rows)


def select_configuration(variant_matrix: pd.DataFrame) -> dict[str, Any]:
    summary_rows: list[dict[str, Any]] = []
    for name, group in variant_matrix.groupby("configuration"):
        n_error = int((group["execution_status"].astype(str).str.startswith("error")).sum())
        n_timeout = int((group["execution_status"] == "timeout").sum())
        n_completed = int((group["execution_status"] == "completed").sum())
        poor_rate = float((group["residual_quality_status"] == "poor").mean())
        median_cost = float(group["cell0_cost"].median())
        max_cost = float(group["cell0_cost"].max())
        summary_rows.append(
            {
                "configuration": name,
                "n_completed": n_completed,
                "n_error": n_error,
                "n_timeout": n_timeout,
                "poor_flag_rate": poor_rate,
                "median_cell0_cost": median_cost,
                "max_cell0_cost": max_cost,
                "eligible": n_error == 0 and n_timeout == 0 and n_completed == len(group),
            }
        )
    table = pd.DataFrame(summary_rows).sort_values("configuration").reset_index(drop=True)
    eligible = table[table["eligible"]].copy()
    if eligible.empty:
        chosen = table.sort_values(["n_error", "n_timeout", "poor_flag_rate", "median_cell0_cost"]).iloc[0]
        status = "indeterminate"
    else:
        ref_rows = eligible[eligible["configuration"] == "reference"]
        reference = ref_rows.iloc[0] if len(ref_rows) else eligible.iloc[0]
        material: list[pd.Series] = []
        for _, row in eligible.iterrows():
            if row["configuration"] == reference["configuration"]:
                continue
            poor_drop = float(reference["poor_flag_rate"]) - float(row["poor_flag_rate"])
            rel_cost = (float(reference["median_cell0_cost"]) - float(row["median_cell0_cost"])) / max(
                abs(float(reference["median_cell0_cost"])), 1.0e-15
            )
            if poor_drop > 1.0e-12 or rel_cost > 0.01:
                material.append(row)
        if material:
            chosen = pd.DataFrame(material).sort_values(["poor_flag_rate", "median_cell0_cost", "configuration"]).iloc[0]
            status = "changed"
        else:
            chosen = reference
            status = "stable"
    unresolved = bool(float(chosen["poor_flag_rate"]) >= 1.0 - 1.0e-12)
    if unresolved and str(chosen["configuration"]) != "reference":
        status = "indeterminate"
    return {
        "selected_configuration": str(chosen["configuration"]),
        "gate_status": status,
        "numerical_quality_unresolved": unresolved,
        "summary": table,
        "selection_rule": (
            "zero execution errors/timeouts; a variant is chosen over reference only if it lowers the poor-flag "
            "rate or reduces median cell0 cost by more than 1%; outlet error is not used"
        ),
    }


def _alpha_xt_value(row: Mapping[str, Any], *, feature: str, sign: float, base_alpha: float, stats: Mapping[str, Mapping[str, float]]) -> float:
    mean = stats[feature]["mean"]
    std = max(stats[feature]["std"], 1.0e-12)
    z = (float(row[feature]) - mean) / std
    return float(np.clip(base_alpha * (1.0 + sign * SLOPE_MAGNITUDE * z), ALPHA_LOW, ALPHA_HIGH))


def _train_feature_stats(rows: Sequence[Mapping[str, Any]], features: Sequence[str] | None = None) -> dict[str, dict[str, float]]:
    train = [row for row in rows if row["split_role"] == "train"]
    stats: dict[str, dict[str, float]] = {}
    for feature in features or FEATURE_COLUMNS:
        values = np.asarray([float(row[feature]) for row in train], dtype=float)
        stats[feature] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=0))}
    return stats


def run_full_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    configuration: str,
    source_commit_hash: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    variant = declared_variants()[configuration]
    fixed_records: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        outcome = _safe_solve(row, variant, alpha_wgs=1.0)
        print(f"m1 fixed {index}/{len(rows)} {row['window_id']} {outcome['execution_status']}", flush=True)
        fixed_records.extend(
            prediction_records(
                row,
                outcome,
                method="fixed_physics",
                configuration=configuration,
                source_commit_hash=source_commit_hash,
            )
        )
    fixed = pd.DataFrame(fixed_records)
    train_rows = [row for row in rows if row["split_role"] == "train"]
    selection_rows: list[dict[str, Any]] = []
    global_preds_by_alpha: dict[float, pd.DataFrame] = {}
    for alpha in ALPHA_GRID:
        records: list[dict[str, Any]] = []
        for row in train_rows:
            outcome = _safe_solve(row, variant, alpha_wgs=alpha)
            records.extend(
                prediction_records(
                    row,
                    outcome,
                    method=f"global_bounded_wgs_scalar_grid_{alpha:g}",
                    configuration=configuration,
                    source_commit_hash=source_commit_hash,
                )
            )
        preds = pd.DataFrame(records)
        global_preds_by_alpha[alpha] = preds
        selection_rows.append(
            {
                "dataset_id": DATASET_ID,
                "method": "global_bounded_wgs_scalar",
                "candidate": f"alpha={alpha:g}",
                "alpha_wgs": float(alpha),
                "selection_split": "train",
                "selection_objective": objective_from_predictions(preds, "train"),
                "bound_hit_rate_train": float(
                    (
                        ((preds["alpha_wgs"] - ALPHA_LOW).abs() < 1e-12)
                        | ((preds["alpha_wgs"] - ALPHA_HIGH).abs() < 1e-12)
                    ).mean()
                ),
                "execution_status": "completed" if (preds["execution_status"] == "completed").all() else "partial",
                "selected": False,
                "source_commit": source_commit_hash,
            }
        )
    global_table = pd.DataFrame(selection_rows)
    ok = global_table[global_table["execution_status"] == "completed"]
    selected_global = float(ok.sort_values(["selection_objective", "alpha_wgs"]).iloc[0]["alpha_wgs"]) if len(ok) else float(ALPHA_GRID[0])
    global_table.loc[global_table["alpha_wgs"] == selected_global, "selected"] = True

    stats = _train_feature_stats(rows)
    xt_rows: list[dict[str, Any]] = []
    xt_preds_by_candidate: dict[str, pd.DataFrame] = {}
    candidates = [("constant_global_alpha", None, 0.0)] + [
        (f"{feature}_{'negative' if sign < 0 else 'positive'}_slope", feature, sign)
        for feature in FEATURE_COLUMNS
        for sign in (-1.0, 1.0)
    ]
    for candidate, feature, sign in candidates:
        records = []
        for row in train_rows:
            alpha = selected_global if feature is None else _alpha_xt_value(row, feature=feature, sign=sign, base_alpha=selected_global, stats=stats)
            outcome = _safe_solve(row, variant, alpha_wgs=alpha)
            records.extend(
                prediction_records(
                    row,
                    outcome,
                    method=f"low_capacity_alpha_wgs_xt_{candidate}",
                    configuration=configuration,
                    source_commit_hash=source_commit_hash,
                )
            )
        preds = pd.DataFrame(records)
        xt_preds_by_candidate[candidate] = preds
        xt_rows.append(
            {
                "dataset_id": DATASET_ID,
                "method": "low_capacity_alpha_wgs_xt",
                "candidate": candidate,
                "alpha_wgs": selected_global if feature is None else math.nan,
                "selected_feature": feature or "",
                "selected_sign": sign,
                "base_alpha": selected_global,
                "selection_split": "train",
                "selection_objective": objective_from_predictions(preds, "train"),
                "execution_status": "completed" if (preds["execution_status"] == "completed").all() else "partial",
                "selected": False,
                "source_commit": source_commit_hash,
            }
        )
    xt_table = pd.DataFrame(xt_rows)
    xt_ok = xt_table[xt_table["execution_status"] == "completed"]
    selected_xt = str(xt_ok.sort_values(["selection_objective", "candidate"]).iloc[0]["candidate"]) if len(xt_ok) else "constant_global_alpha"
    xt_table.loc[xt_table["candidate"] == selected_xt, "selected"] = True
    xt_meta = xt_table[xt_table["candidate"] == selected_xt].iloc[0]

    heldout = [row for row in rows if row["split_role"] != "train"]
    global_full = [global_preds_by_alpha[selected_global]]
    for row in heldout:
        outcome = _safe_solve(row, variant, alpha_wgs=selected_global)
        global_full.append(
            pd.DataFrame(
                prediction_records(
                    row,
                    outcome,
                    method="global_bounded_wgs_scalar",
                    configuration=configuration,
                    source_commit_hash=source_commit_hash,
                )
            )
        )
    global_predictions = pd.concat(global_full, ignore_index=True)
    global_predictions["method"] = "global_bounded_wgs_scalar"

    xt_full = [xt_preds_by_candidate[selected_xt]]
    for row in heldout:
        feature = str(xt_meta["selected_feature"]) or None
        sign = float(xt_meta["selected_sign"])
        alpha = selected_global if not feature else _alpha_xt_value(row, feature=feature, sign=sign, base_alpha=selected_global, stats=stats)
        outcome = _safe_solve(row, variant, alpha_wgs=alpha)
        xt_full.append(
            pd.DataFrame(
                prediction_records(
                    row,
                    outcome,
                    method="low_capacity_alpha_wgs_xt",
                    configuration=configuration,
                    source_commit_hash=source_commit_hash,
                )
            )
        )
    xt_predictions = pd.concat(xt_full, ignore_index=True)
    xt_predictions["method"] = "low_capacity_alpha_wgs_xt"

    stability = pd.concat([fixed, global_predictions, xt_predictions], ignore_index=True)
    ref = fixed.rename(columns={"predicted_value": "reference_predicted_value"})[
        ["window_id", "target", "reference_predicted_value"]
    ]
    stability = stability.merge(ref, on=["window_id", "target"], how="left")
    stability["delta_vs_reference"] = stability["predicted_value"] - stability["reference_predicted_value"]
    stability["relative_delta"] = stability["delta_vs_reference"] / stability["reference_predicted_value"].abs().clip(lower=1.0e-12)
    selection = pd.concat([global_table, xt_table], ignore_index=True)
    return stability, selection, metric_rows(stability, method="mixed", source_commit_hash=source_commit_hash)


def assemble_reference_wgs_from_round56(
    rows: Sequence[Mapping[str, Any]],
    fixed: pd.DataFrame,
    *,
    round56_dir: Path,
    source_commit_hash: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selection = pd.read_csv(round56_dir / "huayi_round56_selection_and_bounds.csv")
    selection["source_commit"] = source_commit_hash
    selection["dataset_id"] = DATASET_ID
    global_preds = pd.read_csv(round56_dir / "_global_selected_full.csv")
    xt_preds = pd.read_csv(round56_dir / "_alpha_xt_selected_full.csv")
    converted: list[pd.DataFrame] = [fixed]
    for frame, method in ((global_preds, "global_bounded_wgs_scalar"), (xt_preds, "low_capacity_alpha_wgs_xt")):
        work = frame.copy()
        work["method"] = method
        work["configuration"] = "reference"
        work["execution_status"] = work["status"].map(lambda s: "completed" if s == "ok" else str(s))
        work["residual_quality_status"] = work["cell0_cost"].map(residual_quality_status)
        work["cell0_max_abs_scaled_residual"] = math.nan
        work["cell0_max_abs_unscaled_residual"] = math.nan
        work["source_commit"] = source_commit_hash
        keep = [
            col
            for col in (
                "dataset_id",
                "window_id",
                "phase",
                "split_role",
                "method",
                "configuration",
                "N_cells",
                "target",
                "target_value",
                "predicted_value",
                "residual",
                "alpha_wgs",
                "poor_convergence_count",
                "cell0_cost",
                "cell0_max_abs_scaled_residual",
                "cell0_max_abs_unscaled_residual",
                "execution_status",
                "residual_quality_status",
                "source_commit",
            )
            if col in work.columns
        ]
        converted.append(work[keep])
    stability = pd.concat(converted, ignore_index=True)
    ref = fixed.rename(columns={"predicted_value": "reference_predicted_value"})[["window_id", "target", "reference_predicted_value"]]
    stability = stability.merge(ref, on=["window_id", "target"], how="left")
    stability["delta_vs_reference"] = stability["predicted_value"] - stability["reference_predicted_value"]
    stability["relative_delta"] = stability["delta_vs_reference"] / stability["reference_predicted_value"].abs().clip(lower=1.0e-12)
    return stability, selection


def run_fixed_fullsplit(rows: Sequence[Mapping[str, Any]], *, configuration: str, source_commit_hash: str) -> pd.DataFrame:
    variant = declared_variants()[configuration]
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        outcome = _safe_solve(row, variant, alpha_wgs=1.0)
        print(f"m1 fixed {index}/{len(rows)} {row['window_id']} {outcome['execution_status']}", flush=True)
        records.extend(
            prediction_records(
                row,
                outcome,
                method="fixed_physics",
                configuration=configuration,
                source_commit_hash=source_commit_hash,
            )
        )
    return pd.DataFrame(records)


def write_m1_summary(
    *,
    output_dir: Path,
    frozen_consumed: bool,
    selected_windows: pd.DataFrame,
    selection: Mapping[str, Any],
    stability: pd.DataFrame | None,
    source_commit_hash: str,
) -> Path:
    selected_name = str(selection["selected_configuration"])
    unresolved = bool(selection["numerical_quality_unresolved"])
    gate = str(selection["gate_status"])
    if unresolved:
        quality_line = (
            "No screened configuration removed or materially reduced the cell-0 quality flag. "
            "The numerical-quality concern remains unresolved and manuscript claims that treat the baseline as a fully converged first-cell solve should be reduced."
        )
    else:
        quality_line = "The selected configuration reduced the cell-0 poor-quality flag rate on the representative screen."
    alpha_line = "WGS reselection was not run in this invocation."
    if stability is not None and not stability.empty:
        wgs = stability[stability["method"] == "global_bounded_wgs_scalar"]
        if not wgs.empty:
            alphas = sorted(set(float(v) for v in wgs["alpha_wgs"].dropna().unique()))
            alpha_line = f"Train-only reselection under {selected_name} produced global alpha values {alphas}."
    text = f"""# Revision M1 Cell-0 Residual-Quality Robustness

- Dataset: `{DATASET_ID}`
- Frozen split consumed: {frozen_consumed}
- Source commit: `{source_commit_hash}`
- Seed: {SEED} (not used for variant selection)
- Poor-quality threshold: cost > {POOR_CONVERGENCE_COST_THRESHOLD}
- Representative windows: {len(selected_windows)} covering phases {sorted(set(int(v) for v in selected_windows['phase']))}
- Selection rule: {selection['selection_rule']}
- Selected configuration: `{selected_name}`
- Gate status: `{gate}`
- Numerical-quality unresolved: {unresolved}

{quality_line}

{alpha_line}

Evidence boundary: {EVIDENCE_BOUNDARY}

Forbidden claims: no production closed-loop control; no learned WGS kinetic identification; no learned multi-alpha success; no z_d value; no oxidation-channel updating; no cross-gasifier generalization.
"""
    path = output_dir / "revision_m1_summary.md"
    path.write_text(text, encoding="utf-8")
    return path


def run_m1(
    *,
    output_dir: Path,
    runtime_path: Path,
    full_split: bool = True,
    reuse_screen: bool = False,
    reuse_round56_wgs: bool = True,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_commit_hash = source_commit()
    frozen_consumed, rows, _primary = load_frozen_split()
    if not frozen_consumed:
        raise RuntimeError("frozen R52-C-75 split was not consumed")
    runtime = pd.read_csv(runtime_path)
    selected = select_representative_windows(runtime, rows)
    variant_path = output_dir / "revision_m1_cell0_variant_matrix.csv"
    component_path = output_dir / "revision_m1_cell0_residual_components.csv"
    if reuse_screen and variant_path.exists() and component_path.exists():
        variant_matrix = pd.read_csv(variant_path)
        components = pd.read_csv(component_path)
    else:
        variant_matrix, components = run_variant_screen(rows, selected, source_commit_hash=source_commit_hash)
        variant_matrix.to_csv(variant_path, index=False)
        components.to_csv(component_path, index=False)
    selection = select_configuration(variant_matrix)
    stability = None
    selection_table = selection["summary"].copy()
    selection_table["selected_configuration"] = selection["selected_configuration"]
    selection_table["gate_status"] = selection["gate_status"]
    selection_table["numerical_quality_unresolved"] = selection["numerical_quality_unresolved"]
    selection_table["source_commit"] = source_commit_hash
    selection_table.to_csv(output_dir / "revision_m1_screen_selection.csv", index=False)
    if full_split:
        config_name = str(selection["selected_configuration"])
        fixed = run_fixed_fullsplit(rows, configuration=config_name, source_commit_hash=source_commit_hash)
        round56_dir = MODEL_ROOT / "outputs" / "diagnostics" / "round56"
        if reuse_round56_wgs and config_name == "reference" and (round56_dir / "huayi_round56_selection_and_bounds.csv").exists():
            stability, wgs_selection = assemble_reference_wgs_from_round56(
                rows,
                fixed,
                round56_dir=round56_dir,
                source_commit_hash=source_commit_hash,
            )
        else:
            stability, wgs_selection, _metrics = run_full_split(
                rows,
                configuration=config_name,
                source_commit_hash=source_commit_hash,
            )
        wgs_selection.to_csv(output_dir / "revision_m1_selection_stability.csv", index=False)
        stability.to_csv(output_dir / "revision_m1_fullsplit_stability.csv", index=False)
    else:
        selection_table.to_csv(output_dir / "revision_m1_selection_stability.csv", index=False)
        pd.DataFrame(columns=list(VARIANT_MATRIX_COLUMNS) + ["target", "predicted_value", "delta_vs_reference", "relative_delta", "alpha_wgs"]).to_csv(
            output_dir / "revision_m1_fullsplit_stability.csv",
            index=False,
        )
    summary = write_m1_summary(
        output_dir=output_dir,
        frozen_consumed=frozen_consumed,
        selected_windows=selected,
        selection=selection,
        stability=stability,
        source_commit_hash=source_commit_hash,
    )
    artifacts = [
        output_dir / "revision_m1_cell0_variant_matrix.csv",
        output_dir / "revision_m1_cell0_residual_components.csv",
        output_dir / "revision_m1_fullsplit_stability.csv",
        output_dir / "revision_m1_selection_stability.csv",
        summary,
    ]
    manifest = write_sha256_manifest(artifacts, output_dir / "revision_m1_manifest.sha256")
    return {"summary": summary, "manifest": manifest, "variant_matrix": artifacts[0]}
