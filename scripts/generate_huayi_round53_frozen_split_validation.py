#!/usr/bin/env python3
"""Round53 frozen-split bounded WGS predictive-validation package.

This script consumes the Round52 R52-C-75 split as immutable input.  It keeps
window IDs, split roles, target definitions and metric aggregation fixed.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DATA_PREP_ROOT = Path("/Users/liuzhen/AI-projects/gasifier-data-prep")
ROUND52_DIR = DATA_PREP_ROOT / "outputs" / "round52"
PRIMARY_DATASET_ID = "R52-C-75"
N_CELLS = 1
WGS_ALPHA_LOW = 0.02
WGS_ALPHA_HIGH = 2.0
ALPHA_GRID = (0.05, 0.20, 0.50, 1.00, 2.00)
TARGETS = (
    ("CO_dry_vol_pct", "co_vol_pct", "pred_CO_vol_pct"),
    ("H2_dry_vol_pct", "h2_vol_pct", "pred_H2_vol_pct"),
    ("CO2_dry_vol_pct", "co2_vol_pct", "pred_CO2_vol_pct"),
    ("H2_CO_ratio", "h2_co_ratio", "pred_H2_CO_ratio"),
    ("dry_syngas_flow", "syngas_flow_mean", "pred_dry_flow_nm3_h"),
)
FIXED_WEIGHTS = {
    "CO_dry_vol_pct": 1.0,
    "H2_dry_vol_pct": 1.0,
    "CO2_dry_vol_pct": 1.0,
    "H2_CO_ratio": 1.0,
    "dry_syngas_flow": 1.0,
}
EVIDENCE_ROLE = "frozen_round52_predictive_validation"
ALLOWED_CLAIM = (
    "bounded WGS adaptive channel evaluated under frozen Huayi R52-C-75 split; "
    "predictive usefulness is limited to this industrial Huayi case"
)
FORBIDDEN_CLAIM = (
    "no learned WGS kinetic identification, no multi-alpha, no z_d, no "
    "oxidation-channel updating, no production control, no cross-gasifier generalization"
)


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _install_model_package_aliases() -> None:
    src_dir = ROOT / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _coal_paths() -> dict[str, Path]:
    return {
        "v0_baseline_coal_assumption": DATA_PREP_ROOT / "config" / "coal_assumption_v0.yaml",
        "huayi_p2_sy3sy2_coal_assumption": DATA_PREP_ROOT / "config" / "coal_assumption_huayi_p2.yaml",
        "huayi_p3_sy3sy2_coal_assumption": DATA_PREP_ROOT / "config" / "coal_assumption_huayi_p3.yaml",
        "huayi_p4_sy3sy2_coal_assumption": DATA_PREP_ROOT / "config" / "coal_assumption_huayi_p4.yaml",
    }


def _phase_label(phase: int) -> str:
    return f"phase{int(phase)}"


def _split_label(split_role: str) -> str:
    if split_role == "train":
        return "train"
    if split_role == "validation":
        return "validation"
    if split_role == "test":
        return "test"
    raise ValueError(f"Unsupported split role: {split_role}")


def _normal_m3_h_to_kmol_s(value_nm3_h: float) -> float:
    return float(value_nm3_h) / 22.414 / 3600.0


def _model_case_rows(primary: pd.DataFrame) -> list[dict[str, Any]]:
    coal_by_id = {key: _read_yaml(path) for key, path in _coal_paths().items()}
    rows: list[dict[str, Any]] = []
    for _, row in primary.sort_values(["split_role", "start_time", "window_id"]).iterrows():
        coal = coal_by_id[str(row["coal_assumption_id"])]
        ultimate = coal["ultimate_analysis_dry"]
        proximate = coal["proximate_analysis"]
        case = {
            "case_id": row["window_id"],
            "case_id_value": row["window_id"],
            "segment_id": row["parent_segment_id"],
            "independence_group_id": row["parent_independence_group_id"],
            "feature_version": row["coal_assumption_id"],
            "model_case_ready": bool(row["model_case_ready"]),
            "phase": _phase_label(int(row["phase"])),
            "phase_number": int(row["phase"]),
            "dataset_id": row["dataset_id"],
            "split": _split_label(str(row["split_role"])),
            "split_role": str(row["split_role"]),
            "window_id": row["window_id"],
            "start_time": row["start_time"],
            "end_time": row["end_time"],
            "reactor_bed_diameter_m_value": 3.8,
            "reactor_bed_height_m_value": 10.3,
            "coal_carbon_mass_fraction_adb_value": ultimate["carbon"]["value"],
            "coal_hydrogen_mass_fraction_adb_value": ultimate["hydrogen"]["value"],
            "coal_oxygen_mass_fraction_adb_value": ultimate["oxygen"]["value"],
            "coal_nitrogen_mass_fraction_adb_value": ultimate["nitrogen"]["value"],
            "coal_sulfur_mass_fraction_adb_value": ultimate["sulfur"]["value"],
            "coal_ash_proximate_d_value": proximate["ash_d"]["value"],
            "coal_volatile_matter_d_value": proximate["volatile_matter_d"]["value"],
            "coal_fixed_carbon_d_value": proximate["fixed_carbon_d"]["value"],
            "coal_heating_value_dry_kj_kg_value": coal["heating_value_dry_kj_kg"]["value"],
            "coal_mass_flow_value": row["coal_flow_mean"],
            "o2_volumetric_flow_value": row["o2_flow_mean"],
            "co2_carrier_volumetric_flow_value": max(float(row["o2_flow_mean"]) * 0.24, 0.0),
            "gasifier_pressure_value": row["pressure_mean"],
            "o2_inlet_temperature_c_value": 160.0,
            "co2_carrier_inlet_temperature_c_value": 80.0,
            "syngas_dry_volumetric_flow_value": row["syngas_flow_mean"],
            "outlet_co_dry_vol_pct_value": row["co_vol_pct"],
            "outlet_h2_dry_vol_pct_value": row["h2_vol_pct"],
            "outlet_co2_dry_vol_pct_value": row["co2_vol_pct"],
            "outlet_ch4_dry_vol_pct_value": row.get("ch4_vol_pct", 0.0),
            "outlet_n2_dry_vol_pct_value": 0.0,
            "outlet_h2s_dry_vol_pct_value": 0.0,
            "co_vol_pct": row["co_vol_pct"],
            "h2_vol_pct": row["h2_vol_pct"],
            "co2_vol_pct": row["co2_vol_pct"],
            "h2_co_ratio": row["h2_co_ratio"],
            "syngas_flow_mean": row["syngas_flow_mean"],
        }
        rows.append(case)
    return rows


def _target_scale(target: str, rows: pd.DataFrame) -> float:
    if target == "dry_syngas_flow":
        return max(float(rows["syngas_flow_mean"].mean()), 1.0)
    if target == "H2_CO_ratio":
        return 0.1
    return 10.0


def _prediction_targets_from_result(result: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, float]:
    return {
        "pred_CO_vol_pct": float(result["CO_volpct_dry"]),
        "pred_H2_vol_pct": float(result["H2_volpct_dry"]),
        "pred_CO2_vol_pct": float(result["CO2_volpct_dry"]),
        "pred_H2_CO_ratio": float(result["H2_CO_ratio"]),
        "pred_dry_flow_nm3_h": float(row["syngas_dry_volumetric_flow_value"]),
    }


def _safe_solve(row: Mapping[str, Any], alpha: float, solve_with_wgs_correction) -> dict[str, Any]:
    try:
        result = dict(solve_with_wgs_correction(row, float(alpha), n_cells=N_CELLS))
        result["status"] = "ok"
        return result
    except Exception as exc:
        return {
            "case_id": row["case_id"],
            "phase": row["phase"],
            "dataset_id": row["dataset_id"],
            "fallback_count": 1,
            "poor_convergence_count": 1,
            "poor_convergence_cell_index": "",
            "cell0_cost": math.nan,
            "status": f"failed:{type(exc).__name__}:{exc}",
            "CO_volpct_dry": math.nan,
            "H2_volpct_dry": math.nan,
            "CO2_volpct_dry": math.nan,
            "H2_CO_ratio": math.nan,
        }


def _solve_rows(rows: Sequence[Mapping[str, Any]], alpha_by_case: Mapping[str, float], solve_with_wgs_correction) -> pd.DataFrame:
    out: list[dict[str, Any]] = []
    for row in rows:
        alpha = float(alpha_by_case[str(row["case_id"])])
        result = _safe_solve(row, alpha, solve_with_wgs_correction)
        preds = _prediction_targets_from_result(result, row)
        out.append(
            {
                "case_id": row["case_id"],
                "window_id": row["window_id"],
                "dataset_id": row["dataset_id"],
                "phase": row["phase_number"],
                "split_role": row["split_role"],
                "split": row["split"],
                "alpha_wgs": alpha,
                "bound_hit": math.isclose(alpha, WGS_ALPHA_LOW) or math.isclose(alpha, WGS_ALPHA_HIGH),
                "fallback_count": int(result.get("fallback_count", 0) or 0),
                "poor_convergence_count": int(result.get("poor_convergence_count", 0) or 0),
                "poor_convergence_cell_index": result.get("poor_convergence_cell_index", ""),
                "cell0_cost": result.get("cell0_cost", math.nan),
                "status": result.get("status", result.get("solve_status", "ok")),
                **{source: float(row[source]) for _, source, _ in TARGETS},
                **preds,
            }
        )
    return pd.DataFrame(out)


def _target_prediction_rows(solved: pd.DataFrame, method: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in solved.iterrows():
        for target, true_col, pred_col in TARGETS:
            y_true = float(row[true_col])
            y_pred = float(row[pred_col])
            rows.append(
                {
                    "dataset_id": row["dataset_id"],
                    "window_id": row["window_id"],
                    "split_role": row["split_role"],
                    "phase": int(row["phase"]),
                    "method": method,
                    "target": target,
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "residual": y_pred - y_true,
                    "alpha_wgs": float(row["alpha_wgs"]),
                    "bound_hit": bool(row["bound_hit"]),
                    "fallback_count": int(row["fallback_count"]),
                    "poor_convergence_count": int(row["poor_convergence_count"]),
                    "cell0_cost": row["cell0_cost"],
                    "status": row["status"],
                }
            )
    return pd.DataFrame(rows)


def _metric_matrix(predictions: pd.DataFrame, method: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (split_role, phase, target), group in predictions.groupby(["split_role", "phase", "target"], dropna=False):
        resid = group["residual"].astype(float).to_numpy()
        metrics = {
            "rmse": float(np.sqrt(np.mean(resid * resid))),
            "mae": float(np.mean(np.abs(resid))),
            "bias": float(np.mean(resid)),
            "n": int(len(group)),
        }
        for metric_name, metric_value in metrics.items():
            rows.append(
                {
                    "dataset_id": PRIMARY_DATASET_ID,
                    "window_id": "AGGREGATE",
                    "split_role": split_role,
                    "phase": int(phase),
                    "method": method,
                    "target": target,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "evidence_role": EVIDENCE_ROLE,
                    "allowed_claim": ALLOWED_CLAIM,
                    "forbidden_claim": FORBIDDEN_CLAIM,
                }
            )
    for (split_role, target), group in predictions.groupby(["split_role", "target"], dropna=False):
        resid = group["residual"].astype(float).to_numpy()
        for metric_name, metric_value in {
            "rmse": float(np.sqrt(np.mean(resid * resid))),
            "mae": float(np.mean(np.abs(resid))),
            "bias": float(np.mean(resid)),
            "n": int(len(group)),
        }.items():
            rows.append(
                {
                    "dataset_id": PRIMARY_DATASET_ID,
                    "window_id": "AGGREGATE",
                    "split_role": split_role,
                    "phase": "ALL",
                    "method": method,
                    "target": target,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "evidence_role": EVIDENCE_ROLE,
                    "allowed_claim": ALLOWED_CLAIM,
                    "forbidden_claim": FORBIDDEN_CLAIM,
                }
            )
    return pd.DataFrame(rows)


def _aggregate_score(predictions: pd.DataFrame, source_rows: pd.DataFrame) -> dict[str, float]:
    score: dict[str, float] = {}
    for split_role, group in predictions.groupby("split_role"):
        total = 0.0
        for target, true_col, _pred_col in TARGETS:
            target_group = group[group["target"] == target]
            scale = _target_scale(target, source_rows)
            rmse = math.sqrt(float(np.mean(np.square(target_group["residual"].astype(float) / scale))))
            total += FIXED_WEIGHTS[target] * rmse
        score[str(split_role)] = total / sum(FIXED_WEIGHTS.values())
    return score


def _select_global_alpha(train_rows: Sequence[Mapping[str, Any]], solve_with_wgs_correction) -> tuple[float, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    for alpha in ALPHA_GRID:
        solved = _solve_rows(train_rows, {str(row["case_id"]): alpha for row in train_rows}, solve_with_wgs_correction)
        preds = _target_prediction_rows(solved, "global_bounded_wgs_scalar")
        score = _aggregate_score(preds, pd.DataFrame(train_rows))["train"]
        records.append({"alpha_wgs": alpha, "train_score": score, "selection_split": "train"})
    grid = pd.DataFrame(records)
    selected = grid.sort_values(["train_score", "alpha_wgs"]).iloc[0]
    return float(selected["alpha_wgs"]), grid


def _fit_direct_residual(train_predictions: pd.DataFrame) -> dict[str, float]:
    return train_predictions.groupby("target")["residual"].mean().astype(float).to_dict()


def _apply_direct_residual(fixed_predictions: pd.DataFrame, corrections: Mapping[str, float]) -> pd.DataFrame:
    out = fixed_predictions.copy()
    out["y_pred"] = out["y_pred"] - out["target"].map(corrections).astype(float)
    out["residual"] = out["y_pred"] - out["y_true"]
    out["method"] = "direct_residual_train_mean_comparator"
    return out


def _feature_table(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rows:
        records.append(
            {
                "case_id": row["case_id"],
                "split": row["split"],
                "split_role": row["split_role"],
                "phase": row["phase_number"],
                "o_c_ratio": float(row["o2_volumetric_flow_value"]) / max(float(row["coal_mass_flow_value"]), 1e-9),
                "coal_flow": float(row["coal_mass_flow_value"]),
                "o2_flow": float(row["o2_volumetric_flow_value"]),
                "pressure": float(row["gasifier_pressure_value"]),
                "syngas_flow": float(row["syngas_dry_volumetric_flow_value"]),
            }
        )
    return pd.DataFrame(records)


def _select_alpha_xt(
    rows: Sequence[Mapping[str, Any]],
    train_rows: Sequence[Mapping[str, Any]],
    solve_with_wgs_correction,
) -> tuple[dict[str, Any], pd.DataFrame, Mapping[str, float]]:
    features = _feature_table(rows)
    train_features = features[features["split"] == "train"].copy()
    columns = ["o_c_ratio", "pressure", "syngas_flow"]
    mean = train_features[columns].mean()
    scale = train_features[columns].std(ddof=0).replace(0.0, 1.0)
    candidate_models: list[dict[str, Any]] = [
        {"name": "constant_selected_global_alpha", "base_alpha": None, "feature": "", "sign": 0.0},
        {"name": "constant_alpha_1p0_control", "base_alpha": 1.0, "feature": "", "sign": 0.0},
    ]
    for feature in columns:
        candidate_models.append({"name": f"{feature}_negative_slope", "base_alpha": None, "feature": feature, "sign": -1.0})
        candidate_models.append({"name": f"{feature}_positive_slope", "base_alpha": None, "feature": feature, "sign": 1.0})

    global_alpha, _grid = _select_global_alpha(train_rows, solve_with_wgs_correction)

    def alpha_map_for(model: Mapping[str, Any]) -> dict[str, float]:
        base = global_alpha if model["base_alpha"] is None else float(model["base_alpha"])
        values: dict[str, float] = {}
        for _, feature_row in features.iterrows():
            alpha = base
            if model["feature"]:
                z = (float(feature_row[model["feature"]]) - float(mean[model["feature"]])) / float(scale[model["feature"]])
                alpha = base * math.exp(0.15 * float(model["sign"]) * z)
            values[str(feature_row["case_id"])] = min(max(alpha, WGS_ALPHA_LOW), WGS_ALPHA_HIGH)
        return values

    records: list[dict[str, Any]] = []
    maps: dict[str, Mapping[str, float]] = {}
    for model in candidate_models:
        amap = alpha_map_for(model)
        maps[str(model["name"])] = amap
        solved = _solve_rows(train_rows, amap, solve_with_wgs_correction)
        preds = _target_prediction_rows(solved, "low_capacity_alpha_wgs_xt")
        score = _aggregate_score(preds, pd.DataFrame(train_rows))["train"]
        records.append(
            {
                "candidate": model["name"],
                "train_score": score,
                "selected_on": "train",
                "feature": model["feature"],
                "sign": model["sign"],
                "alpha_min_train": min(amap[str(row["case_id"])] for row in train_rows),
                "alpha_max_train": max(amap[str(row["case_id"])] for row in train_rows),
            }
        )
    table = pd.DataFrame(records)
    selected = table.sort_values(["train_score", "candidate"]).iloc[0].to_dict()
    return selected, table, maps[str(selected["candidate"])]


def _null_control_summary(
    fixed_predictions: pd.DataFrame,
    adaptive_predictions: pd.DataFrame,
    train_predictions: pd.DataFrame,
) -> pd.DataFrame:
    shuffled = train_predictions.copy()
    rng = np.random.default_rng(5300)
    rows: list[dict[str, Any]] = []
    for target, group in shuffled.groupby("target"):
        residuals = group["residual"].to_numpy().copy()
        rng.shuffle(residuals)
        rows.append(
            {
                "dataset_id": PRIMARY_DATASET_ID,
                "window_id": "AGGREGATE",
                "split_role": "train",
                "phase": "ALL",
                "method": "null_shuffled_train_residuals",
                "target": target,
                "metric_name": "rmse",
                "metric_value": float(np.sqrt(np.mean(residuals * residuals))),
                "evidence_role": "null_control",
                "allowed_claim": "null/control guardrail only",
                "forbidden_claim": FORBIDDEN_CLAIM,
            }
        )
    for split_role in ("validation", "test"):
        base = fixed_predictions[fixed_predictions["split_role"] == split_role]
        adaptive = adaptive_predictions[adaptive_predictions["split_role"] == split_role]
        for target in sorted(base["target"].unique()):
            b = base[base["target"] == target]["residual"].astype(float).to_numpy()
            a = adaptive[adaptive["target"] == target]["residual"].astype(float).to_numpy()
            rows.append(
                {
                    "dataset_id": PRIMARY_DATASET_ID,
                    "window_id": "AGGREGATE",
                    "split_role": split_role,
                    "phase": "ALL",
                    "method": "adaptive_minus_fixed_control_delta",
                    "target": target,
                    "metric_name": "rmse_delta",
                    "metric_value": float(np.sqrt(np.mean(a * a)) - np.sqrt(np.mean(b * b))),
                    "evidence_role": "null_control",
                    "allowed_claim": "negative delta means adaptive beats fixed for this target",
                    "forbidden_claim": FORBIDDEN_CLAIM,
                }
            )
    return pd.DataFrame(rows)


def _worsened_targets(fixed: pd.DataFrame, adaptive: pd.DataFrame, split_role: str) -> list[str]:
    bad: list[str] = []
    for target in sorted(fixed["target"].unique()):
        f = fixed[(fixed["split_role"] == split_role) & (fixed["target"] == target)]["residual"].astype(float).to_numpy()
        a = adaptive[(adaptive["split_role"] == split_role) & (adaptive["target"] == target)]["residual"].astype(float).to_numpy()
        if len(f) and len(a) and math.sqrt(float(np.mean(a * a))) > math.sqrt(float(np.mean(f * f))) * 1.02:
            bad.append(target)
    return bad


def _convergence_table(solved_by_method: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method, solved in solved_by_method.items():
        for split_role, group in solved.groupby("split_role"):
            rows.append(
                {
                    "dataset_id": PRIMARY_DATASET_ID,
                    "window_id": "AGGREGATE",
                    "split_role": split_role,
                    "phase": "ALL",
                    "method": method,
                    "target": "ALL",
                    "metric_name": "fallback_count",
                    "metric_value": int(group["fallback_count"].sum()),
                    "evidence_role": EVIDENCE_ROLE,
                    "allowed_claim": ALLOWED_CLAIM,
                    "forbidden_claim": FORBIDDEN_CLAIM,
                }
            )
            rows.append(
                {
                    "dataset_id": PRIMARY_DATASET_ID,
                    "window_id": "AGGREGATE",
                    "split_role": split_role,
                    "phase": "ALL",
                    "method": method,
                    "target": "ALL",
                    "metric_name": "bound_hit_rate",
                    "metric_value": float(group["bound_hit"].mean()),
                    "evidence_role": EVIDENCE_ROLE,
                    "allowed_claim": ALLOWED_CLAIM,
                    "forbidden_claim": FORBIDDEN_CLAIM,
                }
            )
            rows.append(
                {
                    "dataset_id": PRIMARY_DATASET_ID,
                    "window_id": "AGGREGATE",
                    "split_role": split_role,
                    "phase": "ALL",
                    "method": method,
                    "target": "ALL",
                    "metric_name": "poor_convergence_count",
                    "metric_value": int(group["poor_convergence_count"].sum()),
                    "evidence_role": EVIDENCE_ROLE,
                    "allowed_claim": ALLOWED_CLAIM,
                    "forbidden_claim": FORBIDDEN_CLAIM,
                }
            )
    return pd.DataFrame(rows)


def _write_markdown(path: Path, lines: Sequence[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _base_metric_row(
    *,
    method: str,
    split_role: str,
    phase: Any,
    target: str,
    metric_name: str,
    metric_value: Any,
    evidence_role: str = EVIDENCE_ROLE,
) -> dict[str, Any]:
    return {
        "dataset_id": PRIMARY_DATASET_ID,
        "window_id": "AGGREGATE",
        "split_role": split_role,
        "phase": phase,
        "method": method,
        "target": target,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "evidence_role": evidence_role,
        "allowed_claim": ALLOWED_CLAIM,
        "forbidden_claim": FORBIDDEN_CLAIM,
    }


def _write_unavailable_package(output_dir: Path, primary: pd.DataFrame, consumed_ok: bool) -> dict[str, Any]:
    methods = [
        "fixed_physics_unavailable_existing_api_timeout",
        "direct_residual_comparator_unavailable_depends_on_fixed_physics",
        "global_bounded_wgs_scalar_unavailable_existing_api_timeout",
        "low_capacity_alpha_wgs_xt_unavailable_existing_api_timeout",
    ]
    targets = [target for target, _true, _pred in TARGETS]
    baseline_methods = methods[:3]
    adaptive_methods = methods[3:]

    def rows_for(selected_methods: Sequence[str]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for method in selected_methods:
            for split_role, group in primary.groupby("split_role"):
                phase = "ALL"
                for target in targets:
                    for metric_name in ("rmse", "mae", "bias"):
                        rows.append(
                            _base_metric_row(
                                method=method,
                                split_role=str(split_role),
                                phase=phase,
                                target=target,
                                metric_name=metric_name,
                                metric_value=math.nan,
                                evidence_role="unavailable_existing_solver_runtime",
                            )
                        )
                    rows.append(
                        _base_metric_row(
                            method=method,
                            split_role=str(split_role),
                            phase=phase,
                            target=target,
                            metric_name="n",
                            metric_value=int(len(group)),
                            evidence_role="frozen_split_intake_count",
                        )
                    )
        return pd.DataFrame(rows)

    baseline = rows_for(baseline_methods)
    adaptive = rows_for(adaptive_methods)
    baseline.to_csv(output_dir / "huayi_round53_baseline_metric_matrix.csv", index=False)
    adaptive.to_csv(output_dir / "huayi_round53_wgs_adaptive_metric_matrix.csv", index=False)

    summary_rows = []
    for method in methods:
        for split_role, group in primary.groupby("split_role"):
            summary_rows.append(
                _base_metric_row(
                    method=method,
                    split_role=str(split_role),
                    phase="ALL",
                    target="FIXED_WEIGHT_AGGREGATE",
                    metric_name="aggregate_score",
                    metric_value=math.nan,
                    evidence_role="unavailable_existing_solver_runtime",
                )
            )
            summary_rows.append(
                _base_metric_row(
                    method=method,
                    split_role=str(split_role),
                    phase="ALL",
                    target="ROW_COUNT",
                    metric_name="n",
                    metric_value=int(len(group)),
                    evidence_role="frozen_split_intake_count",
                )
            )
    pd.DataFrame(summary_rows).to_csv(output_dir / "huayi_round53_test_shadow_summary.csv", index=False)

    null_rows = [
        _base_metric_row(
            method="null_control_unavailable_without_completed_fixed_predictions",
            split_role=split_role,
            phase="ALL",
            target=target,
            metric_name="status",
            metric_value=math.nan,
            evidence_role="null_control_blocked",
        )
        for split_role in ("train", "validation", "test")
        for target in targets
    ]
    pd.DataFrame(null_rows).to_csv(output_dir / "huayi_round53_null_control_summary.csv", index=False)

    convergence_rows = []
    for method in methods:
        for split_role, group in primary.groupby("split_role"):
            convergence_rows.extend(
                [
                    _base_metric_row(
                        method=method,
                        split_role=str(split_role),
                        phase="ALL",
                        target="ALL",
                        metric_name="attempted_rows",
                        metric_value=int(len(group)),
                        evidence_role="runtime_audit",
                    ),
                    _base_metric_row(
                        method=method,
                        split_role=str(split_role),
                        phase="ALL",
                        target="ALL",
                        metric_name="completed_rows",
                        metric_value=0,
                        evidence_role="runtime_audit",
                    ),
                    _base_metric_row(
                        method=method,
                        split_role=str(split_role),
                        phase="ALL",
                        target="ALL",
                        metric_name="fallback_count",
                        metric_value=math.nan,
                        evidence_role="runtime_audit",
                    ),
                    _base_metric_row(
                        method=method,
                        split_role=str(split_role),
                        phase="ALL",
                        target="ALL",
                        metric_name="bound_hit_rate",
                        metric_value=math.nan,
                        evidence_role="runtime_audit",
                    ),
                    _base_metric_row(
                        method=method,
                        split_role=str(split_role),
                        phase="ALL",
                        target="ALL",
                        metric_name="convergence_failure_rate",
                        metric_value=math.nan,
                        evidence_role="runtime_audit",
                    ),
                ]
            )
    pd.DataFrame(convergence_rows).to_csv(output_dir / "huayi_round53_convergence_and_bounds.csv", index=False)

    _write_markdown(
        output_dir / "huayi_round53_frozen_split_manifest_intake.md",
        [
            "# Huayi Round53 Frozen Split Intake",
            "",
            f"- Source split: `{ROUND52_DIR / 'huayi_round52_split_freeze.yaml'}`",
            f"- Selected primary dataset: `{PRIMARY_DATASET_ID}`",
            f"- Frozen split consumed: {consumed_ok}",
            "- Train/validation/test rows: 81/44/17",
            "- Split rule preserved: phase 1+2 train, phase 3 validation, phase 4 test/shadow.",
            "- Window rules, target definitions and metric aggregation were not changed after model evaluation.",
        ],
    )
    _write_markdown(
        output_dir / "huayi_round53_claim_boundary.md",
        [
            "# Huayi Round53 Claim Boundary",
            "",
            "Decision: **MODEL_VALIDATION_NO_GO**",
            "",
            "Blocker:",
            "",
            "- Existing full-physics solver API did not complete the frozen 142-row Round52 evaluation tractably, even after reducing the diagnostic mesh. No model-performance metrics are claimed.",
            "",
            "Allowed manuscript claim:",
            "",
            "- Frozen split was consumed and audited, but bounded WGS predictive validation remains unexecuted.",
            "",
            "Forbidden claims:",
            "",
            f"- {FORBIDDEN_CLAIM}",
        ],
    )
    report_lines = [
        "ROUND53_MODEL_VALIDATION_DONE",
        "Decision: MODEL_VALIDATION_NO_GO",
        f"Frozen split consumed: {consumed_ok}; R52-C-75 row IDs and roles preserved",
        "Methods evaluated: packaged as unavailable after existing full-physics solver API did not complete tractably for fixed_physics, direct_residual comparator, global bounded WGS scalar, low_capacity alpha_wgs(x_t), and null/control variants",
        "Train/validation/test rows: 81/44/17",
        "Primary held-out result: unavailable; no phase-4 model-performance metric claimed",
        "Worsened targets: unavailable because model predictions were not completed",
        "Bounds/fallbacks/convergence: solver runtime blocker recorded in huayi_round53_convergence_and_bounds.csv; bound-hit and convergence rates unavailable",
        "Null/control result: unavailable because fixed/adaptive predictions were not completed",
        "Created files: huayi_round53_frozen_split_manifest_intake.md, huayi_round53_baseline_metric_matrix.csv, huayi_round53_wgs_adaptive_metric_matrix.csv, huayi_round53_test_shadow_summary.csv, huayi_round53_null_control_summary.csv, huayi_round53_convergence_and_bounds.csv, huayi_round53_claim_boundary.md, huayi_round53_model_dispatch_result.md",
        "Tests/checks run: frozen split row-ID intake check; split role counts; required CSV schema emitted; attempted full solver run interrupted after existing API exceeded tractable runtime",
        "Known limitations: no model validation result; existing ModelCase solver path is too slow for the frozen 142-row split without a safer batched/cache/JAX execution path; Round52 subwindows require conversion from frozen means plus phase-versioned coal assumptions before model execution.",
        "Allowed manuscript claim: Round52 frozen split is ready, but Round53 predictive validation is blocked pending a tractable solver/evaluation path.",
        f"Forbidden claims: {FORBIDDEN_CLAIM}; no bounded WGS predictive improvement claim from Round53.",
    ]
    _write_markdown(output_dir / "huayi_round53_model_dispatch_result.md", report_lines)
    return {
        "decision": "MODEL_VALIDATION_NO_GO",
        "consumed_ok": consumed_ok,
        "report": "\n".join(report_lines),
    }


def run(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_yaml = _read_yaml(ROUND52_DIR / "huayi_round52_split_freeze.yaml")
    inventory = pd.read_csv(ROUND52_DIR / "huayi_round52_window_inventory.csv")
    primary = inventory[inventory["dataset_id"] == PRIMARY_DATASET_ID].copy()
    expected_ids = {
        role: set(split_yaml["splits"]["row_ids"][role])
        for role in ("train", "validation", "test")
    }
    consumed_ok = all(set(primary.loc[primary["split_role"] == role, "window_id"]) == ids for role, ids in expected_ids.items())
    return _write_unavailable_package(output_dir, primary, consumed_ok)

    _install_model_package_aliases()
    from model.bounded_hybrid_correction_smoke import solve_with_wgs_correction

    rows = _model_case_rows(primary)
    train_rows = [row for row in rows if row["split"] == "train"]

    fixed_solved = _solve_rows(rows, {str(row["case_id"]): 1.0 for row in rows}, solve_with_wgs_correction)
    fixed_predictions = _target_prediction_rows(fixed_solved, "fixed_physics")
    global_alpha, global_grid = _select_global_alpha(train_rows, solve_with_wgs_correction)
    global_solved = _solve_rows(rows, {str(row["case_id"]): global_alpha for row in rows}, solve_with_wgs_correction)
    global_predictions = _target_prediction_rows(global_solved, "global_bounded_wgs_scalar")

    direct_corrections = _fit_direct_residual(fixed_predictions[fixed_predictions["split_role"] == "train"])
    direct_predictions = _apply_direct_residual(fixed_predictions, direct_corrections)

    selected_alpha_xt, alpha_xt_candidates, alpha_xt_map = _select_alpha_xt(rows, train_rows, solve_with_wgs_correction)
    adaptive_solved = _solve_rows(rows, alpha_xt_map, solve_with_wgs_correction)
    adaptive_predictions = _target_prediction_rows(adaptive_solved, "low_capacity_alpha_wgs_xt")

    baseline_metrics = pd.concat(
        [
            _metric_matrix(fixed_predictions, "fixed_physics"),
            _metric_matrix(direct_predictions, "direct_residual_train_mean_comparator"),
            _metric_matrix(global_predictions, "global_bounded_wgs_scalar"),
        ],
        ignore_index=True,
    )
    adaptive_metrics = _metric_matrix(adaptive_predictions, "low_capacity_alpha_wgs_xt")
    null_summary = _null_control_summary(
        fixed_predictions,
        adaptive_predictions,
        fixed_predictions[fixed_predictions["split_role"] == "train"],
    )
    convergence = _convergence_table(
        {
            "fixed_physics": fixed_solved,
            "global_bounded_wgs_scalar": global_solved,
            "low_capacity_alpha_wgs_xt": adaptive_solved,
        }
    )
    test_worse = _worsened_targets(fixed_predictions, adaptive_predictions, "test")
    validation_worse = _worsened_targets(fixed_predictions, adaptive_predictions, "validation")
    score_fixed = _aggregate_score(fixed_predictions, primary)
    score_adaptive = _aggregate_score(adaptive_predictions, primary)
    decision = "MODEL_VALIDATION_GO"
    blockers: list[str] = []
    if not consumed_ok:
        decision = "MODEL_VALIDATION_NO_GO"
        blockers.append("frozen split row IDs did not match Round52 manifest")
    if len(test_worse) > 1 or len(validation_worse) > 1:
        decision = "MODEL_VALIDATION_CONDITIONAL"
        blockers.append("more than one held-out target worsened by >2% RMSE versus fixed physics")
    if score_adaptive.get("test", math.inf) > score_fixed.get("test", math.inf) * 1.02:
        decision = "MODEL_VALIDATION_CONDITIONAL"
        blockers.append("test aggregate score did not stabilize/improve versus fixed physics")

    baseline_metrics.to_csv(output_dir / "huayi_round53_baseline_metric_matrix.csv", index=False)
    adaptive_metrics.to_csv(output_dir / "huayi_round53_wgs_adaptive_metric_matrix.csv", index=False)
    null_summary.to_csv(output_dir / "huayi_round53_null_control_summary.csv", index=False)
    convergence.to_csv(output_dir / "huayi_round53_convergence_and_bounds.csv", index=False)

    test_summary_rows = []
    for method, predictions in [
        ("fixed_physics", fixed_predictions),
        ("direct_residual_train_mean_comparator", direct_predictions),
        ("global_bounded_wgs_scalar", global_predictions),
        ("low_capacity_alpha_wgs_xt", adaptive_predictions),
    ]:
        scores = _aggregate_score(predictions, primary)
        for split_role, score in scores.items():
            test_summary_rows.append(
                {
                    "dataset_id": PRIMARY_DATASET_ID,
                    "window_id": "AGGREGATE",
                    "split_role": split_role,
                    "phase": "ALL",
                    "method": method,
                    "target": "FIXED_WEIGHT_AGGREGATE",
                    "metric_name": "aggregate_score",
                    "metric_value": score,
                    "evidence_role": EVIDENCE_ROLE,
                    "allowed_claim": ALLOWED_CLAIM,
                    "forbidden_claim": FORBIDDEN_CLAIM,
                }
            )
    test_summary = pd.DataFrame(test_summary_rows)
    test_summary.to_csv(output_dir / "huayi_round53_test_shadow_summary.csv", index=False)

    _write_markdown(
        output_dir / "huayi_round53_frozen_split_manifest_intake.md",
        [
            "# Huayi Round53 Frozen Split Intake",
            "",
            f"- Source split: `{ROUND52_DIR / 'huayi_round52_split_freeze.yaml'}`",
            f"- Selected primary dataset: `{PRIMARY_DATASET_ID}`",
            f"- Frozen split consumed: {consumed_ok}",
            "- Train/validation/test rows: 81/44/17",
            "- Split rule preserved: phase 1+2 train, phase 3 validation, phase 4 test/shadow.",
            "- Window rules, target definitions and metric aggregation were not changed after model evaluation.",
        ],
    )
    _write_markdown(
        output_dir / "huayi_round53_claim_boundary.md",
        [
            "# Huayi Round53 Claim Boundary",
            "",
            f"Decision: **{decision}**",
            "",
            "Allowed manuscript claim:",
            "",
            f"- {ALLOWED_CLAIM}",
            "",
            "Forbidden claims:",
            "",
            f"- {FORBIDDEN_CLAIM}",
            "",
            "Blockers/conditions:",
            "",
            *[f"- {item}" for item in (blockers or ["none"])],
        ],
    )
    report_lines = [
        "ROUND53_MODEL_VALIDATION_DONE",
        f"Decision: {decision}",
        f"Frozen split consumed: {consumed_ok}; R52-C-75 row IDs and roles preserved",
        "Methods evaluated: fixed_physics; direct_residual_train_mean_comparator; global_bounded_wgs_scalar; low_capacity_alpha_wgs_xt; null_shuffled_train_residuals",
        "Train/validation/test rows: 81/44/17",
        f"Primary held-out result: fixed test aggregate={score_fixed.get('test', math.nan):.6g}; adaptive test aggregate={score_adaptive.get('test', math.nan):.6g}; selected alpha_xt={selected_alpha_xt['candidate']}; global_alpha={global_alpha:.6g}",
        f"Worsened targets: validation={validation_worse}; test={test_worse}",
        "Bounds/fallbacks/convergence: see huayi_round53_convergence_and_bounds.csv",
        "Null/control result: shuffled-label train residual control emitted; held-out adaptive-minus-fixed deltas emitted in huayi_round53_null_control_summary.csv",
        "Created files: "
        + ", ".join(
            [
                "huayi_round53_frozen_split_manifest_intake.md",
                "huayi_round53_baseline_metric_matrix.csv",
                "huayi_round53_wgs_adaptive_metric_matrix.csv",
                "huayi_round53_test_shadow_summary.csv",
                "huayi_round53_null_control_summary.csv",
                "huayi_round53_convergence_and_bounds.csv",
                "huayi_round53_claim_boundary.md",
                "huayi_round53_model_dispatch_result.md",
            ]
        ),
        "Tests/checks run: script row-ID intake check; frozen split role counts; metric CSV schema check",
        "Known limitations: Round52 subwindows are converted to model-case rows from frozen means and phase-versioned coal assumptions; CO2 carrier uses declared fixed engineering ratio where not present in Round52; dry syngas flow is carried as a target/prediction metadata target because the WGS scalar changes composition, not plant total-flow measurement; diagnostic solve uses N_cells=1 and a predeclared 5-point bounded alpha grid to keep the frozen-split run tractable.",
        f"Allowed manuscript claim: {ALLOWED_CLAIM}",
        f"Forbidden claims: {FORBIDDEN_CLAIM}",
    ]
    _write_markdown(output_dir / "huayi_round53_model_dispatch_result.md", report_lines)
    global_grid.to_csv(output_dir / "huayi_round53_global_alpha_grid.csv", index=False)
    alpha_xt_candidates.to_csv(output_dir / "huayi_round53_alpha_xt_candidate_selection.csv", index=False)
    return {
        "decision": decision,
        "consumed_ok": consumed_ok,
        "score_fixed": score_fixed,
        "score_adaptive": score_adaptive,
        "test_worse": test_worse,
        "validation_worse": validation_worse,
        "report": "\n".join(report_lines),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Huayi Round53 frozen-split validation outputs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "diagnostics" / "round53")
    args = parser.parse_args()
    result = run(args.output_dir)
    print(result["report"])


if __name__ == "__main__":
    main()
