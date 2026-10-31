"""M10 direct outlet residual comparator on the frozen R52-C-75 142-window split."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .direct_residual_baseline import (
    CONSTANT_VARIANT,
    LINEAR_VARIANT,
    direct_residual_prediction_rows,
    fit_direct_residual_models,
    operating_feature_rows,
)
from .revision_evidence_common import (
    DATASET_ID,
    load_frozen_split,
    source_commit,
    write_sha256_manifest,
)

PRIMARY_TARGETS = ("CO_dry_vol_pct", "CO2_dry_vol_pct", "H2_dry_vol_pct", "H2_CO_ratio")
SUPPLEMENTARY_TARGETS = ("dry_syngas_flow",)
ALL_TARGETS = PRIMARY_TARGETS + SUPPLEMENTARY_TARGETS
EFFECT_COLUMNS = (
    "dataset_id",
    "split_role",
    "method",
    "target",
    "rmse",
    "mae",
    "bias",
    "rmse_fixed",
    "delta_rmse",
    "pct_rmse_change",
    "n_attempted",
    "n_completed",
    "n_timeout",
    "n_error",
    "execution_status",
    "source_commit",
)
M10_BOUNDARY = (
    "locked post-outlet residual comparator on R52-C-75; not a physical update, "
    "not a reactor source term, not eligible for monitored evaluation, not "
    "learned WGS kinetics"
)


def _require_frozen_counts(primary: pd.DataFrame) -> dict[str, int]:
    counts = {role: int((primary["split_role"] == role).sum()) for role in ("train", "validation", "test")}
    if counts != {"train": 81, "validation": 44, "test": 17}:
        raise RuntimeError(f"frozen split counts mismatch: {counts}")
    return counts


def load_table4_fixed_physics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    work = frame[frame["method"] == "fixed_physics"].copy()
    if work.empty:
        raise RuntimeError(f"no fixed_physics rows in {path}")
    needed = {
        "dataset_id",
        "window_id",
        "phase",
        "split_role",
        "N_cells",
        "target",
        "target_value",
        "predicted_value",
        "residual",
        "execution_status",
        "residual_quality_status",
        "cell0_cost",
        "poor_convergence_count",
    }
    missing = needed.difference(work.columns)
    if missing:
        raise RuntimeError(f"fixed-physics table missing columns: {sorted(missing)}")
    if "fallback_count" not in work.columns:
        work["fallback_count"] = np.nan
    return work


def load_m2_rmse_fixed(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    table = (
        frame[["split_role", "target", "rmse_fixed"]]
        .drop_duplicates(["split_role", "target"])
        .reset_index(drop=True)
    )
    return table


def train_mean_residual(fixed: pd.DataFrame, target: str) -> float:
    train = fixed[
        (fixed["split_role"] == "train")
        & (fixed["target"] == target)
        & (fixed["execution_status"] == "completed")
    ]
    if train.empty:
        raise RuntimeError(f"no completed train rows for target={target}")
    return float(train["residual"].astype(float).mean())


def apply_constant_patch(fixed: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    corrections: dict[str, float] = {}
    pieces: list[pd.DataFrame] = []
    for target in ALL_TARGETS:
        subset = fixed[fixed["target"] == target].copy()
        correction = train_mean_residual(subset, target)
        corrections[target] = correction
        subset["method"] = CONSTANT_VARIANT
        subset["y_plant"] = subset["target_value"].astype(float)
        subset["y_fixed"] = subset["predicted_value"].astype(float)
        subset["correction_value"] = correction
        subset["y_residual"] = subset["y_fixed"] - correction
        subset["residual_patched"] = subset["y_residual"] - subset["y_plant"]
        subset["fit_split"] = "train"
        subset["validation_excluded_from_fit"] = True
        subset["test_excluded_from_fit"] = True
        pieces.append(subset)
    return pd.concat(pieces, ignore_index=True), corrections


def _builder_fixed_rows(fixed: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    work = fixed[fixed["target"].isin(ALL_TARGETS)].copy()
    for _, item in work.iterrows():
        rows.append(
            {
                "case_id": item["window_id"],
                "phase": item["phase"],
                "dataset_id": item["dataset_id"],
                "split": item["split_role"],
                "split_role": item["split_role"],
                "N_cells": int(item["N_cells"]),
                "target": item["target"],
                "y_true": float(item["target_value"]),
                "y_pred": float(item["predicted_value"]),
                "residual": float(item["residual"]),
                "execution_status": item["execution_status"],
                "residual_quality_status": item["residual_quality_status"],
                "cell0_cost": item["cell0_cost"],
                "fallback_count": item.get("fallback_count", np.nan),
                "poor_convergence_count": item["poor_convergence_count"],
            }
        )
    return rows


def apply_linear_patch(
    fixed: pd.DataFrame,
    model_case_rows: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    builder_rows = _builder_fixed_rows(fixed)
    features = operating_feature_rows(model_case_rows)
    models = [
        model
        for model in fit_direct_residual_models(builder_rows, features)
        if model["baseline_variant"] == LINEAR_VARIANT
    ]
    predictions = direct_residual_prediction_rows(builder_rows, features, models)
    frame = pd.DataFrame(predictions)
    if "window_id" not in frame.columns:
        frame = frame.rename(columns={"case_id": "window_id"})
    if "y_plant" not in frame.columns:
        frame = frame.rename(
            columns={
                "target_value": "y_plant",
                "fixed_prediction": "y_fixed",
                "corrected_prediction": "y_residual",
                "corrected_residual": "residual_patched",
            }
        )
    frame["split_role"] = frame["split"] if "split" in frame.columns else frame["split_role"]
    frame["method"] = LINEAR_VARIANT
    quality_cols = [
        "window_id",
        "target",
        "execution_status",
        "residual_quality_status",
        "cell0_cost",
        "fallback_count",
        "poor_convergence_count",
    ]
    quality = fixed[quality_cols].drop_duplicates(["window_id", "target"])
    merged = frame.merge(quality, on=["window_id", "target"], how="left")
    merged["fit_split"] = "train"
    merged["validation_excluded_from_fit"] = True
    merged["test_excluded_from_fit"] = True
    return merged


def effect_matrix(
    patched: pd.DataFrame,
    rmse_fixed_table: pd.DataFrame,
    *,
    method: str,
    source_commit_hash: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    work = patched[patched["method"] == method].copy()
    for (split_role, target), group in work.groupby(["split_role", "target"], dropna=False):
        attempted = int(group["window_id"].nunique())
        completed = group[group["execution_status"] == "completed"]
        n_completed = int(completed["window_id"].nunique())
        n_timeout = int(group[group["execution_status"] == "timeout"]["window_id"].nunique())
        n_error = int(group[group["execution_status"].astype(str).str.startswith("error")]["window_id"].nunique())
        residual = completed["residual_patched"].astype(float).to_numpy() if n_completed else np.array([])
        rmse = float(np.sqrt(np.mean(residual * residual))) if residual.size else float("nan")
        mae = float(np.mean(np.abs(residual))) if residual.size else float("nan")
        bias = float(np.mean(residual)) if residual.size else float("nan")
        match = rmse_fixed_table[
            (rmse_fixed_table["split_role"] == split_role) & (rmse_fixed_table["target"] == target)
        ]
        if match.empty:
            raise RuntimeError(f"missing Table-4 rmse_fixed for {split_role}/{target}")
        rmse_fixed = float(match.iloc[0]["rmse_fixed"])
        if n_completed:
            fixed_residual = completed["y_fixed"].astype(float) - completed["y_plant"].astype(float)
            recomputed = float(np.sqrt(np.mean(fixed_residual.to_numpy() ** 2)))
            if abs(recomputed - rmse_fixed) > 1.0e-4:
                raise RuntimeError(
                    f"fixed RMSE mismatch for {split_role}/{target}: rows={recomputed} table4={rmse_fixed}"
                )
        delta = rmse - rmse_fixed
        pct = 100.0 * delta / rmse_fixed if rmse_fixed != 0.0 else float("nan")
        rows.append(
            {
                "dataset_id": DATASET_ID,
                "split_role": split_role,
                "method": method,
                "target": target,
                "rmse": rmse,
                "mae": mae,
                "bias": bias,
                "rmse_fixed": rmse_fixed,
                "delta_rmse": delta,
                "pct_rmse_change": pct,
                "n_attempted": attempted,
                "n_completed": n_completed,
                "n_timeout": n_timeout,
                "n_error": n_error,
                "execution_status": "completed" if attempted == n_completed else "partial",
                "source_commit": source_commit_hash,
            }
        )
    return pd.DataFrame(rows).sort_values(["method", "split_role", "target"]).reset_index(drop=True)[list(EFFECT_COLUMNS)]


def row_prediction_table(constant: pd.DataFrame, linear: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "dataset_id",
        "window_id",
        "phase",
        "split_role",
        "method",
        "target",
        "N_cells",
        "y_plant",
        "y_fixed",
        "y_residual",
        "correction_value",
        "residual_patched",
        "execution_status",
        "residual_quality_status",
        "cell0_cost",
        "fallback_count",
        "poor_convergence_count",
        "fit_split",
        "validation_excluded_from_fit",
        "test_excluded_from_fit",
        "source_commit",
    ]
    frames = []
    for frame in (constant, linear):
        work = frame.copy()
        if "source_commit" not in work.columns:
            work["source_commit"] = ""
        for column in keep:
            if column not in work.columns:
                work[column] = np.nan
        frames.append(work[keep])
    return pd.concat(frames, ignore_index=True)


def interpretation_markdown(
    effect: pd.DataFrame,
    protocol: Mapping[str, Any],
    corrections: Mapping[str, float],
) -> str:
    def _row(method: str, split_role: str, target: str) -> pd.Series:
        hit = effect[(effect["method"] == method) & (effect["split_role"] == split_role) & (effect["target"] == target)]
        if hit.empty:
            raise RuntimeError(f"missing effect row {method}/{split_role}/{target}")
        return hit.iloc[0]

    test_constant = {target: _row(CONSTANT_VARIANT, "test", target) for target in PRIMARY_TARGETS}
    lines = [
        "# M10 Direct residual comparator on R52-C-75",
        "",
        "Locked post-outlet residual patch on the same 81/44/17 protocol as Table 4.",
        "This is a black-box comparator / alarm, not a physical update and not a reactor source term.",
        "",
        f"- Dataset: `{protocol['dataset_id']}`",
        f"- Split counts: {protocol['split_counts']}",
        f"- N_cells: {protocol['n_cells']} (copied from Table 4 fixed physics; not the 22-case N=3 path)",
        f"- Primary method: `{CONSTANT_VARIANT}`",
        f"- Exploratory method: `{LINEAR_VARIANT}` (not used to choose the primary form)",
        f"- Fit split: train only; validation/test excluded from the residual mean and from linear scaling",
        f"- Residual definition: `y_residual = y_fixed - mean_train(y_fixed - y_plant)`",
        f"- Source commit: `{protocol['source_commit']}`",
        "",
        "## Train-only constant corrections",
        "",
    ]
    for target, value in corrections.items():
        lines.append(f"- `{target}`: {value:.6g}")
    lines.extend(
        [
            "",
            "## Test n=17 vs Table 4 fixed physics",
            "",
            "| target | fixed RMSE | residual RMSE | ΔRMSE | % | n_completed |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for target, row in test_constant.items():
        lines.append(
            f"| {target} | {row['rmse_fixed']:.6g} | {row['rmse']:.6g} | {row['delta_rmse']:.6g} | {row['pct_rmse_change']:.4g} | {int(row['n_completed'])} |"
        )
    co = test_constant["CO_dry_vol_pct"]
    co2 = test_constant["CO2_dry_vol_pct"]
    h2 = test_constant["H2_dry_vol_pct"]
    worsened = []
    if float(co["delta_rmse"]) > 0:
        worsened.append("CO")
    if float(h2["delta_rmse"]) > 0:
        worsened.append("H2")
    if float(co2["delta_rmse"]) > 0:
        worsened.append("CO2")
    both_improve = float(co["delta_rmse"]) < 0 and float(co2["delta_rmse"]) < 0
    val_co2 = _row(CONSTANT_VARIANT, "validation", "CO2_dry_vol_pct")
    linear_test_co2 = _row(LINEAR_VARIANT, "test", "CO2_dry_vol_pct")
    if both_improve:
        paper_action = (
            "Residual RMSE falls for both test CO and CO2. Keep the method **locked** "
            "(comparator/alarm). Do not promote to audit-supported: there is still no reactor route. "
            "The apparent test-set collapse of RMSE is a large additive bias correction on the N=1 Huayi path. "
            f"Validation CO2 RMSE moves from {float(val_co2['rmse_fixed']):.4g} to {float(val_co2['rmse']):.4g} "
            "under the same constant patch, so the offset does not travel uniformly. "
            f"The exploratory linear patch was not used for selection; on test it changes CO2 RMSE by "
            f"{float(linear_test_co2['delta_rmse']):.4g}. First-cell residual_quality_status is copied from "
            "fixed-physics rows and is not hidden."
        )
    elif worsened:
        paper_action = (
            f"Residual patch worsens held-out {', '.join(worsened)}. Cite this as the reason a "
            "black-box outlet patch is an alarm, not a physical update."
        )
    else:
        paper_action = "Keep locked comparator/alarm; do not change gate state from the model workspace."
    lines.extend(
        [
            "",
            "## Paper action",
            "",
            paper_action,
            "",
            "## Boundary",
            "",
            M10_BOUNDARY,
            "",
            "Do not read this package as learned kinetics, WGS identification, N=5 IFT, oxidation-channel updating, or production control.",
            "",
        ]
    )
    return "\n".join(lines)


def run_m10(
    *,
    output_dir: Path,
    fixed_physics_path: Path,
    m2_effect_path: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_commit_hash = source_commit()
    frozen_consumed, model_case_rows, primary = load_frozen_split()
    if not frozen_consumed:
        raise RuntimeError("frozen R52-C-75 split not consumed")
    counts = _require_frozen_counts(primary)
    frozen_ids = set(primary["window_id"].astype(str))
    fixed = load_table4_fixed_physics(fixed_physics_path)
    fixed = fixed[fixed["window_id"].astype(str).isin(frozen_ids)].copy()
    n_cells = sorted({int(v) for v in fixed["N_cells"].dropna().unique()})
    if n_cells != [1]:
        raise RuntimeError(f"Table 4 fixed physics must be N_cells=1, got {n_cells}")
    role_counts = {
        role: int(fixed.loc[fixed["target"] == PRIMARY_TARGETS[0], "split_role"].eq(role).sum())
        for role in ("train", "validation", "test")
    }
    if role_counts != counts:
        raise RuntimeError(f"fixed-physics role counts {role_counts} != frozen {counts}")
    rmse_fixed_table = load_m2_rmse_fixed(m2_effect_path)
    constant, corrections = apply_constant_patch(fixed)
    constant["source_commit"] = source_commit_hash
    linear = apply_linear_patch(fixed, model_case_rows)
    linear["source_commit"] = source_commit_hash
    predictions = row_prediction_table(constant, linear)
    effect = pd.concat(
        [
            effect_matrix(constant, rmse_fixed_table, method=CONSTANT_VARIANT, source_commit_hash=source_commit_hash),
            effect_matrix(linear, rmse_fixed_table, method=LINEAR_VARIANT, source_commit_hash=source_commit_hash),
        ],
        ignore_index=True,
    )
    protocol = {
        "dataset_id": DATASET_ID,
        "split_id": "R52-C-75",
        "split_counts": counts,
        "frozen_split_consumed": True,
        "n_cells": 1,
        "adapter_contract": "revision M1/M2 Huayi model-case adapter; no re-tune of kinetics, moisture, carrier or steam",
        "primary_method": CONSTANT_VARIANT,
        "exploratory_method": LINEAR_VARIANT,
        "residual_definition": "y_residual = y_fixed - mean_train(y_fixed - y_plant); independent per declared target",
        "train_only": True,
        "validation_excluded_from_fit": True,
        "test_excluded_from_fit": True,
        "selection_rule": "primary form predeclared; validation/test RMSE were not used to choose residual form, targets or clipping",
        "no_clipping": True,
        "source_term_route": False,
        "seed": None,
        "source_commit": source_commit_hash,
        "fixed_physics_source": str(fixed_physics_path.name),
        "rmse_fixed_source": str(m2_effect_path.name),
        "quality_fields": "copied from fixed-physics rows; residual_quality_status is not hidden",
        "evidence_boundary": M10_BOUNDARY,
        "train_mean_corrections": corrections,
    }
    pred_path = output_dir / "m10_residual_row_predictions.csv"
    effect_path = output_dir / "m10_residual_effect_matrix.csv"
    protocol_path = output_dir / "m10_protocol.json"
    md_path = output_dir / "m10_interpretation.md"
    predictions.to_csv(pred_path, index=False)
    effect.to_csv(effect_path, index=False)
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(interpretation_markdown(effect, protocol, corrections), encoding="utf-8")
    manifest = write_sha256_manifest(
        [effect_path, pred_path, protocol_path, md_path],
        output_dir / "m10_sha256.txt",
    )
    return {
        "effect": effect_path,
        "predictions": pred_path,
        "protocol": protocol_path,
        "interpretation": md_path,
        "manifest": manifest,
    }
