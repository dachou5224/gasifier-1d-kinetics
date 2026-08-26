"""M2 WGS effect sizes and phase-aware paired block-bootstrap uncertainty."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .revision_evidence_common import (
    DATASET_ID,
    EVIDENCE_BOUNDARY,
    SEED,
    source_commit,
    write_sha256_manifest,
)

PRIMARY_TARGETS = ("CO_dry_vol_pct", "CO2_dry_vol_pct", "H2_dry_vol_pct", "H2_CO_ratio")
SUPPLEMENTARY_TARGETS = ("dry_syngas_flow",)
N_BOOTSTRAP = 2000
ACF_CUTOFF = 1.0 / math.e
MAX_BLOCK = 8
NEAR_NEIGHBOR_OFFSETS = (-1, 0, 1)


def _acf(values: np.ndarray, lag: int) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size <= lag:
        return math.nan
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(x[:-lag], x[lag:]) / denom)


def select_block_length(train_residuals: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(list(train_residuals), dtype=float)
    acf_rows = []
    chosen = MAX_BLOCK
    for lag in range(1, MAX_BLOCK + 1):
        value = _acf(values, lag)
        acf_rows.append({"lag": lag, "acf": value, "below_cutoff": bool(math.isfinite(value) and abs(value) < ACF_CUTOFF)})
        if chosen == MAX_BLOCK and acf_rows[-1]["below_cutoff"]:
            chosen = lag
    return {
        "selected_block_length": int(chosen),
        "acf_cutoff": ACF_CUTOFF,
        "selection_split": "train",
        "acf": acf_rows,
        "seed": SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "rule": "smallest lag in 1..8 with |ACF| < 1/e on train residuals; neighbors L-1 and L+1 reported",
    }


def moving_block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=int)
    length = max(int(block_length), 1)
    starts = rng.integers(0, n, size=int(math.ceil(n / length)) + 2)
    idx: list[int] = []
    for start in starts:
        for offset in range(length):
            idx.append(int((int(start) + offset) % n))
            if len(idx) >= n:
                return np.asarray(idx[:n], dtype=int)
    return np.asarray(idx[:n], dtype=int)


def paired_rmse(pred: np.ndarray, true: np.ndarray) -> float:
    residual = np.asarray(pred, dtype=float) - np.asarray(true, dtype=float)
    residual = residual[np.isfinite(residual)]
    if residual.size == 0:
        return math.nan
    return float(np.sqrt(np.mean(residual * residual)))


def metric_block(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    residual = np.asarray(pred, dtype=float) - np.asarray(true, dtype=float)
    residual = residual[np.isfinite(residual)]
    if residual.size == 0:
        return {"rmse": math.nan, "mae": math.nan, "bias": math.nan, "n": 0}
    return {
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "n": int(residual.size),
    }


def phase_aware_block_bootstrap(
    frame: pd.DataFrame,
    *,
    method: str,
    target: str,
    split_role: str,
    block_length: int,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> dict[str, Any]:
    work = frame[
        (frame["method"].isin([method, "fixed_physics"]))
        & (frame["target"] == target)
        & (frame["split_role"] == split_role)
        & (frame["execution_status"] == "completed")
    ].copy()
    if work.empty:
        return {"delta_rmse": math.nan, "ci_low": math.nan, "ci_high": math.nan, "n_windows": 0}
    wide = (
        work.pivot_table(index=["window_id", "phase"], columns="method", values=["predicted_value", "target_value"], aggfunc="first")
        .sort_index()
    )
    wide.columns = ["_".join(col).strip() for col in wide.columns.to_flat_index()]
    wide = wide.reset_index()
    pred_m_col = f"predicted_value_{method}"
    pred_f_col = "predicted_value_fixed_physics"
    true_col = f"target_value_{method}" if f"target_value_{method}" in wide.columns else "target_value_fixed_physics"
    if pred_m_col not in wide.columns or pred_f_col not in wide.columns or true_col not in wide.columns:
        return {"delta_rmse": math.nan, "ci_low": math.nan, "ci_high": math.nan, "n_windows": 0}
    observed = paired_rmse(wide[pred_m_col].to_numpy(), wide[true_col].to_numpy()) - paired_rmse(
        wide[pred_f_col].to_numpy(), wide[true_col].to_numpy()
    )
    groups = []
    for _, group in wide.groupby("phase", sort=True):
        groups.append(
            (
                group[pred_m_col].to_numpy(dtype=float),
                group[pred_f_col].to_numpy(dtype=float),
                group[true_col].to_numpy(dtype=float),
            )
        )
    rng = np.random.default_rng(seed)
    samples = np.empty(int(n_bootstrap), dtype=float)
    length = max(int(block_length), 1)
    for i in range(int(n_bootstrap)):
        pm_parts = []
        pf_parts = []
        yt_parts = []
        for pred_m, pred_f, true in groups:
            idx = moving_block_indices(len(pred_m), length, rng)
            pm_parts.append(pred_m[idx])
            pf_parts.append(pred_f[idx])
            yt_parts.append(true[idx])
        pm = np.concatenate(pm_parts)
        pf = np.concatenate(pf_parts)
        yt = np.concatenate(yt_parts)
        samples[i] = paired_rmse(pm, yt) - paired_rmse(pf, yt)
    return {
        "delta_rmse": float(observed),
        "ci_low": float(np.nanpercentile(samples, 2.5)),
        "ci_high": float(np.nanpercentile(samples, 97.5)),
        "n_windows": int(len(wide)),
        "n_bootstrap": int(n_bootstrap),
        "block_length": int(block_length),
        "seed": int(seed),
    }


def recommend_gate(effect_matrix: pd.DataFrame) -> str:
    test = effect_matrix[(effect_matrix["split_role"] == "test") & (effect_matrix["target"].isin(["CO_dry_vol_pct", "CO2_dry_vol_pct"]))]
    if test.empty:
        return "audit_supported_candidate"
    worsened = test[(test["delta_rmse"] > 0) & (test["target"] == "CO2_dry_vol_pct")]
    improved = test[(test["delta_rmse"] < 0) & (test["target"] == "CO_dry_vol_pct")]
    if len(improved) and len(worsened):
        return "audit_supported_candidate"
    if len(improved) and not len(worsened):
        return "provisionally_open"
    return "audit_supported_candidate"


def build_m2_tables(stability: pd.DataFrame, *, source_commit_hash: str) -> dict[str, pd.DataFrame | dict[str, Any]]:
    methods = [m for m in stability["method"].unique() if m != "fixed_physics"]
    effect_rows: list[dict[str, Any]] = []
    relative_rows: list[dict[str, Any]] = []
    for method in methods:
        for split_role in ("train", "validation", "test"):
            for target in PRIMARY_TARGETS + SUPPLEMENTARY_TARGETS:
                sub = stability[(stability["split_role"] == split_role) & (stability["target"] == target)]
                fixed = sub[sub["method"] == "fixed_physics"]
                other = sub[sub["method"] == method]
                merged = fixed.merge(other, on=["window_id"], suffixes=("_fixed", "_method"))
                merged = merged[merged["execution_status_fixed"] == "completed"]
                merged = merged[merged["execution_status_method"] == "completed"]
                fixed_m = metric_block(merged["predicted_value_fixed"], merged["target_value_fixed"])
                other_m = metric_block(merged["predicted_value_method"], merged["target_value_method"])
                delta = other_m["rmse"] - fixed_m["rmse"]
                pct = 100.0 * delta / abs(fixed_m["rmse"]) if math.isfinite(delta) and abs(fixed_m["rmse"]) > 1e-12 else math.nan
                row = {
                    "dataset_id": DATASET_ID,
                    "split_role": split_role,
                    "method": method,
                    "target": target,
                    "rmse": other_m["rmse"],
                    "mae": other_m["mae"],
                    "bias": other_m["bias"],
                    "rmse_fixed": fixed_m["rmse"],
                    "delta_rmse": delta,
                    "pct_rmse_change": pct,
                    "n_attempted": int(other["window_id"].nunique()),
                    "n_completed": int(merged["window_id"].nunique()),
                    "n_timeout": int((other["execution_status"] == "timeout").sum()),
                    "n_error": int(other["execution_status"].astype(str).str.startswith("error").sum()),
                    "execution_status": "completed" if int(merged["window_id"].nunique()) == int(other["window_id"].nunique()) else "partial",
                    "source_commit": source_commit_hash,
                }
                effect_rows.append(row)
                relative_rows.append({k: row[k] for k in ("dataset_id", "split_role", "method", "target", "delta_rmse", "pct_rmse_change", "n_completed", "source_commit")})
    effect = pd.DataFrame(effect_rows)
    train_fixed = stability[(stability["method"] == "fixed_physics") & (stability["split_role"] == "train") & (stability["target"] == "CO_dry_vol_pct")]
    protocol = select_block_length(train_fixed["residual"].astype(float).tolist() if "residual" in train_fixed.columns else [])
    bootstrap_rows: list[dict[str, Any]] = []
    for method in methods:
        for split_role in ("train", "validation", "test"):
            for target in PRIMARY_TARGETS:
                for offset in NEAR_NEIGHBOR_OFFSETS:
                    length = max(protocol["selected_block_length"] + offset, 1)
                    stats = phase_aware_block_bootstrap(
                        stability,
                        method=method,
                        target=target,
                        split_role=split_role,
                        block_length=length,
                    )
                    bootstrap_rows.append(
                        {
                            "dataset_id": DATASET_ID,
                            "split_role": split_role,
                            "method": method,
                            "target": target,
                            "block_length": length,
                            "is_primary_block_length": offset == 0,
                            **stats,
                            "source_commit": source_commit_hash,
                        }
                    )
    return {
        "effect": effect,
        "relative": pd.DataFrame(relative_rows),
        "bootstrap": pd.DataFrame(bootstrap_rows),
        "protocol": protocol,
        "gate": recommend_gate(effect),
    }


def write_m2_summary(tables: Mapping[str, Any], output_dir: Path, source_commit_hash: str) -> Path:
    effect = tables["effect"]
    lines = ["# Revision M2 WGS Effect Sizes", "", f"- Source commit: `{source_commit_hash}`", f"- Seed: {SEED}", f"- Bootstrap replicates: {N_BOOTSTRAP}", f"- Gate recommendation: `{tables['gate']}`", ""]
    test = effect[(effect["split_role"] == "test") & (effect["target"].isin(["CO_dry_vol_pct", "CO2_dry_vol_pct"]))]
    for _, row in test.iterrows():
        lines.append(
            f"- test `{row['method']}` `{row['target']}`: RMSE {row['rmse']:.6g} vs fixed {row['rmse_fixed']:.6g}; "
            f"delta_rmse={row['delta_rmse']:.6g} ({row['pct_rmse_change']:.3g}%)."
        )
    lines.extend(["", f"Evidence boundary: {EVIDENCE_BOUNDARY}"])
    path = output_dir / "revision_m2_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_m2(*, stability_path: Path, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_commit_hash = source_commit()
    stability = pd.read_csv(stability_path)
    tables = build_m2_tables(stability, source_commit_hash=source_commit_hash)
    effect_path = output_dir / "revision_m2_wgs_effect_matrix.csv"
    rel_path = output_dir / "revision_m2_wgs_relative_change.csv"
    boot_path = output_dir / "revision_m2_wgs_block_bootstrap.csv"
    proto_path = output_dir / "revision_m2_bootstrap_protocol.json"
    tables["effect"].to_csv(effect_path, index=False)
    tables["relative"].to_csv(rel_path, index=False)
    tables["bootstrap"].to_csv(boot_path, index=False)
    proto_path.write_text(json.dumps(tables["protocol"], indent=2), encoding="utf-8")
    summary = write_m2_summary(tables, output_dir, source_commit_hash)
    manifest = write_sha256_manifest([effect_path, rel_path, boot_path, proto_path, summary], output_dir / "revision_m2_manifest.sha256")
    return {"summary": summary, "manifest": manifest}
