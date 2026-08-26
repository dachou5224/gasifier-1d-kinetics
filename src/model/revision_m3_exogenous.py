"""M3 exogenous-only forward-prediction WGS comparator."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .revision_evidence_common import (
    DATASET_ID,
    EVIDENCE_BOUNDARY,
    SEED,
    load_frozen_split,
    metric_rows,
    objective_from_predictions,
    prediction_records,
    source_commit,
    write_sha256_manifest,
)
from .revision_m1_cell0 import _alpha_xt_value, _safe_solve, _train_feature_stats, declared_variants

EXOGENOUS_FEATURES = (
    {
        "feature": "o_c_ratio",
        "availability_class": "exogenous_process_input",
        "timestamp_rationale": "window-mean O2/coal ratio from feed measurements available at the window start",
        "allowed": True,
    },
    {
        "feature": "coal_flow_mean",
        "availability_class": "exogenous_process_input",
        "timestamp_rationale": "weighfeeder coal mass flow measured before the prediction window closes",
        "allowed": True,
    },
    {
        "feature": "carrier_to_coal_ratio",
        "availability_class": "exogenous_engineering_input",
        "timestamp_rationale": "CO2 carrier is an engineering ratio of measured O2 feed, not an outlet measurement",
        "allowed": True,
    },
    {
        "feature": "pressure_mpa",
        "availability_class": "exogenous_process_input",
        "timestamp_rationale": "gasifier pressure is a contemporaneous process input, not a syngas product measurement",
        "allowed": True,
    },
)
FORBIDDEN_FEATURES = ("syngas_flow_mean", "co_vol_pct", "h2_vol_pct", "co2_vol_pct", "h2_co_ratio", "pred_dry_flow_nm3_h")
NULL_SHIFT = 7
ALPHA_GRID = (0.05, 0.20, 0.50, 1.00, 2.00)


def feature_contract_rows(source_commit_hash: str) -> pd.DataFrame:
    rows = []
    for item in EXOGENOUS_FEATURES:
        rows.append(
            {
                "dataset_id": DATASET_ID,
                "feature": item["feature"],
                "availability_class": item["availability_class"],
                "timestamp_rationale": item["timestamp_rationale"],
                "allowed_in_forward_comparator": True,
                "source_commit": source_commit_hash,
            }
        )
    for feature in FORBIDDEN_FEATURES:
        rows.append(
            {
                "dataset_id": DATASET_ID,
                "feature": feature,
                "availability_class": "contemporaneous_or_future_output",
                "timestamp_rationale": "outlet composition or syngas flow is not available before the prediction time",
                "allowed_in_forward_comparator": False,
                "source_commit": source_commit_hash,
            }
        )
    return pd.DataFrame(rows)


def cyclic_shift(values: Sequence[float], shift: int = NULL_SHIFT) -> list[float]:
    arr = list(values)
    if not arr:
        return []
    k = int(shift) % len(arr)
    return arr[-k:] + arr[:-k]


def run_m3(
    *,
    output_dir: Path,
    configuration: str,
    selected_global_alpha: float,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_commit_hash = source_commit()
    frozen_consumed, rows, _primary = load_frozen_split()
    if not frozen_consumed:
        raise RuntimeError("frozen split not consumed")
    variant = declared_variants()[configuration]
    train_rows = [row for row in rows if row["split_role"] == "train"]
    stats = _train_feature_stats(rows, [item["feature"] for item in EXOGENOUS_FEATURES])
    contract = feature_contract_rows(source_commit_hash)
    allowed = [item["feature"] for item in EXOGENOUS_FEATURES]
    selection_rows = []
    pred_by_candidate: dict[str, pd.DataFrame] = {}
    for feature in allowed:
        for sign in (-1.0, 1.0):
            candidate = f"{feature}_{'negative' if sign < 0 else 'positive'}_slope"
            records = []
            print(f"m3 train {candidate} start n={len(train_rows)}", flush=True)
            for index, row in enumerate(train_rows, start=1):
                alpha = _alpha_xt_value(row, feature=feature, sign=sign, base_alpha=selected_global_alpha, stats=stats)
                outcome = _safe_solve(row, variant, alpha_wgs=alpha)
                if index == 1 or index % 20 == 0 or index == len(train_rows):
                    print(
                        f"m3 train {candidate} {index}/{len(train_rows)} {outcome.get('execution_status')}",
                        flush=True,
                    )
                records.extend(prediction_records(row, outcome, method=candidate, configuration=configuration, source_commit_hash=source_commit_hash))
            preds = pd.DataFrame(records)
            pred_by_candidate[candidate] = preds
            selection_rows.append(
                {
                    "dataset_id": DATASET_ID,
                    "candidate": candidate,
                    "feature": feature,
                    "sign": sign,
                    "selection_split": "train",
                    "selection_objective": objective_from_predictions(preds, "train"),
                    "execution_status": "completed" if (preds["execution_status"] == "completed").all() else "partial",
                    "selected": False,
                    "source_commit": source_commit_hash,
                }
            )
    selection = pd.DataFrame(selection_rows)
    ok = selection[selection["execution_status"] == "completed"]
    best = ok.sort_values(["selection_objective", "candidate"]).iloc[0]
    selection.loc[selection["candidate"] == best["candidate"], "selected"] = True

    full_records = list(pred_by_candidate[str(best["candidate"])].to_dict("records"))
    heldout = [row for row in rows if row["split_role"] != "train"]
    print(f"m3 heldout start selected={best['candidate']} n={len(heldout)}", flush=True)
    for index, row in enumerate(heldout, start=1):
        alpha = _alpha_xt_value(row, feature=str(best["feature"]), sign=float(best["sign"]), base_alpha=selected_global_alpha, stats=stats)
        outcome = _safe_solve(row, variant, alpha_wgs=alpha)
        if index == 1 or index % 20 == 0 or index == len(heldout):
            print(f"m3 heldout {index}/{len(heldout)} {outcome.get('execution_status')}", flush=True)
        full_records.extend(prediction_records(row, outcome, method="exogenous_alpha_wgs_xt", configuration=configuration, source_commit_hash=source_commit_hash))
    predictions = pd.DataFrame(full_records)
    predictions.loc[predictions["split_role"] != "train", "method"] = "exogenous_alpha_wgs_xt"
    predictions.loc[predictions["split_role"] == "train", "method"] = "exogenous_alpha_wgs_xt"

    # Registered null: cyclic shift of the selected train feature, train-only.
    shifted_values = cyclic_shift([float(row[str(best["feature"])]) for row in train_rows], NULL_SHIFT)
    null_rows = []
    print(f"m3 null cyclic_shift={NULL_SHIFT} n={len(train_rows)}", flush=True)
    for index, (row, shifted) in enumerate(zip(train_rows, shifted_values), start=1):
        fake = dict(row)
        fake[str(best["feature"])] = shifted
        alpha = _alpha_xt_value(fake, feature=str(best["feature"]), sign=float(best["sign"]), base_alpha=selected_global_alpha, stats=stats)
        outcome = _safe_solve(row, variant, alpha_wgs=alpha)
        if index == 1 or index % 20 == 0 or index == len(train_rows):
            print(f"m3 null {index}/{len(train_rows)} {outcome.get('execution_status')}", flush=True)
        null_rows.extend(prediction_records(row, outcome, method="exogenous_null_cyclic_shift", configuration=configuration, source_commit_hash=source_commit_hash))
    null = pd.DataFrame(null_rows)
    control_explains = objective_from_predictions(null, "train") <= objective_from_predictions(predictions, "train") + 1e-12
    null_summary = pd.DataFrame(
        [
            {
                "dataset_id": DATASET_ID,
                "control": "cyclic_shift_train_feature",
                "shift": NULL_SHIFT,
                "seed": SEED,
                "train_objective_model": objective_from_predictions(predictions, "train"),
                "train_objective_control": objective_from_predictions(null, "train"),
                "control_explains": bool(control_explains),
                "execution_status": "completed" if (null["execution_status"] == "completed").all() else "partial",
                "source_commit": source_commit_hash,
            }
        ]
    )
    metrics = metric_rows(predictions, method="exogenous_alpha_wgs_xt", source_commit_hash=source_commit_hash)
    contract.to_csv(output_dir / "revision_m3_exogenous_feature_contract.csv", index=False)
    predictions.to_csv(output_dir / "revision_m3_exogenous_wgs_predictions.csv", index=False)
    metrics.to_csv(output_dir / "revision_m3_exogenous_wgs_metrics.csv", index=False)
    null_summary.to_csv(output_dir / "revision_m3_exogenous_null_control.csv", index=False)
    summary = output_dir / "revision_m3_summary.md"
    summary.write_text(
        f"""# Revision M3 Exogenous-Only WGS Comparator

- Frozen split consumed: True
- Configuration: `{configuration}`
- Selected train-only candidate: `{best['candidate']}`
- Forbidden contemporaneous outputs excluded: {', '.join(FORBIDDEN_FEATURES)}
- Null control explains train fit: {control_explains}
- Forward-prediction claim: not supported from a single split even if held-out metrics improve.

Evidence boundary: {EVIDENCE_BOUNDARY}
""",
        encoding="utf-8",
    )
    manifest = write_sha256_manifest(
        [
            output_dir / "revision_m3_exogenous_feature_contract.csv",
            output_dir / "revision_m3_exogenous_wgs_predictions.csv",
            output_dir / "revision_m3_exogenous_wgs_metrics.csv",
            output_dir / "revision_m3_exogenous_null_control.csv",
            summary,
        ],
        output_dir / "revision_m3_manifest.sha256",
    )
    return {"summary": summary, "manifest": manifest}
