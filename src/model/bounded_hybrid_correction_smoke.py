"""
Bounded correction smoke protocol for the fixed-physics gasifier model.

This module is diagnostic-only. It probes whether a small bounded scalar
reaction-family correction can move a selected fixed-physics objective downhill.
It does not train a hybrid model, alter solver defaults, or claim calibrated
accuracy.
"""
from __future__ import annotations

import math
import time
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .gasifier_system import GasifierSystem
from .model_case_adapter import build_gasifier_inputs_from_model_case
from .model_case_diagnostic_baseline import (
    POOR_CONVERGENCE_COST_THRESHOLD,
    cell_residual_quality,
)

BOUNDED_CORRECTION_EVIDENCE_BOUNDARY = (
    "bounded fixed-physics correction smoke protocol only; not a trained hybrid "
    "model, calibrated accuracy, production control recommendation, learned "
    "multiplier interpretability claim, causal proof, or Yulin/slurry generalization"
)

CORRECTION_NAME = "WGS_CatalyticFactor_bounded_reaction_family"
CORRECTION_TARGET = "weighted_outlet_CO_H2_CO2_squared_error"
CORRECTION_BOUND_LOW = 0.02
CORRECTION_BOUND_HIGH = 2.0
# 判定「目标确实下降」的绝对容差。小于该量的变化视为无改善，避免把平坦目标
# 或求值噪声记成下降；线搜索也用它，防止 theta 在零增益时沿方向漂移。
LOSS_DECREASE_ABS_TOL = 1.0e-12

BOUNDED_CORRECTION_SUMMARY_COLUMNS: tuple[str, ...] = (
    "correction_name",
    "correction_layer",
    "bound_low",
    "bound_high",
    "initial_theta",
    "final_theta",
    "initial_value",
    "final_value",
    "loss_initial",
    "loss_final",
    "loss_delta",
    "relative_loss_delta",
    "loss_decreased",
    "steps",
    "steps_completed",
    "hit_lower_bound",
    "hit_upper_bound",
    "target",
    "case_id",
    "phase",
    "dataset_id",
    "source_candidate_id",
    "independence_group",
    "N_cells",
    "fallback_count_final",
    "poor_convergence_count_final",
    "poor_convergence_cell_index_final",
    "cell0_cost_final",
    "status",
    "failure_reason",
    "evidence_boundary",
)

BOUNDED_CORRECTION_BATCH_COLUMNS: tuple[str, ...] = (
    "scope",
    "phase",
    "n_cases",
    "n_ok",
    "n_failed",
    "n_loss_decreased",
    "median_loss_delta",
    "median_relative_loss_delta",
    "n_hit_lower_bound",
    "n_hit_upper_bound",
    "fallback_count_distribution",
    "poor_convergence_count_distribution",
    "cell0_cost_min",
    "cell0_cost_max",
    "recommendation",
    "evidence_boundary",
)

BOUNDED_CORRECTION_TRAJECTORY_COLUMNS: tuple[str, ...] = (
    "correction_name",
    "case_id",
    "phase",
    "dataset_id",
    "source_candidate_id",
    "independence_group",
    "N_cells",
    "step",
    "bound_low",
    "bound_high",
    "parameter_value",
    "loss",
    "fd_gradient",
    "accepted_step_size",
    "next_parameter_value",
    "next_loss",
    "line_search_trials",
    "solve_status",
    "fallback_count",
    "poor_convergence_count",
    "poor_convergence_cell_index",
    "cell0_cost",
    "CO_volpct_dry",
    "H2_volpct_dry",
    "CO2_volpct_dry",
    "target_CO_volpct_dry",
    "target_H2_volpct_dry",
    "target_CO2_volpct_dry",
    "evidence_boundary",
)


def _get(row: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if hasattr(row, "get"):
        return row.get(key, default)
    return getattr(row, key, default)


def clamp(value: float, low: float, high: float) -> float:
    return min(max(float(value), float(low)), float(high))


def finite_difference_points(value: float, low: float, high: float, fraction: float = 0.05) -> tuple[float, float]:
    step = max(abs(float(value)) * float(fraction), 0.005)
    minus = clamp(float(value) - step, low, high)
    plus = clamp(float(value) + step, low, high)
    if plus == minus:
        raise ValueError("finite-difference points collapsed at correction bounds")
    return minus, plus


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


def _case_meta(model_case_row: Mapping[str, Any], built: Mapping[str, Any], n_cells: int) -> dict[str, Any]:
    return {
        "case_id": built["case_id"],
        "phase": _get(model_case_row, "phase"),
        "dataset_id": _get(model_case_row, "dataset_id"),
        "source_candidate_id": built["segment_id"],
        "independence_group": _get(model_case_row, "independence_group_id", _get(model_case_row, "independence_group_id_value")),
        "N_cells": int(n_cells),
    }


def _loss_from_kpis(kpis: Mapping[str, float], targets: Mapping[str, Any]) -> float:
    target_co = float(targets["outlet_co_dry_vol_pct"])
    target_h2 = float(targets["outlet_h2_dry_vol_pct"])
    target_co2 = float(targets["outlet_co2_dry_vol_pct"])
    # Scale by 10 vol% so the smoke objective is dimensionless and not dominated
    # by one composition component solely due to units.
    return float(
        ((float(kpis["CO_volpct_dry"]) - target_co) / 10.0) ** 2
        + ((float(kpis["H2_volpct_dry"]) - target_h2) / 10.0) ** 2
        + ((float(kpis["CO2_volpct_dry"]) - target_co2) / 10.0) ** 2
    )


def solve_with_wgs_correction(
    model_case_row: Mapping[str, Any],
    correction_value: float,
    *,
    n_cells: int = 3,
) -> dict[str, Any]:
    built = build_gasifier_inputs_from_model_case(model_case_row)
    op_conds = dict(built["op_conds"])
    op_conds["WGS_CatalyticFactor"] = float(correction_value)
    start = time.perf_counter()
    system = GasifierSystem(built["geometry"], built["coal_props"], op_conds)
    profile, _z = system.solve(N_cells=int(n_cells), solver_method="minimize", jacobian_mode="scipy", jax_warmup=False)
    runtime_s = time.perf_counter() - start
    kpis = _dry_syngas_kpis(profile)
    residual_quality = cell_residual_quality(system, profile)
    poor_cells = [
        item["cell_index"]
        for item in residual_quality
        if item["cost"] > POOR_CONVERGENCE_COST_THRESHOLD
    ]
    cell0 = residual_quality[0] if residual_quality else {}
    targets = built["targets"]
    out = {
        **_case_meta(model_case_row, built, n_cells),
        "runtime_s": runtime_s,
        "loss": _loss_from_kpis(kpis, targets),
        "fallback_count": int(system.solve_stats.get("fallback_count", 0)),
        "poor_convergence_count": int(system.solve_stats.get("poor_convergence_count", 0)),
        "poor_convergence_cell_index": ";".join(str(i) for i in poor_cells),
        "cell0_cost": float(cell0.get("cost", math.nan)),
        "solve_status": "ok",
        "target_CO_volpct_dry": float(targets["outlet_co_dry_vol_pct"]),
        "target_H2_volpct_dry": float(targets["outlet_h2_dry_vol_pct"]),
        "target_CO2_volpct_dry": float(targets["outlet_co2_dry_vol_pct"]),
    }
    out.update(kpis)
    return out


def _strictly_decreased(final_loss: float, initial_loss: float) -> bool:
    """仅当最终损失比初始损失低出一个容差以上时才判为下降。

    非有限值（求解失败）一律判为未下降，不能用 NaN 比较得出结论。
    """
    final_value = float(final_loss)
    initial_value = float(initial_loss)
    if not (math.isfinite(final_value) and math.isfinite(initial_value)):
        return False
    return final_value < initial_value - LOSS_DECREASE_ABS_TOL


def _finite_or_none(value: Any) -> float | None:
    """把可能缺失或非有限的诊断量转成 float 或 None，供汇总统计安全读取。"""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _probe_failure_reason(minus_eval: Mapping[str, Any], plus_eval: Mapping[str, Any],
                          minus_theta: float, plus_theta: float) -> str:
    """把差分探针失败的原因拼成可读字符串，保留两侧各自的 failure_reason。"""
    parts = []
    for label, ev, theta_probe in (("minus", minus_eval, minus_theta), ("plus", plus_eval, plus_theta)):
        loss_value = _finite_or_none(ev.get("loss"))
        status = ev.get("solve_status", "unknown")
        detail = ev.get("failure_reason")
        if status == "ok" and loss_value is not None:
            continue
        text = f"{label} probe at theta={float(theta_probe):.6g}: status={status}, loss={ev.get('loss')!r}"
        if detail:
            text += f", reason={detail}"
        parts.append(text)
    if not parts:
        return "finite difference gradient is not finite"
    return "individual probe failed: " + " | ".join(parts)


def _loss(model_case_row: Mapping[str, Any], value: float, n_cells: int) -> dict[str, Any]:
    try:
        return solve_with_wgs_correction(model_case_row, value, n_cells=n_cells)
    except Exception as exc:  # pragma: no cover - diagnostic failure path
        built = build_gasifier_inputs_from_model_case(model_case_row)
        return {
            **_case_meta(model_case_row, built, n_cells),
            "loss": math.nan,
            "solve_status": "failed",
            "failure_reason": f"{type(exc).__name__}: {exc}",
        }


def _build_summary_row(
    *,
    model_case_row: Mapping[str, Any],
    n_cells: int,
    bound_low: float,
    bound_high: float,
    initial_value: float,
    final_value: float,
    initial_loss: float,
    latest_eval: Mapping[str, Any],
    steps_accepted: int,
    final_loss: float,
    status: str,
    failure_reason: str | None,
) -> dict[str, Any]:
    """构造一行 case 级摘要。

    status 必须以 `ok` 或 `failed` 开头：批次汇总按前缀区分成功案例，
    求解失败的案例不得混入成功统计。
    """
    delta = final_loss - initial_loss
    return {
        "correction_name": CORRECTION_NAME,
        "correction_layer": "existing WGS reaction-family multiplier via op_conds.WGS_CatalyticFactor",
        "bound_low": float(bound_low),
        "bound_high": float(bound_high),
        "initial_theta": float(initial_value),
        "final_theta": float(final_value),
        "initial_value": float(initial_value),
        "final_value": float(final_value),
        "loss_initial": initial_loss,
        "loss_final": final_loss,
        "loss_delta": delta,
        "relative_loss_delta": delta / max(abs(initial_loss), 1.0e-12) if math.isfinite(initial_loss) else math.nan,
        "loss_decreased": _strictly_decreased(final_loss, initial_loss),
        "steps": int(steps_accepted),
        "steps_completed": int(steps_accepted),
        "hit_lower_bound": math.isclose(float(final_value), float(bound_low), rel_tol=0.0, abs_tol=1.0e-10),
        "hit_upper_bound": math.isclose(float(final_value), float(bound_high), rel_tol=0.0, abs_tol=1.0e-10),
        "target": CORRECTION_TARGET,
        **{key: latest_eval.get(key) for key in ("case_id", "phase", "dataset_id", "source_candidate_id", "independence_group", "N_cells")},
        "fallback_count_final": latest_eval.get("fallback_count"),
        "poor_convergence_count_final": latest_eval.get("poor_convergence_count"),
        "poor_convergence_cell_index_final": latest_eval.get("poor_convergence_cell_index"),
        "cell0_cost_final": latest_eval.get("cell0_cost"),
        "status": status,
        "failure_reason": failure_reason,
        "evidence_boundary": BOUNDED_CORRECTION_EVIDENCE_BOUNDARY,
    }


def run_bounded_wgs_correction_smoke(
    model_case_rows: Iterable[Mapping[str, Any]],
    *,
    n_cells: int = 3,
    initial_value: float = 1.0,
    bound_low: float = CORRECTION_BOUND_LOW,
    bound_high: float = CORRECTION_BOUND_HIGH,
    steps: int = 5,
    learning_rate: float = 0.2,
    fd_fraction: float = 0.05,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trajectory: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for model_case_row in model_case_rows:
        theta = clamp(initial_value, bound_low, bound_high)
        first_eval = _loss(model_case_row, theta, n_cells)
        initial_loss = float(first_eval["loss"])
        latest_eval = first_eval
        if not math.isfinite(initial_loss):
            # 初始求解就失败：不再做任何梯度步，也不产出轨迹行。
            # 状态必须以 failed 开头，否则会被批次汇总当成成功案例。
            summaries.append(
                _build_summary_row(
                    model_case_row=model_case_row,
                    n_cells=n_cells,
                    bound_low=bound_low,
                    bound_high=bound_high,
                    initial_value=initial_value,
                    final_value=theta,
                    initial_loss=initial_loss,
                    latest_eval=latest_eval,
                    steps_accepted=0,
                    final_loss=initial_loss,
                    status="failed_initial_solve",
                    failure_reason=str(first_eval.get("failure_reason") or "non-finite initial loss"),
                )
            )
            continue
        steps_accepted = 0
        termination_reason: str | None = None
        for step in range(int(steps)):
            minus_theta, plus_theta = finite_difference_points(theta, bound_low, bound_high, fd_fraction)
            minus_eval = _loss(model_case_row, minus_theta, n_cells)
            plus_eval = _loss(model_case_row, plus_theta, n_cells)
            gradient = (float(plus_eval["loss"]) - float(minus_eval["loss"])) / (plus_theta - minus_theta)
            if not math.isfinite(gradient):
                # 差分探针求解失败，梯度不可用：停在当前点，并保留探针的失败原因。
                # 不能把这个案例当成正常的「无改善」——那是优化真的跑完但没找到改进。
                termination_reason = _probe_failure_reason(minus_eval, plus_eval, minus_theta, plus_theta)
                break
            proposed_step = -float(learning_rate) * gradient
            trials = 0
            accepted_theta = theta
            accepted_eval = latest_eval
            accepted_step_size = 0.0
            for factor in (1.0, 0.5, 0.25, 0.1, 0.05):
                trials += 1
                candidate_theta = clamp(theta + proposed_step * factor, bound_low, bound_high)
                if abs(candidate_theta - theta) < 1.0e-14:
                    continue
                candidate_eval = _loss(model_case_row, candidate_theta, n_cells)
                if (
                    candidate_eval.get("solve_status") == "ok"
                    and math.isfinite(float(candidate_eval["loss"]))
                    # 必须严格下降才接受，否则平坦目标下 theta 会一直沿方向漂移。
                    and _strictly_decreased(float(candidate_eval["loss"]), float(latest_eval["loss"]))
                ):
                    accepted_theta = candidate_theta
                    accepted_eval = candidate_eval
                    accepted_step_size = candidate_theta - theta
                    break
            if accepted_step_size != 0.0:
                steps_accepted += 1
            trajectory.append(
                {
                    "correction_name": CORRECTION_NAME,
                    **{key: latest_eval.get(key) for key in ("case_id", "phase", "dataset_id", "source_candidate_id", "independence_group", "N_cells")},
                    "step": step,
                    "bound_low": float(bound_low),
                    "bound_high": float(bound_high),
                    "parameter_value": theta,
                    "loss": float(latest_eval["loss"]),
                    "fd_gradient": gradient,
                    "accepted_step_size": accepted_step_size,
                    "next_parameter_value": accepted_theta,
                    "next_loss": float(accepted_eval["loss"]),
                    "line_search_trials": trials,
                    "solve_status": accepted_eval.get("solve_status", "ok"),
                    "fallback_count": accepted_eval.get("fallback_count"),
                    "poor_convergence_count": accepted_eval.get("poor_convergence_count"),
                    "poor_convergence_cell_index": accepted_eval.get("poor_convergence_cell_index"),
                    "cell0_cost": accepted_eval.get("cell0_cost"),
                    "CO_volpct_dry": accepted_eval.get("CO_volpct_dry"),
                    "H2_volpct_dry": accepted_eval.get("H2_volpct_dry"),
                    "CO2_volpct_dry": accepted_eval.get("CO2_volpct_dry"),
                    "target_CO_volpct_dry": accepted_eval.get("target_CO_volpct_dry"),
                    "target_H2_volpct_dry": accepted_eval.get("target_H2_volpct_dry"),
                    "target_CO2_volpct_dry": accepted_eval.get("target_CO2_volpct_dry"),
                    "evidence_boundary": BOUNDED_CORRECTION_EVIDENCE_BOUNDARY,
                }
            )
            theta = accepted_theta
            latest_eval = accepted_eval

        final_loss = float(latest_eval["loss"])
        decreased = _strictly_decreased(final_loss, initial_loss)
        if termination_reason is not None:
            # 探针失败导致提前终止：不能用 ok_* 掩盖，否则批次会把它计为
            # 「跑完了但没改善」，而真实原因是梯度不可用。
            status = "failed_probe"
            failure_reason = termination_reason
        else:
            status = "ok_loss_decreased" if decreased else "ok_no_loss_decrease"
            failure_reason = None
        summaries.append(
            _build_summary_row(
                model_case_row=model_case_row,
                n_cells=n_cells,
                bound_low=bound_low,
                bound_high=bound_high,
                initial_value=initial_value,
                final_value=theta,
                initial_loss=initial_loss,
                latest_eval=latest_eval,
                steps_accepted=steps_accepted,
                final_loss=final_loss,
                status=status,
                failure_reason=failure_reason,
            )
        )

    return (
        [{column: row.get(column) for column in BOUNDED_CORRECTION_SUMMARY_COLUMNS} for row in summaries],
        [{column: row.get(column) for column in BOUNDED_CORRECTION_TRAJECTORY_COLUMNS} for row in trajectory],
    )


def _distribution(values: Sequence[Any]) -> str:
    counts = Counter(str(value) for value in values)
    return ";".join(f"{key}:{counts[key]}" for key in sorted(counts))


def _recommendation_for_summary(rows: Sequence[Mapping[str, Any]]) -> str:
    n_cases = len(rows)
    n_decreased = sum(1 for row in rows if bool(row.get("loss_decreased")))
    n_lower = sum(1 for row in rows if bool(row.get("hit_lower_bound")))
    if n_cases == 0:
        return "blocked_no_cases"
    if n_decreased == n_cases and n_lower == 0:
        return "supplementary_feasibility_figure_candidate"
    if n_decreased >= max(1, math.ceil(0.75 * n_cases)):
        return "supplementary_or_feasibility_note_due_to_boundary_hits_or_mixed_effects"
    return "feasibility_note_only"


def build_bounded_correction_batch_summary(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate case-level bounded correction smoke rows into batch/phase summaries."""
    groups: list[tuple[str, str, list[Mapping[str, Any]]]] = [("all", "all", list(summaries))]
    phases = sorted({str(row.get("phase")) for row in summaries})
    groups.extend(("phase", phase, [row for row in summaries if str(row.get("phase")) == phase]) for phase in phases)

    out: list[dict[str, Any]] = []
    for scope, phase, rows in groups:
        # 只有 status 以 ok 开头的才是成功案例；failed_* 不参与成功统计。
        ok_rows = [row for row in rows if str(row.get("status", "")).startswith("ok")]
        loss_delta = [v for v in (_finite_or_none(row.get("loss_delta")) for row in ok_rows) if v is not None]
        rel_delta = [v for v in (_finite_or_none(row.get("relative_loss_delta")) for row in ok_rows) if v is not None]
        cell0 = [v for v in (_finite_or_none(row.get("cell0_cost_final")) for row in ok_rows) if v is not None]
        out.append(
            {
                "scope": scope,
                "phase": phase,
                "n_cases": len(rows),
                "n_ok": len(ok_rows),
                "n_failed": sum(1 for row in rows if str(row.get("status", "")).startswith("failed")),
                "n_loss_decreased": sum(1 for row in ok_rows if bool(row.get("loss_decreased"))),
                "median_loss_delta": float(np.median(loss_delta)) if loss_delta else math.nan,
                "median_relative_loss_delta": float(np.median(rel_delta)) if rel_delta else math.nan,
                "n_hit_lower_bound": sum(1 for row in ok_rows if bool(row.get("hit_lower_bound"))),
                "n_hit_upper_bound": sum(1 for row in ok_rows if bool(row.get("hit_upper_bound"))),
                "fallback_count_distribution": _distribution([row.get("fallback_count_final") for row in ok_rows]),
                "poor_convergence_count_distribution": _distribution([row.get("poor_convergence_count_final") for row in ok_rows]),
                "cell0_cost_min": min(cell0) if cell0 else math.nan,
                "cell0_cost_max": max(cell0) if cell0 else math.nan,
                "recommendation": _recommendation_for_summary(ok_rows),
                "evidence_boundary": BOUNDED_CORRECTION_EVIDENCE_BOUNDARY,
            }
        )
    return [{column: row.get(column) for column in BOUNDED_CORRECTION_BATCH_COLUMNS} for row in out]


def bounded_correction_markdown(
    summaries: Sequence[Mapping[str, Any]],
    trajectory: Sequence[Mapping[str, Any]],
    batch_rows: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    batch_rows = list(batch_rows or build_bounded_correction_batch_summary(summaries))
    overall = batch_rows[0] if batch_rows else {}
    lines = [
        "# Bounded hybrid-correction smoke protocol",
        "",
        "This diagnostic uses the existing WGS reaction-family multiplier as a bounded scalar correction smoke.",
        "It intentionally avoids first-cell source-term hooks because those would require new solver/source interfaces.",
        "",
        f"- Correction: {CORRECTION_NAME}",
        f"- Bounds: [{CORRECTION_BOUND_LOW}, {CORRECTION_BOUND_HIGH}]",
        f"- Target: {CORRECTION_TARGET}",
        f"- Summary rows: {len(summaries)}",
        f"- Trajectory rows: {len(trajectory)}",
        f"- Batch recommendation: {overall.get('recommendation', 'n/a')}",
        f"- Loss-decreased cases: {overall.get('n_loss_decreased', 'n/a')}/{overall.get('n_cases', 'n/a')}",
        f"- Bound hits: lower={overall.get('n_hit_lower_bound', 'n/a')}, upper={overall.get('n_hit_upper_bound', 'n/a')}",
        "",
        "## Batch summary",
        "",
        "| scope | phase | n_cases | n_loss_decreased | median_loss_delta | median_relative_loss_delta | lower_hits | upper_hits | recommendation |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in batch_rows:
        lines.append(
            "| {scope} | {phase} | {n} | {decreased} | {delta:.6g} | {rel:.6g} | {low} | {high} | {rec} |".format(
                scope=row["scope"],
                phase=row["phase"],
                n=int(row["n_cases"]),
                decreased=int(row["n_loss_decreased"]),
                delta=float(row["median_loss_delta"]),
                rel=float(row["median_relative_loss_delta"]),
                low=int(row["n_hit_lower_bound"]),
                high=int(row["n_hit_upper_bound"]),
                rec=row["recommendation"],
            )
        )
    lines.extend(
        [
        "",
        "## Case rows",
        "",
        "| case_id | N | initial | final | loss_initial | loss_final | status | poor_final |",
        "|---|---:|---:|---:|---:|---:|---|---:|",
        ]
    )
    for row in summaries:
        lines.append(
            "| {case} | {n} | {initial:.6g} | {final:.6g} | {li:.6g} | {lf:.6g} | {status} | {poor} |".format(
                case=row["case_id"],
                n=int(row["N_cells"]),
                initial=float(row["initial_value"]),
                final=float(row["final_value"]),
                li=float(row["loss_initial"]),
                lf=float(row["loss_final"]),
                status=row["status"],
                poor=row["poor_convergence_count_final"],
            )
        )
    lines.extend(
        [
            "",
            "Protocol decision: first-cell source-term correction remains design-only in this round; "
            "the implemented smoke uses an existing reaction-family parameter so public solver APIs and source-term semantics are unchanged.",
            "",
            f"Evidence boundary: {BOUNDED_CORRECTION_EVIDENCE_BOUNDARY}.",
        ]
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "BOUNDED_CORRECTION_EVIDENCE_BOUNDARY",
    "BOUNDED_CORRECTION_BATCH_COLUMNS",
    "BOUNDED_CORRECTION_SUMMARY_COLUMNS",
    "BOUNDED_CORRECTION_TRAJECTORY_COLUMNS",
    "build_bounded_correction_batch_summary",
    "bounded_correction_markdown",
    "clamp",
    "finite_difference_points",
    "run_bounded_wgs_correction_smoke",
    "solve_with_wgs_correction",
]
