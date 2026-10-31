from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))


def test_bounds_and_fd_points_are_enforced():
    from model.bounded_hybrid_correction_smoke import clamp, finite_difference_points

    assert clamp(-1.0, 0.02, 2.0) == 0.02
    assert clamp(3.0, 0.02, 2.0) == 2.0
    minus, plus = finite_difference_points(1.0, 0.02, 2.0, 0.05)
    assert minus == 0.95
    assert plus == 1.05
    minus, plus = finite_difference_points(0.021, 0.02, 2.0, 0.05)
    assert minus == 0.02
    assert plus > minus


def test_bounded_correction_schema_and_boundary_from_fixture_rows():
    from model.bounded_hybrid_correction_smoke import (
        BOUNDED_CORRECTION_EVIDENCE_BOUNDARY,
        BOUNDED_CORRECTION_BATCH_COLUMNS,
        BOUNDED_CORRECTION_SUMMARY_COLUMNS,
        BOUNDED_CORRECTION_TRAJECTORY_COLUMNS,
        build_bounded_correction_batch_summary,
        bounded_correction_markdown,
    )

    summary = {
        "correction_name": "WGS_CatalyticFactor_bounded_reaction_family",
        "correction_layer": "existing WGS reaction-family multiplier via op_conds.WGS_CatalyticFactor",
        "bound_low": 0.02,
        "bound_high": 2.0,
        "initial_theta": 1.0,
        "final_theta": 0.5,
        "initial_value": 1.0,
        "final_value": 0.5,
        "loss_initial": 2.0,
        "loss_final": 1.0,
        "loss_delta": -1.0,
        "relative_loss_delta": -0.5,
        "loss_decreased": True,
        "steps": 3,
        "steps_completed": 3,
        "hit_lower_bound": False,
        "hit_upper_bound": False,
        "target": "weighted_outlet_CO_H2_CO2_squared_error",
        "case_id": "case_1",
        "phase": "phase1",
        "dataset_id": "dataset",
        "source_candidate_id": "seg_1",
        "independence_group": "IG001",
        "N_cells": 3,
        "fallback_count_final": 0,
        "poor_convergence_count_final": 1,
        "poor_convergence_cell_index_final": "0",
        "cell0_cost_final": 0.01,
        "status": "ok_loss_decreased",
        "failure_reason": None,
        "evidence_boundary": BOUNDED_CORRECTION_EVIDENCE_BOUNDARY,
    }
    trajectory = {
        "correction_name": "WGS_CatalyticFactor_bounded_reaction_family",
        "case_id": "case_1",
        "phase": "phase1",
        "dataset_id": "dataset",
        "source_candidate_id": "seg_1",
        "independence_group": "IG001",
        "N_cells": 3,
        "step": 0,
        "bound_low": 0.02,
        "bound_high": 2.0,
        "parameter_value": 1.0,
        "loss": 2.0,
        "fd_gradient": 1.0,
        "accepted_step_size": -0.2,
        "next_parameter_value": 0.8,
        "next_loss": 1.8,
        "line_search_trials": 1,
        "solve_status": "ok",
        "fallback_count": 0,
        "poor_convergence_count": 1,
        "poor_convergence_cell_index": "0",
        "cell0_cost": 0.01,
        "CO_volpct_dry": 67.0,
        "H2_volpct_dry": 22.0,
        "CO2_volpct_dry": 10.0,
        "target_CO_volpct_dry": 62.0,
        "target_H2_volpct_dry": 28.0,
        "target_CO2_volpct_dry": 9.0,
        "evidence_boundary": BOUNDED_CORRECTION_EVIDENCE_BOUNDARY,
    }

    assert tuple(summary.keys()) == BOUNDED_CORRECTION_SUMMARY_COLUMNS
    assert tuple(trajectory.keys()) == BOUNDED_CORRECTION_TRAJECTORY_COLUMNS
    assert summary["loss_final"] <= summary["loss_initial"]
    batch = build_bounded_correction_batch_summary([summary])
    assert tuple(batch[0].keys()) == BOUNDED_CORRECTION_BATCH_COLUMNS
    assert batch[0]["n_cases"] == 1
    assert batch[0]["n_loss_decreased"] == 1
    assert batch[0]["n_hit_lower_bound"] == 0
    assert batch[0]["recommendation"] == "supplementary_feasibility_figure_candidate"
    text = bounded_correction_markdown([summary], [trajectory], batch)
    assert "Batch summary" in text
    assert "first-cell source-term correction remains design-only" in text
    assert "not a trained hybrid" in text


def _fake_loss_context(monkeypatch_loss):
    """Patch the objective used by the smoke loop so cases need no solver run."""
    import model.bounded_hybrid_correction_smoke as module

    module._loss = monkeypatch_loss
    return module


def test_flat_objective_is_not_reported_as_decreased():
    """恒定损失不得记成「下降」，theta 也不应沿梯度方向漂移。

    回归 Codex 审查意见（P2）：原实现用 `final_loss <= initial_loss`，
    平坦目标下 loss_delta=0 仍被标为 ok_loss_decreased，批次推荐还会给出
    supplementary_feasibility_figure_candidate。
    """
    module = _fake_loss_context(lambda row, value, n_cells: {
        "case_id": "case_flat", "phase": "phase1", "dataset_id": "d",
        "source_candidate_id": "s", "independence_group": "IG001", "N_cells": 3,
        "loss": 1.0, "solve_status": "ok", "cell0_cost": 0.01,
    })
    summaries, trajectory = module.run_bounded_wgs_correction_smoke([{}], n_cells=3, steps=4)
    row = summaries[0]
    assert row["loss_delta"] == 0.0
    assert row["loss_decreased"] is False
    assert row["status"] == "ok_no_loss_decrease"
    # 严格下降才接受步长，因此参数不动、也没有已完成的步
    assert row["final_value"] == row["initial_value"]
    assert row["steps_completed"] == 0
    assert all(item["accepted_step_size"] == 0.0 for item in trajectory)

    batch = module.build_bounded_correction_batch_summary(summaries)
    assert batch[0]["n_loss_decreased"] == 0
    assert batch[0]["recommendation"] == "feasibility_note_only"


def test_failed_initial_solve_is_reported_and_does_not_break_batch():
    """初始求解失败必须传播为 failed 状态，且不炸掉批次汇总。

    回归 Codex 审查意见（P2）：原实现把失败案例标成 ok_no_loss_decrease，
    批次汇总再把缺失的 cell0_cost_final 拿去 float(None)，整张报表生成失败。
    """
    module = _fake_loss_context(lambda row, value, n_cells: {
        "case_id": "case_bad", "phase": "phase1", "dataset_id": "d",
        "source_candidate_id": "s", "independence_group": "IG001", "N_cells": 3,
        "loss": float("nan"), "solve_status": "failed", "failure_reason": "RuntimeError: boom",
    })
    summaries, trajectory = module.run_bounded_wgs_correction_smoke([{}], n_cells=3, steps=4)
    row = summaries[0]
    assert row["status"].startswith("failed")
    assert row["failure_reason"] == "RuntimeError: boom"
    assert row["loss_decreased"] is False
    assert row["steps_completed"] == 0
    assert trajectory == []  # 初始就失败，不产出梯度轨迹

    batch = module.build_bounded_correction_batch_summary(summaries)
    assert batch[0]["n_cases"] == 1
    assert batch[0]["n_ok"] == 0
    assert batch[0]["n_failed"] == 1
    assert batch[0]["recommendation"] == "blocked_no_cases"


def test_failed_case_does_not_break_mixed_batch():
    """一个失败案例不应让同批次其他成功案例的汇总生成失败。"""
    module = _fake_loss_context(lambda row, value, n_cells: {
        "case_id": "mixed", "phase": "phase1", "dataset_id": "d",
        "source_candidate_id": "s", "independence_group": "IG001", "N_cells": 3,
        "loss": float("nan"), "solve_status": "failed", "failure_reason": "boom",
    })
    failures, _ = module.run_bounded_wgs_correction_smoke([{}], n_cells=3, steps=1)
    ok_summary = {
        **{col: None for col in module.BOUNDED_CORRECTION_SUMMARY_COLUMNS},
        "phase": "phase1",
        "loss_initial": 2.0, "loss_final": 1.0, "loss_delta": -1.0,
        "relative_loss_delta": -0.5, "loss_decreased": True,
        "status": "ok_loss_decreased", "cell0_cost_final": 0.02,
        "hit_lower_bound": False, "hit_upper_bound": False,
    }
    batch = module.build_bounded_correction_batch_summary([failures[0], ok_summary])
    overall = batch[0]
    assert overall["n_cases"] == 2
    assert overall["n_ok"] == 1
    assert overall["n_failed"] == 1
    assert overall["n_loss_decreased"] == 1
