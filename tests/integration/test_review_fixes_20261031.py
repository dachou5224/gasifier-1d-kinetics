"""2026-10-31 审查意见的回归测试。

覆盖：
1. null 控制打乱特征时不得改动真值（round56）
2. 缓存必须按指纹校验，不能只看行数（round56）
3. 求解失败时 alpha_wgs_xt 不得因 KeyError 中止（alpha_wgs_xt_baseline）
4. 求解失败时 char-oxidation 计数器不得 `int(nan)` 崩溃
5. 已提交的 manifest 必须是可移植的仓库相对路径
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(_REPO_ROOT / "src"))


def _load_round56():
    """round56 是 scripts 下的模块，按文件加载，避免改动 sys.path 顺序。"""
    path = _REPO_ROOT / "scripts" / "generate_huayi_round56_wgs_comparators.py"
    spec = importlib.util.spec_from_file_location("round56_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- 1. null shuffle

def test_null_shuffle_keeps_ground_truth_from_unshuffled_rows(tmp_path):
    """被打乱的行只提供预测输入，真值必须来自原始行。

    回归审查意见（P2）：`syngas_flow_mean` 同时是被选特征和 `dry_syngas_flow`
    的真值来源。打乱后真值被改、而该 target 的预测是"该行自身输入"的携带值，
    于是凭空产生与该扰动无关的流量误差，抬高了 null objective。
    """
    module = _load_round56()

    class FakeRound54:
        N_CELLS = 1
        TARGETS = (("dry_syngas_flow", "syngas_flow_mean", "pred_flow"),)

    rows = [
        {"window_id": "w1", "split_role": "train", "dataset_id": "d", "phase_number": 1,
         "syngas_flow_mean": 100.0, "syngas_dry_volumetric_flow_value": 100.0},
        {"window_id": "w2", "split_role": "train", "dataset_id": "d", "phase_number": 1,
         "syngas_flow_mean": 200.0, "syngas_dry_volumetric_flow_value": 200.0},
    ]
    original = [dict(r) for r in rows]
    shuffled = [dict(r) for r in rows]
    shuffled[0]["syngas_flow_mean"], shuffled[1]["syngas_flow_mean"] = (
        shuffled[1]["syngas_flow_mean"], shuffled[0]["syngas_flow_mean"])

    # 求解结果：预测 = 传入行自带的携带字段（模拟 worker 行为）
    def fake_run_row(row, *, alpha, n_cells, timeout_s):
        return {"status": "ok", "wall_time_s": 0.0, "pred_flow": float(row["syngas_dry_volumetric_flow_value"])}

    module._run_row = fake_run_row
    alpha_map = {"w1": 0.5, "w2": 0.5}

    corrected = module._prediction_rows_for_alpha(
        FakeRound54, shuffled, method="m", alpha_by_window=alpha_map,
        output_dir=tmp_path, cache_name="c_corrected.csv", timeout_s=1.0,
        target_rows=original)
    buggy = module._prediction_rows_for_alpha(
        FakeRound54, shuffled, method="m", alpha_by_window=alpha_map,
        output_dir=tmp_path, cache_name="c_buggy.csv", timeout_s=1.0)

    # 修正版：真值取自原始行，与携带的预测一致 → 残差 0
    np.testing.assert_allclose(corrected["residual"].to_numpy(), [0.0, 0.0], atol=1e-12)
    # 未修正版：真值被交换，出现虚假误差
    assert (buggy["residual"].abs().to_numpy() > 1e-9).all()

    # 且原始行未被就地修改（调用方可能复用）
    assert [r["syngas_flow_mean"] for r in original] == [100.0, 200.0]


def test_objective_excludes_only_present_targets():
    """objective 只对实际存在的 target 求均值，不做隐含补齐。"""
    module = _load_round56()
    preds = pd.DataFrame([
        {"split_role": "train", "status": "ok", "target": "CO_dry_vol_pct", "residual": 10.0},
        {"split_role": "train", "status": "ok", "target": "dry_syngas_flow", "residual": 0.0},
    ])
    # CO: 10/10 = 1.0；flow: 0/1e5 = 0.0 → 均值 0.5
    assert module._objective(preds, "train") == pytest.approx(0.5)


# ------------------------------------------------------------ 2. cache fingerprint

def test_cache_rejects_a_complete_cache_with_a_different_fingerprint(tmp_path):
    """行数与目标数都齐全、但指纹不同的缓存必须作废。

    回归审查意见（P2）：原校验只比计数，于是「5 个 target / N 个窗口」的旧缓存
    会被静默复用，使报告的选择与指标矩阵描述不同实验。这里构造一个**列完整**
    的缓存，确保测试真正区分「有无指纹校验」，而不是靠缺列触发的读取异常。
    """
    module = _load_round56()
    targets = ["CO_dry_vol_pct", "H2_dry_vol_pct", "CO2_dry_vol_pct",
               "H2_CO_ratio", "dry_syngas_flow"]
    frame = pd.DataFrame({
        "window_id": ["w1"] * 5,
        "target": targets,
        "status": ["ok"] * 5,
        "cache_fingerprint": ["0ldf1ngerpr1nt"] * 5,
    })
    path = tmp_path / "complete.csv"
    frame.to_csv(path, index=False)

    # 指纹不符 -> False
    assert module._cache_has_windows(path, 1, "newf1ngerpr1nt") is False
    # 指纹相符 -> True（证明上一条不是因为其他原因失败）
    assert module._cache_has_windows(path, 1, "0ldf1ngerpr1nt") is True
    # 不传指纹时退化为旧的计数校验（保留给非模型相关的缓存）
    assert module._cache_has_windows(path, 1) is True


def test_cache_fingerprint_separates_alpha_maps():
    """指纹必须随逐窗 alpha 变化，否则不同模型无法区分。"""
    module = _load_round56()
    base = module._cache_fingerprint("m", {"w1": 0.5, "w2": 0.5}, "input_rows")
    assert base == module._cache_fingerprint("m", {"w2": 0.5, "w1": 0.5}, "input_rows")
    assert base != module._cache_fingerprint("m", {"w1": 0.05, "w2": 0.5}, "input_rows")
    assert base != module._cache_fingerprint("other", {"w1": 0.5, "w2": 0.5}, "input_rows")
    assert base != module._cache_fingerprint("m", {"w1": 0.5, "w2": 0.5}, "original_rows")


def test_cache_without_fingerprint_column_is_stale(tmp_path):
    """旧缓存没有指纹列时必须作废（读取失败即视为不可用）。"""
    module = _load_round56()
    stale = pd.DataFrame({
        "window_id": ["w1"] * 5,
        "target": ["CO_dry_vol_pct", "H2_dry_vol_pct", "CO2_dry_vol_pct",
                   "H2_CO_ratio", "dry_syngas_flow"],
        "status": ["ok"] * 5,
    })
    path = tmp_path / "stale.csv"
    stale.to_csv(path, index=False)
    assert module._cache_has_windows(path, 1, "deadbeefdeadbeef") is False


# ------------------------------------------------------- 3. alpha_wgs_xt failures

def test_failed_solve_emits_rows_instead_of_raising_keyerror():
    """求解失败必须产出带失败状态的行，而不是因缺字段抛 KeyError。

    回归审查意见（P2）：`_safe_solve` 的失败记录不含 target/prediction 字段，
    原实现直接索引并抛 KeyError，使单个失败案例中止整份产物。
    """
    from model.alpha_wgs_xt_baseline import _failed_target_values

    fixed_lookup = {
        ("c1", "CO_volpct_dry"): {"y_pred": 60.0, "target_value": 61.0},
        ("c1", "H2_volpct_dry"): {"y_pred": 28.0, "target_value": 29.0},
    }
    values = _failed_target_values(fixed_lookup, "c1")
    assert set(values) == {"CO_volpct_dry", "H2_volpct_dry", "CO2_volpct_dry", "H2_CO_ratio"}
    # 已记录的真值保留，预测置为 NaN
    assert values["CO_volpct_dry"] == (61.0, math.nan)
    assert math.isnan(values["H2_volpct_dry"][1])
    # 未记录的目标留 NaN 而不是抛错
    assert math.isnan(values["CO2_volpct_dry"][0])


# ------------------------------------------------- 4. char oxidation NaN counters

def test_char_oxidation_failed_solve_does_not_crash_on_nan_counters():
    """`_safe_solve` 失败时把计数器置 NaN；`int(nan)` 会让整张表崩掉。"""
    from model.global_char_oxidation_baseline import _count_or_nan

    assert math.isnan(_count_or_nan(math.nan))
    assert math.isnan(_count_or_nan(None))
    assert math.isnan(_count_or_nan("not-a-number"))
    assert _count_or_nan(3) == 3.0
    assert _count_or_nan(3.7) == 3.0


def test_char_oxidation_failed_prediction_row_path_handles_nan_counters():
    """端到端：伪造一个失败的求解，预测表仍能生成。"""
    from model.global_char_oxidation_baseline import char_oxidation_prediction_rows

    def failing_solver(row, alpha, n_cells):
        raise RuntimeError("boom")

    rows = char_oxidation_prediction_rows(
        [{"_model_case_id": "c1", "phase": "phase1", "dataset_id": "d",
          "_split": "train", "_split_role": "train"}],
        selected_alpha=1.0, n_cells=1, solver_fn=failing_solver)
    assert len(rows) == 4
    for row in rows:
        assert str(row["status"]).startswith("failed")
        assert math.isnan(row["fallback_count"])
        assert math.isnan(row["poor_convergence_count"])


# ------------------------------------------------------------ 5. manifest paths

@pytest.mark.parametrize("manifest_name", [
    "figure_data_manifest.json",
    "huayi_expanded_figure_data_manifest.json",
])
def test_committed_manifest_paths_are_portable(manifest_name):
    """已提交的 manifest 不得含作者机器的绝对路径，且每条 file 都能定位到实际文件。

    回归审查意见（P2，2026-10-31）：原文把 `/Users/<author>/...` 写死，
    仓库克隆到别处后按 manifest 读取会指向不存在的路径。
    """
    path = _REPO_ROOT / "outputs" / "figures_data" / manifest_name
    raw = path.read_text(encoding="utf-8")
    assert "/Users/" not in raw, f"{manifest_name} contains an absolute user path"

    manifest = json.loads(raw)
    out_dir = manifest.get("output_directory")
    if out_dir is not None:
        assert not Path(out_dir).is_absolute()
        assert (_REPO_ROOT / out_dir).is_dir()

    for name, entry in manifest.get("artifacts", {}).items():
        file_value = entry.get("file")
        if file_value is None:
            continue
        assert not Path(file_value).is_absolute(), f"{name}: absolute path {file_value}"
        assert (_REPO_ROOT / file_value).exists(), f"{name}: {file_value} does not resolve"


def test_figure_bundle_generator_emits_repo_relative_paths(tmp_path):
    """生成器本身也必须输出仓库相对路径，否则下次重生成会把绝对路径写回来。"""
    from model.figure_data_bundle import _repo_relative

    inside = _REPO_ROOT / "outputs" / "figures_data" / "x.csv"
    assert _repo_relative(inside, _REPO_ROOT) == "outputs/figures_data/x.csv"
    outside = Path("/tmp/elsewhere/x.csv")
    assert Path(_repo_relative(outside, _REPO_ROOT)).is_absolute()


def test_alpha_wgs_xt_prediction_rows_survive_a_failing_solver():
    """端到端：一个 case 求解失败时，整份预测表仍要生成。

    回归审查意见（P2）：原实现直接索引失败记录里不存在的 target 字段，
    抛 KeyError 使整个 baseline/robustness 产物中止。这里走真实主循环。
    """
    from model.alpha_wgs_xt_baseline import alpha_wgs_xt_prediction_rows

    from model.direct_residual_baseline import OPERATING_FEATURE_COLUMNS

    feature_row = {col: 1.0 for col in OPERATING_FEATURE_COLUMNS}
    feature_row.update({"case_id": "c1", "split": "train", "split_role": "train",
                        "dataset_id": "d"})
    fixed_rows = [
        {"case_id": "c1", "target": t, "N_cells": 1, "y_pred": 50.0, "target_value": 51.0}
        for t in ("CO_volpct_dry", "H2_volpct_dry", "CO2_volpct_dry", "H2_CO_ratio")
    ]
    n_features = len(OPERATING_FEATURE_COLUMNS)
    scaler = {"feature_columns": tuple(OPERATING_FEATURE_COLUMNS),
              "mean": tuple(1.0 for _ in OPERATING_FEATURE_COLUMNS),
              "scale": tuple(1.0 for _ in OPERATING_FEATURE_COLUMNS)}
    model = {"coefficients": [0.0] * n_features, "intercept": 0.0}

    def failing_solver(row, alpha, n_cells):
        raise RuntimeError("boom")

    rows = alpha_wgs_xt_prediction_rows(
        [{"_model_case_id": "c1", "phase": "phase1", "dataset_id": "d", "case_id": "c1"}],
        [feature_row], scaler, model, fixed_rows, [],
        n_cells=1, solver_fn=failing_solver,
    )
    assert len(rows) == 4
    for row in rows:
        assert str(row["status"]).startswith("failed")
        assert math.isnan(row["alpha_prediction"])
        assert math.isnan(row["alpha_residual"])
        # 真值从 fixed 基线保留下来，不因失败而丢失
        assert row["target_value"] == 51.0


def test_subset_fingerprints_are_rejected_by_the_full_cache_check(tmp_path):
    """前提验证：带子集指纹的缓存会被完整指纹校验拒绝。

    这条不触达修复代码，只固定 `_cache_has_windows` 的行为契约：它要求整张表
    只有一个指纹且等于期望值。这正是「合并两段子缓存后必须重写指纹」的原因；
    修复本身由 `test_selected_global_full_writes_a_single_fingerprint` 端到端覆盖。
    """
    module = _load_round56()
    targets = ("CO_dry_vol_pct", "H2_dry_vol_pct", "CO2_dry_vol_pct",
               "H2_CO_ratio", "dry_syngas_flow")
    train_ids, held_ids = ["w1", "w2"], ["w3"]
    alpha = {w: 0.5 for w in train_ids + held_ids}

    full_fp = module._cache_fingerprint("meth", alpha, "input_rows")
    train_fp = module._cache_fingerprint("meth", {w: 0.5 for w in train_ids}, "input_rows")
    held_fp = module._cache_fingerprint("meth", {w: 0.5 for w in held_ids}, "input_rows")
    assert len({full_fp, train_fp, held_fp}) == 3

    merged = pd.DataFrame([
        {"window_id": w, "target": t, "status": "ok",
         "cache_fingerprint": fp}
        for w, fp in [(w, train_fp) for w in train_ids] + [(w, held_fp) for w in held_ids]
        for t in targets
    ])
    # 写之前统一成完整指纹 —— 这是修复后的行为
    merged["cache_fingerprint"] = full_fp
    path = tmp_path / "_selected_full.csv"
    merged.to_csv(path, index=False)

    assert module._cache_has_windows(path, len(train_ids + held_ids), full_fp) is True
    # 反向：若保留子集指纹，校验必须失败（这才是修复前的情况）
    stale = merged.copy()
    stale.loc[stale["window_id"].isin(held_ids), "cache_fingerprint"] = held_fp
    stale_path = tmp_path / "_stale_full.csv"
    stale.to_csv(stale_path, index=False)
    assert module._cache_has_windows(stale_path, len(train_ids + held_ids), full_fp) is False


def test_selected_global_full_writes_a_single_fingerprint(tmp_path):
    """端到端：走 concat 路径后，full cache 里只应有一个指纹。"""
    module = _load_round56()
    targets = ("CO_dry_vol_pct", "H2_volpct_dry")  # 数量无关，这里只看指纹列

    class FakeRound54:
        N_CELLS = 1
        TARGETS = tuple((t, "t", f"p_{t}") for t in
                        ("CO_dry_vol_pct", "H2_dry_vol_pct", "CO2_dry_vol_pct",
                         "H2_CO_ratio", "dry_syngas_flow"))

    rows = [
        {"window_id": "w1", "split_role": "train", "dataset_id": "d", "phase_number": 1, "t": 1.0},
        {"window_id": "w2", "split_role": "test", "dataset_id": "d", "phase_number": 1, "t": 1.0},
    ]

    def fake_run_row(r, *, alpha, n_cells, timeout_s):
        return {"status": "ok", "wall_time_s": 0.0,
                **{f"p_{name}": 1.0 for name in
                   ("CO_dry_vol_pct", "H2_dry_vol_pct", "CO2_dry_vol_pct",
                    "H2_CO_ratio", "dry_syngas_flow")}}

    module._run_row = fake_run_row

    # 先造出 train 子缓存（模拟上一阶段已跑过训练段）
    module._prediction_rows_for_alpha(
        FakeRound54, [rows[0]], method="global_bounded_wgs_scalar",
        alpha_by_window={"w1": 0.5}, output_dir=tmp_path,
        cache_name="_global_grid_alpha_0.5.csv", timeout_s=1.0)

    full = module._selected_global_full(
        FakeRound54, rows, tmp_path, 1.0, 0.5)

    stored = set(full["cache_fingerprint"].astype(str))
    assert len(stored) == 1, f"full cache carries {len(stored)} fingerprints: {stored}"
    assert full["window_id"].nunique() == 2
