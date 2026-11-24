"""06A 受控设计与可分辨性检查 —— 产物一致性测试（rev 2）。

只读 `outputs/diagnostics/operating-response-comparison/06A/` 的产物，
不重新求解反应器（求解由 `scripts/p06a_controlled_design_and_separation.py` 完成）。

本测试刻意**不**固化 rev 1 被撤回的普遍结论（「单标量族必然退化」
「任何门槛都不可能改变动作」「WGS 已近平衡」「S2 已被否定」）。
它只验证：底座正确、误差传播分开、计算可复现、以及范围受限的观察。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

OUT = Path("outputs/diagnostics/operating-response-comparison/06A")

PRODUCTS = (
    "baseline_equivalence.csv",
    "c0_reproduction.csv",
    "leverage_scan.csv",
    "response_leverage.csv",
    "separation_check.csv",
    "ablation_trace.csv",
    "scenario_design.csv",
    "controlled_protocol_draft.yaml",
    "expected_outputs.md",
    "commands.md",
    "limitations.md",
)

C0_ABS = 0.2
SIGMA_Y_OPTIMISTIC = 0.10
K_SIGMA = 2.0


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def _read(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def proto() -> dict:
    return json.loads((OUT / "controlled_protocol_draft.yaml").read_text(encoding="utf-8"))


def test_products_exist_and_manifest_matches() -> None:
    for name in PRODUCTS:
        assert (OUT / name).exists(), name
    lines = (OUT / "sha256.txt").read_text(encoding="utf-8").strip().splitlines()
    listed = [ln.split("  ", 1)[1] for ln in lines]
    assert listed == list(PRODUCTS), "清单须与产品名单逐项一致"
    for ln in lines:
        digest, name = ln.split("  ", 1)
        assert _sha(OUT / name) == digest, name
    on_disk = {p.name for p in OUT.iterdir() if p.is_file() and p.name != "sha256.txt"}
    assert on_disk == set(PRODUCTS), f"目录含名单外文件: {on_disk - set(PRODUCTS)}"


# ---------------- P1：底座正确性 ----------------


def test_reference_point_uses_wsr_plus_pfr() -> None:
    """审查 P1：不得走 uniform_full_length 默认网格。"""
    rows = _read("c0_reproduction.csv")
    assert rows
    for r in rows:
        assert r["mesh_family"] == "wsr_plus_pfr", r["window_id"]
        assert int(r["n_cells"]) == 182
        assert r["status"] == "completed"


def test_historical_c0_reproduced_exactly() -> None:
    """对 baseline_142.csv 必须逐位复现，而不是「接近」。"""
    rows = _read("c0_reproduction.csv")
    assert len(rows) >= 6, "至少覆盖三个阶段各两窗"
    for r in rows:
        assert float(r["dCO"]) == 0.0, (r["window_id"], r["dCO"])
        assert float(r["dH2"]) == 0.0, (r["window_id"], r["dH2"])
        assert float(r["dCO2"]) == 0.0, (r["window_id"], r["dCO2"])
        assert float(r["dT_K"]) == 0.0, (r["window_id"], r["dT_K"])


def test_proto_records_reproduction_verdict(proto: dict) -> None:
    rep = proto["reference_point"]["reproduction"]
    assert rep["max_abs_dCO_over_sampled_windows"] == 0.0
    assert rep["verdict"] == "byte-identical"
    assert rep["reference"].endswith("baseline_142.csv")


def test_proto_records_frozen_config(proto: dict) -> None:
    cfg = proto["reference_point"]["frozen_config"]
    assert cfg["MeshFamily"] == "wsr_plus_pfr"
    assert cfg["WSRLength_m"] == 1.0
    assert cfg["PFRCellLength_m"] == 0.0515
    assert cfg["HeatLossPercentTotal"] == 1.0
    assert cfg["n_cells"] == 182
    assert "measured" in cfg["carrier_gas_flow"] or "co2_carrier_measured" in cfg["carrier_gas_flow"]
    assert "explicit dry" in cfg["coal_flow"] or "coal_dry_recon" in cfg["coal_flow"]


def test_historical_c0_equals_explicit_0_2() -> None:
    rows = [
        r for r in _read("baseline_equivalence.csv")
        if r["comparison"] == "unset_vs_explicit_0.2"
    ]
    assert rows
    for r in rows:
        assert float(r["dCO"]) == 0.0, r["phase"]
        assert float(r["dT_K"]) == 0.0, r["phase"]


def test_explicit_1_0_differs_from_c0() -> None:
    rows = [
        r for r in _read("baseline_equivalence.csv")
        if r["comparison"] == "unset_vs_explicit_1.0"
    ]
    assert rows
    for r in rows:
        assert abs(float(r["dCO"])) > 1e-9, r["phase"]
        assert abs(float(r["rel_to_C0"]) - 5.0) < 1e-9, r["phase"]


def test_alt_entry_documented_as_not_substitutable(proto: dict) -> None:
    """另一入口在正确底座上差异更大，不得替代 run_case 验证历史 C0。"""
    rows = [
        r for r in _read("baseline_equivalence.csv")
        if r["comparison"].startswith("alt_entry")
    ]
    assert rows
    assert all(abs(float(r["dCO"])) > 1e-6 for r in rows)
    text = json.dumps(proto, ensure_ascii=False)
    assert "不得直接替代" in text or "not_substitutable" in text or "仍需核对" in text


# ---------------- P2：误差传播分开 ----------------


def test_composition_uses_sigma_y_not_sigma_r() -> None:
    """审查 P2：组成水平不得套用 σ_r（含 ln(1.05)/Δln u）。"""
    rows = _read("separation_check.csv")
    comp = [r for r in rows if r["metric"] == "composition_level_span"]
    assert comp
    for r in comp:
        assert r["estimator_error_model"] == "single_observation_vs_deterministic_model"
        # 组成：error_sigma 应恰等于预设 σ_y
        assert abs(float(r["error_sigma"]) - float(r["sigma_y_preset_pp"])) < 1e-12, r["phase"]
        # required_sigma_y = signal / k
        assert abs(
            float(r["required_sigma_y_pp"]) - float(r["signal"]) / K_SIGMA
        ) < 1e-9, r["phase"]


def test_response_uses_sigma_r_formula() -> None:
    rows = [r for r in _read("separation_check.csv") if r["metric"] == "response_span"]
    assert rows
    for r in rows:
        sy = float(r["required_sigma_y_pp"])
        sr = float(r["required_signal_per_sigma"])
        assert sy > 0 and sr > 0
        # need_r / sigma_y_required = ln(1.05)*sqrt(2)/ln u
        # error_sigma / sigma_y_preset 应为同一比值；CSV 里 required_sigma_y 只存
        # 12 位小数，故用 1e-5 的相对容差而非 1e-9。
        assert math.isclose(
            sr / sy,
            float(r["error_sigma"]) / float(r["sigma_y_preset_pp"]),
            rel_tol=1e-5,
        ), r["phase"]


def test_required_sigma_y_matches_signal_over_k() -> None:
    """rev 1 把组成倍数算成 7.06/7.95/4.94；rev 2 必须是 signal/2。"""
    for r in _read("separation_check.csv"):
        if r["preset_key"] != "optimistic":
            continue
        if r["metric"] == "composition_level_span":
            want = float(r["signal"]) / K_SIGMA
            assert abs(float(r["required_sigma_y_pp"]) - want) < 1e-9, r["phase"]


def test_ratio_is_preset_over_required() -> None:
    for r in _read("separation_check.csv"):
        ratio = float(r["ratio_preset_over_required"])
        sigma = float(r["error_sigma"])
        need = float(r["required_signal_per_sigma"])
        assert abs(ratio - sigma / need) < 1e-6, (r["phase"], r["metric"])


def test_preset_marked_as_assumption(proto: dict) -> None:
    t = proto["thresholds"]
    assert t["preset_status"] == "DESIGN_ASSUMPTION_NOT_CALIBRATED"
    assert "量程" in t["preset_basis"] or "range" in t["preset_basis"].lower()


# ---------------- P3：撤回普遍结论 ----------------


def test_retracted_claims_listed(proto: dict) -> None:
    retracted = proto["verdict"]["retracted_from_rev1"]
    joined = " ".join(retracted)
    assert "degeneracy" in joined
    assert "any_threshold" in joined
    assert "near_equilibrium" in joined
    assert "S2" in joined


def test_s1_scope_is_limited(proto: dict) -> None:
    s1 = proto["s1_ablation"]
    assert "仅限本实现校验" in s1["scope"]
    assert any("必然" in x or "不可能" in x for x in s1["not_claimed"])
    assert "非单调" in s1["counter_evidence"] or "并非单调" in s1["counter_evidence"]


def test_r_dev_not_claimed_monotone(proto: dict) -> None:
    """P3 的 R_dev 在本网格内确实不单调，必须保留为反例。"""
    assert proto["s1_ablation"]["per_phase"]["P3"]["r_dev_monotone_in_grid"] is False


def test_no_universal_degeneracy_in_limitations() -> None:
    text = (OUT / "limitations.md").read_text(encoding="utf-8")
    assert "单标量族必然排序退化" in text, "须显式登记为已撤回"
    assert "任何门槛都不可能改变动作" in text


def test_analysis_仪_precision_claim_removed(proto: dict) -> None:
    """不得声称「任何合理分析仪」都达不到。"""
    text = (OUT / "limitations.md").read_text(encoding="utf-8")
    assert "任何合理分析仪" in text
    assert "预设" in text


# ---------------- P4：S2 实际注入 ----------------


def test_s2_actually_injects_boundary_deviation() -> None:
    rows = _read("scenario_design.csv")
    detail = [r for r in rows if r["theta_cand_abs"] != "SUMMARY"]
    assert detail, "S2 必须有逐候选明细，不能只继承 S1"
    for r in detail:
        assert float(r["carrier_bias_rel"]) > 0
        assert r["r_deviation_injected"] != "", "必须实际注入偏差"
        assert r["r_candidate"] != ""
        assert r["response_gap"] != ""


def test_s2_registers_both_composition_and_response() -> None:
    """组成近似与响应分离须分别登记，不得只报其一。"""
    for r in _read("scenario_design.csv"):
        if r["theta_cand_abs"] != "SUMMARY":
            assert r["composition_matches"] in ("True", "False")
            assert r["response_separable"] in ("True", "False")


def test_s2_verdict_recorded(proto: dict) -> None:
    v = proto["s2_boundary_deviation"]["verdict"]
    assert set(v) == {"P3", "P4"}
    for phase, verdict in v.items():
        assert verdict.startswith(("NOT_SEPARABLE", "SEPARABLE", "PRECONDITION", "UNCLASSIFIED")), (
            phase,
            verdict,
        )


# ---------------- 判定与范围 ----------------


def test_verdict_exit_is_consistent(proto: dict) -> None:
    """STOP 的依据必须只引用本轮实际验证过的三项，不外推。"""
    assert proto["verdict"]["exit"] in ("STOP_TO_07", "RUN_ONCE")
    basis = proto["verdict"]["basis"]
    assert "复现" in basis or "reproduc" in basis.lower()
    assert "2σ" in basis or "2 sigma" in basis.lower()
    if proto["verdict"]["exit"] == "STOP_TO_07":
        assert "S2" in basis


def test_no_plant_outlet_used_as_truth(proto: dict) -> None:
    iso = proto["isolation"]
    assert "真值" in iso["generation_side"]
    assert "锁定后" in iso["scoring_side"]


def test_saturation_fact_recorded_without_mechanism_claim(proto: dict) -> None:
    """保留数值事实，但不得据此断言近平衡机制。"""
    f = proto["separation_findings"]
    assert set(f["saturation_gain_2.0_to_10.0_pp"]) == {"P1", "P3", "P4"}
    joined = " ".join(proto["verdict"]["retracted_from_rev1"])
    assert "near_equilibrium" in joined
