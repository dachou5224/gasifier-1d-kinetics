"""06A 受控设计与可分辨性检查 —— 产物一致性测试。

只读 `outputs/diagnostics/operating-response-comparison/06A/` 的产物，
不重新求解反应器（求解由 `scripts/p06a_controlled_design_and_separation.py` 完成）。

覆盖的关键断言
--------------
1. 产品名单与 sha256 清单一致，无过时产物。
2. 历史路径与显式 0.2 逐位相同（C0 缺省值确实是 0.2）。
3. 另一求解入口在 alpha_wgs=1.0 下与 run_case(1.0) 不等价。
4. 组成水平杠杆与响应杠杆均低于最乐观预设分辨率。
5. C 与 R 的 argmin 相同，响应门槛不改变动作（消融为空）。
6. 放行判定为 STOP_TO_07，且理由码与数据一致。
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
    "leverage_scan.csv",
    "response_leverage.csv",
    "separation_check.csv",
    "ablation_trace.csv",
    "scenario_design.csv",
    "controlled_protocol_draft.yaml",
    "expected_outputs.md",
    "commands.md",
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


def test_historical_path_equals_explicit_c0() -> None:
    """历史路径（不写键）必须与显式 0.2 逐位相同。"""
    rows = _read("baseline_equivalence.csv")
    for r in rows:
        if r["comparison"] in ("historical_no_key", "explicit_0.2"):
            assert float(r["dCO"]) == 0.0, (r["phase"], r["comparison"], r["dCO"])
            assert float(r["dT_K"]) == 0.0, (r["phase"], r["comparison"])


def test_explicit_1_0_differs_from_c0() -> None:
    """显式 1.0 必须与 C0 不同，否则尺度区分无意义。"""
    rows = [r for r in _read("baseline_equivalence.csv") if r["comparison"] == "explicit_1.0"]
    assert rows
    for r in rows:
        assert abs(float(r["dCO"])) > 1e-6, (r["phase"], r["dCO"])
    # 1.0 相对 C0 是 5 倍，不得写成等价
    assert all(abs(float(r["rel_to_C0"]) - 5.0) < 1e-9 for r in rows)


def test_alt_entry_not_equivalent(proto: dict) -> None:
    """另一求解入口不得用于验证历史 C0。"""
    rows = [
        r
        for r in _read("baseline_equivalence.csv")
        if r["comparison"] == "alt_entry_solve_model_case_alpha_1.0"
    ]
    assert rows
    assert all(abs(float(r["dCO"])) > 1e-6 for r in rows)
    text = json.dumps(proto, ensure_ascii=False)
    assert "不等价" in text or "not_equivalent" in text


def test_level_leverage_below_resolution() -> None:
    for r in _read("separation_check.csv"):
        if r["metric"] == "composition_level_span" and r["preset_key"] == "optimistic":
            assert r["separable"] == "False", (r["phase"], r["ratio_sigma_over_need"])
            assert float(r["ratio_sigma_over_need"]) > 1.0


def test_response_leverage_below_resolution() -> None:
    for r in _read("separation_check.csv"):
        if r["metric"] == "response_span" and r["preset_key"] == "optimistic":
            assert r["separable"] == "False", (r["phase"], r["ratio_sigma_over_need"])


def test_all_presets_agree(proto: dict) -> None:
    """三档预设 σ_y 下的结论必须一致，避免结论依赖单档假设。"""
    for metric in ("composition_level_span", "response_span"):
        for phase in ("P1", "P3", "P4"):
            flags = {
                r["separable"]
                for r in _read("separation_check.csv")
                if r["metric"] == metric and r["phase"] == phase
            }
            assert flags == {"False"}, (metric, phase, flags)


def test_required_sigma_y_is_physically_implausible() -> None:
    """要被分辨所需的 σ_y 必须远小于任何合理分析仪精度。

    判据取 0.05 pp（= 500 ppm）。工业在线合成气分析仪（NDIR 测 CO/CO2）
    的典型重复性量级是 0.1 vol%（= 1000 ppm）以上，因此 0.05 pp 已属
    乐观上限；实测最紧的一档是 P4 组成水平 0.0202 pp（≈202 ppm），
    仍比该上限紧一倍以上。
    """
    for r in _read("separation_check.csv"):
        if r["preset_key"] == "optimistic":
            need = float(r["required_sigma_y_pp"])
            assert need < 0.05, (r["phase"], r["metric"], need)


def test_c_and_r_argmin_coincide(proto: dict) -> None:
    """组成与响应的 argmin 相同 → 响应门槛不改变动作。"""
    f = proto["separation_findings"]
    for phase in ("P3", "P4"):
        assert f["c_argmin_theta"][phase] == f["r_argmin_theta"][phase], phase
        assert f["gate_changes_action"][phase] is False, phase


def test_all_candidates_pass_response_gate() -> None:
    """R_dev 全低于门槛 → G 从不拒绝 → 消融为空。"""
    rows = _read("ablation_trace.csv")
    assert rows
    assert all(r["R_gate_pass"] == "True" for r in rows), "应全部通过门槛，说明门槛无判别力"


def test_saturation_not_amplitude_limited(proto: dict) -> None:
    """θ 由 2.0 增至 10.0 的增益必须远小于 [0.02,2.0] 的跨度。"""
    f = proto["separation_findings"]
    for phase in ("P1", "P3", "P4"):
        lev = float(f["composition_level_span_pp"][phase])
        gain = float(f["saturation_gain_2.0_to_10.0_pp"][phase])
        assert gain < 0.2 * lev, (phase, gain, lev)


def test_verdict_is_stop_to_07(proto: dict) -> None:
    assert proto["verdict"]["exit"] == "STOP_TO_07"
    codes = set(proto["verdict"]["reason_codes"])
    assert codes == {
        "signal_below_preset_resolution",
        "single_scalar_family_degeneracy",
        "saturation_not_amplitude_limited",
    }


def test_no_plant_outlet_used_as_truth(proto: dict) -> None:
    """06A 卡面：生成端不得用历史装置出口拟合值当真值。"""
    note = proto["repo"]["note"]
    assert "未读" in note or "not" in note.lower()


def test_scenarios_both_blocked() -> None:
    rows = _read("scenario_design.csv")
    assert len(rows) == 2
    for r in rows:
        assert r["separable"] == "False", r["scenario"]
        assert "不可分辨" in r["verdict"] or "前提不成立" in r["verdict"], r["scenario"]


def test_composition_match_tolerance_exceeds_c_dev_spread(proto: dict) -> None:
    """场景二的前提（两候选出口拟合相似）在整个网格上恒成立，故失去判别意义。"""
    tol = float(proto["thresholds"]["composition_match_tolerance_pp"])
    spread = float(proto["separation_findings"]["c_dev_spread_pp"]["P3"])
    assert spread < tol, (spread, tol)
