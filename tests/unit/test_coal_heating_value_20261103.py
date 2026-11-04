"""P1.7-A（2026-11-03）：热值口径与热解热模型的单元测试。

背景：数据合同的 ``heating_value_dry_kj_kg`` 是**干燥基低位发热量**，修改前被填进
``HHV_d``，``Hf_coal`` 公式又按高位扣了一次汽化潜热（扣两次）。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from model.gasifier_system import GasifierSystem, compute_hf_coal_j_kg  # noqa: E402
from model.input_contract import (  # noqa: E402
    coal_hhv_kj_kg,
    coal_lhv_kj_kg,
    resolve_pyrolysis_heat_model,
)
from model.material import SPECIES_NAMES  # noqa: E402
from model.physics import get_enthalpy_molar  # noqa: E402
from model.pyrolysis_service import PyrolysisService  # noqa: E402
from model.source_terms import PyrolysisSource  # noqa: E402

# P1 合同煤质（config/coal_assumption_v0.yaml），LHV_d = 27720 kJ/kg
P1_COAL = {
    "Cd": 73.01, "Hd": 4.324, "Od": 10.855, "Nd": 0.94, "Sd": 0.371,
    "Ad": 10.5, "FCd": 56.727, "Mt": 2.0,
}


def _pyrolysis_heat_kj_kg(coal_props: dict, T_pyro: float) -> float:
    """按 PyrolysisSource 的口径算热解热 [kJ/kg 干煤]（<0 吸热）。"""
    molar_yields, w_vol = PyrolysisService().calc_yields(coal_props)
    source = PyrolysisSource(
        np.zeros(len(SPECIES_NAMES)), 0.0, target_cell_idx=0, T_pyro=T_pyro,
        coal_mass_flow_kg_s=1.0, coal_props=coal_props,
    )
    h_vol = sum(
        float(molar_yields[i]) * float(get_enthalpy_molar(sp, T_pyro))
        for i, sp in enumerate(SPECIES_NAMES) if molar_yields[i] > 0.0
    )
    m_char = 1.0 - float(w_vol)
    h_coal = 1.0 * (float(coal_props["cp_char"]) * (T_pyro - 298.15) + float(coal_props["Hf_coal"]))
    h_char = m_char * float(coal_props["cp_char"]) * (T_pyro - 298.15)
    return (h_coal - h_char - h_vol) / 1000.0


def test_lhv_given_uses_lhv_directly():
    props = {**P1_COAL, "LHV_d": 27720.0, "cp_char": 1300.0}
    hf, model = compute_hf_coal_j_kg(props, {"T_in": 298.15})
    assert model == "from_heating_value"
    assert hf / 1000.0 == pytest.approx(-1421.1, abs=1.0)
    props["Hf_coal"] = hf
    assert _pyrolysis_heat_kj_kg(props, 298.15) == pytest.approx(-161.8, abs=2.0)


def test_hhv_only_keeps_legacy_behaviour():
    """只给 HHV_d 时保持修改前行为：按高位处理、扣 9·Hd·2442。"""
    props = {**P1_COAL, "HHV_d": 27720.0, "cp_char": 1300.0}
    hf, _ = compute_hf_coal_j_kg(props, {"T_in": 298.15})
    assert hf / 1000.0 == pytest.approx(-2371.4, abs=1.0)
    props["Hf_coal"] = hf
    assert _pyrolysis_heat_kj_kg(props, 298.15) == pytest.approx(-1112.1, abs=2.0)


def test_both_heating_values_is_rejected():
    props = {**P1_COAL, "LHV_d": 27720.0, "HHV_d": 28670.0}
    with pytest.raises(ValueError, match="同时提供"):
        coal_lhv_kj_kg(props)
    geom = {"L": 1.0, "D": 0.5}
    op = {"coal_flow": 1.0, "o2_flow": 0.5, "P": 3.8e6, "T_in": 500.0}
    with pytest.raises(ValueError, match="同时提供"):
        GasifierSystem(geom, props, op)


def test_hhv_is_derived_from_lhv_for_heat_loss_basis():
    props = {**P1_COAL, "LHV_d": 27720.0}
    hd = P1_COAL["Hd"] / 100.0
    assert coal_hhv_kj_kg(props) == pytest.approx(27720.0 + 9.0 * hd * 2442.0)
    assert coal_lhv_kj_kg({**P1_COAL, "HHV_d": 28670.0}) == pytest.approx(
        28670.0 - 9.0 * hd * 2442.0
    )


def test_mj_per_kg_inputs_are_normalised():
    assert coal_hhv_kj_kg({**P1_COAL, "HHV_d": 27.72}) == pytest.approx(27720.0)


def test_pyrolysis_heat_model_zero_gives_exactly_zero_and_ignores_heating_value():
    op = {"T_in": 500.0, "PyrolysisHeatModel": "zero"}
    hf_a, model = compute_hf_coal_j_kg({**P1_COAL, "LHV_d": 27720.0}, op)
    hf_b, _ = compute_hf_coal_j_kg({**P1_COAL, "HHV_d": 34000.0}, op)
    assert model == "zero"
    assert hf_a == pytest.approx(hf_b, rel=0.0, abs=1e-9)
    props = {**P1_COAL, "Hf_coal": hf_a, "cp_char": 1300.0}
    assert _pyrolysis_heat_kj_kg(props, 500.0) == pytest.approx(0.0, abs=1e-6)


def test_pyrolysis_heat_model_default_and_invalid():
    assert resolve_pyrolysis_heat_model({}) == "from_heating_value"
    with pytest.raises(ValueError, match="PyrolysisHeatModel"):
        resolve_pyrolysis_heat_model({"PyrolysisHeatModel": "noconsumption"})


def test_zero_model_overrides_given_hf_coal():
    geom = {"L": 1.0, "D": 0.5}
    op = {"coal_flow": 1.0, "o2_flow": 0.5, "P": 3.8e6, "T_in": 500.0,
          "PyrolysisHeatModel": "zero"}
    system = GasifierSystem(geom, {**P1_COAL, "LHV_d": 27720.0, "Hf_coal": -3.0e6}, op)
    assert system.hf_coal_source == "zero"
    assert system.coal_props["Hf_coal"] != pytest.approx(-3.0e6)
