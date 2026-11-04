"""P1.8-A（2026-11-04）：壁面热损模型开关与 C+H2O/C+CO2 指前因子倍数的最小回归。

只覆盖三件事：
1. ``HeatLossModel`` 的缺省/非法值解析；
2. Fortran 两区 hloss 式子的取值与符号（含 ``tg < tw`` 不 clamp）；
3. 缺省开关下锚点结果不受影响，且 JAX 路径遇到 ``fortran_wall`` / ``f_gas != 1`` 直接报错。
"""
from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from model.fortran_anchor_20261030 import anchor_cases, build_case_inputs, solve_fortran_case
from model.gasifier_system import GasifierSystem
from model.heat_loss import (
    combustion_zone_heat_loss_w,
    gasification_zone_heat_loss_w,
    resolve_heat_loss_model,
)
from model.p1_5_round_20261030 import paper_op_conds


def test_heat_loss_model_default_and_invalid():
    assert resolve_heat_loss_model({}) == "hhv_percent"
    assert resolve_heat_loss_model({"HeatLossModel": "FORTRAN_WALL"}) == "fortran_wall"
    with pytest.raises(ValueError):
        resolve_heat_loss_model({"HeatLossModel": "fortran"})


def test_combustion_zone_uses_thirty_percent_of_absolute_heat():
    assert combustion_zone_heat_loss_w(1.0e6) == pytest.approx(3.0e5, rel=1e-12)
    assert combustion_zone_heat_loss_w(-1.0e6) == pytest.approx(3.0e5, rel=1e-12)


def test_gasification_zone_wall_temperature_and_sign():
    out = gasification_zone_heat_loss_w(1500.0, z_outlet_m=1.65, dz_m=0.05)
    # tw = 2100 − 600·165/330 = 1800 K；tg < tw → hloss 为负且不 clamp
    assert out["tw_K"] == pytest.approx(1800.0, rel=1e-12)
    assert out["hloss_W"] < 0.0
    hand = 3.4e-3 * math.pi * 152.4 * (0.05 * 100.0) * (1500.0 - 1800.0) * 4.184
    assert out["hloss_W"] == pytest.approx(hand, rel=1e-12)


def test_gasification_zone_heat_loss_is_positive_when_gas_is_hotter():
    out = gasification_zone_heat_loss_w(2200.0, z_outlet_m=1.65, dz_m=0.05)
    assert out["hloss_W"] > 0.0


def _case(name="texaco i-1"):
    cases, geo = anchor_cases()
    case = next(c for c in cases if c["case_name"] == name)
    return case, geo


def test_default_heat_loss_model_is_bit_identical():
    case, geo = _case()
    a = solve_fortran_case(case, n_cells=8, mode="F", geometry=geo)
    b = solve_fortran_case(
        case, n_cells=8, mode="F", geometry=geo, switches={"HeatLossModel": "hhv_percent"}
    )
    for key in ("T_gas_K", "X_C_pct", "CO_vol_pct", "H2_vol_pct", "CO2_vol_pct"):
        assert a[key] == b[key]


def test_jax_path_rejects_fortran_wall_and_rate_factor():
    case, geo = _case()
    built = build_case_inputs(case, geometry=geo)
    for extra in ({"HeatLossModel": "fortran_wall"}, {"GasificationRateFactor": 0.5}):
        system = GasifierSystem(built["geometry"], built["coal_props"],
                                paper_op_conds(built, extra=extra))
        with pytest.raises(ValueError):
            system.solve(N_cells=4, solver_method="jax_jit", jax_warmup=False)
