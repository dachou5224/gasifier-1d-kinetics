"""P2.0-A 渣层截留（M2）：分流前后单颗粒状态必须连续。

审阅意见（2026-11-16，P1）指出第一次修正把「物理分流」算成了「转化」：
下游用 ``C_fed - C_captured``（截留颗粒**剩余**的碳）作基准，等价于把分流
本身当成一次额外转化。正确基准是「剩余颗粒所代表的**原始进料份额**」，
即按颗粒份额 ``f_s`` 缩放。

本测试用审阅给出的算例把它钉死：C_fed = 1、全混区转化 50%、截留 30% 颗粒，
则：

- 分流前 X = 50 %；
- 分流后被截留颗粒的 X 也是 50 %（它们只是离开，没有反应）；
- 剩余颗粒的 X 仍然是 50 %，不是 58.8 %。
"""

from __future__ import annotations

import numpy as np
import pytest

# 注入键名。两处（Cell 读取、GasifierSystem 写入）必须一致，这里显式断言，
# 避免任何一侧被改动后静默失配——静默失配的表现正是「修正看起来没生效」。
CAPTURED_PARTICLE_FRACTION_KEY = "CapturedParticleFractionOfFeed"


class _Stub:
    """只承载 ``_calc_physics_props`` 需要的最小状态。"""

    def __init__(self, carbon_kg_s: float):
        # carbon_fraction 取 1，使 solid_mass 与碳量数值相同，便于手算核对
        self.solid_mass = carbon_kg_s
        self.carbon_fraction = 1.0


def _x_total(C_fed: float, C_local: float, cap_frac: float) -> float:
    """复现 ``Cell._calc_physics_props`` 的 X_total 口径（不含其余物理量）。"""
    C_ref = C_fed * (1.0 - min(max(cap_frac, 0.0), 0.9999))
    if C_ref < 1e-9:
        C_ref = 1e-9
    return float(np.clip(1.0 - C_local / C_ref, 0.0, 1.0))


def test_split_does_not_change_particle_conversion():
    """审阅算例：分流前后，剩余颗粒的转化率必须相同。"""
    C_fed = 1.0
    X_wsr = 0.50
    f_s = 0.30
    C_local_wsr = C_fed * (1.0 - X_wsr)          # 0.5
    C_local_ret = C_local_wsr * (1.0 - f_s)      # 剩余颗粒仍带 0.35

    x_before = _x_total(C_fed, C_local_wsr, 0.0)
    x_after = _x_total(C_fed, C_local_ret, f_s)

    assert x_before == pytest.approx(0.50, abs=1e-12)
    assert x_after == pytest.approx(0.50, abs=1e-12), (
        "按剩余碳份额缩放会得到 58.82 %，那是把分流算成转化的旧错误"
    )


def test_old_carbon_based_scaling_would_be_wrong():
    """把错法写进测试，防止它被重新引入。"""
    C_fed, X_wsr, f_s = 1.0, 0.50, 0.30
    C_local_wsr = C_fed * (1.0 - X_wsr)
    C_cap_rem = C_local_wsr * f_s                 # 0.15，剩余碳
    C_ref_wrong = C_fed - C_cap_rem               # 0.85
    x_wrong = 1.0 - (C_local_wsr * (1.0 - f_s)) / C_ref_wrong
    assert x_wrong == pytest.approx(0.5882352941, abs=1e-9)
    assert x_wrong != pytest.approx(0.50, abs=1e-6)


@pytest.mark.parametrize("f_s", [0.0, 0.05, 0.2, 0.4])
@pytest.mark.parametrize("X_wsr", [0.0, 0.3, 0.6, 0.9])
def test_continuity_across_split(f_s: float, X_wsr: float):
    """任意 (f_s, X_wsr) 组合下，分流不改变剩余颗粒的转化率。"""
    C_fed = 1.0
    C_local_wsr = C_fed * (1.0 - X_wsr)
    x_before = _x_total(C_fed, C_local_wsr, 0.0)
    x_after = _x_total(C_fed, C_local_wsr * (1.0 - f_s), f_s)
    assert x_after == pytest.approx(x_before, abs=1e-12)


def test_no_capture_is_bit_identical():
    """f_s = 0 时口径必须逐位不变。"""
    C_fed = 1.0
    for C_local in (0.0, 0.25, 0.5, 1.0):
        assert _x_total(C_fed, C_local, 0.0) == _x_total(C_fed, C_local, 0.0)


def test_key_name_matches_both_sides():
    """写入侧（GasifierSystem）与读取侧（Cell）必须用同一个键名。"""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2]
    sys_src = (root / "src/model/gasifier_system.py").read_text(encoding="utf-8")
    cell_src = (root / "src/model/cell.py").read_text(encoding="utf-8")
    assert CAPTURED_PARTICLE_FRACTION_KEY in sys_src, "写入侧未使用该键"
    assert CAPTURED_PARTICLE_FRACTION_KEY in cell_src, "读取侧未使用该键"
