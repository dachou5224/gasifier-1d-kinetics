"""P1.12-A 审查：整炉总量热损的权重残差必须用未松弛量。"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from model.heat_loss import (  # noqa: E402
    heat_loss_weight_relative_residual,
    temperature_weight,
)


def test_temperature_weight_clips():
    assert temperature_weight(1800.0) == pytest.approx(1.0)
    assert temperature_weight(0.0) == pytest.approx(0.1)
    assert temperature_weight(1800.0 * 2.0) == pytest.approx(5.0)


def test_unrelaxed_residual_is_independent_of_under_relaxation():
    w = np.array([1.0, 2.0, 4.0])
    w_new = w * 1.0005  # 相对不一致 5e-4
    raw = heat_loss_weight_relative_residual(w, w_new)
    assert raw == pytest.approx(5e-4, rel=1e-12)

    alpha = 0.1
    w_step = w + alpha * (w_new - w)
    relaxed = heat_loss_weight_relative_residual(w, w_step)
    assert relaxed == pytest.approx(5e-5, rel=1e-12)
    # 松弛步会通过 1e-4 容差，未松弛残差不会——收敛必须测后者。
    assert relaxed < 1e-4 < raw
