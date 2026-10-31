from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from model.p1_response_20260930 import quality_label, same_sign


def test_interpretability_requires_both_solves():
    assert quality_label(0, 0, 1.0e-6, 1.0e-6) == "interpretable_solved"
    assert quality_label(1, 0, 1.0e-6, 1.0e-6) == "not_interpretable_due_to_residual"
    assert quality_label(0, 0, 0.02, 1.0e-6) == "not_interpretable_due_to_residual"


def test_sign_comparison_uses_strict_product():
    assert same_sign(0.2, 0.1) is True
    assert same_sign(-0.2, -0.1) is True
    assert same_sign(0.2, -0.1) is False
    assert same_sign(0.0, 0.1) is False
