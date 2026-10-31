from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from model.cell import Cell


def test_negative_wgs_stays_within_remaining_hydrogen():
    avail_h2 = 100.0
    r_h2_ox = 99.0
    r_c_h2, r_wgs = Cell.limit_h2_consumers_after_oxidation(avail_h2, r_h2_ox, 40.0, -50.0)
    assert r_wgs < 0.0
    assert r_h2_ox + 2.0 * r_c_h2 + abs(r_wgs) <= avail_h2 + 1.0e-8


def test_positive_wgs_is_not_capped():
    r_c_h2, r_wgs = Cell.limit_h2_consumers_after_oxidation(100.0, 10.0, 1.0, 25.0)
    assert r_wgs == 25.0
    assert r_c_h2 == 1.0
