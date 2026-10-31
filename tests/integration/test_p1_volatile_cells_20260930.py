from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from model.p1_volatile_cells_20260930 import alignment_ok, dry_mol_s_to_nm3_h, first_sign_change


def test_dry_conversion_matches_previous_nm3_definition():
    assert abs(dry_mol_s_to_nm3_h(1000.0) - 22.414 * 3600.0) < 1.0e-9


def test_alignment_tolerances():
    errors = {
        "base_dry_nm3_h": 0.05,
        "plus_dry_nm3_h": 0.1,
        "WGS": 1.0e-4,
        "volatile_oxidation": 0.0,
        "char_oxidation": 0.0,
        "char_gasification": 0.0,
    }
    assert alignment_ok(errors) is True
    errors["plus_dry_nm3_h"] = 0.11
    assert alignment_ok(errors) is False
    errors["plus_dry_nm3_h"] = 0.1
    errors["char_oxidation"] = 1.1e-4
    assert alignment_ok(errors) is False


def test_first_sign_change_index():
    assert first_sign_change([-1.0, -2.0, -0.5]) is None
    assert first_sign_change([-1.0, 0.0, 0.4]) == 2
    assert first_sign_change([0.2, -0.1]) == 1
