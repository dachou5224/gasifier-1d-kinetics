"""P1.15-A：sign_stable 须同时要求区间有符号且点估计同号。"""
from __future__ import annotations

import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import generate_p1_15a_20261108 as p115a  # noqa: E402


def _pair(phase, obs, inp, w_per3, n_per3, w_signed, n_signed) -> pd.DataFrame:
    rows = []
    for model, per3, signed in (
        ("with_carrier", w_per3, w_signed),
        ("no_carrier", n_per3, n_signed),
    ):
        rows.append({
            "phase": phase, "obs": obs, "input": inp, "model": model,
            "per3": per3, "signed": signed,
        })
    return pd.DataFrame(rows)


def test_opposite_signed_coefficients_are_not_direction_stable():
    df = _pair("phase2", "dry_per_coal_mol_kg", "ln_O2", 1.0, -1.0, True, True)
    out = p115a._mark_sign_stable(df)
    assert (out["signed_agree"] == "Y").all()
    assert (out["dir_agree"] == "N").all()
    assert (out["sign_stable"] == "N").all()


def test_both_unsigned_is_not_direction_stable():
    df = _pair("phase3", "CO_vol_pct", "ln_O2", 0.2, 0.1, False, False)
    out = p115a._mark_sign_stable(df)
    assert (out["signed_agree"] == "Y").all()
    assert (out["dir_agree"] == "Y").all()
    assert (out["sign_stable"] == "N").all()


def test_signed_status_mismatch_same_point_sign():
    df = _pair("phase2", "dry_per_coal_mol_kg", "ln_O2", 1.676, 1.565, True, False)
    out = p115a._mark_sign_stable(df)
    assert (out["signed_agree"] == "N").all()
    assert (out["dir_agree"] == "Y").all()
    assert (out["sign_stable"] == "N").all()


def test_both_signed_same_direction_is_stable():
    df = _pair("phase1", "CO_vol_pct", "ln_coal", -0.5, -0.4, True, True)
    out = p115a._mark_sign_stable(df)
    assert (out["signed_agree"] == "Y").all()
    assert (out["dir_agree"] == "Y").all()
    assert (out["sign_stable"] == "Y").all()
