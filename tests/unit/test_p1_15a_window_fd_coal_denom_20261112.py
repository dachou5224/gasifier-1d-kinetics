"""P1.15-A：煤量扰动的单位煤量干气斜率须用该快照实际煤量作分母。"""
from __future__ import annotations

import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import generate_p1_15a_20261108 as p115a  # noqa: E402


def _proportional_snap(coal: float, k: float = 1.0) -> dict:
    """干气摩尔流严格正比于煤量；物种按干气份额拆分。"""
    dry = k * coal
    return {
        "ok": True,
        "coal_dry_kg_s": float(coal),
        "dry_total_mol_s": float(dry),
        "F_CO": 0.55 * dry,
        "F_CO2": 0.15 * dry,
        "F_H2": 0.25 * dry,
        "F_CH4": 0.04 * dry,
        "F_H2S": 0.01 * dry,
        "F_H2O": 0.10 * dry,
        "X_C_pp": 95.0,
        "X_C_at_ceiling": False,
        "wsr_C_CO2_LimitedByCO2": False,
        "wsr_raw_C_CO2": 1.0,
        "wsr_raw_C_H2O": 1.0,
        "limit_len_frac_C_CO2_LimitedByCO2": 0.0,
    }


def test_per3_dry_per_coal_zero_when_dry_proportional_to_coal():
    coal_nom = 10.0
    snaps = {
        "nom": _proportional_snap(coal_nom),
        "o2_p": _proportional_snap(coal_nom),
        "o2_m": _proportional_snap(coal_nom),
        "coal_p": _proportional_snap(coal_nom * 1.03),
        "coal_m": _proportional_snap(coal_nom * 0.97),
    }
    cand = {"kind": "C0", "xi": 0.0}
    fd = p115a.window_fd(snaps, cand, coal_nom)
    got = fd["per3_dry_per_coal_mol_kg__ln_coal"]
    assert math.isfinite(got)
    assert abs(got) < 1e-9


def test_old_nominal_denominator_gives_ln103_when_k_is_one():
    """对照：若仍用名义煤量作分母，k=1 时每+3% ≈ ln(1.03)。"""
    coal_nom = 10.0
    snaps = {
        "nom": _proportional_snap(coal_nom),
        "o2_p": _proportional_snap(coal_nom),
        "o2_m": _proportional_snap(coal_nom),
        "coal_p": _proportional_snap(coal_nom * 1.03),
        "coal_m": _proportional_snap(coal_nom * 0.97),
    }
    cand = {"kind": "C0", "xi": 0.0}
    ys = {
        p: p115a._y_for_post(snaps[p], cand, coal_nom)
        for p in p115a.PERTS
    }
    _sl, per3 = p115a.fd_from_ys(ys["coal_p"], ys["coal_m"], "dry_per_coal_mol_kg")
    assert abs(per3 - math.log(1.03)) < 1e-4
