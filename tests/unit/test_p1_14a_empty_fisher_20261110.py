"""P1.14-A：空 Fisher / 空 jac 不得摔；失败谓词与 rec['ok'] 一致。"""
from __future__ import annotations

import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import generate_p1_14a_20261107 as p114a  # noqa: E402


def test_subsets_split_wall_proxies():
    assert "S3" not in p114a.SUBSETS
    assert "S4" not in p114a.SUBSETS
    assert set(p114a.SUBSETS) >= {
        "S0", "S1", "S2", "S3_conv", "S3_rad", "S4_conv", "S4_rad", "S4_no_wall",
    }
    assert p114a._n_wall_in(p114a.SUBSETS["S3_conv"]) == 1
    assert p114a._n_wall_in(p114a.SUBSETS["S3_rad"]) == 1
    assert p114a._n_wall_in(p114a.SUBSETS["S4_no_wall"]) == 0
    assert p114a._add_lnU(p114a.SUBSETS["S3_conv"])
    assert p114a._add_lnU(p114a.SUBSETS["S4_rad"])
    assert not p114a._add_lnU(p114a.SUBSETS["S0"])
    assert not p114a._add_lnU(p114a.SUBSETS["S4_no_wall"])


def test_build_readings_empty_fisher_no_columns():
    fisher = pd.DataFrame()
    baseline = pd.DataFrame({
        "ok_theta": [True, False],
        "ok_jacobian": [True, True],
    })
    rd = p114a.build_readings(fisher, baseline, pd.DataFrame())
    assert "reading" in rd.columns
    assert "coverage" in set(rd["reading"])
    assert "Q5" in set(rd["reading"])
    assert "Q1" not in set(rd["reading"])
    assert "Q2" not in set(rd["reading"])
    assert "Q3" not in set(rd["reading"])
    failed = rd[rd["metric"] == "n_failed"].iloc[0]["value"]
    assert failed == 1.0


def test_empty_fisher_out_has_columns():
    rec = p114a._empty_fisher_out("S0", n_windows=0, note="no windows in this set")
    assert rec["n_success"] == 0
    assert rec["subset"] == "S0"
    assert "phase" not in rec
    fisher = pd.DataFrame([rec])
    rd = p114a.build_readings(fisher, pd.DataFrame(), pd.DataFrame())
    # 缺 phase/window_set → skip Q1–Q3，仍有 Q5
    assert "Q5" in set(rd["reading"])
    assert "Q1" not in set(rd["reading"])


def test_fisher_from_windows_empty_map():
    rec = p114a.fisher_from_windows({}, ["w1", "w2"], "S0")
    assert rec["n_success"] == 0
    assert rec["n_windows"] == 2
    assert rec["ok"] is False
    assert "sigma_ln_f_gas" in rec


def test_n_failed_matches_ok_predicate():
    base = pd.DataFrame({
        "ok_theta": [True, True, False, True],
        "ok_jacobian": [True, False, True, True],
    })
    # rec['ok'] = ok_theta and ok_jacobian → 失败 2 行
    assert p114a._n_failed_windows(base) == 2
    assert p114a._n_failed_windows(pd.DataFrame()) == 0
