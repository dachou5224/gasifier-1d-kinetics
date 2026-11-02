#!/usr/bin/env python3
"""Wen Fortran 与 Python 的对照：正确煤质 + Fortran 自身几何，模式 M 与模式 F。

前置：`run_wen_fortran_texaco_i3_i6_20261031.py` 已生成 `fortran_outlet.csv`。
本脚本把 Fortran 出口值与 Python 的四种设置并列，用于判定 3400 K 的来源。
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
logging.basicConfig(level=logging.ERROR)

from model.cell import Cell  # noqa: E402
from model.gasifier_system import GasifierSystem  # noqa: E402

OUT = ROOT / "outputs" / "diagnostics" / "round-wen-fortran-texaco-20261031"

# H-coal residue Illinois No.6，干基 wt%（原始论文 JSON）
WEN_COAL = {"C": 74.05, "H": 6.25, "O": 1.32, "S": 1.77, "N": 0.71, "Ash": 15.53}
WEN_MT_PCT = 0.2

# Fortran 源程序自身几何：height = 330 cm（行 32），dt = 152.4 cm
FORTRAN_GEOM = {"L": 3.3, "D": 1.524}
PAPER_GEOM = {"L": 6.096, "D": 1.524}
JSON_GEOM = {"L": 6.0, "D": 2.0}

CASES = {
    "Texaco_I-1": {"tfcoal": 76.66, "foxy": 0.866, "fsteam": 0.241, "ta": 505.22},
    "Texaco_I-3": {"tfcoal": 82.202, "foxy": 0.813, "fsteam": 0.309, "ta": 500.0},
    "Texaco_I-6": {"tfcoal": 87.73, "foxy": 0.774, "fsteam": 0.291, "ta": 500.0},
}
ALIAS = {"texaco i-1": "Texaco_I-1", "texaco i-3": "Texaco_I-3", "texaco i-6": "Texaco_I-6"}


def cap_on(a, r_ox, r_c, r_w):
    rem = max(float(a) - float(r_ox), 0.0)
    lim = min(float(r_c), rem * 0.5 * 0.99)
    rem2 = max(rem - 2.0 * lim, 0.0)
    w = float(r_w)
    if w < 0.0:
        w = max(w, -rem2 * 0.99)
    return lim, w


def cap_off(a, r_ox, r_c, r_w):
    rem = max(float(a) - float(r_ox), 0.0)
    return min(float(r_c), rem * 0.5 * 0.99), float(r_w)


def wen_coal() -> dict:
    hhv = 33.5 * WEN_COAL["C"] / 100 + 144.0 * WEN_COAL["H"] / 100 \
        - 18.0 * WEN_COAL["O"] / 100 + 10.0 * WEN_COAL["S"] / 100
    return {
        "Cd": WEN_COAL["C"], "Hd": WEN_COAL["H"], "Od": WEN_COAL["O"],
        "Sd": WEN_COAL["S"], "Nd": WEN_COAL["N"], "Ad": WEN_COAL["Ash"],
        "Mt": WEN_MT_PCT, "HHV_d": hhv * 1000.0,
    }


def json_coal() -> dict:
    """validation JSON 实际使用的煤（CASE_TO_COAL_MAP 回落值）。"""
    from model.chemistry import COAL_DATABASE

    c = COAL_DATABASE["Paper_Base_Coal"]
    return {k: c[k] for k in ("Cd", "Hd", "Od", "Sd", "Nd", "Ad", "Mt", "HHV_d")}


def run(case: dict, geom: dict, mode: str, coal: dict) -> dict:
    tf = case["tfcoal"] / 1000.0
    op = {
        "coal_flow": tf,
        "o2_flow": case["foxy"] * tf,
        "steam_flow": case["fsteam"] * tf,
        "P": 2431800.0,
        "T_in": case["ta"],
        "HeatLossPercent": 3.0,
        "epsilon": 1.0,
        "ParticleSizeModel": "fortran_constant",
        "AllowTruncatedMesh": True,
        "SolverStrictRoot": False,
    }
    if mode == "F":
        op.update({"UseFortranDiffusion": True, "WGS_RatFactor": True, "CharCombustionRateFactor": 1.0})
    system = GasifierSystem(dict(geom), dict(coal), op)
    res, _z = system.solve(N_cells=25)
    costs = [0.5 * float(np.sum(np.asarray(c.residuals(np.asarray(r, dtype=float)), dtype=float) ** 2))
             for c, r in zip(system.cells, res)]
    gas = res[-1][:8]
    dry = float(np.sum(gas[:7]))
    cin = float(system.W_dry * float(system.coal_props["Cd"]) / 100.0 / 0.012011)
    cout = float(res[-1][8] * res[-1][9] / 0.012011)
    return {
        "T_K": float(res[-1][10]),
        "CO_vol_pct": float(gas[2] / dry * 100.0),
        "H2_vol_pct": float(gas[5] / dry * 100.0),
        "CO2_vol_pct": float(gas[3] / dry * 100.0),
        "X_C_pct": float(100.0 * (1.0 - cout / max(cin, 1e-12))),
        "max_residual_cost": float(max(costs)),
        "converged_1e-4": bool(max(costs) <= 1e-4),
    }


def main() -> None:
    fort = pd.read_csv(OUT / "fortran_outlet.csv")
    fort["paper_case"] = fort["case_name"].map(ALIAS)
    fr = fort.set_index("paper_case")

    rows = []
    settings = [
        ("M_current_json_coal_json_geom", "M", JSON_GEOM, "json"),
        ("M_wen_coal_paper_geom", "M", PAPER_GEOM, "wen"),
        ("M_wen_coal_fortran_geom", "M", FORTRAN_GEOM, "wen"),
        ("F_wen_coal_fortran_geom", "F", FORTRAN_GEOM, "wen"),
    ]
    for tag, mode, geom, coal_tag in settings:
        coal_fn = json_coal if coal_tag == "json" else wen_coal
        for cap_tag, fn in (("capped", cap_on), ("uncapped", cap_off)):
            Cell.limit_h2_consumers_after_oxidation = staticmethod(fn)
            for name, case in CASES.items():
                rec = run(case, geom, mode, coal_fn())
                rec.update({
                    "case_name": name,
                    "setting": tag,
                    "wgs_cap": cap_tag,
                    "coal_basis": coal_tag,
                    "L_m": geom["L"],
                    "D_m": geom["D"],
                    "fortran_T_K": float(fr.loc[name, "T_gas_K"]),
                    "fortran_CO": float(fr.loc[name, "CO_vol_pct"]),
                    "fortran_H2": float(fr.loc[name, "H2_vol_pct"]),
                    "fortran_CO2": float(fr.loc[name, "CO2_vol_pct"]),
                    "fortran_X_C": float(fr.loc[name, "X_C_pct"]),
                })
                rec["dT_vs_fortran_K"] = rec["T_K"] - rec["fortran_T_K"]
                rec["dCO_vs_fortran"] = rec["CO_vol_pct"] - rec["fortran_CO"]
                rows.append(rec)
    Cell.limit_h2_consumers_after_oxidation = staticmethod(cap_on)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "python_vs_fortran.csv", index=False)

    pd.set_option("display.width", 240)
    for tag in df["setting"].unique():
        sub = df[df["setting"] == tag]
        print(f"=== {tag} ===")
        print(sub[["case_name", "wgs_cap", "T_K", "fortran_T_K", "dT_vs_fortran_K",
                   "CO_vol_pct", "fortran_CO", "max_residual_cost", "converged_1e-4"]].to_string(index=False))
        print()


if __name__ == "__main__":
    main()
