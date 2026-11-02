#!/usr/bin/env python3
"""Texaco I-1/I-3/I-6 的全因子归因：煤质 × 反应器几何 × H2 限幅。

用于把 `test_validation_config` 中 3400 K 的失败拆成三项来源，并与原始
Wen & Chaung Fortran 的出口值对照。

- 煤质：「文档口径」= Paper_Base_Coal（CASE_TO_COAL_MAP 现值）；「原文口径」= H-coal residue Illinois No.6
- 几何：「文档口径」= L 6.0 m / D 2.0 m（validation JSON） ；「原文口径」= L 6.096 m / D 1.524 m（20 ft × 5 ft）
- 限幅：「现行」= 保留负向 WGS 截断；「不带截断」= 只保留 C+H2 收紧
"""
from __future__ import annotations

import itertools
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
logging.basicConfig(level=logging.ERROR)

from model.cell import Cell  # noqa: E402
from model.chemistry import COAL_DATABASE  # noqa: E402
from model.gasifier_system import GasifierSystem  # noqa: E402

OUT = ROOT / "outputs" / "diagnostics" / "round-wen-fortran-texaco-20261031"

WEN_COAL_DRY = {"C": 74.05, "H": 6.25, "O": 1.32, "S": 1.77, "N": 0.71, "Ash": 15.53}
WEN_MT_PCT = 0.2
WEN_GEOM = {"L": 6.096, "D": 1.524}
JSON_GEOM = {"L": 6.0, "D": 2.0}

CASES = {
    "Texaco_I-1": {"tfcoal": 76.66, "foxy": 0.866, "fsteam": 0.241, "ta": 505.22},
    "Texaco_I-3": {"tfcoal": 82.202, "foxy": 0.813, "fsteam": 0.309, "ta": 500.0},
    "Texaco_I-6": {"tfcoal": 87.73, "foxy": 0.774, "fsteam": 0.291, "ta": 500.0},
}
FORTRAN_REF = {
    "Texaco_I-1": {"T_K": 1373.16, "CO": 54.912, "H2": 39.322, "CO2": 4.798, "X_C": 92.372},
    "Texaco_I-3": {"T_K": 1350.38, "CO": 51.753, "H2": 41.165, "CO2": 6.142, "X_C": 87.143},
    "Texaco_I-6": {"T_K": 1344.87, "CO": 51.245, "H2": 41.152, "CO2": 6.264, "X_C": 82.798},
}


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


def dulong(c, h, o, s):
    return 33.5 * c / 100.0 + 144.0 * h / 100.0 - 18.0 * o / 100.0 + 10.0 * s / 100.0


def coal(kind: str) -> dict:
    if kind == "wen":
        return {
            "Cd": WEN_COAL_DRY["C"], "Hd": WEN_COAL_DRY["H"], "Od": WEN_COAL_DRY["O"],
            "Sd": WEN_COAL_DRY["S"], "Nd": WEN_COAL_DRY["N"], "Ad": WEN_COAL_DRY["Ash"],
            "Mt": WEN_MT_PCT,
            "HHV_d": dulong(WEN_COAL_DRY["C"], WEN_COAL_DRY["H"], WEN_COAL_DRY["O"], WEN_COAL_DRY["S"]) * 1000.0,
        }
    c = COAL_DATABASE["Paper_Base_Coal"]
    return {k: c[k] for k in ("Cd", "Hd", "Od", "Sd", "Nd", "Ad", "Mt", "HHV_d")}


def run(case: dict, coal_kind: str, geom_kind: str) -> dict:
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
    geo = dict(WEN_GEOM if geom_kind == "wen" else JSON_GEOM)
    system = GasifierSystem(geo, coal(coal_kind), op)
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
        "n_cells_above_1e-4": int(sum(c > 1e-4 for c in costs)),
        "converged": bool(max(costs) <= 1e-4),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for coal_kind, geom_kind, cap_kind in itertools.product(
        ("json", "wen"), ("json", "wen"), ("capped", "uncapped")
    ):
        Cell.limit_h2_consumers_after_oxidation = staticmethod(cap_on if cap_kind == "capped" else cap_off)
        for name, case in CASES.items():
            rec = run(case, coal_kind, geom_kind)
            rec.update({
                "case_name": name,
                "coal_basis": coal_kind,
                "geometry_basis": geom_kind,
                "wgs_cap": cap_kind,
                "fortran_T_K": FORTRAN_REF[name]["T_K"],
                "fortran_CO": FORTRAN_REF[name]["CO"],
                "fortran_H2": FORTRAN_REF[name]["H2"],
                "fortran_CO2": FORTRAN_REF[name]["CO2"],
                "fortran_X_C": FORTRAN_REF[name]["X_C"],
            })
            rec["dT_vs_fortran_K"] = rec["T_K"] - FORTRAN_REF[name]["T_K"]
            rec["dCO_vs_fortran"] = rec["CO_vol_pct"] - FORTRAN_REF[name]["CO"]
            rows.append(rec)
    Cell.limit_h2_consumers_after_oxidation = staticmethod(cap_on)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "full_factorial_attribution.csv", index=False)

    pd.set_option("display.width", 220)
    print(df[["case_name", "coal_basis", "geometry_basis", "wgs_cap", "T_K",
              "max_residual_cost", "converged", "dT_vs_fortran_K"]].to_string(index=False))
    print()
    ok = df[df["converged"]]
    print(f"收敛组合数: {len(ok)}/{len(df)}")
    if len(ok):
        print(ok[["case_name", "coal_basis", "geometry_basis", "wgs_cap", "T_K", "dT_vs_fortran_K"]].to_string(index=False))


if __name__ == "__main__":
    main()
