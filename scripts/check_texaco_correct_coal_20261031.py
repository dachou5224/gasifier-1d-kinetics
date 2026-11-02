#!/usr/bin/env python3
"""用正确煤质（H-coal residue Illinois No.6）复算 Texaco I-1/I-3/I-6。

起因：`validation_cases_final.json` 这些算例的 `coal` 字段为空，加载器默认落成
`Paper_Base_Coal`（Cd=77.56, Od=9.44, Ad=7.11），与 Wen 原文的 H-coal residue
Illinois No.6（C=74.05, O=1.32, Ash=15.53）不是同一个煤。本脚本：

1. 用 Fortran namelist 的 dmmf 参数还原煤质，按 Fortran 的 tfcoal 口径设 `coal_flow`；
2. 在同一套几何/操作条件下分别跑「现行 H2 限幅」与「去掉负向 WGS 截断」；
3. 与原始 Fortran 出口值并列，判断 3400 K 是煤质问题还是限幅问题。
"""
from __future__ import annotations

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
from model.gasifier_system import GasifierSystem  # noqa: E402

OUT = ROOT / "outputs" / "diagnostics" / "round-wen-fortran-texaco-20261031"

# 原始论文 H-coal residue Illinois No.6，干基 wt%（data/archive/validation_cases_OriginalPaper.json）
COAL_DRY = {"C": 74.05, "H": 6.25, "O": 1.32, "S": 1.77, "N": 0.71, "Ash": 15.53}
MT_PCT = 0.2  # xmois = 0.002

# Fortran namelist 口径的工况（tfcoal 是收到基总给料 g/s）
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


def cap_on(avail_h2, r_h2_ox, r_c_h2, r_wgs):
    rem = max(float(avail_h2) - float(r_h2_ox), 0.0)
    limited = min(float(r_c_h2), rem * 0.5 * 0.99)
    rem2 = max(rem - 2.0 * limited, 0.0)
    w = float(r_wgs)
    if w < 0.0:
        w = max(w, -rem2 * 0.99)
    return limited, w


def cap_off(avail_h2, r_h2_ox, r_c_h2, r_wgs):
    rem = max(float(avail_h2) - float(r_h2_ox), 0.0)
    return min(float(r_c_h2), rem * 0.5 * 0.99), float(r_wgs)


def dulong_hhv_mj_kg(c, h, o, s):
    return 33.5 * c / 100.0 + 144.0 * h / 100.0 - 18.0 * o / 100.0 + 10.0 * s / 100.0


def coal_props() -> dict:
    return {
        "Cd": COAL_DRY["C"], "Hd": COAL_DRY["H"], "Od": COAL_DRY["O"],
        "Sd": COAL_DRY["S"], "Nd": COAL_DRY["N"], "Ad": COAL_DRY["Ash"],
        "Mt": MT_PCT,
        "HHV_d": dulong_hhv_mj_kg(COAL_DRY["C"], COAL_DRY["H"], COAL_DRY["O"], COAL_DRY["S"]) * 1000.0,
    }


def run(case: dict, n_cells: int = 25) -> dict:
    tfcoal_kg_s = case["tfcoal"] / 1000.0
    op = {
        "coal_flow": tfcoal_kg_s,
        "o2_flow": case["foxy"] * tfcoal_kg_s,
        "steam_flow": case["fsteam"] * tfcoal_kg_s,
        "P": 2431800.0,
        "T_in": case["ta"],
        "HeatLossPercent": 3.0,
        "epsilon": 1.0,
        "ParticleSizeModel": "fortran_constant",
        "AllowTruncatedMesh": True,
        "SolverStrictRoot": False,
    }
    geo = {"L": 6.096, "D": 1.524}   # Wen: 20 ft x 5 ft
    system = GasifierSystem(geo, coal_props(), op)
    res, _z = system.solve(N_cells=n_cells)
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
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for tag, fn in (("capped_current", cap_on), ("no_wgs_cap", cap_off)):
        Cell.limit_h2_consumers_after_oxidation = staticmethod(fn)
        for name, case in CASES.items():
            rec = run(case)
            rec.update({"case_name": name, "variant": tag, **{f"fortran_{k}": v for k, v in FORTRAN_REF[name].items()}})
            rows.append(rec)
    Cell.limit_h2_consumers_after_oxidation = staticmethod(cap_on)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "python_correct_coal.csv", index=False)
    print(df[["case_name", "variant", "T_K", "CO_vol_pct", "H2_vol_pct", "CO2_vol_pct",
              "X_C_pct", "max_residual_cost", "n_cells_above_1e-4"]].to_string(index=False))
    print()
    print(df[["case_name", "variant", "fortran_T_K", "T_K", "fortran_CO", "CO_vol_pct",
              "fortran_H2", "H2_vol_pct", "fortran_X_C", "X_C_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
