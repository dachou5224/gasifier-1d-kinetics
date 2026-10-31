#!/usr/bin/env python3
"""改正 Texaco 输入后的全算例复算与 Fortran 对照。

数据改动见 `chemistry.py` 的三类 Wen 进料与 `validation_cases_final.json` 的
`coal_type` / `L_reactor_m` / `D_reactor_m`。本脚本核对：
1. 改正后出口温度是否落回物理区间（不再出现 3000 K 以上）；
2. 与 `reference_fortran` 的实际输出差多少；
3. 残差成本是否达到 1e-4。
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

from model.chemistry import COAL_DATABASE  # noqa: E402
from model.gasifier_system import GasifierSystem  # noqa: E402
from model.validation_loader import (  # noqa: E402
    get_validation_cases_final_path,
    iter_validation_cases_final,
    normalize_final_case_to_legacy,
)

OUT = ROOT / "outputs" / "diagnostics" / "round-texaco-input-fix-20261031"

# Fortran 实际输出（round-wen-fortran-texaco-20261031/fortran_outlet.csv）
FORTRAN = ROOT / "outputs" / "diagnostics" / "round-wen-fortran-texaco-20261031" / "fortran_outlet.csv"
ALIAS = {"Texaco_I-1": "texaco i-1", "Texaco_I-2": "texaco i-2", "Texaco_I-5C": "texaco i-5c",
         "Texaco_I-10": "texaco i-10", "Texaco_Exxon": "texaco exxon",
         "Texaco_I-3": "texaco i-3", "Texaco_I-6": "texaco i-6"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = json.load(open(get_validation_cases_final_path(), encoding="utf-8"))
    cases = {n: normalize_final_case_to_legacy(n, d) for n, d in iter_validation_cases_final(raw)}
    fort = pd.read_csv(FORTRAN).set_index("case_name")

    rows = []
    for name, data in cases.items():
        inp = data["inputs"]
        coal_key = data["coal_type"]
        if coal_key not in COAL_DATABASE:
            rows.append({"case_name": name, "status": f"missing_coal:{coal_key}"})
            continue
        coal = COAL_DATABASE[coal_key].copy()
        w = inp["FeedRate_kg_h"] / 3600.0
        op = {
            "coal_flow": w,
            "o2_flow": w * inp["Ratio_OC"],
            "steam_flow": w * inp["SteamRatio_SC"],
            "P": inp["P_Pa"],
            "T_in": inp["T_in_K"],
            "HeatLossPercent": inp["HeatLossPercent"],
            "SlurryConcentration": inp["SlurryConcentration"],
            "epsilon": inp["Voidage"],
        }
        geo = {"L": inp["L_reactor"], "D": inp["D_reactor"]}
        system = GasifierSystem(geo, coal, op)
        try:
            res, z = system.solve(N_cells=25)
        except Exception as exc:
            rows.append({"case_name": name, "coal_type": coal_key, "status": f"error:{type(exc).__name__}"})
            continue
        costs = [0.5 * float(np.sum(np.asarray(
            c.residuals(np.asarray(r, dtype=float)), dtype=float) ** 2))
            for c, r in zip(system.cells, res)]
        gas = res[-1][:8]
        dry = float(np.sum(gas[:7]))
        cin = float(system.W_dry * float(system.coal_props["Cd"]) / 100.0 / 0.012011)
        cout = float(res[-1][8] * res[-1][9] / 0.012011)
        fk = ALIAS.get(name)
        rec = {
            "case_name": name,
            "coal_type": coal_key,
            "status": "ok",
            "L_m": geo["L"], "D_m": geo["D"],
            "T_out_K": float(res[-1][10]),
            "CO_vol_pct": float(gas[2] / dry * 100.0),
            "H2_vol_pct": float(gas[5] / dry * 100.0),
            "CO2_vol_pct": float(gas[3] / dry * 100.0),
            "X_C_pct": float(100.0 * (1.0 - cout / max(cin, 1e-12))),
            "max_residual_cost": float(max(costs)),
            "converged_1e-4": bool(max(costs) <= 1e-4),
            "expected_CO": data["expected"].get("YCO"),
            "expected_H2": data["expected"].get("YH2"),
            "expected_TOUT_C": data["expected"].get("TOUT_C"),
        }
        if fk and fk in fort.index:
            rec["fortran_T_K"] = float(fort.loc[fk, "T_gas_K"])
            rec["fortran_CO"] = float(fort.loc[fk, "CO_vol_pct"])
            rec["fortran_H2"] = float(fort.loc[fk, "H2_vol_pct"])
            rec["fortran_X_C"] = float(fort.loc[fk, "X_C_pct"])
            rec["dT_vs_fortran_K"] = rec["T_out_K"] - rec["fortran_T_K"]
        rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "after_fix_all_cases.csv", index=False)
    ok = df[df["status"] == "ok"]
    pd.set_option("display.width", 240)
    print(ok[["case_name", "coal_type", "T_out_K", "CO_vol_pct", "X_C_pct",
              "max_residual_cost", "converged_1e-4"]].to_string(index=False))
    print()
    print(f"算例 {len(df)}，成功 {len(ok)}，收敛 {int(ok['converged_1e-4'].sum())}")
    print(f"温度区间: {ok['T_out_K'].min():.0f} – {ok['T_out_K'].max():.0f} K")
    print(f"成本区间: {ok['max_residual_cost'].min():.1e} – {ok['max_residual_cost'].max():.1e}")


if __name__ == "__main__":
    main()
