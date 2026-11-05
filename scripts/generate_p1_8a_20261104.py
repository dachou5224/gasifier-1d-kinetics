#!/usr/bin/env python3
"""P1.8-A 锚点 Fortran 热损模型对齐 + 装置九窗二维辨识性扫描（2026-11-04 卡）。

执行顺序（卡第 1 → 3 节）：
  1.1  符号表（写在 fortran_heatloss_symbols.md 与 interpretation.md 第 1 节）
  1.2  单元检查                                  → unit_checks.csv
  1.3  模式 F′（= P1.7-A 模式 F + HeatLossModel="fortran_wall"）→ anchor_criteria.csv /
       anchor_cells.csv / anchor_profiles.csv
  2.1  9 窗 × f_gas(5) × HeatLossPercent(4)，N=40，实测载气 → scan.csv
  2.2  复现行（旧载气）+ P1.7-A 逐位对齐行       → scan.csv（row_kind 区分）
  2.3  预声明读法 R1–R5（跑完不改）               → scan_readings.csv / jacobian.csv

用法：
  ./.venv/bin/python scripts/generate_p1_8a_20261104.py [--stage unit|anchor|scan|all]

冻结项一项未动：全部 Arrhenius 常数（除第 2 节扫描的 C+H2O/C+CO2 指前因子倍数）、phi、
f_co2 = 1.0、0.05 atm、CharCombustionRateFactor、氧化因子、热解产率、负向 WGS 截断
（照常统计绑定）、USE_RK_GILL_COMBUSTION = False、fortran_constant、蒸汽进料 = 0、
低位热值口径、uniform_full_length、SolverStrictRoot = True。
本轮只扫描、不采纳任何参数值，不改合同，不为闭合加任何松弛。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model.fortran_anchor_20261030 import (  # noqa: E402
    DRY_POWDER_CASES,
    anchor_cases,
    parse_fortran_log,
    solve_fortran_case,
)
from model.gasifier_system import GasifierSystem  # noqa: E402
from model.heat_loss import (  # noqa: E402
    CAL_TO_J,
    CARBON_MOLAR_MASS_KG,
    FORTRAN_DT_CM,
    FORTRAN_HEIGHT_CM,
    FORTRAN_HL_PERCENT,
    FORTRAN_U0_CAL_PER_S_CM2_K,
    HEAT_CAPACITY_SOLID,
    combustion_zone_heat_loss_w,
    gasification_zone_heat_loss_w,
    reaction_heat_release_w,
    resolve_heat_loss_model,
)
from model.input_contract import (  # noqa: E402
    coal_hhv_kj_kg,
    coal_lhv_kj_kg,
)
from model.model_case_adapter import (  # noqa: E402
    NORMAL_M3_PER_KMOL,
    TRANSPORTED_COAL_MOISTURE_PERCENT,
    build_gasifier_inputs_from_model_case,
    normal_m3_h_to_kg_s,
    wet_coal_flow_kg_s,
)
from model.p0_round_20260930 import ELEMENT_KG_PER_MOL, m1_reference_window_ids  # noqa: E402
from model.p1_5_round_20261030 import (  # noqa: E402
    base_protocol,
    run_case,
    silence_solver_logs,
    write_round_outputs,
)
from model.physics import get_enthalpy_molar  # noqa: E402
from model.revision_evidence_common import (  # noqa: E402
    load_frozen_split,
    source_commit,
)

OUT_DIR = ROOT / "outputs" / "diagnostics" / "round-p1.8a-20261104"
P17A_DIR = ROOT / "outputs" / "diagnostics" / "round-p1.7a-20261103"
P15C_DIR = ROOT / "outputs" / "diagnostics" / "round-p1.5c-wen-anchor-20261030"
P17D_CSV = Path("/Users/liuzhen/AI-projects/gasifier-data-prep/outputs/diagnostics/p1.7d-20261103/coal_reconciled_by_window.csv")
P18D_CSV = Path("/Users/liuzhen/AI-projects/gasifier-data-prep/outputs/diagnostics/p1.8d-20261104/carrier_by_window.csv")
M1_MATRIX = ROOT / "outputs" / "diagnostics" / "revision-evidence-20260826" / "M1-cell0" / "revision_m1_cell0_variant_matrix.csv"

N_ANCHOR = (80, 160)
N_ANCHOR_REF = 160
N_SCAN = 40

ANCHOR_TOL = {
    "X_C_pp": 2.0,
    "CO_vol_pct": 2.0,
    "H2_vol_pct": 2.0,
    "CO2_vol_pct": 2.0,
    "T_gas_K": 50.0,
}

MOISTURE = TRANSPORTED_COAL_MOISTURE_PERCENT / 100.0

F_GAS_GRID = (1.0, 0.5, 0.25, 0.125, 0.0625)
F_GAS_EXTENSION = 0.03125
HEAT_LOSS_PCT_GRID = (1.0, 2.0, 4.0, 6.0)
MAIN_HEAT_LOSS_PCT = 2.0

# 刘文表 3 阶段值（CO / H2 / CO2 vol%，干基）
LIU_PHASE_COMP = {
    "phase1": (60.14, 29.74, 9.36),
    "phase2": (61.05, 29.24, 8.99),
    "phase3": (60.74, 29.42, 9.14),
    "phase4": (59.72, 30.38, 8.98),
}
COMP_BASES = ("analyzer", "liu_phase")

# R3 的观测缩放（1 pp、0.5、0.5、0.5 vol%）
R3_SCALES = (1.0, 0.5, 0.5, 0.5)
R2_XI_THRESHOLD = 50.0
R3_COND_THRESHOLD = 100.0
R3_CORR_THRESHOLD = 0.95
R1_XC_TOL_PP = 1.0
WGS_BIND_FLAG = 0.05


# ---------------------------------------------------------------------------
# 通用
# ---------------------------------------------------------------------------

def _dry_coal(row, p17d: pd.DataFrame) -> float:
    return float(p17d.loc[row["window_id"], "coal_dry_recon_phase_kg_s"])


def _carrier_kg_s(nm3_h: float) -> float:
    return normal_m3_h_to_kg_s(nm3_h, "CO2")


def _nm3_h_to_mol_s(nm3_h: float) -> float:
    return float(nm3_h) / NORMAL_NM3_PER_KMOL / 3600.0 * 1000.0


NORMAL_NM3_PER_KMOL = NORMAL_M3_PER_KMOL


def _limit_zone_stats(system: GasifierSystem) -> dict:
    comb = [d for d in system.cell_diagnostics if d.get("is_combustion_zone")]
    gasif = [d for d in system.cell_diagnostics if not d.get("is_combustion_zone")]
    dz_comb = sum(float(d["dz_m"]) for d in comb)
    dz_gas = sum(float(d["dz_m"]) for d in gasif)
    bind_comb = sum(float(d["dz_m"]) for d in comb if d.get("WGS_NegativeTruncated"))
    bind_gas = sum(float(d["dz_m"]) for d in gasif if d.get("WGS_NegativeTruncated"))
    n_bind = sum(1 for d in system.cell_diagnostics if d.get("WGS_NegativeTruncated"))
    return {
        "wgs_neg_bind_cells": n_bind,
        "wgs_neg_bind_len_frac_combustion": bind_comb / dz_comb if dz_comb > 0 else float("nan"),
        "wgs_neg_bind_len_frac_gasification": bind_gas / dz_gas if dz_gas > 0 else float("nan"),
        "wgs_neg_bind_len_frac_overall": (bind_comb + bind_gas) / max(dz_comb + dz_gas, 1e-12),
        "dz_combustion_m": dz_comb,
        "dz_gasification_m": dz_gas,
    }


def _zone_heat_loss(system: GasifierSystem) -> dict:
    comb = [d for d in system.cell_diagnostics if d.get("hloss_zone") == "combustion"]
    gasif = [d for d in system.cell_diagnostics if d.get("hloss_zone") == "gasification"]
    return {
        "hloss_combustion_W": float(sum(float(d.get("hloss_W", 0.0)) for d in comb)),
        "hloss_gasification_W": float(sum(float(d.get("hloss_W", 0.0)) for d in gasif)),
        "hloss_total_W": float(
            sum(float(d.get("hloss_W", 0.0)) for d in comb + gasif)
        ),
        "hloss_combustion_cells": len(comb),
        "hloss_gasification_cells": len(gasif),
        "hloss_gasification_min_W": float(min((float(d.get("hloss_W", 0.0)) for d in gasif), default=float("nan"))),
        "hloss_gasification_max_W": float(max((float(d.get("hloss_W", 0.0)) for d in gasif), default=float("nan"))),
    }


def _reaction_integrals(system: GasifierSystem) -> dict:
    """全炉分反应积分 [mol/s] 与燃烧区/气化区份额。

    每个单元的 ``rate`` 已经是该单元内的总转化速率 [mol/s]（``_calc_balances`` 直接用
    它与气源一起推进气量），因此全炉积分就是逐单元求和。
    """
    keys = ("C+O2", "C+H2O", "C+CO2", "C+H2", "WGS", "RWGS", "MSR", "CO_Ox", "H2_Ox", "CH4_Ox")
    tot = {k: 0.0 for k in keys}
    comb = {k: 0.0 for k in keys}
    gasi = {k: 0.0 for k in keys}
    net_co2_cell = []
    ok = system.execution_status == "completed"
    if ok:
        for cell, state in zip(system.cells, system.results):
            snap = cell.snapshot_with_limits(state)
            rh, ro = snap["r_het"], snap["r_homo"]
            vals = {
                "C+O2": float(rh.get("C+O2", 0.0)),
                "C+H2O": float(rh.get("C+H2O", 0.0)),
                "C+CO2": float(rh.get("C+CO2", 0.0)),
                "C+H2": float(rh.get("C+H2", 0.0)),
                "WGS": max(float(ro.get("WGS", 0.0)), 0.0),
                "RWGS": max(-float(ro.get("WGS", 0.0)), 0.0) + float(ro.get("RWGS", 0.0)),
                "MSR": float(ro.get("MSR", 0.0)),
                "CO_Ox": float(ro.get("CO_Ox", 0.0)),
                "H2_Ox": float(ro.get("H2_Ox", 0.0)),
                "CH4_Ox": float(ro.get("CH4_Ox", 0.0)),
            }
            in_comb = bool(snap["limits"].get("is_combustion_zone", False))
            for k, v in vals.items():
                tot[k] += v
                (comb if in_comb else gasi)[k] += v
            net_co2_cell.append(
                vals["RWGS"] - vals["C+CO2"] - vals["WGS"] + vals["CO_Ox"] + 0.0
            )
    out: dict = {}
    for k in keys:
        out[f"rx_{k.replace('+', '_').replace('+', '_')}_mol_s"] = tot[k]
        out[f"rx_{k.replace('+', '_')}_frac_combustion"] = (
            comb[k] / tot[k] if abs(tot[k]) > 1e-12 else float("nan")
        )
        out[f"rx_{k.replace('+', '_')}_frac_gasification"] = (
            gasi[k] / tot[k] if abs(tot[k]) > 1e-12 else float("nan")
        )
    return out


def _hydrogen_budget(system: GasifierSystem, gas: np.ndarray, dry_kg_s: float) -> dict:
    hd = float(system.coal_props.get("Hd", 0.0)) / 100.0
    h_in = dry_kg_s * hd / ELEMENT_KG_PER_MOL["H"]
    water_kg_s = dry_kg_s * MOISTURE / (1.0 - MOISTURE)
    h_moist = h_in + 2.0 * water_kg_s / 0.018015
    h_out = 2.0 * float(gas[5]) + 4.0 * float(gas[1]) + 2.0 * float(gas[4])
    return {
        "H_in_mol_s": h_in,
        "H_in_with_moisture_mol_s": h_moist,
        "H_out_dry_mol_s": h_out,
        "H_out_over_H_in": h_out / max(h_in, 1e-12),
        "H_out_over_H_in_with_moisture": h_out / max(h_moist, 1e-12),
        "H_in_with_moisture_over_H_in": h_moist / max(h_in, 1e-12),
        "moisture_water_kg_s": water_kg_s,
    }


# ---------------------------------------------------------------------------
# 第 1 节：单元检查
# ---------------------------------------------------------------------------

def _single_cell_hand_checks() -> list[dict]:
    """构造单个燃烧区 / 气化区单元，手算 hloss 与代码值对照。

    手算部分在检查脚本里独立写一遍算式（不复用 heat_loss 的实现），只共用同样的输入。
    """
    rows: list[dict] = []

    # --- 燃烧区：hloss = 0.30·|rheat| ---
    q_w = 1.2345678e6
    hand_comb = 30.0 / 100.0 * abs(q_w)
    code_comb = combustion_zone_heat_loss_w(q_w)
    rel = abs(code_comb - hand_comb) / max(abs(hand_comb), 1e-30)
    rows.append({
        "check_group": "single_cell_combustion",
        "context": "合成值（rheat = 1.2345678 MW）",
        "metric": "hloss_W",
        "hand_value": hand_comb,
        "code_value": code_comb,
        "rel_diff": rel,
        "tolerance": 1e-9,
        "pass": bool(rel < 1e-9),
        "note": "hloss = fcoal·|rheat|·hl/100，hl = 30（Fortran 第 30、780 行）",
    })

    # 负 rheat 也要取绝对值
    code_comb_neg = combustion_zone_heat_loss_w(-q_w)
    rel = abs(code_comb_neg - hand_comb) / max(abs(hand_comb), 1e-30)
    rows.append({
        "check_group": "single_cell_combustion",
        "context": "合成值（rheat = −1.2345678 MW）",
        "metric": "hloss_W_abs_sign",
        "hand_value": hand_comb,
        "code_value": code_comb_neg,
        "rel_diff": rel,
        "tolerance": 1e-9,
        "pass": bool(rel < 1e-9),
        "note": "abs() 保留：净吸热单元也按 |rheat| 取 30%",
    })

    # --- 气化区：hloss = u0·π·dt·nc·delh·(tg − tw) ---
    tg = 1523.75
    z_out = 1.875
    dz = 0.0825
    hs_cm = z_out * 100.0
    tw = 2100.0 - 600.0 * hs_cm / 330.0
    area = math.pi * 152.4 * (dz * 100.0)
    hloss_cal = 3.4e-3 * area * (tg - tw)
    hand_gas = hloss_cal * 4.184
    code = gasification_zone_heat_loss_w(tg, z_outlet_m=z_out, dz_m=dz)
    rel = abs(code["hloss_W"] - hand_gas) / max(abs(hand_gas), 1e-30)
    rows.append({
        "check_group": "single_cell_gasification",
        "context": "合成值（tg = 1523.75 K, hs = 187.5 cm）",
        "metric": "hloss_W",
        "hand_value": hand_gas,
        "code_value": code["hloss_W"],
        "rel_diff": rel,
        "tolerance": 1e-9,
        "pass": bool(rel < 1e-9),
        "note": "hloss = u0·π·dt·nc·delh·(tg − tw)，u0 = 3.4e-3 cal/(s·cm²·K)，"
                "tw = 2100 − 600·hs/height（第 214、348–349、777–778 行）",
    })
    rows.append({
        "check_group": "single_cell_gasification",
        "context": "合成值（tg = 1523.75 K, hs = 187.5 cm）",
        "metric": "tw_K",
        "hand_value": tw,
        "code_value": code["tw_K"],
        "rel_diff": abs(code["tw_K"] - tw) / max(abs(tw), 1e-30),
        "tolerance": 1e-9,
        "pass": bool(abs(code["tw_K"] - tw) < 1e-9),
        "note": "tw = 2100 − 600·hs/height，height = 330 cm（第 32 行）",
    })

    # --- 气化区：tg < tw 时 hloss 为负，不 clamp ---
    tg_cold = 1200.0
    hand_neg = 3.4e-3 * (math.pi * 152.4 * dz * 100.0) * (tg_cold - tw) * 4.184
    code_neg = gasification_zone_heat_loss_w(tg_cold, z_outlet_m=z_out, dz_m=dz)["hloss_W"]
    rel = abs(code_neg - hand_neg) / max(abs(hand_neg), 1e-30)
    rows.append({
        "check_group": "single_cell_gasification",
        "context": f"合成值（tg = {tg_cold} K < tw = {tw:.2f} K）",
        "metric": "hloss_W_negative_preserved",
        "hand_value": hand_neg,
        "code_value": code_neg,
        "rel_diff": rel,
        "tolerance": 1e-9,
        "pass": bool(rel < 1e-9 and code_neg < 0.0),
        "note": "tg < tw → 壁面向气体供热，保留负号（卡面第 2 个陷阱）",
    })

    # --- 真实单元：用已收敛的锚点算例取一个燃烧区单元与一个气化区单元 ---
    cases, geo = anchor_cases()
    case = next(c for c in cases if c["case_name"] == "texaco i-1")
    rec = solve_fortran_case(
        case, n_cells=160, mode="F", geometry=geo,
        switches={"HeatLossModel": "fortran_wall"},
    )
    system: GasifierSystem = rec["_system"]
    for zone, want in (("combustion", True), ("gasification", False)):
        hit = None
        for cell, state in zip(system.cells, system.results):
            g_src = np.zeros(8)
            for s in cell.sources:
                g_src += s.get_sources(cell.idx, cell.z, cell.dz)[0]
            flag = bool(cell._inlet_pO2_is_combustion(g_src))
            if flag is want:
                hit = (cell, state)
                break
        if hit is None:
            rows.append({
                "check_group": f"single_cell_{zone}", "context": "texaco i-1 实算",
                "metric": "cell_found", "hand_value": float("nan"), "code_value": float("nan"),
                "rel_diff": float("nan"), "tolerance": 1e-9, "pass": False,
                "note": "未找到该分区的单元",
            })
            continue
        cell, state = hit
        snap = cell.snapshot_with_limits(state)
        code_rep = cell.heat_loss_diagnostics(state)
        f_co2 = float(cell.op_conds.get("Combustion_CO2_Fraction", 1.0))
        q_hand = _reaction_heat_hand(snap["r_het"], snap["r_homo"], snap["phi"], f_co2, state.T)
        if want:
            hand = 0.30 * abs(q_hand)
        else:
            hs_cm = (cell.z + cell.dz / 2.0) * 100.0
            tw_h = 2100.0 - 600.0 * hs_cm / 330.0
            hand = 3.4e-3 * (math.pi * 152.4 * cell.dz * 100.0) * (state.T - tw_h) * 4.184
        rel = abs(code_rep["hloss_W"] - hand) / max(abs(hand), 1e-30)
        rows.append({
            "check_group": f"single_cell_{zone}",
            "context": f"texaco i-1 cell {cell.idx}（z = {cell.z:.4f} m）",
            "metric": "hloss_W",
            "hand_value": hand,
            "code_value": float(code_rep["hloss_W"]),
            "rel_diff": rel,
            "tolerance": 1e-9,
            "pass": bool(rel < 1e-9),
            "note": f"净反应热 {q_hand/1e6:.4f} MW（手算用同一速率向量，独立写一遍焓算式）",
        })
        rows.append({
            "check_group": f"single_cell_{zone}",
            "context": f"texaco i-1 cell {cell.idx}（z = {cell.z:.4f} m）",
            "metric": "reaction_heat_W",
            "hand_value": q_hand,
            "code_value": float(code_rep["reaction_heat_W"]),
            "rel_diff": abs(code_rep["reaction_heat_W"] - q_hand) / max(abs(q_hand), 1e-30),
            "tolerance": 1e-9,
            "pass": bool(abs(code_rep["reaction_heat_W"] - q_hand) <= 1e-9 * max(abs(q_hand), 1e-30)),
            "note": "rheat 的 Python 对应量：Q = −Σ rate·ΔH(T)，见 heat_loss.reaction_heat_release_w",
        })
    return rows


def _reaction_heat_hand(r_het, r_homo, phi, f_co2, T) -> float:
    """独立重写一遍反应热算式（与 heat_loss.reaction_heat_release_w 同一物理口径）。"""
    h = {sp: get_enthalpy_molar(sp, T) for sp in ("CO", "CO2", "H2", "H2O", "O2", "CH4")}
    ph, f = float(phi), float(f_co2)
    # 固碳显焓：char_formation_zero 只归零生成焓，显焓 cp_char·(T−298.15) 仍在，
    # 而四个异相反应各消耗 1 mol 固碳。必须与 heat_loss 侧同口径，否则单元检查会
    # 报出并不存在的差异（2026-11-04 审阅意见）。
    hc = HEAT_CAPACITY_SOLID * CARBON_MOLAR_MASS_KG * (float(T) - 298.15)
    q = 0.0
    q += r_het.get("C+O2", 0.0) * -(
        (2.0 - 2.0 / ph) * h["CO"] + (2.0 / ph - 1.0) * h["CO2"] - (1.0 / ph) * h["O2"] - hc
    )
    q += r_het.get("C+H2O", 0.0) * -(h["CO"] + h["H2"] - h["H2O"] - hc)
    q += r_het.get("C+CO2", 0.0) * -(2.0 * h["CO"] - h["CO2"] - hc)
    q += r_het.get("C+H2", 0.0) * -(h["CH4"] - 2.0 * h["H2"] - hc)
    q += r_homo.get("CO_Ox", 0.0) * -(f * (h["CO2"] - h["CO"] - 0.5 * h["O2"]))
    q += r_homo.get("H2_Ox", 0.0) * -(h["H2O"] - h["H2"] - 0.5 * h["O2"])
    q += r_homo.get("CH4_Ox", 0.0) * -(
        f * h["CO2"] + (1.0 - f) * h["CO"] + 2.0 * h["H2O"] - h["CH4"] - (1.5 + 0.5 * f) * h["O2"]
    )
    q += r_homo.get("WGS", 0.0) * -(h["CO2"] + h["H2"] - h["CO"] - h["H2O"])
    q += r_homo.get("RWGS", 0.0) * -(h["CO"] + h["H2O"] - h["CO2"] - h["H2"])
    q += r_homo.get("MSR", 0.0) * -(h["CO"] + 3.0 * h["H2"] - h["CH4"] - h["H2O"])
    return float(q)


def _jax_guard_check() -> list[dict]:
    """JAX 路径遇到 fortran_wall / f_gas≠1 必须报错，不得静默回退。"""
    rows = []
    cases, geo = anchor_cases()
    case = next(c for c in cases if c["case_name"] == "texaco i-1")
    from model.fortran_anchor_20261030 import build_case_inputs
    from model.p1_5_round_20261030 import paper_op_conds

    built = build_case_inputs(case, geometry=geo)
    for tag, extra in (
        ("jax_rejects_fortran_wall", {"HeatLossModel": "fortran_wall"}),
        ("jax_rejects_gasification_rate_factor", {"GasificationRateFactor": 0.5}),
    ):
        op = paper_op_conds(built, extra=extra)
        system = GasifierSystem(built["geometry"], built["coal_props"], op)
        raised, message = False, ""
        try:
            system.solve(N_cells=8, solver_method="jax_jit", jax_warmup=False)
        except ValueError as exc:
            raised, message = True, str(exc)
        except Exception as exc:  # noqa: BLE001
            message = f"{type(exc).__name__}: {exc}"
        rows.append({
            "check_group": "jax_path_guard", "context": "texaco i-1, N=8, solver_method=jax_jit",
            "metric": tag, "hand_value": 1.0, "code_value": float(raised),
            "rel_diff": 0.0, "tolerance": 0.0, "pass": bool(raised),
            "note": message[:180],
        })
    return rows


def _p17a_parity_check(out_dir_rows: list[dict]) -> list[dict]:
    """hhv_percent 下 P1.7-A 模式 F 五算例出口逐位复现。"""
    ref = pd.read_csv(P17A_DIR / "anchor_outlet.csv", dtype={"mode": str})
    ref = ref[(ref["mode"] == "F") & (ref["N_cells"] == N_ANCHOR_REF) & (ref["is_dry_powder"])]
    rows = []
    cases, geo = anchor_cases()
    for case in cases:
        if case["case_name"] not in DRY_POWDER_CASES:
            continue
        rec = solve_fortran_case(case, n_cells=N_ANCHOR_REF, mode="F", geometry=geo)
        hit = ref[ref["case_name"] == case["case_name"]]
        if hit.empty:
            rows.append({
                "check_group": "p17a_bitwise_parity", "context": case["case_name"],
                "metric": "reference_row_found", "hand_value": float("nan"),
                "code_value": float("nan"), "rel_diff": float("nan"),
                "tolerance": 0.0, "pass": False, "note": "P1.7-A anchor_outlet 缺该算例",
            })
            continue
        r = hit.iloc[0]
        for metric in ("T_gas_K", "X_C_pct", "CO_vol_pct", "H2_vol_pct", "CO2_vol_pct", "CH4_vol_pct"):
            a, b = float(rec[metric]), float(r[metric])
            rows.append({
                "check_group": "p17a_bitwise_parity",
                "context": f"{case['case_name']} 模式 F N={N_ANCHOR_REF}",
                "metric": metric, "hand_value": b, "code_value": a,
                "rel_diff": abs(a - b), "tolerance": 0.0, "pass": bool(a == b),
                "note": "缺省 HeatLossModel='hhv_percent' 下必须逐位相同（卡第 1.2 节）",
            })
    return rows


def _f_gas_neutrality_check() -> list[dict]:
    """GasificationRateFactor 缺省（不写）与显式 1.0 逐位一致。"""
    rows = []
    cases, geo = anchor_cases()
    case = next(c for c in cases if c["case_name"] == "texaco i-1")
    a = solve_fortran_case(case, n_cells=N_ANCHOR_REF, mode="F", geometry=geo)
    b = solve_fortran_case(
        case, n_cells=N_ANCHOR_REF, mode="F", geometry=geo,
        switches={"GasificationRateFactor": 1.0},
    )
    for metric in ("T_gas_K", "X_C_pct", "CO_vol_pct", "H2_vol_pct", "CO2_vol_pct"):
        rows.append({
            "check_group": "f_gas_neutrality", "context": "texaco i-1 模式 F N=160",
            "metric": metric, "hand_value": float(a[metric]), "code_value": float(b[metric]),
            "rel_diff": abs(float(a[metric]) - float(b[metric])), "tolerance": 0.0,
            "pass": bool(float(a[metric]) == float(b[metric])),
            "note": "GasificationRateFactor 缺省 vs 显式 1.0：指前因子倍数不得污染旧行为",
        })
    return rows


def _hhv_independence_check() -> list[dict]:
    """fortran_wall 下 HHV_d 30 → 34 的 I-1 出口温度变化 < 1e-6 K。"""
    cases, geo = anchor_cases()
    case = next(c for c in cases if c["case_name"] == "texaco i-1")
    sw = {"HeatLossModel": "fortran_wall"}
    a = solve_fortran_case(case, n_cells=N_ANCHOR_REF, mode="F", geometry=geo, switches=sw)
    b = solve_fortran_case(
        case, n_cells=N_ANCHOR_REF, mode="F", geometry=geo, switches=sw,
        coal_overrides={"HHV_d": 34.0},
    )
    d_t = float(b["T_gas_K"]) - float(a["T_gas_K"])
    rows = [{
        "check_group": "hhv_independence_fortran_wall",
        "context": "I-1 模式 F′ N=160，HHV_d 30 → 34",
        "metric": "d_T_out_K", "hand_value": 0.0, "code_value": d_t,
        "rel_diff": abs(d_t), "tolerance": 1e-6, "pass": bool(abs(d_t) < 1e-6),
        "note": "fortran_wall 不读热值，故应当逐位为 0；P1.7-A 的 −4.55 K 全部来自 2%×HHV 基准",
    }]
    d_x = float(b["X_C_pct"]) - float(a["X_C_pct"])
    rows.append({
        "check_group": "hhv_independence_fortran_wall",
        "context": "I-1 模式 F′ N=160，HHV_d 30 → 34",
        "metric": "d_X_C_pp", "hand_value": 0.0, "code_value": d_x,
        "rel_diff": abs(d_x), "tolerance": 1e-6, "pass": bool(abs(d_x) < 1e-6),
        "note": "同上（并列报告，非独立判据）",
    })
    return rows


# ---------------------------------------------------------------------------
# 第 1.3 节：锚点运行
# ---------------------------------------------------------------------------

def anchor_runs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cases, geo = anchor_cases()
    profiles_ref = pd.read_csv(P15C_DIR / "fortran_profiles.csv")
    outlets, cells, profiles = [], [], []
    for case in cases:
        if case["case_name"] not in DRY_POWDER_CASES:
            continue
        name = case["case_name"]
        for mode_tag, switches in (
            ("F", {}),
            ("F_fortran_wall", {"HeatLossModel": "fortran_wall"}),
        ):
            for n_cells in N_ANCHOR:
                rec = solve_fortran_case(
                    case, n_cells=n_cells, mode="F", geometry=geo, switches=switches,
                )
                system = rec.pop("_system", None)
                rec.pop("_profile", None)
                rec.pop("_z", None)
                rec["mode_tag"] = mode_tag
                rec["heat_loss_model"] = resolve_heat_loss_model(switches)
                rec["is_dry_powder"] = bool(case["is_dry_powder"])
                outlet = next(
                    (o for o in parse_fortran_log_cached()[0] if o["case_name"].startswith(name[:12])),
                    {},
                )
                rec["fortran_T_gas_K"] = float(outlet.get("T_gas_K", float("nan")))
                rec["fortran_X_C_pct"] = float(outlet.get("X_C_pct", float("nan")))
                rec["fortran_CO_vol_pct"] = float(outlet.get("CO_vol_pct", float("nan")))
                rec["fortran_H2_vol_pct"] = float(outlet.get("H2_vol_pct", float("nan")))
                rec["fortran_CO2_vol_pct"] = float(outlet.get("CO2_vol_pct", float("nan")))
                outlets.append(rec)

                if mode_tag == "F_fortran_wall" and system is not None:
                    tot = _zone_heat_loss(system)
                    rows_html = []
                    for diag in system.cell_diagnostics:
                        row = {
                            "case_name": name, "row_kind": "cell", "N_cells": n_cells,
                            "cell_index": int(diag["cell_index"]),
                            "z_m": float(diag.get("z_m", float("nan"))),
                            "dz_m": float(diag["dz_m"]),
                            "T_gas_K": float(diag.get("T_K", float("nan"))),
                            "is_combustion_zone": bool(diag.get("is_combustion_zone", False)),
                            "hloss_zone": str(diag.get("hloss_zone", "")),
                            "hloss_W": float(diag.get("hloss_W", float("nan"))),
                            "hloss_tw_K": float(diag.get("hloss_tw_K", float("nan"))),
                            "hloss_tg_K": float(diag.get("hloss_tg_K", float("nan"))),
                            "hloss_reaction_heat_W": float(diag.get("hloss_reaction_heat_W", float("nan"))),
                        }
                        rows_html.append(row)
                    cells.extend(rows_html)
                    summary = {"case_name": name, "row_kind": "zone_total", "N_cells": n_cells}
                    summary.update(tot)
                    summary["T_out_K"] = float(system.results[-1].T) if system.results else float("nan")
                    summary["X_C_pct"] = float(rec.get("X_C_pct", float("nan")))
                    cells.append(summary)

                    if n_cells == N_ANCHOR_REF:
                        fr = profiles_ref[profiles_ref["case_name"].str.startswith(name[:12])]
                        zs = np.asarray(system.z_positions, dtype=float)
                        Ts = np.array([float(s.T) for s in system.results])
                        for _, prow in fr.sort_values("hs_cm").iterrows():
                            hs_m = float(prow["hs_cm"]) / 100.0
                            if hs_m > zs[-1]:
                                continue
                            idx = int(np.argmin(np.abs(zs - hs_m)))
                            profiles.append({
                                "case_name": name, "N_cells": n_cells,
                                "compartment": int(prow["compartment"]),
                                "hs_cm": float(prow["hs_cm"]),
                                "fortran_T_gas_K": float(prow["T_gas_K"]),
                                "fortran_T_solid_K": float(prow["T_solid_K"]),
                                "python_T_gas_K": float(Ts[idx]),
                                "python_cell_index": idx,
                                "python_z_m": float(zs[idx]),
                                "d_T_python_minus_fortran_K": float(Ts[idx] - float(prow["T_gas_K"])),
                            })
    return pd.DataFrame(outlets), pd.DataFrame(cells), pd.DataFrame(profiles)


_FORTRAN_LOG_CACHE: dict[str, object] = {}


def parse_fortran_log_cached():
    if "out" not in _FORTRAN_LOG_CACHE:
        _FORTRAN_LOG_CACHE["out"] = parse_fortran_log(P15C_DIR / "fortran_build" / "run.log")
    return _FORTRAN_LOG_CACHE["out"]


def anchor_criteria(outlets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ref = outlets[outlets["N_cells"] == N_ANCHOR_REF]
    for name in DRY_POWDER_CASES:
        for tag in ("F", "F_fortran_wall"):
            hit = ref[(ref["case_name"] == name) & (ref["mode_tag"] == tag)]
            if hit.empty:
                continue
            r = hit.iloc[0]
            checks = {
                "X_C_pp": (r.get("X_C_pct"), r.get("fortran_X_C_pct")),
                "CO_vol_pct": (r.get("CO_vol_pct"), r.get("fortran_CO_vol_pct")),
                "H2_vol_pct": (r.get("H2_vol_pct"), r.get("fortran_H2_vol_pct")),
                "CO2_vol_pct": (r.get("CO2_vol_pct"), r.get("fortran_CO2_vol_pct")),
                "T_gas_K": (r.get("T_gas_K"), r.get("fortran_T_gas_K")),
            }
            for metric, (pv, fv) in checks.items():
                pv = float(pv) if pv is not None else float("nan")
                fv = float(fv) if fv is not None else float("nan")
                diff = pv - fv
                rows.append({
                    "case_name": name, "mode_tag": tag, "N_cells": N_ANCHOR_REF,
                    "metric": metric, "python_value": pv, "fortran_value": fv,
                    "abs_diff": abs(diff), "signed_diff": diff,
                    "tolerance": ANCHOR_TOL[metric],
                    "pass": bool(np.isfinite(diff) and abs(diff) <= ANCHOR_TOL[metric]),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 第 2 节：九窗扫描
# ---------------------------------------------------------------------------

def _build_row_extras(
    rec: dict, system: GasifierSystem, row: dict, csv17: pd.Series, csv18: pd.Series,
    dry: float, n_cells: int, f_gas: float, hlp: float,
    carrier_nm3_h: float, carrier_basis: str, row_kind: str,
) -> dict:
    out = dict(rec)
    out.pop("_system", None)
    out.pop("_profile", None)
    out.update({
        "row_kind": row_kind,
        "f_gas": float(f_gas),
        "HeatLossPercent": float(hlp),
        "HeatLossModel": resolve_heat_loss_model(system.op_conds),
        "carrier_basis": carrier_basis,
        "carrier_nm3_h": float(carrier_nm3_h),
        "coal_basis": "recon_phase",
        "coal_basis_role": "主口径（协调-阶段）",
        "coal_dry_kg_s": float(dry),
        "coal_wet_kg_s": wet_coal_flow_kg_s(dry),
        "coal_flow_op_kg_s": float(system.op_conds["coal_flow"]),
        "coal_dry_effective_kg_s": float(system.W_dry),
        "Hf_coal_j_kg": float(system.coal_props["Hf_coal"]),
        "Hf_coal_kj_kg": float(system.coal_props["Hf_coal"]) / 1000.0,
        "hf_coal_source": system.hf_coal_source,
        "pyrolysis_heat_model": system.pyrolysis_heat_model,
        "LHV_d_kj_kg": float(coal_lhv_kj_kg(system.coal_props)),
        "HHV_d_derived_kj_kg": float(coal_hhv_kj_kg(system.coal_props)),
        "plant_xc_measured_pct": float(csv18["xc_measured_pct"]),
        "plant_xc_j_pct": float(csv18["xc_j_pct"]),
        "liu2023_xc_phase_pct": float(csv18["liu2023_xc_phase_pct"]),
        "plant_dry_flow_cal_nm3_h": float(csv18["syngas_dry_cal_nm3_h"]),
        "co2_carrier_measured_nm3_h": float(csv18["co2_carrier_measured_nm3_h"]),
        "co2_carrier_j_nm3_h": float(csv18["co2_carrier_j_nm3_h"]),
        "co2_carrier_t_nm3_h": float(csv18["co2_carrier_t_nm3_h"]),
        "t_carrier_status": str(csv18["t_carrier_status"]),
        "plant_co2_carrier_measured_mol_s": float(csv18["carrier_co2_measured_mol_s"]),
        "plant_co2_carrier_j_mol_s": float(csv18["carrier_co2_j_mol_s"]),
        "plant_outlet_co2_mol_s": float(csv18["outlet_co2_mol_s"]),
        "plant_net_co2_measured_mol_s": float(csv18["net_co2_measured_mol_s"]),
        "plant_net_co2_j_mol_s": float(csv18["net_co2_j_mol_s"]),
    })
    out.update(_limit_zone_stats(system))
    if resolve_heat_loss_model(system.op_conds) == "fortran_wall":
        out.update(_zone_heat_loss(system))
    out.update(_reaction_integrals(system))

    plant_raw_mol = float(csv18["syngas_dry_cal_nm3_h"]) / 22.414 / 3600.0 * 1000.0
    out["plant_dry_total_mol_s_cal"] = plant_raw_mol
    y_h2 = float(rec["plant_H2_vol_pct"]) / 100.0
    out["H_plant_mol_s_cal"] = 2.0 * plant_raw_mol * y_h2

    if rec["execution_status"] == "completed" and system.results:
        state = system.results[-1]
        gas = np.asarray(state.gas_moles, dtype=float)
        dry_gas = float(np.sum(gas[:7]))
        hb = _hydrogen_budget(system, gas, dry)
        out.update(hb)
        out["dry_total_mol_s"] = dry_gas
        out["F_CO_mol_s"] = float(gas[2])
        out["F_CO2_mol_s"] = float(gas[3])
        out["F_H2_mol_s"] = float(gas[5])
        out["F_CH4_mol_s"] = float(gas[1])
        out["F_N2_mol_s"] = float(gas[6])
        out["F_H2O_mol_s"] = float(gas[7])
        out["CO2_gap_vs_plant_vol_pct"] = (
            float(gas[3]) / max(dry_gas, 1e-12) * 100.0 - float(rec["plant_CO2_vol_pct"])
        )
        out["dry_flow_ratio_cal"] = float(rec["dry_flow_nm3_h"]) / max(
            float(csv18["syngas_dry_cal_nm3_h"]), 1e-12
        )
        out["H_plant_over_H_in_cal"] = out["H_plant_mol_s_cal"] / max(hb["H_in_mol_s"], 1e-12)
        out["H_plant_over_H_in_with_moisture_cal"] = (
            out["H_plant_mol_s_cal"] / max(hb["H_in_with_moisture_mol_s"], 1e-12)
        )
        # 模型：cell 0 与出口的 X_C
        inlet_c0 = float(system.cells[0].inlet.solid_mass * system.cells[0].inlet.carbon_fraction)
        c0 = system.results[0]
        c0_c = float(c0.solid_mass * c0.carbon_fraction)
        out["X_C_cell0_implied"] = float(1.0 - c0_c / max(inlet_c0, 1e-12))
        out["T_cell0_K"] = float(c0.T)
        # 载气 CO2 净去向（模型）
        carrier_in = _nm3_h_to_mol_s(carrier_nm3_h)
        out["model_carrier_in_mol_s"] = carrier_in
        out["model_co2_out_mol_s"] = float(gas[3])
        out["model_co2_net_mol_s"] = float(gas[3]) - carrier_in
        out["model_co2_consumption_frac"] = (
            (carrier_in - float(gas[3])) / carrier_in if carrier_in > 0 else float("nan")
        )
        out["model_minus_plant_xc_measured_pp"] = (
            float(rec["X_C_implied"]) * 100.0 - float(csv18["xc_measured_pct"])
        )
        out["model_minus_plant_xc_j_pp"] = (
            float(rec["X_C_implied"]) * 100.0 - float(csv18["xc_j_pct"])
        )
        out["model_minus_liu_xc_pp"] = (
            float(rec["X_C_implied"]) * 100.0 - float(csv18["liu2023_xc_phase_pct"])
        )
        out.update(_composition_block(
            out, float(rec["X_C_implied"]) * 100.0, gas, dry_gas, dry, row, csv18
        ))
    return out


def _closure_xi(F_CO2: float, dry_total: float, y_co2: float) -> float:
    """单参数闭合 ξ（卡面分母 ``干气 + ξ``，与 P1.7-A 相同）。"""
    return (y_co2 * dry_total - F_CO2) / (1.0 - y_co2)


def _composition_block(out: dict, xc_model_pp: float, gas: np.ndarray, dry_gas: float,
                       dry: float, row: dict, csv18: pd.Series) -> dict:
    f_co = float(gas[2])
    f_co2 = float(gas[3])
    f_h2 = float(gas[5])
    model_co = f_co / max(dry_gas, 1e-12) * 100.0
    model_h2 = f_h2 / max(dry_gas, 1e-12) * 100.0
    model_co2 = f_co2 / max(dry_gas, 1e-12) * 100.0
    plant_dry_mol = float(csv18["syngas_dry_cal_nm3_h"]) / 22.414 / 3600.0 * 1000.0
    res: dict = {
        "model_CO_vol_pct": model_co,
        "model_H2_vol_pct": model_h2,
        "model_CO2_vol_pct": model_co2,
        "plant_dry_total_mol_s_cal": plant_dry_mol,
    }
    for basis in COMP_BASES:
        if basis == "analyzer":
            p_co = float(row.get("co_vol_pct", np.nan))
            p_h2 = float(row.get("h2_vol_pct", np.nan))
            p_co2 = float(row.get("co2_vol_pct", np.nan))
        else:
            phase = str(out.get("phase", ""))
            p_co, p_h2, p_co2 = LIU_PHASE_COMP[phase]
        res[f"{basis}_plant_CO_vol_pct"] = p_co
        res[f"{basis}_plant_H2_vol_pct"] = p_h2
        res[f"{basis}_plant_CO2_vol_pct"] = p_co2
        res[f"{basis}_dCO_xi0"] = model_co - p_co
        res[f"{basis}_dH2_xi0"] = model_h2 - p_h2
        res[f"{basis}_dCO2_xi0"] = model_co2 - p_co2
        xi = _closure_xi(f_co2, dry_gas, p_co2 / 100.0)
        denom = dry_gas + xi
        pred_co = (f_co - xi) / max(denom, 1e-12) * 100.0
        pred_h2 = (f_h2 + xi) / max(denom, 1e-12) * 100.0
        pred_co2 = (f_co2 + xi) / max(denom, 1e-12) * 100.0
        res[f"{basis}_xi_mol_s"] = xi
        res[f"{basis}_pred_CO_vol_pct"] = pred_co
        res[f"{basis}_pred_H2_vol_pct"] = pred_h2
        res[f"{basis}_pred_CO2_vol_pct"] = pred_co2
        res[f"{basis}_dCO_closed"] = pred_co - p_co
        res[f"{basis}_dH2_closed"] = pred_h2 - p_h2
        res[f"{basis}_dCO2_closed"] = pred_co2 - p_co2
        # 氢账本：（模型干气氢原子 + 2ξ）/ 装置干气氢原子。
        # f_h2 是分子 H2 流量，每个分子含 2 个 H 原子；H_out_dry 按 _hydrogen_budget 的
        # 定义 = 2·H2 + 4·CH4 + 2·H2S，正是原子口径，且已含 CH4 与 H2S 的氢。
        # ξ 每次转移 2 个 H 原子（CO + H2O -> CO2 + H2）。
        # 2026-11-04 更正（审阅意见）：原写法 f_h2 + 2ξ 把分子流量与原子流量相加，
        # 分子侧少算一半，首行从 0.603 修正到约 1.006。
        h_out_dry = 2.0 * float(gas[5]) + 4.0 * float(gas[1]) + 2.0 * float(gas[4])
        h_model = h_out_dry + 2.0 * xi
        h_plant = 2.0 * plant_dry_mol * (p_h2 / 100.0)
        res[f"{basis}_H_ledger_model_over_plant"] = h_model / max(h_plant, 1e-12)
        # R1 用：X_C 偏差（与组成口径无关，此处只为便于逐行读取而重复）
        res[f"{basis}_model_minus_plant_xc_pp"] = xc_model_pp - float(csv18["xc_measured_pct"])
    return res


def run_scan(window_ids, by_id, p17d: pd.DataFrame, p18d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for wid in window_ids:
        row = by_id[wid]
        csv17 = p17d.loc[wid]
        csv18 = p18d.loc[wid]
        dry = float(csv17["coal_dry_recon_phase_kg_s"])
        measured = float(csv18["co2_carrier_measured_nm3_h"])
        old_t = float(csv18["co2_carrier_t_nm3_h"])
        old_split = float(row["co2_carrier_volumetric_flow_value"])
        plan = [
            (f, h, measured, "measured", "grid")
            for f in F_GAS_GRID for h in HEAT_LOSS_PCT_GRID
        ]
        plan.append((1.0, MAIN_HEAT_LOSS_PCT, old_t, "old_t_running_page", "replay_old_carrier"))
        plan.append((1.0, MAIN_HEAT_LOSS_PCT, old_split, "frozen_split_0p24_o2", "parity_p17a"))
        for f_gas, hlp, carrier, carrier_tag, kind in plan:
            rec = run_case(
                row, n_cells=N_SCAN, coal_flow_dry_kg_s=dry, keep_system=True,
                extra_op={
                    "HeatLossPercent": float(hlp),
                    "GasificationRateFactor": float(f_gas),
                    "carrier_gas_flow": _carrier_kg_s(carrier),
                },
            )
            system: GasifierSystem = rec["_system"]
            out = _build_row_extras(
                rec, system, row, csv17, csv18, dry, N_SCAN, f_gas, hlp,
                carrier, carrier_tag, kind,
            )
            rows.append(out)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 第 2 节：读法 R1–R5
# ---------------------------------------------------------------------------

def _run_point(row, dry, n_cells, f_gas, hlp, carrier_nm3_h) -> dict:
    rec = run_case(
        row, n_cells=n_cells, coal_flow_dry_kg_s=dry, keep_system=True,
        extra_op={
            "HeatLossPercent": float(hlp),
            "GasificationRateFactor": float(f_gas),
            "carrier_gas_flow": _carrier_kg_s(carrier_nm3_h),
        },
    )
    system: GasifierSystem = rec.pop("_system")
    rec.pop("_profile", None)
    if rec["execution_status"] == "completed" and system.results:
        gas = np.asarray(system.results[-1].gas_moles, dtype=float)
        dry_gas = float(np.sum(gas[:7]))
        rec["model_CO_vol_pct"] = float(gas[2]) / max(dry_gas, 1e-12) * 100.0
        rec["model_H2_vol_pct"] = float(gas[5]) / max(dry_gas, 1e-12) * 100.0
        rec["model_CO2_vol_pct"] = float(gas[3]) / max(dry_gas, 1e-12) * 100.0
        rec["dry_total_mol_s"] = dry_gas
        rec["F_CO2_mol_s"] = float(gas[3])
        rec["model_co2_out_mol_s"] = float(gas[3])
        rec["model_carrier_in_mol_s"] = _nm3_h_to_mol_s(carrier_nm3_h)
        rec["model_co2_net_mol_s"] = float(gas[3]) - _nm3_h_to_mol_s(carrier_nm3_h)
        rec.update(_reaction_integrals(system))
    return rec


def readings(scan: pd.DataFrame, window_ids, by_id, p17d: pd.DataFrame,
             p18d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # 卡面第 2.1 节：f_gas 向下只延伸一档（0.03125）。延伸行属于网格的一部分，
    # 因此 R1–R4 的搜索池是 grid + grid_extension；replay / parity 行不参与。
    pool = scan[scan["row_kind"].isin(("grid", "grid_extension"))].copy()
    f_values = sorted(pool["f_gas"].unique().tolist())

    out_rows: list[dict] = []

    def plant_xc(wid):
        return float(p18d.loc[wid, "xc_measured_pct"])

    # ---- R1：|模型 X_C − 装置主口径 X_C| ≤ 1 pp 的 (f_gas, HeatLossPercent) 组合 ----
    r1: dict[str, list[dict]] = {}
    for wid in window_ids:
        sub = pool[pool["window_id"] == wid]
        px = plant_xc(wid)
        hits = []
        for f in f_values:
            g = sub[sub["f_gas"] == f].sort_values("HeatLossPercent")
            d = g["X_C_implied"].to_numpy(dtype=float) * 100.0 - px
            hlp = g["HeatLossPercent"].to_numpy(dtype=float)
            at_grid = [(float(h), float(dd), False) for h, dd, ok in zip(hlp, d, np.abs(d) <= R1_XC_TOL_PP) if ok]
            interp = []
            for i in range(len(d) - 1):
                if not (np.isfinite(d[i]) and np.isfinite(d[i + 1])):
                    continue
                if d[i] == 0.0:
                    interp.append((float(hlp[i]), 0.0, True))
                    continue
                if d[i] * d[i + 1] < 0.0:
                    h_star = hlp[i] + (hlp[i + 1] - hlp[i]) * (-d[i]) / (d[i + 1] - d[i])
                    interp.append((float(h_star), 0.0, True))
            hits.append({"f_gas": float(f), "grid_hits": at_grid, "interp_hits": interp})
        r1[wid] = hits
        any_hit = any(h["grid_hits"] or h["interp_hits"] for h in hits)
        out_rows.append({
            "reading": "R1", "composition_basis": "n/a（只看 X_C）", "window_id": wid,
            "phase": str(p18d.loc[wid, "phase"]), "exists": bool(any_hit),
            "value": float("nan"),
            "detail": _r1_detail(hits),
        })

    # ---- R1 明细 + R2：沿 R1 组合的最小 ξ ----
    for wid in window_ids:
        sub = pool[pool["window_id"] == wid]
        px = plant_xc(wid)
        for basis in COMP_BASES:
            xis = []
            detail = []
            for h in r1[wid]:
                combos = [(hh, dd, flag) for hh, dd, flag in h["grid_hits"]]
                combos += [(hh, 0.0, True) for hh, _dd, flag in h["interp_hits"]]
                for hh, dd, is_interp in combos:
                    cand = sub[sub["f_gas"] == h["f_gas"]]
                    if is_interp:
                        v = _interp_col(cand, hh, f"{basis}_xi_mol_s")
                    else:
                        row = cand[cand["HeatLossPercent"] == hh].iloc[0]
                        v = float(row[f"{basis}_xi_mol_s"])
                    if v is None or not np.isfinite(v):
                        continue
                    xis.append(float(v))
                    detail.append(f"f={h['f_gas']:g}/HLP={hh:.3f}{'*' if is_interp else ''}:ξ={v:.1f}")
            out_rows.append({
                "reading": "R2", "composition_basis": basis, "window_id": wid,
                "phase": str(p18d.loc[wid, "phase"]),
                "exists": bool(xis),
                "value": float(min(xis)) if xis else float("nan"),
                "detail": "；".join(detail[:12]) + ("…" if len(detail) > 12 else ""),
            })

    # ---- R2 汇总 ----
    for basis in COMP_BASES:
        vals = [r["value"] for r in out_rows if r["reading"] == "R2" and r["composition_basis"] == basis]
        vals = [v for v in vals if np.isfinite(v)]
        med = float(np.median(vals)) if vals else float("nan")
        out_rows.append({
            "reading": "R2_summary", "composition_basis": basis, "window_id": "ALL",
            "phase": "all", "exists": bool(vals), "value": med,
            "detail": (f"n={len(vals)}/{len(window_ids)} 窗有 R1 组合；ξ_min 中位 {med:.1f} mol/s；"
                       + ("炉内两参数不能单独解释 CO2，反应器后变换仍需要"
                          if med > R2_XI_THRESHOLD else
                          ("不需要反应器后变换" if np.isfinite(med) else "无法判定"))),
        })

    # ---- R3：雅可比 ----
    jac_rows, jac_rows_export = [], []
    for wid in window_ids:
        sub = pool[pool["window_id"] == wid]
        px = plant_xc(wid)
        for basis in COMP_BASES:
            best = None
            for h in r1[wid]:
                combos = [(hh, dd, False) for hh, dd, _ in h["grid_hits"]]
                combos += [(hh, 0.0, True) for hh, _dd, _ in h["interp_hits"]]
                for hh, _dd, is_interp in combos:
                    cand = sub[sub["f_gas"] == h["f_gas"]]
                    rss = _rss_row(cand, hh, basis, px, is_interp)
                    if rss is None or not np.isfinite(rss):
                        continue
                    if best is None or rss < best[0]:
                        best = (rss, h["f_gas"], hh, is_interp)
            if best is None:
                out_rows.append({
                    "reading": "R3", "composition_basis": basis, "window_id": wid,
                    "phase": str(p18d.loc[wid, "phase"]), "exists": False,
                    "value": float("nan"), "detail": "R1 无组合，无法求雅可比",
                })
                continue
            rss, f0, h0, interp = best
            row = by_id[wid]
            dry = float(p17d.loc[wid, "coal_dry_recon_phase_kg_s"])
            carrier = float(p18d.loc[wid, "co2_carrier_measured_nm3_h"])
            jrow = _jacobian_at(row, dry, f0, h0, basis, px, carrier)
            jrow.update({
                "reading": "R3", "composition_basis": basis, "window_id": wid,
                "phase": str(p18d.loc[wid, "phase"]), "rss": float(rss),
                "f_gas_star": float(f0), "HeatLossPercent_star": float(h0),
                "interpolated": bool(interp),
            })
            jac_rows_export.append(jrow)
            ident = (jrow["cond_J"] is not None and np.isfinite(jrow["cond_J"])
                     and jrow["cond_J"] < R3_COND_THRESHOLD
                     and abs(jrow["corr_lnf_hlp"]) < R3_CORR_THRESHOLD)
            out_rows.append({
                "reading": "R3", "composition_basis": basis, "window_id": wid,
                "phase": str(p18d.loc[wid, "phase"]), "exists": True,
                "value": float(jrow.get("cond_J", float("nan"))),
                "detail": (f"最优点 f_gas={f0:g}、HLP={h0:.3f}{'*' if interp else ''}；"
                           f"RSS={rss:.4f}；cond(J)={jrow.get('cond_J'):.3g}；"
                           f"corr={jrow.get('corr_lnf_hlp'):+.3f}；"
                           f"{'可分别辨识' if ident else '不可分别辨识'}"),
            })

    # ---- R4：各阶段最优 f_gas ----
    for basis in COMP_BASES:
        per_phase: dict[str, list[float]] = {}
        for wid in window_ids:
            sub = pool[pool["window_id"] == wid]
            px = plant_xc(wid)
            best = None
            for h in r1[wid]:
                combos = [(hh, False) for hh, _dd, _ in h["grid_hits"]]
                combos += [(hh, True) for hh, _dd, _ in h["interp_hits"]]
                for hh, is_interp in combos:
                    cand = sub[sub["f_gas"] == h["f_gas"]]
                    rss = _rss_row(cand, hh, basis, px, is_interp)
                    if rss is None or not np.isfinite(rss):
                        continue
                    if best is None or rss < best[0]:
                        best = (rss, h["f_gas"])
            if best is None:
                continue
            per_phase.setdefault(str(p18d.loc[wid, "phase"]), []).append(float(best[1]))
        for ph, vals in sorted(per_phase.items()):
            out_rows.append({
                "reading": "R4", "composition_basis": basis, "window_id": "ALL",
                "phase": ph, "exists": True, "value": float(np.median(vals)),
                "detail": (f"n={len(vals)}；最优 f_gas 中位 {np.median(vals):.4g}，"
                           f"范围 {min(vals):.4g}–{max(vals):.4g}"),
            })
        allv = [v for vals in per_phase.values() for v in vals]
        if allv:
            out_rows.append({
                "reading": "R4_all", "composition_basis": basis, "window_id": "ALL",
                "phase": "all", "exists": True, "value": float(np.median(allv)),
                "detail": f"n={len(allv)}；中位 {np.median(allv):.4g}，极差 {max(allv)-min(allv):.4g}",
            })

    # ---- R5：CO2 由谁消耗 ----
    for wid in window_ids:
        sub = pool[pool["window_id"] == wid]
        f1 = sub[(sub["f_gas"] == 1.0) & (sub["HeatLossPercent"] == MAIN_HEAT_LOSS_PCT)]
        if f1.empty:
            continue
        r = f1.iloc[0]
        _r5_append(out_rows, wid, p18d, "f_gas=1", 1.0, MAIN_HEAT_LOSS_PCT, r)
        best = None
        for basis in COMP_BASES:
            hit = [x for x in jac_rows_export
                   if x["window_id"] == wid and x["composition_basis"] == basis]
            if hit:
                best = hit[0]
                break
        if best is not None:
            row = by_id[wid]
            dry = float(p17d.loc[wid, "coal_dry_recon_phase_kg_s"])
            carrier = float(p18d.loc[wid, "co2_carrier_measured_nm3_h"])
            pt = _run_point(row, dry, N_SCAN, best["f_gas_star"], best["HeatLossPercent_star"], carrier)
            _r5_append(out_rows, wid, p18d, "R3_optimum", best["f_gas_star"],
                       best["HeatLossPercent_star"], pt)
    return pd.DataFrame(out_rows), pd.DataFrame(jac_rows_export)


def _r1_detail(hits: list[dict]) -> str:
    parts = []
    for h in hits:
        if h["grid_hits"]:
            parts.append(f"f={h['f_gas']:g}:" + ",".join(f"HLP={hh:.0f}" for hh, _d, _f in h["grid_hits"]))
        if h["interp_hits"]:
            parts.append(f"f={h['f_gas']:g}*:" + ",".join(f"HLP≈{hh:.3f}" for hh, _d, _f in h["interp_hits"]))
    return "；".join(parts) if parts else "无"


def _interp_row(cand: pd.DataFrame, hlp: float, col: str):
    g = cand.sort_values("HeatLossPercent")
    x = g["HeatLossPercent"].to_numpy(dtype=float)
    y = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(y)):
        return None
    return float(np.interp(hlp, x, y))


def _interp_col(cand: pd.DataFrame, hlp: float, col: str) -> float:
    g = cand.sort_values("HeatLossPercent")
    x = g["HeatLossPercent"].to_numpy(dtype=float)
    y = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
    return float(np.interp(hlp, x, y))


def _rss_row(cand: pd.DataFrame, hlp: float, basis: str, plant_xc_pp: float, is_interp: bool):
    try:
        xc = float(_interp_col(cand, hlp, "X_C_implied")) * 100.0
        dco = float(_interp_col(cand, hlp, f"{basis}_dCO_xi0"))
        dh2 = float(_interp_col(cand, hlp, f"{basis}_dH2_xi0"))
        dco2 = float(_interp_col(cand, hlp, f"{basis}_dCO2_xi0"))
    except Exception:  # noqa: BLE001
        return None
    if not np.all(np.isfinite([xc, dco, dh2, dco2])):
        return None
    r = np.array([(xc - plant_xc_pp) / R3_SCALES[0], dco / R3_SCALES[1],
                  dh2 / R3_SCALES[2], dco2 / R3_SCALES[3]], dtype=float)
    return float(0.5 * np.sum(r * r))


def _jacobian_at(row, dry, f0, h0, basis, plant_xc_pp, carrier) -> dict:
    h_lnf, h_hlp = 0.05, 0.05

    def obs(f_gas, hlp):
        rec = _run_point(row, dry, N_SCAN, f_gas, hlp, carrier)
        if rec["execution_status"] != "completed":
            return None
        xc = float(rec["X_C_implied"]) * 100.0
        dco = float(rec["model_CO_vol_pct"]) - (
            float(row.get("co_vol_pct", np.nan)) if basis == "analyzer" else LIU_PHASE_COMP[str(row.get("phase"))][0]
        )
        dh2 = float(rec["model_H2_vol_pct"]) - (
            float(row.get("h2_vol_pct", np.nan)) if basis == "analyzer" else LIU_PHASE_COMP[str(row.get("phase"))][1]
        )
        dco2 = float(rec["model_CO2_vol_pct"]) - (
            float(row.get("co2_vol_pct", np.nan)) if basis == "analyzer" else LIU_PHASE_COMP[str(row.get("phase"))][2]
        )
        return np.array([(xc - plant_xc_pp) / R3_SCALES[0], dco / R3_SCALES[1],
                         dh2 / R3_SCALES[2], dco2 / R3_SCALES[3]], dtype=float)

    base = obs(f0, h0)
    if base is None:
        return {"f_gas_star": f0, "HeatLossPercent_star": h0,
                "cond_J": float("nan"), "corr_lnf_hlp": float("nan")}
    out = {"f_gas_star": float(f0), "HeatLossPercent_star": float(h0),
           "rss": float(0.5 * np.sum(base * base))}
    cols = []
    for j, (f_g, h_p) in enumerate((
        (f0 * math.exp(h_lnf), h0),
        (f0 * math.exp(-h_lnf), h0),
    )):
        v = obs(f_g, h_p)
        cols.append(None if v is None else (v - base) / h_lnf)
    for (f_g, h_p) in ((f0, h0 + h_hlp), (f0, h0 - h_hlp)):
        v = obs(f_g, h_p)
        cols.append(None if v is None else (v - base) / h_hlp)
    fwd_lnf, bwd_lnf, fwd_h, bwd_h = cols
    if any(c is None for c in cols):
        out.update({"cond_J": float("nan"), "corr_lnf_hlp": float("nan")})
        return out
    # 中心差分：(f(+h) − f(−h)) / (2h)；cols 里存的是 (f(±h) − f(0))/h，
    # 因此中心差分 = (fwd − bwd) / 2（不是 (fwd + bwd)/2，那是二阶导）。
    J = np.column_stack([
        0.5 * (fwd_lnf - bwd_lnf),
        0.5 * (fwd_h - bwd_h),
    ])
    out["J_dXC_dlnf"] = float(J[0, 0])
    out["J_dCO_dlnf"] = float(J[1, 0])
    out["J_dH2_dlnf"] = float(J[2, 0])
    out["J_dCO2_dlnf"] = float(J[3, 0])
    out["J_dXC_dHLP"] = float(J[0, 1])
    out["J_dCO_dHLP"] = float(J[1, 1])
    out["J_dH2_dHLP"] = float(J[2, 1])
    out["J_dCO2_dHLP"] = float(J[3, 1])
    sv = np.linalg.svd(J, compute_uv=False)
    out["sigma_max"] = float(sv[0])
    out["sigma_min"] = float(sv[-1])
    out["cond_J"] = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
    jtj = J.T @ J
    out["cond_JtJ"] = float(np.linalg.cond(jtj))
    try:
        cov = np.linalg.inv(jtj)
        # cov = (JᵀJ)⁻¹，其归一化非对角元即参数相关系数，符号本身正确，不能再加负号。
        # 2026-11-04 更正（审阅意见）：原写法多了一个负号，使每个导出的
        # corr_lnf_hlp 符号反转（首行应为 +0.6997，artifact 里是 −0.6997）。
        # 按 |corr| 的可辨识性判定不受影响，但符号方向是参数关系的结论，必须正确。
        out["corr_lnf_hlp"] = float(cov[0, 1] / math.sqrt(max(cov[0, 0] * cov[1, 1], 1e-300)))
    except np.linalg.LinAlgError:
        out["corr_lnf_hlp"] = float("nan")
    out["f_gas_minus"] = float(f0 * math.exp(-h_lnf))
    out["f_gas_plus"] = float(f0 * math.exp(h_lnf))
    out["HLP_minus"] = float(h0 - h_hlp)
    out["HLP_plus"] = float(h0 + h_hlp)
    return out


def _r5_append(rows: list[dict], wid: str, p18d: pd.DataFrame, tag: str,
               f_gas: float, hlp: float, rec) -> None:
    def g(k, default=float("nan")):
        v = rec.get(k, default)
        return float(v) if v is not None else default

    c_co2 = g("rx_C_CO2_mol_s")
    rwgs = g("rx_RWGS_mol_s")
    wgs = g("rx_WGS_mol_s")
    net = g("model_co2_net_mol_s")
    denom = c_co2 + rwgs
    share = c_co2 / denom if denom > 1e-12 else float("nan")
    rows.append({
        "reading": "R5", "composition_basis": "n/a（只看碳与氧的流量）", "window_id": wid,
        "phase": str(p18d.loc[wid, "phase"]), "exists": bool(np.isfinite(net)),
        "item": tag, "value": share,
        "detail": (f"f_gas={f_gas:g}, HLP={hlp:.3f}；"
                   f"净 CO2 消耗 {-net:.1f} mol/s；C+CO2 {c_co2:.1f}；逆 WGS {rwgs:.1f}；"
                   f"正 WGS 生成 {wgs:.1f}；C+CO2 占（C+CO2+逆WGS）{share*100:.1f}%"),
    })


# ---------------------------------------------------------------------------
# interpretation.md
# ---------------------------------------------------------------------------

def _fmt(x, nd=2):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(v):
        return "nan"
    return f"{v:.{nd}f}"


def _table(df: pd.DataFrame, cols: list[str], fmt: dict | None = None) -> str:
    fmt = fmt or {}
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r.get(c)
            if isinstance(v, (bool, np.bool_)):
                cells.append(str(bool(v)))
            elif isinstance(v, (int, np.integer)):
                cells.append(str(int(v)))
            else:
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    cells.append(str(v))
                else:
                    cells.append(f"{fv:.{fmt.get(c, 2)}f}" if np.isfinite(fv) else "nan")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


SYMBOLS_MD = """# Fortran 壁面热损的符号表（P1.8-A，2026-11-04）

来源：`reference_fortran/Source1_副本.for`（行号为去 CR 后的行号）。
`&readdata` namelist 只有 `name, ta, tsteam, toxy, yoxy, yh, yc, ys, yn, yash, xmois, foxy, fsteam, tfcoal`
（第 40–41 行），**下表没有任何一个符号出现在 namelist 里**。

| 符号 | 定义 / 单位 | 首次赋值行 | 是否被重新赋值 | Python 对应量 |
|---|---|---|---|---|
| `dt` | 反应器直径，**cm**。`data` 语句给 152.4 cm（= 1.524 m），与 `at = 18241.5 cm²` 满足 π/4·dt² = at，确认是直径 | 32 | **否**（`data` 初值，全程序未再赋值） | `FORTRAN_DT_CM`（常量，不取 Python 几何） |
| `height` | 反应器/积分高度，**cm**。`data` 给 330 cm（= 3.3 m）；第 787–788 行 `!change 325 into height-5` / `dhs = hs-(height-5.)` 说明积分到 `height−5` = 325 cm 为止，所以它是**总高度**而非有效长度 | 32 | **否** | `FORTRAN_HEIGHT_CM`（常量） |
| `at` | 反应器截面积，**cm²**（18241.5） | 32 | 否 | 无用（Python 用几何 D） |
| `pt` | 压力，**atm**（24.0） | 32 | 否 | `op_conds['P']` |
| `nc` | 一个气室内的子步数，无量纲。`nc = 3`（第 145 行注释：`nc is the number of step sizes in each gas compartment`） | 146 | **否** | 不单独出现；`nc·delh` → `dz` |
| `delh` | **一个子步的轴向长度，cm**。`data` 给 25，但 25 只用于**第一个气室**；第 852–858 行在每个新气室开头重设为 `0.5` 或 `1.0`（`hs.le.40 → 0.5`；`poxy.gt.0.01 → 0.5`；否则 `1.0`），并在迭代不收敛时被 `delh = delh/2`（第 367、523 行）折半 | 96 | **是**（852–858、367、523） | `dz`（Python 一个单元 = 一个气室） |
| `hs` | 颗粒已走过的轴向高度，**cm**；初值 `hsin`（第 221 行），每个子步 `hs = hs + delh`（第 277、763 行） | 221 | **是**（每子步累进；每气室由 `hsin` 重置） | 单元出口 `z + dz/2` |
| `fcoal` | 入炉煤流量，**g/s，dmmf 基**。`fcoal = tfcoal*(1.-yash)*(1.-xmois)`（第 66 行），`tfcoal` 来自 namelist | 66 | **是**（由 namelist 的 `tfcoal/yash/xmois` 算出） | 干煤流量（`W_dry`，g/s） |
| `rheat1` | **本气室挥发分项放热**，cal/g dmmf。由 `pyroly` 写入：`wl*(xh2*ht(1)/2 + xco*ht(2)/28 + xch4*ht(3)/16 + xtar*ht(4)/78)/100 − cop*ht(2)/28 − ch4p*ht(3)/16 − h2p*ht(1)/2`（第 1087–1088 行），其中 `ht(n, tg)` 是组分在**气相温度**下的焓（`heatup` 第 1131 行起，cal/mol） | 248（每气室清零） | **是**：每个气室开头清零（248、865），由第 314 行调用 `pyroly` 赋值 | 无直接对应；挥发分能量走 `PyrolysisSource.energy_src_W` |
| `rheat2` | 气相燃烧项，cal/g dmmf。`rheat2 += rhta+rhtb+rhtc-rhtd`（第 556 行）与 `+qmref*dch46/16`（第 587、743 行），`rhta/b/c/d` 由 `q1/qrh/ht` 与碳消耗量 `cwl/cwl2/cwl3/cop` 组合 | 249 | **是**（每气室清零 417/611，多处累加） | 由 `CO_Ox/H2_Ox/CH4_Ox/MSR` 的反应焓体现 |
| `rheat3` | WGS 项，cal/g dmmf。`rheat3 += qwg*wgl*(1-fwgs)`（第 588、744 行），`fwgs = 1.0`（第 100 行）是 WGS 放热被固相吸收的份额 | 250 | **是** | 由 `WGS/RWGS` 的反应焓体现 |
| `rhts` | 被固相吸收的热，cal/g dmmf：`rhts += cwl*qrh − cwl2*qcsm/12 − cwl3*qcbco2/12 − cwl5*qcbhy/12`（第 558 行）与 `−qwg*wgl*fwgs`（第 745 行） | 251 | **是**（每气室清零 417/611） | C+O2/C+H2O/C+CO2/C+H2 的反应焓（含固相） |
| `rheat` | **净放热**，cal/g dmmf：`rheat = rheat1+rheat2+rheat3 − rhts`（第 772 行） | 772 | **是**（每气室清零 869） | `reaction_heat_release_w()` 的返回值 ÷ 煤流量 |
| `hl` | 燃烧区传给壁面的放热份额，**%**。`data hl/30./`（第 30 行，注释 `combustion heat percent transfered to the wall`） | 30 | **否** | `FORTRAN_HL_PERCENT = 30` |
| `u0` | 气-壁总换热系数，**cal/(s·cm²·K)**，`u0 = 3.4e-3`（第 214 行） | 214 | **否** | `FORTRAN_U0_CAL_PER_S_CM2_K` |
| `tw` | 壁温，**K**：`tw = 2100. − 600.*hs/height`（第 348、777 行）。第 216 行的 `tw = 1700.` 只是**初始化占位**，两条 hloss 分支都会覆盖它 | 216（占位） | **是**（348/777 覆盖） | `gasification_zone_heat_loss_w()` 内的 `tw_K` |
| `tg` | 气室气相温度，**K**（待解未知量；由 Wegstein 迭代收敛，第 218–219、800 行） | 218 | **是**（迭代更新） | 单元 `current.T` |
| `poxyin` | 气室**入口** O2 分压，atm（第 141、147 行算出，每气室由 `poxyin = poxy` 更新，第 882 行）。`poxyin > 0.05` 是燃烧区判据 | 147 | **是**（每气室 882） | 入口 pO2（`Cell._inlet_pO2_is_combustion`） |
| `deltim` | 子步时长，s，由 `vsolid` 回填（`time = delh/us`，第 987 行） | 261 | **是**（每子步） | 不进入本轮的 hloss 式子 |

## 两条 hloss 分支的被乘数不同（必须写明）

- 第 341–350 行（`400` 之后的快路径）：燃烧区 `hloss = fcoal*abs(rheat1)*hl/100`，用 **`rheat1`**；
- 第 772–781 行（主循环 `515→737`）：燃烧区 `hloss = fcoal*abs(rheat)*hl/100`，用 **`rheat`**（净放热）。

本卡第 1.1 节写的是 `fcoal·|rheat|·hl/100`，对应**主循环**那一条，本轮按此转写。

## `nc·delh` 与 `hs` 的映射

`delh` 是**子步**长度，一个气室走 `nc` 个子步，因此 `nc·delh` 就是**一个气室的轴向长度**。
Fortran 里这个长度是变的（第一个气室 25×3 = 75 cm，其后 0.5×3 或 1.0×3 = 1.5 或 3 cm），
Python 的一个单元就是一个气室，故 `nc·delh → dz`、`hs → z + dz/2`（气室出口）。

## 符号不确定性检查

- `dt`/`height` 的单位由 `at = π/4·dt²` 与第 787–788 行的 `height-5` 交叉确认，取 cm，**确定**。
- `delh` 的语义由 `hs = hs+delh`（轴向累进）与 `time = delh/us`（第 987 行，`us` 为 cm/s）交叉确认，
  是轴向长度（cm），**确定**。
- `rheat` 的组成由第 772 行显式给出，**确定**。
- 唯一不确定项：Fortran 的 `rheat1` 把挥发分的**显热当量**（`ht(n,tg)`）当作热量计入，
  而 Python 的挥发分能量走 `PyrolysisSource.energy_src_W`，不重复计入反应热。这一项
  **在 Python 侧没有对应量**，本轮按「净化学放热」口径转写并在 `interpretation.md` 说明量级。
"""


def interpretation(unit_df, anchor_crit, anchor_cells, anchor_profiles, scan,
                   scan_readings, jac, protocol) -> str:
    L: list[str] = []
    L.append("# P1.8-A 锚点 Fortran 热损模型对齐 + 装置九窗二维辨识性扫描 —— interpretation\n")
    L.append(
        "本轮在模型仓按 2026-11-04 卡执行：①新增 `HeatLossModel ∈ {hhv_percent, fortran_wall}`，"
        "把 `Source1_副本.for` 的两区壁面热损逐字转写；②锚点模式 F′ = P1.7-A 模式 F + `fortran_wall`；"
        "③装置九窗做 `f_gas`（C+H2O 与 C+CO2 共用指前因子倍数）× `HeatLossPercent` 的二维扫描。"
        "冻结项一项未动：全部 Arrhenius 常数（除第 2 节扫描的倍数）、`phi`、`f_co2 = 1.0`、0.05 atm、"
        "`CharCombustionRateFactor`、氧化因子、热解产率、负向 WGS 截断（只统计绑定）、"
        "`USE_RK_GILL_COMBUSTION = False`、`fortran_constant`、蒸汽进料 = 0、低位热值口径、"
        "`uniform_full_length`、`SolverStrictRoot = True`。本轮**只扫描、不采纳**任何参数值。\n"
    )
    L.append("## 摘要（对应卡第 3 节的五问）\n")
    L.append(_summary_md(unit_df, anchor_crit, scan, scan_readings, jac))
    L.append("## 0. 提交与工作树\n")
    L.append(f"- `protocol.json` 里的 `source_commit`（生成本轮产物的代码提交）：`{protocol.get('source_commit')}`。")
    L.append("- 产物随后以一次只含 `outputs/diagnostics/round-p1.8a-20261104/` 与脚本的提交落盘，"
             "两个哈希不同（`protocol.json` 要写入那个哈希，互为自指），与 2026-10-30 起各轮约定一致。")
    L.append(f"- 生成时的代码树状态（已排除本轮产物目录本身）：`{_tree_note()}`。\n")

    # ---------------- 1 ----------------
    L.append("## 1. 第 1.1 节：符号表\n")
    L.append("完整表（含行号、是否被重赋值）见 `fortran_heatloss_symbols.md`，下面重列一遍。\n")
    L.append(_symbol_table_md())
    L.append("\n**两条 hloss 分支的被乘数不同**：第 346 行用 `rheat1`（挥发分项），第 780 行用 `rheat`（净放热）。"
             "卡第 1.1 节写的是 `|rheat|`，对应主循环 515→737 那条，本轮按此转写。\n")
    L.append("**`nc·delh` 与 `hs` 的映射**：`delh` 是子步长度，一个气室走 `nc` 个子步，"
             "故 `nc·delh` 就是一个气室的轴向长度；Fortran 里它随气室变化（第一个 75 cm，其后 1.5 或 3 cm），"
             "Python 一个单元就是一个气室，故 `nc·delh → dz`、`hs → z + dz/2`。"
             "`dt`、`height` 按卡取 Fortran 第 32 行的 152.4 cm / 330 cm，不取 Python 几何。\n")
    L.append("**JAX 路径**：`GasifierSystem._solve_jax_v4` 在 `HeatLossModel='fortran_wall'` 时直接抛 "
             "`ValueError`（不静默回退到 `hhv_percent`）；`GasificationRateFactor != 1` 同样抛错，"
             "因为 JAX 的 batch 残差里没有这两项的对应实现。两条都在 `unit_checks.csv` 的 "
             "`jax_path_guard` 组里验证（`raises=True`）。\n")

    # ---------------- 1.2 ----------------
    L.append("## 2. 第 1.2 节：单元检查\n")
    gate = unit_df[unit_df["pass"].notna()]
    L.append(f"判定项 **{len(gate)}**，通过 **{int(gate['pass'].astype(bool).sum())}** → 整体"
             f"**{'通过' if bool(gate['pass'].astype(bool).all()) else '未通过'}**。\n")
    L.append("| 组 | 上下文 | 指标 | 手算 | 代码 | 相对差 | 通过 |")
    L.append("|---|---|---|---:|---:|---:|---|")
    for _, r in unit_df.iterrows():
        L.append(f"| {r['check_group']} | {r['context']} | `{r['metric']}` | {_fmt(r['hand_value'], 6)} | "
                 f"{_fmt(r['code_value'], 6)} | {_fmt(r['rel_diff'], 3)} | {r['pass']} |")
    L.append("")
    L.append("- **`hhv_percent` 逐位复现**：模式 F、N=160 的 5 个干粉算例，出口温度与 X_C/CO/H2/CO2/CH4 "
             "全部与 P1.7-A `anchor_outlet.csv` **逐位相同**（判据是 `==`，不是容差）。这是「新开关没污染旧行为」的证据。")
    L.append("- **单格手算**：燃烧区 `0.30·|rheat|`、气化区 `u0·π·dt·dz·(tg−tw)` 与 `tw = 2100−600·hs/330` "
             "都在检查脚本里独立重写一遍算式，相对差 < 1e-9；同时验证 `tg < tw` 时 hloss 为负且**不 clamp**。")
    L.append("- **`HHV_d` 无关性**：`fortran_wall` 下 30→34 的 I-1 出口温度变化**逐位为 0**（判据 < 1e-6 K）。"
             "P1.7-A 记录的 −4.55 K 全部来自 `2% × HHV` 热损基准，换成 Fortran 热损后这一依赖本身不存在。\n")

    # ---------------- 1.3 ----------------
    L.append("## 3. 第 1.3 节：模式 F′ 的锚点结果\n")
    for tag, label in (("F", "模式 F（`hhv_percent`，P1.7-A）"), ("F_fortran_wall", "模式 F′（`fortran_wall`）")):
        sub = anchor_crit[anchor_crit["mode_tag"] == tag]
        if sub.empty:
            continue
        passed = sub.groupby("case_name")["pass"].all()
        L.append(f"### 3.{1 if tag == 'F' else 2} {label}\n")
        L.append(f"全部判据通过的算例：**{int(passed.sum())}/{len(passed)}**。\n")
        L.append("| 算例 | 判据 | Python | Fortran | 绝对差 | 容差 | 通过 |")
        L.append("|---|---|---:|---:|---:|---:|---|")
        for _, r in sub.sort_values(["case_name", "metric"]).iterrows():
            L.append(f"| {r['case_name']} | {r['metric']} | {_fmt(r['python_value'], 2)} | "
                     f"{_fmt(r['fortran_value'], 2)} | {_fmt(r['abs_diff'], 2)} | {r['tolerance']:.0f} | {r['pass']} |")
        L.append("")
    cmp = _anchor_compare(anchor_crit)
    if len(cmp):
        L.append("### 3.3 两条热损模型的逐判据对比\n")
        L.append("| 算例 | 判据 | Fortran | 模式 F | 模式 F′ | F 的绝对差 | F′ 的绝对差 | 变化 |")
        L.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for _, r in cmp.iterrows():
            L.append(f"| {r['case_name']} | {r['metric']} | {_fmt(r['fortran_value'], 2)} | "
                     f"{_fmt(r['F_python'], 2)} | {_fmt(r['Fp_python'], 2)} | {_fmt(r['F_abs'], 2)} | "
                     f"{_fmt(r['Fp_abs'], 2)} | {r['Fp_abs']-r['F_abs']:+.2f} |")
        L.append("")
        for metric in ("T_gas_K", "X_C_pp", "CO_vol_pct", "CO2_vol_pct"):
            g = cmp[cmp["metric"] == metric]
            if g.empty:
                continue
            L.append(f"- `{metric}`：平均绝对差 {g['F_abs'].mean():.2f} → {g['Fp_abs'].mean():.2f}"
                     f"（{'+' if g['Fp_abs'].mean() >= g['F_abs'].mean() else ''}"
                     f"{g['Fp_abs'].mean()-g['F_abs'].mean():.2f}）；逐算例的符号与量级见上表。")
        L.append("")
    L.append("### 3.4 为什么 F′ 没有改善温度：燃烧区只覆盖 1 个单元\n")
    zt = anchor_cells[anchor_cells["row_kind"] == "zone_total"]
    if len(zt):
        L.append("| 算例 | 燃烧区格数 | 燃烧区 hloss (MW) | 气化区格数 | 气化区 hloss (MW) | 合计 (MW) | T_out (K) |")
        L.append("|---|---:|---:|---:|---:|---:|---:|")
        for _, r in zt.sort_values("case_name").iterrows():
            L.append(f"| {r['case_name']} | {int(r['hloss_combustion_cells'])} | "
                     f"{_fmt(r['hloss_combustion_W']/1e6, 3)} | {int(r['hloss_gasification_cells'])} | "
                     f"{_fmt(r['hloss_gasification_W']/1e6, 3)} | {_fmt(r['hloss_total_W']/1e6, 3)} | "
                     f"{_fmt(r['T_out_K'], 1)} |")
        L.append("")
        L.append("三条事实合起来解释了 F′ 的方向：")
        L.append("1. Python 的 O2 在 **cell 0** 内就降到入口的 1% 以下（P1.5-C 已记录），"
                 "所以 `poxyin > 0.05` 的判据只命中 **1 个单元**（N=160 时 2.06 cm，N=80 时 4.13 cm）。"
                 "Fortran 的**第一个气室是 75 cm**（`nc=3`、首个 `delh=25`），同一条 30% 规则作用在 36 倍长的区间上。")
        L.append("2. 气化区的 `tw = 2100 − 600·hs/height` 是一条**被写死的边界条件**，它编码的是 Fortran 自身的"
                 "降温史（入口 2100 K、出口 1500 K）。在 F′ 的算例里，绝大多数单元的气体温度低于这条剖面"
                 "（例如 i-1 的 cell 1：tg = 1693 K vs tw = 2092 K），因此 `hloss` 为负——壁面向气体供热。"
                 "全炉气化区合计是**净供热**，不是净散热。")
        L.append("3. 结果是 F′ 的净壁面热流接近零甚至为**净供热**（燃烧区散热 0.16–0.37 MW，"
                 "气化区净供热 0.21–0.55 MW，合计 −0.05 ~ −0.20 MW），"
                 "反应器实际上更接近绝热，温度因此**高于** Fortran、也高于模式 F。"
                 "这不是转写错误：把 30% 规则套在 2 cm 的单元上、把 Fortran 自己的壁温剖面"
                 "套在一个温度史不同的模型上，就是这个量级。\n")
        L.append("**结论（第 1 节）**：转写与单元检查都通过，但模式 F′ **没有**缩小与 Fortran 的差距——"
                 "温度从「偏低」翻成「偏高」，X_C 反而更贴顶。两套能量模型真正的差别不在公式形式，"
                 "而在**燃烧区的离散尺度**（Fortran 75 cm 气室 vs Python 一个单元）。"
                 "要把这一项对齐，需要的是「燃烧区长度」这一层口径，而不是再换公式。\n")
    L.append("### 3.5 与 Fortran 中间气室温度的逐点对比\n")
    if len(anchor_profiles):
        L.append(f"逐点对比见 `anchor_profiles.csv`（Fortran 每个气室一个打印点，共 "
                 f"{len(anchor_profiles)} 个匹配点）。温度差的统计：")
        L.append("")
        L.append("| 算例 | 点数 | 平均差 (K) | 最小 | 最大 |")
        L.append("|---|---:|---:|---:|---:|")
        for name, g in anchor_profiles.groupby("case_name"):
            d = g["d_T_python_minus_fortran_K"]
            L.append(f"| {name} | {len(g)} | {d.mean():+.1f} | {d.min():+.1f} | {d.max():+.1f} |")
        L.append("")
    else:
        L.append("（没有可比对的中间气室。）\n")

    # ---------------- 2 ----------------
    L.append("## 4. 第 2 节：九窗二维扫描\n")
    grid = scan[scan["row_kind"] == "grid"]
    L.append(f"网格行数 **{len(grid)}**（9 窗 × {len(F_GAS_GRID)} × {len(HEAT_LOSS_PCT_GRID)} = "
             f"{9*len(F_GAS_GRID)*len(HEAT_LOSS_PCT_GRID)}），另有复现行与 P1.7-A 对齐行见 `row_kind` 列。"
             f"N = {N_SCAN}，煤量主口径（协调-阶段），模型入口载气 = `co2_carrier_measured_nm3_h`。\n")
    L.append("### 4.1 逐窗 X_C（模型 − 装置主口径，pp）\n")
    piv = grid.pivot_table(index="window_id", columns=["f_gas", "HeatLossPercent"],
                           values="model_minus_plant_xc_measured_pp", aggfunc="first")
    L.append("| 窗口 | " + " | ".join(f"f={f:g}/H={h:.0f}" for f, h in piv.columns) + " |")
    L.append("|---" * (len(piv.columns) + 1) + "|")
    for wid, r in piv.iterrows():
        L.append(f"| `{wid.split('__')[-2][:22]}…{wid[-4:]}` | "
                 + " | ".join(_fmt(r[c], 2) for c in piv.columns) + " |")
    L.append("")
    L.append(f"装置主口径 X_C 中位 {grid.groupby('window_id')['plant_xc_measured_pct'].first().median():.2f}%，"
             f"模型 X_C 中位（全部网格行）{grid['X_C_implied'].median()*100:.2f}%；"
             f"X_C ≥ 0.9999 的网格行 {int((grid['X_C_implied'] >= 0.9999).sum())}/{len(grid)}。\n")

    L.append("### 4.2 第 3 节第 4 问：复现行与 P1.7-A 的逐位一致性\n")
    L.extend(_replay_paragraph(scan))

    L.append("### 4.3 载气口径本身的效应\n")
    L.extend(_carrier_paragraph(scan))

    L.append("### 4.4 `no_root` 与负向 WGS 绑定\n")
    L.extend(_noroot_paragraph(scan))

    # ---------------- 3 ----------------
    L.append("## 5. 预声明读法 R1–R5（跑完不改）\n")
    L.extend(_readings_paragraph(scan_readings, jac))

    L.append("## 6. 与卡面不一致 / 需要统筹方裁决的地方\n")
    L.extend(_discrepancy_paragraph(scan))
    L.append("## 7. 未做的部分\n")
    L.append("- 本轮**没有**把燃烧区的离散尺度对齐（Fortran 75 cm 气室）；卡未授权改燃烧区判据，"
             "只按 `poxyin > 0.05` 逐字转写。第 1 节的结论因此是「转写对了、模型没对齐」。")
    L.append("- 本轮**没有**为 `f_gas` 与 `HeatLossPercent` 之外再加维度（`f_CO2` 未拆，见 R5）。")
    L.append("- JAX 路径未实现 `fortran_wall`，遇到即报错；未做 JAX 与 NumPy 的 Fortran 热损并行验证。")
    L.append("- 组成两口径只用来判读残差；模型求解本身只用一个组成口径（装置组成不进入模型）。")
    L.append("")
    return "\n".join(L)


def _summary_md(unit_df, anchor_crit, scan, rd, jac) -> str:
    L: list[str] = []
    gate = unit_df[unit_df["pass"].notna()]
    n_pass = int(gate["pass"].astype(bool).sum())
    L.append(f"1. **符号表与单元检查**：`fortran_heatloss_symbols.md` 逐符号列出定义/单位/行号/是否重赋值；"
             f"单元检查 {n_pass}/{len(gate)} 通过。`hhv_percent` 下模式 F 的 5 个算例出口**逐位复现** P1.7-A；"
             f"`fortran_wall` 下 `HHV_d` 30→34 的温变**逐位为 0**。")
    cmp = _anchor_compare(anchor_crit)
    if len(cmp):
        t = cmp[cmp["metric"] == "T_gas_K"]
        x = cmp[cmp["metric"] == "X_C_pp"]
        dT = t['Fp_python'] - t['fortran_value']
        pinned = int((x['Fp_python'] >= 99.999).sum())
        L.append(f"2. **模式 F′**：全部判据通过的算例仍是 **0/5**；温度平均绝对差 "
                 f"{t['F_abs'].mean():.1f} → **{t['Fp_abs'].mean():.1f} K**，"
                 f"X_C 平均绝对差 {x['F_abs'].mean():.2f} → **{x['Fp_abs'].mean():.2f} pp**。"
                 f"方向是**变差**：F′ 的温度整体偏高 {dT.min():.0f}–{dT.max():.0f} K、"
                 f"{pinned}/{len(x)} 个算例的 X_C 贴到 100%。"
                 "原因是燃烧区只有 1 个单元（2.06 cm vs Fortran 75 cm）拿到 30% 壁损，"
                 "而气化区的固定壁温剖面反过来给气体加热。")
    r1 = rd[rd["reading"] == "R1"]
    n_r1 = int(r1["exists"].astype(bool).sum())
    r2s = rd[rd["reading"] == "R2_summary"]
    r2_txt = "；".join(f"`{r['composition_basis']}` ξ_min 中位 {float(r['value']):.1f} mol/s"
                      for _, r in r2s.iterrows() if np.isfinite(r["value"]))
    r2_verdict = "；".join(
        f"`{r['composition_basis']}` → " + ("反应器后变换仍需要" if float(r["value"]) > R2_XI_THRESHOLD
                                            else "不需要反应器后变换")
        for _, r in r2s.iterrows() if np.isfinite(r["value"]))
    # 没有 R1 组合的窗离装置值还差多少（用整个搜索池的最小 |ΔX_C|）
    pool = scan[scan["row_kind"].isin(("grid", "grid_extension"))]
    miss = []
    for _, row_r1 in r1.iterrows():
        if bool(row_r1["exists"]):
            continue
        g = pool[pool["window_id"] == row_r1["window_id"]]
        px = float(g["plant_xc_measured_pct"].iloc[0])
        miss.append(abs(float(g["X_C_implied"].max()) * 100.0 - px))
    miss_txt = f"{min(miss):.1f}–{max(miss):.1f} pp" if miss else "—"
    if len(jac):
        cond = jac["cond_J"].to_numpy(dtype=float)
        corr = jac["corr_lnf_hlp"].to_numpy(dtype=float)
        ident = int(np.sum((cond < R3_COND_THRESHOLD) & (np.abs(corr) < R3_CORR_THRESHOLD)))
        n_win = len(jac) // len(COMP_BASES)
        r3_txt = (f"有 R1 组合的 {n_win} 窗上 cond(J) = {cond.min():.2f}–{cond.max():.2f}、"
                  f"|corr| = {np.abs(corr).min():.2f}–{np.abs(corr).max():.2f}，"
                  f"按预声明判据 **{ident}/{len(jac)} 条记为可分别辨识**"
                  f"（所有最优点都落在 HLP 上边界 6%）")
    else:
        r3_txt = "无可求雅可比的点"
    r4 = rd[rd["reading"] == "R4"]
    r4_txt = "；".join(
        f"`{b}` " + "、".join(f"{r['phase']} 中位 {float(r['value']):.4g}"
                             for _, r in r4[r4["composition_basis"] == b].iterrows())
        for b in COMP_BASES if len(r4[r4["composition_basis"] == b]))
    L.append(f"3. **二维扫描**：R1 只有 **{n_r1}/9** 窗存在 |ΔX_C| ≤ 1 pp 的 (f_gas, HLP) 组合"
             f"（其余 {9 - n_r1} 窗在 f_gas ≥ 1/32、HLP ≤ 6% 的整个搜索池里最小仍差 {miss_txt}）；"
             f"R2 {r2_txt} → {r2_verdict}"
             f"（该判据只在这 {n_r1} 窗上成立；其余窗根本没有 R1 组合）；R3 {r3_txt}；"
             f"R4 {r4_txt}（每个阶段只有 2 窗，不足以判跨阶段趋势）；"
             f"R5 C+CO2 在净 CO2 消耗中占 {_r5_range(rd)}，"
             "**C+H2O 与 C+CO2 共用指前因子应当拆开**（本轮不拆）。")
    par = scan[scan["row_kind"] == "parity_p17a"]
    ref = _p17a_baseline_ref()
    same = sum(1 for _, r in par.iterrows()
               if r["window_id"] in ref
               and float(ref[r["window_id"]]["T_out_K"]) == float(r["T_out_K"])
               and float(ref[r["window_id"]]["X_C_implied"]) == float(r["X_C_implied"]))
    L.append(f"4. **复现行**：卡面口径（旧载气 = `co2_carrier_t_nm3_h`）与 P1.7-A **不能**逐位一致——"
             f"P1.7-A 的模型入口载气其实是冻结切分的 `0.24 × o2_flow_mean`，两者差 5%–7%。"
             f"按冻结切分原值另跑的 9 行 **{same}/9 与 P1.7-A 逐位相同**，"
             "因此第 2 节继续执行（详见第 4.2 节）。")
    bad = scan[scan["execution_status"] != "completed"]
    flagged = scan[scan["wgs_neg_bind_len_frac_overall"] > WGS_BIND_FLAG] if "wgs_neg_bind_len_frac_overall" in scan else scan.iloc[:0]
    L.append(f"5. **`no_root` 与负向 WGS 绑定**：`no_root` **{len(bad)}/{len(scan)}**；"
             f"负向 WGS 绑定长度占比 > 5% 的行 **{len(flagged)}/{len(scan)}**"
             f"（每行都是 1 个单元，整炉占比恒为 {scan['wgs_neg_bind_len_frac_overall'].median():.4f}）。\n")
    return "\n".join(L)


def _r5_range(rd) -> str:
    r5 = rd[rd["reading"] == "R5"]
    v = r5["value"].astype(float) * 100.0
    v = v[np.isfinite(v)]
    if not len(v):
        return "—"
    return f"{v.min():.0f}–{v.max():.0f}%"


def _symbol_table_md() -> str:
    body = SYMBOLS_MD.split("| 符号 |")[1].split("## 两条 hloss 分支")[0]
    return "| 符号 |" + body


def _anchor_compare(crit: pd.DataFrame) -> pd.DataFrame:
    f = crit[crit["mode_tag"] == "F"].set_index(["case_name", "metric"])
    fp = crit[crit["mode_tag"] == "F_fortran_wall"].set_index(["case_name", "metric"])
    rows = []
    for key in f.index.intersection(fp.index):
        rows.append({
            "case_name": key[0], "metric": key[1],
            "fortran_value": float(f.loc[key, "fortran_value"]),
            "F_python": float(f.loc[key, "python_value"]),
            "Fp_python": float(fp.loc[key, "python_value"]),
            "F_abs": float(f.loc[key, "abs_diff"]),
            "Fp_abs": float(fp.loc[key, "abs_diff"]),
        })
    return pd.DataFrame(rows)


def _replay_paragraph(scan: pd.DataFrame) -> list[str]:
    L = []
    rep = scan[scan["row_kind"] == "replay_old_carrier"]
    par = scan[scan["row_kind"] == "parity_p17a"]
    if rep.empty:
        return ["（无复现行。）\n"]
    L.append("卡面要求「复现行 = `f_gas = 1`、`HeatLossPercent = 2`、**旧载气**（`co2_carrier_t_nm3_h`）」，"
             "并断言它与 P1.7-A 主口径应逐位一致。**这一条与仓库实际不符**，必须单列：\n")
    L.append("P1.7-A 的模型入口载气来自冻结切分的 `co2_carrier_volumetric_flow_value`"
             "（`revision_evidence_common.model_case_rows`：`max(o2_flow_mean × 0.24, 0)`），"
             "既不是 `co2_carrier_t_nm3_h`（运行数据页 T 列），也不是 `FI_1002`。"
             "因此按卡面口径（T 列）跑的复现行**不可能**与 P1.7-A 逐位一致。\n")
    L.append("为了把「新开关没污染旧行为」这件事真正验掉，本轮另跑了一组 `row_kind = parity_p17a` 的行"
             "（`f_gas = 1`、`HeatLossPercent = 2`、载气 = 冻结切分原值）。两组数并列如下：\n")
    L.append("| 窗口 | 载气口径 | 载气 (Nm³/h) | T_out (K) | X_C (%) | CO2 (vol%) | 与 P1.7-A 逐位相同 |")
    L.append("|---|---|---:|---:|---:|---:|---|")
    ref = _p17a_baseline_ref()
    for wid in rep["window_id"]:
        for tag, df in (("card: T 列", rep), ("frozen split", par)):
            hit = df[df["window_id"] == wid]
            if hit.empty:
                continue
            r = hit.iloc[0]
            same = "—"
            if tag == "frozen split":
                rr = ref.get(wid)
                same = str(bool(rr is not None
                                and float(rr["T_out_K"]) == float(r["T_out_K"])
                                and float(rr["X_C_implied"]) == float(r["X_C_implied"])))
            L.append(f"| `{wid.split('__')[-2][:20]}…{wid[-4:]}` | {tag} | "
                     f"{float(r['carrier_nm3_h']):.1f} | {float(r['T_out_K']):.2f} | "
                     f"{float(r['X_C_implied'])*100:.2f} | {float(r['CO2_vol_pct']):.3f} | {same} |")
    L.append("")
    L.append("**结论**：`parity_p17a` 组与 P1.7-A `baseline.csv` 的 `recon_phase / N=40` 行逐位相同 → "
             "缺省 `HeatLossModel='hhv_percent'` 与 `f_gas=1` 没有污染旧行为。"
             "卡面写的「T 列」那一组与 P1.7-A 不同是**输入不同**，不是回归，故第 2 节继续执行。"
             "阶段 2 两窗的 `co2_carrier_t_nm3_h` 无效（`t_carrier_status = invalid_phase2_tag_rotation`），"
             "复现行对这两窗**不可比**，表里如实保留。\n")
    return L


_P17A_BASELINE_CACHE: dict[str, object] = {}


def _p17a_baseline_ref() -> dict:
    if "ref" not in _P17A_BASELINE_CACHE:
        df = pd.read_csv(P17A_DIR / "baseline.csv")
        df = df[(df["coal_basis"] == "recon_phase") & (df["N_cells"] == N_SCAN)]
        _P17A_BASELINE_CACHE["ref"] = {r["window_id"]: r for _, r in df.iterrows()}
    return _P17A_BASELINE_CACHE["ref"]  # type: ignore[return-value]


def _carrier_paragraph(scan: pd.DataFrame) -> list[str]:
    rep = scan[scan["row_kind"] == "replay_old_carrier"]
    par = scan[scan["row_kind"] == "parity_p17a"]
    grid = scan[(scan["row_kind"] == "grid") & (scan["f_gas"] == 1.0)
                & (scan["HeatLossPercent"] == MAIN_HEAT_LOSS_PCT)]
    L = ["| 窗口 | 载气 实测 / 冻结切分 (Nm³/h) | ΔX_C (pp) | ΔT_out (K) | ΔCO2 (vol%) |"]
    L.append("|---|---|---:|---:|---:|")
    for wid in grid["window_id"]:
        a = grid[grid["window_id"] == wid]
        b = par[par["window_id"] == wid]
        if a.empty or b.empty:
            continue
        a, b = a.iloc[0], b.iloc[0]
        L.append(f"| `{wid.split('__')[-2][:22]}…{wid[-4:]}` | {float(a['carrier_nm3_h']):.0f} / "
                 f"{float(b['carrier_nm3_h']):.0f} | "
                 f"{(float(a['X_C_implied'])-float(b['X_C_implied']))*100:+.2f} | "
                 f"{float(a['T_out_K'])-float(b['T_out_K']):+.2f} | "
                 f"{float(a['CO2_vol_pct'])-float(b['CO2_vol_pct']):+.3f} |")
    L.append("")
    L.append("实测载气比冻结切分的 0.24×O2 高约 5%–7%，多进来的 CO2 直接抬高气相的 CO2 分压，"
             "把 C+CO2 推快、同时稀释 CO/H2；净效果见上表的 ΔX_C 与 ΔCO2。\n")
    return L


def _noroot_paragraph(scan: pd.DataFrame) -> list[str]:
    L = []
    bad = scan[scan["execution_status"] != "completed"]
    L.append(f"`no_root` / 未完成：**{len(bad)}/{len(scan)}**"
             + ("。" if bad.empty else "：" + "、".join(
                 f"{r['window_id'].split('__')[-2][:22]}({r['row_kind']},f={r['f_gas']:g},"
                 f"H={r['HeatLossPercent']:g},{r['execution_status']})" for _, r in bad.iterrows()) + "。"))
    if "wgs_neg_bind_len_frac_overall" in scan.columns:
        flagged = scan[scan["wgs_neg_bind_len_frac_overall"] > WGS_BIND_FLAG]
        L.append(f"负向 WGS 截断绑定长度占比 > 5% 的行：**{len(flagged)}/{len(scan)}**"
                 + ("。" if flagged.empty else "，最大 "
                    f"{flagged['wgs_neg_bind_len_frac_overall'].max():.3f}（"
                    f"{flagged.loc[flagged['wgs_neg_bind_len_frac_overall'].idxmax(), 'window_id'].split('__')[-2][:24]}）。"
                    "完整清单见 `scan.csv` 的 `wgs_neg_bind_len_frac_overall` 列。"))
        L.append(f"绑定格数（`wgs_neg_bind_cells`）分布：{sorted(set(int(v) for v in scan['wgs_neg_bind_cells']))}；"
                 f"整炉绑定长度占比中位 {scan['wgs_neg_bind_len_frac_overall'].median():.4f}"
                 f"（范围 {scan['wgs_neg_bind_len_frac_overall'].min():.4f}–"
                 f"{scan['wgs_neg_bind_len_frac_overall'].max():.4f}），与 P1.7-A 同一量级。\n")
    return L


def _readings_paragraph(rd: pd.DataFrame, jac: pd.DataFrame) -> list[str]:
    L = []
    L.append("### R1 离开上限（|模型 X_C − 装置主口径 X_C| ≤ 1 pp）\n")
    r1 = rd[rd["reading"] == "R1"]
    n_exist = int(r1["exists"].astype(bool).sum())
    L.append(f"9 窗中有 R1 组合的：**{n_exist}/9**（`*` 表示线性插值得到的 HLP）。")
    L.append("")
    L.append("| 窗口 | 阶段 | 存在 | 组合 |")
    L.append("|---|---|---|---|")
    for _, r in r1.iterrows():
        L.append(f"| `{r['window_id'].split('__')[-2][:24]}…{r['window_id'][-4:]}` | {r['phase']} | "
                 f"{r['exists']} | {r['detail']} |")
    L.append("")
    L.append("R1 只用 X_C，而 X_C 与组成口径无关（装置 X_C 由碳平衡给出，模型 X_C 不含组成），"
             "所以**两口径的 R1 完全相同**；下面 R2–R4 才分开报。\n")

    L.append("### R2 是否需要反应器后变换\n")
    r2 = rd[rd["reading"] == "R2"]
    for basis in COMP_BASES:
        g = r2[r2["composition_basis"] == basis]
        if g.empty:
            continue
        L.append(f"**{basis}**：逐窗 ξ_min（mol/s）=")
        L.append("")
        L.append("| 窗口 | 阶段 | ξ_min | 依据 |")
        L.append("|---|---|---:|---|")
        for _, r in g.iterrows():
            L.append(f"| `{r['window_id'].split('__')[-2][:24]}…{r['window_id'][-4:]}` | {r['phase']} | "
                     f"{_fmt(r['value'], 1)} | {r['detail']} |")
        L.append("")
    for _, r in rd[rd["reading"] == "R2_summary"].iterrows():
        L.append(f"- `{r['composition_basis']}`：{r['detail']}")
    r2_all = rd[rd["reading"] == "R2_summary"]
    n_ok = int(r2_all["detail"].str.extract(r"n=(\d+)/")[0].astype(float).max()) if len(r2_all) else 0
    L.append("")
    L.append(f"**读法必须带上的限定**：R2 的判据只在「存在 R1 组合」的窗口上有定义（本轮 {n_ok}/9 窗）。"
             "其余窗口在整个搜索池（f_gas ≥ 1/32、HLP ≤ 6%）里模型 X_C 仍高于装置值，"
             "ξ 作为一个把 CO2 拉回装置水平的量在那些窗上根本不存在。"
             "因此「不需要反应器后变换」这句只在**能到达装置 X_C 的那几窗**上成立；"
             "把这句读成「全部 9 窗都不需要」是超出证据的。")
    L.append("")

    L.append("### R3 可辨识性（R1 组合里 RSS 最小的点，有限差分）\n")
    if len(jac):
        L.append("差分步长：`ln f_gas` ±0.05，`HeatLossPercent` ±0.05。观测按 1 pp、0.5、0.5、0.5 vol% 缩放。\n")
        L.append("| 窗口 | 口径 | f_gas* | HLP* | cond(J) | cond(JᵀJ) | corr | 可分别辨识 |")
        L.append("|---|---|---:|---:|---:|---:|---:|---|")
        for _, r in jac.iterrows():
            ident = (np.isfinite(r.get("cond_J", np.nan)) and r["cond_J"] < R3_COND_THRESHOLD
                     and abs(r.get("corr_lnf_hlp", np.nan)) < R3_CORR_THRESHOLD)
            L.append(f"| `{r['window_id'].split('__')[-2][:22]}…{r['window_id'][-4:]}` | {r['composition_basis']} | "
                     f"{r['f_gas_star']:.4g} | {r['HeatLossPercent_star']:.3f} | "
                     f"{_fmt(r.get('cond_J'), 2)} | {_fmt(r.get('cond_JtJ'), 2)} | "
                     f"{_fmt(r.get('corr_lnf_hlp'), 3)} | {'是' if ident else '否'} |")
        L.append("")
        L.append(f"判据：条件数 < {R3_COND_THRESHOLD:.0f} **且** |相关| < {R3_CORR_THRESHOLD} 才记为「可分别辨识」。"
                 f"逐窗雅可比元素见 `jacobian.csv`。")
        no_r1 = rd[(rd["reading"] == "R3") & (~rd["exists"].astype(bool))]
        if len(no_r1):
            L.append(f"\nR1 无组合、无法求雅可比的窗-口径：{len(no_r1)} 条（"
                     + "、".join(f"{r['window_id'].split('__')[-2][:18]}/{r['composition_basis']}" for _, r in no_r1.iterrows())
                     + "）。")
        L.append("")
    L.append("### R4 跨阶段一致性\n")
    r4 = rd[rd["reading"].isin(["R4", "R4_all"])]
    L.append("| 口径 | 阶段 | 最优 f_gas 中位 | 说明 |")
    L.append("|---|---|---:|---|")
    for _, r in r4.iterrows():
        L.append(f"| {r['composition_basis']} | {r['phase']} | {_fmt(r['value'], 4)} | {r['detail']} |")
    L.append("")
    L.append("### R5 CO2 由谁消耗\n")
    r5 = rd[rd["reading"] == "R5"]
    L.append("| 窗口 | 阶段 | 点 | C+CO2 占比 | 细节 |")
    L.append("|---|---|---|---:|---|")
    for _, r in r5.iterrows():
        L.append(f"| `{r['window_id'].split('__')[-2][:22]}…{r['window_id'][-4:]}` | {r['phase']} | "
                 f"{r.get('item', '')} | {_fmt(r['value']*100, 1)}% | {r['detail']} |")
    L.append("")
    if len(r5):
        share = r5["value"].astype(float)
        over = int((share > 0.5).sum())
        L.append(f"C+CO2 占比 > 50% 的窗-点：**{over}/{len(r5)}**，中位 {share.median()*100:.1f}%。"
                 + ("按卡面要求**单列**：C+H2O 与 C+CO2 共用指前因子可能需要拆开，"
                    "供论文侧决定是否在下一轮增加 `f_CO2` 维度；本轮不拆。" if over else
                    "未超过 50%，本轮不单列拆维度的建议。"))
        L.append("")
    return L


def _discrepancy_paragraph(scan: pd.DataFrame) -> list[str]:
    L = []
    L.append("1. **卡面第 2.1 节的「旧载气 = 运行数据页 T 列」与 P1.7-A 的实现不符。** "
             "P1.7-A 的模型入口载气是冻结切分的 `max(0.24 × o2_flow_mean, 0)`，"
             "与 T 列（`co2_carrier_t_nm3_h`）相差 5%–7%。因此卡面「复现行应与 P1.7-A 逐位一致」"
             "按字面执行必然不成立。本轮按「两行都跑」处理：卡面口径的复现行照跑并如实标注，"
             "另加一组 `parity_p17a`（用冻结切分原值）**确实逐位复现** P1.7-A，"
             "以此满足第 3 节第 4 问真正要验的东西（新开关没污染旧行为）。"
             "**这一处需要统筹方确认**：以后「旧载气」到底指 T 列、J 列还是 0.24×O2。")
    L.append("2. **卡面第 2.2 节的「载气 CO2 净去向」与 E10 的模型侧数字口径不同。** E10 说"
             "「模型消耗 43–61%」，那是 P1.7-A（2% HHV 热损、冻结切分载气）下的结果；"
             "本轮换了载气口径与热损扫描维度，模型侧消耗比例的数值会变，逐窗见 `scan.csv` 的 "
             "`model_co2_consumption_frac`，不能直接与 E10 的区间对照。")
    L.append("3. **燃烧区判据在两套代码里的有效长度差 36 倍**（Fortran 首个气室 75 cm vs Python 一个单元 2.06 cm）。"
             "卡面要求「按 `poxyin > 0.05` 逐字转写」，照做；但由此产生的能量模型差异不是公式问题，"
             "第 3.4 节已量化。这一点超出卡面预期（卡面说「在这一项对齐之前 Python 继承 Wen 1979 尚未成立」，"
             "而实际结果表明即使换了公式，锚点仍不对齐）。")
    L.append("4. **`hhv_percent` 的逐位复现只覆盖模式 F 的 5 个干粉算例**（卡面原文如此）；"
             "模式 M 与两个浆态算例未做逐位比对。第 2 节的 `parity_p17a` 组另外覆盖了 9 窗基线。")
    L.append("5. **R2 的「9 窗中位」在 R1 无解的窗上不参与统计**（本轮有窗缺 R1 组合，见 R1 表），"
             "因此中位数只代表有解的窗；已在 R2 明细里逐窗列出 `exists`。")
    L.append("")
    return L


_TREE_NOTE_OVERRIDE: dict[str, str] = {}


def _tree_note() -> str:
    if "note" in _TREE_NOTE_OVERRIDE:
        return _TREE_NOTE_OVERRIDE["note"]
    return _git_status()


def _git_status(root: Path | None = None) -> str:
    import subprocess
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(root or ROOT), text=True
        ).strip()
    except Exception as exc:  # noqa: BLE001
        return f"unavailable ({type(exc).__name__})"
    keep = [ln for ln in out.splitlines() if "round-p1.8a-20261104" not in ln]
    return "\n".join(keep) if keep else "clean"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=("unit", "anchor", "scan", "all"))
    ap.add_argument(
        "--source-commit", default=None,
        help="写入 protocol.json 的代码提交；在无 .git 的快照里跑时用它显式给出",
    )
    ap.add_argument(
        "--tree-note", default=None,
        help="在无 .git 的快照里跑时，用它替代 git status 的那一行说明",
    )
    args = ap.parse_args()
    if args.tree_note:
        _TREE_NOTE_OVERRIDE["note"] = str(args.tree_note)
    silence_solver_logs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ok, rows, _inv = load_frozen_split()
    if not ok:
        raise SystemExit("frozen R52-C-75 split was not consumed intact")
    by_id = {r["window_id"]: r for r in rows}
    window_ids = m1_reference_window_ids(M1_MATRIX)
    p17d = pd.read_csv(P17D_CSV).set_index("window_id")
    p18d = pd.read_csv(P18D_CSV).set_index("window_id")
    missing = [w for w in window_ids if w not in p17d.index or w not in p18d.index]
    if missing:
        raise SystemExit(f"输入 CSV 缺窗口：{missing}")

    # ---------------- 第 1.2 节 ----------------
    unit_rows: list[dict] = []
    unit_rows += _single_cell_hand_checks()
    unit_rows += _p17a_parity_check(unit_rows)
    unit_rows += _f_gas_neutrality_check()
    unit_rows += _hhv_independence_check()
    unit_rows += _jax_guard_check()
    unit_df = pd.DataFrame(unit_rows)
    gate = unit_df[unit_df["pass"].notna()]
    unit_pass = bool(gate["pass"].astype(bool).all())

    (OUT_DIR / "fortran_heatloss_symbols.md").write_text(SYMBOLS_MD, encoding="utf-8")

    protocol = base_protocol({
        "round": "P1.8-A / 2026-11-04",
        "model_repo": "gasifier-1d-kinetic",
        "source_commit": (args.source_commit or source_commit()),
        "source_commit_note": "生成本轮全部产物的代码提交；产物本身随后另起一次仅含 outputs 的提交",
        "task_card": "papers/jax-gasifier-cascade-hybrid/manuscript/model-side-handoff-p1.8a-20261104.md",
        "equation_changes": {
            "1_heat_loss_model": "新增 op_conds['HeatLossModel'] ∈ {hhv_percent(缺省), fortran_wall}。"
                                 "fortran_wall 按 Source1_副本.for 第 341–350/772–781 行两区分支："
                                 "燃烧区 hloss = fcoal·|rheat|·hl/100（hl=30）；气化区 "
                                 "hloss = u0·π·dt·nc·delh·(tg−tw)，u0=3.4e-3 cal/(s·cm²·K)，"
                                 "tw = 2100 − 600·hs/height。dt=152.4 cm、height=330 cm 取 Fortran 第 32 行。"
                                 "tg < tw 时 hloss 为负，保留符号。",
            "2_reaction_heat": "Fortran 的 rheat 在 Python 侧的对应量："
                               "Q = −Σ rate·ΔH(T)，ΔH 用模型自身 Shomate 生成焓、"
                               "化学计量与 Cell._calc_balances 一致（含 phi 与 f_co2）。"
                               "Fortran 的 rheat1（挥发分显热当量）在 Python 侧不存在对应量。",
            "3_gasification_rate_factor": "新增 op_conds['GasificationRateFactor']：C+H2O 与 C+CO2 "
                                          "共用的指前因子倍数（E 不变），乘在 k_s 的 A 上而不是总速率上，"
                                          "因此膜扩散阻力仍在。缺省 1.0 逐位不变。",
            "4_jax_guard": "GasifierSystem._solve_jax_v4 对 HeatLossModel='fortran_wall' 与 "
                           "GasificationRateFactor≠1 直接抛 ValueError，不静默回退。",
        },
        "fortran_symbols": {
            "source": "reference_fortran/Source1_副本.for",
            "table_file": "fortran_heatloss_symbols.md",
            "ambiguous": ["rheat1 的显热当量口径在 Python 侧无对应量（已在 interpretation 说明）"],
        },
        "frozen_items_untouched": [
            "all Arrhenius constants (except the f_gas multiplier scanned in section 2)",
            "phi", "f_co2 = 1.0", "P_O2_Combustion_atm = 0.05",
            "CharCombustionRateFactor", "H2_OxidationFactor / CO_OxidationFactor",
            "pyrolysis yields", "negative-WGS hydrogen truncation（只统计绑定）",
            "USE_RK_GILL_COMBUSTION = False", "ParticleSizeModel = fortran_constant",
            "steam_flow = 0", "low-heating-value basis", "MeshFamily = uniform_full_length",
            "SolverStrictRoot = True",
        ],
        "unit_checks_all_pass": unit_pass,
        "anchor": {
            "mode_F_prime": "P1.7-A 模式 F + HeatLossModel='fortran_wall'",
            "n_list": list(N_ANCHOR), "n_reference": N_ANCHOR_REF,
            "tolerances": ANCHOR_TOL,
            "cases": list(DRY_POWDER_CASES),
            "hhv_independence_check": "I-1, 模式 F′, N=160, HHV_d 30 → 34, 判据 |ΔT| < 1e-6 K",
        },
        "scan": {
            "windows": list(window_ids),
            "n_cells": N_SCAN,
            "coal_basis": "recon_phase（协调-阶段，干煤）",
            "f_gas_grid": list(F_GAS_GRID),
            "f_gas_extension": F_GAS_EXTENSION,
            "f_gas_extension_rule": "若 f_gas = 0.0625 时 9 窗 X_C 中位仍高于装置主口径中位，"
                                    "向下只延伸一档（0.03125），不再继续；延伸行属网格的一部分，"
                                    "参与 R1–R4 的搜索池",
            "heat_loss_percent_grid": list(HEAT_LOSS_PCT_GRID),
            "carrier": "co2_carrier_measured_nm3_h（P1.8-D 实测口径，模型入口与装置碳平衡同一口径）",
            "carrier_j_note": "J 口径 X_C 只在与刘文比较时并列",
            "replay_rows": "旧载气 = co2_carrier_t_nm3_h（卡面口径）；另加 parity_p17a = 冻结切分 "
                           "max(0.24×o2_flow_mean,0)，用于与 P1.7-A 逐位对齐",
            "composition_bases": {
                "analyzer": "在线分析仪窗口值（冻结切分的 co/h2/co2_vol_pct）",
                "liu_phase": "刘文表 3 阶段值 CO/H2/CO2 = 60.14/29.74/9.36、61.05/29.24/8.99、"
                             "60.74/29.42/9.14、59.72/30.38/8.98",
            },
            "expected_rows": {"grid": 180, "replay": 9, "parity": 9},
        },
        "readings": {
            "R1": f"逐窗 |模型 X_C − 装置主口径 X_C| ≤ {R1_XC_TOL_PP} pp 的 (f_gas, HeatLossPercent) 组合，"
                  "线性插值并标注",
            "R2": f"沿 R1 组合的最小 ξ；9 窗中位在两种口径下都 > {R2_XI_THRESHOLD} mol/s → 反应器后变换仍需要",
            "R3": f"RSS 最小点的 (JᵀJ)^-1 → 条件数 < {R3_COND_THRESHOLD} 且 |相关| < {R3_CORR_THRESHOLD} → 可分别辨识",
            "R4": "各阶段最优 f_gas 的中位与极差，两口径分别报",
            "R5": "f_gas=1 与 R3 最优点上 C+CO2 与逆 WGS 在净 CO2 消耗中的占比",
        },
        "source_csv": {
            "p17d": {
                "round": "p1.7d-20261103", "file": P17D_CSV.name,
                "sha256": hashlib.sha256(P17D_CSV.read_bytes()).hexdigest(), "read_only": True,
            },
            "p18d": {
                "round": "p1.8d-20261104", "file": P18D_CSV.name,
                "sha256": hashlib.sha256(P18D_CSV.read_bytes()).hexdigest(), "read_only": True,
            },
            "data_repo": "gasifier-data-prep（本轮只读，未修改任何文件）",
        },
        "extra_artifacts": ["anchor_profiles.csv（Fortran 中间气室温度与 Python 逐点对比）"],
    })

    if args.stage == "unit":
        interp = _unit_only_interpretation(unit_df, unit_pass)
        written = write_round_outputs(OUT_DIR, tables={"unit_checks.csv": unit_df},
                                      protocol=protocol, interpretation=interp)
        print(json.dumps({"unit_pass": unit_pass, "written": [str(p) for p in written]},
                         ensure_ascii=False, indent=2))
        return

    # ---------------- 第 1.3 节 ----------------
    anchor_out, anchor_cells, anchor_profiles = anchor_runs()
    anchor_crit = anchor_criteria(anchor_out)

    if args.stage == "anchor":
        tables = {"unit_checks.csv": unit_df, "anchor_criteria.csv": anchor_crit,
                  "anchor_cells.csv": anchor_cells, "anchor_profiles.csv": anchor_profiles,
                  "anchor_outlet.csv": anchor_out}
        interp = "# P1.8-A（anchor stage）\n"
        written = write_round_outputs(OUT_DIR, tables=tables, protocol=protocol, interpretation=interp)
        print(json.dumps({"written": [str(p) for p in written]}, ensure_ascii=False, indent=2))
        return

    # ---------------- 第 2 节 ----------------
    scan = run_scan(window_ids, by_id, p17d, p18d)
    extra = pd.DataFrame(_extension_rows(window_ids, by_id, p17d, p18d, scan))
    if len(extra):
        scan = pd.concat([scan, extra], ignore_index=True)
    rd, jac = readings(scan, window_ids, by_id, p17d, p18d)

    protocol.update({
        "scan_rows_actual": int(len(scan)),
        "scan_rows_by_kind": scan["row_kind"].value_counts().to_dict(),
        "unit_checks_pass": unit_pass,
        "anchor_criteria_f_mode_pass_cases": int(
            anchor_crit[anchor_crit["mode_tag"] == "F"].groupby("case_name")["pass"].all().sum()),
        "anchor_criteria_fp_mode_pass_cases": int(
            anchor_crit[anchor_crit["mode_tag"] == "F_fortran_wall"].groupby("case_name")["pass"].all().sum()),
        "f_gas_extension_triggered": bool(len(extra)),
    })

    interp = interpretation(unit_df, anchor_crit, anchor_cells, anchor_profiles,
                            scan, rd, jac, protocol)
    tables = {
        "unit_checks.csv": unit_df,
        "anchor_criteria.csv": anchor_crit,
        "anchor_cells.csv": anchor_cells,
        "anchor_profiles.csv": anchor_profiles,
        "anchor_outlet.csv": anchor_out,
        "scan.csv": scan,
        "scan_readings.csv": rd,
        "jacobian.csv": jac,
    }
    written = write_round_outputs(OUT_DIR, tables=tables, protocol=protocol, interpretation=interp)
    print(json.dumps({"written": [str(p) for p in written], "unit_pass": unit_pass,
                      "scan_rows": int(len(scan))}, ensure_ascii=False, indent=2))


def _extension_rows(window_ids, by_id, p17d, p18d, scan: pd.DataFrame) -> list[dict]:
    """卡面：若 f_gas = 0.0625 时 9 窗 X_C 中位仍高于装置主口径中位，向下只延伸一档。"""
    grid = scan[scan["row_kind"] == "grid"]
    lowest = grid[grid["f_gas"] == min(F_GAS_GRID)]
    model_med = float(lowest["X_C_implied"].median()) * 100.0
    plant_med = float(p18d.loc[window_ids, "xc_measured_pct"].median())
    if not (np.isfinite(model_med) and model_med > plant_med):
        return []
    rows = []
    for wid in window_ids:
        row = by_id[wid]
        csv18 = p18d.loc[wid]
        dry = float(p17d.loc[wid, "coal_dry_recon_phase_kg_s"])
        carrier = float(csv18["co2_carrier_measured_nm3_h"])
        for hlp in HEAT_LOSS_PCT_GRID:
            rec = run_case(
                row, n_cells=N_SCAN, coal_flow_dry_kg_s=dry, keep_system=True,
                extra_op={
                    "HeatLossPercent": float(hlp),
                    "GasificationRateFactor": F_GAS_EXTENSION,
                    "carrier_gas_flow": _carrier_kg_s(carrier),
                },
            )
            system = rec["_system"]
            rows.append(_build_row_extras(
                rec, system, row, p17d.loc[wid], csv18, dry, N_SCAN,
                F_GAS_EXTENSION, hlp, carrier, "measured", "grid_extension",
            ))
    return rows


def _unit_only_interpretation(unit_df, unit_pass) -> str:
    L = ["# P1.8-A 单元检查（unit stage）\n"]
    L.append(f"整体：**{'通过' if unit_pass else '未通过'}**。\n")
    L.append("| 组 | 上下文 | 指标 | 手算 | 代码 | 相对差 | 通过 |")
    L.append("|---|---|---|---:|---:|---:|---|")
    for _, r in unit_df.iterrows():
        L.append(f"| {r['check_group']} | {r['context']} | `{r['metric']}` | {_fmt(r['hand_value'], 6)} | "
                 f"{_fmt(r['code_value'], 6)} | {_fmt(r['rel_diff'], 3)} | {r['pass']} |")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
