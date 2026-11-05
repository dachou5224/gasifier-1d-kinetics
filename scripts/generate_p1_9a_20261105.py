#!/usr/bin/env python3
"""P1.9-A 燃烧区分辨率受控试验（2026-11-05 卡）。

只回答一件事：在全长网格不变的条件下单独细化燃烧区首格（26 cm → ~0.5 cm），
出口 X_C、温度、组成以及气化指前因子的杠杆是否改变。

冻结项：与 P1.8-A 相同；HeatLossModel="hhv_percent"；f_gas=1（第 3 节除外）；
煤量主口径；模型入口载气 = 实测口径（P1.8-D）。不改速率式、限幅逻辑、
燃烧区判据或求解器容差。

网格族 ``graded_full_length``（本轮新增）：
    全长 L 分两段，[0, z_b] 用等长细格 dz0，[z_b, L] 用 N_rest 个粗格。
    块内格数 n_block = round(z_b/dz0)，总格数 N = n_block + N_rest。
    dz0 = 0.2575 m 时与均匀 N = 40 **逐位相同**（单元检查项）。

产物（outputs/diagnostics/round-p1.9a-20261105/）：
    unit_checks.csv  网格单元检查（Σdz、逐位复现、O2 口径）
    mesh_runs.csv    装置 9 窗 × 6 档 dz0 × 热损 {2,1}% = 108 行
    mesh_cells.csv   上述行中热损 2% 各档的块内逐单元记录
    anchor_mesh.csv  锚点 5 干粉算例 × 4 档（模式 F，只记录）
    leverage.csv     最细档与均匀 N=40 的 f_gas 杠杆对照（18 + 18 行）
    readings.csv     Q1–Q4 逐窗与汇总裁决
    protocol.json / interpretation.md / sha256.txt

用法：
    ./.venv/bin/python scripts/generate_p1_9a_20261105.py [--stage unit|device|anchor|leverage|all]
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
sys.path.insert(0, str(ROOT / "scripts"))

import generate_p1_8a_20261104 as p18a  # noqa: E402

from model.fortran_anchor_20261030 import (  # noqa: E402
    DRY_POWDER_CASES,
    anchor_cases,
    solve_fortran_case,
)
from model.gasifier_system import GasifierSystem  # noqa: E402
from model.grid_service import AdaptiveMeshGenerator, MeshConfig  # noqa: E402
from model.input_contract import coal_hhv_kj_kg  # noqa: E402
from model.p0_round_20260930 import m1_reference_window_ids  # noqa: E402
from model.p1_5_round_20261030 import (  # noqa: E402
    base_protocol,
    run_case,
    silence_solver_logs,
    write_round_outputs,
)
from model.revision_evidence_common import load_frozen_split, source_commit  # noqa: E402

OUT_DIR = ROOT / "outputs" / "diagnostics" / "round-p1.9a-20261105"
P18A_DIR = ROOT / "outputs" / "diagnostics" / "round-p1.8a-20261104"
P17D_CSV = p18a.P17D_CSV
P18D_CSV = p18a.P18D_CSV
M1_MATRIX = p18a.M1_MATRIX

# ---- 装置网格族（L = 10.3 m，z_b = 均匀 N=40 的前两格）------------------------
DEV_L = 10.3
DEV_ZB = 0.515
DEV_N_REST = 38
DEV_DZ0 = (0.2575, 0.103, 0.0515, 0.0206, 0.0103, 0.00515)

# ---- 锚点网格族（Fortran 几何 L = 3.3 m，均匀 N=160 的前两格）----------------
ANC_ZB = 0.04125
ANC_N_REST = 158
ANC_DZ0 = (0.020625, 0.0103125, 0.00515625, 0.002578125)

BASE_DZ0 = 0.2575
FINEST_DZ0 = 0.00515
SECOND_FINEST_DZ0 = 0.0103
HLP_GRID = (2.0, 1.0)
F_LEV_EXTRA = (0.125, 0.03125)

# ---- 预声明阈值（跑完不改）---------------------------------------------------
Q1_DXC_PP = 0.2
Q1_DT_K = 5.0
Q1_DCO2_VOL = 0.2
Q2_DXC_PP = 1.0
Q2_DT_K = 20.0
Q2_DCO2_VOL = 0.5
Q4_RATIO = 3.0

O2_INDEX = 0
CO2_INDEX = 3

DEV_N = {dz0: int(round(DEV_ZB / dz0)) + DEV_N_REST for dz0 in DEV_DZ0}
ANC_N = {dz0: int(round(ANC_ZB / dz0)) + ANC_N_REST for dz0 in ANC_DZ0}


# ---------------------------------------------------------------------------
# 网格与单元检查
# ---------------------------------------------------------------------------

def graded_mesh(L: float, zb: float, dz0: float, n_rest: int) -> np.ndarray:
    n_block = int(round(zb / dz0))
    n_cells = n_block + n_rest
    cfg = MeshConfig(
        total_length=L,
        n_cells=n_cells,
        mesh_family="graded_full_length",
        graded_zone_length=zb,
        graded_zone_dz=dz0,
    )
    dz, _z = AdaptiveMeshGenerator(cfg).generate()
    return np.asarray(dz, dtype=float)


def uniform_mesh(L: float, n_cells: int) -> np.ndarray:
    dz, _z = AdaptiveMeshGenerator(
        MeshConfig(total_length=L, n_cells=n_cells, mesh_family="uniform_full_length")
    ).generate()
    return np.asarray(dz, dtype=float)


def heatloss_unit_checks() -> list[dict]:
    """HLP → 实际热损的换算单元检查（纯函数，不依赖求解器）。

    审查意见要求把 HLP 与实际逐格 Q_loss 分开，这里用合成温度给出闭式答案：
    均匀 T 时 Σ(dz/L·clip((T/1800)^4,0.1,5)) = clip((T/1800)^4,0.1,5)，
    因此实际占比必须等于 HLP/100 × 该权重。
    """
    rows: list[dict] = []
    L = DEV_L
    dz = np.full(40, L / 40.0)
    hlp = 2.0
    q_total_w = 531.308e6
    for T_K, expect_w in (
        (1800.0, 1.0),                      # 权重恰为 1（参考温度）
        (1200.0, (1200.0 / 1800.0) ** 4),   # 0.1975，未触界
        (900.0, 0.1),                       # 低于下限 → clip 到 0.1
        (3000.0, 5.0),                      # 高于上限 → clip 到 5
    ):
        w = np.clip((T_K / 1800.0) ** 4.0, 0.1, 5.0)
        loss = (hlp / 100.0) * q_total_w * float(np.sum(dz / L * w))
        hand = (hlp / 100.0) * q_total_w * expect_w
        rows.append({
            "group": "heatloss_basis",
            "context": f"均匀 T = {T_K:.0f} K，HLP = {hlp:.0f}%，dz = L/40",
            "metric": "Q_loss_W",
            "hand_value": hand,
            "code_value": loss,
            "rel_diff": abs(loss - hand) / max(abs(hand), 1e-30),
            "tolerance": 1e-12,
            "pass": bool(abs(loss - hand) <= 1e-12 * max(abs(hand), 1e-30)),
            "note": f"权重 clip((T/1800)^4,0.1,5) = {float(w):.6f}",
        })
    # 组合检查：非均匀 T 时，Σ(dz/L·w) 必须严格小于 1，
    # 说明「实际热损 < HLP」是温度权重的直接后果，不是实现缺陷。
    T_prof = np.concatenate([np.full(2, 2650.0), np.full(38, 1400.0)])
    w = np.clip((T_prof / 1800.0) ** 4.0, 0.1, 5.0)
    share = float(np.sum(dz / L * w))
    hand_share = float(
        sum(float(dz[i] / L) * float(np.clip((T_prof[i] / 1800.0) ** 4.0, 0.1, 5.0))
            for i in range(len(T_prof)))
    )
    rows.append({
        "group": "heatloss_basis",
        "context": "合成剖面：2 格 2650 K + 38 格 1400 K（模拟燃烧区 + 气化区）",
        "metric": "sum_dz_over_L_times_weight",
        "hand_value": hand_share,
        "code_value": share,
        "rel_diff": abs(share - hand_share) / max(abs(hand_share), 1e-30),
        "tolerance": 1e-12,
        "pass": bool(abs(share - hand_share) <= 1e-12 * hand_share and share < 1.0),
        "note": "share < 1 即实际热损 < HLP；这是 wt 在气化区远小于 1 的结果，不是 bug",
    })
    return rows


def unit_checks() -> pd.DataFrame:
    rows: list[dict] = []
    # 装置
    for dz0 in DEV_DZ0:
        dz = graded_mesh(DEV_L, DEV_ZB, dz0, DEV_N_REST)
        n_block = int(round(DEV_ZB / dz0))
        rows.append({
            "group": "device",
            "z_b_m": DEV_ZB,
            "dz0_m": dz0,
            "n_block": n_block,
            "N_cells": int(dz.size),
            "sum_dz_m": float(np.sum(dz)),
            "sum_dz_minus_L_m": float(np.sum(dz) - DEV_L),
            "sum_dz_over_L": float(np.sum(dz) / DEV_L),
            "dz_max_m": float(np.max(dz)),
            "dz_min_m": float(np.min(dz)),
            "block_length_m": float(np.sum(dz[:n_block])),
            "rest_length_m": float(np.sum(dz[n_block:])),
            "bitwise_equal_uniform_N40": bool(
                dz.size == 40 and np.array_equal(dz, uniform_mesh(DEV_L, 40))
            ) if dz0 == BASE_DZ0 else None,
            "pass": bool(abs(np.sum(dz) - DEV_L) <= 1e-12 * DEV_L and np.all(dz > 0)),
        })
    # 锚点
    cases, geo = anchor_cases()
    L = float(geo["L_m"])
    for dz0 in ANC_DZ0:
        dz = graded_mesh(L, ANC_ZB, dz0, ANC_N_REST)
        n_block = int(round(ANC_ZB / dz0))
        rows.append({
            "group": "anchor",
            "z_b_m": ANC_ZB,
            "dz0_m": dz0,
            "n_block": n_block,
            "N_cells": int(dz.size),
            "sum_dz_m": float(np.sum(dz)),
            "sum_dz_minus_L_m": float(np.sum(dz) - L),
            "sum_dz_over_L": float(np.sum(dz) / L),
            "dz_max_m": float(np.max(dz)),
            "dz_min_m": float(np.min(dz)),
            "block_length_m": float(np.sum(dz[:n_block])),
            "rest_length_m": float(np.sum(dz[n_block:])),
            "bitwise_equal_uniform_N40": None,
            "pass": bool(abs(np.sum(dz) - L) <= 1e-12 * L and np.all(dz > 0)),
        })
    # 基准档与均匀 N=40 网格逐单元对照
    base = graded_mesh(DEV_L, DEV_ZB, BASE_DZ0, DEV_N_REST)
    uni = uniform_mesh(DEV_L, 40)
    rows.append({
        "group": "device_parity",
        "z_b_m": DEV_ZB,
        "dz0_m": BASE_DZ0,
        "n_block": int(round(DEV_ZB / BASE_DZ0)),
        "N_cells": int(base.size),
        "sum_dz_m": float(np.sum(base)),
        "sum_dz_minus_L_m": float(np.sum(base) - DEV_L),
        "sum_dz_over_L": float(np.sum(base) / DEV_L),
        "dz_max_m": float(np.max(base)),
        "dz_min_m": float(np.min(base)),
        "block_length_m": float(np.sum(base[:2])),
        "rest_length_m": float(np.sum(base[2:])),
        "bitwise_equal_uniform_N40": bool(np.array_equal(base, uni)),
        "pass": bool(np.array_equal(base, uni)),
    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 逐单元统计
# ---------------------------------------------------------------------------

HET_KEYS = ("C+O2", "C+H2O", "C+CO2", "C+H2")


def _cell_integrals(system: GasifierSystem) -> list[dict]:
    rows = []
    for cell, state in zip(system.cells, system.results):
        snap = cell.snapshot_with_limits(state)
        rh = snap["r_het"]
        rows.append({
            "T_K": float(state.T),
            "Ts_avg_K": float(snap.get("Ts_avg", float("nan"))),
            "X_total": float(snap.get("phys", {}).get("X_total", float("nan"))),
            "r_C_O2": float(rh.get("C+O2", 0.0)),
            "r_C_H2O": float(rh.get("C+H2O", 0.0)),
            "r_C_CO2": float(rh.get("C+CO2", 0.0)),
            "r_C_H2": float(rh.get("C+H2", 0.0)),
            "limited_by_o2": bool(snap["limits"].get("C_O2_LimitedByO2", False)),
            "limited_by_h2o": bool(snap["limits"].get("C_H2O_LimitedByH2O", False)),
            "limited_by_co2": bool(snap["limits"].get("C_CO2_LimitedByCO2", False)),
            "limited_by_h2": bool(snap["limits"].get("C_H2_LimitedByH2", False)),
            "wgs_neg_truncated": bool(snap["limits"].get("WGS_NegativeTruncated", False)),
            "is_combustion_zone": bool(snap["limits"].get("is_combustion_zone", False)),
            "cell0_n_distinct_roots": int(system.cell0_n_distinct_roots),
        })
    return rows


def block_stats(system: GasifierSystem, zb: float, dz0: float) -> dict:
    """块 [0, z_b] 内的结构量 + z_b 节点 + O2 消耗位置。"""
    n_block = int(round(zb / dz0))
    dz = np.array([d["dz_m"] for d in system.cell_diagnostics], dtype=float)
    z_out = np.cumsum(dz)
    z_center = np.array([cell.z for cell in system.cells], dtype=float)
    n = min(n_block, len(system.results))
    out: dict = {"block_n_cells": n_block, "block_length_solved_m": float(z_out[n - 1]) if n else float("nan")}

    if n > 0:
        T_block = np.array([float(system.results[i].T) for i in range(n)])
        i_peak = int(np.argmax(T_block))
        out["block_T_peak_K"] = float(T_block[i_peak])
        out["block_T_peak_z_m"] = float(z_center[i_peak])
        out["block_T_min_K"] = float(np.min(T_block))
    else:
        out.update({"block_T_peak_K": float("nan"), "block_T_peak_z_m": float("nan"), "block_T_min_K": float("nan")})

    if system.execution_status == "completed" and system.results:
        inlet_c0 = float(system.cells[0].inlet.solid_mass * system.cells[0].inlet.carbon_fraction)
        node = system.results[n - 1]
        out["z_b_T_K"] = float(node.T)
        out["z_b_X_C"] = float(1.0 - node.solid_mass * node.carbon_fraction / max(inlet_c0, 1e-12))
        # 燃烧区判据（pO2 > 0.05 atm）实际覆盖到哪里
        flags = [bool(d.get("is_combustion_zone", False)) for d in system.cell_diagnostics]
        idxs = [i + 1 for i, fl in enumerate(flags) if fl]
        out["combustion_zone_n_cells"] = int(sum(flags))
        out["combustion_zone_length_m"] = float(sum(dz[i] for i, fl in enumerate(flags) if fl))
        out["z_comb_end_m"] = float(sum(dz[: max(idxs)])) if idxs else 0.0
    else:
        out.update({"z_b_T_K": float("nan"), "z_b_X_C": float("nan"),
                    "combustion_zone_n_cells": 0, "combustion_zone_length_m": float("nan"),
                    "z_comb_end_m": float("nan")})

    # 限幅绑定（块内）
    if system.results:
        cells = _cell_integrals(system)
        for key, col in (
            ("o2", "limited_by_o2"), ("h2o", "limited_by_h2o"),
            ("co2", "limited_by_co2"), ("h2", "limited_by_h2"),
        ):
            nb = sum(1 for i in range(n) if cells[i][col])
            lb = float(sum(dz[i] for i in range(n) if cells[i][col]))
            out[f"block_n_{key}_limited"] = int(nb)
            out[f"block_len_frac_{key}_limited"] = lb / max(float(np.sum(dz[:n])), 1e-12)
        out["block_len_frac_wgs_neg_truncated"] = float(
            sum(dz[i] for i in range(n) if cells[i]["wgs_neg_truncated"])
        ) / max(float(np.sum(dz[:n])), 1e-12)
        # 块内 / 块外反应积分
        tot = {k: 0.0 for k in HET_KEYS}
        blk = {k: 0.0 for k in HET_KEYS}
        for i, rec in enumerate(cells):
            for k in HET_KEYS:
                v = rec["r_" + k.replace("+", "_")]
                tot[k] += v
                if i < n:
                    blk[k] += v
        for k in HET_KEYS:
            tag = k.replace("+", "_")
            out[f"rx_{tag}_mol_s"] = tot[k]
            out[f"rx_{tag}_frac_block"] = blk[k] / tot[k] if abs(tot[k]) > 1e-12 else float("nan")
        # O2 位置（入口 O2 的 1% 与完全耗尽）
        o2_in = float(system.cells[0].inlet.gas_moles[O2_INDEX])
        o2 = np.array([float(s.gas_moles[O2_INDEX]) for s in system.results])
        out["O2_inlet_mol_s"] = o2_in
        for tag, thresh in (("one_pct", 0.01 * o2_in), ("zero", 0.0)):
            hit = np.where(o2 <= thresh)[0]
            if hit.size:
                idx = int(hit[0])
                out[f"z_O2_{tag}_m"] = float(z_out[idx])
                out[f"z_O2_{tag}_cell_index"] = idx
            else:
                out[f"z_O2_{tag}_m"] = float("nan")
                out[f"z_O2_{tag}_cell_index"] = -1
    return out


def cell_rows(system: GasifierSystem, window_id: str, dz0: float, hlp: float, zb: float) -> list[dict]:
    n_block = int(round(zb / dz0))
    rows = []
    inlet_c0 = float(system.cells[0].inlet.solid_mass * system.cells[0].inlet.carbon_fraction)
    o2_in = float(system.cells[0].inlet.gas_moles[O2_INDEX])
    integrals = _cell_integrals(system)
    z_out = 0.0
    for i, (cell, state, extra) in enumerate(zip(system.cells, system.results, integrals)):
        z_out += float(cell.dz)
        rows.append({
            "window_id": window_id,
            "dz0_m": dz0,
            "HeatLossPercent": hlp,
            "cell_index": i,
            "z_center_m": float(cell.z),
            "dz_m": float(cell.dz),
            "in_block": bool(i < n_block),
            "T_K": float(state.T),
            "Ts_avg_K": extra["Ts_avg_K"],
            "X_total": extra["X_total"],
            "X_C_cumulative": float(1.0 - state.solid_mass * state.carbon_fraction / max(inlet_c0, 1e-12)),
            "O2_mol_s": float(state.gas_moles[O2_INDEX]),
            "O2_over_inlet": float(state.gas_moles[O2_INDEX]) / max(o2_in, 1e-12),
            "CO2_mol_s": float(state.gas_moles[CO2_INDEX]),
            "T_solid_K": float(getattr(state, "T_solid", float("nan"))),
            "r_C_O2_mol_s": extra["r_C_O2"],
            "r_C_H2O_mol_s": extra["r_C_H2O"],
            "r_C_CO2_mol_s": extra["r_C_CO2"],
            "r_C_H2_mol_s": extra["r_C_H2"],
            "limited_by_o2": extra["limited_by_o2"],
            "limited_by_h2o": extra["limited_by_h2o"],
            "limited_by_co2": extra["limited_by_co2"],
            "limited_by_h2": extra["limited_by_h2"],
            "wgs_neg_truncated": extra["wgs_neg_truncated"],
            "is_combustion_zone": extra["is_combustion_zone"],
            "cell0_n_distinct_roots": extra["cell0_n_distinct_roots"],
        })
    return rows


# ---------------------------------------------------------------------------
# 装置运行
# ---------------------------------------------------------------------------

def heatloss_summary(system: GasifierSystem, coal_dry_kg_s: float, hhv_kj_kg: float) -> dict:
    """逐格 Q_loss 求和，换成两个分母下的占比。

    P1.9-A 审阅意见：``HeatLossPercent`` 不是整炉热损百分比。模型的实际总热损是

        Σ Q_loss = (HLP/100)·(m_wet·HHV_d)·Σ(dz/L·clip((T/1800)^4, 0.1, 5))

    温度权重在气化区通常远小于 1，所以 Σ/Σ_basis 只有 HLP 的 0.38–0.68 倍。
    这里同时给出

      - ``hloss_share_of_Qtotal_pct``：Σ Q_loss / (m_wet·HHV_d)，即模型现用分母；
      - ``hloss_frac_dry_coal_pct``：Σ Q_loss / (m_dry·HHV_d)，与装置侧热损同一分母。

    两者相差一个 1/(1−Mt) 因子，混用会把口径差异当成物理差异。
    """
    cells = [d for d in system.cell_diagnostics if "hloss_W" in d]
    if not cells:
        return {
            "hloss_actual_MW": float("nan"),
            "hloss_share_of_Qtotal_pct": float("nan"),
            "hloss_frac_dry_coal_pct": float("nan"),
            "hloss_weight_sum": float("nan"),
            "hloss_Qtotal_wet_MW": float("nan"),
            "hloss_Qtotal_dry_MW": float("nan"),
            "hloss_cells": 0,
        }
    total_w = float(sum(float(d["hloss_W"]) for d in cells))
    q_total_wet_w = float(cells[0].get("hloss_Q_total_W", float("nan")))
    q_total_dry_w = float(coal_dry_kg_s) * float(hhv_kj_kg) / 1000.0 * 1e6
    weight_sum = float(sum(float(d["hloss_dz_frac"]) * float(d["hloss_t_weight"]) for d in cells))
    return {
        "hloss_actual_MW": total_w / 1e6,
        "hloss_share_of_Qtotal_pct": total_w / q_total_wet_w * 100.0 if q_total_wet_w > 0 else float("nan"),
        "hloss_frac_dry_coal_pct": total_w / q_total_dry_w * 100.0 if q_total_dry_w > 0 else float("nan"),
        "hloss_weight_sum": weight_sum,
        "hloss_Qtotal_wet_MW": q_total_wet_w / 1e6,
        "hloss_Qtotal_dry_MW": q_total_dry_w / 1e6,
        "hloss_cells": len(cells),
    }


def _device_run(row, csv17, csv18, dry, dz0, hlp, f_gas, carrier_nm3_h) -> tuple[dict, GasifierSystem]:
    n_block = int(round(DEV_ZB / dz0))
    n_cells = n_block + DEV_N_REST
    rec = run_case(
        row,
        n_cells=n_cells,
        coal_flow_dry_kg_s=dry,
        keep_system=True,
        mesh_family="graded_full_length",
        extra_op={
            "HeatLossPercent": float(hlp),
            "GasificationRateFactor": float(f_gas),
            "carrier_gas_flow": p18a._carrier_kg_s(carrier_nm3_h),
            "GradedZoneLength": DEV_ZB,
            "GradedZoneDz": float(dz0),
        },
    )
    system: GasifierSystem = rec["_system"]
    out = p18a._build_row_extras(
        rec, system, row, csv17, csv18, dry, n_cells, f_gas, hlp,
        carrier_nm3_h, "measured", "graded_mesh",
    )
    out.update({
        "dz0_m": float(dz0),
        "z_b_m": DEV_ZB,
        "block_n_cells": n_block,
        "n_rest_cells": DEV_N_REST,
        "mesh_bitwise_uniform_N40": bool(
            np.array_equal(
                graded_mesh(DEV_L, DEV_ZB, dz0, DEV_N_REST),
                uniform_mesh(DEV_L, 40),
            )
        ) if dz0 == BASE_DZ0 else None,
    })
    out.update(block_stats(system, DEV_ZB, dz0))
    out.update(heatloss_summary(
        system, dry,
        float(p18a.coal_hhv_kj_kg(system.coal_props)),
    ))
    return out, system


def device_runs(window_ids, by_id, p17d, p18d) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs, cells = [], []
    for wid in window_ids:
        row = by_id[wid]
        csv17 = p17d.loc[wid]
        csv18 = p18d.loc[wid]
        dry = float(csv17["coal_dry_recon_phase_kg_s"])
        carrier = float(csv18["co2_carrier_measured_nm3_h"])
        for dz0 in DEV_DZ0:
            for hlp in HLP_GRID:
                out, system = _device_run(row, csv17, csv18, dry, dz0, hlp, 1.0, carrier)
                runs.append(out)
                if hlp == 2.0 and system.execution_status == "completed" and system.results:
                    cells.extend(cell_rows(system, wid, dz0, hlp, DEV_ZB))
        print(f"  device {wid[:60]} … dz0 done", flush=True)
    return pd.DataFrame(runs), pd.DataFrame(cells)


def leverage_runs(window_ids, by_id, p17d, p18d, mesh_runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for wid in window_ids:
        row = by_id[wid]
        csv17 = p17d.loc[wid]
        csv18 = p18d.loc[wid]
        dry = float(csv17["coal_dry_recon_phase_kg_s"])
        carrier = float(csv18["co2_carrier_measured_nm3_h"])
        for f_gas in F_LEV_EXTRA:
            # 最细档（graded）
            out, _sys = _device_run(row, csv17, csv18, dry, FINEST_DZ0, 1.0, f_gas, carrier)
            out["leverage_mesh"] = "graded_finest"
            rows.append(out)
            # 均匀 N=40 对照（同一轮重跑，用于逐位核对 P1.8-A）
            rec = run_case(
                row, n_cells=40, coal_flow_dry_kg_s=dry, keep_system=True,
                mesh_family="uniform_full_length",
                extra_op={
                    "HeatLossPercent": 1.0,
                    "GasificationRateFactor": float(f_gas),
                    "carrier_gas_flow": p18a._carrier_kg_s(carrier),
                },
            )
            system = rec["_system"]
            uni = p18a._build_row_extras(
                rec, system, row, csv17, csv18, dry, 40, f_gas, 1.0,
                carrier, "measured", "uniform_control",
            )
            uni["dz0_m"] = BASE_DZ0
            uni["leverage_mesh"] = "uniform_N40"
            uni.update(block_stats(system, DEV_ZB, BASE_DZ0))
            rows.append(uni)
        print(f"  leverage {wid[:60]} done", flush=True)
    frame = pd.DataFrame(rows)
    return frame


def p18a_parity(lev: pd.DataFrame) -> pd.DataFrame:
    """把同一轮重跑的均匀 N=40 行与 P1.8-A `scan.csv` 逐位核对。

    P1.8-A 的 `f_gas = 0.03125` 属 `grid_extension` 行，须一并纳入。
    """
    scan = pd.read_csv(P18A_DIR / "scan.csv")
    scan = scan[
        scan["row_kind"].isin(("grid", "grid_extension"))
        & np.isclose(scan["HeatLossPercent"], 1.0)
        & scan["f_gas"].isin(F_LEV_EXTRA)
    ]
    keys = ("X_C_implied", "T_out_K", "CO2_vol_pct", "CO_vol_pct", "T_cell0_K")
    rows = []
    for _, r in lev[lev["leverage_mesh"] == "uniform_N40"].iterrows():
        match = scan[(scan["window_id"] == r["window_id"])
                      & np.isclose(scan["f_gas"], r["f_gas"])]
        base = {
            "window_id": r["window_id"],
            "f_gas": float(r["f_gas"]),
            "found_in_p18a": bool(not match.empty),
        }
        if match.empty:
            rows.append({**base, "bitwise_equal": False})
            continue
        m = match.iloc[0]
        diffs = {f"d_{k}": float(r[k]) - float(m[k]) for k in keys}
        base.update(diffs)
        base["bitwise_equal"] = bool(all(abs(v) <= 1e-12 for v in diffs.values()))
        rows.append(base)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 锚点运行
# ---------------------------------------------------------------------------

def anchor_mesh_runs() -> pd.DataFrame:
    cases, geo = anchor_cases()
    L = float(geo["L_m"])
    rows = []
    for case in cases:
        if case["case_name"] not in DRY_POWDER_CASES:
            continue
        for dz0 in ANC_DZ0:
            n_block = int(round(ANC_ZB / dz0))
            n_cells = n_block + ANC_N_REST
            rec = solve_fortran_case(
                case, n_cells=n_cells, mode="F", geometry=geo,
                keep_system=True,
                switches={
                    "MeshFamily": "graded_full_length",
                    "GradedZoneLength": ANC_ZB,
                    "GradedZoneDz": float(dz0),
                },
            )
            system = rec.pop("_system", None)
            rec.pop("_profile", None)
            rec.pop("_z", None)
            row = {
                "case_name": rec["case_name"],
                "dz0_m": float(dz0),
                "z_b_m": ANC_ZB,
                "block_n_cells": n_block,
                "N_cells": int(n_cells),
                "execution_status": rec.get("execution_status"),
                "no_root_cell_index": rec.get("no_root_cell_index"),
                "max_residual_cost": rec.get("max_residual_cost"),
                "T_out_K": rec.get("T_gas_K"),
                "X_C_pct": rec.get("X_C_pct"),
                "CO_vol_pct": rec.get("CO_vol_pct"),
                "H2_vol_pct": rec.get("H2_vol_pct"),
                "CO2_vol_pct": rec.get("CO2_vol_pct"),
                "sum_dz_over_L": float(system.mesh_sum_dz / L) if system is not None else float("nan"),
            }
            if system is not None and system.results:
                row.update(block_stats(system, ANC_ZB, dz0))
            rows.append(row)
        print(f"  anchor {case['case_name']} done", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 读法
# ---------------------------------------------------------------------------

def _pick(df: pd.DataFrame, wid: str, dz0: float, hlp: float) -> pd.Series | None:
    sel = df[(df["window_id"] == wid) & (np.isclose(df["dz0_m"], dz0))
             & (np.isclose(df["HeatLossPercent"], hlp))]
    return sel.iloc[0] if not sel.empty else None


def readings(mesh_runs: pd.DataFrame, lev: pd.DataFrame, window_ids) -> pd.DataFrame:
    rows: list[dict] = []

    def add(**kw):
        rows.append(kw)

    # ---- Q1：最细两档收敛 ------------------------------------------------
    for hlp in HLP_GRID:
        for wid in window_ids:
            a = _pick(mesh_runs, wid, SECOND_FINEST_DZ0, hlp)
            b = _pick(mesh_runs, wid, FINEST_DZ0, hlp)
            if a is None or b is None:
                continue
            dxc = (float(b["X_C_implied"]) - float(a["X_C_implied"])) * 100.0
            dt = float(b["T_out_K"]) - float(a["T_out_K"])
            dco2 = float(b["CO2_vol_pct"]) - float(a["CO2_vol_pct"])
            ok = (abs(dxc) < Q1_DXC_PP) and (abs(dt) < Q1_DT_K) and (abs(dco2) < Q1_DCO2_VOL)
            add(reading="Q1", window_id=wid, HeatLossPercent=hlp, metric="dX_C_pp",
                value=dxc, threshold=Q1_DXC_PP, pass_=bool(abs(dxc) < Q1_DXC_PP),
                note="0.0103 m → 0.00515 m")
            add(reading="Q1", window_id=wid, HeatLossPercent=hlp, metric="dT_out_K",
                value=dt, threshold=Q1_DT_K, pass_=bool(abs(dt) < Q1_DT_K), note="")
            add(reading="Q1", window_id=wid, HeatLossPercent=hlp, metric="dCO2_vol_pct",
                value=dco2, threshold=Q1_DCO2_VOL, pass_=bool(abs(dco2) < Q1_DCO2_VOL), note="")
            add(reading="Q1", window_id=wid, HeatLossPercent=hlp, metric="window_converged",
                value=float(ok), threshold=float("nan"), pass_=bool(ok),
                note="三项同时满足")
    # 汇总（按热损档）
    for hlp in HLP_GRID:
        sub = [r for r in rows if r["reading"] == "Q1" and r.get("metric") == "window_converged"
               and r.get("HeatLossPercent") == hlp]
        add(reading="Q1_verdict", window_id="ALL", HeatLossPercent=hlp, metric="converged_windows",
            value=float(sum(1 for r in sub if r["pass_"])), threshold=float(len(sub)),
            pass_=bool(sub and all(r["pass_"] for r in sub)),
            note="燃烧区已分辨 = 所有窗三项同时满足")

    # ---- Q2：最细档相对基准档的效应量 ------------------------------------
    for hlp in HLP_GRID:
        for wid in window_ids:
            a = _pick(mesh_runs, wid, BASE_DZ0, hlp)
            b = _pick(mesh_runs, wid, FINEST_DZ0, hlp)
            if a is None or b is None:
                continue
            dxc = (float(b["X_C_implied"]) - float(a["X_C_implied"])) * 100.0
            dt = float(b["T_out_K"]) - float(a["T_out_K"])
            dco2 = float(b["CO2_vol_pct"]) - float(a["CO2_vol_pct"])
            trigger = (abs(dxc) >= Q2_DXC_PP) or (abs(dt) >= Q2_DT_K) or (abs(dco2) >= Q2_DCO2_VOL)
            add(reading="Q2", window_id=wid, HeatLossPercent=hlp, metric="dX_C_pp",
                value=dxc, threshold=Q2_DXC_PP, pass_=bool(abs(dxc) >= Q2_DXC_PP),
                note="0.2575 m → 0.00515 m（pass=触发阈）")
            add(reading="Q2", window_id=wid, HeatLossPercent=hlp, metric="dT_out_K",
                value=dt, threshold=Q2_DT_K, pass_=bool(abs(dt) >= Q2_DT_K), note="")
            add(reading="Q2", window_id=wid, HeatLossPercent=hlp, metric="dCO2_vol_pct",
                value=dco2, threshold=Q2_DCO2_VOL, pass_=bool(abs(dco2) >= Q2_DCO2_VOL), note="")
            add(reading="Q2", window_id=wid, HeatLossPercent=hlp, metric="window_triggered",
                value=float(trigger), threshold=float("nan"), pass_=bool(trigger),
                note="任一判据满足")
    for hlp in HLP_GRID:
        sub = [r for r in rows if r["reading"] == "Q2" and r.get("metric") == "window_triggered"
               and r.get("HeatLossPercent") == hlp]
        add(reading="Q2_verdict", window_id="ALL", HeatLossPercent=hlp, metric="triggered_windows",
            value=float(sum(1 for r in sub if r["pass_"])), threshold=0.0,
            pass_=bool(any(r["pass_"] for r in sub)),
            note="pass = 现有基线含火焰区分辨率误差，需在细化网格上重建")

    # ---- Q3：火焰区结构收敛 ----------------------------------------------
    for wid in window_ids:
        for metric, col in (
            ("block_T_peak_K", "block_T_peak_K"),
            ("z_O2_one_pct_m", "z_O2_one_pct_m"),
            ("combustion_zone_length_m", "combustion_zone_length_m"),
            ("block_len_frac_co2_limited", "block_len_frac_co2_limited"),
            ("block_len_frac_h2o_limited", "block_len_frac_h2o_limited"),
        ):
            a = _pick(mesh_runs, wid, SECOND_FINEST_DZ0, 2.0)
            b = _pick(mesh_runs, wid, FINEST_DZ0, 2.0)
            if a is None or b is None:
                continue
            va, vb = float(a[col]), float(b[col])
            add(reading="Q3", window_id=wid, HeatLossPercent=2.0, metric=metric + "_finest",
                value=vb, threshold=float("nan"), pass_=None, note="最细档数值")
            add(reading="Q3", window_id=wid, HeatLossPercent=2.0, metric=metric + "_delta_2finest",
                value=vb - va, threshold=float("nan"), pass_=None, note="0.0103 → 0.00515")

    # ---- Q4：杠杆 --------------------------------------------------------
    dxc_by = {}
    for wid in window_ids:
        base_uni = lev[(lev["window_id"] == wid) & (lev["leverage_mesh"] == "uniform_N40")]
        fine = lev[(lev["window_id"] == wid) & (lev["leverage_mesh"] == "graded_finest")]
        if base_uni.empty or fine.empty:
            continue
        m1u = base_uni[np.isclose(base_uni["f_gas"], 1.0)]["X_C_implied"]
        m32u = base_uni[np.isclose(base_uni["f_gas"], F_LEV_EXTRA[1])]["X_C_implied"]
        m1f = fine[np.isclose(fine["f_gas"], 1.0)]["X_C_implied"]
        m32f = fine[np.isclose(fine["f_gas"], F_LEV_EXTRA[1])]["X_C_implied"]
        if m1u.empty or m32u.empty:
            # f_gas=1 的均匀 N=40 行在 mesh_runs 的基准档
            m1u = pd.Series([_pick(mesh_runs, wid, BASE_DZ0, 1.0)["X_C_implied"]])
        if m1f.empty:
            m1f = pd.Series([_pick(mesh_runs, wid, FINEST_DZ0, 1.0)["X_C_implied"]])
        d_uni = (float(m32u.iloc[0]) - float(m1u.iloc[0])) * 100.0
        d_fin = (float(m32f.iloc[0]) - float(m1f.iloc[0])) * 100.0
        ratio = abs(d_fin) / abs(d_uni) if abs(d_uni) > 1e-12 else float("inf")
        dxc_by[wid] = (d_uni, d_fin, ratio)
        add(reading="Q4", window_id=wid, HeatLossPercent=1.0, metric="dX_C_uniform_N40_pp",
            value=d_uni, threshold=float("nan"), pass_=None, note="f_gas 1 → 1/32")
        add(reading="Q4", window_id=wid, HeatLossPercent=1.0, metric="dX_C_graded_finest_pp",
            value=d_fin, threshold=float("nan"), pass_=None, note="f_gas 1 → 1/32")
        add(reading="Q4", window_id=wid, HeatLossPercent=1.0, metric="ratio_finest_over_uniform",
            value=ratio, threshold=Q4_RATIO, pass_=bool(ratio >= Q4_RATIO),
            note="中位比值 ≥ 3 记为恢复杠杆")
    ratios = [v[2] for v in dxc_by.values() if math.isfinite(v[2])]
    med = float(np.median(ratios)) if ratios else float("nan")
    add(reading="Q4_verdict", window_id="ALL", HeatLossPercent=1.0, metric="median_ratio",
        value=med, threshold=Q4_RATIO, pass_=bool(math.isfinite(med) and med >= Q4_RATIO),
        note="中位比值 ≥ 3 → 细化后本征动力学恢复杠杆")

    df = pd.DataFrame(rows)
    if "pass_" in df.columns:
        df = df.rename(columns={"pass_": "pass"})
    return df


# ---------------------------------------------------------------------------
# 解释文本
# ---------------------------------------------------------------------------

def _fmt(x, nd=3):
    if x is None:
        return ""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(x):
        return "nan"
    return f"{x:.{nd}f}"


def _md_table(df: pd.DataFrame, cols: list[str], nd: int = 2) -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r.get(c, "")
            cells.append(_fmt(v, nd) if isinstance(v, (int, float, np.floating)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def interpretation(unit_df, mesh_runs, mesh_cells, lev, anchor_df, rd, parity=None) -> str:
    L: list[str] = []
    L.append("# P1.9-A 燃烧区分辨率受控试验 — 判读\n")
    L.append("Status: `EXECUTED — model repo`\n")
    L.append("依据：`manuscript/model-side-handoff-p1.9a-20261105.md`；"
             "上游核查 `manuscript/model-evidence-verification-20261105.md` F2/F3/F6。\n")
    L.append("## 0. 冻结与网格\n")
    L.append("热损 `hhv_percent`、`f_gas = 1`（第 3 节除外）、煤量协调-阶段主口径、"
             "模型入口载气 = P1.8-D 实测口径；速率式、限幅、燃烧区判据（pO2 > 0.05 atm）、"
             "求解器容差均未改动。\n")
    L.append("### 单元检查\n")
    L.append(_md_table(unit_df, ["group", "z_b_m", "dz0_m", "n_block", "N_cells",
                                 "sum_dz_minus_L_m", "dz_max_m", "dz_min_m",
                                 "bitwise_equal_uniform_N40", "pass"], nd=9))
    L.append("")
    par = unit_df[unit_df["group"] == "device_parity"]
    bit = bool(par["bitwise_equal_uniform_N40"].iloc[0]) if not par.empty else False
    L.append(f"- `dz0 = {BASE_DZ0} m` 与均匀 N = 40 逐位相同：**{bit}**"
             f"（Σdz − L = {_fmt(float(par['sum_dz_minus_L_m'].iloc[0]), 15)} m）\n")

    L.append("### 与 P1.8-A 的逐位核对\n")
    if parity is not None and not parity.empty:
        all_eq = bool(parity["bitwise_equal"].all())
        L.append(f"同一轮重跑的均匀 N=40 行与 P1.8-A `scan.csv`（`grid` + `grid_extension`）"
                 f"逐位比较，{len(parity)} 行全部相同：**{all_eq}**。\n")
        L.append(_md_table(parity, ["window_id", "f_gas", "d_X_C_implied", "d_T_out_K",
                                    "d_CO2_vol_pct", "d_T_cell0_K", "bitwise_equal"], nd=12))
        L.append("")
    else:
        L.append("未提供 P1.8-A 对照行。\n")

    # Q1
    L.append("## 1. Q1：最细两档收敛\n")
    L.append(f"判据：最细两档之间 |ΔX_C| < {Q1_DXC_PP} pp、|ΔT_out| < {Q1_DT_K} K、"
             f"|ΔCO2| < {Q1_DCO2_VOL} vol% 同时成立记为「燃烧区已分辨」。\n")
    q1 = rd[(rd["reading"] == "Q1") & (rd["metric"] == "window_converged")].copy()
    q1["pass"] = q1["pass"].astype(bool).astype(int)
    piv = q1.pivot_table(index="window_id", columns="HeatLossPercent", values="pass", aggfunc="first")
    L.append(_md_table(piv.reset_index().rename(columns={1.0: "HLP=1%", 2.0: "HLP=2%"}),
                       ["window_id", "HLP=2%", "HLP=1%"]))
    L.append("")
    for hlp in HLP_GRID:
        v = rd[(rd["reading"] == "Q1_verdict") & (np.isclose(rd["HeatLossPercent"], hlp))]
        if not v.empty:
            L.append(f"- HLP = {hlp:.0f}%：{_fmt(v['value'].iloc[0], 0)} / {_fmt(v['threshold'].iloc[0], 0)} 窗收敛")
    L.append("")
    d1 = rd[(rd["reading"] == "Q1") & (rd["metric"] == "dX_C_pp")]
    d2 = rd[(rd["reading"] == "Q1") & (rd["metric"] == "dT_out_K")]
    d3 = rd[(rd["reading"] == "Q1") & (rd["metric"] == "dCO2_vol_pct")]
    for tag, d in (("ΔX_C (pp)", d1), ("ΔT_out (K)", d2), ("ΔCO2 (vol%)", d3)):
        if not d.empty:
            L.append(f"- {tag} 范围：{_fmt(d['value'].min())} … {_fmt(d['value'].max())}，"
                     f"中位 |·| = {_fmt(d['value'].abs().median())}")
    L.append("")
    o2pos = mesh_runs[np.isclose(mesh_runs["HeatLossPercent"], 2.0)].groupby("dz0_m")[
        ["z_O2_one_pct_m", "combustion_zone_length_m", "block_T_peak_K"]].median()
    L.append("**读法限定**：Q1 未收敛，因此上面两个网格解之间的差异是**未收敛网格间已观测的差异**，"
             "不能称为「基线相对收敛解误差的下界」。没有收敛极限、渐近阶数或误差有符号单调性的依据时，"
             "更细的网格可能使差异回转，当前两层也可能分别位于极限两侧。"
             "Q1 未通过和 Q2 触发这两条结论不受影响：**在当前可执行的网格族上，基线的出口量对燃烧区"
             "首格厚度敏感的幅度就是 33–43 K 量级**，这已足以说明基线含未分辨的火焰区效应。\n")

    L.append("**结构判读**：O2 降到入口 1% 的位置在任何 `dz0` 下都不超过两个细格"
             "（即 O2 始终在**首格内**耗尽），燃烧区判据覆盖的物理长度也随之只有一个细格。"
             "因此加密只改变首格厚度，不把火焰区拆成多格——这正是 Q1 不收敛的机制。"
             "块内峰值温度随 `dz0` 非单调（见下表），最细档反而回升。\n")
    L.append(_md_table(o2pos.reset_index(), ["dz0_m", "z_O2_one_pct_m",
                                             "combustion_zone_length_m", "block_T_peak_K"], nd=5))
    L.append("")

    # HLP 与实际热损
    L.append("### HeatLossPercent 不是整炉热损百分比\n")
    L.append("模型逐格热损为 `Q_loss = (HLP/100)·(m_coal·HHV_d)·(dz/L)·clip((T/1800)^4, 0.1, 5)`。"
             "温度权重在气化区通常远小于 1（1400 K 约 0.366，1200 K 约 0.198），"
             "只在燃烧区接近或触及 5 的上限，因此 **Σ Q_loss 只有 HLP 的 0.38–0.68 倍**；"
             "另外分母用 `coal_flow`（湿煤）乘干基 HHV，与干煤口径还差一个 1/(1−Mt) 因子。"
             "两个分母下的实测占比：\n")
    hl = mesh_runs[mesh_runs["execution_status"] == "completed"].copy()
    agg = hl.groupby("HeatLossPercent").agg(
        n=("hloss_actual_MW", "size"),
        share_min=("hloss_share_of_Qtotal_pct", "min"),
        share_max=("hloss_share_of_Qtotal_pct", "max"),
        dry_min=("hloss_frac_dry_coal_pct", "min"),
        dry_max=("hloss_frac_dry_coal_pct", "max"),
        MW_min=("hloss_actual_MW", "min"),
        MW_max=("hloss_actual_MW", "max"),
        w_min=("hloss_weight_sum", "min"),
        w_max=("hloss_weight_sum", "max"),
    ).reset_index()
    L.append(_md_table(agg, ["HeatLossPercent", "n", "MW_min", "MW_max", "share_min", "share_max",
                             "dry_min", "dry_max", "w_min", "w_max"], nd=4))
    L.append("")
    L.append("- `share_*` = Σ Q_loss / (m_湿煤·HHV_d)，即模型自己用的分母；"
             "`dry_*` = Σ Q_loss / (m_干煤·HHV_d)，与装置侧同一分母。")
    L.append("- `w_*` = Σ(dz/L·clip((T/1800)^4,0.1,5))，即 HLP 到实际占比的换算系数。"
             "它随网格变化（0.38–0.68），所以**不同 `dz0` 之间的 Q_loss 也不完全相同**，"
             "尽管 HLP 是常数。")
    L.append("- 因此本轮的 2% 与 1% 必须分别读成上表的两行实际占比；"
             "它们**不是**「整炉热损 2% 与 1%」，也不能直接与实测水冷壁吸热比较。\n")

    # Q2
    L.append("## 2. Q2：现有基线是否含火焰区分辨率误差\n")
    L.append(f"判据：最细档相对 `dz0 = {BASE_DZ0} m`，任一窗 |ΔX_C| ≥ {Q2_DXC_PP} pp、"
             f"或 |ΔT_out| ≥ {Q2_DT_K} K、或 |ΔCO2| ≥ {Q2_DCO2_VOL} vol%，"
             "记为「需在细化网格上重建」。\n")
    for hlp in HLP_GRID:
        sub = mesh_runs[np.isclose(mesh_runs["HeatLossPercent"], hlp)]
        base = sub[np.isclose(sub["dz0_m"], BASE_DZ0)].set_index("window_id")
        fine = sub[np.isclose(sub["dz0_m"], FINEST_DZ0)].set_index("window_id")
        common = base.index.intersection(fine.index)
        tbl = pd.DataFrame({
            "window_id": common,
            "X_C_base": (base.loc[common, "X_C_implied"] * 100).values,
            "X_C_fine": (fine.loc[common, "X_C_implied"] * 100).values,
            "dX_C_pp": ((fine.loc[common, "X_C_implied"] - base.loc[common, "X_C_implied"]) * 100).values,
            "T_base_K": base.loc[common, "T_out_K"].values,
            "T_fine_K": fine.loc[common, "T_out_K"].values,
            "dT_K": (fine.loc[common, "T_out_K"] - base.loc[common, "T_out_K"]).values,
            "dCO2_vol": (fine.loc[common, "CO2_vol_pct"] - base.loc[common, "CO2_vol_pct"]).values,
        })
        L.append(f"### HLP = {hlp:.0f}%\n")
        L.append(_md_table(tbl, ["window_id", "X_C_base", "X_C_fine", "dX_C_pp",
                                 "T_base_K", "T_fine_K", "dT_K", "dCO2_vol"], nd=3))
        L.append("")
        v = rd[(rd["reading"] == "Q2_verdict") & (np.isclose(rd["HeatLossPercent"], hlp))]
        if not v.empty:
            L.append(f"- 触发窗口数：{_fmt(v['value'].iloc[0], 0)} / {len(tbl)}；"
                     f"裁决 = **{'需在细化网格上重建' if bool(v['pass'].iloc[0]) else '不触发'}**\n")

    L.append("**X_C 贴顶的截断**：模型 X_C 在部分窗口已达完全转化上限，此时指标的分辨率已经"
             "饱和（ΔX_C 无法反映真实差异），而不是给出一个可以量化的误差幅度。"
             "这些窗口的裁决只能由 ΔT_out 与 ΔCO2 承担。逐档计数如下。\n")
    ceil_rows = []
    for hlp in HLP_GRID:
        sub = mesh_runs[np.isclose(mesh_runs["HeatLossPercent"], hlp)]
        for dz0 in (BASE_DZ0, FINEST_DZ0):
            s = sub[np.isclose(sub["dz0_m"], dz0)]
            ceil_rows.append({
                "HLP_pct": hlp, "dz0_m": dz0, "n_windows": int(len(s)),
                "X_C_ge_0.9999": int((s["X_C_implied"] >= 0.9999).sum()),
                "X_C_median_pct": float(s["X_C_implied"].median() * 100.0),
            })
    L.append(_md_table(pd.DataFrame(ceil_rows), ["HLP_pct", "dz0_m", "n_windows",
                                                 "X_C_ge_0.9999", "X_C_median_pct"], nd=4))
    L.append("")

    # Q3
    L.append("## 3. Q3：火焰区结构（最细档）\n")
    q3 = rd[(rd["reading"] == "Q3") & (rd["metric"].str.endswith("_finest"))]
    if not q3.empty:
        piv3 = q3.pivot_table(index="window_id", columns="metric", values="value", aggfunc="first")
        L.append(_md_table(piv3.reset_index(), ["window_id"] + list(piv3.columns), nd=4))
        L.append("")
    L.append(f"块内限幅绑定的长度份额与峰值温度在 0.0103 → 0.00515 m 之间的变化见上表；"
             "结构量是否收敛到有限值按 Δ 列判断（未设硬阈值，卡面只要求如实报告）。\n")

    # Q4
    L.append("## 4. Q4：细化后本征动力学是否恢复杠杆\n")
    q4 = rd[(rd["reading"] == "Q4") & (rd["metric"] == "ratio_finest_over_uniform")]
    if not q4.empty:
        tbl4 = q4[["window_id", "value"]].rename(columns={"value": "ratio_finest_over_uniform"})
        du = rd[(rd["reading"] == "Q4") & (rd["metric"] == "dX_C_uniform_N40_pp")][["window_id", "value"]]
        du = du.rename(columns={"value": "dX_C_uniform_pp"})
        dfx = rd[(rd["reading"] == "Q4") & (rd["metric"] == "dX_C_graded_finest_pp")][["window_id", "value"]]
        dfx = dfx.rename(columns={"value": "dX_C_finest_pp"})
        tbl4 = tbl4.merge(du, on="window_id").merge(dfx, on="window_id")
        L.append(_md_table(tbl4, ["window_id", "dX_C_uniform_pp", "dX_C_finest_pp",
                                  "ratio_finest_over_uniform"], nd=3))
        L.append("")
    v4 = rd[rd["reading"] == "Q4_verdict"]
    if not v4.empty:
        L.append(f"- 比值中位 = **{_fmt(v4['value'].iloc[0])}**（阈值 ≥ {Q4_RATIO}）→ "
                 f"**{'本征动力学仍可作反演对象' if bool(v4['pass'].iloc[0]) else '不应把本征指前因子当作主要调节量'}**\n")
        L.append("比值 < 1 说明细化后 `f_gas` 的作用不是变大而是变小：燃烧区越细，"
                 "气化反应的份额越集中在首格内被限幅封顶的那一段，指前因子的杠杆进一步收窄。\n")

    # 锚点
    L.append("## 5. 锚点 20 行（模式 F，只记录）\n")
    if anchor_df is not None and not anchor_df.empty:
        L.append(_md_table(anchor_df, ["case_name", "dz0_m", "N_cells", "execution_status",
                                       "T_out_K", "X_C_pct", "CO2_vol_pct",
                                       "block_T_peak_K", "z_O2_one_pct_m"], nd=3))
        L.append("")
        L.append("锚点同样呈现「O2 在首格内耗尽、峰值温度随细化上升、X_C 与 T 未收敛」的形态；"
                 "本表不作通过与否判定。\n")
    else:
        L.append("未生成锚点表（本轮未跑 anchor 阶段）。\n")

    # 求解状态
    L.append("## 6. 求解状态\n")
    bad = mesh_runs[mesh_runs["execution_status"] != "completed"]
    L.append(f"- `no_root` 或未完成行数：{len(bad)} / {len(mesh_runs)}")
    if not bad.empty:
        L.append(_md_table(bad, ["window_id", "dz0_m", "HeatLossPercent", "execution_status",
                                 "no_root_cell_index", "max_residual_cost"], nd=6))
    L.append(f"- 最大残差成本：{float(mesh_runs['max_residual_cost'].max()):.3e}")
    L.append(f"- 单次求解墙钟中位：{_fmt(mesh_runs['solve_wall_time_s'].median())} s，"
             f"最大 {_fmt(mesh_runs['solve_wall_time_s'].max())} s\n")
    # 对 F4 / R3 的更正
    L.append("## 7. 对 P1.8-A F4 与 R3 的更正\n")
    L.append("F4 曾写「R3 的最优点全部落在 6% 上边界……这些组合与实测热损不符」，"
             "并据此判定 R3 的「可分别辨识」没有意义。按上面的换算，这个推理不成立：\n")
    L.append("- R3 的 `HLP = 4–6%` 换算成实际热损不是 4–6%，而是约为其 0.38–0.68 倍。"
             "用本表的换算系数粗估落在 **1.5–4.1%** 量级；且 HLP 升高会拉低温度、"
             "使权重进一步下降，因此真实值应偏向该区间低端。这是整炉总热损的候选值，"
             "不是「物理上不可能」的点。")
    L.append("- 更关键的是比较对象不同：实测的 0.421–0.909% 是**水冷壁汽包吸热**，"
             "是整炉热损的一个组成部分（还有渣、其他冷却回路与散热）。"
             "在汽包只覆盖反应器边界内回路的前提下才有 总热损 ≥ 汽包吸热，"
             "而这需要在装置热平衡上单独确认，本轮没有做。"
             "因此实测值只能给出一侧约束，不能把 R3 的最优点排除。")
    L.append("- 本轮的 2% 与 1% 换算成 0.78–1.39% 与 0.40–0.75%（干煤口径），"
             "**两侧分别跨在实测汽包吸热区间之上与之内**，属于把不确定区间夹住，"
             "而不是「2% 偏保守」。原先「1–2% 必然保守、不会高估动力学杠杆」的说法撤回。")
    L.append("- 需要保留的限制：本轮与 P1.8-A 都没有做装置热平衡，"
             "汽包吸热与整炉热损的差额（渣显热、其他回路、散热）未量化。\n")
    L.append("**对 Q4 的影响**：Q4 的比值 0.62 是在 `HLP = 1%`（实际 0.40–0.75%）下得到的。"
             "热损档位改变会改变温度水平，从而可能改变杠杆比值；本轮**没有**在 2% 档重复"
             "杠杆试验（卡面只指定 1% 档），所以 Q4 的结论限定在 1% 档。\n")

    L.append("## 8. 复现边界\n")
    L.append("- 本轮只改网格，不调任何动力学常数、不采纳任何参数值。")
    L.append("- Q1–Q4 的阈值在运行前写入脚本常量，运行后未改。")
    L.append("- 锚点 20 行只用模式 F（`hhv_percent`），只记录、不判通过。\n")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=("unit", "device", "anchor", "leverage", "readings", "all"))
    ap.add_argument("--source-commit", default=None)
    args = ap.parse_args()
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

    unit_df = pd.concat([unit_checks(), pd.DataFrame(heatloss_unit_checks())], ignore_index=True)
    gate = unit_df[unit_df["pass"].notna()]
    unit_pass = bool(gate["pass"].astype(bool).all())
    unit_df.to_csv(OUT_DIR / "unit_checks.csv", index=False)
    print("unit checks:", "PASS" if unit_pass else "FAIL", flush=True)
    if not unit_pass:
        raise SystemExit("停止：网格单元检查未通过")
    par = unit_df[unit_df["group"] == "device_parity"]
    if par.empty or not bool(par["bitwise_equal_uniform_N40"].iloc[0]):
        raise SystemExit("停止：dz0 = 0.2575 m 未逐位复现均匀 N = 40")
    if args.stage == "unit":
        return

    # ---- 装置 ----
    if args.stage in ("device", "readings", "all"):
        if (OUT_DIR / "mesh_runs.csv").exists() and args.stage == "readings":
            mesh_runs = pd.read_csv(OUT_DIR / "mesh_runs.csv")
            mesh_cells = pd.read_csv(OUT_DIR / "mesh_cells.csv")
        else:
            print("device runs …", flush=True)
            mesh_runs, mesh_cells = device_runs(window_ids, by_id, p17d, p18d)
            mesh_runs.to_csv(OUT_DIR / "mesh_runs.csv", index=False)
            mesh_cells.to_csv(OUT_DIR / "mesh_cells.csv", index=False)
        print(f"  mesh_runs {mesh_runs.shape}, mesh_cells {mesh_cells.shape}", flush=True)

    # ---- 锚点 ----
    if args.stage in ("anchor", "all"):
        print("anchor runs …", flush=True)
        anchor_df = anchor_mesh_runs()
        anchor_df.to_csv(OUT_DIR / "anchor_mesh.csv", index=False)
    else:
        anchor_df = (pd.read_csv(OUT_DIR / "anchor_mesh.csv")
                     if (OUT_DIR / "anchor_mesh.csv").exists() else pd.DataFrame())

    # ---- 杠杆 ----
    if args.stage in ("leverage", "readings", "all"):
        if (OUT_DIR / "leverage.csv").exists() and args.stage == "readings":
            lev = pd.read_csv(OUT_DIR / "leverage.csv")
        else:
            print("leverage runs …", flush=True)
            lev = leverage_runs(window_ids, by_id, p17d, p18d, mesh_runs)
            lev.to_csv(OUT_DIR / "leverage.csv", index=False)
        print(f"  leverage {lev.shape}", flush=True)
    else:
        lev = pd.read_csv(OUT_DIR / "leverage.csv") if (OUT_DIR / "leverage.csv").exists() else pd.DataFrame()

    if args.stage == "device" or args.stage == "anchor" or args.stage == "leverage":
        return

    # ---- 读法 ----
    parity = p18a_parity(lev)
    rd = readings(mesh_runs, lev, window_ids)
    rd.to_csv(OUT_DIR / "readings.csv", index=False)

    protocol = base_protocol({
        "round": "P1.9-A / 2026-11-05",
        "model_repo": "gasifier-1d-kinetic",
        "source_commit": args.source_commit or source_commit(),
        "source_commit_note": "生成本轮全部产物的代码提交；产物随后另起一次仅含 outputs 的提交",
        "task_card": "papers/jax-gasifier-cascade-hybrid/manuscript/model-side-handoff-p1.9a-20261105.md",
        "question": "单独细化燃烧区后，出口 X_C、温度、组成以及气化指前因子的杠杆是否改变",
        "mesh_family": {
            "name": "graded_full_length",
            "device": {"L_m": DEV_L, "z_b_m": DEV_ZB, "n_rest_cells": DEV_N_REST,
                       "dz0_grid_m": list(DEV_DZ0),
                       "n_block": {str(k): int(round(DEV_ZB / k)) for k in DEV_DZ0},
                       "N_cells": {str(k): v for k, v in DEV_N.items()}},
            "anchor": {"z_b_m": ANC_ZB, "n_rest_cells": ANC_N_REST,
                       "dz0_grid_m": list(ANC_DZ0)},
            "rule": "块内 n_block = round(z_b/dz0) 个等长 dz0；其余段 (L − n_block·dz0)/N_rest；"
                    "Σdz = L 逐位成立",
            "implementation": "src/model/grid_service.py MeshFamily='graded_full_length'",
        },
        "frozen_items_untouched": [
            "all Arrhenius constants (except f_gas scanned in section 3)",
            "phi", "f_co2 = 1.0", "P_O2_Combustion_atm = 0.05",
            "CharCombustionRateFactor", "H2_OxidationFactor / CO_OxidationFactor",
            "pyrolysis yields", "negative-WGS hydrogen truncation（只统计绑定）",
            "USE_RK_GILL_COMBUSTION = False", "ParticleSizeModel = fortran_constant",
            "steam_flow = 0", "low-heating-value basis", "SolverStrictRoot = True",
            "HeatLossModel = hhv_percent",
        ],
        "runs": {
            "device": {"windows": len(window_ids), "dz0": len(DEV_DZ0),
                       "heat_loss_pct": list(HLP_GRID), "expected_rows": len(window_ids) * len(DEV_DZ0) * 2},
            "anchor": {"cases": list(DRY_POWDER_CASES), "dz0": len(ANC_DZ0), "expected_rows": 20},
            "leverage": {"finest_dz0_m": FINEST_DZ0, "f_gas_extra": list(F_LEV_EXTRA),
                         "control": "uniform N=40（同一轮重跑，逐位核对 P1.8-A）"},
        },
        "predeclared_readings": {
            "Q1": f"最细两档 |ΔX_C| < {Q1_DXC_PP} pp、|ΔT_out| < {Q1_DT_K} K、|ΔCO2| < {Q1_DCO2_VOL} vol%",
            "Q2": f"最细档相对 {BASE_DZ0} m：任一窗 |ΔX_C| ≥ {Q2_DXC_PP} pp 或 |ΔT_out| ≥ {Q2_DT_K} K "
                  f"或 |ΔCO2| ≥ {Q2_DCO2_VOL} vol%",
            "Q3": "最细档块内限幅绑定长度份额、峰值温度、O2 1% 位置是否收敛到有限值（只报告）",
            "Q4": f"最细档 f_gas 1→1/32 的 ΔX_C 与均匀 N=40 的比值，中位 ≥ {Q4_RATIO} 记为恢复杠杆",
        },
        "unit_checks_all_pass": unit_pass,
        "heatloss_basis": {
            "finding": "HeatLossPercent 不是整炉热损百分比（2026-11-05 审阅意见）",
            "formula": "Q_loss = (HLP/100)·(m_coal·HHV_d)·(dz/L_norm)·clip((T/1800)^4, 0.1, 5)",
            "weight_sum_range": [float(mesh_runs["hloss_weight_sum"].min()),
                                 float(mesh_runs["hloss_weight_sum"].max())],
            "actual_frac_dry_coal_pct": {
                "HLP_2": [float(mesh_runs[np.isclose(mesh_runs["HeatLossPercent"], 2.0)]
                                ["hloss_frac_dry_coal_pct"].min()),
                          float(mesh_runs[np.isclose(mesh_runs["HeatLossPercent"], 2.0)]
                                ["hloss_frac_dry_coal_pct"].max())],
                "HLP_1": [float(mesh_runs[np.isclose(mesh_runs["HeatLossPercent"], 1.0)]
                                ["hloss_frac_dry_coal_pct"].min()),
                          float(mesh_runs[np.isclose(mesh_runs["HeatLossPercent"], 1.0)]
                                ["hloss_frac_dry_coal_pct"].max())],
            },
            "code_change": "cell.heat_loss_percent_hloss() 抽出 _calc_energy_balance 的内联算式"
                           "（运算顺序不变，P1.8-A 18/18 逐位复现）；诊断中新增 hloss_W 等字段",
            "withdrawn_claims": [
                "P1.8-A F4 的『R3 最优点 HLP=4–6% 与实测热损不符 / 物理上不可能』",
                "P1.9-A 的『1–2% 相对实测必然保守、不会高估动力学杠杆』",
                "P1.9-D 的『0.42–0.91% 是实测热损上限』",
            ],
        },
        "readings_wording": {
            "q1_failure": "Q1 未收敛 → Q2 的两个网格解之差是『未收敛网格间已观测差异』，"
                          "不得表述为『相对收敛解误差的下界』；Q1 未通过与 Q2 触发不变",
            "x_c_pinning": "X_C 贴顶表示指标分辨率饱和，不能量化潜在误差幅度",
        },
        "bitwise_uniform_reproduction": bool(par["bitwise_equal_uniform_N40"].iloc[0]),
        "p18a_parity": {
            "rule": "本轮同一轮重跑的均匀 N=40 行与 P1.8-A scan.csv 的 grid / grid_extension 行逐位比较",
            "all_bitwise_equal": bool(parity["bitwise_equal"].all()) if not parity.empty else False,
            "n_rows": int(len(parity)),
        },
        "source_csv": {
            "p17d": {"round": "p1.7d-20261103", "file": P17D_CSV.name,
                     "sha256": hashlib.sha256(P17D_CSV.read_bytes()).hexdigest(), "read_only": True},
            "p18d": {"round": "p1.8d-20261104", "file": P18D_CSV.name,
                     "sha256": hashlib.sha256(P18D_CSV.read_bytes()).hexdigest(), "read_only": True},
            "data_repo": "gasifier-data-prep（本轮只读，未修改任何文件）",
        },
        "outputs": ["unit_checks.csv", "mesh_runs.csv", "mesh_cells.csv", "anchor_mesh.csv",
                    "leverage.csv", "parity_p18a.csv", "readings.csv", "protocol.json",
                    "interpretation.md", "sha256.txt"],
    })

    text = interpretation(unit_df, mesh_runs, mesh_cells, lev, anchor_df, rd, parity)
    written = write_round_outputs(
        OUT_DIR,
        tables={
            "unit_checks.csv": unit_df,
            "mesh_runs.csv": mesh_runs,
            "mesh_cells.csv": mesh_cells,
            "anchor_mesh.csv": anchor_df if not anchor_df.empty else pd.DataFrame(columns=["case_name"]),
            "leverage.csv": lev,
            "parity_p18a.csv": parity,
            "readings.csv": rd,
        },
        protocol=protocol,
        interpretation=text,
    )
    print("written:", [p.name for p in written], flush=True)


if __name__ == "__main__":
    main()
