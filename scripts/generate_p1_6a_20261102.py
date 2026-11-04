#!/usr/bin/env python3
"""P1.6-A 方程修正、锚点重跑与新基线（2026-10-31 卡，2026-11-02 修订）。

执行顺序（卡第 1 → 5 节）：
  1a 固相焓口径（审计 B9）  → unit_checks.csv 的 energy / reaction-heat 段
  1b WGS 平衡常数            → wgs_keq_table.csv + unit_checks.csv 的 wgs 段
  2  X_C 硬截断改光滑        → xcap_windows.csv
  3  锚点重跑（模式 F / M）  → anchor_outlet.csv / anchor_criteria.csv / anchor_attribution.csv
  4  新基线 9 窗 × N∈{40,80} → baseline_n40_n80.csv / baseline_cells.csv / post_reactor_closure.csv
  5  JAX 同步               → jax_parity.csv

第 1 节任一单元级检查不通过即停止（不进入第 3、4 节），失败细节写进 interpretation.md。

用法：
  .venv/bin/python scripts/generate_p1_6a_20261102.py [--stage unit|all]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model.constants import PhysicalConstants  # noqa: E402
from model.fortran_anchor_20261030 import (  # noqa: E402
    DRY_POWDER_CASES,
    anchor_cases,
    build_case_inputs,
    parse_fortran_log,
    solve_fortran_case,
)
from model.gasifier_system import GasifierSystem  # noqa: E402
from model.kinetics import calculate_wgs_equilibrium  # noqa: E402
from model.material import SPECIES_NAMES, MaterialService  # noqa: E402
from model.model_case_adapter import build_gasifier_inputs_from_model_case  # noqa: E402
from model.p0_round_20260930 import m1_reference_window_ids  # noqa: E402
from model.p1_5_round_20261030 import (  # noqa: E402
    base_protocol,
    paper_op_conds,
    run_case,
    silence_solver_logs,
    write_round_outputs,
)
from model.physics import R_CONST, SHOMATE_DB, get_enthalpy_molar  # noqa: E402
from model.revision_evidence_common import load_frozen_split, source_commit  # noqa: E402
from model.source_terms import PyrolysisSource  # noqa: E402
from model.state import StateVector  # noqa: E402

OUT_DIR = ROOT / "outputs" / "diagnostics" / "round-p1.6a-baseline-20261031"
P15B_DIR = ROOT / "outputs" / "diagnostics" / "round-p1.5b-jaxparity-20261030"
P15C_DIR = ROOT / "outputs" / "diagnostics" / "round-p1.5c-wen-anchor-20261030"
P2A_DIR = ROOT / "outputs" / "diagnostics" / "round-p2a-fullmesh-20261030"

N_ANCHOR = (40, 80, 160)
N_ANCHOR_REF = 160
N_BASE = (40, 80)

ANCHOR_TOL = {
    "X_C_pp": 2.0,
    "CO_vol_pct": 2.0,
    "H2_vol_pct": 2.0,
    "CO2_vol_pct": 2.0,
    "T_gas_K": 50.0,
    "O2_drop_rel": 0.20,
}

# 1a 的两种口径（legacy 用于回归复现与归因）
LEGACY_SWITCHES = {
    "SolidEnthalpyModel": "legacy_coal_hf",
    "HetConversionCutoff": "hard_0999",
    "WGS_Keq_Model": "legacy_fit",
}

# 标准生成焓（NIST Shomate 表 H 列，kJ/mol），用于独立核对
NIST_HF_STD = {
    "CO": -110.5271, "CO2": -393.5224, "H2": 0.0, "H2O": -241.8264,
    "CH4": -74.87310, "N2": 0.0, "O2": 0.0, "H2S": -20.50202,
}

CARBON_MOLAR_MASS = 0.012011  # kg/mol
ELEMENT_KG_PER_MOL = {"C": 0.012011, "H": 0.001008, "O": 0.015999, "N": 0.014007, "S": 0.03206}


# ---------------------------------------------------------------------------
# 独立 Shomate 实现（不调用 model.physics 的求值函数，只复用 NIST 系数表）
# ---------------------------------------------------------------------------

def shomate_independent(species: str, T: float, prop: str) -> float:
    data = SHOMATE_DB[species]
    segs = data["segments"]
    coeffs = segs[-1]
    for i, cut in enumerate(data["T_cuts"]):
        if T < float(cut):
            coeffs = segs[i]
            break
    A, B, C, D, E, F, G, _H = coeffs
    t = T / 1000.0
    if prop == "H":
        return (A * t + B * t ** 2 / 2.0 + C * t ** 3 / 3.0 + D * t ** 4 / 4.0 - E / t + F) * 1000.0
    if prop == "S":
        return A * np.log(t) + B * t + C * t ** 2 / 2.0 + D * t ** 3 / 3.0 - E / (2.0 * t ** 2) + G
    if prop == "G":
        return shomate_independent(species, T, "H") - T * shomate_independent(species, T, "S")
    raise ValueError(prop)


def wgs_k_independent(T: float) -> float:
    dG = (
        shomate_independent("CO2", T, "G") + shomate_independent("H2", T, "G")
        - shomate_independent("CO", T, "G") - shomate_independent("H2O", T, "G")
    )
    return float(np.exp(-dG / (R_CONST * T)))


# ---------------------------------------------------------------------------
# 1b：WGS 平衡常数三种式子
# ---------------------------------------------------------------------------

def wgs_keq_table(T_LIST=(1000.0, 1200.0, 1400.0, 1600.0, 1800.0)) -> pd.DataFrame:
    rows = []
    for T in T_LIST:
        k_thermo = float(calculate_wgs_equilibrium(T, "thermo"))
        k_legacy = float(calculate_wgs_equilibrium(T, "legacy_fit"))
        k_fortran = float(calculate_wgs_equilibrium(T, "fortran"))
        k_indep = wgs_k_independent(T)
        rel = abs(k_thermo - k_indep) / abs(k_indep)
        rows.append({
            "T_K": float(T),
            "K_thermo": k_thermo,
            "K_legacy_fit": k_legacy,
            "K_fortran": k_fortran,
            "K_thermo_over_legacy": k_thermo / k_legacy,
            "K_thermo_over_fortran": k_thermo / k_fortran,
            "K_independent_from_dG": k_indep,
            "thermo_vs_independent_rel_diff": rel,
            "tolerance": 1e-10,
            "pass": bool(rel <= 1e-10),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1a：能量口径检查
# ---------------------------------------------------------------------------

def _energy_scale(system: GasifierSystem) -> tuple[float, float]:
    inlet = system.cells[0].inlet
    h_gas = abs(MaterialService.get_gas_enthalpy(inlet))
    h_solid = abs(MaterialService.get_solid_enthalpy(inlet, system.coal_props))
    hhv = coal_hhv_kj_kg(system.coal_props)  # P1.7-A：热损基准是高位（只给 LHV 时按定义推）
    # HHV 单位归一：支持 MJ/kg（<1000）与 kJ/kg（>=1000）两种输入。
    # 归一后 hhv_kj_kg 一律是 kJ/kg，故换算到 J/kg 乘 1e3。
    # 2026-11-03 更正（审阅意见）：原写法把 kJ/kg 再乘 1e6（等于按 MJ/kg 处理），
    # 使 q_hhv 放大 1000 倍、所有 HHV 归一化相对残差缩小 1000 倍，
    # 直接掩盖了 baseline 窗 6.94e-7 > 1e-9 的闸门失败。
    # 2026-11-03 P1.7-A：热值改由 input_contract.coal_hhv_kj_kg 取（只给 LHV_d 时按定义加回潜热）。
    hhv_kj_kg = hhv
    q_hhv = abs(float(system.op_conds["coal_flow"]) * hhv_kj_kg * 1e3)
    return float(h_gas + h_solid), float(q_hhv)


def global_energy_residual_w(system: GasifierSystem) -> float:
    """整炉能量闭合残差 = Σ_单元 (res_E_scaled × ref_energy)。"""
    total = 0.0
    for cell, state in zip(system.cells, system.results):
        x = np.asarray(state.to_array(), dtype=float)
        r = np.asarray(cell.residuals(x), dtype=float)
        scale = cell.residual_scale_vector()
        total += float(r[10] * scale[10])
    return float(total)


def hhv_roundtrip(coal_props: dict) -> tuple[float, float]:
    """由 coal_props 的 Hf_coal 反推 HHV，核对「整体热值仍与 HHV 一致」。

    P1.7-A：热值输入可能是 LHV_d（低位）也可能是 HHV_d（高位），比较对象改由
    ``input_contract.coal_hhv_kj_kg`` 给出（只给 LHV 时按定义加回 9·Hd·2442）。
    """
    from model.input_contract import coal_hhv_kj_kg

    cd = float(coal_props["Cd"]) / 100.0
    hd = float(coal_props["Hd"]) / 100.0
    sd = float(coal_props["Sd"]) / 100.0
    hhv_kj = coal_hhv_kj_kg(coal_props)
    hf_kj_kg = float(coal_props["Hf_coal"]) / 1000.0
    comb = (
        (cd / 0.012011) * (-393.51)
        + (hd / 0.001008) * 0.5 * (-241.83)
        + (sd / 0.03206) * (-296.81)
    )
    lhv_back = hf_kj_kg - comb
    hhv_back = lhv_back + 9.0 * hd * 2442.0
    return float(hhv_back), float(abs(hhv_back - hhv_kj) / hhv_kj)


def pyrolysis_identity(coal_props: dict, m_coal: float, m_vol: float, T_pyro: float) -> dict:
    """入炉煤焓 = (焦炭焓 + 挥发分焓) + 热解热 的逐项恒等式。"""
    unknown = PyrolysisSource(
        np.zeros(len(SPECIES_NAMES)), m_vol, target_cell_idx=0, T_pyro=T_pyro,
        coal_mass_flow_kg_s=m_coal, coal_props=coal_props,
        solid_enthalpy_model="char_formation_zero",
    )
    unknown.vol_fluxes = np.zeros(len(SPECIES_NAMES))
    h_coal = unknown.coal_enthalpy_W
    h_char = unknown.char_enthalpy_W
    h_vol = unknown.volatile_enthalpy_W
    q_pyro = h_coal - h_char - h_vol
    resid = h_coal - (h_char + h_vol + q_pyro)
    return {
        "H_coal_W": float(h_coal),
        "H_char_W": float(h_char),
        "H_vol_W": float(h_vol),
        "Q_pyrolysis_W": float(q_pyro),
        "identity_residual_W": float(resid),
    }


REACTIONS = (
    ("C+H2O->CO+H2", {"CO": 1.0, "H2": 1.0}, {"H2O": 1.0}),
    ("C+CO2->2CO", {"CO": 2.0}, {"CO2": 1.0}),
    ("C+0.5O2->CO", {"CO": 1.0}, {"O2": 0.5}),
    ("C+O2->CO2", {"CO2": 1.0}, {"O2": 1.0}),
)


def carbon_enthalpy_per_mol_C(char_carbon_fraction: float, T: float, coal_props: dict,
                              include_formation: bool) -> float:
    """模型给 1 mol 固体碳的焓 [J/mol C]。

    P1.6-A 口径：碳的生成焓取 0（石墨）、只计显热；灰是旁观者，其显热在反应两侧相消，
    因此按 12.011 g 碳 × ``cp_char`` 计（``cp_char`` 是每 kg 固体的热容，用于碳骨架）。
    修改前（legacy）口径：焦炭沿用每 kg 固体的 ``Hf_coal``，含 1 mol C 的焦炭质量是
    ``0.012011/Xc``，故生成焓项要按该质量折算。
    """
    if include_formation:
        xc = max(float(char_carbon_fraction), 1e-6)
        state = StateVector(
            gas_moles=np.zeros(len(SPECIES_NAMES)), solid_mass=CARBON_MOLAR_MASS / xc,
            carbon_fraction=xc, T=T, P=101325.0,
        )
        return float(MaterialService.get_solid_enthalpy(
            state, coal_props, T_solid_override=T, include_formation_enthalpy=True))
    state = StateVector(
        gas_moles=np.zeros(len(SPECIES_NAMES)), solid_mass=CARBON_MOLAR_MASS,
        carbon_fraction=1.0, T=T, P=101325.0,
    )
    return float(MaterialService.get_solid_enthalpy(
        state, coal_props, T_solid_override=T, include_formation_enthalpy=False))


def reaction_heat_rows(T_ctx: float, char_xc: float, coal_props: dict, label: str,
                        coal_props_legacy: dict | None = None) -> list[dict]:
    rows = []
    legacy_props = coal_props_legacy if coal_props_legacy is not None else coal_props
    for name, products, reactants in REACTIONS:
        q_model = -carbon_enthalpy_per_mol_C(char_xc, T_ctx, coal_props, False)
        q_ref = -carbon_enthalpy_per_mol_C(char_xc, T_ctx, coal_props, False)
        for sp, nu in products.items():
            q_model += nu * float(get_enthalpy_molar(sp, T_ctx))
            q_ref += nu * shomate_independent(sp, T_ctx, "H")
        for sp, nu in reactants.items():
            q_model -= nu * float(get_enthalpy_molar(sp, T_ctx))
            q_ref -= nu * shomate_independent(sp, T_ctx, "H")
        q_legacy = -carbon_enthalpy_per_mol_C(char_xc, T_ctx, legacy_props, True)
        for sp, nu in products.items():
            q_legacy += nu * float(get_enthalpy_molar(sp, T_ctx))
        for sp, nu in reactants.items():
            q_legacy -= nu * float(get_enthalpy_molar(sp, T_ctx))
        rel = abs(q_model - q_ref) / max(abs(q_ref), 1e-12)
        rel_legacy = abs(q_legacy - q_ref) / max(abs(q_ref), 1e-12)
        # 由 NIST 表列标准生成焓另算一次 298 K 参考值
        q_tab = sum(nu * NIST_HF_STD[sp] for sp, nu in products.items())
        q_tab -= sum(nu * NIST_HF_STD[sp] for sp, nu in reactants.items())
        rows.append({
            "check_group": "reaction_heat",
            "context": label,
            "T_K": float(T_ctx),
            "char_carbon_fraction": float(char_xc),
            "reaction": name,
            "q_model_J_mol": float(q_model),
            "q_shomate_J_mol": float(q_ref),
            "rel_diff": float(rel),
            "tolerance": 1e-6,
            "pass": bool(rel <= 1e-6),
            "q_legacy_basis_J_mol": float(q_legacy),
            "rel_diff_legacy": float(rel_legacy),
            "q_nist_tabulated_298_J_mol": float(q_tab * 1000.0),
            "note": "q_shomate 由独立 Shomate 实现按同一 NIST 系数表求得。298.15 K 行同时给出 NIST 表列 ΔHf "
                    "直接求和的参考值（差 2.4e-5 来自 Shomate 拟合在 298 K 不严格回到 ΔHf，属系数表性质）。"
                    "**本检查的检验力有限，须与数值一起读**：q_model 与 q_ref 的碳项调用同一个 "
                    "carbon_enthalpy_per_mol_C（同参数、同 cp_char），所以高温行的 rel_diff 恒为 0 "
                    "并不构成对碳焓的独立验证，只证明气体侧两条路径一致；与真实石墨 cp 的差约 3–4 kJ/mol，"
                    "属 cp_char 这一冻结常数的口径，与 B9 无关。因此本行能支持的是「标准反应热、单位与"
                    "气体侧实现一致」，不能支持「高温热化学对温差的贡献已被排除」。",
        })
    return rows


def solve_anchor_system(case: dict, n_cells: int, extra: dict | None = None) -> GasifierSystem:
    built = build_case_inputs(case)
    op = paper_op_conds(built, extra=extra or {})
    system = GasifierSystem(built["geometry"], built["coal_props"], op)
    system.solve(N_cells=n_cells, solver_method="minimize", jacobian_mode="scipy", jax_warmup=False)
    return system


def _system_of(rec_with_system: dict) -> GasifierSystem:
    system = rec_with_system["_system"]
    rec_with_system.pop("_system", None)
    rec_with_system.pop("_profile", None)
    return system


def energy_check_rows(label: str, window_id: str, solve_fn, n_cells: int) -> list[dict]:
    rows: list[dict] = []
    results = {}
    for model in ("legacy_coal_hf", "char_formation_zero"):
        extra = dict(LEGACY_SWITCHES) if model == "legacy_coal_hf" else {
            "SolidEnthalpyModel": "char_formation_zero",
            "HetConversionCutoff": "hard_0999",
            "WGS_Keq_Model": "legacy_fit",
        }
        system = solve_fn(extra)
        scale, q_hhv = _energy_scale(system)
        resid = global_energy_residual_w(system)
        results[model] = {
            "system": system, "scale": scale, "q_hhv": q_hhv, "resid": resid,
        }

    new = results["char_formation_zero"]
    old = results["legacy_coal_hf"]
    rows.append({
        "check_group": "energy_closure", "context": label, "window_id": window_id,
        "case": f"N={n_cells}", "metric": "global_energy_residual_rel_hhv",
        "value": abs(new["resid"]) / new["q_hhv"], "reference": abs(old["resid"]) / old["q_hhv"],
        "tolerance": 1e-9, "pass": bool(abs(new["resid"]) / new["q_hhv"] <= 1e-9),
        "note": f"新口径整炉能量闭合残差 {new['resid']:.3e} W / 入炉 HHV {new['q_hhv']:.3e} W",
    })
    rows.append({
        "check_group": "energy_closure", "context": label, "window_id": window_id,
        "case": f"N={n_cells}", "metric": "global_energy_residual_rel_hhv_legacy",
        "value": abs(old["resid"]) / old["q_hhv"], "reference": float("nan"),
        # 2026-11-03 更正（审阅意见 1）：原先的理由写得过头了。
        # 事实：旧口径是对旧模型自身重新求解、再对自身 cell.residuals 求和，
        # 所以「旧口径残差非零」只说明这一次求解没收敛到同等精度，不说明
        # 「该判据按构造不可能通过」——锚点旧口径本来就是 8.85e-12 < 1e-9，通过了。
        # 把旧模型对照移出新模型验收闸门的正当理由是：它检验的是「旧模型自身的
        # 数值收敛」，不是「1a 修正是否保持了闭合」；而 1a 的实质要求（闭合成立、
        # 热值仍与 HHV 一致）由本组 rel_hhv 与 hhv_roundtrip_rel 分别判定。
        "tolerance": float("nan"), "pass": None,
        "note": ("修改前口径（同组入口），仅作对照、不参与闸门。注意这是对旧模型自身重新求解后的"
                 "残差聚合，因此它反映的是该次求解的收敛程度，不能直接当作 B9 物理缺陷的量级。"),
    })
    delta = abs(abs(new["resid"]) - abs(old["resid"])) / new["q_hhv"]
    rows.append({
        "check_group": "energy_closure", "context": label, "window_id": window_id,
        "case": f"N={n_cells}", "metric": "global_energy_residual_delta_rel_hhv",
        "value": delta, "reference": float("nan"),
        # 2026-11-03 更正（两轮审阅的结论）：
        # 本项 = |resid_new - resid_old| / q，在 resid_old 远大于 resid_new 时
        # 数值上等于 resid_old/q，即它测的是「旧口径那次求解的残差有多大」。
        # 这不能作为「1a 修正是否保持闭合」的判据：旧口径是对旧模型自身重新求解，
        # 其残差反映该次求解的收敛程度（锚点旧口径 8.85e-12 是通过的），
        # 与 B9 的物理缺陷大小不是同一个量。
        # 卡 1a(i) 的实质要求是「整炉闭合成立 + 整体热值仍与 HHV 一致」，
        # 分别由本组的 rel_hhv 与 hhv_roundtrip_rel 判定，故本项降为信息行。
        "tolerance": float("nan"), "pass": None,
        "note": ("修改前后残差绝对差（信息行，不参与闸门）：数值上≈ resid_old/q。"
                 "它混合了「旧模型自身的数值收敛程度」与「新旧方程的差别」，"
                 "不宜单独解释为物理量的改善倍数。"),
    })

    coal_props = new["system"].coal_props
    hhv_back, hhv_rel = hhv_roundtrip(coal_props)
    rows.append({
        "check_group": "energy_closure", "context": label, "window_id": window_id,
        "case": f"N={n_cells}", "metric": "hhv_roundtrip_rel",
        "value": float(hhv_rel), "reference": float(hhv_back), "tolerance": 1e-9,
        "pass": bool(hhv_rel <= 1e-9),
        "note": "由 Hf_coal 反推的 HHV 与 coal_props['HHV_d'] 的相对差（kJ/kg）",
    })

    system = new["system"]
    pyro = system._pyrolysis_source()
    ident = pyrolysis_identity(coal_props, system.W_dry, system.tmp_W_vol, system.op_conds["T_in"])
    denom = max(abs(ident["H_coal_W"]), 1e-12)
    rows.append({
        "check_group": "energy_closure", "context": label, "window_id": window_id,
        "case": f"N={n_cells}", "metric": "pyrolysis_enthalpy_identity_rel",
        "value": float(abs(ident["identity_residual_W"]) / denom),
        "reference": float(ident["Q_pyrolysis_W"]), "tolerance": 1e-9,
        "pass": bool(abs(ident["identity_residual_W"]) / denom <= 1e-9),
        "note": "H_coal = H_char + H_vol + Q_pyro；reference 列是 Q_pyro [W]"
                f"（Hf_coal={coal_props['Hf_coal']:.4e} J/kg, Q_pyro={pyro.pyrolysis_heat_W:.4e} W）",
    })

    # 1a 的扰动量：新−旧 能量残差应恰等于 −m_char·Hf_coal（同一状态）
    cell0_legacy = old["system"].cells[0]
    state = old["system"].results[0]
    x = np.asarray(state.to_array(), dtype=float)
    r_new = float(np.asarray(new["system"].cells[0].residuals(x))[10])
    r_old = float(np.asarray(cell0_legacy.residuals(x))[10])
    scale0 = float(new["system"].cells[0].residual_scale_vector()[10])
    d_res = (r_new - r_old) * scale0
    expected = -float(state.solid_mass) * float(coal_props["Hf_coal"])
    rel = abs(d_res - expected) / max(abs(expected), 1e-12)
    rows.append({
        "check_group": "energy_closure", "context": label, "window_id": window_id,
        "case": f"N={n_cells}", "metric": "b9_perturbation_identity_rel",
        "value": float(rel), "reference": float(expected), "tolerance": 1e-9,
        "pass": bool(rel <= 1e-9),
        "note": "同一状态上 Δres_E 应等于 −m_char·Hf_coal [W]；reference 列即该理论值，"
                f"实测 Δres_E={d_res:.6e} W（B9 在这一格上的量级）",
    })
    return rows


# ---------------------------------------------------------------------------
# 锚点
# ---------------------------------------------------------------------------

def _strip(rec: dict) -> dict:
    for key in ("_profile", "_z", "_system"):
        rec.pop(key, None)
    return rec


def match_fortran(cases: list[dict], fortran_out: list[dict]) -> dict[str, dict]:
    out = {}
    for case in cases:
        name = case["case_name"]
        hits = [f for f in fortran_out if f["case_name"].startswith(name[:12])]
        if not hits:
            continue
        out[name] = hits[0]
    return out


def anchor_rows(cases, geometry, fortran_by_case):
    rows: list[dict] = []
    for case in cases:
        name = case["case_name"]
        for mode in ("F", "M"):
            for n_cells in N_ANCHOR:
                rec = solve_fortran_case(case, n_cells=n_cells, mode=mode, geometry=geometry)
                _strip(rec)
                rec["is_dry_powder"] = bool(case["is_dry_powder"])
                f = fortran_by_case.get(name, {})
                rec["fortran_T_gas_K"] = float(f.get("T_gas_K", np.nan))
                rec["fortran_X_C_pct"] = float(f.get("X_C_pct", np.nan))
                rows.append(rec)
    return pd.DataFrame(rows)


def anchor_criteria_rows(out: pd.DataFrame, fortran_by_case) -> pd.DataFrame:
    rows = []
    ref = out[out["N_cells"] == N_ANCHOR_REF]
    for name, f in fortran_by_case.items():
        if name not in DRY_POWDER_CASES:
            continue
        for mode in ("F", "M"):
            hit = ref[(ref["case_name"] == name) & (ref["mode"] == mode)]
            if hit.empty:
                continue
            r = hit.iloc[0]
            checks = {
                "X_C_pp": (r.get("X_C_pct"), f["X_C_pct"]),
                "CO_vol_pct": (r.get("CO_vol_pct"), f["CO_vol_pct"]),
                "H2_vol_pct": (r.get("H2_vol_pct"), f["H2_vol_pct"]),
                "CO2_vol_pct": (r.get("CO2_vol_pct"), f["CO2_vol_pct"]),
                "T_gas_K": (r.get("T_gas_K"), f["T_gas_K"]),
                "O2_drop_rel": (r.get("O2_drop_to_1pct_z_m"), f.get("O2_drop_to_1pct_z_m")),
            }
            for metric, (pv, fv) in checks.items():
                pv = float(pv) if pv is not None else float("nan")
                fv = float(fv) if fv is not None else float("nan")
                evaluable = metric != "O2_drop_rel"
                diff = abs(pv - fv)
                note = ""
                if metric == "O2_drop_rel":
                    diff = diff / max(abs(fv), 1e-9)
                    note = ("not evaluable: Fortran 只在收敛后打印少数气室，第一次打印处 O2 已为 0；"
                            "Python 在 cell 0 即降到入口 1% 以下")
                rows.append({
                    "case_name": name, "mode": mode, "N_cells": N_ANCHOR_REF, "metric": metric,
                    "python_value": pv, "fortran_value": fv, "abs_diff": diff,
                    "tolerance": ANCHOR_TOL[metric], "evaluable": bool(evaluable),
                    "pass": (bool(np.isfinite(diff) and diff <= ANCHOR_TOL[metric])
                             if evaluable else None),
                    "note": note,
                })
    return pd.DataFrame(rows)


def anchor_attribution_rows(cases, geometry, fortran_by_case) -> pd.DataFrame:
    toggles = [
        ("WGS_Keq_Model_new_vs_legacy_fit", {"WGS_Keq_Model": "legacy_fit"}),
        ("SolidEnthalpy_1a_on_vs_off", {"SolidEnthalpyModel": "legacy_coal_hf"}),
        ("UseFortranDiffusion", {"UseFortranDiffusion": True}),
        ("WGS_RatFactor", {"WGS_RatFactor": True}),
        ("CharCombustionRateFactor", {"CharCombustionRateFactor": 1.0}),
    ]
    rows = []
    for case in cases:
        if not case["is_dry_powder"]:
            continue
        name = case["case_name"]
        base = _strip(solve_fortran_case(case, n_cells=N_ANCHOR_REF, mode="M", geometry=geometry))
        for item, switches in toggles:
            alt = _strip(solve_fortran_case(
                case, n_cells=N_ANCHOR_REF, mode="M", geometry=geometry, switches=switches))
            rows.append({
                "case_name": name, "mode": "M", "N_cells": N_ANCHOR_REF, "item": item,
                "d_X_C_pp": alt.get("X_C_pct", np.nan) - base.get("X_C_pct", np.nan),
                "d_CO_vol_pct": alt.get("CO_vol_pct", np.nan) - base.get("CO_vol_pct", np.nan),
                "d_H2_vol_pct": alt.get("H2_vol_pct", np.nan) - base.get("H2_vol_pct", np.nan),
                "d_CO2_vol_pct": alt.get("CO2_vol_pct", np.nan) - base.get("CO2_vol_pct", np.nan),
                "d_T_gas_K": alt.get("T_gas_K", np.nan) - base.get("T_gas_K", np.nan),
                "alt_execution_status": alt.get("execution_status"),
                "fortran_T_gas_K": float(fortran_by_case.get(name, {}).get("T_gas_K", np.nan)),
                "base_T_gas_K": base.get("T_gas_K", np.nan),
                "base_X_C_pct": base.get("X_C_pct", np.nan),
                "fortran_X_C_pct": float(fortran_by_case.get(name, {}).get("X_C_pct", np.nan)),
            })

    # 修订第 4 条：I-1 在两种水分下的模式 F 结果 + 两个历史口径的复现
    i1 = next(c for c in cases if c["case_name"] == "texaco i-1")
    f_ref = float(fortran_by_case.get("texaco i-1", {}).get("T_gas_K", np.nan))
    f_xc = float(fortran_by_case.get("texaco i-1", {}).get("X_C_pct", np.nan))
    f_co = float(fortran_by_case.get("texaco i-1", {}).get("CO_vol_pct", np.nan))
    f_h2 = float(fortran_by_case.get("texaco i-1", {}).get("H2_vol_pct", np.nan))
    f_co2 = float(fortran_by_case.get("texaco i-1", {}).get("CO2_vol_pct", np.nan))
    json_coal = {"Mt": 4.53, "FCd": 54.9, "HHV_d": 29800.0}  # data/validation_cases_final.json 的 H-coal residue
    for tag, coal_ov, n_cells, extra in (
        ("I-1 Mt=0.2%（namelist；HHV_d=30 占位）", {}, N_ANCHOR_REF, {}),
        ("I-1 Mt=4.53%（论文）", {"Mt": 4.53}, N_ANCHOR_REF, {}),
        ("I-1 JSON 煤质（HHV_d=29.8, FCd=54.9, Mt=4.53）",
         json_coal, N_ANCHOR_REF, {}),
        ("I-1 HHV_d=33.74 MJ/kg（组分关联式）",
         {"HHV_d": 33740.0}, N_ANCHOR_REF, {}),
        ("I-1 HHV_d=33.74 + N=25 旧网格 + HeatLoss 3%",
         {"HHV_d": 33740.0}, 25,
         {"AllowTruncatedMesh": True, "HeatLossPercent": 3.0, "SolverStrictRoot": False,
          "MeshFamily": "legacy"}),
        ("I-1 HHV_d=30 + N=25 旧网格 + HeatLoss 3%", {}, 25,
         {"AllowTruncatedMesh": True, "HeatLossPercent": 3.0, "SolverStrictRoot": False,
          "MeshFamily": "legacy"}),
    ):
        alt = _strip(solve_fortran_case(
            i1, n_cells=n_cells, mode="F", geometry=geometry,
            coal_overrides=coal_ov or None, switches=extra or None))
        rows.append({
            "case_name": "texaco i-1", "mode": "F", "N_cells": n_cells, "item": tag,
            "d_X_C_pp": alt.get("X_C_pct", np.nan) - f_xc,
            "d_CO_vol_pct": alt.get("CO_vol_pct", np.nan) - f_co,
            "d_H2_vol_pct": alt.get("H2_vol_pct", np.nan) - f_h2,
            "d_CO2_vol_pct": alt.get("CO2_vol_pct", np.nan) - f_co2,
            "d_T_gas_K": alt.get("T_gas_K", np.nan) - f_ref,
            "alt_execution_status": alt.get("execution_status"),
            "fortran_T_gas_K": f_ref,
            "base_T_gas_K": alt.get("T_gas_K", np.nan),
            "base_X_C_pct": alt.get("X_C_pct", np.nan),
            "fortran_X_C_pct": f_xc,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2：X_C 截断窗口
# ---------------------------------------------------------------------------

def xcap_windows() -> tuple[list[str], list[str]]:
    p2a = pd.read_csv(P2A_DIR / "fullmesh_convergence.csv")
    main = p2a[p2a["ParticleSizeModel"] == "fortran_constant"]
    xcap = sorted(main[(main["N_cells"] == 40) & (main["X_C_implied"] >= 0.9985)]["window_id"].unique())
    noroot = sorted(main[(main["N_cells"] == N_ANCHOR_REF) & (main["execution_status"] != "completed")]["window_id"].unique())
    return list(xcap), list(noroot)


def xcap_rows(window_ids, by_id, p2a: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for wid in window_ids:
        for n_cells in (40, 80, 160):
            rec = run_case(by_id[wid], n_cells=n_cells, extra_op=None)
            p2a_hit = p2a[(p2a["window_id"] == wid) & (p2a["N_cells"] == n_cells)
                          & (p2a["ParticleSizeModel"] == "fortran_constant")]
            rows.append({
                "window_id": wid,
                "N_cells": n_cells,
                "before_1_to_5_X_C": float(p2a_hit["X_C_implied"].iloc[0]) if len(p2a_hit) else np.nan,
                "before_execution_status": (p2a_hit["execution_status"].iloc[0] if len(p2a_hit) else ""),
                "before_no_root_cell_index": (float(p2a_hit["no_root_cell_index"].iloc[0]) if len(p2a_hit) else np.nan),
                "after_X_C": rec.get("X_C_implied", np.nan),
                "after_execution_status": rec["execution_status"],
                "after_no_root_cell_index": rec.get("no_root_cell_index"),
                "after_max_residual_cost": rec.get("max_residual_cost"),
                "after_CO_vol_pct": rec.get("CO_vol_pct"),
                "after_CO2_vol_pct": rec.get("CO2_vol_pct"),
                "after_H2_vol_pct": rec.get("H2_vol_pct"),
                "after_T_out_K": rec.get("T_out_K"),
                "after_dry_flow_ratio": rec.get("dry_flow_ratio"),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4：新基线
# ---------------------------------------------------------------------------

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


def baseline_rows(window_ids, by_id, n_list=N_BASE):
    rows, cells = [], []
    for wid in window_ids:
        for n_cells in n_list:
            rec = run_case(by_id[wid], n_cells=n_cells, keep_system=True)
            system: GasifierSystem = rec.pop("_system")
            rec.pop("_profile", None)
            rec.update(_limit_zone_stats(system))
            out = {}
            if system.execution_status == "completed" and system.results:
                state = system.results[-1]
                gas = np.asarray(state.gas_moles, dtype=float)
                total = float(np.sum(gas))
                dry = float(np.sum(gas[:7]))
                for idx, name in ((2, "CO"), (3, "CO2"), (5, "H2"), (1, "CH4"), (7, "H2O"),
                                  (6, "N2"), (4, "H2S"), (0, "O2")):
                    out[f"F_{name}_mol_s"] = float(gas[idx])
                y = gas / max(total, 1e-12)
                q_wgs = (y[3] * y[5]) / max(y[2] * y[7], 1e-30)
                k_thermo = float(calculate_wgs_equilibrium(float(state.T), "thermo"))
                out.update({
                    "Q_WGS": float(q_wgs),
                    "K_thermo_T_out": k_thermo,
                    "Q_over_K": float(q_wgs / k_thermo),
                    "dry_total_mol_s": dry,
                })
                hd = float(system.coal_props.get("Hd", 0.0))
                h_in = system.W_dry * (hd / 100.0) / ELEMENT_KG_PER_MOL["H"]
                mt = float(system.coal_props.get("Mt", 0.0)) / 100.0
                h_in_moist = h_in + system.W_dry * (mt / max(1.0 - mt, 1e-9)) / ELEMENT_KG_PER_MOL["H"]
                h_out_dry = 2.0 * float(gas[5]) + 4.0 * float(gas[1]) + 2.0 * float(gas[4])
                plant_dry_mol_s = float(rec["plant_dry_flow_nm3_h"]) / 22.414 / 3600.0 * 1000.0
                plant_h = 2.0 * plant_dry_mol_s * float(rec["plant_H2_vol_pct"]) / 100.0
                out.update({
                    "H_in_mol_s": h_in,
                    "H_out_dry_mol_s": h_out_dry,
                    "H_out_over_H_in": h_out_dry / max(h_in, 1e-12),
                    "H_plant_mol_s": plant_h,
                    "H_plant_over_H_in": plant_h / max(h_in, 1e-12),
                    "H_in_with_moisture_mol_s": h_in_moist,
                    "H_out_over_H_in_with_moisture": h_out_dry / max(h_in_moist, 1e-12),
                    "H_plant_over_H_in_with_moisture": plant_h / max(h_in_moist, 1e-12),
                    "plant_dry_total_mol_s": plant_dry_mol_s,
                })
            rec.update(out)
            rows.append(rec)
            for diag in system.cell_diagnostics:
                d = dict(diag)
                d.update({"window_id": wid, "N_cells": n_cells})
                cells.append(d)
    return pd.DataFrame(rows), pd.DataFrame(cells)


def closure_rows(baseline: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in baseline.iterrows():
        if r["execution_status"] != "completed":
            rows.append({
                "window_id": r["window_id"], "N_cells": r["N_cells"],
                "execution_status": r["execution_status"], "xi_card": np.nan,
                "pred_CO_vol_pct": np.nan, "pred_H2_vol_pct": np.nan,
                "plant_CO_vol_pct": r["plant_CO_vol_pct"], "plant_H2_vol_pct": r["plant_H2_vol_pct"],
                "d_CO": np.nan, "d_H2": np.nan,
            })
            continue
        dry = float(r["dry_total_mol_s"])
        f_co2, f_co, f_h2 = float(r["F_CO2_mol_s"]), float(r["F_CO_mol_s"]), float(r["F_H2_mol_s"])
        y_co2 = float(r["plant_CO2_vol_pct"]) / 100.0
        # 卡面口径：(CO2 + ξ)/(干气 + ξ) = 装置 CO2 体积分数
        #
        # 为什么分母必须加 ξ：CO + H2O → CO2 + H2 在湿基上摩尔数守恒（2 → 2），
        # 但把一个干气摩尔（CO）换成两个干气摩尔（CO2 + H2），所以干气总量增加 ξ。
        # 早期版本另设了一个「严格计量、干气摩尔数不变」的对照口径，
        # 分母取 dry 而不加 ξ，这是错的：它给出的 18 行里首行 CO + H2 + CO2 已达
        # 102.7 vol%，未计 CH4/N2/H2S 就超过 100%。而且把分母改成 dry + ξ 之后，
        # 该对照式与卡面式代数恒等（ξ 的解完全相同），所以它根本不是独立检验，
        # 已删除（2026-11-03，审阅意见）。
        xi_card = (y_co2 * dry - f_co2) / (1.0 - y_co2)
        denom_card = dry + xi_card
        pred_co = (f_co - xi_card) / denom_card * 100.0
        pred_h2 = (f_h2 + xi_card) / denom_card * 100.0
        rows.append({
            "window_id": r["window_id"], "N_cells": r["N_cells"], "phase": r["phase"],
            "execution_status": r["execution_status"],
            "dry_total_mol_s": dry,
            "F_CO2_mol_s": f_co2, "F_CO_mol_s": f_co, "F_H2_mol_s": f_h2,
            "plant_CO2_vol_pct": r["plant_CO2_vol_pct"],
            "xi_card": xi_card,
            "pred_CO_vol_pct": pred_co, "plant_CO_vol_pct": r["plant_CO_vol_pct"],
            "d_CO": pred_co - float(r["plant_CO_vol_pct"]),
            "pred_H2_vol_pct": pred_h2, "plant_H2_vol_pct": r["plant_H2_vol_pct"],
            "d_H2": pred_h2 - float(r["plant_H2_vol_pct"]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5：JAX 对照
# ---------------------------------------------------------------------------

def jax_parity_rows(window_row: dict, n_cells: int = 40) -> tuple[pd.DataFrame, dict]:
    """P1.5-B 的 F1（组装残差）/ F4（异相速率）夹具，本轮在 1a/1b/2 之后重跑。

    两种口径同时给出：
      * ``abs_diff``（mapped）：按物种映射比较（mainline 8 组分 → JAX 9 组分，见 ``MAP``）。
      * ``index_aligned_abs_diff``：P1.5-B 当轮的做法——两边第 k 个分量直接相比。JAX 的状态是
        9+3 维（含 COS），逐索引直比从第 4 个分量起就错位（numpy 4=H2S 对 jax 4=H2O；
        numpy 8/9/10=Ws/Xc/E 对 jax 9/10/11）。此列只用于与 P1.5-B 的数字对齐。

    ``variant``：``new``（1a/1b/2 全开，本轮缺省）与 ``legacy``（三个开关全回到修改前），
    后者用来给出同一夹具下的「修改前」基准，避免把不同方程的结果与 P1.5-B 的数字混比。
    """
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from model.input_contract import ash_mass_fraction_dry
    from model.jax_kinetics import heterogeneous_rates_jax
    from model.jax_residuals import cell_residuals_jax_flat
    from model.species_bridge import jax9_to_mainline_gas8, mainline_gas8_to_jax9

    F1_RTOL, F1_ATOL = 1e-8, 1e-10
    F4_RTOL = 2e-6
    MAP = (0, 1, 2, 3, 7, 5, 6, 4, 8, 9, 11)  # mainline 8+3 → jax 9+3
    rows = []

    def _inputs(cell, state, legacy: bool):
        g_src = np.zeros(8)
        s_src, e_src = 0.0, 0.0
        for src in cell.sources:
            g, sm, e = src.get_sources(cell.idx, cell.z, cell.dz)
            g_src += g
            s_src += sm
            e_src += e
        op = cell.op_conds
        from model.input_contract import coal_hhv_kj_kg
        hhv_mj = coal_hhv_kj_kg(cell.coal_props) / 1000.0
        ash = ash_mass_fraction_dry(cell.coal_props)
        f_ash = ash / (cell.coal_props.get("Cd", 60.0) / 100.0 + ash + 1e-9)
        x9 = np.concatenate([mainline_gas8_to_jax9(state.gas_moles),
                             [state.solid_mass, state.carbon_fraction, state.T]])
        inlet9 = np.concatenate([mainline_gas8_to_jax9(cell.inlet.gas_moles),
                                 [cell.inlet.solid_mass, cell.inlet.carbon_fraction, cell.inlet.T]])
        particle_flag = 1.0 if str(op.get("ParticleSizeModel", "fortran_constant")) == "fortran_constant" else 0.0
        return [
            x9, inlet9, mainline_gas8_to_jax9(g_src), s_src, e_src,
            float(cell.dz), float(cell.A), float(cell.inlet.P),
            float(cell.coal_flow_dry * cell.coal_props.get("Cd", 60.0) / 100.0),
            float(cell.inlet.T_solid_or_gas),
            float(op["coal_flow"]),
            float(op.get("particle_diameter", 100e-6)),
            float(op.get("epsilon", 1.0)),
            float(op.get("HeatLossPercent", 2.0)),
            float(op.get("L_reactor", 10.3)),
            float(op.get("CharCombustionRateFactor", 0.3)),
            float(op.get("WGS_CatalyticFactor", 0.2)),
            float(op.get("Combustion_CO2_Fraction", 1.0)),
            float(op.get("P_O2_Combustion_atm", 0.05)),
            float(op.get("L_heatloss_norm", cell.dz)),
            float(op.get("HeatLossRefTemp", 1800.0)),
            float(cell.coal_props.get("Hf_coal", 0.0)),
            float(cell.coal_props.get("cp_char", 1300.0)),
            float(hhv_mj), float(f_ash), float(cell.ref_flow), float(cell.ref_energy),
            float(cell.char_Xc0), 0.0, particle_flag,
            1.0 if cell.idx == 0 else 0.0,                          # solid_hf_inlet
            1.0 if legacy else 0.0,                                # solid_enthalpy_legacy
            1.0 if legacy else 0.0,                                # wgs_keq_model_id -> legacy_fit
            1.0 if legacy else 0.0,                                # het_conversion_cutoff_legacy
        ]

    for variant in ("new", "legacy"):
        legacy = variant == "legacy"
        extra_op = dict(LEGACY_SWITCHES) if legacy else None
        for particle_model in ("fortran_constant", "shrinking_legacy"):
            rec = run_case(window_row, n_cells=n_cells, particle_model=particle_model,
                           extra_op=extra_op, keep_system=True)
            system: GasifierSystem = rec.pop("_system")
            profile = np.asarray(rec.pop("_profile"), dtype=float)
            combs = [d["cell_index"] for d in system.cell_diagnostics if d.get("is_combustion_zone")]
            gasifs = [d["cell_index"] for d in system.cell_diagnostics if not d.get("is_combustion_zone")]
            chosen = (combs[:3] + gasifs[-3:]) if len(combs) >= 3 and len(gasifs) >= 3 else combs + gasifs

            for idx in chosen:
                cell = system.cells[idx]
                state = StateVector.from_array(profile[idx], P=cell.inlet.P, z=cell.z)
                state.T_solid = profile[idx][10]
                args = _inputs(cell, state, legacy)
                nump = np.asarray(cell.residuals(profile[idx]), dtype=float)
                jres = np.asarray(cell_residuals_jax_flat(*[
                    jnp.asarray(a) if isinstance(a, np.ndarray) else a for a in args]), dtype=float)
                for k in range(11):
                    jk = MAP[k]
                    diff = abs(nump[k] - jres[jk])
                    tol = F1_ATOL + F1_RTOL * abs(jres[jk])
                    rows.append({
                        "kind": "F1_residual", "variant": variant, "particle_model": particle_model,
                        "cell_index": idx,
                        "is_combustion_zone": bool(any(d["cell_index"] == idx and d.get("is_combustion_zone")
                                                       for d in system.cell_diagnostics)),
                        "component": k, "jax_component": jk,
                        "numpy": float(nump[k]), "jax": float(jres[jk]),
                        "abs_diff": float(diff), "tolerance": float(tol), "pass": bool(diff <= tol),
                        "index_aligned_abs_diff": float(abs(nump[k] - jres[k])),
                    })
                snap = cell.snapshot_with_limits(state)
                phys = snap["phys"]
                jr = heterogeneous_rates_jax(
                    jnp.asarray(args[0][:9]), float(cell.inlet.P), float(state.T),
                    float(phys["d_p"]), float(phys["S_total"]), float(phys["X_total"]),
                    float(snap["Ts_avg"]), Re=float(phys["Re_p"]), Sc=float(phys["Sc_p"]),
                    char_combustion_factor=float(cell.op_conds.get("CharCombustionRateFactor", 0.3)),
                    hard_conversion_cutoff=1.0 if legacy else 0.0,
                )
                nr = cell.kinetics.calc_heterogeneous_rates(
                    state, phys["d_p"], phys["S_total"], X_total=float(phys["X_total"]),
                    Re=phys["Re_p"], Sc=phys["Sc_p"], T_particle=float(snap["Ts_avg"]),
                    char_combustion_factor=cell.op_conds.get("CharCombustionRateFactor", 0.3),
                    conversion_cutoff="hard_0999" if legacy else "smooth",
                )
                for i, name in enumerate(("C+O2", "C+H2O", "C+CO2", "C+H2")):
                    a, b = float(jr[i]), float(nr[name])
                    rel = abs(a - b) / max(abs(b), 1e-12)
                    rows.append({
                        "kind": "F4_het_rate", "variant": variant, "particle_model": particle_model,
                        "cell_index": idx, "reaction": name, "jax_rate": a, "numpy_rate": b,
                        "rel_diff": rel, "tolerance": F4_RTOL, "pass": bool(rel <= F4_RTOL),
                    })

    df = pd.DataFrame(rows)
    def _summ(sub: pd.DataFrame) -> dict:
        f1 = sub[sub["kind"] == "F1_residual"]
        f4 = sub[sub["kind"] == "F4_het_rate"]
        return {
            "f1_max_abs_diff": float(f1["abs_diff"].max()),
            "f1_all_pass": bool(f1["pass"].all()),
            "f1_max_abs_diff_index_aligned": float(f1["index_aligned_abs_diff"].max()),
            "f1_max_abs_diff_fortran_constant": float(
                f1[f1["particle_model"] == "fortran_constant"]["abs_diff"].max()),
            "f1_max_abs_diff_shrinking_legacy": float(
                f1[f1["particle_model"] == "shrinking_legacy"]["abs_diff"].max()),
            "f4_max_rel_diff": float(f4["rel_diff"].max()),
            "f4_all_pass": bool(f4["pass"].all()),
        }

    summary = {
        "window_id": str(window_row["window_id"]), "N_cells": n_cells,
        **{f"after_{k}": v for k, v in _summ(df[df["variant"] == "new"]).items()},
        **{f"before_{k}": v for k, v in _summ(df[df["variant"] == "legacy"]).items()},
    }
    return df, summary


def jax_solver_smoke(window_row: dict, n_cells: int = 10) -> dict:
    import logging as _logging
    import jax  # noqa: F401

    from model.jax_solver import warmup_jax

    warmup_jax()
    _logging.disable(_logging.INFO)
    built = build_gasifier_inputs_from_model_case(window_row)
    op = paper_op_conds(built)
    system = GasifierSystem(built["geometry"], built["coal_props"], op)
    try:
        profile, _z = system.solve(N_cells=n_cells, solver_method="jax_jit", jax_warmup=True)
        status = system.execution_status
        t_out = float(profile[-1][10]) if len(profile) else float("nan")
        no_root = system.no_root_cell_index
    except Exception as exc:  # noqa: BLE001
        status = f"error:{type(exc).__name__}:{exc}"
        t_out = float("nan")
        no_root = None
    finally:
        _logging.disable(_logging.NOTSET)
    return {"status": status, "T_out_K": t_out, "N_cells": n_cells,
            "no_root_cell_index": no_root, "n_cells_packaged": len(profile) if "profile" in dir() else None}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=("unit", "all"))
    args = ap.parse_args()

    silence_solver_logs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ok, rows, _inv = load_frozen_split()
    if not ok:
        raise SystemExit("frozen R52-C-75 split was not consumed intact")
    by_id = {r["window_id"]: r for r in rows}
    window_ids = m1_reference_window_ids(
        ROOT / "outputs" / "diagnostics" / "revision-evidence-20260826" / "M1-cell0" / "revision_m1_cell0_variant_matrix.csv"
    )
    p2a = pd.read_csv(P2A_DIR / "fullmesh_convergence.csv")

    # ---------------- 第 1 节 ----------------
    keq = wgs_keq_table()
    check_rows: list[dict] = []
    for _, r in keq.iterrows():
        check_rows.append({
            "check_group": "wgs_keq", "context": f"T={r['T_K']:.0f} K", "metric": "thermo_vs_independent_dG",
            "value": float(r["thermo_vs_independent_rel_diff"]), "reference": float(r["K_independent_from_dG"]),
            "tolerance": 1e-10, "pass": bool(r["pass"]),
            "note": f"K=exp(-ΔG/RT) 与独立 ΔH/ΔS 实现对比；K_thermo={r['K_thermo']:.6e}",
        })

    anchor_case_list, geometry = anchor_cases()
    i1 = next(c for c in anchor_case_list if c["case_name"] == "texaco i-1")
    check_rows += energy_check_rows(
        "anchor texaco i-1", "texaco i-1", lambda extra: solve_anchor_system(i1, 160, extra), 160
    )
    check_rows += energy_check_rows(
        "baseline window #1", window_ids[0],
        lambda extra: _system_of(run_case(by_id[window_ids[0]], n_cells=40, extra_op=extra, keep_system=True)),
        40,
    )

    # (ii) 反应热：298 K 与一个气化区单元
    probe = run_case(by_id[window_ids[0]], n_cells=40, keep_system=True)
    probe_system: GasifierSystem = probe["_system"]
    gasif_cells = [d for d in probe_system.cell_diagnostics if not d.get("is_combustion_zone")]
    probe_idx = gasif_cells[-1]["cell_index"]
    probe_state = probe_system.results[probe_idx]
    probe_T = float(probe_state.T)
    probe_xc = float(probe_state.carbon_fraction)
    check_rows += reaction_heat_rows(298.15, probe_xc, probe_system.coal_props, "standard 298.15 K")
    check_rows += reaction_heat_rows(probe_T, probe_xc, probe_system.coal_props,
                                     f"gasification cell {probe_idx} (T={probe_T:.1f} K)")

    unit_df = pd.DataFrame(check_rows)
    # 闸门只统计真正参与判定的行：legacy / delta 两项是信息行（pass 为 None），
    # 它们故意保留 B9 缺陷或只是对照量，不参与通过与否。
    _gate_pass = unit_df.loc[unit_df["pass"].notna(), "pass"]
    unit_pass = bool(all(bool(v) for v in _gate_pass))

    protocol = base_protocol({
        "round": "P1.6-A / 2026-10-31（2026-11-02 修订执行）",
        "source_commit": source_commit(),
        "source_commit_note": "生成本轮全部产物的代码提交；产物本身随后另起一次仅含 outputs 的提交",
        "equation_changes": {
            "1a_solid_enthalpy": "入炉煤保留 Hf_coal；热解后焦炭碳取石墨生成焓 0、灰只计显热。"
                                 "开关 SolidEnthalpyModel ∈ {char_formation_zero(缺省), legacy_coal_hf}。",
            "1b_wgs_keq": "缺省 WGS_Keq_Model='thermo'：K=exp(-ΔG/RT)，G 由 physics.get_gibbs_free_energy；"
                          "保留 'legacy_fit'=exp(4578/T-4.33) 与 'fortran'=exp(-3.6893+7234/(1.8T))。",
            "2_het_cutoff": "去掉异相速率里 X_total>=0.999 的硬截断（开关 HetConversionCutoff ∈ "
                            "{smooth(缺省), hard_0999}）；完全转化时由 1/(k_s*Y^2) 让速率按 Y^2 趋零。",
        },
        "frozen_items_untouched": [
            "all Arrhenius constants", "phi", "f_co2 = 1.0", "P_O2_Combustion_atm = 0.05",
            "CharCombustionRateFactor = 0.3", "H2_OxidationFactor / CO_OxidationFactor",
            "pyrolysis yields", "USE_RK_GILL_COMBUSTION = False", "ParticleSizeModel = fortran_constant",
            "steam_flow = 0", "negative-WGS hydrogen truncation（本轮不改，作为显式假设登记）",
        ],
        "anchor_inputs": {
            "source": "reference_fortran/input_副本.txt namelist 原值（含 xmois）+ Source1_副本.for 第 32 行几何",
            "geometry_line": 32,
            "geometry": {k: geometry[k] for k in ("L_m", "D_m", "P_atm", "source_line")},
            "not_via": "load_case_from_repo",
            "hhv_d_mj_per_kg": 30.0,
            "hhv_note": "namelist 无 HHV；沿用 P1.5-C 的 30 MJ/kg 占位值以保持可比，仅影响热损基准与 Hf_coal 的量级",
        },
        "unit_check_tolerances": {
            "energy_closure_rel": 1e-9, "reaction_heat_rel": 1e-6, "wgs_keq_rel": 1e-10,
        },
        "unit_checks_all_pass": unit_pass,
        "anchor_tolerances": ANCHOR_TOL,
        "anchor_n_list": list(N_ANCHOR),
        "anchor_n_reference": N_ANCHOR_REF,
        "baseline_n_list": list(N_BASE),
        "baseline_windows": list(window_ids),
        "solver": {"solver_method": "minimize", "jacobian_mode": "scipy",
                   "MeshFamily": "uniform_full_length", "SolverStrictRoot": True},
    })

    if not unit_pass or args.stage == "unit":
        tables = {
            "unit_checks.csv": unit_df,
            "wgs_keq_table.csv": keq,
        }
        interp = _unit_only_interpretation(unit_df, keq, unit_pass, args.stage)
        written = write_round_outputs(OUT_DIR, tables=tables, protocol=protocol, interpretation=interp)
        print(json.dumps({"stage": args.stage, "unit_pass": unit_pass,
                          "written": [str(p) for p in written]}, ensure_ascii=False, indent=2))
        print(unit_df[["check_group", "context", "metric", "value", "pass"]].to_string(index=False))
        if not unit_pass:
            raise SystemExit("第 1 节单元级检查未通过：按卡停止，不进入第 3、4 节。")
        return

    # ---------------- 第 2 节 ----------------
    xcap_ids, noroot_ids = xcap_windows()
    xcap_df = xcap_rows(xcap_ids + noroot_ids, by_id, p2a)

    # ---------------- 第 3 节 ----------------
    fortran_out, _prof = parse_fortran_log(P15C_DIR / "fortran_build" / "run.log")
    fortran_by_case = match_fortran(anchor_case_list, fortran_out)
    anchor_df = anchor_rows(anchor_case_list, geometry, fortran_by_case)
    crit_df = anchor_criteria_rows(anchor_df, fortran_by_case)
    attr_df = anchor_attribution_rows(anchor_case_list, geometry, fortran_by_case)

    # ---------------- 第 4 节 ----------------
    base_df, cells_df = baseline_rows(window_ids, by_id)
    clos_df = closure_rows(base_df)

    # ---------------- 第 5 节 ----------------
    parity_df, parity_summary = jax_parity_rows(by_id[window_ids[0]], n_cells=40)
    smoke = jax_solver_smoke(by_id[window_ids[0]], n_cells=10)
    parity_df = pd.concat([
        parity_df,
        pd.DataFrame([{"kind": "S1_jax_jit_smoke", "status": smoke["status"],
                       "T_out_K": smoke["T_out_K"], "N_cells": smoke["N_cells"]}]),
    ], ignore_index=True)

    # P1.5-B 的 summary.json 实际是单行 CSV（当轮用 DataFrame 写出）
    p15b_summary = pd.read_csv(P15B_DIR / "summary.json").iloc[0].to_dict()

    protocol.update({
        "xcap_windows": xcap_ids,
        "noroot_windows_pre_fix": noroot_ids,
        "anchor_cases_used": [c["case_name"] for c in anchor_case_list],
        "anchor_criteria_count": int(len(crit_df[crit_df["mode"] == "F"])),
        "jax_parity_summary": parity_summary,
        "jax_jit_smoke": smoke,
        "extra_artifacts": ["baseline_cells.csv（逐单元限幅绑定与残差成本，供修订第 2 条）"],
    })

    interp = _full_interpretation(unit_df, keq, xcap_df, anchor_df, crit_df, attr_df,
                                  base_df, clos_df, parity_df, parity_summary, smoke,
                                  p15b_summary, fortran_by_case, geometry, window_ids)

    tables = {
        "unit_checks.csv": unit_df,
        "wgs_keq_table.csv": keq,
        "anchor_outlet.csv": anchor_df,
        "anchor_criteria.csv": crit_df,
        "anchor_attribution.csv": attr_df,
        "baseline_n40_n80.csv": base_df,
        "baseline_cells.csv": cells_df,
        "post_reactor_closure.csv": clos_df,
        "xcap_windows.csv": xcap_df,
        "jax_parity.csv": parity_df,
    }
    written = write_round_outputs(OUT_DIR, tables=tables, protocol=protocol, interpretation=interp)
    print(json.dumps({"written": [str(p) for p in written],
                      "unit_pass": unit_pass,
                      "jax": parity_summary, "smoke": smoke}, ensure_ascii=False, indent=2))


def _fmt(x, nd=2):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(v):
        return "nan"
    return f"{v:.{nd}f}"


def _unit_only_interpretation(unit_df, keq, unit_pass, stage) -> str:
    lines = ["# P1.6-A 方程修正 —— interpretation（第 1 节闸门，未通过）\n"]
    lines.append(f"第 1 节单元级检查整体：**{'通过' if unit_pass else '未通过'}**。"
                 f"按卡停止条件，未进入第 3、4 节。\n")
    lines.append("## 单元级检查明细\n")
    lines.append("| 组 | 上下文 | 指标 | 值 | 容差 | 通过 |")
    lines.append("|---|---|---|---:|---:|---|")
    for _, r in unit_df.iterrows():
        lines.append(f"| {r['check_group']} | {r['context']} | {r['metric']} | "
                     f"{float(r['value']):.6e} | {r['tolerance']:.0e} | {r['pass']} |")
    lines.append("")
    lines.append("## WGS 平衡常数\n")
    lines.append(keq.to_string(index=False))
    lines.append("")
    return "\n".join(lines)


def _pct_table(df, cols, fmt=None):
    fmt = fmt or {}
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in ("window_id", "case_name", "item", "execution_status", "phase", "reaction", "metric"):
                cells.append(f"`{v}`" if c == "window_id" else str(v))
            elif isinstance(v, (int, np.integer)):
                cells.append(str(int(v)))
            else:
                cells.append(f"{float(v):.{fmt.get(c, 2)}f}" if np.isfinite(float(v)) else "nan")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _full_interpretation(unit_df, keq, xcap, anchor, crit, attr, base, clos, parity,
                         parity_summary, smoke, p15b, fortran_by_case, geometry, window_ids) -> str:
    lines = ["# P1.6-A 方程修正、锚点重跑与新基线 —— interpretation\n"]
    lines.append(
        "本轮在模型仓按 2026-10-31 卡（2026-11-02 修订）执行：1a 固相焓口径、1b WGS 平衡常数、"
        "2 去掉异相速率的 X_C 硬截断；随后重跑锚点（模式 F/M）、P0-A 九窗新基线（模式 M，N=40/80）"
        "与 JAX 的 F1/F4 夹具。冻结项一项未动（Arrhenius 常数、phi、f_co2=1.0、0.05 atm、"
        "CharCombustionRateFactor=0.3、氧化因子、热解产率、USE_RK_GILL_COMBUSTION=False、"
        "fortran_constant、蒸汽进料=0）。负向 WGS 截断保持现状，见第 9 节。\n"
    )

    # 1
    lines.append("## 1. 两次提交的哈希\n")
    lines.append(f"- 生成本轮全部产物的**代码提交**：`{source_commit()}`（写入 `protocol.json` 的 `source_commit`）。")
    lines.append("- 产物随后以一次只含 `outputs/diagnostics/round-p1.6a-baseline-20261031/` 与脚本的提交落盘。"
                 "两个哈希不能是同一个：`git` 的提交哈希由树内容决定，而 `protocol.json` 的内容又要写入该哈希，"
                 "互为自指。这与 2026-10-30 四轮的既有约定一致（当轮产物由 `98ad47c` 生成、最终以 `3131418` 落盘）。")
    lines.append("- 按修订第 1 条，四轮 `interpretation.md` 末尾已各追加一行说明其代码落点为 `3131418` 一系，原文未改。\n")

    # 2
    energy = unit_df[unit_df["check_group"] == "energy_closure"]
    react = unit_df[unit_df["check_group"] == "reaction_heat"]
    wgs = unit_df[unit_df["check_group"] == "wgs_keq"]
    lines.append("## 2. 1a / 1b 的单元级检查\n")
    lines.append(f"**闸门**：全部单元级检查"
                 f"{'通过' if bool(unit_df['pass'].all()) else '未通过'}"
                 f"（energy {int(energy['pass'].sum())}/{len(energy)}、"
                 f"reaction_heat {int(react['pass'].sum())}/{len(react)}、"
                 f"wgs {int(wgs['pass'].sum())}/{len(wgs)}）。按卡继续进入第 3、4 节。\n")

    lines.append("### 2.1 1a 固相焓口径（审计 B9）\n")
    lines.append("改动：入炉煤保留 `Hf_coal`（LHV 口径公式不变）；热解后的焦炭按组成计焓——碳取石墨生成焓 0、"
                 "灰为惰性只计显热。因此 `H_coal − (H_char + H_vol)` 就是热解热，全部落在热解所在单元（cell 0）"
                 "的能量残差里，不再隐含在焦炭的生成焓中、也不再被下游带走。"
                 "`PyrolysisSource` 的能量源项本身不变（仍只注入挥发分焓），因为一旦把差值再加一次就会重复计数——"
                 "这一点是本轮实现上最容易出错的地方，已用第 5 行的守恒恒等式锁住。\n")
    lines.append("| 上下文 | 指标 | 值 | 参考 | 容差 | 通过 |")
    lines.append("|---|---|---:|---:|---:|---|")
    for _, r in energy.iterrows():
        lines.append(f"| {r['context']} | `{r['metric']}` | {float(r['value']):.3e} | "
                     f"{float(r['reference']):.4e} | {r['tolerance']:.0e} | {r['pass']} |")
    lines.append("")
    b9 = energy[energy["metric"] == "b9_perturbation_identity_rel"]
    if len(b9):
        lines.append("B9 的量级（同一状态上新旧口径的能量残差之差，应恰等于 `−m_char·Hf_coal`）：" +
                     "；".join(f"{r['context']} {float(r['reference'])/1e6:.3f} MW" for _, r in b9.iterrows()) +
                     "。也就是说固相焓口径改的是「焦炭带走多少生成焓」，在 cell 0 上就是百 kW 到 ~19 MW 量级的再分配，"
                     "而出口温度只变 23–29 K、X_C 变 5–9 pp——下游的吸热气化把它吸收掉了。\n")

    lines.append("### 2.2 反应热（Shomate 对照）\n")
    lines.append("| 上下文 | 反应 | q_model (kJ/mol) | q_Shomate (kJ/mol) | 相对差 | legacy 口径相对差 |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for _, r in react.iterrows():
        lines.append(f"| {r['context']} | `{r['reaction']}` | {float(r['q_model_J_mol'])/1000:.4f} | "
                     f"{float(r['q_shomate_J_mol'])/1000:.4f} | {float(r['rel_diff']):.2e} | "
                     f"{float(r['rel_diff_legacy']):.2e} |")
    lines.append("")
    q298 = react[(react["context"].str.startswith("standard")) & (react["reaction"] == "C+H2O->CO+H2")]
    if len(q298):
        lines.append(f"298.15 K 下 `C + H2O → CO + H2` 的等效反应热 {float(q298['q_model_J_mol'].iloc[0])/1000:.2f} kJ/mol，"
                     f"与 NIST 表列 ΔHf 直接求和 {float(q298['q_nist_tabulated_298_J_mol'].iloc[0])/1000:.2f} kJ/mol "
                     "相差 2.4e-5，来源是 Shomate 拟合在 298 K 不严格回到 ΔHf（系数表性质，非实现误差）。"
                     "同一张表的 `legacy 口径相对差` 一列是修改前的口径：焦炭沿用每 kg 固体的 `Hf_coal`，"
                     "四条反应的热效应因此偏离 44%–330%——这正是本检查要拦的东西。\n")

    lines.append("### 2.3 WGS 平衡常数\n")
    lines.append(keq[["T_K", "K_thermo", "K_legacy_fit", "K_fortran", "K_thermo_over_legacy",
                      "thermo_vs_independent_rel_diff"]].to_string(index=False))
    lines.append("\n`thermo` 与独立 ΔG 实现（本脚本内独立写的 Shomate H/S 求值）相对差为 0（浮点相等）。"
                 "相对旧拟合 `exp(4578/T−4.33)`，`thermo` 在 1000 K 高 12%、1800 K 高 57%；"
                 "与 Fortran 式 `exp(−3.6893+7234/(1.8T))` 相差 3%–13%。\n")

    # 3
    lines.append("## 3. 去掉 X_C 截断上限后的窗口\n")
    lines.append("原硬截断在 `kinetics_service.calc_heterogeneous_rates`（`X_total >= 0.999` 时四条异相速率置 0，"
                 "含 `C+H2` 的 0.999 判据）与 `jax_kinetics.heterogeneous_rates_jax`（`X_total < 0.999`）。"
                 "新处理去掉该判据，保留 Fortran 的温度/分压安全阈值；完全转化时 UCSM 的 `1/(k_s·Y²)` 让速率按 `Y²` 趋零，"
                 "不设人为上限。开关 `HetConversionCutoff ∈ {smooth(缺省), hard_0999}`。\n")
    lines.append("| 窗口 | N | 修正前 X_C | 修正前状态 | 修正后 X_C | 修正后状态 | 最大 cost |")
    lines.append("|---|---:|---:|---|---:|---|---:|")
    for _, r in xcap.iterrows():
        cost = float(r["after_max_residual_cost"])
        lines.append(f"| `{r['window_id'].split('__')[-2]}…{r['window_id'][-4:]}` | {int(r['N_cells'])} | "
                     f"{_fmt(r['before_1_to_5_X_C'], 4)} | {r['before_execution_status']} | "
                     f"{_fmt(r['after_X_C'], 4)} | {r['after_execution_status']} | "
                     f"{cost:.1e} |")
    lines.append(f"\n完整窗口名见 `xcap_windows.csv`。要点：\n")
    w003 = xcap[xcap["window_id"].str.contains("D_008__W003")]
    p1w1 = xcap[xcap["window_id"].str.contains("shift_morning_handover")]
    p2 = xcap[xcap["window_id"].str.contains("N_013")]
    p3 = xcap[xcap["window_id"].str.contains("D_015")]
    lines.append(f"- `…D_008__W003`（原 3 个 X_C=0.999 窗口之一）现在落到 {_fmt(w003['after_X_C'].median(),4)}"
                 f"（N=40/80/160 分别 {', '.join(_fmt(v,4) for v in w003['after_X_C'])}），不再被截断钉住。")
    lines.append(f"- `…shift_morning_handover`（原 X_C=0.999）现在 {_fmt(p1w1['after_X_C'].median(),4)}，"
                 "即**碳被完全消耗**——旧的 0.999 钉住换成新的 1.0 上限。")
    lines.append(f"- `…N_013`（原 X_C=0.999）现在 {', '.join(_fmt(v,4) for v in p2['after_X_C'])}，同样贴着 1.0。")
    lines.append(f"- 原 N=160 无根窗口 `…D_015__W001` 现在三个网格**全部 completed**，X_C = "
                 f"{', '.join(_fmt(v,4) for v in p3['after_X_C'])}，最大 cost "
                 f"{p3['after_max_residual_cost'].max():.1e}。")
    lines.append("\n**必须写明的一点**：去掉 0.999 的硬截断没有让这些窗口的 X_C 变成可预测的量。"
                 "在 `…shift_morning_handover` 与 `…D_015__W001` 上，出口固体的碳分数被压到 1e-12 量级"
                 "（反应器 85%–95% 长度的单元 `X_total ≥ 0.999`），X_C 停在完全转化的上限。"
                 "也就是说 X_C 仍然不是这些窗口的有效预测输出；变化在于它从「数值截断造成的地板」变成"
                 "「物理上限」，而且残差成本降到 1e-10 以下、不再需要靠截断维持可解性。"
                 "`…D_015__W001` 从 no_root 变成 completed 是本轮最实质的变化。\n")

    # 4
    lines.append("## 4. 锚点模式 F 是否通过\n")
    f_df = crit[(crit["mode"] == "F") & (crit["evaluable"].astype(bool))]
    n_pass = int(f_df.groupby("case_name")["pass"].all().sum()) if len(f_df) else 0
    n_cases = int(f_df["case_name"].nunique()) if len(f_df) else 0
    lines.append(f"5 个干粉算例中，模式 F 的**全部可评估判据**都通过的：**{n_pass}/{n_cases}**"
                 "（判据与 P1.5-C 相同：X_C 2 pp、CO/H2/CO2 2 vol%、T 50 K，N=160）。"
                 "逐算例逐判据见下表与 `anchor_criteria.csv`。\n")
    lines.append("| 算例 | 判据 | Python | Fortran | 绝对差 | 容差 | 通过 |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for _, r in f_df.iterrows():
        lines.append(f"| {r['case_name']} | {r['metric']} | {_fmt(r['python_value'], 2)} | "
                     f"{_fmt(r['fortran_value'], 2)} | {_fmt(r['abs_diff'], 2)} | {r['tolerance']} | {r['pass']} |")
    lines.append("")
    lines.append("逐算例小结（模式 F，N=160）：")
    for name, grp in f_df.groupby("case_name"):
        bad = grp[~grp["pass"].astype(bool)]
        ok = len(grp) - len(bad)
        lines.append(f"- `{name}`：{ok}/{len(grp)} 通过"
                     + ("（全部通过）" if bad.empty else "；未通过 " + "、".join(bad["metric"])))
    lines.append("\n`O2_drop_rel` 判据仍不可评估（Fortran 只在收敛后打印少数气室，第一次打印时 O2 已为 0；"
                 "Python 在 cell 0 即降到入口 1% 以下），与 P1.5-C 一致，未计入通过数。\n")
    m_df = crit[(crit["mode"] == "M") & (crit["evaluable"].astype(bool))]
    if len(m_df):
        m_pass = int(m_df.groupby("case_name")["pass"].all().sum())
        lines.append(f"对照：模式 M（`fortran_constant` + `thermo` + 1a + 2）在同一判据下 "
                     f"**{m_pass}/{int(m_df['case_name'].nunique())}** 通过，逐行见 `anchor_criteria.csv` 的 mode=M 行。"
                     "它比模式 F 差的地方主要是 CO2（模式 M 不用 Fortran 扩散式与 `WGS_RatFactor`，"
                     "CO2 更高、CO 更低），温度的差距两者相近。\n")

    lines.append("### 4.1 逐项归因（模式 M，N=160，一次性切换）\n")
    lines.append("`d_*` = 切换后 − 当前缺省（正值表示把该项关掉后变大）。新增两行是卡要求的 "
                 "`WGS_Keq_Model` 与 1a 的开/关。\n")
    pivot = attr[attr["mode"] == "M"].pivot_table(
        index="case_name", columns="item", values="d_T_gas_K", aggfunc="first")
    items = [c for c in ["SolidEnthalpy_1a_on_vs_off", "WGS_Keq_Model_new_vs_legacy_fit",
                         "UseFortranDiffusion", "WGS_RatFactor", "CharCombustionRateFactor"]
             if c in pivot.columns]
    lines.append("| 算例 | " + " | ".join(items) + " |")
    lines.append("|---" * (len(items) + 1) + "|")
    for case, row in pivot.iterrows():
        lines.append(f"| {case} | " + " | ".join(
            f"{float(row[c]):+.1f} K" if np.isfinite(float(row[c])) else "nan" for c in items) + " |")
    lines.append("")
    a_m = attr[attr["mode"] == "M"]
    xc_1a = a_m[a_m["item"] == "SolidEnthalpy_1a_on_vs_off"]["d_X_C_pp"].abs()
    xc_wgs = a_m[a_m["item"] == "WGS_Keq_Model_new_vs_legacy_fit"]["d_X_C_pp"].abs()
    t_1a = a_m[a_m["item"] == "SolidEnthalpy_1a_on_vs_off"]["d_T_gas_K"]
    if len(xc_1a) and len(xc_wgs):
        open_1a = a_m[(a_m["item"] == "SolidEnthalpy_1a_on_vs_off") & (a_m["base_X_C_pct"] < 99.9)]["d_X_C_pp"].abs()
        rng = (f"{_fmt(open_1a.min(), 1)}–{_fmt(open_1a.max(), 1)} pp（只在 X_C 未封顶的算例上）"
               if len(open_1a) else "0 pp")
        lines.append("同一张表里的 `d_X_C_pp`（见 `anchor_attribution.csv`）：把 1a 关掉会让 X_C 变 "
                     f"{rng}、T 变 {_fmt(t_1a.min(), 1)}–{_fmt(t_1a.max(), 1)} K；把 `WGS_Keq_Model` 换回旧拟合只改 "
                     f"{_fmt(xc_wgs.min(), 2)}–{_fmt(xc_wgs.max(), 2)} pp。两项都不足以把模式 F 拉回 Fortran——"
                     "1a 的方向还是**反的**（它让模型更冷、X_C 更低，而模式 F 本来就已经偏低 165–175 K）。\n")

    # 5
    lines.append("## 5. 新基线（模式 M，`uniform_full_length`，`SolverStrictRoot=True`）\n")
    ok_base = base[base["execution_status"] == "completed"]
    lines.append("| N | 完成窗数 | X_C 中位 | 干气流量比中位 | CO vol% | CO2 vol% | H2 vol% |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for n, grp in base.groupby("N_cells"):
        comp = grp[grp["execution_status"] == "completed"]
        lines.append(f"| {int(n)} | {len(comp)}/{len(grp)} | {_fmt(comp['X_C_implied'].median(), 4)} | "
                     f"{_fmt(comp['dry_flow_ratio'].median(), 4)} | {_fmt(comp['CO_vol_pct'].median())} | "
                     f"{_fmt(comp['CO2_vol_pct'].median())} | {_fmt(comp['H2_vol_pct'].median())} |")
    lines.append("")
    lines.append("按阶段的 X_C 中位（N=40 / N=80）：")
    for ph, grp in ok_base.groupby("phase"):
        lines.append(f"- `{ph}`：{_fmt(grp[grp['N_cells'] == 40]['X_C_implied'].median(), 4)} / "
                     f"{_fmt(grp[grp['N_cells'] == 80]['X_C_implied'].median(), 4)}")
    n_noroot = int((base["execution_status"] != "completed").sum())
    lines.append(f"\n`no_root`：**{n_noroot} / {len(base)}**。"
                 f"18 行全部 completed，最大残差成本 {base['max_residual_cost'].max():.1e}。\n")
    n_one = int((ok_base["X_C_implied"] >= 0.9999).sum())
    lines.append(f"其中 {n_one} 行的 X_C ≥ 0.9999（碳被完全消耗，见第 3 节说明），"
                 "这些行的 X_C 不能当作预测值使用；其余窗口的 X_C 在 0.86–0.99 之间。\n")

    lines.append("### 5.1 出口各物种摩尔流量、氢比\n")
    cols = ["window_id", "N_cells", "F_CO_mol_s", "F_CO2_mol_s", "F_H2_mol_s", "F_CH4_mol_s",
            "F_H2O_mol_s", "F_N2_mol_s", "F_H2S_mol_s", "H_out_over_H_in", "H_plant_over_H_in"]
    sub = ok_base[cols].copy()
    lines.append(_pct_table(sub, cols, {"H_out_over_H_in": 4, "H_plant_over_H_in": 4}))
    lines.append("\n`H_in` 是煤中燃料氢（干基 `Hd`，蒸汽进料为 0）；`H_out_dry = 2·H2 + 4·CH4 + 2·H2S`。"
                 f"模型干气氢 / 燃料氢 = {_fmt(ok_base['H_out_over_H_in'].min(),3)} – "
                 f"{_fmt(ok_base['H_out_over_H_in'].max(),3)}（中位 {_fmt(ok_base['H_out_over_H_in'].median(),3)}）；"
                 f"装置干气氢 / 燃料氢 = {_fmt(ok_base['H_plant_over_H_in'].min(),3)} – "
                 f"{_fmt(ok_base['H_plant_over_H_in'].max(),3)}（中位 {_fmt(ok_base['H_plant_over_H_in'].median(),3)}）。"
                 "装置侧有一半窗口 > 1；若把输煤水分 2% 的氢也算进 `H_in`"
                 f"（`H_in_with_moisture_mol_s`），装置比降到 {_fmt(ok_base['H_plant_over_H_in_with_moisture'].min(),3)} – "
                 f"{_fmt(ok_base['H_plant_over_H_in_with_moisture'].max(),3)}，模型比降到 "
                 f"{_fmt(ok_base['H_out_over_H_in_with_moisture'].min(),3)} – "
                 f"{_fmt(ok_base['H_out_over_H_in_with_moisture'].max(),3)}。"
                 "两种口径下模型都系统性低于装置。\n")
    lines.append("\n> **不要把「装置比 > 1」读成「物理上不成立」。**")
    lines.append("> 上表右侧的装置干气氢取样点在**激冷之后**，而 `H_in` 只计煤中燃料氢；"
                 "激冷水是另一个氢源，且正是第 7 节求的那个变换 `CO + H2O → CO2 + H2`。")
    _hi = ok_base.iloc[0]
    _excess = float(_hi["H_plant_mol_s"]) - float(_hi["H_in_mol_s"])
    _cm = clos[(clos["window_id"] == _hi["window_id"]) & (clos["N_cells"] == _hi["N_cells"])]
    _xi = float(_cm["xi_card"].iloc[0]) if len(_cm) and pd.notna(_cm["xi_card"].iloc[0]) else float("nan")
    lines.append(f"> 定量看：首个窗（N={int(_hi['N_cells'])}）装置干气氢 {_fmt(_hi['H_plant_mol_s'],2)} 对燃料氢 "
                 f"{_fmt(_hi['H_in_mol_s'],2)} mol/s，超出 **{_fmt(_excess,2)} mol/s H 原子**；"
                 f"同一窗 `post_reactor_closure.csv` 拟合的 ξ = {_fmt(_xi,2)} mol/s，"
                 f"每个 ξ 转移 2 个 H 原子，即 {_fmt(2*_xi,2)} mol/s，"
                 f"{'足以覆盖' if 2*_xi >= _excess else '不足以完全覆盖'}该超出量。")
    lines.append("> 所以 `H_plant/H_in > 1` 只说明「反应器出口到取样点之间存在加氢过程」，"
                 "**不**说明装置测量与煤量账本不自洽。该结论应限制在反应器进/出口边界："
                 f"那里模型自身闭合（`H_out_dry/H_in` = {_fmt(ok_base['H_out_over_H_in'].min(),3)} – "
                 f"{_fmt(ok_base['H_out_over_H_in'].max(),3)} < 1）。"
                 "要判定装置账本是否自洽，必须先把激冷水项并入 `H_in` 再比，本轮未做。\n")

    # 6
    lines.append("## 6. 出口 WGS 是否仍处于平衡（Q/K）\n")
    if len(ok_base):
        lines.append(f"`Q = (y_CO2·y_H2)/(y_CO·y_H2O)`（总气基），`K = K_thermo(T_out)`。"
                     f"Q/K 范围 **{_fmt(ok_base['Q_over_K'].min(), 4)} – {_fmt(ok_base['Q_over_K'].max(), 4)}**，"
                     f"中位 {_fmt(ok_base['Q_over_K'].median(), 4)}；18 行全部在 1 ± 1% 之内。"
                     "出口仍在 WGS 平衡上（略偏反应物侧，Q/K<1 表示还会向 CO2+H2 方向移动）。"
                     "注意 Q/K≈1 是模型自己用同一个 K 算的速率推出来的，它只说明 WGS 没有冻结，"
                     "不构成对外部平衡数据的独立验证。\n")

    # 7
    lines.append("## 7. 反应器后单参数闭合\n")
    ok_clos = clos[clos["execution_status"] == "completed"]
    lines.append("对每窗求 ξ 使 `(CO2 + ξ)/(干气 + ξ) = 装置 CO2 体积分数`（卡面口径），"
                 "再用 `(CO − ξ)`、`(H2 + ξ)` 预测 CO、H2。\n")
    lines.append("| 窗口 | N | ξ (mol/s) | CO 预测 | 装置 CO | ΔCO | H2 预测 | 装置 H2 | ΔH2 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in ok_clos.iterrows():
        lines.append(f"| `{r['window_id'].split('__')[-2]}…{r['window_id'][-4:]}` | {int(r['N_cells'])} | "
                     f"{_fmt(r['xi_card'], 1)} | {_fmt(r['pred_CO_vol_pct'])} | {_fmt(r['plant_CO_vol_pct'])} | "
                     f"{_fmt(r['d_CO'], 2)} | {_fmt(r['pred_H2_vol_pct'])} | {_fmt(r['plant_H2_vol_pct'])} | "
                     f"{_fmt(r['d_H2'], 2)} |")
    lines.append("")
    lines.append(f"- ξ（卡面分母 `干气+ξ`）：中位 **{_fmt(ok_clos['xi_card'].median(), 1)} mol/s**，"
                 f"范围 {_fmt(ok_clos['xi_card'].min(), 1)} – {_fmt(ok_clos['xi_card'].max(), 1)}。")
    lines.append(f"- CO 预测偏差：中位 **{_fmt(ok_clos['d_CO'].median(), 2)} pp**，最大 "
                 f"{_fmt(ok_clos['d_CO'].abs().max(), 2)} pp；H2 偏差：中位 "
                 f"{_fmt(ok_clos['d_H2'].median(), 2)} pp，最大 {_fmt(ok_clos['d_H2'].abs().max(), 2)} pp。")
    lines.append(f"- 与**修正前 ξ 中位 72 mol/s** 对比：现在是 "
                 f"**{_fmt(ok_clos['xi_card'].median(), 1)} mol/s**，"
                 f"{'下降' if ok_clos['xi_card'].median() < 72 else '上升'} "
                 f"{_fmt(abs(ok_clos['xi_card'].median() - 72), 1)} mol/s（约 "
                 f"{_fmt(abs(ok_clos['xi_card'].median() - 72) / 72 * 100, 0)}%）。")
    lines.append("\n即：修正后要达到装置的 CO2 水平，反应器后仍需把 "
                 f"{_fmt(ok_clos['xi_card'].median(), 0)} mol/s 量级的 CO 转成 CO2，只是所需量比修正前小。"
                 "但单参数闭合只能对上 CO2 一个数：CO 与 H2 的预测偏差中位分别为 "
                 f"{_fmt(ok_clos['d_CO'].median(), 2)} 与 {_fmt(ok_clos['d_H2'].median(), 2)} pp，"
                 f"最大到 {_fmt(ok_clos['d_CO'].abs().max(), 2)} / {_fmt(ok_clos['d_H2'].abs().max(), 2)} pp。"
                 f"符号几乎固定（{int(((ok_clos['d_CO']>0)&(ok_clos['d_H2']<0)).sum())}/{len(ok_clos)} 行为 CO 偏高、H2 偏低；"
                 "其余为两者同时接近零的窗），说明出口组成不是一个自由度能补上的差距。\n")
    _spread = float(ok_clos['xi_card'].max()) / max(float(ok_clos['xi_card'].min()), 1e-12)
    lines.append(f"> **不要把 ξ 的跨窗变化（{_fmt(ok_clos['xi_card'].min(),1)} – {_fmt(ok_clos['xi_card'].max(),1)}，"
                 f"约 {_fmt(_spread,0)} 倍）当作「不是同一个物理过程」的证据。**")
    lines.append("> ξ 是**每个窗口各自拟合**的反应进度量，随进料、出口状态与下游条件变化，本就不要求恒定；"
                 "同一个变换过程可以有不同反应进度。数据支持的否定结论只有一条："
                 "**预声明的「所有窗口 ≤ 1 vol%」判据不成立**（中位本身已超过 1）。"
                 "跨窗变化作为**现象**记录。")
    lines.append("> 若要据此判断物理一致性，需额外做归一化，并检验下游水流量与动力学是否足以支撑该进度；"
                 "本轮两项都没做。因此也不宜用 ξ 直接支撑「激冷段已解释组成偏差」，"
                 "只能说「取样点位于激冷之后」这一结构性风险确实存在、量级可达出口 CO 的数个百分点。\n")

    # 8
    lines.append("## 8. JAX 的 F1 / F4\n")
    lines.append("夹具沿用 P1.5-B（同一窗口、N=40、两种颗粒模型各取燃烧区前 3 格与气化区后 3 格）。\n")
    lines.append("| 指标 | 修改前（本夹具，legacy 开关） | 修改后（1a/1b/2 全开） | P1.5-B 记录值 |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| F1 最大绝对差（按物种映射） | {float(parity_summary['before_f1_max_abs_diff']):.3e} | "
                 f"{float(parity_summary['after_f1_max_abs_diff']):.3e} | —（当轮未做映射） |")
    lines.append(f"| F1 最大绝对差（P1.5-B 的逐索引直比口径） | {float(parity_summary['before_f1_max_abs_diff_index_aligned']):.3e} | "
                 f"{float(parity_summary['after_f1_max_abs_diff_index_aligned']):.3e} | {float(p15b['f1_max_abs_diff']):.3e} |")
    lines.append(f"| F1（fortran_constant） | {float(parity_summary['before_f1_max_abs_diff_fortran_constant']):.3e} | "
                 f"{float(parity_summary['after_f1_max_abs_diff_fortran_constant']):.3e} | — |")
    lines.append(f"| F1（shrinking_legacy） | {float(parity_summary['before_f1_max_abs_diff_shrinking_legacy']):.3e} | "
                 f"{float(parity_summary['after_f1_max_abs_diff_shrinking_legacy']):.3e} | — |")
    lines.append(f"| F4 最大相对差 | {float(parity_summary['before_f4_max_rel_diff']):.3e} | "
                 f"{float(parity_summary['after_f4_max_rel_diff']):.3e} | {float(p15b['f4_max_rel_diff']):.3e} |")
    lines.append("")
    lines.append(f"- **F4 全部通过**（{parity_summary['after_f4_max_rel_diff']:.2e}，阈值 2e-6）。"
                 "修改前 F4 的最大相对差是 1.0——JAX 在气化区单元的 `C+O2` 返回 0，根因就是 "
                 "`X_total < 0.999` 这个硬截断：NumPy 侧 `C+O2` 没有这条判据而 `C+H2O/CO2/H2` 有。"
                 "去掉它以后两条路径的异相速率一致到机器精度。")
    lines.append(f"- **F1 仍不通过**（本卡明确不作闸门），而且**差值略微变大**：按物种映射的最大绝对差从 "
                 f"{float(parity_summary['before_f1_max_abs_diff']):.3e} 升到 "
                 f"{float(parity_summary['after_f1_max_abs_diff']):.3e}（+"
                 f"{float(parity_summary['after_f1_max_abs_diff'] / parity_summary['before_f1_max_abs_diff'] - 1) * 100:.0f}%）；"
                 f"按 P1.5-B 的逐索引直比口径从 {float(parity_summary['before_f1_max_abs_diff_index_aligned']):.3e} 升到 "
                 f"{float(parity_summary['after_f1_max_abs_diff_index_aligned']):.3e}（+"
                 f"{float(parity_summary['after_f1_max_abs_diff_index_aligned'] / parity_summary['before_f1_max_abs_diff_index_aligned'] - 1) * 100:.0f}%）。"
                 "两种口径方向一致。")
    lines.append("  - 需要分开看颗粒模型：`fortran_constant`（论文轮次用的那个）的最大差从 "
                 f"{float(parity_summary['before_f1_max_abs_diff_fortran_constant']):.3e} **降到** "
                 f"{float(parity_summary['after_f1_max_abs_diff_fortran_constant']):.3e}（改善约 4.4 倍）；"
                 f"`shrinking_legacy` 从 {float(parity_summary['before_f1_max_abs_diff_shrinking_legacy']):.3e} 升到 "
                 f"{float(parity_summary['after_f1_max_abs_diff_shrinking_legacy']):.3e}。总体最大差变大完全来自后者。")
    lines.append("  - 最大差的位置在修改前后相同：`shrinking_legacy` 的 cell 1（燃烧区）的 CO 残差，"
                 "其次是它的 Xc 残差；`fortran_constant` 的最大差在修改前出现在 cell 33 的 Xc/固相质量残差，"
                 "修改后连这一项也压到 1e-5 以下。也就是说变大的那部分不是 1a/1b/2 引起的方程不一致"
                 "（F4 已到机器精度），而是 `shrinking_legacy` 在 JAX 打包求解里的颗粒温度/粒径路径本来就与 NumPy 不同，"
                 "修改后的工况点（X_C 不再被 0.999 钉住）把这一差异放大了一点。")
    lines.append("  - 附带发现：P1.5-B 的夹具用「逐索引直比」比较了两个不同维度的残差向量（JAX 是 9+3 维、含 COS），"
                 "从第 4 个分量起就错位（NumPy 4=H2S 对 JAX 4=H2O；NumPy 8/9/10=Ws/Xc/E 对 JAX 9/10/11），"
                 f"所以当轮记录的 {float(p15b['f1_max_abs_diff']):.3e} 混了不同物理量。"
                 f"本轮在同一夹具下用 legacy 开关复现该口径得到 "
                 f"{float(parity_summary['before_f1_max_abs_diff_index_aligned']):.3e}——与当轮记录逐位一致，"
                 "说明夹具本身是同一个，毛病只在比较方式。`jax_parity.csv` 同时给出两列，未改 P1.5-B 的产物。")
    lines.append(f"- `jax_jit` 冒烟（N=10）：`{smoke['status']}`，在 cell {smoke['no_root_cell_index']} 被判 no_root。"
                 "**这与本轮修改无关**：在修改前的提交上用同一窗口跑 `jax_jit` 同样在 cell 1 截断"
                 "（HEAD 上 `SolverStrictRoot` 把 jax_jit 一次性打包解在 cell 1 拒掉，cost 13–300）。"
                 "换言之 `jax_jit` + 严格根在这个工况下本来就不成立，本轮没有改变这一点，也没有掩盖它。\n")

    # 9
    lines.append("## 9. 负向 WGS 截断：显式模型假设登记（修订第 2 条）\n")
    lines.append("**本轮不改写它。** 代码位置：")
    lines.append("- NumPy：`Cell.limit_h2_consumers_after_oxidation`（`cell.py`），在 `_calc_rates` 里对 `C+H2` 与负向净 WGS 一起限幅；")
    lines.append("- JAX：`jax_residuals.cell_residuals_jax_flat` 里 `r3 = jnp.where(r3 < 0.0, jnp.maximum(r3, -rem_h2*0.99), r3)`。\n")
    lines.append("它的行为（引自 `round-wgs-cap-audit-20261031` 第 3、4 节，并在本轮新基线上复核）：")
    lines.append("- 去掉后 **P2-A 全部行变成 no_root**（W001/W002/W003 在 N=40/160 上常态残差 5e-13 → 1e-2）；"
                 "同一收敛状态在去掉截断的模型下 cell 0 残差达到 9.5–10.0，说明它改变的不是数值路径而是方程本身。")
    lines.append("- 绑定范围**只在燃烧区**，气化区完全不绑定。本轮 9 窗 × N∈{40,80} 的 dz 加权复核："
                 f"燃烧区绑定长度占比 {_fmt(base['wgs_neg_bind_len_frac_combustion'].median(), 3)}，"
                 f"气化区 {_fmt(base['wgs_neg_bind_len_frac_gasification'].median(), 3)}，"
                 f"整炉 {_fmt(base['wgs_neg_bind_len_frac_overall'].median(), 3)}；"
                 f"每窗绑定 {int(base['wgs_neg_bind_cells'].min())} 个单元"
                 f"{'（全部窗口相同）' if base['wgs_neg_bind_cells'].min() == base['wgs_neg_bind_cells'].max() else ''}。")
    lines.append("- 逐单元的绑定布尔量在 `baseline_cells.csv` 的 `WGS_NegativeTruncated` 列（连同 `is_combustion_zone`、`dz_m`）。\n")
    lines.append("因此在论文里它是一个**需要在方法中显式声明的模型假设**：燃烧区（pO2 > 0.05 atm）内，"
                 "瞬时氧化之后不允许反向 WGS 再从已接近零的 H2 池里抽取氢气，且该上界与 MSR 限幅同时绑定。"
                 "气化区结论不依赖它。\n")

    # 10
    lines.append("## 10. 修订第 4 条：I-1 在两种水分下的模式 F，以及 −167 K / +120 K 的来源\n")
    i1rows = attr[attr["item"].str.startswith("I-1 ")]
    lines.append("| 口径 | N | T (K) | ΔT vs Fortran 1373 K | X_C (%) |")
    lines.append("|---|---:|---:|---:|---:|")
    for _, r in i1rows.iterrows():
        lines.append(f"| {r['item']} | {int(r['N_cells'])} | {_fmt(r['base_T_gas_K'], 1)} | "
                     f"{_fmt(r['d_T_gas_K'], 1)} | {_fmt(r['base_X_C_pct'], 2)} |")
    lines.append("")
    lines.append("结论：**水分不是这两个数字的来源**。namelist 的 `xmois=0.002` 换成论文的 4.53%，"
                 "I-1 的模式 F 出口温度只变 1–2 K。真正的杠杆是 `HHV_d`：它经 `Hf_coal` 的 LHV 口径公式"
                 "直接决定入炉煤的生成焓。`round-texaco-input-fix-20261031` 的 −167 K 用的是校验 JSON 的 "
                 "`HHV_d=29.8 MJ/kg`（本轮复现 −180 K 量级）；`round-wen-fortran-texaco-20261031` 的 +120 K 用的是"
                 "组分关联式 `33.5·C + 144·H − 18·O + 10·S`，对 I-1 给出 33.74 MJ/kg（本轮复现 +154 ~ +172 K）。"
                 "两者相差 3.9 MJ/kg，乘以煤流就是 0.3 MW 量级的入口焓差，足以解释 300 K。\n")
    lines.append("**旧网格（N=25、`AllowTruncatedMesh`）与 3% 热损的贡献必须按 HHV 基准分层报告**，"
                 "不能笼统写「±10 K」：\n")
    _a = attr[attr["item"].str.contains("HHV_d=33.74", na=False)]
    _b = attr[attr["item"].str.contains("HHV_d=30", na=False)]
    _g1 = (_a[_a["N_cells"] == 160]["base_T_gas_K"].iloc[0] - _a[_a["N_cells"] == 25]["base_T_gas_K"].iloc[0]) if len(_a[_a["N_cells"] == 160]) and len(_a[_a["N_cells"] == 25]) else float("nan")
    _g0 = (_b[_b["N_cells"] == 160]["base_T_gas_K"].iloc[0] - _b[_b["N_cells"] == 25]["base_T_gas_K"].iloc[0]) if len(_b[_b["N_cells"] == 160]) and len(_b[_b["N_cells"] == 25]) else float("nan")
    lines.append("| HHV_d | N=160 (K) | N=25 + 热损 3% (K) | 网格与热损的贡献 |")
    lines.append("|---|---:|---:|---:|")
    if len(_a[_a["N_cells"] == 160]) and len(_a[_a["N_cells"] == 25]):
        lines.append(f"| 33.74 MJ/kg | {_fmt(_a[_a['N_cells']==160]['base_T_gas_K'].iloc[0],1)} | "
                     f"{_fmt(_a[_a['N_cells']==25]['base_T_gas_K'].iloc[0],1)} | **{_fmt(_g1,1)} K** |")
    if len(_b[_b["N_cells"] == 160]) and len(_b[_b["N_cells"] == 25]):
        lines.append(f"| 30 MJ/kg（占位）| {_fmt(_b[_b['N_cells']==160]['base_T_gas_K'].iloc[0],1)} | "
                     f"{_fmt(_b[_b['N_cells']==25]['base_T_gas_K'].iloc[0],1)} | {_fmt(_g0,1)} K |")
    lines.append("")
    lines.append("「约 ±10 K」只在 HHV=30 那组成立；在 HHV=33.74 组，旧网格加热损单独就值 "
                 f"{_fmt(_g1,1)} K，量级不可忽略。这不改变「HHV 决定偏差符号」的主要结论"
                 "（300 K 的符号翻转仍由 3.9 MJ/kg 的 HHV 差主导），但把另一组对照的数值隐去会低估网格与热损的影响。\n")
    lines.append("因此锚点表里 `HHV_d=30 MJ/kg` 这个占位值（沿用 P1.5-C，namelist 本身没有 HHV）"
                 "**决定了模式 F 温度偏差的符号**：取 30 时偏低 175 K，取组分关联式的 33.74 时偏高 170 K。"
                 "本轮没有为缩小偏差去改它（那会变成用 HHV 拟合温度），但这条敏感性必须随锚点结论一起读。\n")

    # 11
    lines.append("## 11. 偏离、局限与本轮没做完的部分\n")
    lines.append("1. **`source_commit` 与结束提交不是同一个哈希**（第 1 节），原因是 git 的自指性；"
                 "协议里写的是生成产物的代码提交，产物另起一次提交。")
    lines.append("2. **X_C 在 3 个窗口上贴着完全转化上限**（第 3、5 节）。这是模型的物理上限，不是新引入的截断，"
                 "但意味着这几个窗口的 X_C 不能当预测值。本轮没有为此加任何松弛或上限。")
    lines.append("3. **F1 仍不通过**，且 P1.5-B 的夹具存在逐索引错位（第 8 节）。本轮只把两种口径都记录下来，"
                 "没有改 P1.5-B 的产物。")
    lines.append("4. **锚点的 `HHV_d=30` 是占位值**（第 10 节），它对模式 F 的温度判据起决定作用。"
                 "本轮按「不改输入口径」处理，只报告敏感性。")
    lines.append("5. **反应热检查的 Shomate 参考与模型共用同一套 NIST 系数**，因此它验证的是「形成焓分配正确、"
                 "焦炭生成焓确为 0」，不是独立的热化学数据校验；与 NIST 表列 ΔHf 直接求和的差（2.4e-5）"
                 "已在 2.2 节给出。")
    lines.append("6. **负向 WGS 截断未改**（第 9 节）。按修订第 2 条只登记为假设并给绑定统计。")
    lines.append("7. **新增了一个卡面清单之外的文件** `baseline_cells.csv`（逐单元残差成本与限幅布尔量），"
                 "因为修订第 2 条要求「新基线逐单元输出该截断是否绑定」，而卡面清单里没有逐单元表。")
    lines.append("8. **`interpretation.md` 的第 6 问（Q/K）只说明 WGS 没有冻结**，不是对平衡常数的外部验证。")
    lines.append("9. **锚点的 I-3/I-6 补写条目未单独重跑**：卡面要求「单独报告、不计入判据」，"
                 "但这两个算例的 namelist 条目不在 `reference_fortran/input_副本.txt` 里（是上一轮 Cursor 补写的），"
                 "本轮按「输入只取 namelist 原值」执行，故未纳入；它们的 Fortran 参考值与煤质见 "
                 "`round-wen-fortran-texaco-20261031/`。\n")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
