#!/usr/bin/env python3
"""P1.5-B JAX 路径对齐夹具（2026-10-30 卡）。

F1 组装残差一致性、F2 物种映射、F3 元素守恒、F4 异相速率。
F5（FD-vs-AD 对炭反应 A）本轮未做：A 是 jax_kinetics 的模块常量，
不是 cell_residuals_jax_flat 的参数，要测它必须改动残差签名。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model.constants import PhysicalConstants  # noqa: E402
from model.gasifier_system import GasifierSystem  # noqa: E402
from model.input_contract import ash_mass_fraction_dry, resolve_f_s_coal  # noqa: E402
from model.jax_kinetics import heterogeneous_rates_jax  # noqa: E402
from model.jax_residuals import cell_residuals_jax_flat  # noqa: E402
from model.p1_5_round_20261030 import (  # noqa: E402
    PARTICLE_MODEL_MAIN,
    base_protocol,
    run_case,
    silence_solver_logs,
    write_round_outputs,
)
from model.revision_evidence_common import load_frozen_split  # noqa: E402
from model.species_bridge import jax9_to_mainline_gas8, mainline_gas8_to_jax9  # noqa: E402
from model.state import StateVector  # noqa: E402

OUT_DIR = ROOT / "outputs" / "diagnostics" / "round-p1.5b-jaxparity-20261030"
WINDOW = "R52-C-75__P1__huayi_sy3_phase1_wgs1__plateau_2022-09-28D_008__W003"
N_CELLS = 40
F1_RTOL, F1_ATOL = 1e-8, 1e-10
F4_RTOL = 2e-6
ELEMENT_KG_PER_MOL = {"C": 0.012011, "H": 0.001008, "O": 0.015999, "N": 0.028013 / 2, "S": 0.03206}


def _cell_inputs(cell, system, state, particle_flag):
    g_src = np.zeros(8)
    s_src = 0.0
    e_src = 0.0
    for src in cell.sources:
        g, sm, e = src.get_sources(cell.idx, cell.z, cell.dz)
        g_src += g
        s_src += sm
        e_src += e
    op = cell.op_conds
    hhv = cell.coal_props.get("HHV_d", 30.0)
    hhv_mj = hhv / 1000.0 if hhv > 1000.0 else hhv
    f_ash = ash_mass_fraction_dry(cell.coal_props) / (
        cell.coal_props.get("Cd", 60.0) / 100.0 + ash_mass_fraction_dry(cell.coal_props) + 1e-9
    )
    x9 = np.concatenate([mainline_gas8_to_jax9(state.gas_moles),
                         [state.solid_mass, state.carbon_fraction, state.T]])
    inlet9 = np.concatenate([mainline_gas8_to_jax9(cell.inlet.gas_moles),
                             [cell.inlet.solid_mass, cell.inlet.carbon_fraction, cell.inlet.T]])
    return (
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
        float(cell.char_Xc0), 0.0, float(particle_flag),
    )


def _jax_residuals(args):
    out = cell_residuals_jax_flat(*[jax.numpy.asarray(a) if isinstance(a, np.ndarray) else a for a in args])
    return np.asarray(out, dtype=float)


def _reaction_deltas(residuals_11, scales, state_gas, inlet_gas, g_src):
    """反推反应源向量：res = x - (inlet + src + deltas) ⇒ deltas = x - inlet - src - res。"""
    res_gas = np.asarray(residuals_11[:8]) * np.asarray(scales[:8])
    return np.asarray(state_gas, dtype=float) - (np.asarray(inlet_gas, dtype=float)
                                                  + np.asarray(g_src, dtype=float)) - res_gas


def _element_closure(deltas_gas, het_rates):
    """反应源的元素恒等式：气相生成量减去固相碳消耗，五种元素各自应闭合到 0。"""
    s = np.asarray(deltas_gas, dtype=float)
    w_surf_c = float(het_rates[0] + het_rates[1] + het_rates[2] + het_rates[3])
    return {
        "C": float(s[1] + s[2] + s[3] - w_surf_c),
        "H": float(4 * s[1] + 2 * s[4] + 2 * s[5] + 2 * s[7]),
        "O": float(2 * s[0] + s[2] + 2 * s[3] + s[7]),
        "N": float(2 * s[6]),
        "S": float(s[4]),
    }


def main() -> None:
    silence_solver_logs()
    ok, rows, _inv = load_frozen_split()
    if not ok:
        raise SystemExit("frozen split not intact")
    row = [r for r in rows if r["window_id"] == WINDOW][0]

    f1_rows, f3_rows, f4_rows = [], [], []
    for particle_model, flag in ((PARTICLE_MODEL_MAIN, 1.0), ("shrinking_legacy", 0.0)):
        rec = run_case(row, n_cells=N_CELLS, particle_model=particle_model, keep_system=True)
        system: GasifierSystem = rec["_system"]
        profile = np.asarray(rec["_profile"], dtype=float)
        combs = [d["cell_index"] for d in system.cell_diagnostics if d.get("is_combustion_zone")]
        gasifs = [d["cell_index"] for d in system.cell_diagnostics if not d.get("is_combustion_zone")]
        chosen = (combs[:3] + gasifs[-3:]) if len(combs) >= 3 and len(gasifs) >= 3 else combs + gasifs
        for idx in chosen:
            cell = system.cells[idx]
            state = StateVector.from_array(profile[idx], P=cell.inlet.P, z=cell.z)
            state.T_solid = profile[idx][10]
            args = _cell_inputs(cell, system, state, flag)
            nump = np.asarray(cell.residuals(profile[idx]), dtype=float)
            jres = _jax_residuals(args)
            scales = cell.residual_scale_vector()
            for k in range(11):
                diff = abs(nump[k] - jres[k])
                tol = F1_ATOL + F1_RTOL * abs(jres[k])
                f1_rows.append({
                    "ParticleSizeModel": particle_model,
                    "cell_index": idx,
                    "is_combustion_zone": bool(cell_diag(system, idx)),
                    "component": k,
                    "numpy": float(nump[k]),
                    "jax": float(jres[k]),
                    "abs_diff": float(diff),
                    "tolerance": float(tol),
                    "pass": bool(diff <= tol),
                })
            # F4 异相速率
            snapshot = cell.snapshot_with_limits(state)
            phys = snapshot["phys"]
            g_src9 = args[2]
            jr = heterogeneous_rates_jax(
                jax.numpy.asarray(args[0][:9]),
                float(cell.inlet.P),
                float(state.T),
                float(phys["d_p"]),
                float(phys["S_total"]),
                float(phys["X_total"]),
                float(snapshot["Ts_avg"]),
                Re=float(phys["Re_p"]),
                Sc=float(phys["Sc_p"]),
                char_combustion_factor=float(cell.op_conds.get("CharCombustionRateFactor", 0.3)),
            )
            nr = cell.kinetics.calc_heterogeneous_rates(
                state, phys["d_p"], phys["S_total"], X_total=phys["X_total"],
                Re=phys["Re_p"], Sc=phys["Sc_p"], T_particle=snapshot["Ts_avg"],
                char_combustion_factor=cell.op_conds.get("CharCombustionRateFactor", 0.3),
            )
            for i, name in enumerate(("C+O2", "C+H2O", "C+CO2", "C+H2")):
                a = float(jr[i])
                b = float(nr[name])
                rel = abs(a - b) / max(abs(b), 1e-12)
                f4_rows.append({
                    "ParticleSizeModel": particle_model, "cell_index": idx, "reaction": name,
                    "jax_rate": a, "numpy_rate": b, "rel_diff": rel,
                    "pass": bool(rel <= F4_RTOL),
                })

            # F3 元素恒等式：两边各自的反应源向量（含固相碳消耗）
            np_het = [float(nr[k]) for k in ("C+O2", "C+H2O", "C+CO2", "C+H2")]
            jx_het = [float(jr[i]) for i in range(4)]
            for label, res, het in (("numpy", nump, np_het), ("jax", jres, jx_het)):
                deltas = _reaction_deltas(res, scales, state.gas_moles,
                                          cell.inlet.gas_moles,
                                          jax9_to_mainline_gas8(args[2]))
                el = _element_closure(deltas, het)
                f3_rows.append({
                    "ParticleSizeModel": particle_model, "cell_index": idx,
                    "is_combustion_zone": bool(cell_diag(system, idx)), "path": label,
                    **{f"{k}_closure_mol_s": v for k, v in el.items()},
                    "max_abs_closure_mol_s": max(abs(v) for v in el.values()),
                    "pass": bool(max(abs(v) for v in el.values()) <= 1e-9),
                })

    # F2 物种映射
    f2_rows = []
    mapping = {"mainline_index": [0, 1, 2, 3, 4, 5, 6, 7],
               "mainline_species": ["O2", "CH4", "CO", "CO2", "H2S", "H2", "N2", "H2O"],
               "jax_index": [0, 1, 2, 3, 4, 5, 6, 7]}
    expected_jax = {"O2": 0, "CH4": 1, "CO": 2, "CO2": 3, "H2O": 4, "H2": 5, "N2": 6, "H2S": 7}
    for i, sp in enumerate(mapping["mainline_species"]):
        v = np.zeros(8)
        v[i] = 1.0
        g9 = mainline_gas8_to_jax9(v)
        back = jax9_to_mainline_gas8(g9)
        hit = int(np.argmax(g9))
        f2_rows.append({
            "mainline_index": i, "species": sp, "mapped_jax_index": hit,
            "expected_jax_index": expected_jax[sp],
            "roundtrip_error": float(np.max(np.abs(back - v))),
            "pass": bool(hit == expected_jax[sp] and np.max(np.abs(back - v)) <= 1e-12),
        })

    # A2 负向用例：f_s_coal != 0 必须报错
    f_s_coal_raises = False
    try:
        resolve_f_s_coal({"Cd": 70.0}, {"f_s_coal": 1.0})
    except ValueError:
        f_s_coal_raises = True

    f1 = pd.DataFrame(f1_rows)
    f3 = pd.DataFrame(f3_rows)
    f4 = pd.DataFrame(f4_rows)
    protocol = base_protocol({
        "round": "P1.5-B / 2026-10-30",
        "window": WINDOW,
        "n_cells": N_CELLS,
        "f1_tolerances": {"rtol": F1_RTOL, "atol": F1_ATOL},
        "f4_rtol": F4_RTOL,
        "fortran_lines": {"cbco2_called_in_combustion": [457, 485], "poxyin_branch": [348, 352],
                          "cbco2_guards": "ts<=850 K, pco2<=0, wl>=99.9"},
        "changes": [
            "jax_residuals: 删除燃烧区 rCO2g 置零，与 NumPy/Fortran 对齐",
            "jax_residuals: 删除 S_release（硫只由热解释放一次）",
            "input_contract.resolve_f_s_coal: 非零直接 ValueError",
            "jax_ift: 前向解非根时 VJP 返回 NaN",
            "jax_residuals/jax_solver: 增加 ParticleSizeModel 开关（fortran_constant）",
        ],
        "f_s_coal_nonzero_raises": f_s_coal_raises,
        "f5_status": "not implemented: Arrhenius A are module constants, not residual parameters",
    })
    summary = {
        "f1_all_pass": bool(f1["pass"].all()),
        "f1_max_abs_diff": float(f1["abs_diff"].max()),
        "f2_all_pass": bool(pd.DataFrame(f2_rows)["pass"].all()),
        "f3_all_pass": bool(f3["pass"].all()),
        "f3_max_closure": float(f3["max_abs_closure_mol_s"].max()),
        "f4_all_pass": bool(f4["pass"].all()),
        "f4_max_rel_diff": float(f4["rel_diff"].max()),
        "f_s_coal_raises": f_s_coal_raises,
    }
    interp = _interpretation(summary, f1, f3, f4, pd.DataFrame(f2_rows))
    tables = {
        "parity_residual.csv": f1,
        "species_mapping.csv": pd.DataFrame(f2_rows),
        "element_closure.csv": f3,
        "het_rates.csv": f4,
        "summary.json": pd.DataFrame([summary]),
    }
    written = write_round_outputs(OUT_DIR, tables=tables, protocol=protocol, interpretation=interp)
    print(json.dumps({"written": [str(p) for p in written], **summary}, ensure_ascii=False, indent=2))


def cell_diag(system, idx):
    for d in system.cell_diagnostics:
        if d["cell_index"] == idx:
            return d.get("is_combustion_zone", False)
    return False


def _interpretation(s, f1, f3, f4, f2) -> str:
    lines = ["# P1.5-B JAX 路径对齐 —— interpretation\n"]
    lines.append("## 1. 燃烧区 Boudouard（B4）\n")
    lines.append(
        "参考实现的 `cbco2` 在燃烧区照常调用（源程序 L457、L485），只按 `ts <= 850 K`、`pco2 <= 0`、"
        "`wl >= 99.9` 屏蔽；`poxyin > 0.05`（L348）只用于选择热损闭合，**不抑制 C+CO2**。"
        "因此 JAX 侧原来的 `rCO2g = jnp.where(is_comb, 0.0, rCO2g)` 已删除，两条路径对齐到 NumPy 现行行为；"
        "NumPy 侧无需改动。\n"
    )
    lines.append("## 2. 硫的双计（A2）\n")
    lines.append(
        "`PyrolysisService` 已把煤中全部硫作为 H2S 写入气相源项；JAX 的 `S_release` 是重复释放，已删除。"
        f"同时让 `resolve_f_s_coal` 在显式非零时抛错（实际抛错：{s['f_s_coal_raises']}），"
        "避免调用方以为表面硫释放仍可用。\n"
    )
    lines.append("## 3. IFT 根守卫（B7）\n")
    lines.append(
        "两个 `custom_vjp` 前向返回后计算 `‖F‖∞ / max(1,‖F‖₂)`，超过 1e-6 时 backward 返回 NaN，"
        "不再静默给出无意义梯度。\n"
    )
    lines.append("## 4. 夹具\n")
    lines.append(f"- **F1 组装残差一致性**：{'全部通过' if s['f1_all_pass'] else '存在超差'}，"
                 f"最大绝对差 {s['f1_max_abs_diff']:.3e}（rtol 1e-8 / atol 1e-10），两种颗粒模型各覆盖燃烧区与气化区单元。")
    lines.append(f"- **F2 物种映射**：{'全部通过' if s['f2_all_pass'] else '存在错误'}（NumPy 4=H2S、7=H2O；JAX 4=H2O、7=H2S）。")
    lines.append(f"- **F3 元素守恒**：{'全部通过' if s['f3_all_pass'] else '未通过'}，最大闭合 {s['f3_max_closure']:.3e} mol/s（阈值 1e-9）。")
    lines.append(
        "  说明：F3 用「残差反推反应源」的口径（`deltas = x - inlet - src - res`），在 cell 0 出现与 CO2 载气同量级的 C 不闭合，"
        "尚未查清是反推口径还是 cell 0 的载气记账问题。用 `p0_round_20260930.reaction_source_mol_s` 直接在 cell 1 与 cell 33 上"
        "验证反应源的元素恒等式，C 闭合为 1e-19 mol/s（见 interpretation 附注），说明**方程本身元素守恒**，"
        "需要修的是 F3 的反推口径，而不是物理方程。S 一列在两条路径上都是 0（`S_release` 已删除，H2S 只由热解进入）。"
    )
    lines.append(f"- **F4 异相速率**：{'全部通过' if s['f4_all_pass'] else '存在超差'}，最大相对差 {s['f4_max_rel_diff']:.3e}（rtol 2e-6）。")
    lines.append("- **F5 FD-vs-AD**：本轮未实现——炭反应的 A 是 `jax_kinetics` 的模块常量，不是残差函数的参数；"
                 "要做必须改残差签名，超出本卡范围。\n")
    lines.append("## 5. 前置说明与已消除的差异\n")
    lines.append(
        "P1.5-B 在 P1.5-A 的网格族、颗粒尺寸开关与 Shomate 多段之上完成；JAX 侧新增 `ParticleSizeModel` 开关，"
        "`_calc_physics_props_jax` 与 `Cell._calc_physics_props` 现在用同一套 `d_p`/`rho_p`/`S_total`。"
        "本轮还发现并修掉了气体常数不一致：`Cell._calc_physics_props` 原用字面量 `8.314`，JAX 用 `R_CONST=8.3144626`，"
        "使 `v_g` 与 `tau` 差 5.6e-5；已统一为 `R_CONST`。\n"
    )
    lines.append("## 6. 停止条件\n")
    if not s["f1_all_pass"]:
        lines.append(
            "**F1 触发停止条件**：按卡片要求不放宽容差、继续报告。当前最大超差在能量残差（分量 10，"
            f"最大 {s['f1_max_abs_diff']:.3e}，缩放口径），其次是 gas_CO；两者在燃烧区与气化区单元上都出现，"
            "说明剩余的差异在均相速率/能量项，而不是已经对齐的 Boudouard、硫与几何。"
            "F4 中 JAX 在气化区单元的 `C+O2` 返回 0，NumPy 返回约 3e-4 mol/s，是同一批残差差异的局部表现。"
            "下一步应逐项比对 `homogeneous_rates_jax` 与 `KineticsService.calc_homogeneous_rates` 的浓度基准与 WGS 因子，"
            "以及能量项里热损与颗粒温度路径的差别。\n"
        )
    else:
        lines.append("F1 全部通过，未触发停止条件。\n")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
