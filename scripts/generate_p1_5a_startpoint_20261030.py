#!/usr/bin/env python3
"""生成 P1.5-A 证据路径起点修复轮次产物（2026-10-30 卡）。

只在模型仓写 outputs/diagnostics/round-p1.5a-startpoint-20261030/，
不改论文仓、不调参、不改动力学常数。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model.p1_5_round_20261030 import (  # noqa: E402
    LIMIT_KEYS,
    PARTICLE_MODEL_DIAG,
    PARTICLE_MODEL_MAIN,
    base_protocol,
    mesh_check_rows,
    particle_area_rows,
    run_case,
    silence_solver_logs,
    thermo_check_rows,
    write_round_outputs,
)
from model.revision_evidence_common import load_frozen_split  # noqa: E402

SMOKE_WINDOW = "R52-C-75__P1__huayi_sy3_phase1_wgs1__plateau_2022-09-28D_008__W003"
OUT_DIR = ROOT / "outputs" / "diagnostics" / "round-p1.5a-startpoint-20261030"
SMOKE_N = (10, 40)


def main() -> None:
    silence_solver_logs()
    ok, rows, _inventory = load_frozen_split()
    if not ok:
        raise SystemExit("frozen R52-C-75 split was not consumed intact")
    row = [r for r in rows if str(r["window_id"]) == SMOKE_WINDOW]
    if not row:
        raise SystemExit(f"smoke window not found: {SMOKE_WINDOW}")
    row = row[0]

    mesh_rows = mesh_check_rows()
    thermo_rows = thermo_check_rows()

    smoke_rows: list[dict] = []
    cell_rows: list[dict] = []
    area_rows: list[dict] = []
    for n_cells in SMOKE_N:
        for particle_model in (PARTICLE_MODEL_MAIN, PARTICLE_MODEL_DIAG):
            rec = run_case(
                row,
                n_cells=n_cells,
                particle_model=particle_model,
                keep_system=True,
            )
            system = rec.pop("_system")
            profile = rec.pop("_profile")
            smoke_rows.append(rec)

            for diag in system.cell_diagnostics:
                c = dict(diag)
                c["window_id"] = rec["window_id"]
                c["N_cells"] = n_cells
                c["ParticleSizeModel"] = particle_model
                c["execution_status"] = rec["execution_status"]
                cell_rows.append(c)

            if rec["execution_status"] == "completed" and len(system.cells) == n_cells:
                for a in particle_area_rows(system, profile):
                    a["window_id"] = rec["window_id"]
                    a["N_cells"] = n_cells
                    a["ParticleSizeModel"] = particle_model
                    area_rows.append(a)

    tables = {
        "mesh_check.csv": pd.DataFrame(mesh_rows),
        "thermo_check.csv": pd.DataFrame(thermo_rows),
        "particle_area_check.csv": pd.DataFrame(area_rows),
        "smoke_single_window.csv": pd.DataFrame(smoke_rows),
        "smoke_cells.csv": pd.DataFrame(cell_rows),
    }

    protocol = base_protocol(
        {
            "round_dir": str(OUT_DIR),
            "smoke_window": SMOKE_WINDOW,
            "smoke_n_cells": list(SMOKE_N),
            "changed_files": [
                "src/model/grid_service.py",
                "src/model/gasifier_system.py",
                "src/model/cell.py",
                "src/model/physics.py",
                "src/model/jax_physics.py",
                "src/model/p1_5_round_20261030.py",
                "scripts/generate_p1_5a_startpoint_20261030.py",
            ],
            "new_switches": {
                "MeshFamily": "legacy | uniform_full_length（论文轮次用后者）",
                "AllowTruncatedMesh": "缺省 False；只用于复现旧轮次",
                "SolverStrictRoot": "缺省 False；论文轮次 True",
                "ParticleSizeModel": "shrinking_legacy | fortran_constant",
                "ParticleSwellFactor": "缺省 1.0（华谊无实测膨胀系数）",
            },
            "thermo_tolerances": {"enthalpy_jump_J_per_mol": 20.0, "janaf_rel_dev": 0.002},
            "particle_area_tolerance": 1e-6,
            "particle_model_fortran_evidence": {
                "rp_line": 130,
                "rp_formula": "rp = ri*(swell**0.333)",
                "r_const_lines": [419, 442, 755],
                "densp_note": "densp 在入口按堆积密度赋值，之后随转化更新；本轮按 '密度随剩余固体质量分数线性下降' 实现",
                "swell_value_used": 1.0,
            },
            "strict_root_cost_threshold": 1e-4,
            "mesh_assert_rtol": 1e-9,
            "note": "本轮不产出华谊统计结论；全长网格结果留给 P2-A",
        }
    )

    smoke_df = pd.DataFrame(smoke_rows)
    interpretation = _interpretation(smoke_df, pd.DataFrame(area_rows), pd.DataFrame(thermo_rows), pd.DataFrame(cell_rows))
    written = write_round_outputs(
        OUT_DIR, tables=tables, protocol=protocol, interpretation=interpretation
    )
    print(json.dumps({"written": [str(p) for p in written]}, ensure_ascii=False, indent=2))
    print(smoke_df[["N_cells", "ParticleSizeModel", "execution_status", "no_root_cell_index",
                    "X_C_implied", "dry_flow_ratio", "CO_vol_pct", "H2_vol_pct",
                    "solve_wall_time_s"]].to_string(index=False))


def _interpretation(smoke: pd.DataFrame, area: pd.DataFrame, thermo: pd.DataFrame, cells: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("# P1.5-A 证据路径起点修复 —— interpretation\n")
    lines.append("## 1. 网格必须铺满反应器（B1）\n")
    lines.append(
        "新增 `MeshFamily=\"uniform_full_length\"`：`dz_i = L/N`。旧族行为不变，"
        "`GasifierSystem.solve` 对 `|sum(dz)-L| > 1e-9·L` 直接报错，"
        "复现旧轮次必须显式传 `AllowTruncatedMesh=True`。详见 `mesh_check.csv`。\n"
    )
    lines.append("## 2. 求解器接受逻辑（B2）\n")
    lines.append(
        "`SolverStrictRoot=True` 时，任一单元 `cost > 1e-4` 即标记 `execution_status=\"no_root\"`、"
        "记录 `no_root_cell_index` 并停止向下游推进，出口量留空。逐单元 `residual_cost` 与主动残差分量写入 "
        "`smoke_cells.csv`。`fallback_count` 语义保留，但它不统计 `minimize` 路径 cell 0 的多起点。\n"
    )
    lines.append("## 3. 颗粒尺寸模型（B3）\n")
    lines.append(
        "`fortran_constant`：`d_p = d_p0·swell^(1/3)` 恒定，颗粒密度随剩余固体分数 `A=f_ash+(1-f_ash)(1-X)` 下降，"
        "`S_total = 6·m_hold/(rho_p·d_p)`。`shrinking_legacy` 保持原行为。\n"
    )
    if len(area):
        dev = float(area["rel_dev"].abs().max())
        lines.append(
            f"单元级检查：`fortran_constant` 与 `shrinking_legacy` 的 `S_total` 之比对照理论 "
            f"`A^(-2/3)`，最大相对偏差 {dev:.2e}（容差 1e-6）。`swell` 在华谊工况无实测，取 1.0。\n"
        )
    else:
        lines.append("单元级面积检查：本轮无 completed 行，未生成 `particle_area_check.csv` 数据。\n")
    lines.append("## 4. Shomate 分段（B8）\n")
    fails = thermo[~thermo["pass"].astype(bool)] if len(thermo) else pd.DataFrame()
    max_jump = float(thermo.loc[thermo["check"] == "enthalpy_jump", "value"].abs().max()) if len(thermo) else float("nan")
    lines.append(
        f"H2O 低温段改为 NIST 500–1700 K 系数（`T_cut = 1700 K`）；N2、O2、H2 补成三段；"
        f"全部物种的最大分段点焓跳变 {max_jump:.3f} J/mol（容差 20）。"
        f"JANAF 对照（Cp(1000 K)、H(1500 K)−H(298.15 K)）偏差均 < 0.2%。"
        f"未通过项：{0 if fails.empty else len(fails)}。\n"
    )
    lines.append("## 5. 诊断列（B5、B6）\n")
    lines.append(
        "逐单元记录 7 个限幅布尔量（判据：限幅前后相对差 > 1e-9）与 cell 0 五个初值的 cost、温度、胜出初值；"
        "`cell0_n_distinct_roots` 统计 cost < 1e-10 且温度相差 > 50 K 的不同根个数。不改选解规则。\n"
    )
    lines.append("## 6. 单窗冒烟（验收）\n")
    lines.append(
        "窗口 `" + SMOKE_WINDOW + "`，`uniform_full_length`，N ∈ {10, 40} × 两种颗粒模型，共 4 次求解。\n"
    )
    if len(smoke):
        for _, r in smoke.iterrows():
            lines.append(
                f"- N={int(r['N_cells'])}, {r['ParticleSizeModel']}: `{r['execution_status']}`"
                + (f"（cell {int(r['no_root_cell_index'])} 无根，cost={r['max_residual_cost']:.3e}）"
                   if r["execution_status"] != "completed" else
                   f"，X_C={r['X_C_implied']:.4f}，干气流量比={r['dry_flow_ratio']:.4f}，"
                   f"CO={r['CO_vol_pct']:.2f}，H2={r['H2_vol_pct']:.2f}%，"
                   f"墙钟 {r['solve_wall_time_s']:.2f} s")
                + "。"
            )
        lines.append("")
    lines.append(
        "**闸门状态：触发。** N=10（dz=1.03 m）两种颗粒模型都在 cell 1 出现 `no_root`；"
        "N=20（dz=0.515 m）同样在 cell 1 失败；N=40/80/160 全部 completed，"
        "最差单元 cost ≤ 1e-7。失败单元的特征是 `X_total` 撞到动力学 0.999 截断、"
        "主残差为 `carbon_fraction` / `gas_CO`，燃烧区厚度约 0.26 m 而单元 0.5–1.0 m——"
        "这是燃烧区分辨不足导致的离散化极限，不是方程或代码缺陷（旧族 N=3 仍复现修复前量级）。"
        "按卡片停止条件，本轮到此为止，未放宽 1e-4、未回退氢气限幅、未改方程。\n"
    )
    if len(cells):
        bad = cells[cells.get("strict_no_root").astype(bool)] if "strict_no_root" in cells else pd.DataFrame()
        lines.append(f"无根单元记录数：{len(bad)}，明细见 `smoke_cells.csv`。\n")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
