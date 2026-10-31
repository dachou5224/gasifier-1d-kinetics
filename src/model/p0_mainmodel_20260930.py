"""P0 主模型：先核对 cell 0 氢气汇是否重复扣减，再决定是否改顺序限幅。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .p0_followup_20260930 import _dry_kpis, _identity_from_solve
from .p0_round_20260930 import (
    DATASET_ID,
    HANDOFF_MODEL_HEAD,
    POOR_CONVERGENCE_COST_THRESHOLD,
    SPECIES_NAMES,
    assert_no_local_routing,
    evaluate_cell_state,
    silence_solver_logs,
    solve_reference_case,
    source_commit,
    write_sha256_manifest,
)
from .p0d_20260930 import ELEMENT_IDENTITY_STOP_MOL_S

SINK_ATOL_MOL_S = 1.0e-6
STAGE0_MIN_WINDOWS = 7
STAGE0_N_CELLS = 3


def h2_sink_row(row: Mapping[str, Any], solved: Mapping[str, Any]) -> dict[str, Any]:
    """在 cell 0 的接受解上取可用氢气和三个汇，不改速率。"""
    cell = solved["system"].cells[0]
    state = np.asarray(solved["profile"][0], dtype=float)
    evaluated = evaluate_cell_state(cell, state)
    avail_h2 = max(float(cell.inlet.gas_moles[5]) + float(evaluated["gas_src"][5]), 0.0)
    r_h2_ox = float(evaluated["r_homo"].get("H2_Ox", 0.0))
    r_c_h2 = float(evaluated["r_het"].get("C+H2", 0.0))
    r_rwgs = float(evaluated["r_homo"].get("RWGS", 0.0))
    sink = r_h2_ox + 2.0 * r_c_h2 + r_rwgs
    return {
        "dataset_id": DATASET_ID,
        "window_id": row["window_id"],
        "phase": int(row["phase_number"]),
        "split_role": row["split_role"],
        "N_cells": STAGE0_N_CELLS,
        "cell_index": 0,
        "avail_H2_mol_s": avail_h2,
        "r_H2_Ox_mol_s": r_h2_ox,
        "r_C_H2_mol_s": r_c_h2,
        "r_RWGS_mol_s": r_rwgs,
        "r_WGS_mol_s": float(evaluated["r_homo"].get("WGS", 0.0)),
        "r_C_H2O_mol_s": float(evaluated["r_het"].get("C+H2O", 0.0)),
        "r_MSR_mol_s": float(evaluated["r_homo"].get("MSR", 0.0)),
        "h2_sink_mol_s": sink,
        "sink_minus_avail_mol_s": sink - avail_h2,
        "double_draw_exceeds": bool(sink > avail_h2 + SINK_ATOL_MOL_S),
    }


def stage0_passes(frame: pd.DataFrame) -> bool:
    return int(frame["double_draw_exceeds"].astype(bool).sum()) >= STAGE0_MIN_WINDOWS


def run_stage0(rows: Sequence[Mapping[str, Any]], window_ids: Sequence[str], output_dir: Path) -> dict[str, Any]:
    silence_solver_logs()
    output_dir.mkdir(parents=True, exist_ok=True)
    by_id = {str(row["window_id"]): row for row in rows}
    selected = [by_id[str(window_id)] for window_id in window_ids]
    records = []
    for index, row in enumerate(selected, start=1):
        print(f"P0-main stage0 {index}/{len(selected)} {row['window_id']}", flush=True)
        solved = solve_reference_case(row, n_cells=STAGE0_N_CELLS)
        records.append(h2_sink_row(row, solved))
    frame = pd.DataFrame(records)
    path = output_dir / "stage0_h2_sinks.csv"
    frame.to_csv(path, index=False)
    passed = stage0_passes(frame)
    return {
        "passed": passed,
        "n_exceed": int(frame["double_draw_exceeds"].astype(bool).sum()),
        "n_windows": len(frame),
        "path": path,
        "frame": frame,
    }


def write_stage0_stop(output_dir: Path, result: Mapping[str, Any]) -> None:
    """7/9 不成立时只写停止记录，不改限幅。"""
    commit = source_commit()
    n_exceed = int(result["n_exceed"])
    interpretation = (
        "# P0 主模型阶段 0\n\n"
        f"{n_exceed}/9 个窗口满足 r_H2_Ox + 2*r(C+H2) + r_RWGS > avail_H2 + 1e-6。"
        "没有达到 7/9。没有改顺序限幅。\n\n"
        "每个窗口的三项见 stage0_h2_sinks.csv。\n"
    )
    protocol = {
        "dataset_id": DATASET_ID,
        "stage0_n_cells": STAGE0_N_CELLS,
        "stage0_criterion": "r_H2_Ox + 2*r_C_H2 + r_RWGS > avail_H2 + 1e-6",
        "stage0_min_windows": STAGE0_MIN_WINDOWS,
        "stage0_n_exceed": n_exceed,
        "stage0_passed": False,
        "code_changed": False,
        "selected_N": None,
        "source_commit": commit,
        "handoff_model_head": HANDOFF_MODEL_HEAD,
        "solver_modules_match_handoff_head": commit.startswith(HANDOFF_MODEL_HEAD),
    }
    main_model = {"selected_N": None, "rule": "no grid is named when stage 0 does not pass 7/9"}
    protocol_path = output_dir / "protocol.json"
    interpretation_path = output_dir / "interpretation.md"
    main_path = output_dir / "main_model.json"
    protocol_text = json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    main_text = json.dumps(main_model, indent=2, sort_keys=True) + "\n"
    assert_no_local_routing(protocol_text)
    assert_no_local_routing(interpretation)
    assert_no_local_routing(main_text)
    protocol_path.write_text(protocol_text, encoding="utf-8")
    interpretation_path.write_text(interpretation, encoding="utf-8")
    main_path.write_text(main_text, encoding="utf-8")
    write_sha256_manifest(
        [result["path"], protocol_path, interpretation_path, main_path],
        output_dir / "sha256.txt",
    )


H2_INDEX = SPECIES_NAMES.index("H2")


def _cell_records(row: Mapping[str, Any], solved: Mapping[str, Any], n_cells: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    system = solved["system"]
    profile = solved["profile"]
    _closure, _projection, identity_gap, _flags = _identity_from_solve(solved)
    cell_rows: list[dict[str, Any]] = []
    term_rows: list[dict[str, Any]] = []
    for cell, state_row in zip(system.cells, profile):
        state = np.asarray(state_row, dtype=float)
        components = cell.residual_components(state)
        evaluated = evaluate_cell_state(cell, state)
        names = list(components["names"])
        cost = float(components["cost"])
        avail_h2 = max(float(cell.inlet.gas_moles[H2_INDEX]) + float(evaluated["gas_src"][H2_INDEX]), 0.0)
        cell_rows.append(
            {
                "dataset_id": DATASET_ID,
                "window_id": row["window_id"],
                "phase": int(row["phase_number"]),
                "split_role": row["split_role"],
                "N_cells": int(n_cells),
                "cell_index": int(cell.idx),
                "residual_cost": cost,
                "exceeds_1e-4": bool(cost > POOR_CONVERGENCE_COST_THRESHOLD),
                "is_combustion": bool(evaluated["is_combustion"]),
                "H2_residual_mol_s": float(components["unscaled"][names.index("gas_H2")]),
                "element_species_identity_max_abs_mol_s": float(identity_gap),
            }
        )
        if cell.idx != 0:
            continue
        gas_unscaled = np.asarray(components["unscaled"][:8], dtype=float)
        reconstructed = evaluated["convection"] - evaluated["gas_src"] - evaluated["reaction"]
        term_rows.append(
            {
                "dataset_id": DATASET_ID,
                "window_id": row["window_id"],
                "phase": int(row["phase_number"]),
                "split_role": row["split_role"],
                "N_cells": int(n_cells),
                "cell_index": 0,
                "avail_H2_mol_s": avail_h2,
                "convection_inout_mol_s": float(evaluated["convection"][H2_INDEX]),
                "reaction_source_mol_s": float(evaluated["reaction"][H2_INDEX]),
                "inlet_source_mol_s": float(evaluated["gas_src"][H2_INDEX]),
                "residual_mol_s": float(gas_unscaled[H2_INDEX]),
                "term_sum_minus_residual_mol_s": float(reconstructed[H2_INDEX] - gas_unscaled[H2_INDEX]),
                "r_H2_Ox_mol_s": float(evaluated["r_homo"].get("H2_Ox", 0.0)),
                "r_C_H2_mol_s": float(evaluated["r_het"].get("C+H2", 0.0)),
                "r_RWGS_mol_s": float(evaluated["r_homo"].get("RWGS", 0.0)),
                "r_WGS_mol_s": float(evaluated["r_homo"].get("WGS", 0.0)),
                "r_C_H2O_mol_s": float(evaluated["r_het"].get("C+H2O", 0.0)),
                "r_MSR_mol_s": float(evaluated["r_homo"].get("MSR", 0.0)),
                "residual_cost": cost,
                "exceeds_1e-4": bool(cost > POOR_CONVERGENCE_COST_THRESHOLD),
                "element_species_identity_max_abs_mol_s": float(identity_gap),
            }
        )
    return cell_rows, term_rows, float(identity_gap)


def run_modified_n3(rows: Sequence[Mapping[str, Any]], window_ids: Sequence[str], output_dir: Path) -> dict[str, Any]:
    """修改后的 9 窗 N=3。任一单元超过 1e-4 或元素恒等式超过 1e-2 就停止，不跑更细网格。"""
    silence_solver_logs()
    output_dir.mkdir(parents=True, exist_ok=True)
    stage0_path = output_dir / "stage0_h2_sinks.csv"
    if not stage0_path.exists():
        raise RuntimeError("stage0_h2_sinks.csv is missing; refuse to recreate it in this stage")
    by_id = {str(row["window_id"]): row for row in rows}
    selected = [by_id[str(window_id)] for window_id in window_ids]
    cell_rows: list[dict[str, Any]] = []
    term_rows: list[dict[str, Any]] = []
    identity_max = 0.0
    for index, row in enumerate(selected, start=1):
        print(f"P0-main N=3 {index}/{len(selected)} {row['window_id']}", flush=True)
        solved = solve_reference_case(row, n_cells=3)
        part_cells, part_terms, identity_gap = _cell_records(row, solved, 3)
        cell_rows.extend(part_cells)
        term_rows.extend(part_terms)
        identity_max = max(identity_max, identity_gap)
        if identity_gap > ELEMENT_IDENTITY_STOP_MOL_S:
            break
    cell_frame = pd.DataFrame(cell_rows)
    term_frame = pd.DataFrame(term_rows)
    n_exceed = int(cell_frame["exceeds_1e-4"].astype(bool).sum())
    identity_failed = bool(identity_max > ELEMENT_IDENTITY_STOP_MOL_S)
    residual_passed = n_exceed == 0 and len(cell_frame) == 9 * 3
    passed = residual_passed and not identity_failed
    cell_path = output_dir / "stage2_cell_residual.csv"
    term_path = output_dir / "stage2_cell0_h2_terms.csv"
    decision_path = output_dir / "stage2_decision.json"
    protocol_path = output_dir / "protocol.json"
    interpretation_path = output_dir / "interpretation.md"
    main_path = output_dir / "main_model.json"
    cell_frame.to_csv(cell_path, index=False)
    term_frame.to_csv(term_path, index=False)
    commit = source_commit()
    if identity_failed:
        reason = "element identity exceeded 1e-2 mol/s"
    elif not residual_passed:
        reason = "N=3 has cells with residual_cost above 1e-4"
    else:
        reason = "N=3 passed"
    decision = {
        "grids_ran": [3],
        "N3_passed": passed,
        "identity_max_abs_mol_s": identity_max,
        "n_cells_above_1e-4": n_exceed,
        "finer_grids_ran": False,
        "stopped_at": None if passed else "N=3",
        "stop_reason": reason,
        "revert_limit_change": identity_failed,
        "selected_N": None,
    }
    protocol = {
        "dataset_id": DATASET_ID,
        "stage0_csv_preserved": True,
        "previous_rwgs_stage1_executed": False,
        "change": "after H2_Ox, cap C+H2 and negative net WGS on remaining H2; positive WGS is not capped",
        "jax_change": "same negative WGS cap using avail[5] - r_H2_ox, then C+H2; no new RWGS; S_release unchanged",
        "stoichiometry_difference": "numpy H2 balance has no S_release; JAX dH2 includes S_release",
        "unchanged": [
            "H2_OxidationFactor",
            "pyrolysis yields",
            "phi",
            "f_co2",
            "P_O2_Combustion_atm=0.05",
        ],
        "acceptance": "every cell residual_cost <= 1e-4 and element identity <= 1e-2 mol/s",
        "N3_passed": passed,
        "identity_max_abs_mol_s": identity_max,
        "n_cells_above_1e-4": n_exceed,
        "selected_N": None,
        "source_commit": commit,
        "handoff_model_head": HANDOFF_MODEL_HEAD,
        "solver_modules_match_handoff_head": commit.startswith(HANDOFF_MODEL_HEAD),
    }
    poor = cell_frame[cell_frame["exceeds_1e-4"].astype(bool)]
    poor_cells = []
    if len(poor):
        for cell_index, group in poor.groupby("cell_index"):
            poor_cells.append(
                f"cell {int(cell_index)} 有 {int(group['window_id'].nunique())} 个窗口，cost {float(group['residual_cost'].min()):.6g}–{float(group['residual_cost'].max()):.6g}"
            )
    if passed:
        interpretation = (
            "# P0 主模型 N=3\n\n"
            "9 个窗口的每个单元 residual_cost 都不超过 1e-4。"
            f"元素恒等式最大绝对偏差 {identity_max:.6g} mol/s。\n"
        )
    else:
        listed = "；".join(poor_cells) if poor_cells else "没有单元超过 1e-4"
        interpretation = (
            "# P0 主模型 N=3 停止\n\n"
            f"{reason}。{listed}。元素恒等式最大绝对偏差 {identity_max:.6g} mol/s。"
            "没有跑 N=5、N=9 和 142 窗，没有加第二种修改。selected_N 为 null。\n\n"
            "cell 0 的 H2 三项和七个速率见 stage2_cell0_h2_terms.csv。\n"
        )
    main_model = {
        "selected_N": None,
        "rule": "coarsest of {3, 5, 9} whose 142 windows all have every cell residual_cost <= 1e-4; null when no such grid",
    }
    decision_text = json.dumps(decision, indent=2, sort_keys=True) + "\n"
    protocol_text = json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    main_text = json.dumps(main_model, indent=2, sort_keys=True) + "\n"
    assert_no_local_routing(decision_text)
    assert_no_local_routing(protocol_text)
    assert_no_local_routing(interpretation)
    assert_no_local_routing(main_text)
    decision_path.write_text(decision_text, encoding="utf-8")
    protocol_path.write_text(protocol_text, encoding="utf-8")
    interpretation_path.write_text(interpretation, encoding="utf-8")
    main_path.write_text(main_text, encoding="utf-8")
    write_sha256_manifest(
        [stage0_path, cell_path, term_path, decision_path, protocol_path, interpretation_path, main_path],
        output_dir / "sha256.txt",
    )
    return {
        "passed": passed,
        "identity_failed": identity_failed,
        "identity_max_abs_mol_s": identity_max,
        "n_cells_above_1e-4": n_exceed,
    }


def _solve_representative_grid(
    selected: Sequence[Mapping[str, Any]],
    n_cells: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float, int]:
    cell_rows: list[dict[str, Any]] = []
    term_rows: list[dict[str, Any]] = []
    identity_max = 0.0
    for index, row in enumerate(selected, start=1):
        print(f"P0-main N={n_cells} {index}/{len(selected)} {row['window_id']}", flush=True)
        solved = solve_reference_case(row, n_cells=n_cells)
        part_cells, part_terms, identity_gap = _cell_records(row, solved, n_cells)
        cell_rows.extend(part_cells)
        term_rows.extend(part_terms)
        identity_max = max(identity_max, identity_gap)
        if identity_gap > ELEMENT_IDENTITY_STOP_MOL_S:
            break
    n_exceed = int(sum(bool(item["exceeds_1e-4"]) for item in cell_rows))
    return cell_rows, term_rows, identity_max, n_exceed


def _effect_rows(status: pd.DataFrame, commit: str) -> list[dict[str, Any]]:
    targets = (
        ("CO_dry_vol_pct", "pred_CO_dry_vol_pct", "plant_CO_dry_vol_pct"),
        ("CO2_dry_vol_pct", "pred_CO2_dry_vol_pct", "plant_CO2_dry_vol_pct"),
        ("H2_dry_vol_pct", "pred_H2_dry_vol_pct", "plant_H2_dry_vol_pct"),
        ("dry_syngas_flow", "pred_dry_flow_nm3_h", "plant_dry_flow_nm3_h"),
    )
    rows = []
    test = status[status["split_role"] == "test"]
    for n_cells, group in test.groupby("N_cells"):
        for target, pred_col, plant_col in targets:
            residual = group[pred_col].astype(float).to_numpy() - group[plant_col].astype(float).to_numpy()
            residual = residual[np.isfinite(residual)]
            ratio = group["pred_dry_flow_nm3_h"].astype(float) / group["plant_dry_flow_nm3_h"].astype(float)
            rows.append(
                {
                    "dataset_id": DATASET_ID,
                    "split_role": "test",
                    "N_cells": int(n_cells),
                    "target": target,
                    "rmse": float(np.sqrt(np.mean(residual * residual))) if residual.size else float("nan"),
                    "mae": float(np.mean(np.abs(residual))) if residual.size else float("nan"),
                    "bias": float(np.mean(residual)) if residual.size else float("nan"),
                    "n_completed": int(residual.size),
                    "mean_pred_dry_over_plant_dry": float(ratio.mean()) if len(ratio) else float("nan"),
                    "source_commit": commit,
                }
            )
    return rows


def run_finer_grids_and_full_split(rows: Sequence[Mapping[str, Any]], window_ids: Sequence[str], output_dir: Path) -> dict[str, Any]:
    """N=3 的 9 窗已通过。继续 N=5、N=9，再对通过的网格跑 142 窗。"""
    silence_solver_logs()
    stage0_path = output_dir / "stage0_h2_sinks.csv"
    if not stage0_path.exists():
        raise RuntimeError("stage0_h2_sinks.csv is missing")
    by_id = {str(row["window_id"]): row for row in rows}
    selected = [by_id[str(window_id)] for window_id in window_ids]
    cell_frame = pd.read_csv(output_dir / "stage2_cell_residual.csv")
    term_frame = pd.read_csv(output_dir / "stage2_cell0_h2_terms.csv")
    if set(cell_frame["N_cells"].astype(int)) != {3}:
        raise RuntimeError("stage2 residual file is not the N=3 result")
    passed_grids = [3]
    identity_max = float(cell_frame["element_species_identity_max_abs_mol_s"].max())
    identity_failed = False
    failed_grid = None
    for n_cells in (5, 9):
        part_cells, part_terms, grid_identity, n_exceed = _solve_representative_grid(selected, n_cells)
        cell_frame = pd.concat([cell_frame, pd.DataFrame(part_cells)], ignore_index=True)
        term_frame = pd.concat([term_frame, pd.DataFrame(part_terms)], ignore_index=True)
        identity_max = max(identity_max, grid_identity)
        expected = len(selected) * n_cells
        grid_ok = n_exceed == 0 and len(part_cells) == expected and grid_identity <= ELEMENT_IDENTITY_STOP_MOL_S
        if grid_identity > ELEMENT_IDENTITY_STOP_MOL_S:
            identity_failed = True
            failed_grid = n_cells
            break
        if not grid_ok:
            failed_grid = n_cells
            break
        passed_grids.append(n_cells)
    full_status: list[dict[str, Any]] = []
    full_pass: list[int] = []
    if not identity_failed:
        for n_cells in passed_grids:
            grid_identity = 0.0
            grid_above = 0
            completed = 0
            for index, row in enumerate(rows, start=1):
                print(f"P0-main full N={n_cells} {index}/{len(rows)} {row['window_id']}", flush=True)
                solved = solve_reference_case(row, n_cells=n_cells)
                part_cells, _part_terms, identity_gap = _cell_records(row, solved, n_cells)
                costs = [float(item["residual_cost"]) for item in part_cells]
                above = int(sum(bool(item["exceeds_1e-4"]) for item in part_cells))
                grid_above += above
                grid_identity = max(grid_identity, identity_gap)
                identity_max = max(identity_max, identity_gap)
                kpis = _dry_kpis(solved["profile"])
                full_status.append(
                    {
                        "dataset_id": DATASET_ID,
                        "window_id": row["window_id"],
                        "split_role": row["split_role"],
                        "phase": int(row["phase_number"]),
                        "N_cells": int(n_cells),
                        "n_cells_above_1e-4": above,
                        "max_residual_cost": max(costs) if costs else float("nan"),
                        "element_species_identity_max_abs_mol_s": identity_gap,
                        "pred_CO_dry_vol_pct": kpis["pred_CO_dry_vol_pct"],
                        "pred_CO2_dry_vol_pct": kpis["pred_CO2_dry_vol_pct"],
                        "pred_H2_dry_vol_pct": kpis["pred_H2_dry_vol_pct"],
                        "pred_dry_flow_nm3_h": kpis["pred_dry_flow_nm3_h"],
                        "plant_CO_dry_vol_pct": float(row["co_vol_pct"]),
                        "plant_CO2_dry_vol_pct": float(row["co2_vol_pct"]),
                        "plant_H2_dry_vol_pct": float(row["h2_vol_pct"]),
                        "plant_dry_flow_nm3_h": float(row["syngas_flow_mean"]),
                    }
                )
                completed += 1
                if identity_gap > ELEMENT_IDENTITY_STOP_MOL_S:
                    identity_failed = True
                    failed_grid = n_cells
                    break
            if identity_failed:
                break
            if grid_above == 0 and completed == len(rows) and grid_identity <= ELEMENT_IDENTITY_STOP_MOL_S:
                full_pass.append(n_cells)
    selected_n = min(full_pass) if full_pass else None
    commit = source_commit()
    cell_path = output_dir / "stage2_cell_residual.csv"
    term_path = output_dir / "stage2_cell0_h2_terms.csv"
    decision_path = output_dir / "stage2_decision.json"
    status_path = output_dir / "stage3_window_status.csv"
    effect_path = output_dir / "stage3_effect_matrix.csv"
    protocol_path = output_dir / "protocol.json"
    interpretation_path = output_dir / "interpretation.md"
    main_path = output_dir / "main_model.json"
    cell_frame.to_csv(cell_path, index=False)
    term_frame.to_csv(term_path, index=False)
    status_frame = pd.DataFrame(full_status)
    if len(status_frame):
        status_frame.to_csv(status_path, index=False)
        effect_source = status_frame[status_frame["N_cells"].astype(int).isin(full_pass)]
        if len(effect_source):
            effect_frame = pd.DataFrame(_effect_rows(effect_source, commit))
            effect_frame.to_csv(effect_path, index=False)
        else:
            effect_path = None
    decision = {
        "representative_passed_grids": passed_grids,
        "full_split_passed_grids": full_pass,
        "failed_grid": failed_grid,
        "identity_failed": identity_failed,
        "identity_max_abs_mol_s": identity_max,
        "revert_limit_change": identity_failed,
        "selected_N": selected_n,
    }
    protocol = {
        "dataset_id": DATASET_ID,
        "stage0_csv_preserved": True,
        "previous_rwgs_stage1_executed": False,
        "change": "after H2_Ox, cap C+H2 and negative net WGS on remaining H2; positive WGS is not capped",
        "jax_change": "same negative WGS cap using avail[5] - r_H2_ox, then C+H2; no new RWGS; S_release unchanged",
        "stoichiometry_difference": "numpy H2 balance has no S_release; JAX dH2 includes S_release",
        "unchanged": [
            "H2_OxidationFactor",
            "pyrolysis yields",
            "phi",
            "f_co2",
            "P_O2_Combustion_atm=0.05",
        ],
        "acceptance": "every cell residual_cost <= 1e-4 and element identity <= 1e-2 mol/s",
        "representative_passed_grids": passed_grids,
        "full_split_passed_grids": full_pass,
        "selected_N": selected_n,
        "identity_max_abs_mol_s": identity_max,
        "source_commit": commit,
        "handoff_model_head": HANDOFF_MODEL_HEAD,
        "solver_modules_match_handoff_head": commit.startswith(HANDOFF_MODEL_HEAD),
    }
    main_model = {
        "selected_N": selected_n,
        "rule": "coarsest of {3, 5, 9} whose 142 windows all have every cell residual_cost <= 1e-4; null when no such grid",
    }
    lines = ["# P0 主模型", ""]
    lines.append(f"9 窗通过的网格：{passed_grids}。142 窗全部单元不超过 1e-4 的网格：{full_pass or '无'}。")
    lines.append(f"selected_N = {selected_n if selected_n is not None else 'null'}。")
    lines.append(f"元素恒等式最大绝对偏差 {identity_max:.6g} mol/s。")
    if len(status_frame) and full_pass:
        test = status_frame[(status_frame["split_role"] == "test") & (status_frame["N_cells"].astype(int).isin(full_pass))]
        lines.append("")
        lines.append("| N | CO RMSE | CO2 RMSE | H2 RMSE | 干气 RMSE | 预测干气/装置干气 |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for n_cells, group in test.groupby("N_cells"):
            def _rmse(pred, plant):
                residual = group[pred].astype(float).to_numpy() - group[plant].astype(float).to_numpy()
                return float(np.sqrt(np.mean(residual * residual)))
            ratio = float((group["pred_dry_flow_nm3_h"] / group["plant_dry_flow_nm3_h"]).mean())
            lines.append(
                "| {n} | {co:.6g} | {co2:.6g} | {h2:.6g} | {dry:.6g} | {ratio:.6g} |".format(
                    n=int(n_cells),
                    co=_rmse("pred_CO_dry_vol_pct", "plant_CO_dry_vol_pct"),
                    co2=_rmse("pred_CO2_dry_vol_pct", "plant_CO2_dry_vol_pct"),
                    h2=_rmse("pred_H2_dry_vol_pct", "plant_H2_dry_vol_pct"),
                    dry=_rmse("pred_dry_flow_nm3_h", "plant_dry_flow_nm3_h"),
                    ratio=ratio,
                )
            )
    interpretation = "\n".join(lines) + "\n"
    texts = {
        decision_path: json.dumps(decision, indent=2, sort_keys=True) + "\n",
        protocol_path: json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        main_path: json.dumps(main_model, indent=2, sort_keys=True) + "\n",
    }
    for text in list(texts.values()) + [interpretation]:
        assert_no_local_routing(text)
    for path, text in texts.items():
        path.write_text(text, encoding="utf-8")
    interpretation_path.write_text(interpretation, encoding="utf-8")
    manifest = [stage0_path, cell_path, term_path, decision_path, protocol_path, interpretation_path, main_path]
    if len(status_frame):
        manifest.append(status_path)
        if effect_path is not None and effect_path.exists():
            manifest.append(effect_path)
    write_sha256_manifest(manifest, output_dir / "sha256.txt")
    return {
        "passed_grids": passed_grids,
        "full_pass": full_pass,
        "selected_N": selected_n,
        "identity_failed": identity_failed,
        "identity_max_abs_mol_s": identity_max,
    }
