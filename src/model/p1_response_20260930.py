"""P1：在已有根的主模型上重算 P0-B 的单向通道响应。不改限幅，不改 selected_N。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .alpha_candidate_identifiability import inject_alpha
from .p0_followup_20260930 import _dry_kpis, _family_rates, _identity_from_solve
from .p0_round_20260930 import (
    DATASET_ID,
    HANDOFF_MODEL_HEAD,
    POOR_CONVERGENCE_COST_THRESHOLD,
    assert_no_local_routing,
    evaluate_cell_state,
    silence_solver_logs,
    solve_reference_case,
    source_commit,
    write_sha256_manifest,
)
from .p0d_20260930 import ELEMENT_IDENTITY_STOP_MOL_S

SELECTED_N = 3
GRIDS = (3, 5, 9)
CHANNELS = (
    ("alpha_wgs", 1.0, 0.03),
    ("alpha_char_oxidation", 0.3, 0.03),
    ("alpha_volatile_oxidation", 1.0, 0.10),
)
WINDOWS = (
    {
        "window_id": "R52-C-75__P1__huayi_sy3_phase1_wgs1__plateau_2022-09-28D_008__W003",
        "phase": 1,
        "split_role": "train",
    },
    {
        "window_id": "R52-C-75__P2__huayi_sy3sy2_phase2_wgs1__plateau_2022-10-03N_013__W001",
        "phase": 2,
        "split_role": "train",
    },
    {
        "window_id": "R52-C-75__P3__huayi_sy3sy2_phase3_wgs1__plateau_2022-10-09D_014__W004",
        "phase": 3,
        "split_role": "validation",
    },
    {
        "window_id": "R52-C-75__P4__huayi_sy3sy2_phase4_wgs1__plateau_2022-10-11N_009__W002",
        "phase": 4,
        "split_role": "test",
    },
)


def quality_label(base_n_above: int, plus_n_above: int, base_identity: float, plus_identity: float) -> str:
    solved = (
        int(base_n_above) == 0
        and int(plus_n_above) == 0
        and float(base_identity) <= ELEMENT_IDENTITY_STOP_MOL_S
        and float(plus_identity) <= ELEMENT_IDENTITY_STOP_MOL_S
    )
    return "interpretable_solved" if solved else "not_interpretable_due_to_residual"


def same_sign(left: float, right: float) -> bool:
    return float(left) * float(right) > 0.0


def summarize_solve(solved: Mapping[str, Any]) -> dict[str, Any]:
    system = solved["system"]
    profile = solved["profile"]
    _closure, _projection, identity, _flags = _identity_from_solve(solved)
    costs = []
    families = {"WGS": 0.0, "char_oxidation": 0.0, "volatile_oxidation": 0.0, "char_gasification": 0.0}
    for cell, state_row in zip(system.cells, profile):
        components = cell.residual_components(np.asarray(state_row, dtype=float))
        costs.append(float(components["cost"]))
        evaluated = evaluate_cell_state(cell, np.asarray(state_row, dtype=float))
        for key, value in _family_rates(evaluated["r_het"], evaluated["r_homo"]).items():
            families[key] += value
    kpis = _dry_kpis(profile)
    n_above = int(sum(cost > POOR_CONVERGENCE_COST_THRESHOLD for cost in costs))
    return {
        "max_residual_cost": max(costs) if costs else float("nan"),
        "n_cells_above_1e-4": n_above,
        "element_identity_max_abs_mol_s": float(identity),
        "kpis": kpis,
        "families": families,
        "acceptable": n_above == 0 and float(identity) <= ELEMENT_IDENTITY_STOP_MOL_S,
    }


def _response_row(window: Mapping[str, Any], n_cells: int, alpha_name: str, base_alpha: float, step: float, base: Mapping[str, Any], plus: Mapping[str, Any]) -> dict[str, Any]:
    metrics = (
        ("CO", "pred_CO_dry_vol_pct", "vol_pct"),
        ("CO2", "pred_CO2_dry_vol_pct", "vol_pct"),
        ("H2", "pred_H2_dry_vol_pct", "vol_pct"),
        ("dry_flow", "pred_dry_flow_nm3_h", "nm3_h"),
    )
    row = {
        "dataset_id": DATASET_ID,
        "window_id": window["window_id"],
        "phase": int(window["phase"]),
        "split_role": window["split_role"],
        "N_cells": int(n_cells),
        "selected_N": SELECTED_N,
        "alpha_name": alpha_name,
        "base_alpha": float(base_alpha),
        "perturbation": float(step),
        "plus_alpha": float(base_alpha) + float(step),
        "base_max_residual_cost": base["max_residual_cost"],
        "base_n_cells_above_1e-4": base["n_cells_above_1e-4"],
        "base_element_identity_max_abs_mol_s": base["element_identity_max_abs_mol_s"],
        "plus_max_residual_cost": plus["max_residual_cost"],
        "plus_n_cells_above_1e-4": plus["n_cells_above_1e-4"],
        "plus_element_identity_max_abs_mol_s": plus["element_identity_max_abs_mol_s"],
        "interpretability": quality_label(
            base["n_cells_above_1e-4"],
            plus["n_cells_above_1e-4"],
            base["element_identity_max_abs_mol_s"],
            plus["element_identity_max_abs_mol_s"],
        ),
    }
    for label, key, _unit in metrics:
        base_value = float(base["kpis"][key])
        plus_value = float(plus["kpis"][key])
        delta = plus_value - base_value
        row[f"base_{label}"] = base_value
        row[f"plus_{label}"] = plus_value
        row[f"delta_{label}"] = delta
        row[f"one_sided_d{label}_dalpha"] = delta / float(step)
    for family in ("WGS", "volatile_oxidation", "char_oxidation", "char_gasification"):
        row[f"base_{family}_mol_s"] = float(base["families"][family])
        row[f"plus_{family}_mol_s"] = float(plus["families"][family])
    return row


def _fmt(value: float) -> str:
    return f"{float(value):.6g}"


def render_interpretation(frame: pd.DataFrame) -> str:
    n_ok = int((frame["interpretability"] == "interpretable_solved").sum())
    lines = [
        "# P1 通道响应",
        "",
        f"36 行里 interpretable_solved 有 {n_ok} 行，not_interpretable_due_to_residual 有 {len(frame) - n_ok} 行。",
        "selected_N 仍是 3。N=5 和 N=9 只比较 CO 差值的符号。",
        "",
        "N=3 的差值范围：",
        "",
        "| 通道 | CO vol% | CO2 vol% | H2 vol% | 干气 Nm3/h |",
        "|---|---:|---:|---:|---:|",
    ]
    n3 = frame[frame["N_cells"].astype(int) == SELECTED_N]
    for alpha_name, _base, _step in CHANNELS:
        part = n3[n3["alpha_name"] == alpha_name]
        lines.append(
            "| {name} | {co} | {co2} | {h2} | {dry} |".format(
                name=alpha_name,
                co=f"{_fmt(part['delta_CO'].min())} 至 {_fmt(part['delta_CO'].max())}",
                co2=f"{_fmt(part['delta_CO2'].min())} 至 {_fmt(part['delta_CO2'].max())}",
                h2=f"{_fmt(part['delta_H2'].min())} 至 {_fmt(part['delta_H2'].max())}",
                dry=f"{_fmt(part['delta_dry_flow'].min())} 至 {_fmt(part['delta_dry_flow'].max())}",
            )
        )
    lines.extend(["", "相对 N=3，同一窗口同一通道的 CO 差值：", ""])
    flips = []
    same_5 = 0
    same_9 = 0
    pairs = 0
    for window_id, group in frame.groupby("window_id"):
        for alpha_name, _base, _step in CHANNELS:
            part = group[group["alpha_name"] == alpha_name]
            by_n = {int(record["N_cells"]): float(record["delta_CO"]) for _, record in part.iterrows()}
            pairs += 1
            sign_5 = same_sign(by_n[5], by_n[3])
            sign_9 = same_sign(by_n[9], by_n[3])
            same_5 += int(sign_5)
            same_9 += int(sign_9)
            if not sign_5 or not sign_9:
                flips.append(f"{window_id} / {alpha_name}：N=3 {_fmt(by_n[3])}，N=5 {_fmt(by_n[5])}，N=9 {_fmt(by_n[9])}")
    lines.append(f"N=5 与 N=3 同号 {same_5}/{pairs}。N=9 与 N=3 同号 {same_9}/{pairs}。")
    if flips:
        lines.append("不同号的行：")
        lines.extend(f"- {item}" for item in flips)
    else:
        lines.append("N=5 和 N=9 的 CO 差值都与 N=3 同号。")
    return "\n".join(lines) + "\n"


def _protocol(commit: str, *, stopped: bool, stop_reason: str) -> dict[str, Any]:
    return {
        "dataset_id": DATASET_ID,
        "selected_N": SELECTED_N,
        "selection_unchanged": True,
        "grids": list(GRIDS),
        "n5_n9_not_used_for_selection": True,
        "windows": list(WINDOWS),
        "channels": [
            {"name": name, "base": base, "perturbation": step, "plus": base + step}
            for name, base, step in CHANNELS
        ],
        "direction": "positive one-sided perturbation, one channel at a time",
        "volatile_injection": "alpha_volatile_oxidation sets CO_OxidationFactor and H2_OxidationFactor together on the perturbed solve only, matching the previous one-sided channel; baseline factors are unchanged",
        "residual_cost_threshold": POOR_CONVERGENCE_COST_THRESHOLD,
        "element_identity_stop_mol_s": ELEMENT_IDENTITY_STOP_MOL_S,
        "interpretability": "interpretable_solved only when both solves have every cell residual_cost <= 1e-4 and both element identities <= 1e-2 mol/s",
        "unchanged": [
            "H2 sequential limit",
            "H2_OxidationFactor baseline",
            "pyrolysis yields",
            "phi",
            "f_co2",
            "P_O2_Combustion_atm=0.05",
        ],
        "n1_rerun": False,
        "full_split_rerun": False,
        "perturbations_executed": not stopped,
        "stopped": stopped,
        "stop_reason": stop_reason,
        "source_commit": commit,
        "handoff_model_head": HANDOFF_MODEL_HEAD,
        "solver_modules_match_handoff_head": commit.startswith(HANDOFF_MODEL_HEAD),
    }


def _write(output_dir: Path, frame: pd.DataFrame | None, protocol: dict[str, Any], interpretation: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = output_dir / "protocol.json"
    interpretation_path = output_dir / "interpretation.md"
    protocol_text = json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    assert_no_local_routing(protocol_text)
    assert_no_local_routing(interpretation)
    protocol_path.write_text(protocol_text, encoding="utf-8")
    interpretation_path.write_text(interpretation, encoding="utf-8")
    paths = [protocol_path, interpretation_path]
    if frame is not None:
        response_path = output_dir / "channel_response.csv"
        frame.to_csv(response_path, index=False)
        paths.insert(0, response_path)
    write_sha256_manifest(paths, output_dir / "sha256.txt")


def run_p1_response(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> dict[str, Any]:
    silence_solver_logs()
    output_dir.mkdir(parents=True, exist_ok=True)
    by_id = {str(row["window_id"]): row for row in rows}
    missing = [item["window_id"] for item in WINDOWS if item["window_id"] not in by_id]
    if missing:
        raise RuntimeError(f"phase-median windows missing from the frozen split: {missing}")
    baselines: dict[tuple[str, int], dict[str, Any]] = {}
    for index, window in enumerate(WINDOWS, start=1):
        print(f"P1 baseline N=3 {index}/4 {window['window_id']}", flush=True)
        solved = solve_reference_case(by_id[window["window_id"]], n_cells=SELECTED_N)
        baselines[(window["window_id"], SELECTED_N)] = summarize_solve(solved)
    failed = [
        window["window_id"]
        for window in WINDOWS
        if not baselines[(window["window_id"], SELECTED_N)]["acceptable"]
    ]
    commit = source_commit()
    if failed:
        interpretation = (
            "# P1 通道响应停止\n\n"
            "N=3 基准解没有全部达到每个单元 residual_cost <= 1e-4 且元素恒等式 <= 1e-2 mol/s。"
            f"未通过的窗口：{', '.join(failed)}。没有做扰动。selected_N 仍是 3。\n"
        )
        protocol = _protocol(commit, stopped=True, stop_reason="N=3 baseline residual or element identity failed")
        _write(output_dir, None, protocol, interpretation)
        return {"stopped": True, "n_rows": 0, "n_interpretable": 0}
    records = []
    for n_cells in GRIDS:
        for window in WINDOWS:
            key = (window["window_id"], n_cells)
            if key not in baselines:
                print(f"P1 baseline N={n_cells} {window['window_id']}", flush=True)
                baselines[key] = summarize_solve(solve_reference_case(by_id[window["window_id"]], n_cells=n_cells))
            base = baselines[key]
            for alpha_name, base_alpha, step in CHANNELS:
                plus_alpha = base_alpha + step
                print(f"P1 N={n_cells} {alpha_name} {window['window_id']}", flush=True)
                solved = solve_reference_case(
                    by_id[window["window_id"]],
                    n_cells=n_cells,
                    extra_op_conds=inject_alpha({}, alpha_name, plus_alpha),
                )
                plus = summarize_solve(solved)
                records.append(_response_row(window, n_cells, alpha_name, base_alpha, step, base, plus))
    frame = pd.DataFrame(records)
    if len(frame) != 36:
        raise RuntimeError(f"expected 36 channel rows, found {len(frame)}")
    interpretation = render_interpretation(frame)
    protocol = _protocol(commit, stopped=False, stop_reason="")
    _write(output_dir, frame, protocol, interpretation)
    n_ok = int((frame["interpretability"] == "interpretable_solved").sum())
    return {"stopped": False, "n_rows": len(frame), "n_interpretable": n_ok}
