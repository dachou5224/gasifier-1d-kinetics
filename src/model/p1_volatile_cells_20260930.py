"""P1：把挥发分通道的基准解和 ×1.10 解拆到每个单元。不覆盖上一轮通道响应。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .alpha_candidate_identifiability import inject_alpha
from .p0_followup_20260930 import _family_rates, _identity_from_solve, rate_branch_criterion
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
from .p1_response_20260930 import WINDOWS

SELECTED_N = 3
GRIDS = (3, 5, 9)
VOLATILE_PLUS = 1.10
DRY_NM3_ATOL = 0.1
RATE_DELTA_ATOL_MOL_S = 1.0e-4
FAMILIES = ("WGS", "volatile_oxidation", "char_oxidation", "char_gasification")
# 与 pred_dry_flow_nm3_h 相同：mol/s → Nm3/h。
NM3_PER_MOL_S = 22.414 * 3600.0 / 1000.0


def dry_mol_s_to_nm3_h(dry_mol_s: float) -> float:
    return float(dry_mol_s) * NM3_PER_MOL_S


def first_sign_change(deltas: Sequence[float]) -> int | None:
    """沿单元顺序，干气差第一次从已有符号变成相反符号的 cell_index。全程不变号返回 None。"""
    established = 0
    for index, value in enumerate(deltas):
        current = 0 if abs(float(value)) <= 1.0e-12 else (1 if float(value) > 0.0 else -1)
        if current == 0:
            continue
        if established == 0:
            established = current
            continue
        if current != established:
            return int(index)
    return None


def alignment_errors(
    *,
    base_dry_nm3_h: float,
    plus_dry_nm3_h: float,
    previous_base_dry_nm3_h: float,
    previous_plus_dry_nm3_h: float,
    rate_deltas: Mapping[str, float],
    previous_rate_deltas: Mapping[str, float],
) -> dict[str, float]:
    errors = {
        "base_dry_nm3_h": abs(float(base_dry_nm3_h) - float(previous_base_dry_nm3_h)),
        "plus_dry_nm3_h": abs(float(plus_dry_nm3_h) - float(previous_plus_dry_nm3_h)),
    }
    for family in FAMILIES:
        errors[family] = abs(float(rate_deltas[family]) - float(previous_rate_deltas[family]))
    return errors


def alignment_ok(errors: Mapping[str, float]) -> bool:
    if float(errors["base_dry_nm3_h"]) > DRY_NM3_ATOL or float(errors["plus_dry_nm3_h"]) > DRY_NM3_ATOL:
        return False
    return all(float(errors[family]) <= RATE_DELTA_ATOL_MOL_S for family in FAMILIES)


def _cell_side(solved: Mapping[str, Any]) -> tuple[list[dict[str, Any]], float]:
    system = solved["system"]
    profile = solved["profile"]
    _closure, _projection, identity, _flags = _identity_from_solve(solved)
    rows = []
    for cell, state_row in zip(system.cells, profile):
        state = np.asarray(state_row, dtype=float)
        components = cell.residual_components(state)
        evaluated = evaluate_cell_state(cell, state)
        gas = state[:8]
        combustion = bool(rate_branch_criterion(cell, evaluated["current"], evaluated["gas_src"])["is_combustion"])
        families = _family_rates(evaluated["r_het"], evaluated["r_homo"])
        rows.append(
            {
                "cell_index": int(cell.idx),
                "is_combustion": combustion,
                "residual_cost": float(components["cost"]),
                "dry_mol_s": float(np.sum(gas[:7])),
                "CO_mol_s": float(gas[2]),
                "CO2_mol_s": float(gas[3]),
                "H2_mol_s": float(gas[5]),
                **families,
            }
        )
    return rows, float(identity)


def _paired_rows(window: Mapping[str, Any], n_cells: int, base_rows: Sequence[Mapping[str, Any]], plus_rows: Sequence[Mapping[str, Any]], base_identity: float, plus_identity: float) -> list[dict[str, Any]]:
    paired = []
    for base, plus in zip(base_rows, plus_rows):
        record = {
            "dataset_id": DATASET_ID,
            "window_id": window["window_id"],
            "phase": int(window["phase"]),
            "split_role": window["split_role"],
            "N_cells": int(n_cells),
            "cell_index": int(base["cell_index"]),
            "is_combustion": bool(base["is_combustion"]),
            "plus_is_combustion": bool(plus["is_combustion"]),
            "residual_cost": max(float(base["residual_cost"]), float(plus["residual_cost"])),
            "base_residual_cost": float(base["residual_cost"]),
            "plus_residual_cost": float(plus["residual_cost"]),
            "base_element_identity_max_abs_mol_s": float(base_identity),
            "plus_element_identity_max_abs_mol_s": float(plus_identity),
            "base_dry_mol_s": float(base["dry_mol_s"]),
            "plus_dry_mol_s": float(plus["dry_mol_s"]),
            "delta_dry_mol_s": float(plus["dry_mol_s"]) - float(base["dry_mol_s"]),
        }
        for species in ("CO", "CO2", "H2"):
            record[f"base_{species}_mol_s"] = float(base[f"{species}_mol_s"])
            record[f"plus_{species}_mol_s"] = float(plus[f"{species}_mol_s"])
            record[f"delta_{species}_mol_s"] = float(plus[f"{species}_mol_s"]) - float(base[f"{species}_mol_s"])
        for family in FAMILIES:
            record[f"base_{family}_mol_s"] = float(base[family])
            record[f"plus_{family}_mol_s"] = float(plus[family])
            record[f"delta_{family}_mol_s"] = float(plus[family]) - float(base[family])
        paired.append(record)
    return paired


def _previous_volatile(previous_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(previous_csv)
    volatile = frame[frame["alpha_name"] == "alpha_volatile_oxidation"].copy()
    if len(volatile) != 12:
        raise RuntimeError(f"expected 12 previous volatile rows, found {len(volatile)}")
    return volatile


def _previous_match(previous: pd.DataFrame, window_id: str, n_cells: int) -> pd.Series:
    part = previous[(previous["window_id"] == window_id) & (previous["N_cells"].astype(int) == int(n_cells))]
    if len(part) != 1:
        raise RuntimeError(f"expected one previous volatile row for N={n_cells} {window_id}")
    return part.iloc[0]


def _fmt(value: float) -> str:
    return f"{float(value):.6g}"


def render_stop(reason: str) -> str:
    return f"# P1 挥发分单元分解停止\n\n{reason}\n\n没有解释干气差值的符号。selected_N 仍是 3。\n"


def render_interpretation(frame: pd.DataFrame) -> str:
    lines = [
        "# P1 挥发分单元分解",
        "",
        "selected_N 仍是 3。末单元干气和四个积分速率差与上一轮挥发分通道对齐。",
        "",
    ]
    for window_id, group in frame.groupby("window_id", sort=False):
        lines.append(f"## {window_id}")
        lines.append("")
        for n_cells in (3, 9):
            part = group[group["N_cells"].astype(int) == n_cells].sort_values("cell_index")
            outlet = part.iloc[-1]
            outlet_nm3 = dry_mol_s_to_nm3_h(float(outlet["delta_dry_mol_s"]))
            change = first_sign_change(part["delta_dry_mol_s"].astype(float).tolist())
            position = int(np.argmax(np.abs(part["delta_char_gasification_mol_s"].to_numpy(dtype=float))))
            peak = part.iloc[position]
            lines.append(
                f"- N={n_cells} 出口干气差 {_fmt(outlet_nm3)} Nm3/h。"
                f"干气差第一次变号的 cell_index：{change if change is not None else 'none'}。"
                f"焦炭气化速率差绝对值最大在 cell {int(peak['cell_index'])}，"
                f"基准解燃烧区={bool(peak['is_combustion'])}，扰动解燃烧区={bool(peak['plus_is_combustion'])}。"
            )
        lines.append("")
    return "\n".join(lines)


def _protocol(commit: str, *, stopped: bool, stop_reason: str, n_rows: int) -> dict[str, Any]:
    return {
        "dataset_id": DATASET_ID,
        "selected_N": SELECTED_N,
        "selection_unchanged": True,
        "grids": list(GRIDS),
        "windows": list(WINDOWS),
        "channel": "alpha_volatile_oxidation",
        "plus_factor": VOLATILE_PLUS,
        "injection": "perturbed solve sets CO_OxidationFactor and H2_OxidationFactor to 1.10; baseline factors stay at their current values",
        "dry_nm3_conversion": "mol/s / 1000 * 22.414 * 3600, same as pred_dry_flow_nm3_h",
        "alignment_dry_atol_nm3_h": DRY_NM3_ATOL,
        "alignment_rate_delta_atol_mol_s": RATE_DELTA_ATOL_MOL_S,
        "residual_cost_threshold": POOR_CONVERGENCE_COST_THRESHOLD,
        "element_identity_stop_mol_s": ELEMENT_IDENTITY_STOP_MOL_S,
        "unchanged": [
            "H2 sequential limit",
            "baseline H2_OxidationFactor",
            "baseline CO_OxidationFactor",
            "pyrolysis yields",
            "phi",
            "f_co2",
            "P_O2_Combustion_atm=0.05",
        ],
        "full_split_rerun": False,
        "n_rows": n_rows,
        "stopped": stopped,
        "stop_reason": stop_reason,
        "source_commit": commit,
        "handoff_model_head": HANDOFF_MODEL_HEAD,
        "solver_modules_match_handoff_head": commit.startswith(HANDOFF_MODEL_HEAD),
    }


def _write(output_dir: Path, frame: pd.DataFrame, protocol: dict[str, Any], interpretation: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cell_path = output_dir / "cell_rates.csv"
    protocol_path = output_dir / "protocol.json"
    interpretation_path = output_dir / "interpretation.md"
    frame.to_csv(cell_path, index=False)
    protocol_text = json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    assert_no_local_routing(protocol_text)
    assert_no_local_routing(interpretation)
    protocol_path.write_text(protocol_text, encoding="utf-8")
    interpretation_path.write_text(interpretation, encoding="utf-8")
    write_sha256_manifest([cell_path, protocol_path, interpretation_path], output_dir / "sha256.txt")


def run_volatile_cells(rows: Sequence[Mapping[str, Any]], previous_csv: Path, output_dir: Path) -> dict[str, Any]:
    silence_solver_logs()
    output_dir.mkdir(parents=True, exist_ok=True)
    by_id = {str(row["window_id"]): row for row in rows}
    missing = [item["window_id"] for item in WINDOWS if item["window_id"] not in by_id]
    if missing:
        raise RuntimeError(f"phase-median windows missing from the frozen split: {missing}")
    previous = _previous_volatile(previous_csv)
    records: list[dict[str, Any]] = []
    commit = source_commit()
    stopped = False
    stop_reason = ""
    for n_cells in GRIDS:
        for window in WINDOWS:
            label = f"N={n_cells} {window['window_id']}"
            print(f"P1-volatile baseline {label}", flush=True)
            base_solved = solve_reference_case(by_id[window["window_id"]], n_cells=n_cells)
            base_rows, base_identity = _cell_side(base_solved)
            print(f"P1-volatile plus {label}", flush=True)
            plus_solved = solve_reference_case(
                by_id[window["window_id"]],
                n_cells=n_cells,
                extra_op_conds=inject_alpha({}, "alpha_volatile_oxidation", VOLATILE_PLUS),
            )
            plus_rows, plus_identity = _cell_side(plus_solved)
            paired = _paired_rows(window, n_cells, base_rows, plus_rows, base_identity, plus_identity)
            records.extend(paired)
            poor = [
                item["cell_index"]
                for item in paired
                if float(item["base_residual_cost"]) > POOR_CONVERGENCE_COST_THRESHOLD
                or float(item["plus_residual_cost"]) > POOR_CONVERGENCE_COST_THRESHOLD
            ]
            if poor or base_identity > ELEMENT_IDENTITY_STOP_MOL_S or plus_identity > ELEMENT_IDENTITY_STOP_MOL_S:
                stopped = True
                stop_reason = (
                    f"{label} residual cells {poor}, "
                    f"base identity {base_identity:.6g} mol/s, plus identity {plus_identity:.6g} mol/s."
                )
                break
            prior = _previous_match(previous, window["window_id"], n_cells)
            base_dry_nm3 = dry_mol_s_to_nm3_h(float(paired[-1]["base_dry_mol_s"]))
            plus_dry_nm3 = dry_mol_s_to_nm3_h(float(paired[-1]["plus_dry_mol_s"]))
            rate_deltas = {
                family: float(sum(item[f"delta_{family}_mol_s"] for item in paired))
                for family in FAMILIES
            }
            previous_deltas = {
                family: float(prior[f"plus_{family}_mol_s"]) - float(prior[f"base_{family}_mol_s"])
                for family in FAMILIES
            }
            errors = alignment_errors(
                base_dry_nm3_h=base_dry_nm3,
                plus_dry_nm3_h=plus_dry_nm3,
                previous_base_dry_nm3_h=float(prior["base_dry_flow"]),
                previous_plus_dry_nm3_h=float(prior["plus_dry_flow"]),
                rate_deltas=rate_deltas,
                previous_rate_deltas=previous_deltas,
            )
            if not alignment_ok(errors):
                stopped = True
                detail = ", ".join(f"{key}={value:.6g}" for key, value in errors.items())
                stop_reason = f"{label} does not match the previous volatile channel: {detail}."
                break
        if stopped:
            break
    frame = pd.DataFrame(records)
    if stopped:
        interpretation = render_stop(stop_reason)
    else:
        if len(frame) != 68:
            raise RuntimeError(f"expected 68 cell rows, found {len(frame)}")
        interpretation = render_interpretation(frame)
    protocol = _protocol(commit, stopped=stopped, stop_reason=stop_reason, n_rows=len(frame))
    _write(output_dir, frame, protocol, interpretation)
    return {"stopped": stopped, "n_rows": len(frame), "stop_reason": stop_reason}
