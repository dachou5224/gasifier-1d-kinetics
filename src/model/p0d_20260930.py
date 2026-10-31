"""P0-D：9 个代表窗口在 N=3、5、9 上的单元残差与干气缺口分解。

不改平衡系数、热解产率、phi、f_co2、H2_OxidationFactor 或燃烧阈值。
不重跑 N=1，不重跑 142 窗。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .model_case_adapter import normal_m3_h_to_kmol_s
from .p0_followup_20260930 import _identity_from_solve, rate_branch_criterion
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

ELEMENT_IDENTITY_STOP_MOL_S = 1.0e-2
DRY_GAP_ACCOUNTING_ATOL_MOL_S = 1.0e-6
O2_SHARE_CONTINUE_LIMIT = 0.01
EXPECTED_SOLVES = 27
GAP_SHARE_THRESHOLD = 0.7
GRIDS = (3, 5, 9)
# 干气 = O2+CH4+CO+CO2+H2S+H2+N2，与 pred_dry_flow_nm3_h 的 gas[:7] 相同。
# 其余干气只计 CH4、H2S、N2。缺口恒等式把预测出口 O2 加在四列之外。
OTHER_DRY_INDICES = (1, 4, 6)
SPECIES_INDEX = {name: index for index, name in enumerate(SPECIES_NAMES)}


def nm3_h_to_mol_s(value_nm3_h: float) -> float:
    """与 pred_dry_flow_nm3_h 互逆：Nm3/h → mol/s。"""
    return float(normal_m3_h_to_kmol_s(value_nm3_h)) * 1000.0


def dry_gap_components(gas_mol_s: np.ndarray, *, plant_dry_nm3_h: float, plant_co_vol_pct: float, plant_co2_vol_pct: float, plant_h2_vol_pct: float) -> dict[str, float]:
    """预测减装置。装置 CO/CO2/H2 由干基体积分数乘干气摩尔流量得到。"""
    gas = np.asarray(gas_mol_s, dtype=float)
    pred_dry = float(np.sum(gas[:7]))
    pred_co = float(gas[SPECIES_INDEX["CO"]])
    pred_co2 = float(gas[SPECIES_INDEX["CO2"]])
    pred_h2 = float(gas[SPECIES_INDEX["H2"]])
    pred_other = float(sum(gas[index] for index in OTHER_DRY_INDICES))
    pred_o2 = float(gas[SPECIES_INDEX["O2"]])
    plant_dry = nm3_h_to_mol_s(plant_dry_nm3_h)
    plant_co = plant_dry * float(plant_co_vol_pct) / 100.0
    plant_co2 = plant_dry * float(plant_co2_vol_pct) / 100.0
    plant_h2 = plant_dry * float(plant_h2_vol_pct) / 100.0
    plant_unmeasured = plant_dry - plant_co - plant_co2 - plant_h2
    gap_co = pred_co - plant_co
    gap_co2 = pred_co2 - plant_co2
    gap_h2 = pred_h2 - plant_h2
    gap_other = pred_other - plant_unmeasured
    dry_gap = pred_dry - plant_dry
    gap_sum = gap_co + gap_co2 + gap_h2 + gap_other + pred_o2
    return {
        "pred_dry_mol_s": pred_dry,
        "plant_dry_mol_s": plant_dry,
        "dry_gap_mol_s": dry_gap,
        "pred_CO_mol_s": pred_co,
        "plant_CO_mol_s": plant_co,
        "gap_CO_mol_s": gap_co,
        "pred_CO2_mol_s": pred_co2,
        "plant_CO2_mol_s": plant_co2,
        "gap_CO2_mol_s": gap_co2,
        "pred_H2_mol_s": pred_h2,
        "plant_H2_mol_s": plant_h2,
        "gap_H2_mol_s": gap_h2,
        "predicted_other_dry_mol_s": pred_other,
        "plant_unmeasured_dry_mol_s": plant_unmeasured,
        "gap_other_dry_mol_s": gap_other,
        "predicted_O2_mol_s": pred_o2,
        "pred_H2O_mol_s": float(gas[SPECIES_INDEX["H2O"]]),
        "gap_columns_sum_mol_s": gap_sum,
        "gap_columns_minus_dry_gap_mol_s": gap_sum - dry_gap,
        "o2_share_of_abs_dry_gap": abs(pred_o2) / max(abs(dry_gap), 1.0e-30),
    }


def refresh_gap_identity(gap: Mapping[str, Any]) -> dict[str, Any]:
    """四列加上预测出口 O2。已有行只改这两列，不重解。"""
    updated = dict(gap)
    four = (
        float(updated["gap_CO_mol_s"])
        + float(updated["gap_CO2_mol_s"])
        + float(updated["gap_H2_mol_s"])
        + float(updated["gap_other_dry_mol_s"])
    )
    predicted_o2 = float(updated["predicted_O2_mol_s"])
    dry_gap = float(updated["dry_gap_mol_s"])
    gap_sum = four + predicted_o2
    updated["gap_columns_sum_mol_s"] = gap_sum
    updated["gap_columns_minus_dry_gap_mol_s"] = gap_sum - dry_gap
    updated["o2_share_of_abs_dry_gap"] = abs(predicted_o2) / max(abs(dry_gap), 1.0e-30)
    return updated


def gap_stop_reason(gap: Mapping[str, Any]) -> str:
    """恒等式对不上、且差额不是预测 O2 时停止。O2 不到缺口 1% 则继续。"""
    residual = float(gap["gap_columns_minus_dry_gap_mol_s"])
    if abs(residual) > DRY_GAP_ACCOUNTING_ATOL_MOL_S:
        return (
            f"四列加上预测 O2 与干气缺口相差 {residual:.6g} mol/s，差额不是预测 O2。"
            "停止。"
        )
    share = float(gap["o2_share_of_abs_dry_gap"])
    if share >= O2_SHARE_CONTINUE_LIMIT:
        return (
            f"预测 O2 占干气缺口绝对值的 {share:.6g}，不低于 {O2_SHARE_CONTINUE_LIMIT:.6g}。"
            "停止。"
        )
    return ""


def classify_downstream(cell_frame: pd.DataFrame) -> dict[str, Any]:
    """句 1：下游全部不超过阈值且 cell 0 全部超过。句 2：下游至少有一个单元超过。"""
    downstream = cell_frame[cell_frame["cell_index"].astype(int) > 0]
    poor = downstream[downstream["exceeds_1e-4"].astype(bool)]
    cell0 = cell_frame[cell_frame["cell_index"].astype(int) == 0]
    cell0_all_exceed = bool(len(cell0) and cell0["exceeds_1e-4"].astype(bool).all())
    if poor.empty and cell0_all_exceed:
        return {
            "sentence_id": 1,
            "sentence": "下游单元全部低于 1e-4，只有 cell 0 超过阈值。",
            "cells": [],
        }
    cells = []
    for (n_cells, cell_index), group in poor.groupby(["N_cells", "cell_index"], sort=True):
        cells.append(
            {
                "N_cells": int(n_cells),
                "cell_index": int(cell_index),
                "n_windows": int(group["window_id"].nunique()),
                "cost_min": float(group["residual_cost"].min()),
                "cost_max": float(group["residual_cost"].max()),
            }
        )
    if cells:
        listed = "、".join(
            f"N={item['N_cells']} 的 cell {item['cell_index']}（{item['n_windows']} 个窗口，cost {item['cost_min']:.6g}–{item['cost_max']:.6g}）"
            for item in cells
        )
        return {
            "sentence_id": 2,
            "sentence": f"下游仍有单元超过 1e-4。{listed}。",
            "cells": cells,
        }
    return {
        "sentence_id": 0,
        "sentence": "下游没有超过 1e-4 的单元，cell 0 也没有在全部门窗口上超过阈值。",
        "cells": [],
    }


def _gap_shares(frame: pd.DataFrame) -> dict[str, pd.Series]:
    total = frame["dry_gap_mol_s"].astype(float).abs().clip(lower=1.0e-30)
    columns = {
        "CO": "gap_CO_mol_s",
        "CO2": "gap_CO2_mol_s",
        "H2": "gap_H2_mol_s",
        "other_dry": "gap_other_dry_mol_s",
    }
    return {name: frame[column].astype(float).abs() / total for name, column in columns.items()}


def classify_dry_gap(frame: pd.DataFrame) -> dict[str, Any]:
    """句 3：同一列在每一行都占干气缺口绝对值的 70% 以上。否则用句 4，并给出四列中位数。"""
    shares = _gap_shares(frame)
    medians = {
        "CO": float(frame["gap_CO_mol_s"].median()),
        "CO2": float(frame["gap_CO2_mol_s"].median()),
        "H2": float(frame["gap_H2_mol_s"].median()),
        "other_dry": float(frame["gap_other_dry_mol_s"].median()),
    }
    median_share = {name: float(series.median()) for name, series in shares.items()}
    unanimous = [name for name, series in shares.items() if bool((series >= GAP_SHARE_THRESHOLD).all())]
    if len(unanimous) == 1:
        name = unanimous[0]
        return {
            "sentence_id": 3,
            "sentence": f"干气缺口的主项是 {name}，该列占缺口绝对值的 70% 以上。",
            "dominant_column": name,
            "median_share": median_share,
            "medians_mol_s": medians,
            "min_share": float(shares[name].min()),
        }
    any_row_reaches = {name: int((series >= GAP_SHARE_THRESHOLD).sum()) for name, series in shares.items()}
    return {
        "sentence_id": 4,
        "sentence": (
            "没有任何一列在全部门窗口和网格上都达到缺口绝对值的 70%。"
            f"四列中位数为 CO {_fmt(medians['CO'])}、CO2 {_fmt(medians['CO2'])}、"
            f"H2 {_fmt(medians['H2'])}、other dry {_fmt(medians['other_dry'])} mol/s。"
        ),
        "dominant_column": "",
        "median_share": median_share,
        "medians_mol_s": medians,
        "rows_reaching_70pct": any_row_reaches,
    }


def _fmt(value: float) -> str:
    if not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.6g}"


def _window_record(row: Mapping[str, Any], n_cells: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    solved = solve_reference_case(row, n_cells=n_cells)
    system = solved["system"]
    profile = solved["profile"]
    closure, _projection, identity_gap, _combustion_flags = _identity_from_solve(solved)
    cell_rows: list[dict[str, Any]] = []
    term_rows: list[dict[str, Any]] = []
    for cell, state_row in zip(system.cells, profile):
        state = np.asarray(state_row, dtype=float)
        components = cell.residual_components(state)
        evaluated = evaluate_cell_state(cell, state)
        names = list(components["names"])
        criterion = rate_branch_criterion(cell, evaluated["current"], evaluated["gas_src"])
        cost = float(components["cost"])
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
                "is_combustion": bool(criterion["is_combustion"]),
                "H2_residual_mol_s": float(components["unscaled"][names.index("gas_H2")]),
                "CO2_residual_mol_s": float(components["unscaled"][names.index("gas_CO2")]),
                "element_species_identity_max_abs_mol_s": float(identity_gap),
            }
        )
        if cell.idx != 0:
            continue
        gas_unscaled = np.asarray(components["unscaled"][:8], dtype=float)
        reconstructed = evaluated["convection"] - evaluated["gas_src"] - evaluated["reaction"]
        for species in ("H2", "CO2"):
            index = SPECIES_INDEX[species]
            term_rows.append(
                {
                    "dataset_id": DATASET_ID,
                    "window_id": row["window_id"],
                    "phase": int(row["phase_number"]),
                    "split_role": row["split_role"],
                    "N_cells": int(n_cells),
                    "cell_index": 0,
                    "species": species,
                    "convection_inout_mol_s": float(evaluated["convection"][index]),
                    "reaction_source_mol_s": float(evaluated["reaction"][index]),
                    "inlet_source_mol_s": float(evaluated["gas_src"][index]),
                    "residual_mol_s": float(gas_unscaled[index]),
                    "term_sum_minus_residual_mol_s": float(reconstructed[index] - gas_unscaled[index]),
                }
            )
    outlet = np.asarray(profile[-1, :8], dtype=float)
    gap = dry_gap_components(
        outlet,
        plant_dry_nm3_h=float(row["syngas_flow_mean"]),
        plant_co_vol_pct=float(row["co_vol_pct"]),
        plant_co2_vol_pct=float(row["co2_vol_pct"]),
        plant_h2_vol_pct=float(row["h2_vol_pct"]),
    )
    gap.update(
        {
            "dataset_id": DATASET_ID,
            "window_id": row["window_id"],
            "phase": int(row["phase_number"]),
            "split_role": row["split_role"],
            "N_cells": int(n_cells),
            "outlet_solid_carbon_mol_s": float(closure["C_outlet_solid_mol_s"]),
            "C_closure_error_mol_s": float(closure["C_closure_error_mol_s"]),
            "element_species_identity_max_abs_mol_s": float(identity_gap),
        }
    )
    return cell_rows, term_rows, gap


def render_interpretation(
    downstream: Mapping[str, Any],
    dry_gap: Mapping[str, Any],
    cell_frame: pd.DataFrame,
    gap_frame: pd.DataFrame,
) -> str:
    lines = ["# P0-D 判断", "", downstream["sentence"], "", dry_gap["sentence"], ""]
    if downstream["cells"]:
        lines.extend(
            [
                "| N_cells | cell_index | n_windows | cost_min | cost_max |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for item in downstream["cells"]:
            lines.append(
                f"| {item['N_cells']} | {item['cell_index']} | {item['n_windows']} | {_fmt(item['cost_min'])} | {_fmt(item['cost_max'])} |"
            )
        lines.append("")
    cell0 = cell_frame[cell_frame["cell_index"].astype(int) == 0]
    lines.append(
        f"cell 0 超过 1e-4 的行数：{int(cell0['exceeds_1e-4'].astype(bool).sum())} / {len(cell0)}。"
        f" cost 范围 {_fmt(float(cell0['residual_cost'].min()))}–{_fmt(float(cell0['residual_cost'].max()))}。"
    )
    lines.append("")
    lines.extend(
        [
            "| 列 | 中位数 mol/s | 中位份额 | 达到 70% 的行数 |",
            "|---|---:|---:|---:|",
        ]
    )
    shares = _gap_shares(gap_frame)
    medians = dry_gap["medians_mol_s"]
    reaching = dry_gap.get("rows_reaching_70pct")
    if reaching is None and dry_gap.get("dominant_column"):
        reaching = {name: int((series >= GAP_SHARE_THRESHOLD).sum()) for name, series in shares.items()}
    for name in ("CO", "CO2", "H2", "other_dry"):
        lines.append(
            f"| {name} | {_fmt(medians[name])} | {_fmt(dry_gap['median_share'][name])} | {int(reaching[name])} |"
        )
    lines.extend(
        [
            "",
            "份额 = 该列绝对值 / |预测干气 − 装置干气|。中位数跨 9 个窗口和 N=3、5、9。",
            "预测其余干气是 CH4+H2S+N2。装置没有这三项的独立测量；`plant_unmeasured_dry_mol_s` 是干气总量减去 CO、CO2、H2。",
            "干气缺口等于这四列加上预测出口 O2。O2 不并入 other dry。",
            "元素恒等式是反应器元素误差与各单元气相残差原子加总之差的最大绝对值。",
        ]
    )
    return "\n".join(lines) + "\n"


def render_stop(reason: str, detail: Mapping[str, Any]) -> str:
    lines = [
        "# P0-D 停止",
        "",
        reason,
        "",
        f"window_id: {detail.get('window_id', '')}",
        f"N_cells: {detail.get('N_cells', '')}",
        f"element_species_identity_max_abs_mol_s: {_fmt(float(detail.get('element_species_identity_max_abs_mol_s', math.nan)))}",
        f"gap_columns_minus_dry_gap_mol_s: {_fmt(float(detail.get('gap_columns_minus_dry_gap_mol_s', math.nan)))}",
        f"predicted_O2_mol_s: {_fmt(float(detail.get('predicted_O2_mol_s', math.nan)))}",
        "",
        "没有改平衡系数、热解产率、phi、f_co2、H2_OxidationFactor 或燃烧阈值。",
    ]
    return "\n".join(lines) + "\n"


def _protocol(window_ids: Sequence[str], commit: str, *, stopped: bool, stop_reason: str, identity_max: float, accounting_max: float) -> dict[str, Any]:
    return {
        "dataset_id": DATASET_ID,
        "configuration": "reference",
        "alpha_wgs": 1.0,
        "grids": list(GRIDS),
        "window_ids": list(window_ids),
        "n_windows": len(list(window_ids)),
        "n1_reproduction_rerun": False,
        "full_split_rerun": False,
        "dry_gas_definition": "sum of O2, CH4, CO, CO2, H2S, H2, N2 mol/s; Nm3/h uses 22.414 and the same conversion as pred_dry_flow_nm3_h",
        "plant_species_from_dry_fraction": "plant mol/s = plant dry mol/s * dry vol pct / 100",
        "predicted_other_dry": "CH4 + H2S + N2 mol/s; plant N2, CH4 and H2S are not measured",
        "plant_unmeasured_dry": "plant dry mol/s minus plant CO, CO2 and H2; not an independent measurement",
        "gap_columns": ["CO", "CO2", "H2", "other_dry"],
        "gap_accounting_rule": "pred dry minus plant dry equals gap_CO + gap_CO2 + gap_H2 + gap_other_dry + predicted_O2",
        "gap_accounting_atol_mol_s": DRY_GAP_ACCOUNTING_ATOL_MOL_S,
        "o2_share_continue_limit": O2_SHARE_CONTINUE_LIMIT,
        "o2_alone_below_one_percent_continues": True,
        "gap_accounting_max_abs_mol_s": accounting_max,
        "gap_share": "abs(column) / abs(pred_dry - plant_dry)",
        "gap_sentence_3_rule": "one column has share >= 0.7 on every completed window and grid",
        "downstream_sentence_1_rule": "every cell_index > 0 has residual_cost <= 1e-4 and every cell 0 exceeds 1e-4",
        "element_identity": "max abs difference between reactor element closure error and the sum of cell gas-residual atom projections; H=4*CH4+2*H2S+2*H2+2*H2O, C=CH4+CO+CO2, O=2*O2+CO+2*CO2+H2O, S=H2S, N=2*N2",
        "element_identity_stop_mol_s": ELEMENT_IDENTITY_STOP_MOL_S,
        "element_identity_max_abs_mol_s": identity_max,
        "unchanged": [
            "balance coefficients",
            "pyrolysis yields",
            "phi",
            "f_co2",
            "H2_OxidationFactor",
            "P_O2_Combustion_atm=0.05",
        ],
        "residual_cost_threshold": POOR_CONVERGENCE_COST_THRESHOLD,
        "stopped": stopped,
        "stop_reason": stop_reason,
        "source_commit": commit,
        "handoff_model_head": HANDOFF_MODEL_HEAD,
        "solver_modules_match_handoff_head": commit.startswith(HANDOFF_MODEL_HEAD),
    }


def _write_outputs(
    output_dir: Path,
    *,
    cell_rows: list[dict[str, Any]],
    term_rows: list[dict[str, Any]],
    gap_rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    interpretation: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cell_frame = pd.DataFrame(cell_rows)
    term_frame = pd.DataFrame(term_rows)
    gap_frame = pd.DataFrame(gap_rows)
    paths = {
        "cells": output_dir / "p0d_cell_residual.csv",
        "terms": output_dir / "p0d_cell0_h2_co2_terms.csv",
        "gaps": output_dir / "p0d_dry_flow_gap.csv",
        "protocol": output_dir / "p0d_protocol.json",
        "interpretation": output_dir / "p0d_interpretation.md",
    }
    cell_frame.to_csv(paths["cells"], index=False)
    term_frame.to_csv(paths["terms"], index=False)
    gap_frame.to_csv(paths["gaps"], index=False)
    protocol_text = json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    assert_no_local_routing(protocol_text)
    assert_no_local_routing(interpretation)
    paths["protocol"].write_text(protocol_text, encoding="utf-8")
    paths["interpretation"].write_text(interpretation, encoding="utf-8")
    write_sha256_manifest(list(paths.values()), output_dir / "p0d_sha256.txt")


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _load_reusable(output_dir: Path) -> dict[tuple[str, int], tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]]:
    """已有出口 O2 的窗口×网格不再重解。缺 O2 或缺单元行则留给求解。"""
    gap_path = output_dir / "p0d_dry_flow_gap.csv"
    cell_path = output_dir / "p0d_cell_residual.csv"
    term_path = output_dir / "p0d_cell0_h2_co2_terms.csv"
    if not (gap_path.exists() and cell_path.exists() and term_path.exists()):
        return {}
    gaps = pd.read_csv(gap_path)
    cells = pd.read_csv(cell_path)
    terms = pd.read_csv(term_path)
    if "predicted_O2_mol_s" not in gaps.columns:
        return {}
    reusable: dict[tuple[str, int], tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]] = {}
    for record in gaps.to_dict(orient="records"):
        if not math.isfinite(float(record["predicted_O2_mol_s"])):
            continue
        key = (str(record["window_id"]), int(record["N_cells"]))
        cell_part = cells[(cells["window_id"].astype(str) == key[0]) & (cells["N_cells"].astype(int) == key[1])]
        if len(cell_part) != key[1]:
            continue
        term_part = terms[(terms["window_id"].astype(str) == key[0]) & (terms["N_cells"].astype(int) == key[1])]
        cell_rows = cell_part.to_dict(orient="records")
        for item in cell_rows:
            item["exceeds_1e-4"] = _as_bool(item["exceeds_1e-4"])
            item["is_combustion"] = _as_bool(item["is_combustion"])
        reusable[key] = (cell_rows, term_part.to_dict(orient="records"), refresh_gap_identity(record))
    return reusable


def run_p0d(rows: Sequence[Mapping[str, Any]], window_ids: Sequence[str], output_dir: Path) -> dict[str, Any]:
    silence_solver_logs()
    output_dir.mkdir(parents=True, exist_ok=True)
    commit = source_commit()
    by_id = {str(row["window_id"]): row for row in rows}
    selected_ids = [str(window_id) for window_id in window_ids]
    missing = [window_id for window_id in selected_ids if window_id not in by_id]
    if missing:
        raise RuntimeError(f"representative window ids are not all in the frozen split: {missing}")
    reusable = _load_reusable(output_dir)
    cell_rows: list[dict[str, Any]] = []
    term_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    reused_keys: list[str] = []
    identity_max = 0.0
    accounting_max = 0.0
    o2_share_max = 0.0
    stopped = False
    stop_reason = ""
    stop_detail: dict[str, Any] = {}
    for n_cells in GRIDS:
        for index, window_id in enumerate(selected_ids, start=1):
            key = (window_id, int(n_cells))
            if key in reusable:
                print(f"P0-D reuse N={n_cells} {index}/{len(selected_ids)} {window_id}", flush=True)
                part_cells, part_terms, gap = reusable[key]
                reused_keys.append(f"{window_id}|N={n_cells}")
            else:
                print(f"P0-D N={n_cells} {index}/{len(selected_ids)} {window_id}", flush=True)
                part_cells, part_terms, gap = _window_record(by_id[window_id], n_cells)
            cell_rows.extend(part_cells)
            term_rows.extend(part_terms)
            gap_rows.append(gap)
            identity_gap = float(gap["element_species_identity_max_abs_mol_s"])
            accounting_gap = abs(float(gap["gap_columns_minus_dry_gap_mol_s"]))
            identity_max = max(identity_max, identity_gap)
            accounting_max = max(accounting_max, accounting_gap)
            o2_share_max = max(o2_share_max, float(gap["o2_share_of_abs_dry_gap"]))
            if identity_gap > ELEMENT_IDENTITY_STOP_MOL_S:
                stopped = True
                stop_reason = (
                    f"元素恒等式最大绝对偏差 {identity_gap:.6g} mol/s，大于 {ELEMENT_IDENTITY_STOP_MOL_S:.6g} mol/s。"
                    "停止，不改系数。"
                )
                stop_detail = gap
                break
            reason = gap_stop_reason(gap)
            if reason:
                stopped = True
                stop_reason = reason
                stop_detail = gap
                break
        if stopped:
            break
    n_completed = len(gap_rows)
    complete = n_completed == EXPECTED_SOLVES and not stopped
    if not complete:
        interpretation = render_stop(stop_reason or "27 个求解没有齐，判断句保持为 0。", stop_detail)
        downstream_id = 0
        gap_id = 0
    else:
        cell_frame = pd.DataFrame(cell_rows)
        gap_frame = pd.DataFrame(gap_rows)
        downstream = classify_downstream(cell_frame)
        dry_gap = classify_dry_gap(gap_frame)
        interpretation = render_interpretation(downstream, dry_gap, cell_frame, gap_frame)
        downstream_id = int(downstream["sentence_id"])
        gap_id = int(dry_gap["sentence_id"])
    protocol = _protocol(
        selected_ids,
        commit,
        stopped=stopped,
        stop_reason=stop_reason,
        identity_max=identity_max,
        accounting_max=accounting_max,
    )
    protocol["downstream_sentence_id"] = downstream_id
    protocol["dry_gap_sentence_id"] = gap_id
    protocol["n_solves_completed"] = n_completed
    protocol["n_solves_expected"] = EXPECTED_SOLVES
    protocol["o2_share_max"] = o2_share_max
    protocol["reused_without_resolve"] = reused_keys
    _write_outputs(
        output_dir,
        cell_rows=cell_rows,
        term_rows=term_rows,
        gap_rows=gap_rows,
        protocol=protocol,
        interpretation=interpretation,
    )
    return {
        "stopped": stopped,
        "stop_reason": stop_reason,
        "identity_max_abs_mol_s": identity_max,
        "accounting_max_abs_mol_s": accounting_max,
        "o2_share_max": o2_share_max,
        "n_solves_completed": n_completed,
        "downstream_sentence_id": downstream_id,
        "dry_gap_sentence_id": gap_id,
    }
