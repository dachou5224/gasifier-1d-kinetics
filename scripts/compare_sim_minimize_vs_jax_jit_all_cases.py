"""
对比验证：`minimize` vs `jax_jit`（当前 19 case benchmark）。

输出内容：
1) 每个 case：出口 `T_out` (K) 与干基 mol%（忽略 H2O）的 `yCO/yH2/yCO2` 差值
2) 每个 case：profile 最大误差（max|ΔT|, max|ΔyCO|, max|ΔyH2|, max|ΔyCO2|）

注意：
- `warmup`：仅对第 1 个 case 打开 `jax_warmup=True`，避免编译成本污染全部统计。
- `干基 mol%`：按 UI 口径，F_dry = sum(gas[:7])（gas 顺序 [O2,CH4,CO,CO2,H2S,H2,N2,H2O]）
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np


def _ensure_import_path() -> None:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src_path = os.path.join(project_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def _dry_y(profile: np.ndarray, idx: int) -> np.ndarray:
    """
    profile: (N_cells, 11) ; gas: (:, :8) -> 干基 mol%（忽略 H2O）
    """
    gas = profile[:, :8].astype(float)
    denom = gas[:, :7].sum(axis=1) + 1e-12  # dry basis excludes H2O
    return gas[:, idx] / denom * 100.0


def _build_system(
    name: str,
    case_data: Dict,
    coal_database: Dict,
    cases_pressure: float,
):
    from model.gasifier_system import GasifierSystem
    from model.chemistry import COAL_DATABASE, CASE_TO_COAL_MAP

    cond = case_data["operating_conditions"]
    expected = case_data.get("expected_results") or case_data.get("expected") or {}

    geometry = {"L": cond.get("L_reactor_m", 6.0), "D": cond.get("D_reactor_m", 1.5)}

    # 煤种映射
    coal_type = case_data.get("coal_type")
    if name in CASE_TO_COAL_MAP:
        mapped = CASE_TO_COAL_MAP[name]
        if mapped in COAL_DATABASE:
            coal_type = mapped
    if not coal_type or coal_type not in COAL_DATABASE:
        coal_type = "Paper_Base_Coal"
    coal_props = COAL_DATABASE[coal_type]

    # feed_type 判定：沿用 run_all_validation_cases.py 的 feedstock 逻辑
    orig_feedstock = case_data.get("feedstock_type_orig", "").lower()
    feed_type_explicit = case_data.get("feed_type")

    name_lower = name.lower()
    is_true_slurry = "slurry" in name_lower or "coal_water_slurry" in orig_feedstock
    is_residue = ("residue" in orig_feedstock) or ("molten" in orig_feedstock)

    if feed_type_explicit:
        feed_type = feed_type_explicit
    elif is_true_slurry and not is_residue:
        feed_type = "Slurry-fed"
    else:
        feed_type = "Dry-fed"

    slurry_conc = cond.get("slurry_concentration_pct", 100.0)
    coal_flow_kg_h = cond.get("coal_feed_rate_kg_hr", 100.0)
    if cond.get("coal_feed_rate_g_s") is not None:
        coal_flow_kg_h = cond["coal_feed_rate_g_s"] * 3.6
    coal_flow = coal_flow_kg_h / 3600.0

    ratio_oc = cond.get("O2_to_coal_ratio", cond.get("O2_to_fuel_ratio", cond.get("Ratio_OC", 0.9)))
    raw_water_ratio = cond.get(
        "water_to_coal_ratio",
        cond.get("steam_to_fuel_ratio", cond.get("Ratio_SC", 0.0)),
    )

    if feed_type == "Slurry-fed":
        if slurry_conc == 100.0 or slurry_conc is None:
            slurry_conc = 100.0 / (1.0 + raw_water_ratio)
        ratio_sc = 0.0
    else:
        ratio_sc = raw_water_ratio
        slurry_conc = 100.0

    P = cond.get("pressure_Pa", cond.get("pressure_atm", 1.0) * 101325.0)

    op_conds = cond.copy()
    op_conds.update(
        {
            "coal_flow": coal_flow,
            "o2_flow": coal_flow * ratio_oc,
            "steam_flow": coal_flow * ratio_sc,
            "P": P,
            "T_in": cond.get("inlet_temperature_K", 400.0),
            "HeatLossPercent": cond.get("heat_loss_percent", 2.0),
            "SlurryConcentration": slurry_conc,
            "AdaptiveFirstCellLength": True,
        }
    )

    # 与当前模型默认对齐
    if "Combustion_CO2_Fraction" not in op_conds:
        op_conds["Combustion_CO2_Fraction"] = 0.15
    if "WGS_CatalyticFactor" not in op_conds:
        op_conds["WGS_CatalyticFactor"] = 1.5

    return GasifierSystem(geometry, coal_props, op_conds)


def _collect_cases() -> List[Tuple[str, str, str, Dict]]:
    from model.validation_loader import (
        get_validation_cases_final_path,
        iter_validation_cases_final,
    )

    import json

    with open(get_validation_cases_final_path(), "r", encoding="utf-8") as f:
        data = json.load(f)

    cases: List[Tuple[str, str, str, Dict]] = []
    for name, case_data in iter_validation_cases_final(data):
        cases.append(
            (
                str(case_data.get("capacity", "")),
                str(case_data.get("feed_type", "")),
                name,
                case_data,
            )
        )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N_cells", type=int, default=20)
    parser.add_argument(
        "--out",
        type=str,
        default="docs/minimize_vs_jax_jit_report.md",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-level", type=str, default="ERROR")
    args = parser.parse_args()

    _ensure_import_path()
    from model.chemistry import COAL_DATABASE  # noqa: F401

    cases = _collect_cases()
    if args.limit:
        cases = cases[: args.limit]

    try:
        import jax  # noqa: F401
        import jaxlib  # noqa: F401
    except Exception as e:
        out_path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        blocked_lines = [
            "# minimize vs jax_jit：全量 profile/出口对比（干基 mol%）",
            "",
            f"- N_cells: {args.N_cells}",
            f"- cases: {len(cases)}",
            "- status: blocked",
            f"- reason: 当前环境缺少 JAX 运行时，无法执行 `solver_method='jax_jit'`（{type(e).__name__}: {e}）",
            "",
            "如需生成有效对比报告，请在可用环境中安装 `jax` 与 `jaxlib` 后重跑本脚本。",
            "",
        ]
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(blocked_lines))
        print(f"[BLOCKED] jax_jit runtime unavailable: {e}")
        print(f"Done. Report: {out_path}")
        return

    print(f"Comparing minimize vs jax_jit for {len(cases)} cases, N_cells={args.N_cells}")

    md_lines: List[str] = []
    md_lines.append("# minimize vs jax_jit：全量 profile/出口对比（干基 mol%）")
    md_lines.append("")
    md_lines.append(f"- N_cells: {args.N_cells}")
    md_lines.append(f"- cases: {len(cases)}")
    md_lines.append("")

    md_lines.append(
        "| Case | T_out_min(K) | T_out_jit(K) | ΔT(K) | max|ΔT|(K) | max|ΔyCO|(mol%) | max|ΔyH2|(mol%) | max|ΔyCO2|(mol%) |"
    )
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    ok = 0
    fail = 0
    t0_all = time.perf_counter()
    fail_reasons: Dict[str, int] = {}

    for i, (_cap, _ftype, name, case_data) in enumerate(cases):
        sys_obj = None
        try:
            sys_obj = _build_system(name, case_data, COAL_DATABASE, cases_pressure=0.0)
            t0 = time.perf_counter()
            prof_min, _z = sys_obj.solve(N_cells=args.N_cells, solver_method="minimize", jacobian_mode="scipy", jax_warmup=False)
            t_min = time.perf_counter() - t0

            # jax_jit：仅第 1 个 case warmup（编译摊销）
            # 这里重建 system，确保它走系统自己的缓存逻辑
            sys_obj2 = _build_system(name, case_data, COAL_DATABASE, cases_pressure=0.0)
            t1 = time.perf_counter()
            prof_jit, _z2 = sys_obj2.solve(
                N_cells=args.N_cells,
                solver_method="jax_jit",
                jax_warmup=(i == 0),
            )
            t_jit = time.perf_counter() - t1
            _ = (t_min, t_jit)

            T_min = float(prof_min[-1, 10])
            T_jit = float(prof_jit[-1, 10])
            dT = T_jit - T_min
            max_dT = float(np.max(np.abs(prof_jit[:, 10] - prof_min[:, 10])))

            # 干基 mol%：CO idx=2, H2 idx=5, CO2 idx=3
            yCO_min = _dry_y(prof_min, 2)
            yCO_jit = _dry_y(prof_jit, 2)
            yH2_min = _dry_y(prof_min, 5)
            yH2_jit = _dry_y(prof_jit, 5)
            yCO2_min = _dry_y(prof_min, 3)
            yCO2_jit = _dry_y(prof_jit, 3)

            max_dyCO = float(np.max(np.abs(yCO_jit - yCO_min)))
            max_dyH2 = float(np.max(np.abs(yH2_jit - yH2_min)))
            max_dyCO2 = float(np.max(np.abs(yCO2_jit - yCO2_min)))

            ok += 1
            md_lines.append(
                "| "
                + " | ".join(
                    [
                        name,
                        f"{T_min:.3f}",
                        f"{T_jit:.3f}",
                        f"{dT:+.3f}",
                        f"{max_dT:.3f}",
                        f"{max_dyCO:.3f}",
                        f"{max_dyH2:.3f}",
                        f"{max_dyCO2:.3f}",
                    ]
                )
                + " |"
            )
        except Exception as e:
            fail += 1
            err = str(e).replace("\n", " ")[:120]
            md_lines.append("| " + " | ".join([name, "-", "-", "-", "-", "-", "-", "-"]) + " |")
            fail_reasons[err] = fail_reasons.get(err, 0) + 1
            print(f"[FAIL] {name}: {err}")

        if (i + 1) % 3 == 0:
            elapsed = time.perf_counter() - t0_all
            print(f"progress {i+1}/{len(cases)} ok={ok} fail={fail} elapsed={elapsed:.1f}s")

    md_lines.append("")
    md_lines.append(f"- ok: {ok}, fail: {fail}")
    md_lines.append(f"- wall time: {time.perf_counter()-t0_all:.1f}s")
    if fail_reasons:
        md_lines.append("- fail reasons:")
        for reason, count in sorted(fail_reasons.items(), key=lambda item: (-item[1], item[0])):
            md_lines.append(f"  - {count}x {reason}")

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"Done. Report: {out_path}")


if __name__ == "__main__":
    main()
