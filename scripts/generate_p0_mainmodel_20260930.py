#!/usr/bin/env python3
"""P0 主模型阶段 0：9 个窗口、N=3，核对氢气汇是否重复扣减。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model.p0_mainmodel_20260930 import run_finer_grids_and_full_split, run_modified_n3, run_stage0, write_stage0_stop
from model.p0_round_20260930 import load_frozen_split, m1_reference_window_ids


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("stage0", "n3", "continue"), default="n3")
    args = parser.parse_args()
    output_dir = ROOT / "outputs" / "diagnostics" / "round-p0-mainmodel-20260930"
    variant = ROOT / "outputs" / "diagnostics" / "revision-evidence-20260826" / "M1-cell0" / "revision_m1_cell0_variant_matrix.csv"
    frozen_ok, rows, _inventory = load_frozen_split()
    if not frozen_ok:
        raise SystemExit("frozen R52-C-75 split was not consumed intact")
    window_ids = m1_reference_window_ids(variant)
    if len(window_ids) != 9:
        raise SystemExit(f"expected 9 P0-A windows, found {len(window_ids)}")
    if args.stage == "stage0":
        if (output_dir / "stage0_h2_sinks.csv").exists():
            raise SystemExit("stage0_h2_sinks.csv already exists; refuse to overwrite")
        result = run_stage0(rows, window_ids, output_dir)
        print(f"P0-main stage0 exceed={result['n_exceed']}/9 passed={result['passed']}", flush=True)
        if not result["passed"]:
            write_stage0_stop(output_dir, result)
            raise SystemExit(2)
        return
    if args.stage == "continue":
        result = run_finer_grids_and_full_split(rows, window_ids, output_dir)
        print(
            "P0-main continue "
            f"representative={result['passed_grids']} "
            f"full={result['full_pass']} "
            f"selected_N={result['selected_N']} "
            f"identity={result['identity_max_abs_mol_s']:.6g} "
            f"revert={result['identity_failed']}",
            flush=True,
        )
        if result["identity_failed"] or result["selected_N"] is None:
            raise SystemExit(2)
        return
    result = run_modified_n3(rows, window_ids, output_dir)
    print(
        "P0-main N=3 "
        f"passed={result['passed']} "
        f"above={result['n_cells_above_1e-4']} "
        f"identity={result['identity_max_abs_mol_s']:.6g} "
        f"revert={result['identity_failed']}",
        flush=True,
    )
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
