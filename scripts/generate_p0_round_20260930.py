#!/usr/bin/env python3
"""生成 P0-A 残差诊断，并在其后执行 P0-B 的 N=1 复现锁。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model.p0_round_20260930 import (
    load_frozen_split,
    m1_reference_window_ids,
    m2_window_ids,
    run_p0a,
    run_p0b_n1_lock,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("a", "b-lock", "a-then-b-lock"),
        default="a-then-b-lock",
    )
    parser.add_argument(
        "--a-dir",
        type=Path,
        default=ROOT / "outputs" / "diagnostics" / "round-p0-20260930" / "A-residual",
    )
    parser.add_argument(
        "--b-dir",
        type=Path,
        default=ROOT / "outputs" / "diagnostics" / "round-p0-20260930" / "B-mesh",
    )
    parser.add_argument(
        "--m1-variant-matrix",
        type=Path,
        default=ROOT / "outputs" / "diagnostics" / "revision-evidence-20260826" / "M1-cell0" / "revision_m1_cell0_variant_matrix.csv",
    )
    parser.add_argument(
        "--m2-stability",
        type=Path,
        default=ROOT / "outputs" / "diagnostics" / "revision-evidence-20260826" / "M1-cell0" / "revision_m1_fullsplit_stability.csv",
    )
    args = parser.parse_args()
    frozen_ok, rows, _inventory = load_frozen_split()
    if not frozen_ok:
        raise SystemExit("frozen R52-C-75 split was not consumed intact")
    if args.stage in {"a", "a-then-b-lock"}:
        window_ids = m1_reference_window_ids(args.m1_variant_matrix)
        judgment = run_p0a(rows, window_ids, args.a_dir)
        print(judgment["sentence"], flush=True)
    if args.stage in {"b-lock", "a-then-b-lock"}:
        result = run_p0b_n1_lock(rows, m2_window_ids(args.m2_stability), args.b_dir)
        print(
            f"P0-B N=1 passed={result['passed']} rmse={result['reproduced_rmse']} abs_error={result['abs_error']}",
            flush=True,
        )
        if not result["passed"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
