#!/usr/bin/env python3
"""P0 后续：先做燃烧区分支诊断，恒等式成立后再比较 N=3、5、9。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model.p0_followup_20260930 import run_mesh, run_task1
from model.p0_round_20260930 import load_frozen_split, m1_reference_window_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("c", "mesh", "c-then-mesh"), default="c-then-mesh")
    parser.add_argument(
        "--c-dir",
        type=Path,
        default=ROOT / "outputs" / "diagnostics" / "round-p0-20260930" / "C-branch",
    )
    parser.add_argument(
        "--b-dir",
        type=Path,
        default=ROOT / "outputs" / "diagnostics" / "round-p0-20260930" / "B-mesh",
    )
    parser.add_argument(
        "--p0a-term",
        type=Path,
        default=ROOT / "outputs" / "diagnostics" / "round-p0-20260930" / "A-residual" / "p0a_cell0_term_split.csv",
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
    if args.stage in {"c", "c-then-mesh"}:
        result = run_task1(rows, m1_reference_window_ids(args.m1_variant_matrix), args.c_dir, args.p0a_term)
        print(
            f"P0-C identity_holds={result['identity_holds']} gap={result['identity_gap']} classes={result['stop_classes']}",
            flush=True,
        )
        if not result["identity_holds"]:
            raise SystemExit(2)
    if args.stage in {"mesh", "c-then-mesh"}:
        mesh = run_mesh(
            rows,
            output_dir=args.b_dir,
            stability_csv=args.m2_stability,
            variant_matrix_csv=args.m1_variant_matrix,
            n1_reproduction_json=args.b_dir / "p0b_n1_reproduction.json",
        )
        print(
            f"P0-B mesh n1_s={mesh['n1_time_s']:.3f} n9_s={mesh['n9_time_s']:.3f} finer_subset={mesh['finer_mesh_subset']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
