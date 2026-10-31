#!/usr/bin/env python3
"""P0-D：只用 P0-A 的 9 个窗口，比较 N=3、5、9 的单元残差和干气缺口。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model.p0_round_20260930 import load_frozen_split, m1_reference_window_ids
from model.p0d_20260930 import run_p0d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "diagnostics" / "round-p0-20260930" / "D-cells",
    )
    parser.add_argument(
        "--m1-variant-matrix",
        type=Path,
        default=ROOT / "outputs" / "diagnostics" / "revision-evidence-20260826" / "M1-cell0" / "revision_m1_cell0_variant_matrix.csv",
    )
    args = parser.parse_args()
    frozen_ok, rows, _inventory = load_frozen_split()
    if not frozen_ok:
        raise SystemExit("frozen R52-C-75 split was not consumed intact")
    window_ids = m1_reference_window_ids(args.m1_variant_matrix)
    if len(window_ids) != 9:
        raise SystemExit(f"expected 9 P0-A windows, found {len(window_ids)}")
    result = run_p0d(rows, window_ids, args.output_dir)
    print(
        "P0-D "
        f"stopped={result['stopped']} "
        f"identity={result['identity_max_abs_mol_s']:.6g} "
        f"accounting={result['accounting_max_abs_mol_s']:.6g} "
        f"downstream={result['downstream_sentence_id']} "
        f"gap={result['dry_gap_sentence_id']}",
        flush=True,
    )
    if result["stopped"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
