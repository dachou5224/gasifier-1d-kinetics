#!/usr/bin/env python3
"""Generate M1 cell-0 residual-quality robustness artifacts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model.revision_m1_cell0 import run_m1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "diagnostics" / "revision-evidence-20260826" / "M1-cell0",
    )
    parser.add_argument(
        "--runtime-csv",
        type=Path,
        default=ROOT / "outputs" / "diagnostics" / "round55" / "huayi_round55_runtime_and_convergence.csv",
    )
    parser.add_argument("--screen-only", action="store_true")
    parser.add_argument("--reuse-screen", action="store_true")
    args = parser.parse_args()
    run_m1(
        output_dir=args.output_dir,
        runtime_path=args.runtime_csv,
        full_split=not args.screen_only,
        reuse_screen=args.reuse_screen,
    )


if __name__ == "__main__":
    main()
