#!/usr/bin/env python3
"""Generate M10 residual comparator artifacts on the frozen 142-window split."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model.revision_m10_residual_142 import run_m10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixed-physics",
        type=Path,
        default=ROOT / "outputs" / "diagnostics" / "revision-evidence-20260826" / "M1-cell0" / "revision_m1_fullsplit_stability.csv",
    )
    parser.add_argument(
        "--m2-effect",
        type=Path,
        default=ROOT / "outputs" / "diagnostics" / "revision-evidence-20260826" / "M2-wgs-effects" / "revision_m2_wgs_effect_matrix.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "diagnostics" / "round-r4-m10-residual-142",
    )
    args = parser.parse_args()
    paths = run_m10(
        output_dir=args.output_dir,
        fixed_physics_path=args.fixed_physics,
        m2_effect_path=args.m2_effect,
    )
    for key, path in paths.items():
        print(f"{key}={path}")


if __name__ == "__main__":
    main()
