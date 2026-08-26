#!/usr/bin/env python3
"""Generate M2 WGS effect-size and bootstrap artifacts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model.revision_m2_wgs_effects import run_m2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stability-csv",
        type=Path,
        default=ROOT / "outputs" / "diagnostics" / "revision-evidence-20260826" / "M1-cell0" / "revision_m1_fullsplit_stability.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "diagnostics" / "revision-evidence-20260826" / "M2-wgs-effects",
    )
    args = parser.parse_args()
    run_m2(stability_path=args.stability_csv, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
