#!/usr/bin/env python3
"""Generate M4 local estimability artifacts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model.revision_m4_estimability import run_m4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", default="reference")
    parser.add_argument(
        "--runtime-csv",
        type=Path,
        default=ROOT / "outputs" / "diagnostics" / "round55" / "huayi_round55_runtime_and_convergence.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "diagnostics" / "revision-evidence-20260826" / "M4-estimability",
    )
    args = parser.parse_args()
    run_m4(output_dir=args.output_dir, runtime_path=args.runtime_csv, configuration=args.configuration)


if __name__ == "__main__":
    main()
