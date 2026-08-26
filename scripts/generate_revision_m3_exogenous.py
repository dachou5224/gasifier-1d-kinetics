#!/usr/bin/env python3
"""Generate M3 exogenous-only WGS comparator artifacts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model.revision_m3_exogenous import run_m3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", default="reference")
    parser.add_argument("--selected-global-alpha", type=float, default=0.05)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "diagnostics" / "revision-evidence-20260826" / "M3-exogenous-comparator",
    )
    args = parser.parse_args()
    run_m3(
        output_dir=args.output_dir,
        configuration=args.configuration,
        selected_global_alpha=args.selected_global_alpha,
    )


if __name__ == "__main__":
    main()
