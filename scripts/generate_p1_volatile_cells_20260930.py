#!/usr/bin/env python3
"""P1：挥发分通道在 N=3、5、9 上的单元分解。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model.p0_round_20260930 import load_frozen_split
from model.p1_volatile_cells_20260930 import run_volatile_cells


def main() -> None:
    output_dir = ROOT / "outputs" / "diagnostics" / "round-p1-volatile-cells-20260930"
    previous = ROOT / "outputs" / "diagnostics" / "round-p1-response-20260930" / "channel_response.csv"
    frozen_ok, rows, _inventory = load_frozen_split()
    if not frozen_ok:
        raise SystemExit("frozen R52-C-75 split was not consumed intact")
    result = run_volatile_cells(rows, previous, output_dir)
    print(f"P1-volatile stopped={result['stopped']} rows={result['n_rows']}", flush=True)
    if result["stopped"]:
        print(result["stop_reason"], flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
