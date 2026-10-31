#!/usr/bin/env python3
"""P1：4 个阶段中位窗口在 N=3、5、9 上的单向通道响应。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model.p0_round_20260930 import load_frozen_split
from model.p1_response_20260930 import run_p1_response


def main() -> None:
    output_dir = ROOT / "outputs" / "diagnostics" / "round-p1-response-20260930"
    frozen_ok, rows, _inventory = load_frozen_split()
    if not frozen_ok:
        raise SystemExit("frozen R52-C-75 split was not consumed intact")
    result = run_p1_response(rows, output_dir)
    print(
        f"P1 stopped={result['stopped']} rows={result['n_rows']} interpretable={result['n_interpretable']}",
        flush=True,
    )
    if result["stopped"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
