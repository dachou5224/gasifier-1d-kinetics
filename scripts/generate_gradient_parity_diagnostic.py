#!/usr/bin/env python3
"""Generate FD-vs-IFT/JAX gradient parity diagnostics."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _ensure_src_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return repo_root


def main() -> None:
    repo_root = _ensure_src_path()
    from model.gradient_parity_diagnostic import (
        GRADIENT_PARITY_COLUMNS,
        build_gradient_parity_rows,
        gradient_parity_markdown,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--step-fraction", type=float, default=1.0e-3)
    parser.add_argument("--out-dir", default=str(repo_root / "outputs" / "diagnostics"))
    parser.add_argument("--basename", default="gradient_parity_diagnostic")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = build_gradient_parity_rows(step_fraction=args.step_fraction)
    table = pd.DataFrame(rows, columns=list(GRADIENT_PARITY_COLUMNS))
    csv_path = out_dir / f"{args.basename}.csv"
    md_path = out_dir / f"{args.basename}.md"
    table.to_csv(csv_path, index=False)
    md_path.write_text(gradient_parity_markdown(rows), encoding="utf-8")

    ok = table[table["status"] == "ok"]
    print(f"wrote_csv={csv_path}")
    print(f"wrote_markdown={md_path}")
    print(f"rows={len(table)}")
    print(f"ok_rows={len(ok)}")
    print(f"blocked_rows={int((table['status'] == 'blocked').sum())}")
    print(f"max_abs_error={ok['abs_error'].max():.12g}")
    print(f"max_rel_error={ok['rel_error'].max():.12g}")


if __name__ == "__main__":
    main()
