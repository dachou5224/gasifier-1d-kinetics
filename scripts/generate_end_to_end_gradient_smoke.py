#!/usr/bin/env python3
"""Generate end-to-end gradient smoke artifacts for the fixed-branch JAX/IFT path."""
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
    from model.end_to_end_gradient_smoke import (
        END_TO_END_GRADIENT_SMOKE_COLUMNS,
        gradient_smoke_markdown,
        run_heat_loss_two_cell_ift_smoke,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--theta0", type=float, default=2.0)
    parser.add_argument("--target-delta-k", type=float, default=-3.0)
    parser.add_argument("--out-dir", default=str(repo_root / "outputs" / "diagnostics"))
    parser.add_argument("--basename", default="end_to_end_gradient_smoke")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = run_heat_loss_two_cell_ift_smoke(
        theta0=args.theta0,
        learning_rate=args.learning_rate,
        n_steps=args.steps,
        target_delta_k=args.target_delta_k,
    )
    table = pd.DataFrame(rows, columns=list(END_TO_END_GRADIENT_SMOKE_COLUMNS))

    csv_path = out_dir / f"{args.basename}.csv"
    md_path = out_dir / f"{args.basename}.md"
    table.to_csv(csv_path, index=False)
    md_path.write_text(gradient_smoke_markdown(rows), encoding="utf-8")

    initial_loss = float(table.iloc[0]["loss"])
    final_loss = float(table.iloc[-1]["next_loss"])
    print(f"wrote_csv={csv_path}")
    print(f"wrote_markdown={md_path}")
    print(f"steps={len(table)}")
    print(f"initial_loss={initial_loss:.12g}")
    print(f"final_loss={final_loss:.12g}")
    print(f"overall_decrease={final_loss < initial_loss}")
    print(f"per_step_direction_passed={bool(table['direction_check_passed'].all())}")


if __name__ == "__main__":
    main()
