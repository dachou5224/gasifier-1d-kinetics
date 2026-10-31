#!/usr/bin/env python3
"""Generate IFT/custom-VJP vs direct-unrolled AD cost diagnostics."""
from __future__ import annotations

import argparse
import multiprocessing as mp
import queue
import sys
from pathlib import Path

import pandas as pd


def _ensure_src_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return repo_root


def _worker(method: str, n_cells: int, newton_iterations: int, warm_repeats: int, out_q) -> None:
    _ensure_src_path()
    from model.ift_cost_diagnostic import failed_row, run_cost_case

    try:
        out_q.put(run_cost_case(method, n_cells, newton_iterations=newton_iterations, warm_repeats=warm_repeats))
    except Exception as exc:  # pragma: no cover - exercised only by diagnostic failures
        out_q.put(failed_row(method, n_cells, newton_iterations=newton_iterations, failure_reason=f"{type(exc).__name__}: {exc}"))


def _run_with_timeout(method: str, n_cells: int, newton_iterations: int, warm_repeats: int, timeout_s: float) -> dict:
    from model.ift_cost_diagnostic import timeout_row

    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(method, n_cells, newton_iterations, warm_repeats, out_q))
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join(10)
        return timeout_row(method, n_cells, newton_iterations=newton_iterations, timeout_s=timeout_s)
    try:
        return out_q.get_nowait()
    except queue.Empty:
        return timeout_row(method, n_cells, newton_iterations=newton_iterations, timeout_s=timeout_s)


def main() -> None:
    repo_root = _ensure_src_path()
    from model.ift_cost_diagnostic import (
        IFT_COST_COLUMNS,
        METHOD_IFT,
        METHOD_UNROLLED,
        annotate_reference_differences,
        collect_protocol_metadata,
        ift_cost_markdown,
        ift_cost_protocol_markdown,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--ift-n-cells", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--unrolled-n-cells", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--newton-iterations", type=int, default=8)
    parser.add_argument("--warm-repeats", type=int, default=2)
    parser.add_argument("--timeout-s", type=float, default=90.0)
    parser.add_argument("--out-dir", default=str(repo_root / "outputs" / "diagnostics"))
    parser.add_argument("--basename", default="ift_cost_diagnostic")
    parser.add_argument("--protocol-only", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = out_dir / "ift_cost_protocol.md"
    protocol_text = ift_cost_protocol_markdown(
        collect_protocol_metadata(timeout_s=args.timeout_s, warm_repeats=args.warm_repeats)
    )
    protocol_path.write_text(protocol_text, encoding="utf-8")
    if args.protocol_only:
        print(f"wrote_protocol={protocol_path}")
        return

    rows = []
    for n_cells in args.ift_n_cells:
        rows.append(_run_with_timeout(METHOD_IFT, int(n_cells), args.newton_iterations, args.warm_repeats, args.timeout_s))
    for n_cells in args.unrolled_n_cells:
        rows.append(_run_with_timeout(METHOD_UNROLLED, int(n_cells), args.newton_iterations, args.warm_repeats, args.timeout_s))
    rows = annotate_reference_differences(rows)

    csv_path = out_dir / f"{args.basename}.csv"
    md_path = out_dir / f"{args.basename}.md"
    pd.DataFrame(rows, columns=list(IFT_COST_COLUMNS)).to_csv(csv_path, index=False)
    md_path.write_text(ift_cost_markdown(rows), encoding="utf-8")

    table = pd.DataFrame(rows)
    print(f"wrote_csv={csv_path}")
    print(f"wrote_markdown={md_path}")
    print(f"wrote_protocol={protocol_path}")
    print(f"rows={len(table)}")
    print("status_counts=" + ",".join(f"{k}:{v}" for k, v in table["status"].value_counts().sort_index().items()))
    ok = table[table["status"] == "ok"]
    if not ok.empty:
        print(f"ok_methods={','.join(sorted(ok['method'].unique()))}")
        print(f"max_cold_time_s={ok['compile_or_cold_time_s'].max():.12g}")
        print(f"max_warm_time_s={ok['warm_eval_time_s'].max():.12g}")


if __name__ == "__main__":
    main()
