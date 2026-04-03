#!/usr/bin/env python3
"""Precompile `jax_jit` for representative cases/shapes so services can warm up on startup."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Iterable, List

# 仓库根目录：供 `from scripts....` 解析；src：供 `model` 包（src 优先）
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_SRC, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model.chemistry import COAL_DATABASE
from model.validation_loader import get_validation_cases_final_path, iter_validation_cases_final

try:
    import jax  # noqa: F401
    import jaxlib  # noqa: F401
except Exception as exc:
    print(f"[BLOCKED] JAX runtime unavailable: {type(exc).__name__}: {exc}")
    raise SystemExit(1)

from scripts.compare_sim_minimize_vs_jax_jit_all_cases import _build_system


def _load_cases_map() -> Dict[str, Dict]:
    path = get_validation_cases_final_path()
    with open(path, "r", encoding="utf-8") as fp:
        raw = json.load(fp)
    return {name: case for name, case in iter_validation_cases_final(raw)}


def _resolve_case_names(available: Iterable[str], requested: Iterable[str]) -> List[str]:
    choices = []
    for name in requested:
        if name not in available:
            raise ValueError(f"Unknown case '{name}' (available: {sorted(available)})")
        choices.append(name)
    return choices


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompile jax_jit solver for representative cases/shapes")
    parser.add_argument(
        "--cases",
        type=str,
        nargs="+",
        default=["Paper_Case_6", "Texaco_I-2", "Coal_Water_Slurry_Western"],
        help="Case names from data/validation_cases_final.json",
    )
    parser.add_argument("--n-cells", type=int, nargs="+", default=[20], help="Mesh sizes to precompile")
    parser.add_argument("--jax-warmup", action="store_true", help="Run the lightweight JAX warmup before compiling")
    args = parser.parse_args()

    cases_map = _load_cases_map()
    case_names = _resolve_case_names(cases_map.keys(), args.cases)

    if args.jax_warmup:
        from model.jax_solver import warmup_jax

        warmup_jax()

    durations: List[float] = []
    for nc in args.n_cells:
        for name in case_names:
            case_data = cases_map[name]
            sys_obj = _build_system(name, case_data, COAL_DATABASE, cases_pressure=0.0)
            print(f"Precompiling {name} with {nc} cells...", flush=True)
            start = time.perf_counter()
            sys_obj.solve(N_cells=nc, solver_method="jax_jit", jax_warmup=False)
            elapsed = time.perf_counter() - start
            durations.append(elapsed)
            print(f"  done in {elapsed:.2f}s/shape", flush=True)

    total = sum(durations)
    print("Precompile summary:")
    print(f"  shapes: {len(args.n_cells) * len(case_names)}")
    print(f"  total time: {total:.2f}s")


if __name__ == "__main__":
    main()
