#!/usr/bin/env python3
"""Build figure/table-ready CSVs from existing diagnostic artifacts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_src_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return repo_root


def main() -> None:
    repo_root = _ensure_src_path()
    from model.figure_data_bundle import build_figure_data_bundle

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(repo_root / "outputs" / "figures_data"))
    args = parser.parse_args()

    manifest = build_figure_data_bundle(repo_root=repo_root, out_dir=args.out_dir)
    print(f"output_directory={manifest['output_directory']}")
    for name, entry in manifest["artifacts"].items():
        print(f"{name}: rows={entry['row_count']} file={entry['file']}")


if __name__ == "__main__":
    main()
