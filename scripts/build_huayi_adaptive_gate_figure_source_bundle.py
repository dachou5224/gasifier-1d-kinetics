#!/usr/bin/env python3
"""Build Huayi adaptive-gate figure/table source-data bundle."""
from __future__ import annotations

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
    from model.adaptive_gate_figure_source import build_adaptive_gate_figure_source_bundle

    result = build_adaptive_gate_figure_source_bundle(repo_root)
    for key in (
        "fig4_metric_matrix",
        "fig4_method_ladder",
        "channel_status",
        "reaction_family_gate_map",
        "alpha_similarity",
        "multialpha_activation",
        "manifest",
        "markdown",
    ):
        print(f"wrote_{key}={result[key]}")
    print(f"row_counts={result['row_counts']}")
    print(f"method_labels={result['method_labels']}")
    print(f"channel_statuses={result['channel_statuses']}")
    print(f"not_available_fields={result['not_available_fields']}")


if __name__ == "__main__":
    main()
