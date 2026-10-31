#!/usr/bin/env python3
"""Build paper-facing Huayi hybrid diagnostic evidence bundle."""
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
    from model.hybrid_diagnostic_evidence_bundle import build_hybrid_diagnostic_evidence_bundle

    result = build_hybrid_diagnostic_evidence_bundle(repo_root)
    print(f"wrote_manifest={result['manifest']}")
    print(f"wrote_metric_matrix={result['metric_matrix']}")
    print(f"wrote_claim_table={result['claim_table']}")
    print(f"wrote_recommendations={result['recommendations']}")
    print(f"wrote_markdown={result['markdown']}")
    print(f"method_labels={result['method_labels']}")


if __name__ == "__main__":
    main()
