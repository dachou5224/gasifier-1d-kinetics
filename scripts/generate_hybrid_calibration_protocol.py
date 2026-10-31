#!/usr/bin/env python3
"""Generate the round40 hybrid calibration protocol artifacts."""
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
    from model.hybrid_calibration_protocol import write_protocol_artifacts

    json_path, md_path, protocol = write_protocol_artifacts(repo_root)
    audit = protocol["readiness_audit"]
    print(f"wrote_json={json_path}")
    print(f"wrote_markdown={md_path}")
    print(f"protocol_version={protocol['protocol_version']}")
    print(f"huayi_ready={audit['huayi_ready_model_cases']}/{audit['huayi_contract_rows']}")
    print(f"candidate_alpha_groups={len(protocol['candidate_alpha_groups'])}")
    print(f"baseline_count={len(protocol['baseline_matrix'])}")
    print(f"claim_boundary={protocol['claim_boundary']}")


if __name__ == "__main__":
    main()
