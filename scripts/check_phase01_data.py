"""Validate persisted Phase 1 artifacts without regenerating or altering them."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from answerability_rag.config import load_phase01_config  # noqa: E402
from answerability_rag.data.integrity import validate_persisted_phase01  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs/phase01_data.json")
    args = parser.parse_args()
    config = load_phase01_config(args.config)
    report = validate_persisted_phase01(config.artifact_root)
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    report.require_pass()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
