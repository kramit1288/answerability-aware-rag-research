"""Validate Phase 7 upstream immutability and no-post-TEST-tuning invariants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from answerability_rag.final_test.integrity import write_integrity_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    result = write_integrity_report(
        ROOT, ROOT / "configs/phase07_final_test.json", require_complete=not args.allow_incomplete,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
