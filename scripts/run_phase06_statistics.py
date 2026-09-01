"""Thin entry point for the frozen Phase 6 statistical analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from answerability_rag.statistics.pipeline import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase06_statistics.json")
    args = parser.parse_args()
    print(json.dumps(run(ROOT, ROOT / args.config), indent=2))


if __name__ == "__main__":
    main()
