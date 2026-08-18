"""Thin entry point for the frozen Phase 2 controlled-retrieval experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from answerability_rag.retrieval.config import Phase02Config
from answerability_rag.retrieval.pipeline import run_phase02


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/phase02_retrieval.json"))
    arguments = parser.parse_args()
    root = ROOT
    manifest = run_phase02(Phase02Config.load(root / arguments.config), root)
    print(f"Phase 2 complete: {manifest['run_id']}")


if __name__ == "__main__":
    main()
