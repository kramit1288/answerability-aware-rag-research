"""Construct all frozen Phase 3 automatic labels and manual-validation material."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from answerability_rag.sufficiency.config import Phase03Config
from answerability_rag.sufficiency.pipeline import run_phase03


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/phase03_sufficiency.json"))
    arguments = parser.parse_args()
    manifest = run_phase03(Phase03Config.load(ROOT / arguments.config), ROOT)
    print(f"Phase 3 automatic construction complete: {manifest['run_id']}")


if __name__ == "__main__":
    main()
