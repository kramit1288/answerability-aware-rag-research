"""Reproduce the Phase 3 blinded manual-annotation pack from frozen inputs."""

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
    print(f"Prepared {manifest['manual_sample_rows']} blinded Phase 3 annotation rows")


if __name__ == "__main__":
    main()
