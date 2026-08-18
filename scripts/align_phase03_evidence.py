"""Run the frozen Phase 3 evidence-alignment audit and 90% development gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from answerability_rag.sufficiency.config import Phase03Config
from answerability_rag.sufficiency.pipeline import run_alignment_stage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/phase03_sufficiency.json"))
    arguments = parser.parse_args()
    result = run_alignment_stage(Phase03Config.load(ROOT / arguments.config), ROOT)
    print(json.dumps(result["feasibility"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
