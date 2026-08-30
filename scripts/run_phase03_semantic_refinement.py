"""Run Phase 3.6 semantic refinement and stop after the blinded confirmation pack."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from answerability_rag.sufficiency.semantic import Phase03SemanticConfig
from answerability_rag.sufficiency.semantic_pipeline import run_phase03_semantic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/phase03_semantic.json"))
    arguments = parser.parse_args()
    result = run_phase03_semantic(Phase03SemanticConfig.load(ROOT / arguments.config), ROOT)
    print(
        f"Phase 3.6 stopped for human confirmation: {result['run_id']} "
        f"template=artifacts/results/phase03_semantic_confirmation_template.csv"
    )


if __name__ == "__main__":
    main()
