"""Run one resumable Phase 5 stage without opening TechQA TEST."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from answerability_rag.generation.pipeline import run_stage  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase05_generation_grounding.json")
    parser.add_argument(
        "--stage", required=True,
        choices=("preflight", "contexts", "generate", "quality", "ragtruth", "techqa_grounding", "analysis"),
    )
    args = parser.parse_args()
    result = run_stage(ROOT, ROOT / args.config, args.stage)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
