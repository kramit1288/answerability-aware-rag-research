"""Run Phase 3.6b and stop at its decision or human-confirmation checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from answerability_rag.sufficiency.rescue_pipeline import run_phase03_rescue


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = run_phase03_rescue(args.root.resolve())
    print(json.dumps({
        "candidate_grid_sha256": result["candidate_grid_sha256"],
        "candidate_count": result["candidate_count"],
        "development_gates_passed": result["development_gates_passed"],
        "selected": result["selected"],
        "final_target_config_sha256": result.get("final_target_config_sha256"),
        "primary_population": result.get("primary_population"),
        "confirmation": result.get("confirmation"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
