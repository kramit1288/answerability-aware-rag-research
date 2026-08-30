"""Run final Phase 3.6c automatic rescue refinement and stop at its frozen checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from answerability_rag.sufficiency.rescue_expanded_pipeline import run_phase03_rescue_expanded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = run_phase03_rescue_expanded(args.root.resolve())
    print(json.dumps({
        "expanded_grid_sha256": result["expanded_grid_sha256"],
        "candidate_count": result["candidate_count"],
        "new_candidate_count": result["new_candidate_count"],
        "development_gates_passed": result["development_gates_passed"],
        "selected": result["selected"],
        "precision_wilson_95": result["precision_wilson_95"],
        "final_target_config_sha256": result.get("final_target_config_sha256"),
        "primary_population": result.get("primary_population"),
        "confirmation": result.get("confirmation"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
