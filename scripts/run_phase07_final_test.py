"""Thin CLI for the frozen Phase 7 final TEST workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from answerability_rag.final_test.evaluation import (
    run_benchmark_impossible_sensitivity,
    run_test_inference,
    write_test_feature_manifest,
)
from answerability_rag.final_test.analysis import build_test_policy_views, run_test_statistics
from answerability_rag.final_test.generation import (
    assemble_test_contexts,
    evaluate_test_quality,
    run_test_generation,
)
from answerability_rag.final_test.grounding import evaluate_test_grounding
from answerability_rag.final_test.reporting import run_final_reporting
from answerability_rag.final_test.target import construct_test_target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/phase07_final_test.json"))
    parser.add_argument(
        "--stage", choices=("target", "inference", "feature-manifest", "sensitivity", "contexts", "generation", "quality", "grounding", "policy-views", "statistics", "reporting"),
        default="target",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    stages = {
        "target": construct_test_target, "inference": run_test_inference,
        "feature-manifest": write_test_feature_manifest,
        "sensitivity": run_benchmark_impossible_sensitivity,
        "contexts": assemble_test_contexts, "generation": run_test_generation,
        "quality": evaluate_test_quality, "grounding": evaluate_test_grounding,
        "policy-views": build_test_policy_views, "statistics": run_test_statistics,
        "reporting": run_final_reporting,
    }
    result = stages[args.stage](root, root / args.config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
