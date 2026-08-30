"""Check the completed and frozen final Phase 3 human confirmation."""

from __future__ import annotations

import json
from pathlib import Path

from answerability_rag.hashing import canonical_json_sha256, sha256_file
from answerability_rag.sufficiency.final_confirmation import (
    EXPECTED_THRESHOLDS,
    FINAL_CONFIG_SHA256,
    evaluate_confirmation,
    validate_confirmation,
)


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str, checks: list[str]) -> None:
    if not condition:
        raise AssertionError(message)
    checks.append(message)


def main() -> None:
    checks: list[str] = []
    validation, human = validate_confirmation(ROOT)
    require(validation["status"] == "pass" and validation["answer_key_opened"] is False,
            "completed human CSV passes the pre-key validation boundary", checks)
    persisted_validation = json.loads((ROOT / "artifacts/results/phase03_final_confirmation_prekey_validation.json").read_text(encoding="utf-8"))
    require(persisted_validation == validation,
            "persisted pre-key validation reproduces exactly", checks)

    evaluation, disagreements, population = evaluate_confirmation(ROOT, validation, human)
    persisted_evaluation = json.loads((ROOT / "artifacts/results/phase03_final_confirmation_evaluation.json").read_text(encoding="utf-8"))
    persisted_population = json.loads((ROOT / "artifacts/results/phase03_final_target_population_manifest.json").read_text(encoding="utf-8"))
    require(persisted_evaluation == evaluation and persisted_population == population,
            "confirmation metrics and target population reproduce exactly", checks)
    require(evaluation["gates"]["both_passed"] is True
            and evaluation["precision"] >= 0.90 and evaluation["weighted_f1"] >= 0.85,
            "both unchanged independent-confirmation gates pass", checks)
    require(evaluation["confusion_matrix"] == {"tn": 13, "fp": 3, "fn": 3, "tp": 31}
            and evaluation["human_label_counts"] == {"sufficient": 34, "insufficient": 16, "ambiguous": 0},
            "human counts and confusion matrix reproduce", checks)
    require(len(disagreements) == 6 and evaluation["error_type_counts"] == {"false_negative": 3, "false_positive": 3},
            "six frozen disagreements reproduce", checks)

    config = json.loads((ROOT / "configs/phase03_final_target.json").read_text(encoding="utf-8"))
    stored = config.pop("final_target_config_sha256")
    require(stored == FINAL_CONFIG_SHA256 == canonical_json_sha256(config)
            and config["selected_thresholds"] == EXPECTED_THRESHOLDS
            and config["strict_positive_demotion_permitted"] is False,
            "frozen strict-preserving final rule and SHA reproduce", checks)
    require(population["eligible_questions"] == 513 and population["eligible_conditions"] == 6156
            and population["final_positive_conditions"] == 2802
            and population["final_negative_conditions"] == 3354,
            "frozen PRIMARY target census reproduces", checks)
    require(population["exclusions"]["semantic_unevaluable"]["question_ids"] == ["DEV_Q066", "TRAIN_Q526"]
            and population["exclusions"]["benchmark_impossible"]["primary_use"] is False,
            "semantic-unevaluable and benchmark-impossible primary exclusions remain frozen", checks)

    status = json.loads((ROOT / "artifacts/results/phase03_final_confirmation_evaluation_status.json").read_text(encoding="utf-8"))
    require(status["status"] == "passed_phase03_complete"
            and status["test_outcomes_sealed"] is True and status["phase4_started"] is False,
            "Phase 3 is complete while TEST remains sealed and Phase 4 unstarted", checks)
    manifest = json.loads((ROOT / "artifacts/results/phase03_final_artifact_manifest.json").read_text(encoding="utf-8"))
    require(manifest["phase03_status"] == "complete_pending_phase4_review"
            and all(sha256_file(ROOT / item["path"]) == item["physical_sha256"]
                    for item in manifest["artifacts"].values()),
            "all final Phase 3 artifact hashes reproduce", checks)
    print(json.dumps({"status": "pass", "checks": checks,
                      "precision": evaluation["precision"],
                      "weighted_f1": evaluation["weighted_f1"],
                      "final_target_config_sha256": FINAL_CONFIG_SHA256,
                      "test_outcomes_sealed": True, "phase4_started": False}, indent=2))


if __name__ == "__main__":
    main()
