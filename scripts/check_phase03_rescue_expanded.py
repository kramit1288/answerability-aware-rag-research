"""Validate the Phase 3.6c final-label and blinded-confirmation checkpoint."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from answerability_rag.hashing import canonical_json_sha256, sha256_file
from answerability_rag.sufficiency.rescue import final_prediction
from answerability_rag.sufficiency.rescue_expanded import expanded_candidate_grid, wilson_interval


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str, checks: list[str]) -> None:
    if not condition:
        raise AssertionError(message)
    checks.append(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    checks: list[str] = []
    grid_path = ROOT / "configs/phase03_rescue_grid_expanded.json"
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    freeze = json.loads((ROOT / "artifacts/results/phase03_rescue_expanded_candidate_grid_freeze.json").read_text(encoding="utf-8"))
    require(sha256_file(grid_path) == freeze["candidate_grid_physical_sha256"],
            "expanded grid physical hash matches pre-outcome freeze", checks)
    require(canonical_json_sha256(grid) == freeze["candidate_grid_canonical_sha256"],
            "expanded grid canonical hash matches pre-outcome freeze", checks)
    candidates = expanded_candidate_grid(grid)
    require(len(candidates) == 77 and sum(row["new_phase03_6c_candidate"] for row in candidates) == 74,
            "expanded grid has exactly 77 total and 74 new candidates", checks)
    require(freeze["frozen_before_development_outcome_calculation"] is True,
            "expanded grid existed before development outcome calculation", checks)
    require(grid["iteration_governance"]["final_automatic_threshold_refinement_iteration"] is True
            and grid["iteration_governance"]["allow_further_autonomous_threshold_search"] is False,
            "final automatic-refinement hard stop is frozen", checks)

    prior = grid["frozen_phase03_6b"]
    prior_files = {
        "candidate_grid_physical_sha256": ROOT / "configs/phase03_rescue_grid.json",
        "candidate_results_physical_sha256": ROOT / "artifacts/results/phase03_rescue_candidate_results.csv",
        "development_report_physical_sha256": ROOT / "artifacts/results/phase03_rescue_development_report.json",
        "artifact_manifest_physical_sha256": ROOT / "artifacts/results/phase03_rescue_artifact_manifest.json",
    }
    require(all(sha256_file(path) == prior[key] for key, path in prior_files.items()),
            "all frozen Phase 3.6b artifacts remain byte-identical", checks)

    report = json.loads((ROOT / "artifacts/results/phase03_rescue_expanded_development_report.json").read_text(encoding="utf-8"))
    selected = report["selected"]
    require(report["development_gates_passed"] is True
            and selected["sample_precision"] >= 0.90 and selected["weighted_f1"] >= 0.85,
            "selected candidate passes both unchanged development gates", checks)
    expected_interval = wilson_interval(selected["tp"], selected["tp"] + selected["fp"])
    require(report["precision_wilson_95"] == expected_interval,
            "selected precision Wilson interval recomputes exactly", checks)
    strict_row = {"y_suff_strict": 1, "maximum_span_coverage_fraction": 0.0,
                  "mean_claim_entailment": 0.0, "minimum_claim_entailment": 0.0,
                  "maximum_selected_premise_contradiction": 1.0}
    require(all(final_prediction(strict_row, candidate) == 1 for candidate in candidates),
            "strict positives cannot be demoted by any expanded candidate", checks)

    config_path = ROOT / "configs/phase03_final_target.json"
    final_config = json.loads(config_path.read_text(encoding="utf-8"))
    final_hash = final_config.pop("final_target_config_sha256")
    require(canonical_json_sha256(final_config) == final_hash == report["final_target_config_sha256"],
            "final target configuration SHA-256 recomputes exactly", checks)
    require(final_config["test_outcomes_sealed"] is True and final_config["phase4_started"] is False,
            "final target config keeps TEST sealed and Phase 4 unstarted", checks)

    final_rows = pq.read_table(ROOT / "artifacts/data/context_sufficiency_final_labels.parquet").to_pylist()
    primary_rows = pq.read_table(ROOT / "artifacts/data/phase03_final_primary_target.parquet").to_pylist()
    require(len(final_rows) == 6180 and len(primary_rows) == 6156,
            "final labels retain 6180 rows and primary export contains 6156 eligible rows", checks)
    require(not any(row["split"] == "test" for row in final_rows + primary_rows),
            "no TEST row enters Phase 3.6c labels or primary target", checks)
    require(not any(row["y_suff_strict"] == 1 and row["y_suff_final"] != 1
                    for row in final_rows if row["y_suff_final"] is not None),
            "final label artifact never demotes a strict positive", checks)
    unevaluable = [row for row in final_rows if not row["primary_phase4_target_eligible"]]
    require(len(unevaluable) == 24 and {row["question_id"] for row in unevaluable} == {"DEV_Q066", "TRAIN_Q526"}
            and all(row["y_suff_final"] is None for row in unevaluable),
            "semantic-unevaluable questions remain NA and excluded", checks)
    primary_ids = {row["example_id"] for row in primary_rows}
    strict_source = {row["example_id"]: row for row in pq.read_table(
        ROOT / "artifacts/data/context_sufficiency_labels.parquet",
        filters=[("split", "in", ["train", "validation"])],
    ).to_pylist()}
    require(all(strict_source[example_id]["benchmark_is_impossible"] is False
                and strict_source[example_id]["y_suff"] is not None
                for example_id in primary_ids),
            "benchmark-impossible and unresolved rows cannot enter primary target", checks)

    governance = json.loads((ROOT / "artifacts/data/phase03_semantic_column_governance.json").read_text(encoding="utf-8"))
    require(governance["future_classifier_feature_allowlist"] == ["k", "retrieval_strategy"],
            "label-only evidence and NLI fields remain blocked from Phase 4 features", checks)

    original = read_csv(ROOT / "artifacts/results/phase03_manual_sample_manifest.csv")
    previous = read_csv(ROOT / "artifacts/results/phase03_semantic_confirmation_sample_manifest.csv")
    manifest = read_csv(ROOT / "artifacts/results/phase03_final_confirmation_sample_manifest.csv")
    template = read_csv(ROOT / "artifacts/results/phase03_final_confirmation_template.csv")
    blinded = pq.read_table(ROOT / "artifacts/results/phase03_final_confirmation_blinded.parquet").to_pylist()
    original_questions = {row["question_id"] for row in original}
    previous_questions = {row["question_id"] for row in previous}
    sample_questions = {row["question_id"] for row in manifest}
    require(len(manifest) == len(template) == len(blinded) == 50 and len(sample_questions) == 50,
            "final confirmation contains exactly 50 unique questions", checks)
    require(not sample_questions & original_questions and not sample_questions & previous_questions,
            "final confirmation has zero overlap with both prior reviewed-question sets", checks)
    require(not sample_questions & {row["question_id"] for row in unevaluable},
            "final confirmation contains no semantic-unevaluable question", checks)
    require(not any(row["split"] == "test" for row in manifest),
            "final confirmation contains no TEST row", checks)
    require(all(strict_source[row["example_id"]]["benchmark_is_impossible"] is False for row in manifest),
            "final confirmation contains no benchmark-impossible row", checks)
    forbidden = {"y_suff_strict", "y_suff_semantic", "y_suff_final",
                 "maximum_span_coverage_fraction", "minimum_claim_entailment",
                 "mean_claim_entailment", "maximum_selected_premise_contradiction",
                 "model_prediction", "confirmation_sampling_stratum", "example_id",
                 "final_target_config_sha256"}
    require(not forbidden & set(template[0]) and not forbidden & set(blinded[0]),
            "blinded confirmation views expose no label, score, prediction, or answer-key field", checks)
    require(all(not row["annotator_id"] and not row["manual_label"] and not row["rationale"]
                and not row["annotation_timestamp"] for row in template),
            "final confirmation human fields remain blank", checks)
    answer_key = pq.ParquetFile(ROOT / "artifacts/results/phase03_final_confirmation_answer_key.parquet")
    require(answer_key.metadata.num_rows == 50,
            "separate sealed answer key has 50 rows without reading its values", checks)
    status = json.loads((ROOT / "artifacts/results/phase03_final_confirmation_status.json").read_text(encoding="utf-8"))
    require(status["status"] == "pending_unannotated"
            and status["answer_key_evaluation_status"] == "sealed_until_complete_genuine_human_confirmation",
            "final confirmation is pending, unannotated, and answer-key sealed", checks)
    supersession = json.loads((ROOT / "artifacts/results/phase03_semantic_confirmation_supersession.json").read_text(encoding="utf-8"))
    require(supersession["prior_confirmation_status"] == "superseded_unannotated"
            and supersession["answer_key_values_opened"] is False,
            "superseded semantic-only confirmation remains unused", checks)

    available_questions = {row["question_id"]: row for row in final_rows
        if row["primary_phase4_target_eligible"]
        and row["question_id"] not in original_questions
        and row["question_id"] not in previous_questions}
    available_splits = Counter(row["split"] for row in available_questions.values())
    require(available_splits.get("validation", 0) == 0 and Counter(row["split"] for row in manifest) == {"train": 50},
            "all-train confirmation split is the only feasible disjoint allocation", checks)

    print(json.dumps({"status": "pass", "checks": checks,
        "expanded_grid_sha256": freeze["candidate_grid_canonical_sha256"],
        "candidate_count": 77, "selected_candidate_order": selected["candidate_order"],
        "final_target_config_sha256": final_hash,
        "primary_conditions": len(primary_rows), "confirmation_rows": len(manifest),
        "confirmation_strata": dict(Counter(row["confirmation_sampling_stratum"] for row in manifest)),
        "available_question_splits_after_disjoint_exclusions": dict(available_splits),
        "test_outcomes_sealed": True, "phase4_started": False}, indent=2))


if __name__ == "__main__":
    main()
