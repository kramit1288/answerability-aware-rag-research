"""Validate the Phase 3.6b decision checkpoint without opening answer-key values."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from answerability_rag.hashing import canonical_json_sha256, sha256_file
from answerability_rag.sufficiency.rescue import candidate_grid, final_prediction


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str, checks: list[str]) -> None:
    if not condition:
        raise AssertionError(message)
    checks.append(message)


def main() -> None:
    checks: list[str] = []
    grid = json.loads((ROOT / "configs/phase03_rescue_grid.json").read_text(encoding="utf-8"))
    freeze = json.loads((ROOT / "artifacts/results/phase03_rescue_candidate_grid_freeze.json").read_text(encoding="utf-8"))
    require(sha256_file(ROOT / "configs/phase03_rescue_grid.json") == freeze["candidate_grid_physical_sha256"],
            "candidate grid physical hash matches its pre-evaluation freeze", checks)
    require(canonical_json_sha256(grid) == freeze["candidate_grid_canonical_sha256"],
            "candidate grid canonical hash matches its pre-evaluation freeze", checks)
    candidates = candidate_grid(grid)
    require(len(candidates) == 59, "candidate grid contains exactly 59 rules", checks)
    require(freeze["frozen_before_development_label_evaluation"] is True,
            "candidate grid was frozen before label evaluation", checks)

    expected_annotations = {
        "phase03_annotation_annotator_1.csv": ("04c0f33db4ca3ac9d2a58322f6399c1688ef66a48855bbb7c7546ca7d5851959", 86, 64, "sufficient"),
        "phase03_annotation_annotator_1_adjudicated.csv": ("3e7a87f2da694cbb40930c142549cd70dde86c0d89429885711549cf91b99cc4", 85, 65, "insufficient"),
    }
    for name, (digest, sufficient, insufficient, q217) in expected_annotations.items():
        path = ROOT / "artifacts/results" / name
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        counts = Counter(row["manual_label"] for row in rows)
        changed = [row for row in rows if row["question_id"] == "DEV_Q217"]
        require(sha256_file(path) == digest, f"{name} provenance SHA-256 is correct", checks)
        require(len(rows) == 150 and counts == {"sufficient": sufficient, "insufficient": insufficient},
                f"{name} has its frozen label counts", checks)
        require(len(changed) == 1 and changed[0]["manual_label"] == q217,
                f"{name} has the correct DEV_Q217 provenance", checks)

    report = json.loads((ROOT / "artifacts/results/phase03_rescue_development_report.json").read_text(encoding="utf-8"))
    require(report["candidate_grid_sha256"] == freeze["candidate_grid_canonical_sha256"],
            "development report references the frozen grid", checks)
    require(report["candidate_count"] == 59, "all and only 59 candidates were evaluated", checks)
    require(report["development_gates_passed"] is False,
            "development checkpoint records the frozen gate failure", checks)
    selected = report["selected"]
    require(selected["sample_precision"] >= 0.90 and selected["weighted_f1"] < 0.85,
            "selected eligible rule passed precision but failed weighted-F1 gate", checks)
    require(selected["fp"] == 0 and selected["fn"] == 25,
            "persisted decision confusion matrix is internally expected", checks)
    require(report["test_outcomes_sealed"] is True and report["phase4_started"] is False,
            "TEST and Phase 4 seals remain active", checks)

    row = {"y_suff_strict": 1, "maximum_span_coverage_fraction": 0.0,
           "mean_claim_entailment": 0.0, "minimum_claim_entailment": 0.0,
           "maximum_selected_premise_contradiction": 1.0}
    require(all(final_prediction(row, candidate) == 1 for candidate in candidates),
            "no frozen candidate can demote a strict positive", checks)
    require(grid["test_seal"] == {"allow_test_load": False, "allow_test_aggregate_outcomes": False},
            "TEST cannot influence rescue selection", checks)

    semantic_governance = json.loads((ROOT / "artifacts/data/phase03_semantic_column_governance.json").read_text(encoding="utf-8"))
    require(semantic_governance["future_classifier_feature_allowlist"] == ["k", "retrieval_strategy"],
            "gold/evidence/NLI label-construction columns remain forbidden inference features", checks)
    supersession = json.loads((ROOT / "artifacts/results/phase03_semantic_confirmation_supersession.json").read_text(encoding="utf-8"))
    require(supersession["prior_confirmation_status"] == "superseded_unannotated"
            and supersession["answer_key_values_opened"] is False
            and supersession["human_annotations_present"] is False,
            "prior semantic-only confirmation is superseded, unannotated, and unopened", checks)

    forbidden_outputs = [
        ROOT / "configs/phase03_final_target.json",
        ROOT / "artifacts/data/context_sufficiency_final_labels.parquet",
        ROOT / "artifacts/data/phase03_final_primary_target.parquet",
        ROOT / "artifacts/results/phase03_final_confirmation_template.csv",
        ROOT / "artifacts/results/phase03_final_confirmation_blinded.parquet",
        ROOT / "artifacts/results/phase03_final_confirmation_answer_key.parquet",
        ROOT / "artifacts/results/phase03_final_confirmation_sample_manifest.csv",
    ]
    if any(path.exists() for path in forbidden_outputs):
        successor_path = ROOT / "artifacts/results/phase03_rescue_expanded_development_report.json"
        require(successor_path.exists(),
                "later final artifacts are attributable to a separately frozen successor iteration", checks)
        successor = json.loads(successor_path.read_text(encoding="utf-8"))
        require(successor["development_gates_passed"] is True
                and successor["final_automatic_threshold_refinement_iteration"] is True,
                "Phase 3.6c, not failed Phase 3.6b, produced the later final artifacts", checks)
    else:
        require(True, "gate failure produced no final target or replacement confirmation sample", checks)
    require((ROOT / "docs/PHASE_03_6B_DECISION_REQUEST.md").exists(),
            "gate failure produced the required decision request", checks)
    print(json.dumps({"status": "pass", "checks": checks,
                      "candidate_grid_sha256": freeze["candidate_grid_canonical_sha256"],
                      "candidate_count": 59, "development_gates_passed": False,
                      "test_outcomes_sealed": True, "phase4_started": False}, indent=2))


if __name__ == "__main__":
    main()
