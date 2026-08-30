"""Validate frozen Phase 3.6 outputs without reading confirmation answer-key values."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from answerability_rag.hashing import canonical_json_sha256, sha256_file
from answerability_rag.retrieval.artifacts import read_parquet_records
from answerability_rag.sufficiency.semantic import Phase03SemanticConfig


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _require(condition: bool, message: str, checks: dict[str, str], key: str) -> None:
    checks[key] = "pass" if condition else f"fail: {message}"
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/phase03_semantic.json"))
    args = parser.parse_args()
    config = Phase03SemanticConfig.load(ROOT / args.config)
    checks: dict[str, str] = {}

    selected_path = ROOT / "artifacts/results/phase03_semantic_selected_config.json"
    results_path = ROOT / "artifacts/results/phase03_semantic_refinement_results.json"
    hashes_path = ROOT / "artifacts/results/phase03_semantic_artifact_hashes.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
    label_governance = json.loads((
        ROOT / "configs/phase03_semantic_label_governance.json"
    ).read_text(encoding="utf-8"))
    label_governance_sha256 = canonical_json_sha256(label_governance)

    _require(
        selected["base_semantic_config_sha256"] == config.config_sha256,
        "selected method does not reference the current frozen base config", checks,
        "base_config_hash",
    )
    _require(
        selected["frozen_before_confirmation_sampling"] is True,
        "semantic method was not marked frozen before confirmation sampling", checks,
        "selection_before_sampling",
    )
    _require(
        selected["semantic_config_sha256"] == results["semantic_config_sha256"]
        == hashes["semantic_config_sha256"],
        "semantic configuration hashes disagree", checks, "semantic_config_hash",
    )
    _require(
        results["semantic_label_governance_sha256"]
        == hashes["semantic_label_governance_sha256"]
        == label_governance_sha256,
        "semantic label governance hashes disagree", checks,
        "label_governance_hash",
    )
    _require(
        label_governance["semantic_scoring_configuration"]["sha256"]
        == selected["semantic_config_sha256"]
        and results["semantic_configuration_relationship"][
            "scoring_configuration_changed_by_governance"
        ] is False,
        "label governance does not preserve the frozen scoring configuration", checks,
        "scoring_governance_relationship",
    )
    _require(
        results["test_semantic_scores_calculated"] is False
        and results["test_semantic_aggregates_calculated"] is False,
        "TEST semantic results were calculated", checks, "test_seal",
    )
    _require(results["phase4_started"] is False, "Phase 4 is marked started", checks, "phase4")

    semantic = read_parquet_records(
        ROOT / "artifacts/data/context_sufficiency_semantic_labels.parquet"
    )
    keys = [
        (row["question_id"], row["retrieval_strategy"], int(row["k"])) for row in semantic
    ]
    _require(len(keys) == len(set(keys)), "semantic condition keys are duplicated", checks,
             "semantic_unique_keys")
    _require(
        set(row["split"] for row in semantic) <= {"train", "validation"},
        "semantic label artifact contains a non-TRAIN/VALIDATION split", checks,
        "semantic_splits",
    )
    evaluable = [row for row in semantic if row["semantic_label_status"] == "evaluable"]
    unevaluable = [row for row in semantic if row["semantic_label_status"] == "unevaluable"]
    _require(
        len(evaluable) + len(unevaluable) == len(semantic)
        and all(row["primary_population_eligible"] is True for row in evaluable)
        and all(row["primary_population_eligible"] is False for row in unevaluable),
        "semantic eligibility/status relationship is inconsistent", checks,
        "primary_eligibility",
    )
    _require(
        all(row["y_suff_semantic"] in {0, 1} for row in evaluable)
        and all(
            row["y_suff_semantic"] is None
            and row["semantic_exclusion_reason"]
            == "claim_exceeds_frozen_nli_pair_budget"
            and row["strict_semantic_category"] == "semantic_unevaluable"
            for row in unevaluable
        ),
        "semantic-unevaluable rows were forced to a semantic class or misidentified",
        checks, "semantic_unevaluable_na",
    )
    _require(
        len(semantic) == int(results["primary_population"]["condition_rows"]),
        "semantic artifact count differs from results", checks, "primary_count",
    )
    _require(
        all(row["semantic_config_sha256"] == selected["semantic_config_sha256"] for row in semantic),
        "semantic label row has a different configuration hash", checks,
        "row_config_hashes",
    )
    _require(
        all(
            row["semantic_label_governance_sha256"] == label_governance_sha256
            for row in semantic
        ),
        "semantic label row has a different governance hash", checks,
        "row_governance_hashes",
    )
    exclusions = read_parquet_records(
        ROOT / "artifacts/data/phase03_semantic_unevaluable_questions.parquet"
    )
    exclusion_questions = {row["question_id"] for row in exclusions}
    _require(
        len(exclusion_questions) == len(exclusions)
        == int(results["primary_population"]["semantic_unevaluable_question_rows"])
        and len(unevaluable)
        == int(results["primary_population"]["semantic_unevaluable_condition_rows"])
        and {row["question_id"] for row in unevaluable} == exclusion_questions,
        "semantic exclusion artifact/counts do not match NA condition rows", checks,
        "semantic_exclusion_accounting",
    )

    manifest = _csv(ROOT / "artifacts/results/phase03_semantic_confirmation_sample_manifest.csv")
    blinded = read_parquet_records(
        ROOT / "artifacts/results/phase03_semantic_confirmation_blinded.parquet"
    )
    template = _csv(ROOT / "artifacts/results/phase03_semantic_confirmation_template.csv")
    status = json.loads((
        ROOT / "artifacts/results/phase03_semantic_confirmation_status.json"
    ).read_text(encoding="utf-8"))
    _require(
        len(manifest) == len(blinded) == len(template) == status["sample_rows"] == 100,
        "confirmation pack does not contain exactly 100 rows in every blinded view", checks,
        "confirmation_size",
    )
    sample_ids = [row["sample_id"] for row in manifest]
    _require(len(sample_ids) == len(set(sample_ids)), "confirmation sample IDs are duplicated",
             checks, "confirmation_unique_ids")
    _require(
        len({row["question_id"] for row in manifest}) == 100,
        "confirmation sample violates one-condition-per-question", checks,
        "confirmation_unique_questions",
    )
    _require(
        set(row["split"] for row in manifest) <= {"train", "validation"},
        "confirmation sample contains a non-TRAIN/VALIDATION row", checks,
        "confirmation_splits",
    )
    original_questions = {
        row["question_id"] for row in _csv(
            ROOT / "artifacts/results/phase03_manual_sample_manifest.csv"
        )
    }
    _require(
        not (original_questions & {row["question_id"] for row in manifest}),
        "confirmation sample overlaps an original development question", checks,
        "development_non_overlap",
    )
    _require(
        not (exclusion_questions & {row["question_id"] for row in manifest})
        and int(status["semantic_unevaluable_rows_in_sample"]) == 0,
        "confirmation sample contains a semantic-unevaluable question", checks,
        "confirmation_excludes_unevaluable",
    )
    forbidden = {
        "y_suff_strict", "y_suff_semantic", "strict_label", "semantic_label",
        "semantic_support_score", "minimum_claim_entailment", "mean_claim_entailment",
        "automatic_prediction", "answer_key", "semantic_model_id", "semantic_model_revision",
    }
    blinded_columns = set(pq.read_schema(
        ROOT / "artifacts/results/phase03_semantic_confirmation_blinded.parquet"
    ).names)
    template_columns = set(template[0]) if template else set()
    _require(
        not (forbidden & (blinded_columns | template_columns)),
        "a blinded confirmation view exposes semantic/automatic answer-key fields", checks,
        "confirmation_blinding",
    )
    _require(
        all(
            not row["annotator_id"] and not row["manual_label"]
            and not row["rationale"] and not row["annotation_timestamp"]
            for row in template
        ),
        "confirmation human-entry fields are not blank", checks,
        "confirmation_unannotated",
    )
    # Deliberately inspect only Parquet metadata/schema, never answer-key row values.
    answer_key_file = ROOT / "artifacts/results/phase03_semantic_confirmation_answer_key.parquet"
    answer_key_metadata = pq.ParquetFile(answer_key_file).metadata
    _require(
        answer_key_metadata.num_rows == 100,
        "separate confirmation answer key does not contain 100 rows", checks,
        "answer_key_metadata_only",
    )
    _require(
        status["primary_human_annotation"] == "pending"
        and status["answer_key_evaluation_status"]
        == "sealed_until_complete_genuine_human_confirmation",
        "confirmation status is not pending and sealed", checks,
        "confirmation_pending",
    )
    _require(
        float(status["automatic_sufficient_precision_gate"]) == 0.90
        and float(status["prevalence_weighted_f1_gate"]) == 0.85,
        "independent confirmation gates differ from the frozen 0.90/0.85 values", checks,
        "confirmation_gates",
    )

    for name, artifact in hashes["artifacts"].items():
        path = ROOT / artifact["path"]
        if not path.exists() or sha256_file(path) != artifact["physical_sha256"]:
            raise ValueError(f"artifact physical hash mismatch: {name}: {path}")
    checks["artifact_physical_hashes"] = "pass"

    model_files = [
        path for path in (ROOT / "artifacts/models").rglob("*")
        if path.is_file() and path.name != ".gitkeep"
    ]
    _require(not model_files, f"Phase 4/model artifact files exist: {model_files}", checks,
             "no_model_artifacts")

    report = {
        "overall_status": "pass",
        "checks": checks,
        "semantic_rows": len(semantic),
        "semantic_evaluable_rows": len(evaluable),
        "semantic_unevaluable_rows": len(unevaluable),
        "semantic_unevaluable_questions": len(exclusion_questions),
        "semantic_split_counts": dict(sorted(Counter(row["split"] for row in semantic).items())),
        "confirmation_rows": len(manifest),
        "confirmation_strata": dict(sorted(Counter(
            row["confirmation_sampling_stratum"] for row in manifest
        ).items())),
        "answer_key_values_read": False,
        "test_outcomes_sealed": True,
        "phase4_started": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
