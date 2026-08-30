"""Frozen independent-human confirmation and Phase 3 finalization.

Validation is deliberately separated from answer-key loading.  The public
``validate_confirmation`` function never opens the answer-key artifact; callers
must receive a passing validation result before invoking ``evaluate_confirmation``.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

from answerability_rag.hashing import canonical_json_sha256, sha256_file


HUMAN_COLUMNS = (
    "schema_version", "sample_id", "blind_order", "question_id", "split",
    "retrieval_strategy", "k", "question", "benchmark_status", "benchmark_answer",
    "retrieved_context_json", "annotation_guideline_version",
    "annotation_guideline_sha256", "annotator_id", "manual_label", "rationale",
    "annotation_timestamp",
)
ANNOTATION_COLUMNS = {"annotator_id", "manual_label", "rationale", "annotation_timestamp"}
PERMITTED_LABELS = {"sufficient", "insufficient", "ambiguous"}
FORBIDDEN_HUMAN_COLUMNS = {
    "y_suff", "y_suff_strict", "y_suff_semantic", "y_suff_final", "prediction",
    "automatic_prediction", "model_prediction", "maximum_span_coverage_fraction",
    "minimum_claim_entailment", "mean_claim_entailment",
    "maximum_selected_premise_contradiction", "confirmation_sampling_stratum",
    "example_id", "answer_key",
}
FINAL_CONFIG_SHA256 = "5b9a394e5e97776844054f3282af815462148adc62b528c77245d61707406977"
EXPECTED_THRESHOLDS = {
    "T_contradiction": 0.5, "T_cov": 0.2, "T_mean": 0.35, "T_min": 0.05,
}
STRATA = (
    "strict_positive_retained", "strict_negative_rescued",
    "strict_negative_not_rescued",
)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), list(reader)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", delete=False, dir=path.parent,
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    fields = tuple(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", delete=False, dir=path.parent,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _validate_timestamp(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _frozen_rule(root: Path, errors: list[dict[str, Any]]) -> dict[str, Any]:
    path = root / "configs/phase03_final_target.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    stored = value.pop("final_target_config_sha256")
    recomputed = canonical_json_sha256(value)
    if stored != recomputed or recomputed != FINAL_CONFIG_SHA256:
        errors.append({"scope": "final_config", "stored": stored, "recomputed": recomputed})
    if value.get("selected_thresholds") != EXPECTED_THRESHOLDS:
        errors.append({"scope": "final_rule", "issue": "frozen thresholds changed"})
    if value.get("strict_positive_demotion_permitted") is not False:
        errors.append({"scope": "final_rule", "issue": "strict-positive preservation changed"})
    if value.get("test_outcomes_sealed") is not True or value.get("phase4_started") is not False:
        errors.append({"scope": "governance", "issue": "TEST seal or Phase 4 boundary changed"})
    return {"value": value, "stored_sha256": stored, "physical_sha256": sha256_file(path)}


def validate_confirmation(root: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Validate the completed human file without opening the answer key."""
    root = root.resolve()
    paths = {
        "human": root / "artifacts/results/phase03_final_confirmation_annotator_1.csv",
        "template": root / "artifacts/results/phase03_final_confirmation_template.csv",
        "manifest": root / "artifacts/results/phase03_final_confirmation_sample_manifest.csv",
        "development": root / "artifacts/results/phase03_manual_sample_manifest.csv",
        "prior": root / "artifacts/results/phase03_semantic_confirmation_sample_manifest.csv",
        "final_labels": root / "artifacts/data/context_sufficiency_final_labels.parquet",
        "strict_labels": root / "artifacts/data/context_sufficiency_labels.parquet",
    }
    errors: list[dict[str, Any]] = []
    frozen = _frozen_rule(root, errors)
    human_fields, human = _read_csv(paths["human"])
    template_fields, template = _read_csv(paths["template"])
    _, manifest = _read_csv(paths["manifest"])
    _, development = _read_csv(paths["development"])
    _, prior = _read_csv(paths["prior"])

    if human_fields != list(HUMAN_COLUMNS) or human_fields != template_fields:
        errors.append({"scope": "schema", "actual": human_fields,
                       "expected": list(HUMAN_COLUMNS)})
    if set(human_fields) & FORBIDDEN_HUMAN_COLUMNS:
        errors.append({"scope": "blinding", "forbidden_columns":
                       sorted(set(human_fields) & FORBIDDEN_HUMAN_COLUMNS)})
    if len(human) != 50:
        errors.append({"scope": "row_count", "actual": len(human), "expected": 50})

    template_by_id = {row["sample_id"]: row for row in template}
    manifest_by_id = {row["sample_id"]: row for row in manifest}
    sample_ids: list[str] = []
    annotators: set[str] = set()
    for csv_row, row in enumerate(human, start=2):
        sample_id = row.get("sample_id", "")
        blind_order = row.get("blind_order", "")
        sample_ids.append(sample_id)
        if sample_id not in template_by_id or sample_id not in manifest_by_id:
            errors.append({"csv_row": csv_row, "blind_order": blind_order,
                           "sample_id": sample_id, "issue": "unknown frozen sample ID"})
            continue
        for field in HUMAN_COLUMNS:
            if field not in ANNOTATION_COLUMNS and row.get(field, "") != template_by_id[sample_id].get(field, ""):
                errors.append({"csv_row": csv_row, "blind_order": blind_order,
                               "sample_id": sample_id, "issue": f"frozen field changed: {field}"})
        for field in ("blind_order", "question_id", "split", "retrieval_strategy", "k",
                      "annotation_guideline_version", "annotation_guideline_sha256"):
            if row.get(field, "") != manifest_by_id[sample_id].get(field, ""):
                errors.append({"csv_row": csv_row, "blind_order": blind_order,
                               "sample_id": sample_id, "issue": f"manifest mismatch: {field}"})
        label = row.get("manual_label", "").strip().lower()
        if label not in PERMITTED_LABELS:
            errors.append({"csv_row": csv_row, "blind_order": blind_order,
                           "sample_id": sample_id, "issue": "invalid or incomplete manual label",
                           "value": row.get("manual_label", "")})
        annotator = row.get("annotator_id", "").strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", annotator):
            errors.append({"csv_row": csv_row, "blind_order": blind_order,
                           "sample_id": sample_id, "issue": "invalid annotator ID"})
        else:
            annotators.add(annotator)
        timestamp = row.get("annotation_timestamp", "").strip()
        if not timestamp or not _validate_timestamp(timestamp):
            errors.append({"csv_row": csv_row, "blind_order": blind_order,
                           "sample_id": sample_id, "issue": "invalid timezone-aware ISO-8601 timestamp",
                           "value": timestamp})

    duplicates = {key: count for key, count in Counter(sample_ids).items() if count != 1}
    if duplicates:
        errors.append({"scope": "sample_id_uniqueness", "duplicates": duplicates})
    if set(sample_ids) != set(manifest_by_id):
        errors.append({"scope": "sample_id_set",
                       "missing": sorted(set(manifest_by_id) - set(sample_ids)),
                       "extra": sorted(set(sample_ids) - set(manifest_by_id))})
    questions = [row.get("question_id", "") for row in human]
    question_duplicates = {key: count for key, count in Counter(questions).items() if count != 1}
    if question_duplicates:
        errors.append({"scope": "question_uniqueness", "duplicates": question_duplicates})
    if len(annotators) != 1:
        errors.append({"scope": "annotator", "issue": "expected one genuine annotator ID",
                       "values": sorted(annotators)})

    question_set = set(questions)
    development_overlap = question_set & {row["question_id"] for row in development}
    prior_overlap = question_set & {row["question_id"] for row in prior}
    if development_overlap:
        errors.append({"scope": "development_overlap", "question_ids": sorted(development_overlap)})
    if prior_overlap:
        errors.append({"scope": "superseded_confirmation_overlap",
                       "question_ids": sorted(prior_overlap)})
    if any(row.get("split") == "test" for row in human):
        errors.append({"scope": "test_seal", "issue": "TEST row present"})
    if any(row.get("benchmark_status") != "answerable_with_reference" for row in human):
        errors.append({"scope": "benchmark", "issue": "benchmark-impossible row present"})

    final_by_example = {row["example_id"]: row for row in pq.read_table(paths["final_labels"]).to_pylist()}
    strict_by_example = {row["example_id"]: row for row in pq.read_table(
        paths["strict_labels"], filters=[("split", "in", ["train", "validation"])],
    ).to_pylist()}
    for row in manifest:
        final_row = final_by_example.get(row["example_id"])
        strict_row = strict_by_example.get(row["example_id"])
        if final_row is None or final_row.get("primary_phase4_target_eligible") is not True:
            errors.append({"scope": "primary_eligibility", "blind_order": row["blind_order"],
                           "sample_id": row["sample_id"], "issue": "not semantic-evaluable/eligible"})
        if strict_row is None or strict_row.get("benchmark_is_impossible") is not False:
            errors.append({"scope": "benchmark", "blind_order": row["blind_order"],
                           "sample_id": row["sample_id"], "issue": "benchmark-impossible"})

    phase36c = json.loads((root / "artifacts/results/phase03_rescue_expanded_artifact_manifest.json").read_text(encoding="utf-8"))
    frozen_artifacts = phase36c["artifacts"]
    for key, path in {
        "confirmation_manifest": paths["manifest"],
        "confirmation_template": paths["template"],
        "final_labels": paths["final_labels"],
        "primary_target": root / "artifacts/data/phase03_final_primary_target.parquet",
        "final_config": root / "configs/phase03_final_target.json",
    }.items():
        if sha256_file(path) != frozen_artifacts[key]["physical_sha256"]:
            errors.append({"scope": "frozen_artifact", "artifact": key, "issue": "hash changed"})

    result = {
        "schema_version": "phase03-final-confirmation-prekey-validation-v1",
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "answer_key_opened": False,
        "human_annotation_path": paths["human"].relative_to(root).as_posix(),
        "human_annotation_physical_sha256": sha256_file(paths["human"]),
        "sample_rows": len(human),
        "unique_sample_ids": len(set(sample_ids)),
        "unique_questions": len(question_set),
        "manual_label_counts": dict(Counter(row.get("manual_label", "").strip().lower() for row in human)),
        "annotator_ids": sorted(annotators),
        "test_rows": sum(row.get("split") == "test" for row in human),
        "benchmark_impossible_rows": sum(
            row.get("benchmark_status") != "answerable_with_reference" for row in human
        ),
        "semantic_unevaluable_rows": sum(
            not final_by_example.get(row["example_id"], {}).get("primary_phase4_target_eligible", False)
            for row in manifest
        ),
        "development_question_overlap": len(development_overlap),
        "superseded_confirmation_question_overlap": len(prior_overlap),
        "confirmation_manifest_physical_sha256": sha256_file(paths["manifest"]),
        "final_target_config_sha256": frozen["stored_sha256"],
        "final_target_config_physical_sha256": frozen["physical_sha256"],
        "test_outcomes_sealed": True,
        "phase4_started": False,
    }
    return result, human


def _confusion(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    cells = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    for row in rows:
        truth = int(row["manual_label"] == "sufficient")
        prediction = int(row["prediction"])
        cell = "tp" if truth and prediction else "fn" if truth else "fp" if prediction else "tn"
        cells[cell] += 1
    return cells


def _metrics(cells: dict[str, float]) -> dict[str, float | None]:
    tn, fp, fn, tp = (cells[key] for key in ("tn", "fp", "fn", "tp"))
    total = tn + fp + fn + tp
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else 0.0 if precision == 0 or recall == 0 else None
    )
    return {"precision": precision, "recall": recall, "f1": f1,
            "accuracy": (tp + tn) / total if total else None}


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> dict[str, float | None]:
    if trials == 0:
        return {"lower": None, "upper": None}
    proportion = successes / trials
    denominator = 1 + z * z / trials
    centre = (proportion + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)) / denominator
    return {"lower": max(0.0, centre - half), "upper": min(1.0, centre + half)}


def _population_stratum(row: dict[str, Any]) -> str:
    if int(row["y_suff_strict"]) == 1:
        return "strict_positive_retained"
    return "strict_negative_rescued" if int(row["y_suff_final"]) == 1 else "strict_negative_not_rescued"


def evaluate_confirmation(
    root: Path, validation: dict[str, Any], human: list[dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Open the frozen answer key only after a passing pre-key validation."""
    if validation.get("status") != "pass" or validation.get("answer_key_opened") is not False:
        raise ValueError("answer key remains sealed until pre-key validation passes")
    root = root.resolve()
    key_path = root / "artifacts/results/phase03_final_confirmation_answer_key.parquet"
    phase36c = json.loads((root / "artifacts/results/phase03_rescue_expanded_artifact_manifest.json").read_text(encoding="utf-8"))
    if sha256_file(key_path) != phase36c["artifacts"]["confirmation_answer_key"]["physical_sha256"]:
        raise ValueError("frozen confirmation answer-key hash changed")
    keys = pq.read_table(key_path).to_pylist()
    _, manifest = _read_csv(root / "artifacts/results/phase03_final_confirmation_sample_manifest.csv")
    by_human = {row["sample_id"]: row for row in human}
    by_manifest = {row["sample_id"]: row for row in manifest}
    by_key = {row["sample_id"]: row for row in keys}
    if set(by_human) != set(by_manifest) or set(by_human) != set(by_key) or len(by_key) != 50:
        raise ValueError("answer-key identity does not match the validated frozen sample")

    rows: list[dict[str, Any]] = []
    for sample_id in sorted(by_human, key=lambda value: int(by_human[value]["blind_order"])):
        key = by_key[sample_id]
        if key["final_target_config_sha256"] != FINAL_CONFIG_SHA256:
            raise ValueError(f"answer-key target hash mismatch: {sample_id}")
        row = {**by_human[sample_id], **by_manifest[sample_id], **key}
        row["prediction"] = int(row["y_suff_final"])
        rows.append(row)

    binary = [row for row in rows if row["manual_label"] != "ambiguous"]
    raw_cells = _confusion(binary)
    raw_metrics = _metrics(raw_cells)
    final_rows = pq.read_table(root / "artifacts/data/context_sufficiency_final_labels.parquet").to_pylist()
    primary_rows = [row for row in final_rows if row["primary_phase4_target_eligible"]]
    population_counts = Counter(_population_stratum(row) for row in primary_rows)
    population_total = sum(population_counts.values())

    weighted_cells = {cell: 0.0 for cell in ("tn", "fp", "fn", "tp")}
    by_stratum = []
    for stratum in STRATA:
        selected = [row for row in binary if row["confirmation_sampling_stratum"] == stratum]
        if not selected:
            raise ValueError(f"no non-ambiguous confirmation row in stratum: {stratum}")
        cells = _confusion(selected)
        weight = population_counts[stratum] / population_total
        for cell in weighted_cells:
            weighted_cells[cell] += weight * cells[cell] / len(selected)
        by_stratum.append({
            "confirmation_sampling_stratum": stratum,
            "sample_rows": len(selected),
            "ambiguous_rows": sum(
                row["manual_label"] == "ambiguous" and row["confirmation_sampling_stratum"] == stratum
                for row in rows
            ),
            "primary_population_conditions": population_counts[stratum],
            "primary_population_weight": weight,
            "confusion_matrix": cells,
            **_metrics(cells),
        })
    weighted_metrics = _metrics(weighted_cells)

    disagreements = []
    for row in rows:
        truth = None if row["manual_label"] == "ambiguous" else int(row["manual_label"] == "sufficient")
        prediction = int(row["prediction"])
        if truth is None or truth != prediction:
            error_type = "ambiguous" if truth is None else "false_positive" if prediction else "false_negative"
            structural_pattern = (
                "ambiguous_human_judgement" if truth is None
                else "rescued_context_human_insufficient" if error_type == "false_positive"
                else "unrescued_context_human_sufficient"
            )
            disagreements.append({
                "blind_order": int(row["blind_order"]), "sample_id": row["sample_id"],
                "question_id": row["question_id"], "example_id": row["example_id"],
                "confirmation_sampling_stratum": row["confirmation_sampling_stratum"],
                "retrieval_strategy": row["retrieval_strategy"], "k": int(row["k"]),
                "automatic_label": "sufficient" if prediction else "insufficient",
                "human_label": row["manual_label"], "error_type": error_type,
                "structural_error_pattern": structural_pattern, "human_rationale": row["rationale"],
            })

    config = json.loads((root / "configs/phase03_final_target.json").read_text(encoding="utf-8"))
    gates = config["confirmation_gates"]
    precision_pass = bool(raw_metrics["precision"] is not None and
                          raw_metrics["precision"] >= gates["minimum_ordinary_precision"])
    weighted_pass = bool(weighted_metrics["f1"] is not None and
                         weighted_metrics["f1"] >= gates["minimum_prevalence_weighted_f1"])
    label_counts = Counter(row["manual_label"] for row in rows)
    result = {
        "schema_version": "phase03-final-human-confirmation-evaluation-v1",
        "evaluation_design": "independent_stratified_confirmation; not a natural-prevalence sample",
        "answer_key_opened_after_prekey_validation": True,
        "prekey_validation_status": validation["status"],
        "sample_size": len(rows), "binary_evaluation_rows": len(binary),
        "human_label_counts": {
            "sufficient": label_counts["sufficient"], "insufficient": label_counts["insufficient"],
            "ambiguous": label_counts["ambiguous"],
        },
        "ambiguous_rate": label_counts["ambiguous"] / len(rows),
        "confusion_matrix": raw_cells, "precision": raw_metrics["precision"],
        "recall": raw_metrics["recall"], "f1": raw_metrics["f1"],
        "accuracy": raw_metrics["accuracy"],
        "precision_wilson_95": wilson_interval(raw_cells["tp"], raw_cells["tp"] + raw_cells["fp"]),
        "weighting": {
            "method": "within-stratum confusion proportions weighted by frozen PRIMARY TRAIN+VALIDATION condition frequencies",
            "population_counts": {stratum: population_counts[stratum] for stratum in STRATA},
            "population_total": population_total,
            "estimated_confusion_proportions": weighted_cells,
        },
        "weighted_precision": weighted_metrics["precision"],
        "weighted_recall": weighted_metrics["recall"],
        "weighted_f1": weighted_metrics["f1"],
        "weighted_accuracy": weighted_metrics["accuracy"],
        "metrics_by_confirmation_stratum": by_stratum,
        "disagreement_count": len(disagreements),
        "error_type_counts": dict(Counter(row["error_type"] for row in disagreements)),
        "structural_error_pattern_counts": dict(Counter(
            row["structural_error_pattern"] for row in disagreements
        )),
        "gates": {
            "minimum_ordinary_precision": gates["minimum_ordinary_precision"],
            "minimum_prevalence_weighted_f1": gates["minimum_prevalence_weighted_f1"],
            "ordinary_precision_passed": precision_pass,
            "prevalence_weighted_f1_passed": weighted_pass,
            "both_passed": precision_pass and weighted_pass,
        },
        "hashes": {
            "human_annotation_physical_sha256": validation["human_annotation_physical_sha256"],
            "confirmation_answer_key_physical_sha256": sha256_file(key_path),
            "confirmation_manifest_physical_sha256": validation["confirmation_manifest_physical_sha256"],
            "final_target_config_sha256": FINAL_CONFIG_SHA256,
        },
        "test_outcomes_sealed": True, "phase4_started": False,
    }

    strict_rows = pq.read_table(
        root / "artifacts/data/context_sufficiency_labels.parquet",
        filters=[("split", "in", ["train", "validation"])],
    ).to_pylist()
    benchmark_impossible = [row for row in strict_rows if row["benchmark_is_impossible"]]
    unresolved = [row for row in strict_rows if not row["benchmark_is_impossible"] and row["y_suff"] is None]
    excluded_final = [row for row in final_rows if not row["primary_phase4_target_eligible"]]
    target_population = {
        "schema_version": "phase03-final-target-population-manifest-v1",
        "final_target_config_sha256": FINAL_CONFIG_SHA256,
        "eligible_questions": len({row["question_id"] for row in primary_rows}),
        "eligible_conditions": len(primary_rows),
        "strict_positive_conditions": population_counts["strict_positive_retained"],
        "rescued_positive_conditions": population_counts["strict_negative_rescued"],
        "final_positive_conditions": sum(int(row["y_suff_final"]) == 1 for row in primary_rows),
        "final_negative_conditions": sum(int(row["y_suff_final"]) == 0 for row in primary_rows),
        "exclusions": {
            "benchmark_impossible": {
                "questions": len({row["question_id"] for row in benchmark_impossible}),
                "conditions": len(benchmark_impossible), "primary_use": False,
                "retained_for_sensitivity_analysis": True,
            },
            "unresolved_reference_or_evidence": {
                "questions": len({row["question_id"] for row in unresolved}),
                "conditions": len(unresolved), "primary_use": False,
            },
            "semantic_unevaluable": {
                "questions": len({row["question_id"] for row in excluded_final}),
                "conditions": len(excluded_final), "primary_use": False,
                "reason": "claim_exceeds_frozen_nli_pair_budget",
                "question_ids": sorted({row["question_id"] for row in excluded_final}),
            },
        },
        "test_rows": 0, "test_outcomes_sealed": True, "phase4_started": False,
    }
    return result, disagreements, target_population


def run_final_confirmation(root: Path, validate_only: bool = False) -> dict[str, Any]:
    root = root.resolve()
    validation, human = validate_confirmation(root)
    validation_path = root / "artifacts/results/phase03_final_confirmation_prekey_validation.json"
    write_json(validation_path, validation)
    if validation["status"] != "pass":
        raise ValueError(f"human confirmation validation failed: {validation['errors']}")
    if validate_only:
        return {"validation": validation}

    evaluation, disagreements, population = evaluate_confirmation(root, validation, human)
    results_dir = root / "artifacts/results"
    evaluation_path = results_dir / "phase03_final_confirmation_evaluation.json"
    disagreement_path = results_dir / "phase03_final_confirmation_disagreements.csv"
    population_path = results_dir / "phase03_final_target_population_manifest.json"
    status_path = results_dir / "phase03_final_confirmation_evaluation_status.json"
    write_json(evaluation_path, evaluation)
    write_csv(disagreement_path, disagreements, (
        "blind_order", "sample_id", "question_id", "example_id",
        "confirmation_sampling_stratum", "retrieval_strategy", "k", "automatic_label",
        "human_label", "error_type", "structural_error_pattern", "human_rationale",
    ))
    write_json(population_path, population)
    status = {
        "schema_version": "phase03-final-confirmation-evaluation-status-v1",
        "status": "passed_phase03_complete" if evaluation["gates"]["both_passed"] else "failed_decision_required",
        "prekey_validation_status": "pass", "answer_key_opened_after_validation": True,
        "ordinary_precision_gate_passed": evaluation["gates"]["ordinary_precision_passed"],
        "prevalence_weighted_f1_gate_passed": evaluation["gates"]["prevalence_weighted_f1_passed"],
        "test_outcomes_sealed": True, "phase4_started": False,
    }
    write_json(status_path, status)
    return {"validation": validation, "evaluation": evaluation,
            "disagreements": disagreements, "target_population": population, "status": status}


def build_final_artifact_manifest(root: Path) -> dict[str, Any]:
    root = root.resolve()
    files = {
        "human_confirmation": "artifacts/results/phase03_final_confirmation_annotator_1.csv",
        "confirmation_template": "artifacts/results/phase03_final_confirmation_template.csv",
        "confirmation_blinded": "artifacts/results/phase03_final_confirmation_blinded.parquet",
        "confirmation_answer_key": "artifacts/results/phase03_final_confirmation_answer_key.parquet",
        "confirmation_sample_manifest": "artifacts/results/phase03_final_confirmation_sample_manifest.csv",
        "confirmation_frozen_status": "artifacts/results/phase03_final_confirmation_status.json",
        "prekey_validation": "artifacts/results/phase03_final_confirmation_prekey_validation.json",
        "confirmation_evaluation": "artifacts/results/phase03_final_confirmation_evaluation.json",
        "confirmation_disagreements": "artifacts/results/phase03_final_confirmation_disagreements.csv",
        "confirmation_evaluation_status": "artifacts/results/phase03_final_confirmation_evaluation_status.json",
        "target_population_manifest": "artifacts/results/phase03_final_target_population_manifest.json",
        "final_target_config": "configs/phase03_final_target.json",
        "expanded_rescue_grid": "configs/phase03_rescue_grid_expanded.json",
        "semantic_label_governance": "configs/phase03_semantic_label_governance.json",
        "final_labels": "artifacts/data/context_sufficiency_final_labels.parquet",
        "final_primary_target": "artifacts/data/phase03_final_primary_target.parquet",
        "strict_labels": "artifacts/data/context_sufficiency_labels.parquet",
        "semantic_labels": "artifacts/data/context_sufficiency_semantic_labels.parquet",
        "experiment_contract": "docs/EXPERIMENT_CONTRACT.md",
        "final_validation_document": "docs/PHASE_03_FINAL_VALIDATION.md",
        "phase03_execution_document": "docs/PHASE_03_EXECUTION.md",
        "research_decisions_document": "docs/RESEARCH_DECISIONS.md",
        "final_confirmation_implementation": "src/answerability_rag/sufficiency/final_confirmation.py",
        "final_confirmation_runner": "scripts/evaluate_phase03_final_confirmation.py",
        "final_confirmation_checker": "scripts/check_phase03_final_confirmation.py",
        "original_prototype": "notebooks/original_prototype.ipynb",
    }
    artifacts = {
        name: {"path": relative, "bytes": (root / relative).stat().st_size,
               "physical_sha256": sha256_file(root / relative)}
        for name, relative in files.items()
    }
    manifest = {
        "schema_version": "phase03-final-artifact-hash-manifest-v1",
        "phase03_status": "complete_pending_phase4_review",
        "final_target_config_sha256": FINAL_CONFIG_SHA256,
        "artifacts": artifacts,
        "test_outcomes_sealed": True, "phase4_started": False,
    }
    write_json(root / "artifacts/results/phase03_final_artifact_manifest.json", manifest)
    return manifest
