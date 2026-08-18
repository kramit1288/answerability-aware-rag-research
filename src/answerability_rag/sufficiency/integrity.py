"""Phase 3 label, blindness, monotonicity, and test-seal checks."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .governance import validate_feature_selection
from .labeling import LABEL_FIELDS


def monotonicity_violations(labels: list[dict[str, Any]]) -> list[tuple[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in labels:
        if row["label_method"] == "gold_span_coverage":
            grouped[(row["question_id"], row["retrieval_strategy"])].append(row)
    violations = []
    for key, rows in grouped.items():
        values = [row["y_suff"] for row in sorted(rows, key=lambda item: int(item["k"]))]
        if any(left > right for left, right in zip(values, values[1:])):
            violations.append(key)
    return violations


def validate_phase03(
    labels: list[dict[str, Any]], conditions: list[dict[str, Any]], development_summary: list[dict],
    sample_manifest: list[dict], blinded: list[dict], answer_key: list[dict], governance: dict,
    *, expected_sample_size: int,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(check_id: str, passed: bool, observed: Any, expected: Any) -> None:
        checks.append({"check_id": check_id, "status": "pass" if passed else "fail",
                       "observed": observed, "expected": expected})

    add("label_rows", len(labels) == 10920, len(labels), 10920)
    label_keys = [(row["question_id"], row["retrieval_strategy"], int(row["k"])) for row in labels]
    condition_keys = {
        (row["question_id"], row["retrieval_strategy"], int(row["k"])) for row in conditions
    }
    add("unique_condition_labels", len(label_keys) == len(set(label_keys)),
        len(set(label_keys)), len(label_keys))
    add("exact_phase02_condition_coverage", set(label_keys) == condition_keys,
        len(set(label_keys) ^ condition_keys), 0)
    split_counts = Counter(row["split"] for row in labels)
    add("split_row_counts", split_counts == {"train": 7644, "validation": 1644, "test": 1632},
        dict(split_counts), {"train": 7644, "validation": 1644, "test": 1632})
    monotonic_violations = monotonicity_violations(labels)
    add("k_monotonicity", not monotonic_violations, len(monotonic_violations), 0)
    resolved_rule = all(
        (row["y_suff"] == 1) == (int(row["fully_covered_span_count"]) > 0)
        for row in labels if row["label_method"] == "gold_span_coverage"
    )
    add("full_span_rule", resolved_rule, resolved_rule, True)
    impossible = [row for row in labels if row["label_method"] == "benchmark_impossible"]
    impossible_ok = all(
        row["y_suff"] == 0 and row["label_status"] == "preliminary_benchmark_negative"
        and row["gold_document_hit"] is None for row in impossible
    )
    add("benchmark_negative_provenance", len(impossible) == 3600 and impossible_ok,
        {"rows": len(impossible), "valid": impossible_ok}, {"rows": 3600, "valid": True})
    unresolved = [row for row in labels if row["label_status"] == "unresolved"]
    unresolved_ok = all(row["y_suff"] is None and row["exclusion_reason"] for row in unresolved)
    add("unresolved_are_na", unresolved_ok, unresolved_ok, True)
    test_summaries = sum(row.get("split") == "test" for row in development_summary)
    add("test_aggregate_sealed", test_summaries == 0, test_summaries, 0)
    sample_ids = [row["sample_id"] for row in sample_manifest]
    sample_questions = [row["question_id"] for row in sample_manifest]
    sample_ok = (
        len(sample_manifest) == expected_sample_size
        and len(sample_ids) == len(set(sample_ids))
        and len(sample_questions) == len(set(sample_questions))
        and all(row["split"] in {"train", "validation"} for row in sample_manifest)
    )
    add("manual_sample", sample_ok, {
        "rows": len(sample_manifest), "unique_questions": len(set(sample_questions)),
        "test_rows": sum(row["split"] == "test" for row in sample_manifest),
    }, {"rows": expected_sample_size, "unique_questions": expected_sample_size, "test_rows": 0})
    forbidden_blind = {
        "automatic_y_suff", "y_suff", "label_method", "label_status", "gold_document_hit",
        "maximum_span_coverage_fraction", "partial_overlap", "sampling_stratum",
    }
    blind_columns = set(blinded[0]) if blinded else set()
    add("annotation_blinding", not (forbidden_blind & blind_columns),
        sorted(forbidden_blind & blind_columns), [])
    blind_ids = {row["sample_id"] for row in blinded}
    key_ids = {row["sample_id"] for row in answer_key}
    add("annotation_key_separation", blind_ids == key_ids == set(sample_ids),
        {"blind": len(blind_ids), "key": len(key_ids), "sample": len(sample_ids)},
        {"blind": expected_sample_size, "key": expected_sample_size,
         "sample": expected_sample_size})
    classified = set(governance["artifacts"]["context_sufficiency_labels"])
    add("column_governance_complete", classified == set(LABEL_FIELDS),
        sorted(set(LABEL_FIELDS) - classified), [])
    try:
        validate_feature_selection(
            governance["future_classifier_feature_allowlist_from_phase03_labels"], governance
        )
        leakage_guard = True
    except ValueError:
        leakage_guard = False
    add("feature_allowlist_guard", leakage_guard, leakage_guard, True)
    return {
        "schema_version": "phase03-integrity-v1",
        "overall_status": "pass" if all(row["status"] == "pass" for row in checks) else "fail",
        "checks": checks,
    }
