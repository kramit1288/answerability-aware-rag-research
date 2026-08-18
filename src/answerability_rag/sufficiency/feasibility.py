"""Train/validation-only alignment feasibility summaries."""

from __future__ import annotations

from collections import Counter
from statistics import mean, median
from typing import Any


def _distribution(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"minimum": None, "median": None, "mean": None, "maximum": None}
    return {
        "minimum": min(values), "median": median(values), "mean": mean(values),
        "maximum": max(values),
    }


def development_alignment_report(
    alignment_rows: list[dict[str, Any]], *, minimum_coverage: float,
) -> dict[str, Any]:
    development = [row for row in alignment_rows if row["split"] in {"train", "validation"}]
    by_question: dict[str, list[dict[str, Any]]] = {}
    for row in development:
        by_question.setdefault(row["question_id"], []).append(row)
    status_counts: Counter[str] = Counter()
    multiple = 0
    character_lengths: list[int] = []
    token_lengths: list[int] = []
    candidates_per_question: list[int] = []
    for rows in by_question.values():
        status = rows[0]["gold_alignment_status"]
        if status == "aligned_exact":
            status_counts["exact_aligned"] += 1
        elif status == "aligned_normalized":
            status_counts["normalized_only_aligned"] += 1
        else:
            status_counts["unresolved"] += 1
        count = int(rows[0]["candidate_span_count"])
        candidates_per_question.append(count)
        if count > 1:
            multiple += 1
        for row in rows:
            if row["character_length"] is not None:
                character_lengths.append(int(row["character_length"]))
                token_lengths.append(int(row["token_length"]))
    total = len(by_question)
    aligned = status_counts["exact_aligned"] + status_counts["normalized_only_aligned"]
    coverage = aligned / total if total else 0.0
    return {
        "schema_version": "phase03-alignment-feasibility-v1",
        "summary_splits": ["train", "validation"],
        "answerable_questions": total,
        "exact_aligned_questions": status_counts["exact_aligned"],
        "normalized_only_aligned_questions": status_counts["normalized_only_aligned"],
        "multiple_match_questions": multiple,
        "unresolved_questions": status_counts["unresolved"],
        "aligned_questions": aligned,
        "alignment_coverage": coverage,
        "minimum_required_coverage": minimum_coverage,
        "gate_status": "pass" if coverage >= minimum_coverage else "fail",
        "character_length_distribution": _distribution(character_lengths),
        "token_length_distribution": _distribution(token_lengths),
        "accepted_candidate_spans_per_question": _distribution(candidates_per_question),
        "test_aggregate_alignment_statistics_calculated": False,
    }
