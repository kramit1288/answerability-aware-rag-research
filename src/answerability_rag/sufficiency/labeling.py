"""Deterministic interval-union sufficiency labels for frozen retrieval bundles."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Iterable

from answerability_rag.hashing import canonical_json, canonical_json_sha256


LABEL_FIELDS = (
    "schema_version", "example_id", "run_id", "question_id", "split", "split_group_id",
    "retrieval_strategy", "k", "benchmark_is_impossible", "reference_status",
    "gold_alignment_status", "label_status", "y_suff", "label_method", "label_provenance",
    "gold_document_hit", "accepted_gold_span_count", "maximum_span_coverage_fraction",
    "fully_covered_span_count", "partial_overlap", "first_covering_rank",
    "covering_chunk_ids_json", "exclusion_reason", "phase01_split_sha256",
    "phase02_conditions_semantic_sha256", "phase02_retrieval_config_sha256",
    "phase03_label_config_sha256",
)


def merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted((int(start), int(end)) for start, end in intervals if int(end) > int(start))
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def interval_coverage(span: tuple[int, int], intervals: Iterable[tuple[int, int]]) -> float:
    start, end = span
    if end <= start:
        raise ValueError("evidence span must be non-empty")
    covered = 0
    for left, right in merge_intervals(intervals):
        covered += max(0, min(end, right) - max(start, left))
    return min(1.0, covered / (end - start))


def _coverage_diagnostics(
    span_rows: list[dict[str, Any]], chunk_ids: list[str], chunks: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    ranked_chunks = [chunks[chunk_id] for chunk_id in chunk_ids]
    intervals_by_doc: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for chunk in ranked_chunks:
        intervals_by_doc[chunk["doc_id"]].append((chunk["char_start"], chunk["char_end"]))
    fractions = [
        interval_coverage(
            (int(span["source_char_start"]), int(span["source_char_end"])),
            intervals_by_doc.get(span["gold_doc_id"], []),
        ) for span in span_rows
    ]
    fully = [span for span, fraction in zip(span_rows, fractions) if fraction >= 1.0]
    partial = any(0.0 < fraction < 1.0 for fraction in fractions)
    gold_docs = {span["gold_doc_id"] for span in span_rows}
    gold_hit = any(chunk["doc_id"] in gold_docs for chunk in ranked_chunks)
    first_covering_rank = None
    for rank in range(1, len(ranked_chunks) + 1):
        prefix_by_doc: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for chunk in ranked_chunks[:rank]:
            prefix_by_doc[chunk["doc_id"]].append((chunk["char_start"], chunk["char_end"]))
        if any(interval_coverage(
            (int(span["source_char_start"]), int(span["source_char_end"])),
            prefix_by_doc.get(span["gold_doc_id"], []),
        ) >= 1.0 for span in span_rows):
            first_covering_rank = rank
            break
    covering: list[str] = []
    if fully:
        full_keys = [
            (span["gold_doc_id"], int(span["source_char_start"]), int(span["source_char_end"]))
            for span in fully
        ]
        for chunk_id, chunk in zip(chunk_ids, ranked_chunks):
            if any(
                chunk["doc_id"] == doc_id and int(chunk["char_start"]) < end
                and int(chunk["char_end"]) > start
                for doc_id, start, end in full_keys
            ):
                covering.append(chunk_id)
    return {
        "y_suff": int(bool(fully)), "gold_document_hit": gold_hit,
        "maximum_span_coverage_fraction": max(fractions),
        "fully_covered_span_count": len(fully), "partial_overlap": partial,
        "first_covering_rank": first_covering_rank, "covering_chunk_ids": covering,
    }


def build_label_rows(
    conditions: list[dict[str, Any]], chunks: list[dict[str, Any]],
    questions: dict[str, Any], assignments: dict[str, dict[str, str]],
    alignment_rows: list[dict[str, Any]], *, run_id: str, config: dict[str, Any],
    phase03_config_sha256: str,
) -> list[dict[str, Any]]:
    expected = config["expected_inputs"]
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in alignment_rows:
        if row["gold_alignment_status"].startswith("aligned_"):
            by_question[row["question_id"]].append(row)
    chunk_map = {row["chunk_id"]: row for row in chunks}
    output: list[dict[str, Any]] = []
    for condition in sorted(
        conditions, key=lambda row: (row["question_id"], row["retrieval_strategy"], int(row["k"]))
    ):
        question = questions[condition["question_id"]]
        assignment = assignments[question.question_id]
        key = {
            "question_id": question.question_id,
            "retrieval_strategy": condition["retrieval_strategy"], "k": int(condition["k"]),
            "phase02_conditions_semantic_sha256": expected["phase02_conditions_semantic_sha256"],
            "phase03_label_config_sha256": phase03_config_sha256,
        }
        base = {
            "schema_version": "phase03-labels-v1", "example_id": canonical_json_sha256(key),
            "run_id": run_id, "question_id": question.question_id, "split": assignment["split"],
            "split_group_id": assignment["split_group_id"],
            "retrieval_strategy": condition["retrieval_strategy"], "k": int(condition["k"]),
            "benchmark_is_impossible": question.is_impossible,
            "reference_status": assignment["reference_status"],
            "phase01_split_sha256": expected["phase01_split_sha256"],
            "phase02_conditions_semantic_sha256": expected["phase02_conditions_semantic_sha256"],
            "phase02_retrieval_config_sha256": expected["phase02_retrieval_config_sha256"],
            "phase03_label_config_sha256": phase03_config_sha256,
        }
        if question.is_impossible:
            output.append({**base, "gold_alignment_status": "not_applicable_benchmark_impossible",
                "label_status": config["label"]["benchmark_impossible_status"], "y_suff": 0,
                "label_method": "benchmark_impossible",
                "label_provenance": "techqa_benchmark_relative_impossible_annotation",
                "gold_document_hit": None, "accepted_gold_span_count": None,
                "maximum_span_coverage_fraction": None, "fully_covered_span_count": None,
                "partial_overlap": None, "first_covering_rank": None,
                "covering_chunk_ids_json": None, "exclusion_reason": None})
            continue
        spans = by_question.get(question.question_id, [])
        if not spans:
            reason = "empty_answerable_reference" if not question.answer else "no_defensible_gold_alignment"
            output.append({**base, "gold_alignment_status": "unresolved", "label_status": "unresolved",
                "y_suff": None, "label_method": "none",
                "label_provenance": "phase03_alignment_audit", "gold_document_hit": None,
                "accepted_gold_span_count": 0, "maximum_span_coverage_fraction": None,
                "fully_covered_span_count": None, "partial_overlap": None,
                "first_covering_rank": None, "covering_chunk_ids_json": None,
                "exclusion_reason": reason})
            continue
        chunk_ids = json.loads(condition["ordered_chunk_ids_json"])
        diagnostics = _coverage_diagnostics(spans, chunk_ids, chunk_map)
        output.append({**base, "gold_alignment_status": spans[0]["gold_alignment_status"],
            "label_status": "automatic_resolved", "y_suff": diagnostics["y_suff"],
            "label_method": "gold_span_coverage",
            "label_provenance": canonical_json({
                "alignment_ids": sorted(span["alignment_id"] for span in spans),
                "rule": config["label"]["version"],
            }), "gold_document_hit": diagnostics["gold_document_hit"],
            "accepted_gold_span_count": len(spans),
            "maximum_span_coverage_fraction": diagnostics["maximum_span_coverage_fraction"],
            "fully_covered_span_count": diagnostics["fully_covered_span_count"],
            "partial_overlap": diagnostics["partial_overlap"],
            "first_covering_rank": diagnostics["first_covering_rank"],
            "covering_chunk_ids_json": canonical_json(diagnostics["covering_chunk_ids"]),
            "exclusion_reason": None})
    return output
