from __future__ import annotations

import json

from answerability_rag.data.schemas import TechQAQuestion
from answerability_rag.sufficiency.integrity import monotonicity_violations
from answerability_rag.sufficiency.labeling import (
    build_label_rows, interval_coverage, merge_intervals,
)


def test_interval_union_full_partial_and_adjacent_coverage() -> None:
    assert merge_intervals([(0, 5), (5, 10), (8, 12), (20, 25)]) == [(0, 12), (20, 25)]
    assert interval_coverage((2, 10), [(0, 5), (5, 12)]) == 1.0
    assert interval_coverage((0, 10), [(0, 4), (6, 10)]) == 0.8
    assert interval_coverage((0, 10), []) == 0.0


def _config() -> dict:
    return {
        "expected_inputs": {
            "phase01_split_sha256": "p1", "phase02_conditions_semantic_sha256": "p2",
            "phase02_retrieval_config_sha256": "r2",
        },
        "label": {
            "version": "rule", "benchmark_impossible_status": "preliminary_benchmark_negative"
        },
    }


def test_build_labels_distinguishes_full_partial_wrong_document_and_special_statuses() -> None:
    questions = {
        "A": TechQAQuestion("A", "q", "answer", False, ()),
        "I": TechQAQuestion("I", "q", "-", True, ()),
        "U": TechQAQuestion("U", "q", "", False, ()),
    }
    assignments = {
        key: {"split": "train", "split_group_id": key, "reference_status": "available"}
        for key in questions
    }
    chunks = [
        {"chunk_id": "c1", "doc_id": "d", "char_start": 0, "char_end": 7},
        {"chunk_id": "c2", "doc_id": "d", "char_start": 7, "char_end": 15},
        {"chunk_id": "x", "doc_id": "other", "char_start": 0, "char_end": 50},
    ]
    alignments = [{
        "question_id": "A", "gold_alignment_status": "aligned_exact", "gold_doc_id": "d",
        "source_char_start": 5, "source_char_end": 12, "alignment_id": "a1",
    }]
    conditions = [
        {"question_id": "A", "retrieval_strategy": "bm25", "k": 1,
         "ordered_chunk_ids_json": json.dumps(["c1"])},
        {"question_id": "A", "retrieval_strategy": "bm25", "k": 3,
         "ordered_chunk_ids_json": json.dumps(["c1", "c2", "x"])},
        {"question_id": "I", "retrieval_strategy": "bm25", "k": 1,
         "ordered_chunk_ids_json": json.dumps(["x"])},
        {"question_id": "U", "retrieval_strategy": "bm25", "k": 1,
         "ordered_chunk_ids_json": json.dumps(["x"])},
    ]
    rows = build_label_rows(
        conditions, chunks, questions, assignments, alignments, run_id="run", config=_config(),
        phase03_config_sha256="p3",
    )
    by_question_k = {(row["question_id"], row["k"]): row for row in rows}
    assert by_question_k[("A", 1)]["y_suff"] == 0
    assert by_question_k[("A", 1)]["partial_overlap"] is True
    assert by_question_k[("A", 3)]["y_suff"] == 1
    assert by_question_k[("A", 3)]["first_covering_rank"] == 2
    assert json.loads(by_question_k[("A", 3)]["covering_chunk_ids_json"]) == ["c1", "c2"]
    assert by_question_k[("I", 1)]["label_status"] == "preliminary_benchmark_negative"
    assert by_question_k[("I", 1)]["gold_document_hit"] is None
    assert by_question_k[("U", 1)]["y_suff"] is None
    assert by_question_k[("U", 1)]["exclusion_reason"] == "empty_answerable_reference"


def test_monotonicity_check_detects_but_never_repairs_violation() -> None:
    rows = [
        {"question_id": "q", "retrieval_strategy": "dense", "k": 1,
         "label_method": "gold_span_coverage", "y_suff": 1},
        {"question_id": "q", "retrieval_strategy": "dense", "k": 3,
         "label_method": "gold_span_coverage", "y_suff": 0},
    ]
    assert monotonicity_violations(rows) == [("q", "dense")]
    assert [row["y_suff"] for row in rows] == [1, 0]
