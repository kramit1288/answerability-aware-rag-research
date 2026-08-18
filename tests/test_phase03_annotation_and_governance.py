from __future__ import annotations

import pytest

from answerability_rag.sufficiency.annotation import select_annotation_sample
from answerability_rag.sufficiency.evaluation import evaluate_human_annotations
from answerability_rag.sufficiency.governance import (
    build_column_governance, validate_feature_selection,
)


def _label(index: int) -> dict:
    categories = index % 5
    return {
        "question_id": f"q{index:03d}", "example_id": f"e{index:03d}",
        "split": "validation" if index % 4 == 0 else "train",
        "retrieval_strategy": ("bm25", "dense", "hybrid")[index % 3],
        "k": (1, 3, 5, 10)[index % 4], "y_suff": 1 if categories == 0 else 0,
        "label_method": "benchmark_impossible" if categories == 4 else "gold_span_coverage",
        "partial_overlap": categories == 1, "gold_document_hit": categories in {1, 2},
    }


def test_annotation_sample_is_deterministic_question_disjoint_and_test_free() -> None:
    labels = [_label(index) for index in range(250)]
    config = {
        "eligible_splits": ["train", "validation"], "sample_seed": 42, "target_size": 150,
        "version": "v1", "target_strata": {
            "automatic_positive": 30, "partial_overlap": 30,
            "correct_document_insufficient": 30, "wrong_document_retrieval": 30,
            "benchmark_impossible": 30,
        },
    }
    first = select_annotation_sample(labels, config)
    second = select_annotation_sample(list(reversed(labels)), config)
    assert [(row["sample_id"], row["blind_order"]) for row in first] == [
        (row["sample_id"], row["blind_order"]) for row in second
    ]
    assert len(first) == len({row["question_id"] for row in first}) == 150
    assert not any(row["split"] == "test" for row in first)


def test_feature_governance_rejects_gold_and_label_fields() -> None:
    governance = build_column_governance()
    validate_feature_selection(["retrieval_strategy", "k"], governance)
    with pytest.raises(ValueError, match="forbidden"):
        validate_feature_selection(["k", "maximum_span_coverage_fraction", "y_suff"], governance)


def test_human_validation_supports_real_second_annotator_and_pending_state() -> None:
    keys = [
        {"sample_id": "a", "automatic_y_suff": 1},
        {"sample_id": "b", "automatic_y_suff": 0},
    ]
    one = evaluate_human_annotations([
        {"sample_id": "a", "annotator_id": "human-1", "manual_label": "sufficient"},
        {"sample_id": "b", "annotator_id": "human-1", "manual_label": "ambiguous"},
    ], keys)
    assert one["annotator_results"][0]["eligible_non_ambiguous_count"] == 1
    assert one["inter_annotator_agreement"]["cohens_kappa"] is None
    two = evaluate_human_annotations([
        {"sample_id": "a", "annotator_id": "human-1", "manual_label": "sufficient"},
        {"sample_id": "b", "annotator_id": "human-1", "manual_label": "insufficient"},
        {"sample_id": "a", "annotator_id": "human-2", "manual_label": "sufficient"},
        {"sample_id": "b", "annotator_id": "human-2", "manual_label": "sufficient"},
    ], keys)
    agreement = two["inter_annotator_agreement"]
    assert agreement["double_annotated"] == 2
    assert agreement["raw_agreement"] == 0.5
    assert agreement["cohens_kappa"] == 0.0
