from __future__ import annotations

import pytest

from answerability_rag.sufficiency.annotation import select_annotation_sample
from answerability_rag.sufficiency.evaluation import (
    development_stratum_frequencies,
    evaluate_human_annotations,
)
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
    strata = (
        "automatic_positive", "partial_overlap", "correct_document_insufficient",
        "wrong_document_retrieval", "benchmark_impossible",
    )
    keys = [
        {"sample_id": str(index), "automatic_y_suff": int(index == 0),
         "sampling_stratum": stratum}
        for index, stratum in enumerate(strata)
    ]
    population = {stratum: 10 for stratum in strata}
    one_rows = [
        {"sample_id": str(index), "annotator_id": "human-1",
         "manual_label": "sufficient" if index == 0 else "insufficient",
         "annotation_timestamp": "2026-08-19T12:00:00Z"}
        for index in range(5)
    ]
    one = evaluate_human_annotations(one_rows, keys, population, primary_annotator_id="human-1")
    assert one["annotator_results"][0]["eligible_non_ambiguous_count"] == 5
    assert one["annotator_results"][0]["prevalence_weighted_metrics"]["f1"] == 1.0
    assert one["predeclared_gate_report"]["automatic_label_quality_status"] == "pass"
    assert one["inter_annotator_agreement"]["cohens_kappa"] is None
    two_rows = one_rows + [
        {"sample_id": "0", "annotator_id": "human-2", "manual_label": "sufficient",
         "annotation_timestamp": "2026-08-19T13:00:00Z"},
        {"sample_id": "1", "annotator_id": "human-2", "manual_label": "sufficient",
         "annotation_timestamp": "2026-08-19T13:01:00Z"},
    ]
    two = evaluate_human_annotations(
        two_rows, keys, population, primary_annotator_id="human-1",
    )
    agreement = two["inter_annotator_agreement"]
    assert agreement["double_annotated"] == 2
    assert agreement["raw_agreement"] == 0.5
    assert agreement["cohens_kappa"] == 0.0
    assert agreement["overlap_limitation"] is not None


def test_prevalence_weighting_uses_development_stratum_frequencies() -> None:
    strata = (
        "automatic_positive", "partial_overlap", "correct_document_insufficient",
        "wrong_document_retrieval", "benchmark_impossible",
    )
    population = dict(zip(strata, (50, 10, 10, 20, 10)))
    keys = []
    annotations = []
    for stratum_index, stratum in enumerate(strata):
        for index in range(10):
            sample_id = f"{stratum_index}-{index}"
            automatic = int(stratum == "automatic_positive")
            manual = "sufficient" if (
                (stratum == "automatic_positive" and index < 9)
                or (stratum == "partial_overlap" and index < 2)
                or (stratum == "benchmark_impossible" and index == 0)
            ) else "insufficient"
            keys.append({
                "sample_id": sample_id, "automatic_y_suff": automatic,
                "sampling_stratum": stratum,
            })
            annotations.append({
                "sample_id": sample_id, "annotator_id": "human-1", "manual_label": manual,
                "annotation_timestamp": "2026-08-19T12:00:00Z",
            })
    report = evaluate_human_annotations(
        annotations, keys, population, primary_annotator_id="human-1",
    )
    result = report["annotator_results"][0]
    assert result["automatic_sufficient_precision"] == 0.9
    assert result["prevalence_weighted_metrics"]["estimated_confusion_proportions"] == {
        "tn": pytest.approx(0.47), "fp": pytest.approx(0.05),
        "fn": pytest.approx(0.03), "tp": pytest.approx(0.45),
    }
    assert result["prevalence_weighted_metrics"]["f1"] == pytest.approx(0.9183673469)
    assert report["predeclared_gate_report"][
        "benchmark_impossible_human_sufficient_rate"
    ]["status"] == "within_predeclared_gate"
    assert "not a natural development-population estimate" in report[
        "stratified_sample_interpretation"
    ]


def test_benchmark_impossible_gate_and_blinding_are_enforced() -> None:
    strata = (
        "automatic_positive", "partial_overlap", "correct_document_insufficient",
        "wrong_document_retrieval", "benchmark_impossible",
    )
    population = {stratum: 1 for stratum in strata}
    keys = [
        {"sample_id": str(index), "automatic_y_suff": int(index == 0),
         "sampling_stratum": stratum}
        for index, stratum in enumerate(strata)
    ]
    annotations = [
        {"sample_id": str(index), "annotator_id": "human-1",
         "manual_label": "sufficient" if index in {0, 4} else "insufficient",
         "annotation_timestamp": "2026-08-19T12:00:00Z"}
        for index in range(5)
    ]
    report = evaluate_human_annotations(
        annotations, keys, population, primary_annotator_id="human-1",
    )
    gate = report["predeclared_gate_report"]["benchmark_impossible_human_sufficient_rate"]
    assert gate["status"] == "exclusion_required"
    assert "exclude benchmark-impossible" in gate["required_phase4_action"]
    with pytest.raises(ValueError, match="violates blinding"):
        evaluate_human_annotations(
            [{**annotations[0], "automatic_y_suff": 1}], keys, population,
            primary_annotator_id="human-1",
        )


def test_development_stratum_frequencies_exclude_test_and_unresolved() -> None:
    rows = [_label(index) for index in range(5)]
    rows.append({**_label(5), "split": "test"})
    rows.append({**_label(6), "y_suff": None})
    counts = development_stratum_frequencies(rows)
    assert sum(counts.values()) == 5
    assert set(counts) == {
        "automatic_positive", "partial_overlap", "correct_document_insufficient",
        "wrong_document_retrieval", "benchmark_impossible",
    }
