from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from answerability_rag.answerability.features import (
    agreement_features, context_features, lexical_features, score_features, semantic_features,
)
from answerability_rag.answerability.registry import load_feature_registry


ROOT = Path(__file__).resolve().parents[1]


def test_feature_registry_has_exact_inference_boundary() -> None:
    registry = load_feature_registry(ROOT / "configs/phase04_feature_registry.json")
    assert len(registry.model_features) == 39
    assert len(set(registry.model_features)) == 39
    registry.validate_model_columns(registry.model_features)


@pytest.mark.parametrize("forbidden", [
    "y_suff_strict", "y_suff_semantic", "maximum_span_coverage_fraction",
    "selected_nli_entailment_score", "selected_nli_contradiction_score",
    "benchmark_answer", "human_annotation", "gold_evidence_spans",
])
def test_feature_leakage_guard_rejects_representative_forbidden_fields(forbidden: str) -> None:
    registry = load_feature_registry(ROOT / "configs/phase04_feature_registry.json")
    with pytest.raises(ValueError, match="forbidden"):
        registry.validate_model_columns([registry.model_features[0], forbidden])


def test_feature_leakage_guard_rejects_unknown_fields() -> None:
    registry = load_feature_registry(ROOT / "configs/phase04_feature_registry.json")
    with pytest.raises(ValueError, match="unknown"):
        registry.validate_model_columns(["answer_key_secret_proxy"])


def test_score_aggregates_and_k1_missing_top2() -> None:
    values = score_features([4.0, 1.0, 0.0])
    assert values["retrieval_top1_score"] == 4.0
    assert values["retrieval_top2_score"] == 1.0
    assert values["retrieval_top1_top2_gap"] == 3.0
    assert values["retrieval_score_mean"] == pytest.approx(5 / 3)
    assert values["retrieval_score_range"] == 4.0
    singleton = score_features([2.0])
    assert math.isnan(singleton["retrieval_top2_score"])
    assert math.isnan(singleton["retrieval_top1_top2_gap"])
    assert singleton["retrieval_score_std"] == 0.0


def test_lexical_query_coverage_and_idf_weighting() -> None:
    values = lexical_features(
        ["alpha", "beta", "beta", "gamma"],
        [{"alpha", "x"}, {"gamma", "y"}],
        {"alpha": 1.0, "beta": 2.0, "gamma": 3.0}, oov_idf=9.0,
    )
    assert values["unique_query_token_context_coverage"] == pytest.approx(2 / 3)
    assert values["idf_weighted_query_token_context_coverage"] == pytest.approx(4 / 6)
    assert values["max_chunk_query_lexical_overlap"] == pytest.approx(1 / 3)
    assert values["mean_chunk_query_lexical_overlap"] == pytest.approx(1 / 3)
    assert values["std_chunk_query_lexical_overlap"] == 0.0


def test_semantic_similarity_and_pairwise_redundancy() -> None:
    query = np.array([1.0, 0.0])
    chunks = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    semantic = semantic_features(query, chunks)
    assert semantic["query_chunk_similarity_max"] == 1.0
    assert semantic["query_chunk_similarity_min"] == 0.0
    assert semantic["query_chunk_similarity_top1_top2_gap"] == pytest.approx(0.4)
    context = context_features([10, 20, 30], ["a", "a", "b"], chunks)
    assert context["retrieved_unique_document_count"] == 2
    assert context["dominant_document_chunk_fraction"] == pytest.approx(2 / 3)
    assert context["duplicate_document_ratio"] == pytest.approx(1 / 3)
    assert context["chunk_pairwise_similarity_mean"] == pytest.approx((0 + 0.6 + 0.8) / 3)
    assert context["chunk_pairwise_similarity_max"] == pytest.approx(0.8)


def test_k1_pairwise_redundancy_is_missing() -> None:
    context = context_features([12], ["a"], np.array([[1.0, 0.0]]))
    assert math.isnan(context["chunk_pairwise_similarity_mean"])
    assert math.isnan(context["chunk_pairwise_similarity_max"])


def test_cross_retriever_overlap_and_jaccard() -> None:
    values = agreement_features(["a", "b", "c"], ["b", "c", "d"])
    assert values["bm25_dense_chunk_overlap_count"] == 2
    assert values["bm25_dense_chunk_jaccard"] == 0.5
