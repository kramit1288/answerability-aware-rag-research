"""Deterministic inference-time feature construction for Phase 4."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Mapping, Sequence

import numpy as np

from answerability_rag.retrieval.tokenization import tokenize_bm25


def _fraction(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _population(values: Sequence[float]) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0)), float(array.min()), float(array.max())


def score_features(scores: Sequence[float]) -> dict[str, float]:
    if not scores:
        raise ValueError("retrieved bundle must contain at least one score")
    mean, std, minimum, maximum = _population(scores)
    top2 = float(scores[1]) if len(scores) >= 2 else math.nan
    return {
        "retrieval_top1_score": float(scores[0]),
        "retrieval_top2_score": top2,
        "retrieval_top1_top2_gap": float(scores[0] - scores[1]) if len(scores) >= 2 else math.nan,
        "retrieval_score_mean": mean,
        "retrieval_score_std": std,
        "retrieval_score_min": minimum,
        "retrieval_score_max": maximum,
        "retrieval_score_range": maximum - minimum,
    }


def lexical_features(
    query_tokens: Sequence[str], chunk_token_sets: Sequence[set[str]], idf: Mapping[str, float],
    *, oov_idf: float,
) -> dict[str, float]:
    unique_query = set(query_tokens)
    context_tokens = set().union(*chunk_token_sets) if chunk_token_sets else set()
    covered = unique_query.intersection(context_tokens)
    denominator = sum(float(idf.get(token, oov_idf)) for token in unique_query)
    weighted = sum(float(idf.get(token, oov_idf)) for token in covered)
    overlaps = [_fraction(len(unique_query.intersection(tokens)), len(unique_query))
                for tokens in chunk_token_sets]
    mean, std, _, maximum = _population(overlaps)
    return {
        "unique_query_token_context_coverage": _fraction(len(covered), len(unique_query)),
        "idf_weighted_query_token_context_coverage": _fraction(weighted, denominator),
        "max_chunk_query_lexical_overlap": maximum,
        "mean_chunk_query_lexical_overlap": mean,
        "std_chunk_query_lexical_overlap": std,
    }


def semantic_features(query_embedding: np.ndarray, chunk_embeddings: np.ndarray) -> dict[str, float]:
    similarities = np.asarray(chunk_embeddings @ query_embedding, dtype=np.float64)
    mean, std, minimum, maximum = _population(similarities)
    ordered = np.sort(similarities)[::-1]
    return {
        "query_chunk_similarity_max": maximum,
        "query_chunk_similarity_mean": mean,
        "query_chunk_similarity_min": minimum,
        "query_chunk_similarity_std": std,
        "query_chunk_similarity_top1_top2_gap": (
            float(ordered[0] - ordered[1]) if len(ordered) >= 2 else math.nan
        ),
    }


def context_features(
    token_lengths: Sequence[int], doc_ids: Sequence[str], chunk_embeddings: np.ndarray,
) -> dict[str, float | int]:
    if not token_lengths or len(token_lengths) != len(doc_ids):
        raise ValueError("context bundle arrays must be nonempty and aligned")
    lengths = np.asarray(token_lengths, dtype=np.float64)
    counts = Counter(doc_ids)
    pairwise_mean = math.nan
    pairwise_max = math.nan
    if len(chunk_embeddings) >= 2:
        matrix = np.asarray(chunk_embeddings @ chunk_embeddings.T, dtype=np.float64)
        upper = matrix[np.triu_indices(len(matrix), k=1)]
        pairwise_mean, pairwise_max = float(upper.mean()), float(upper.max())
    return {
        "retrieved_chunk_count": len(token_lengths),
        "retrieved_total_token_count": int(lengths.sum()),
        "retrieved_mean_chunk_token_count": float(lengths.mean()),
        "retrieved_std_chunk_token_count": float(lengths.std(ddof=0)),
        "retrieved_unique_document_count": len(counts),
        "dominant_document_chunk_fraction": max(counts.values()) / len(doc_ids),
        "duplicate_document_ratio": 1.0 - len(counts) / len(doc_ids),
        "chunk_pairwise_similarity_mean": pairwise_mean,
        "chunk_pairwise_similarity_max": pairwise_max,
    }


def agreement_features(bm25_chunk_ids: Sequence[str], dense_chunk_ids: Sequence[str]) -> dict[str, float | int]:
    left, right = set(bm25_chunk_ids), set(dense_chunk_ids)
    intersection = len(left.intersection(right))
    union = len(left.union(right))
    return {
        "bm25_dense_chunk_overlap_count": intersection,
        "bm25_dense_chunk_jaccard": _fraction(intersection, union),
    }


def query_features(question: str, *, identifier_regex: str) -> dict[str, float | int]:
    tokens = tokenize_bm25(question)
    unique = set(tokens)
    digits = sum(token.isdigit() for token in tokens)
    identifier = re.compile(identifier_regex)
    identifiers = sum(identifier.fullmatch(token) is not None for token in tokens)
    return {
        "query_char_count": len(question),
        "query_token_count": len(tokens),
        "query_unique_token_count": len(unique),
        "query_unique_token_fraction": _fraction(len(unique), len(tokens)),
        "query_digit_token_count": digits,
        "query_digit_token_fraction": _fraction(digits, len(tokens)),
        "query_identifier_like_token_count": identifiers,
        "query_identifier_like_token_fraction": _fraction(identifiers, len(tokens)),
    }


def condition_features(
    *, question: str, scores: Sequence[float], chunk_texts: Sequence[str],
    token_lengths: Sequence[int], doc_ids: Sequence[str], query_embedding: np.ndarray,
    chunk_embeddings: np.ndarray, bm25_chunk_ids: Sequence[str], dense_chunk_ids: Sequence[str],
    idf: Mapping[str, float], oov_idf: float, identifier_regex: str,
    retrieval_strategy: str, k: int,
) -> dict[str, float | int | str]:
    """Compute the frozen 39-feature vector without answer/gold/label inputs."""
    query_tokens = tokenize_bm25(question)
    result: dict[str, float | int | str] = {}
    result.update(query_features(question, identifier_regex=identifier_regex))
    result.update(score_features(scores))
    result.update(lexical_features(
        query_tokens, [set(tokenize_bm25(text)) for text in chunk_texts], idf, oov_idf=oov_idf,
    ))
    result.update(semantic_features(query_embedding, chunk_embeddings))
    result.update(context_features(token_lengths, doc_ids, chunk_embeddings))
    result.update(agreement_features(bm25_chunk_ids, dense_chunk_ids))
    result["retrieval_strategy"] = retrieval_strategy
    result["k"] = int(k)
    return result
