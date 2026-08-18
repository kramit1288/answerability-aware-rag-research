"""Frozen rank-bm25 indexing, caching, and deterministic search."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from answerability_rag.retrieval.cache import validate_cache_metadata, write_json_atomic
from answerability_rag.retrieval.ranking import stable_top_k
from answerability_rag.retrieval.tokenization import tokenize_bm25


def build_bm25(texts: list[str], *, k1: float, b: float, epsilon: float) -> BM25Okapi:
    return BM25Okapi([tokenize_bm25(text) for text in texts], k1=k1, b=b, epsilon=epsilon)


def load_or_build_bm25(
    texts: list[str], cache_path: Path, metadata_path: Path, *, key: str,
    k1: float, b: float, epsilon: float,
) -> tuple[BM25Okapi, bool]:
    if cache_path.exists() or metadata_path.exists():
        validate_cache_metadata(metadata_path, expected_key=key, expected_kind="bm25")
        if not cache_path.exists():
            raise ValueError("BM25 metadata exists without index")
        with cache_path.open("rb") as handle:
            index = pickle.load(handle)
        if index.corpus_size != len(texts):
            raise ValueError("BM25 cached corpus size mismatch")
        return index, True
    index = build_bm25(texts, k1=k1, b=b, epsilon=epsilon)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(index, handle, protocol=5)
    temporary.replace(cache_path)
    write_json_atomic(metadata_path, {"kind": "bm25", "cache_key": key, "corpus_size": len(texts)})
    return index, False


def search_bm25(index: BM25Okapi, query: str, chunk_ids: list[str], depth: int) -> list[tuple[int, float]]:
    scores = np.asarray(index.get_scores(tokenize_bm25(query)), dtype=np.float64)
    indexes = stable_top_k(scores, chunk_ids, depth)
    return [(int(index_), float(scores[index_])) for index_ in indexes]
