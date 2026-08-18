import numpy as np

from answerability_rag.retrieval.bm25 import build_bm25, search_bm25
from answerability_rag.retrieval.dense import exact_search
from answerability_rag.retrieval.hybrid import reciprocal_rank_fusion
from answerability_rag.retrieval.ranking import stable_top_k


def test_bm25_known_ranking_and_fixed_tokenizer() -> None:
    texts = ["reset x-42 error", "configure server:8080", "database backup"]
    ids = ["c", "b", "a"]
    index = build_bm25(texts, k1=1.5, b=0.75, epsilon=0.25)
    hits = search_bm25(index, "SERVER:8080", ids, 3)
    assert hits[0][0] == 1
    assert hits[0][1] > hits[1][1]
    assert len({index for index, _ in hits}) == 3


def test_stable_score_ties_use_chunk_id() -> None:
    assert stable_top_k(np.array([1.0, 1.0, 0.0]), ["z", "a", "m"], 2).tolist() == [1, 0]


def test_dense_exact_similarity_normalization_and_ties() -> None:
    corpus = np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float32)
    query = np.array([[1, 0]], dtype=np.float32)
    assert np.allclose(np.linalg.norm(corpus, axis=1), 1.0)
    hits = exact_search(query, corpus, ["z", "x", "a"], 3)[0]
    assert [index for index, _ in hits] == [2, 0, 1]


def test_rrf_expected_ranking_constituent_provenance_and_tie() -> None:
    bm25 = [(0, 9.0), (1, 8.0), (2, 7.0)]
    dense = [(1, 0.9), (0, 0.8), (3, 0.7)]
    hits = reciprocal_rank_fusion(bm25, dense, ["z", "a", "b", "c"], constant=60, depth=4)
    assert [hit.chunk_index for hit in hits[:2]] == [1, 0]  # exact RRF tie -> chunk_id "a" first
    assert hits[1].bm25_rank == 1 and hits[1].dense_rank == 2
    assert hits[2].bm25_score == 7.0 and hits[2].dense_rank is None
    assert len({hit.chunk_index for hit in hits}) == 4
