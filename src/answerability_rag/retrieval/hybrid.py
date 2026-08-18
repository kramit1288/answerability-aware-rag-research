"""Frozen equal-weight reciprocal-rank fusion."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HybridHit:
    chunk_index: int
    score: float
    bm25_rank: int | None
    bm25_score: float | None
    dense_rank: int | None
    dense_score: float | None


def reciprocal_rank_fusion(
    bm25_hits: list[tuple[int, float]], dense_hits: list[tuple[int, float]],
    chunk_ids: list[str], *, constant: int = 60, depth: int = 10,
) -> list[HybridHit]:
    bm = {index: (rank, score) for rank, (index, score) in enumerate(bm25_hits, 1)}
    de = {index: (rank, score) for rank, (index, score) in enumerate(dense_hits, 1)}
    hits = []
    for index in bm.keys() | de.keys():
        bm_value, de_value = bm.get(index), de.get(index)
        score = (1.0 / (constant + bm_value[0]) if bm_value else 0.0) + (
            1.0 / (constant + de_value[0]) if de_value else 0.0
        )
        hits.append(HybridHit(index, score,
                              bm_value[0] if bm_value else None, bm_value[1] if bm_value else None,
                              de_value[0] if de_value else None, de_value[1] if de_value else None))
    hits.sort(key=lambda hit: (-hit.score, chunk_ids[hit.chunk_index]))
    return hits[:depth]
