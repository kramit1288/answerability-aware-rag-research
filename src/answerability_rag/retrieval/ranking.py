"""Exact deterministic ranking utilities shared by all retrievers."""

from __future__ import annotations

import numpy as np


def stable_top_k(scores: np.ndarray, chunk_ids: list[str], k: int) -> np.ndarray:
    """Return exact top-k indexes sorted by (-score, chunk_id)."""
    values = np.asarray(scores)
    if values.ndim != 1 or len(values) != len(chunk_ids):
        raise ValueError("scores and chunk IDs must be aligned one-dimensional arrays")
    if len(values) < k:
        raise ValueError(f"cannot return {k} unique hits from corpus of {len(values)}")
    if not np.isfinite(values).all():
        raise ValueError("ranking scores contain non-finite values")
    if k == len(values):
        candidates = np.arange(len(values))
    else:
        partition = np.argpartition(values, len(values) - k)[len(values) - k:]
        boundary = values[partition].min()
        above = np.flatnonzero(values > boundary)
        tied = np.flatnonzero(values == boundary)
        tied = np.asarray(sorted(tied, key=lambda index: chunk_ids[int(index)]), dtype=np.int64)
        candidates = np.concatenate((above, tied[: k - len(above)]))
    return np.asarray(
        sorted(candidates, key=lambda index: (-float(values[int(index)]), chunk_ids[int(index)])),
        dtype=np.int64,
    )[:k]
