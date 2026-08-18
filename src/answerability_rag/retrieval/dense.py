"""Resumable float32 embedding and exact normalized-dot retrieval."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from answerability_rag.retrieval.cache import validate_cache_metadata, write_json_atomic
from answerability_rag.retrieval.ranking import stable_top_k


def validate_completed_embeddings(
    embeddings: np.ndarray, completed_batches: set[int], *, batch_size: int,
    row_count: int, dimension: int, atol: float = 1e-5,
) -> dict[str, float | int]:
    """Validate every row covered by the persisted completed-batch bitmap."""
    if embeddings.shape != (row_count, dimension) or embeddings.dtype != np.float32:
        raise ValueError(
            f"embedding checkpoint array mismatch: shape={embeddings.shape}, dtype={embeddings.dtype}"
        )
    checked = 0
    min_norm = float("inf")
    max_norm = float("-inf")
    for batch in sorted(completed_batches):
        start, end = batch * batch_size, min((batch + 1) * batch_size, row_count)
        if start >= row_count or batch < 0:
            raise ValueError(f"invalid completed batch index {batch}")
        values = np.asarray(embeddings[start:end])
        if not np.isfinite(values).all():
            raise ValueError(f"completed embedding batch {batch} contains non-finite values")
        norms = np.linalg.norm(values, axis=1)
        if not np.allclose(norms, 1.0, atol=atol):
            raise ValueError(
                f"completed embedding batch {batch} is not unit normalized: "
                f"min={norms.min()}, max={norms.max()}"
            )
        checked += end - start
        min_norm = min(min_norm, float(norms.min()))
        max_norm = max(max_norm, float(norms.max()))
    return {
        "completed_batches": len(completed_batches), "completed_rows": checked,
        "minimum_l2_norm": min_norm if checked else 0.0,
        "maximum_l2_norm": max_norm if checked else 0.0,
    }


def encode_resumable(
    model: SentenceTransformer, texts: list[str], embedding_path: Path, metadata_path: Path,
    *, key: str, batch_size: int, dimension: int,
) -> tuple[np.memmap, bool]:
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    resumed = False
    completed: set[int] = set()
    if embedding_path.exists() or metadata_path.exists():
        metadata = validate_cache_metadata(metadata_path, expected_key=key, expected_kind="dense")
        if metadata.get("shape") != [len(texts), dimension] or not embedding_path.exists():
            raise ValueError("dense cache shape/file mismatch")
        if metadata.get("dtype") != "float32" or int(metadata.get("batch_size", -1)) != batch_size:
            raise ValueError("dense cache dtype/batch-size mismatch")
        expected_bytes = len(texts) * dimension * np.dtype(np.float32).itemsize
        if embedding_path.stat().st_size != expected_bytes:
            raise ValueError(
                f"dense cache byte-size mismatch: {embedding_path.stat().st_size} != {expected_bytes}"
            )
        completed_values = [int(value) for value in metadata.get("completed_batches", [])]
        if len(completed_values) != len(set(completed_values)):
            raise ValueError("dense completed-batch bitmap contains duplicates")
        completed = set(completed_values)
        if sorted(completed) != list(range(len(completed))):
            raise ValueError("dense completed-batch bitmap must be a contiguous prefix")
        resumed = bool(completed)
        embeddings = np.memmap(embedding_path, dtype=np.float32, mode="r+", shape=(len(texts), dimension))
        validation = validate_completed_embeddings(
            embeddings, completed, batch_size=batch_size, row_count=len(texts), dimension=dimension
        )
        print(
            "validated embedding checkpoint: "
            f"{validation['completed_rows']}/{len(texts)} rows, "
            f"norms=[{validation['minimum_l2_norm']:.8f}, {validation['maximum_l2_norm']:.8f}]",
            flush=True,
        )
    else:
        embeddings = np.memmap(embedding_path, dtype=np.float32, mode="w+", shape=(len(texts), dimension))
        metadata = {"kind": "dense", "cache_key": key, "shape": [len(texts), dimension],
                    "dtype": "float32", "batch_size": batch_size, "completed_batches": []}
        write_json_atomic(metadata_path, metadata)
    batch_count = (len(texts) + batch_size - 1) // batch_size
    initial_completed_rows = sum(
        min((batch + 1) * batch_size, len(texts)) - batch * batch_size for batch in completed
    )
    resume_started = time.perf_counter()
    last_progress = resume_started
    for batch in range(batch_count):
        if batch in completed:
            continue
        start, end = batch * batch_size, min((batch + 1) * batch_size, len(texts))
        values = model.encode(
            texts[start:end], batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        ).astype(np.float32, copy=False)
        if values.shape != (end - start, dimension):
            raise ValueError(f"unexpected embedding batch shape {values.shape}")
        norms = np.linalg.norm(values, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5):
            raise ValueError("model returned non-normalized embeddings")
        embeddings[start:end] = values
        embeddings.flush()
        completed.add(batch)
        metadata["completed_batches"] = sorted(completed)
        write_json_atomic(metadata_path, metadata)
        now = time.perf_counter()
        if now - last_progress >= 60 or len(completed) == batch_count:
            completed_rows = sum(
                min((value + 1) * batch_size, len(texts)) - value * batch_size
                for value in completed
            )
            session_rows = completed_rows - initial_completed_rows
            elapsed = max(now - resume_started, 1e-9)
            rate = session_rows / elapsed
            remaining = len(texts) - completed_rows
            eta_seconds = remaining / rate if rate > 0 else float("inf")
            print(
                f"embedding progress: {completed_rows}/{len(texts)} chunks; "
                f"{rate:.3f} chunks/sec; ETA {eta_seconds / 3600:.2f} hours",
                flush=True,
            )
            last_progress = now
    final_validation = validate_completed_embeddings(
        embeddings, completed, batch_size=batch_size, row_count=len(texts), dimension=dimension
    )
    if final_validation["completed_rows"] != len(texts):
        raise ValueError(
            f"embedding cache incomplete after encode: {final_validation['completed_rows']}/{len(texts)}"
        )
    return embeddings, resumed


def exact_search(
    query_embeddings: np.ndarray, corpus_embeddings: np.ndarray, chunk_ids: list[str], depth: int,
) -> list[list[tuple[int, float]]]:
    scores = np.asarray(query_embeddings, dtype=np.float32) @ np.asarray(corpus_embeddings, dtype=np.float32).T
    output = []
    for row in scores:
        indexes = stable_top_k(row, chunk_ids, depth)
        output.append([(int(index), float(row[index])) for index in indexes])
    return output
