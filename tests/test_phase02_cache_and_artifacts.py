import json
from pathlib import Path

import numpy as np
import pytest

from answerability_rag.retrieval.artifacts import write_canonical_parquet
from answerability_rag.retrieval.cache import cache_key, validate_cache_metadata
from answerability_rag.retrieval.dense import encode_resumable, validate_completed_embeddings


class FixtureModel:
    def encode(self, texts, **kwargs):
        values = np.array([[len(text), 1.0] for text in texts], dtype=np.float32)
        return values / np.linalg.norm(values, axis=1, keepdims=True)


def test_cache_key_and_stale_cache_rejection(tmp_path: Path) -> None:
    key = cache_key("dense", {"model": "m", "corpus": "c"})
    metadata = tmp_path / "cache.json"
    metadata.write_text(json.dumps({"kind": "dense", "cache_key": key}), encoding="utf-8")
    assert validate_cache_metadata(metadata, expected_key=key, expected_kind="dense")["cache_key"] == key
    with pytest.raises(ValueError, match="stale/incompatible"):
        validate_cache_metadata(metadata, expected_key="other", expected_kind="dense")


def test_resumable_embeddings_skip_completed_batches(tmp_path: Path) -> None:
    path, metadata = tmp_path / "vectors.f32", tmp_path / "vectors.json"
    first, resumed = encode_resumable(FixtureModel(), ["a", "bb", "ccc"], path, metadata,
                                      key="key", batch_size=2, dimension=2)
    expected = np.array(first)
    del first
    second, resumed = encode_resumable(FixtureModel(), ["a", "bb", "ccc"], path, metadata,
                                       key="key", batch_size=2, dimension=2)
    assert resumed is True
    assert np.array_equal(np.array(second), expected)


def test_completed_embedding_validation_rejects_nonfinite_or_nonunit_values() -> None:
    good = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    result = validate_completed_embeddings(
        good, {0}, batch_size=2, row_count=2, dimension=2
    )
    assert result["completed_rows"] == 2
    bad = good.copy()
    bad[1] = [2.0, 0.0]
    with pytest.raises(ValueError, match="not unit normalized"):
        validate_completed_embeddings(bad, {0}, batch_size=2, row_count=2, dimension=2)


def test_parquet_is_byte_and_semantic_deterministic_after_canonical_sort(tmp_path: Path) -> None:
    rows = [{"id": "b", "value": 2}, {"id": "a", "value": 1}]
    left = write_canonical_parquet(tmp_path / "left.parquet", rows, ("id", "value"), ("id",))
    right = write_canonical_parquet(tmp_path / "right.parquet", list(reversed(rows)),
                                    ("id", "value"), ("id",))
    assert left["semantic_sha256"] == right["semantic_sha256"]
    assert left["physical_sha256"] == right["physical_sha256"]
