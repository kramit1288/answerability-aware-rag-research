"""Fail-fast interoperability smoke test for the frozen Phase 2 stack."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow
import scipy
import torch
import transformers
from huggingface_hub import HfApi
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from answerability_rag.retrieval.tokenization import tokenize_bm25


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs" / "phase02_retrieval.json").read_text(encoding="utf-8"))


def version(distribution: str) -> str:
    return importlib.metadata.version(distribution)


def main() -> None:
    dense = CONFIG["dense"]
    resolved = HfApi().model_info(dense["model_name"], revision=dense["model_revision"]).sha
    if resolved != dense["model_revision"]:
        raise RuntimeError(f"model revision mismatch: expected {dense['model_revision']}, got {resolved}")
    cache = ROOT / "data" / "derived" / "phase02" / "model_cache"
    model = SentenceTransformer(
        dense["model_name"], revision=dense["model_revision"], device="cpu", cache_folder=str(cache)
    )
    vectors = model.encode(
        ["reset error code x-42", "configure server port"],
        batch_size=2,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32, copy=False)
    norms = np.linalg.norm(vectors, axis=1)
    if vectors.dtype != np.float32 or not np.allclose(norms, 1.0, atol=1e-6):
        raise RuntimeError(f"dense normalization failed: dtype={vectors.dtype}, norms={norms}")
    scores = vectors @ vectors.T
    ranks = np.argsort(-scores, axis=1, kind="stable")
    if ranks.shape != (2, 2) or not np.array_equal(ranks[:, 0], np.array([0, 1])):
        raise RuntimeError(f"tiny exact search failed: {ranks.tolist()}")

    corpus = [
        tokenize_bm25("reset error-code x-42"),
        tokenize_bm25("configure server port"),
        tokenize_bm25("database backup schedule"),
    ]
    bm25 = BM25Okapi(corpus, k1=1.5, b=0.75, epsilon=0.25)
    bm25_scores = bm25.get_scores(tokenize_bm25("server port"))
    if int(np.argmax(bm25_scores)) != 1:
        raise RuntimeError(f"tiny BM25 search failed: {bm25_scores.tolist()}")

    packages = {
        name: version(name)
        for name in [
            "numpy", "scipy", "pandas", "torch", "transformers", "sentence-transformers",
            "rank-bm25", "pyarrow", "scikit-learn", "huggingface-hub", "tokenizers",
            "safetensors",
        ]
    }
    manifest = {
        "schema_version": "phase02-dependencies-v1",
        "status": "pass",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
        "model_name": dense["model_name"],
        "requested_model_revision": dense["model_revision"],
        "resolved_model_revision": resolved,
        "tokenizer_revision": CONFIG["chunking"]["tokenizer_revision"],
        "device": str(model.device),
        "dtype": "float32",
        "embedding_dimension": int(model.get_sentence_embedding_dimension()),
        "max_sequence_length": int(model.max_seq_length),
        "normalize_embeddings": True,
        "similarity": "exact float32 dot product",
        "tiny_dense_scores": scores.tolist(),
        "tiny_bm25_scores": bm25_scores.tolist(),
        "imports": {
            "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "pyarrow": pyarrow.__version__,
        },
    }
    destination = ROOT / "artifacts" / "results" / "phase02_dependency_manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
