"""End-to-end deterministic Phase 2 controlled-retrieval orchestration."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from answerability_rag.data.techqa import load_techqa_rows
from answerability_rag.hashing import canonical_json, canonical_json_sha256, sha256_file
from answerability_rag.retrieval.artifacts import (
    read_parquet_records, semantic_records_sha256, write_canonical_parquet, write_json,
)
from answerability_rag.retrieval.bm25 import load_or_build_bm25, search_bm25
from answerability_rag.retrieval.cache import cache_key, validate_cache_metadata, write_json_atomic
from answerability_rag.retrieval.chunking import CHUNK_FIELDS, ChunkSpec, build_chunk_corpus, load_corpus_manifest
from answerability_rag.retrieval.config import Phase02Config
from answerability_rag.retrieval.dense import encode_resumable, exact_search
from answerability_rag.retrieval.hybrid import HybridHit, reciprocal_rank_fusion
from answerability_rag.retrieval.integrity import validate_retrieval_outputs
from answerability_rag.retrieval.metrics import document_metrics_for_ranking, summarize_development_metrics


RANKED_FIELDS = (
    "schema_version", "run_id", "question_id", "split", "provenance_partition",
    "retrieval_strategy", "rank", "chunk_id", "doc_id", "filename", "strategy_score",
    "bm25_score", "bm25_rank", "dense_score", "dense_rank", "is_confirmed_gold_document",
    "confirmed_gold_eligible", "unresolved_reference", "corpus_semantic_sha256",
    "chunk_config_sha256", "retrieval_config_sha256",
)
CONDITION_FIELDS = (
    "schema_version", "run_id", "question_id", "split", "provenance_partition",
    "retrieval_strategy", "k", "ordered_chunk_ids_json", "ordered_doc_ids_json",
    "ordered_raw_scores_json", "gold_doc_ids_json", "confirmed_gold_eligible",
    "unresolved_reference", "first_gold_rank", "doc_recall_at_k", "reciprocal_rank_at_k",
    "reciprocal_rank_at_10", "corpus_semantic_sha256", "chunk_config_sha256",
    "retrieval_config_sha256",
)
METRIC_FIELDS = (
    "schema_version", "split", "retrieval_strategy", "k", "eligible_questions",
    "document_recall_at_k", "mrr_at_10", "limitation",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows([{field: row.get(field) for field in fields} for row in rows])


def _git_sha(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "HEAD"],
        cwd=root, text=True,
    ).strip()


def _verify_phase01(root: Path, config: Phase02Config) -> dict:
    expected = config.values["expected_phase01"]
    split_hash = (root / "artifacts/data/techqa_split.sha256").read_text(encoding="ascii").strip()
    if split_hash != config.values["phase01_split_semantic_sha256"]:
        raise ValueError(f"Phase 1 split hash drift: {split_hash}")
    assignments = _read_csv(root / "artifacts/data/techqa_split_assignments.csv")
    corpus = _read_csv(root / "artifacts/data/techqa_corpus_manifest.csv")
    components = _read_csv(root / "artifacts/data/techqa_split_components.csv")
    counts = Counter(row["split"] for row in assignments)
    observed = {"questions": len(assignments), "corpus_documents": len(corpus),
                "components": len(components), "split_questions": dict(counts)}
    if observed != expected:
        raise ValueError(f"Phase 1 frozen counts drift: expected={expected}, observed={observed}")
    return {"split_sha256": split_hash, "counts": observed, "git_commit_sha": _git_sha(root)}


def _load_or_build_chunks(root: Path, config: Phase02Config, tokenizer) -> tuple[list[dict], dict, bool]:
    manifest_path = root / "artifacts/data/techqa_corpus_manifest.csv"
    corpus_manifest_hash = sha256_file(manifest_path)
    spec_values = config.values["chunking"]
    spec = ChunkSpec(
        tokenizer_name=spec_values["tokenizer_name"], tokenizer_revision=spec_values["tokenizer_revision"],
        max_content_tokens=spec_values["max_content_tokens"], overlap_tokens=spec_values["overlap_tokens"],
        step_tokens=spec_values["step_tokens"], version=spec_values["version"],
    )
    key = cache_key("chunks", {"corpus_manifest_sha256": corpus_manifest_hash,
                                "chunk_config_sha256": spec.sha256})
    path = root / "artifacts/data/techqa_chunk_manifest.parquet"
    metadata_path = root / "data/derived/phase02/chunks/cache.json"
    if path.exists() or metadata_path.exists():
        metadata = validate_cache_metadata(metadata_path, expected_key=key, expected_kind="chunks")
        if not path.exists() or sha256_file(path) != metadata["artifact"]["physical_sha256"]:
            raise ValueError("chunk artifact missing or physical hash mismatch")
        chunks = read_parquet_records(path)
        if len(chunks) != metadata["artifact"]["rows"]:
            raise ValueError("chunk row count mismatch")
        semantic_sha256 = semantic_records_sha256(chunks, CHUNK_FIELDS)
        if semantic_sha256 != metadata["artifact"]["semantic_sha256"]:
            raise ValueError(
                "chunk semantic hash mismatch: "
                f"expected {metadata['artifact']['semantic_sha256']}, got {semantic_sha256}"
            )
        if metadata.get("chunk_config_sha256") != spec.sha256:
            raise ValueError("chunk configuration hash mismatch")
        print(
            "validated chunk corpus: "
            f"physical={metadata['artifact']['physical_sha256']}, "
            f"semantic={semantic_sha256}, config={spec.sha256}",
            flush=True,
        )
        return chunks, metadata, True
    phase01 = json.loads((root / "configs/phase01_data.json").read_text(encoding="utf-8"))
    revision = phase01["techqa"]["revision"]
    corpus_root = root / f"data/derived/techqa/{revision}/corpus"
    manifest = load_corpus_manifest(manifest_path)
    chunks, zero_docs = build_chunk_corpus(manifest, corpus_root, tokenizer, spec, corpus_manifest_hash)
    artifact = write_canonical_parquet(path, chunks, CHUNK_FIELDS, ("chunk_id",))
    metadata = {"kind": "chunks", "cache_key": key, "artifact": artifact,
                "chunk_config_sha256": spec.sha256, "corpus_manifest_sha256": corpus_manifest_hash,
                "zero_chunk_documents": zero_docs}
    write_json_atomic(metadata_path, metadata)
    return chunks, metadata, False


def _ranking_rows_for_strategy(question: dict, strategy: str, hits, chunks: list[dict], run: dict) -> list[dict]:
    gold = question["gold_doc_ids"]
    rows = []
    for rank, hit in enumerate(hits, 1):
        if isinstance(hit, HybridHit):
            index, score = hit.chunk_index, hit.score
            bm_rank, bm_score, de_rank, de_score = hit.bm25_rank, hit.bm25_score, hit.dense_rank, hit.dense_score
        else:
            index, score = hit
            bm_rank = rank if strategy == "bm25" else None
            bm_score = score if strategy == "bm25" else None
            de_rank = rank if strategy == "dense" else None
            de_score = score if strategy == "dense" else None
        chunk = chunks[index]
        rows.append({
            "schema_version": "phase02-ranked-v1", "run_id": run["run_id"],
            "question_id": question["question_id"], "split": question["split"],
            "provenance_partition": question["provenance_partition"],
            "retrieval_strategy": strategy, "rank": rank, "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"], "filename": chunk["filename"], "strategy_score": score,
            "bm25_score": bm_score, "bm25_rank": bm_rank, "dense_score": de_score,
            "dense_rank": de_rank, "is_confirmed_gold_document": chunk["doc_id"] in gold,
            "confirmed_gold_eligible": question["confirmed_gold_eligible"],
            "unresolved_reference": question["unresolved_reference"],
            "corpus_semantic_sha256": run["corpus_semantic_sha256"],
            "chunk_config_sha256": run["chunk_config_sha256"],
            "retrieval_config_sha256": run["retrieval_config_sha256"],
        })
    return rows


def _condition_rows(ranked: list[dict], questions: dict[str, dict], run: dict) -> list[dict]:
    grouped = {}
    for row in ranked:
        grouped.setdefault((row["question_id"], row["retrieval_strategy"]), []).append(row)
    output = []
    for (question_id, strategy), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["rank"])
        question = questions[question_id]
        docs10 = [row["doc_id"] for row in rows]
        first_gold = next((rank for rank, doc in enumerate(docs10, 1) if doc in question["gold_doc_ids"]), None)
        rr10 = (1.0 / first_gold) if first_gold and question["confirmed_gold_eligible"] else (
            0.0 if question["confirmed_gold_eligible"] else None
        )
        for k in (1, 3, 5, 10):
            prefix = rows[:k]
            if question["confirmed_gold_eligible"]:
                recall, rr = document_metrics_for_ranking(
                    [row["doc_id"] for row in prefix], question["gold_doc_ids"], k
                )
            else:
                recall, rr = None, None
            output.append({
                "schema_version": "phase02-conditions-v1", "run_id": run["run_id"],
                "question_id": question_id, "split": question["split"],
                "provenance_partition": question["provenance_partition"],
                "retrieval_strategy": strategy, "k": k,
                "ordered_chunk_ids_json": canonical_json([row["chunk_id"] for row in prefix]),
                "ordered_doc_ids_json": canonical_json([row["doc_id"] for row in prefix]),
                "ordered_raw_scores_json": canonical_json([row["strategy_score"] for row in prefix]),
                "gold_doc_ids_json": canonical_json(sorted(question["gold_doc_ids"])),
                "confirmed_gold_eligible": question["confirmed_gold_eligible"],
                "unresolved_reference": question["unresolved_reference"],
                "first_gold_rank": first_gold if question["confirmed_gold_eligible"] else None,
                "doc_recall_at_k": recall, "reciprocal_rank_at_k": rr,
                "reciprocal_rank_at_10": rr10,
                "corpus_semantic_sha256": run["corpus_semantic_sha256"],
                "chunk_config_sha256": run["chunk_config_sha256"],
                "retrieval_config_sha256": run["retrieval_config_sha256"],
            })
    return output


def run_phase02(config: Phase02Config, root: Path) -> dict:
    started = time.perf_counter()
    timings = {}
    phase01 = _verify_phase01(root, config)
    print(f"validated Phase 1 split: {phase01['split_sha256']}", flush=True)
    dense_config, chunk_config, bm_config = config.values["dense"], config.values["chunking"], config.values["bm25"]
    resolved_model_revision = HfApi().model_info(
        dense_config["model_name"], revision=dense_config["model_revision"]
    ).sha
    if resolved_model_revision != dense_config["model_revision"]:
        raise ValueError(
            f"model revision mismatch: expected {dense_config['model_revision']}, "
            f"got {resolved_model_revision}"
        )
    print(f"validated MiniLM revision: {resolved_model_revision}", flush=True)
    model_cache = root / "data/derived/phase02/model_cache"
    tokenizer = AutoTokenizer.from_pretrained(
        chunk_config["tokenizer_name"], revision=chunk_config["tokenizer_revision"],
        use_fast=True, cache_dir=model_cache,
    )
    model = SentenceTransformer(
        dense_config["model_name"], revision=dense_config["model_revision"],
        device="cpu", cache_folder=str(model_cache),
    )
    dimension = int(model.get_embedding_dimension())

    stage = time.perf_counter()
    chunks, chunk_metadata, chunks_cached = _load_or_build_chunks(root, config, tokenizer)
    timings["chunks_seconds"] = time.perf_counter() - stage
    chunk_ids = [row["chunk_id"] for row in chunks]
    texts = [row["text"] for row in chunks]
    corpus_semantic = chunk_metadata["artifact"]["semantic_sha256"]
    run_values = {"phase01_split_sha256": phase01["split_sha256"], "corpus_semantic_sha256": corpus_semantic,
                  "config_sha256": config.config_sha256, "git_commit_sha": phase01["git_commit_sha"]}
    run = {"run_id": "phase02-" + canonical_json_sha256(run_values)[:16],
           "corpus_semantic_sha256": corpus_semantic,
           "chunk_config_sha256": chunk_metadata["chunk_config_sha256"],
           "retrieval_config_sha256": config.config_sha256}

    assignments = {row["question_id"]: row for row in _read_csv(root / "artifacts/data/techqa_split_assignments.csv")}
    phase01_config = json.loads((root / "configs/phase01_data.json").read_text(encoding="utf-8"))
    raw_path = root / "data/raw/techqa" / phase01_config["techqa"]["revision"] / "train.json"
    raw_questions = load_techqa_rows(raw_path)
    questions = []
    for item in raw_questions:
        assignment = assignments[item.question_id]
        unresolved = item.question_id in {"DEV_Q014", "DEV_Q094"}
        questions.append({"question_id": item.question_id, "question": item.question,
                          "split": assignment["split"], "provenance_partition": assignment["provenance_partition"],
                          "gold_doc_ids": set(json.loads(assignment["gold_doc_ids_json"])),
                          "confirmed_gold_eligible": (not item.is_impossible and not unresolved and
                                                       assignment["gold_evidence_analysis_status"] == "confirmed_attached_context"),
                          "unresolved_reference": unresolved})
    questions.sort(key=lambda row: row["question_id"])

    versions = {name: importlib.metadata.version(name) for name in
                ("rank-bm25", "numpy", "sentence-transformers", "transformers", "torch")}
    bm_key = cache_key("bm25", {"corpus": corpus_semantic, "chunk_config": run["chunk_config_sha256"],
                                 "bm25": bm_config, "rank_bm25_version": versions["rank-bm25"]})
    stage = time.perf_counter()
    bm_index, bm_cached = load_or_build_bm25(
        texts, root / f"data/derived/phase02/cache/bm25-{bm_key}.pkl",
        root / f"data/derived/phase02/cache/bm25-{bm_key}.json", key=bm_key,
        k1=bm_config["k1"], b=bm_config["b"], epsilon=bm_config["epsilon"],
    )
    timings["bm25_index_seconds"] = time.perf_counter() - stage

    dense_key = cache_key("dense", {"corpus": corpus_semantic, "chunk_config": run["chunk_config_sha256"],
        "model": dense_config["model_name"], "revision": dense_config["model_revision"],
        "tokenizer_revision": dense_config["tokenizer_revision"], "normalize": True, "dtype": "float32",
        "dimension": dimension, "versions": versions})
    stage = time.perf_counter()
    embeddings, dense_resumed = encode_resumable(
        model, texts, root / f"data/derived/phase02/cache/dense-{dense_key}.f32",
        root / f"data/derived/phase02/cache/dense-{dense_key}.json", key=dense_key,
        batch_size=dense_config["batch_size"], dimension=dimension,
    )
    timings["embedding_seconds"] = time.perf_counter() - stage

    stage = time.perf_counter()
    bm_hits = []
    for position, question in enumerate(questions, 1):
        bm_hits.append(search_bm25(bm_index, question["question"], chunk_ids, 100))
        if position % 100 == 0:
            print(f"BM25 ranked {position}/{len(questions)}", flush=True)
    timings["bm25_query_seconds"] = time.perf_counter() - stage
    stage = time.perf_counter()
    query_embeddings = model.encode(
        [row["question"] for row in questions], batch_size=dense_config["batch_size"],
        convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
    ).astype(np.float32, copy=False)
    dense_hits = []
    block = dense_config["query_block_size"]
    for start in range(0, len(questions), block):
        dense_hits.extend(exact_search(query_embeddings[start:start + block], embeddings, chunk_ids, 100))
        print(f"dense ranked {min(start + block, len(questions))}/{len(questions)}", flush=True)
    timings["dense_query_seconds"] = time.perf_counter() - stage

    ranked = []
    for index, question in enumerate(questions):
        hybrid = reciprocal_rank_fusion(bm_hits[index], dense_hits[index], chunk_ids, constant=60, depth=10)
        ranked.extend(_ranking_rows_for_strategy(question, "bm25", bm_hits[index][:10], chunks, run))
        ranked.extend(_ranking_rows_for_strategy(question, "dense", dense_hits[index][:10], chunks, run))
        ranked.extend(_ranking_rows_for_strategy(question, "hybrid", hybrid, chunks, run))
    conditions = _condition_rows(ranked, {row["question_id"]: row for row in questions}, run)
    metrics = summarize_development_metrics(conditions)
    integrity = validate_retrieval_outputs(ranked, conditions, metrics)
    if integrity["overall_status"] != "pass":
        raise ValueError(f"Phase 2 integrity failed: {integrity}")

    ranked_path = root / "artifacts/results/retrieval_ranked_hits.parquet"
    condition_path = root / "artifacts/results/retrieval_query_level.parquet"
    ranked_artifact = write_canonical_parquet(
        ranked_path, ranked, RANKED_FIELDS, ("question_id", "retrieval_strategy", "rank")
    )
    condition_artifact = write_canonical_parquet(
        condition_path, conditions, CONDITION_FIELDS, ("question_id", "retrieval_strategy", "k")
    )
    metrics.sort(key=lambda row: (row["split"], row["retrieval_strategy"], row["k"]))
    metrics_path = root / "artifacts/results/phase02_metrics_train_validation.csv"
    _write_csv(metrics_path, metrics, METRIC_FIELDS)
    write_json(root / "artifacts/results/phase02_integrity_report.json", integrity)

    lengths = [int(row["token_length"]) for row in chunks]
    per_doc = Counter(row["doc_id"] for row in chunks)
    summary = {
        "schema_version": "phase02-chunk-summary-v1", "corpus_documents": 28481,
        "total_chunks": len(chunks), "zero_chunk_documents": chunk_metadata["zero_chunk_documents"],
        "token_length": {"min": min(lengths), "max": max(lengths), "mean": sum(lengths) / len(lengths)},
        "chunks_per_document": {"min": min(per_doc.values()), "max": max(per_doc.values()),
            "mean": len(chunks) / 28481, "distribution": dict(sorted(Counter(per_doc.values()).items()))},
        "chunk_artifact": chunk_metadata["artifact"], "chunk_config_sha256": run["chunk_config_sha256"],
    }
    write_json(root / "artifacts/data/phase02_chunk_summary.json", summary)
    write_json(root / "artifacts/data/phase02_chunk_checksums.json", chunk_metadata)
    artifacts = {
        "chunks": chunk_metadata["artifact"], "ranked_hits": ranked_artifact,
        "conditions": condition_artifact,
        "metrics": {"path": metrics_path.as_posix(), "rows": len(metrics), "physical_sha256": sha256_file(metrics_path)},
    }
    artifact_hash_path = root / "artifacts/results/phase02_artifact_hashes.json"
    prior_hashes = json.loads(artifact_hash_path.read_text(encoding="utf-8")) if artifact_hash_path.exists() else None
    reproducibility = {"prior_completed_run_available": bool(prior_hashes)}
    if prior_hashes:
        for name in ("ranked_hits", "conditions"):
            prior = prior_hashes["artifacts"][name]
            current = artifacts[name]
            if prior["semantic_sha256"] != current["semantic_sha256"]:
                raise ValueError(f"cached rerun changed {name} semantic hash")
            if prior["physical_sha256"] != current["physical_sha256"]:
                raise ValueError(f"cached rerun changed {name} physical hash in pinned environment")
        reproducibility.update({"ranking_semantic_hashes_equal": True,
                                "ranking_physical_hashes_equal": True})
    write_json(artifact_hash_path,
               {"schema_version": "phase02-artifact-hashes-v1", "artifacts": artifacts,
                "reproducibility": reproducibility})
    timings["total_seconds"] = time.perf_counter() - started
    cache_root = root / "data/derived/phase02/cache"
    run_manifest_path = root / "artifacts/results/phase02_run_manifest.json"
    prior_run = json.loads(run_manifest_path.read_text(encoding="utf-8")) if run_manifest_path.exists() else None
    run_manifest = {
        "schema_version": "phase02-run-v1", **run, "git_commit_sha": phase01["git_commit_sha"],
        "phase01_split_sha256": phase01["split_sha256"], "phase01_counts": phase01["counts"],
        "packages": versions, "model_name": dense_config["model_name"],
        "model_revision": dense_config["model_revision"], "tokenizer_revision": dense_config["tokenizer_revision"],
        "device": "cpu", "dtype": "float32", "embedding_dimension": dimension,
        "chunks_cached": chunks_cached, "bm25_cached": bm_cached, "dense_resumed_or_cached": dense_resumed,
        "timings": timings, "artifact_hashes": artifacts,
        "cache_sizes": {
            "bm25_bytes": (root / f"data/derived/phase02/cache/bm25-{bm_key}.pkl").stat().st_size,
            "embeddings_bytes": (root / f"data/derived/phase02/cache/dense-{dense_key}.f32").stat().st_size,
        },
        "ranked_hit_rows": len(ranked), "condition_rows": len(conditions),
        "eligible_confirmed_gold_questions": sum(row["confirmed_gold_eligible"] for row in questions),
        "aggregate_metric_splits": ["train", "validation"], "test_aggregate_metrics_calculated": False,
        "phase_boundary": "stopped before Phase 3; no sufficiency labels or later-phase logic",
        "reproducibility": reproducibility,
        "prior_completed_run": ({"timings": prior_run.get("timings"),
                                 "artifact_hashes": prior_run.get("artifact_hashes")}
                                if prior_run else None),
    }
    write_json(run_manifest_path, run_manifest)
    return run_manifest
