"""Mechanical application of the frozen Phase 5 grounding proxy to TEST generations."""

from __future__ import annotations

import gc
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from answerability_rag.generation.claims import segment_claims
from answerability_rag.generation.config import Phase05Config
from answerability_rag.generation.grounding import (
    GroundingModels,
    aggregate_candidate_nli,
    candidate_json,
    response_grounding_metrics,
    select_candidate_chunks,
)
from answerability_rag.generation.pipeline import CLAIM_FIELDS, _grounding_config_sha
from answerability_rag.hashing import canonical_json_sha256, sha256_file, sha256_text
from answerability_rag.retrieval.artifacts import write_canonical_parquet

from .common import RESULTS, Phase07Config, require_unsealed, write_csv, write_json


DERIVED = Path("data/derived/phase07")
RESPONSE_FIELDS = (
    "schema_version", "response_id", "question_id", "k", "generation_status",
    "claim_count", "evaluable_claim_count", "unevaluable_claim_count",
    "mean_claim_support_score", "minimum_claim_support_score", "unsupported_claim_count",
    "unsupported_claim_rate", "maximum_claim_contradiction", "fully_supported_response",
    "response_with_any_unsupported_claim", "response_grounding_status", "y_suff_final", "t_support",
)
TEST_CLAIM_FIELDS = (*CLAIM_FIELDS, "t_support", "predicted_unsupported_claim")


def _phase05(root: Path) -> Phase05Config:
    return Phase05Config.load(root / "configs/phase05_generation_grounding.json", root)


def _load_checkpoint(path: Path, grounding_sha: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row["grounding_config_sha256"] != grounding_sha:
                raise ValueError(f"stale Phase 7 grounding checkpoint at line {number}")
            claim_id = str(row["claim_id"])
            if claim_id in rows and rows[claim_id] != row:
                raise ValueError(f"conflicting grounding checkpoint claim {claim_id}")
            rows[claim_id] = row
    return rows


def _append(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _score_claims(
    root: Path, grounding_sha: str, models: GroundingModels, records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    checkpoint = root / DERIVED / "phase07_test_grounding_checkpoint.jsonl"
    cached = _load_checkpoint(checkpoint, grounding_sha)
    claims: list[dict[str, Any]] = []
    contexts: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        response_id = str(record["response_id"])
        contexts[response_id] = record["chunks"]
        for claim in segment_claims(str(record["response_text"])):
            claim_id = sha256_text(
                f"techqa\n{response_id}\n{claim.claim_index}\n{claim.start}\n{claim.end}\n{claim.text}"
            )
            claims.append({
                "schema_version": "phase07-test-claim-grounding-v1", "dataset": "techqa",
                "response_id": response_id, "source_id": None,
                "question_id": record["question_id"], "official_split": "test", "quality": None,
                "context_id": response_id, "claim_id": claim_id, "claim_index": claim.claim_index,
                "claim_text": claim.text, "claim_start": claim.start, "claim_end": claim.end,
                "human_unsupported": None,
            })
    required = {row["claim_id"] for row in claims}
    extra = set(cached) - required
    if extra:
        raise ValueError(f"grounding checkpoint contains {len(extra)} claims outside immutable TEST responses")
    remaining = [row for row in claims if row["claim_id"] not in cached]
    if remaining:
        unique_chunks: dict[tuple[str, str], dict[str, Any]] = {}
        for row in remaining:
            for chunk in contexts[row["context_id"]]:
                unique_chunks[(row["context_id"], str(chunk["chunk_id"]))] = chunk
        chunk_keys = sorted(unique_chunks)
        chunk_values = models.encode([str(unique_chunks[key]["text"]) for key in chunk_keys])
        chunk_embeddings = dict(zip(chunk_keys, chunk_values))
        claim_embeddings = models.encode([str(row["claim_text"]) for row in remaining])
        for start in range(0, len(remaining), 128):
            batch = remaining[start:start + 128]
            embeddings = claim_embeddings[start:start + len(batch)]
            prepared: list[tuple[dict[str, Any], list[dict[str, Any]], int, int]] = []
            pairs: list[dict[str, Any]] = []
            for row, embedding in zip(batch, embeddings):
                chunks = contexts[row["context_id"]]
                matrix = np.stack([chunk_embeddings[(row["context_id"], str(chunk["chunk_id"]))] for chunk in chunks])
                candidates = select_candidate_chunks((matrix @ embedding).tolist(), chunks, 3)
                fits, claim_length, special = models.claim_fits(str(row["claim_text"]))
                prepared.append((row, candidates, claim_length, special))
                if fits:
                    for candidate_index, candidate in enumerate(candidates, 1):
                        unit = {
                            "unit_id": canonical_json_sha256({
                                "claim_id": row["claim_id"], "chunk_id": candidate["chunk_id"],
                                "rank": int(candidate["rank"]), "text_sha256": sha256_text(str(candidate["text"])),
                            }),
                            "unit_type": "independent_candidate_chunk",
                            "chunk_ids": [str(candidate["chunk_id"])], "ranks": [int(candidate["rank"])],
                            "constituents": [str(candidate["text"])],
                        }
                        pairs.append({
                            "claim_id": row["claim_id"], "claim_text": row["claim_text"],
                            "candidate_index": candidate_index, "chunk_id": str(candidate["chunk_id"]),
                            "rank": int(candidate["rank"]), "similarity": float(candidate["similarity"]),
                            "unit": unit,
                        })
            by_claim: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for pair in models.score_pairs(pairs) if pairs else []:
                by_claim[str(pair["claim_id"])].append({
                    "evaluation_status": "evaluable", "candidate_index": int(pair["candidate_index"]),
                    "chunk_id": str(pair["chunk_id"]), "rank": int(pair["rank"]),
                    "similarity": float(pair["similarity"]), "entailment": float(pair["entailment"]),
                    "neutral": float(pair["neutral"]), "contradiction": float(pair["contradiction"]),
                })
            completed = []
            for row, candidates, claim_length, special in prepared:
                nli = sorted(by_claim.get(row["claim_id"], []), key=lambda item: int(item["candidate_index"]))
                completed.append({
                    **{key: row.get(key) for key in (
                        "schema_version", "dataset", "response_id", "source_id", "question_id",
                        "official_split", "quality", "claim_id", "claim_index", "claim_text",
                        "claim_start", "claim_end", "human_unsupported",
                    )},
                    "candidate_chunk_ids_json": candidate_json(candidates, ["chunk_id", "rank"]),
                    "candidate_similarities_json": candidate_json(candidates, ["chunk_id", "rank", "similarity"]),
                    "candidate_nli_json": candidate_json(nli, ["candidate_index", "chunk_id", "rank", "similarity", "entailment", "neutral", "contradiction"]),
                    "claim_token_length": claim_length, "pair_special_token_count": special,
                    **aggregate_candidate_nli(nli), "grounding_config_sha256": grounding_sha,
                })
            _append(checkpoint, completed)
            cached.update({row["claim_id"]: row for row in completed})
            print(f"Phase 7 TEST grounding {min(start + len(batch), len(remaining))}/{len(remaining)} new claims", flush=True)
    if set(cached) != required:
        raise ValueError("Phase 7 TEST grounding checkpoint is incomplete")
    rows = sorted(cached.values(), key=lambda row: (str(row["question_id"]), str(row["response_id"]), int(row["claim_index"])))
    counts = {
        "response_count": len(records), "claim_count": len(rows),
        "evaluable_claim_count": sum(row["evaluation_status"] == "evaluable" for row in rows),
        "unevaluable_claim_count": sum(row["evaluation_status"] != "evaluable" for row in rows),
        "zero_claim_response_count": len(records) - len({row["response_id"] for row in rows}),
    }
    return rows, counts


def evaluate_test_grounding(root: Path, config_path: Path) -> dict[str, Any]:
    config = Phase07Config.load(config_path)
    require_unsealed(root, config)
    phase05 = _phase05(root)
    generation = pq.read_table(root / RESULTS / "phase07_test_generation_cache.parquet").to_pylist()
    contexts = pq.read_table(root / RESULTS / "phase07_test_context_manifest.parquet").to_pylist()
    if len(generation) != len(contexts):
        raise ValueError("generation and context state counts differ")
    by_response = {str(row["response_id"]): row for row in contexts}
    records = []
    for row in generation:
        if row["generation_status"] != "generated":
            continue
        context = by_response[str(row["response_id"])]
        records.append({
            "response_id": row["response_id"], "question_id": row["question_id"],
            "response_text": row["normalized_generated_text"],
            "chunks": json.loads(context["prompt_visible_chunks_json"]),
        })
    evaluator = phase05.values["grounding_evaluator"]
    grounding_sha = _grounding_config_sha(phase05)
    if grounding_sha != config.values["upstream_freeze"]["phase05_grounding_config_sha256"]:
        raise ValueError("frozen Phase 5 grounding evaluator configuration changed")
    amendment = json.loads((root / RESULTS / "phase05_execution_optimization.json").read_text(encoding="utf-8"))
    execution = amendment["selected_execution"]
    models = GroundingModels(
        evaluator["candidate_embedding_model"], evaluator["candidate_embedding_revision"],
        evaluator["nli_model_id"], evaluator["nli_model_revision"],
        root / phase05.values["generator"]["cache_directory"],
        int(evaluator["nli_max_pair_tokens"]), int(execution["nli_batch_size"]),
        bool(execution["length_bucketing"]),
    )
    claims, counts = _score_claims(root, grounding_sha, models, records)
    models.close(); del models
    gc.collect()
    t_support = float(config.values["grounding"]["support_threshold"])
    if t_support != 0.16:
        raise ValueError("frozen grounding threshold changed")
    claims = [{
        **row, "t_support": t_support,
        "predicted_unsupported_claim": (
            None if row["claim_support_score"] is None
            else bool(float(row["claim_support_score"]) < t_support)
        ),
    } for row in claims]
    claim_path = root / RESULTS / "phase07_test_claim_grounding.parquet"
    claim_artifact = write_canonical_parquet(claim_path, claims, TEST_CLAIM_FIELDS, ("question_id", "response_id", "claim_index"))
    generated_fields = ("schema_version", "response_id", "question_id", "claim_id", "claim_index", "claim_text", "claim_start", "claim_end")
    generated_claims = [{
        **{field: row.get(field) for field in generated_fields},
        "schema_version": "phase07-test-generated-claims-v1",
    } for row in claims]
    generated_artifact = write_canonical_parquet(
        root / RESULTS / "phase07_test_generated_claims.parquet", generated_claims,
        generated_fields, ("question_id", "response_id", "claim_index"),
    )
    claims_by_response: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in claims:
        claims_by_response[str(row["response_id"])].append(row)
    context_y = {str(row["response_id"]): int(row["y_suff_final"]) for row in contexts}
    response_rows = []
    unevaluable = []
    for row in sorted(generation, key=lambda item: (item["question_id"], int(item["k"]))):
        response_id = str(row["response_id"])
        response_claims = claims_by_response.get(response_id, [])
        if row["generation_status"] == "generated":
            metric = response_grounding_metrics(response_claims, t_support)
        else:
            metric = {
                "claim_count": 0, "evaluable_claim_count": 0, "unevaluable_claim_count": 0,
                "mean_claim_support_score": None, "minimum_claim_support_score": None,
                "unsupported_claim_count": 0, "unsupported_claim_rate": None,
                "maximum_claim_contradiction": None, "fully_supported_response": None,
                "response_with_any_unsupported_claim": None,
                "response_grounding_status": row["generation_status"],
            }
        response_rows.append({
            "schema_version": "phase07-test-response-grounding-v1", "response_id": response_id,
            "question_id": row["question_id"], "k": int(row["k"]),
            "generation_status": row["generation_status"], **metric,
            "y_suff_final": context_y[response_id], "t_support": t_support,
        })
        for claim in response_claims:
            if claim["evaluation_status"] != "evaluable":
                unevaluable.append({
                    "response_id": response_id, "question_id": row["question_id"], "k": int(row["k"]),
                    "claim_id": claim["claim_id"], "claim_text": claim["claim_text"],
                    "claim_token_length": claim["claim_token_length"],
                    "reason": "claim_exceeds_frozen_nli_pair_budget",
                })
    response_path = root / RESULTS / "phase07_test_response_grounding.parquet"
    response_artifact = write_canonical_parquet(response_path, response_rows, RESPONSE_FIELDS, ("question_id", "k"))
    write_csv(
        root / RESULTS / "phase07_test_evaluator_unevaluable_claims.csv", unevaluable,
        ("response_id", "question_id", "k", "claim_id", "claim_text", "claim_token_length", "reason"),
    )
    manifest = {
        "schema_version": "phase07-test-grounding-manifest-v1", "claim_artifact": claim_artifact,
        "generated_claim_artifact": generated_artifact, "response_artifact": response_artifact,
        "counts": counts, "unevaluable_claim_count": len(unevaluable),
        "unevaluable_claim_rate": len(unevaluable) / len(claims) if claims else None,
        "selected_t_support": t_support, "grounding_config_sha256": grounding_sha,
        "interpretation": "automatic retrieved-context support proxy; not authoritative hallucination labels",
        "generation_cache_sha256": sha256_file(root / RESULTS / "phase07_test_generation_cache.parquet"),
    }
    write_json(root / RESULTS / "phase07_test_grounding_manifest.json", manifest)
    return manifest
