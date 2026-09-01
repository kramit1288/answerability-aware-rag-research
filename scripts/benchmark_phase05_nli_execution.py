"""Benchmark execution-only NLI settings on already checkpointed claims.

This script never reads human labels, computes metrics, or writes the Phase 5
grounding checkpoint. It compares candidate execution outputs only with the
already persisted NLI probabilities for a deterministic checkpoint sample.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from answerability_rag.generation.config import Phase05Config
from answerability_rag.generation.grounding import score_pairs_length_bucketed
from answerability_rag.generation.pipeline import _ragtruth_records
from answerability_rag.hashing import canonical_json_sha256, sha256_text
from answerability_rag.io import write_json_atomic


CHECKPOINT = Path("data/derived/phase05/phase05_ragtruth_grounding_checkpoint.jsonl")
OUTPUT = Path("artifacts/results/phase05_execution_benchmark.json")


def _checkpoint_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    with (root / CHECKPOINT).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _sample_pairs(root: Path, config: Phase05Config, sample_size: int) -> list[dict[str, Any]]:
    """Reconstruct exact persisted claim/chunk pairs without reading labels."""
    checkpoint = _checkpoint_rows(root)
    rows = sorted(checkpoint, key=lambda row: str(row["claim_id"]))[:sample_size]
    records, _ = _ragtruth_records(root, config)
    by_response = {str(record["response_id"]): record for record in records}
    output: list[dict[str, Any]] = []
    for row in rows:
        record = by_response[str(row["response_id"])]
        chunks = {str(chunk["chunk_id"]): chunk for chunk in record["chunks"]}
        persisted = json.loads(str(row["candidate_nli_json"]))
        for candidate in persisted:
            chunk = chunks[str(candidate["chunk_id"])]
            unit = {
                "unit_id": canonical_json_sha256({
                    "claim_id": row["claim_id"], "chunk_id": candidate["chunk_id"],
                    "rank": int(candidate["rank"]),
                    "text_sha256": sha256_text(str(chunk["text"])),
                }),
                "unit_type": "independent_candidate_chunk",
                "chunk_ids": [str(candidate["chunk_id"])],
                "ranks": [int(candidate["rank"])],
                "constituents": [str(chunk["text"])],
            }
            output.append({
                "claim_id": str(row["claim_id"]),
                "candidate_index": int(candidate["candidate_index"]),
                "claim_text": str(row["claim_text"]),
                "chunk_id": str(candidate["chunk_id"]),
                "rank": int(candidate["rank"]),
                "similarity": float(candidate["similarity"]),
                "unit": unit,
                "baseline": {
                    "entailment": float(candidate["entailment"]),
                    "neutral": float(candidate["neutral"]),
                    "contradiction": float(candidate["contradiction"]),
                },
            })
    return output


def _worker(
    root: Path, batch_size: int, threads: int, interop: int,
    sample_size: int, length_bucketing: bool,
) -> None:
    import torch
    from answerability_rag.sufficiency.semantic_nli import NLIScorer

    torch.set_num_threads(int(threads))
    torch.set_num_interop_threads(int(interop))
    config = Phase05Config.load(root / "configs/phase05_generation_grounding.json", root)
    evaluator = config.values["grounding_evaluator"]
    pairs = _sample_pairs(root, config, sample_size)
    scorer = NLIScorer(
        evaluator["nli_model_id"], evaluator["nli_model_revision"],
        root / config.values["generator"]["cache_directory"],
        int(evaluator["nli_max_pair_tokens"]), int(batch_size),
    )
    started = time.perf_counter()
    scored = score_pairs_length_bucketed(scorer, pairs) if length_bucketing else scorer.score(pairs)
    elapsed = time.perf_counter() - started
    scorer.close()
    components = ("entailment", "neutral", "contradiction")
    differences = []
    argmax_changes = 0
    for row, score in zip(pairs, scored):
        baseline = np.asarray([row["baseline"][name] for name in components], dtype=np.float64)
        candidate = np.asarray([float(score[name]) for name in components], dtype=np.float64)
        differences.extend(np.abs(candidate - baseline).tolist())
        argmax_changes += int(int(np.argmax(candidate)) != int(np.argmax(baseline)))
    values = np.asarray(differences, dtype=np.float64)
    print(json.dumps({
        "batch_size": int(batch_size), "torch_threads": int(threads),
        "torch_interop_threads": int(interop), "sample_claims": int(sample_size),
        "length_bucketing": bool(length_bucketing),
        "sample_pairs": len(pairs), "elapsed_seconds": elapsed,
        "claims_per_second": sample_size / elapsed,
        "pairs_per_second": len(pairs) / elapsed,
        "max_absolute_difference": float(values.max()),
        "mean_absolute_difference": float(values.mean()),
        "count_exceeding_1e-6": int((values > 1e-6).sum()),
        "count_exceeding_1e-5": int((values > 1e-5).sum()),
        "argmax_nli_class_changes": argmax_changes,
    }, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--interop", type=int, default=2)
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--length-bucketing", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.worker:
        _worker(root, args.batch_size, args.threads, args.interop, args.sample_size, args.length_bucketing)
        return
    config = Phase05Config.load(root / "configs/phase05_generation_grounding.json", root)
    sample = _sample_pairs(root, config, args.sample_size)
    if len(sample) != args.sample_size * 3:
        raise RuntimeError("checkpoint sample does not contain exactly three pairs per claim")
    candidates = [(16, 2, 2)] + [
        (batch, threads, 1, False)
        for batch in (16, 32, 64)
        for threads in (1, 2, 4)
    ]
    candidates += [(batch, 4, 1, True) for batch in (16, 32, 64)]
    results = []
    for candidate in candidates:
        if len(candidate) == 3:
            batch, threads, interop = candidate
            length_bucketing = False
        else:
            batch, threads, interop, length_bucketing = candidate
        command = [
            sys.executable, str(Path(__file__).resolve()), "--worker",
            "--batch-size", str(batch), "--threads", str(threads),
            "--interop", str(interop), "--sample-size", str(args.sample_size),
        ]
        if length_bucketing:
            command.append("--length-bucketing")
        completed = subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        result["execution_equivalent"] = (
            result["argmax_nli_class_changes"] == 0
            and result["max_absolute_difference"] <= 1e-5
        )
        results.append(result)
    result = {
        "schema_version": "phase05-execution-benchmark-v1",
        "scientific_inputs_unchanged": True,
        "labels_read": False,
        "partial_performance_metrics_calculated": False,
        "checkpoint_path": str((root / CHECKPOINT).resolve()).replace("\\", "/"),
        "checkpoint_claim_count_at_benchmark": len(_checkpoint_rows(root)),
        "sample_claims": args.sample_size,
        "sample_pairs": len(sample),
        "baseline_execution": {"batch_size": 16, "torch_threads": 2, "torch_interop_threads": 2},
        "candidates": results,
    }
    write_json_atomic(root / OUTPUT, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
