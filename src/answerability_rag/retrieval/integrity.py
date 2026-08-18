"""Phase 2 output integrity checks, deliberately excluding test performance."""

from __future__ import annotations

import json
from collections import Counter, defaultdict


def validate_retrieval_outputs(ranked: list[dict], conditions: list[dict], metrics: list[dict]) -> dict:
    checks = []

    def add(check_id: str, passed: bool, observed, expected) -> None:
        checks.append({"check_id": check_id, "status": "pass" if passed else "fail",
                       "observed": observed, "expected": expected})

    add("ranked_hit_rows", len(ranked) == 27300, len(ranked), 27300)
    add("condition_rows", len(conditions) == 10920, len(conditions), 10920)
    keys = [(row["question_id"], row["retrieval_strategy"], row["rank"]) for row in ranked]
    add("unique_rank_keys", len(keys) == len(set(keys)), len(set(keys)), len(keys))
    groups = defaultdict(list)
    for row in ranked:
        groups[(row["question_id"], row["retrieval_strategy"])].append(row)
    ranks_ok = all(sorted(int(row["rank"]) for row in rows) == list(range(1, 11)) for rows in groups.values())
    chunks_ok = all(len({row["chunk_id"] for row in rows}) == 10 for rows in groups.values())
    add("rank_continuity", ranks_ok, ranks_ok, True)
    add("unique_chunks_per_ranking", chunks_ok, chunks_ok, True)
    split_counts = Counter(row["split"] for row in ranked if row["retrieval_strategy"] == "bm25" and row["rank"] == 1)
    add("question_split_counts", split_counts == {"train": 637, "validation": 137, "test": 136},
        dict(split_counts), {"train": 637, "validation": 137, "test": 136})
    prefixes_ok = True
    by_condition = {(row["question_id"], row["retrieval_strategy"], int(row["k"])): row for row in conditions}
    for question, strategy in groups:
        ten = json.loads(by_condition[(question, strategy, 10)]["ordered_chunk_ids_json"])
        for k in (1, 3, 5):
            if json.loads(by_condition[(question, strategy, k)]["ordered_chunk_ids_json"]) != ten[:k]:
                prefixes_ok = False
    add("nested_prefixes", prefixes_ok, prefixes_ok, True)
    test_summary_rows = sum(row.get("split") == "test" for row in metrics)
    add("test_aggregate_sealed", test_summary_rows == 0, test_summary_rows, 0)
    strategy_counts = Counter(row["retrieval_strategy"] for row in ranked)
    add("strategy_rows", strategy_counts == {"bm25": 9100, "dense": 9100, "hybrid": 9100},
        dict(strategy_counts), {"bm25": 9100, "dense": 9100, "hybrid": 9100})
    status = "pass" if all(check["status"] == "pass" for check in checks) else "fail"
    return {"schema_version": "phase02-integrity-v1", "overall_status": status, "checks": checks}
