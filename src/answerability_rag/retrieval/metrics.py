"""Document-level retrieval metrics with frozen eligibility and test sealing."""

from __future__ import annotations

from collections import defaultdict


def document_metrics_for_ranking(doc_ids: list[str], gold_doc_ids: set[str], k: int) -> tuple[int, float]:
    ranks = [rank for rank, doc_id in enumerate(doc_ids[:k], 1) if doc_id in gold_doc_ids]
    return (int(bool(ranks)), (1.0 / ranks[0]) if ranks else 0.0)


def summarize_development_metrics(
    conditions: list[dict], allowed_splits: tuple[str, ...] = ("train", "validation"),
) -> list[dict]:
    if "test" in allowed_splits:
        raise ValueError("aggregate test retrieval metrics are sealed during Phase 2")
    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    rr10: dict[tuple[str, str, str], float] = {}
    for row in conditions:
        if row["split"] not in allowed_splits or not row["confirmed_gold_eligible"]:
            continue
        grouped[(row["split"], row["retrieval_strategy"], int(row["k"]))].append(row)
        if int(row["k"]) == 10:
            rr10[(row["split"], row["retrieval_strategy"], row["question_id"])] = float(
                row["reciprocal_rank_at_10"]
            )
    output = []
    for (split, strategy, k), rows in sorted(grouped.items()):
        question_keys = {(split, strategy, row["question_id"]) for row in rows}
        output.append({
            "schema_version": "phase02-metrics-v1", "split": split,
            "retrieval_strategy": strategy, "k": k, "eligible_questions": len(rows),
            "document_recall_at_k": sum(int(row["doc_recall_at_k"]) for row in rows) / len(rows),
            "mrr_at_10": sum(rr10[key] for key in question_keys) / len(question_keys),
            "limitation": "correct-document retrieval is not proof of sufficient answer evidence",
        })
    if any(row["split"] == "test" for row in output):
        raise AssertionError("development summary exposed aggregate test performance")
    return output
