import pytest

from answerability_rag.retrieval.metrics import document_metrics_for_ranking, summarize_development_metrics


def test_document_recall_multiple_gold_and_mrr_duplicate_chunks() -> None:
    docs = ["irrelevant", "gold-b", "gold-b", "gold-a"]
    assert document_metrics_for_ranking(docs, {"gold-a", "gold-b"}, 1) == (0, 0.0)
    assert document_metrics_for_ranking(docs, {"gold-a", "gold-b"}, 4) == (1, 0.5)


def condition(question, split, eligible, k, recall, rr):
    return {"question_id": question, "split": split, "retrieval_strategy": "bm25", "k": k,
            "confirmed_gold_eligible": eligible, "doc_recall_at_k": recall,
            "reciprocal_rank_at_10": rr}


def test_impossible_unresolved_excluded_and_test_sealed() -> None:
    rows = []
    for k in (1, 3, 5, 10):
        rows.extend([condition("eligible", "train", True, k, 1, 0.5),
                     condition("impossible", "train", False, k, None, None),
                     condition("unresolved", "validation", False, k, None, None),
                     condition("sealed", "test", True, k, 1, 1.0)])
    summary = summarize_development_metrics(rows)
    assert {row["split"] for row in summary} == {"train"}
    assert all(row["eligible_questions"] == 1 for row in summary)
    with pytest.raises(ValueError, match="sealed"):
        summarize_development_metrics(rows, ("test",))
