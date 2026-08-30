from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from answerability_rag.sufficiency.semantic import (
    aggregate_claim_scores,
    build_context_units,
    confirmation_stratum,
    segment_reference_answer,
    semantic_prediction,
    threshold_grid,
    weighted_metrics,
)
from answerability_rag.sufficiency.semantic_nli import NLIScorer
from answerability_rag.sufficiency.semantic_pipeline import (
    _claim_budget_metadata,
    _evaluate_thresholds,
    _score_pairs_checkpointed,
)


def test_reference_claim_segmentation_is_deterministic_and_conservative() -> None:
    answer = "<ul><li>Enable TLS.</li><li>Set the port to 443; restart the server after saving.</li></ul>"
    claims = segment_reference_answer(answer)
    assert claims == ["Enable TLS.", "Set the port to 443", "restart the server after saving."]
    assert segment_reference_answer(answer) == claims
    assert segment_reference_answer("-") == []


def test_context_unit_ranks_all_chunks_and_builds_cross_chunk_window() -> None:
    chunks = [
        {"rank": 1, "chunk_id": "a", "text": "TLS setup"},
        {"rank": 2, "chunk_id": "b", "text": "unrelated"},
        {"rank": 3, "chunk_id": "c", "text": "restart TLS server"},
    ]
    units = build_context_units("configure TLS and restart server", chunks)
    assert len(units) == 1
    assert units[0]["unit_type"] == "claim_ranked_multi_chunk_window"
    assert units[0]["chunk_ids"] == ["a", "b", "c"]
    assert units == build_context_units("configure TLS and restart server", chunks)


def test_claim_score_aggregation_uses_max_entailment_and_stable_unit_tie() -> None:
    rows = [
        {"unit_id": "b", "unit_type": "single_chunk", "chunk_ids": ["b"],
         "entailment": 0.8, "neutral": 0.1, "contradiction": 0.1},
        {"unit_id": "a", "unit_type": "adjacent_chunk_pair", "chunk_ids": ["a", "b"],
         "entailment": 0.8, "neutral": 0.15, "contradiction": 0.05},
    ]
    result = aggregate_claim_scores(rows)
    assert result["best_unit_id"] == "a"
    assert result["entailment"] == 0.8
    assert result["maximum_unit_contradiction"] == 0.1


def test_semantic_prediction_requires_minimum_mean_and_contradiction_rules() -> None:
    row = {
        "minimum_claim_entailment": 0.75,
        "mean_claim_entailment": 0.82,
        "maximum_selected_premise_contradiction": 0.10,
    }
    thresholds = {
        "minimum_claim_entailment": 0.70,
        "mean_claim_entailment": 0.80,
        "maximum_selected_premise_contradiction": 0.25,
    }
    assert semantic_prediction(row, thresholds) == 1
    assert semantic_prediction({**row, "mean_claim_entailment": 0.79}, thresholds) == 0
    assert semantic_prediction({**row, "maximum_selected_premise_contradiction": 0.25}, thresholds) == 0


def test_threshold_grid_is_finite_and_enforces_mean_at_least_minimum() -> None:
    config = {
        "threshold_search": {
            "minimum_claim_entailment": [0.5, 0.7],
            "mean_claim_entailment": [0.5, 0.7],
            "mean_threshold_must_be_at_least_minimum_threshold": True,
            "maximum_selected_premise_contradiction": [0.5, 1.01],
        }
    }
    grid = threshold_grid(config)
    assert len(grid) == 6
    assert all(row["mean_claim_entailment"] >= row["minimum_claim_entailment"] for row in grid)


def test_weighted_metrics_use_stratum_confusion_rates_not_pooled_counts() -> None:
    rows = [
        {"sampling_stratum": "a", "manual_label": "sufficient", "prediction": 1},
        {"sampling_stratum": "b", "manual_label": "sufficient", "prediction": 0},
    ]
    result = weighted_metrics(rows, {"a": 3, "b": 1}, ("a", "b"))
    assert result["estimated_confusion_proportions"] == {
        "tn": 0.0, "fp": 0.0, "fn": 0.25, "tp": 0.75,
    }
    assert result["recall"] == 0.75


def test_confirmation_strata_are_mutually_exclusive() -> None:
    assert {
        confirmation_stratum(strict, semantic)
        for strict in (0, 1) for semantic in (0, 1)
    } == {
        "strict_negative_semantic_negative", "strict_negative_semantic_positive",
        "strict_positive_semantic_negative", "strict_positive_semantic_positive",
    }


def test_development_precision_gate_is_unweighted_human_precision() -> None:
    base = {
        "evaluation_status": "evaluable", "minimum_claim_entailment": 1.0,
        "mean_claim_entailment": 1.0, "maximum_selected_premise_contradiction": 0.0,
    }
    rows = []
    for stratum, positives, negatives in (
        ("automatic_positive", 1, 0),
        ("partial_overlap", 1, 0),
        ("correct_document_insufficient", 5, 1),
        ("wrong_document_retrieval", 1, 0),
    ):
        rows.extend({**base, "sampling_stratum": stratum, "manual_label": "sufficient"}
                    for _ in range(positives))
        rows.extend({**base, "sampling_stratum": stratum, "manual_label": "insufficient"}
                    for _ in range(negatives))
    config = SimpleNamespace(values={"threshold_search": {
        "minimum_claim_entailment": [0.5], "mean_claim_entailment": [0.5],
        "mean_threshold_must_be_at_least_minimum_threshold": True,
        "maximum_selected_premise_contradiction": [0.5], "minimum_precision": 0.90,
        "primary_population_stratum_counts": {
            "automatic_positive": 1686, "partial_overlap": 1014,
            "correct_document_insufficient": 787, "wrong_document_retrieval": 2693,
        },
    }})
    result = _evaluate_thresholds(
        {"model_id": "candidate", "revision": "revision", "candidate_order": 1},
        rows, config,
    )[0]
    assert result["sample_precision"] == 8 / 9
    assert result["weighted_precision"] > 0.90
    assert result["precision_gate_status"] == "fail"


class _RetokenizingBudgetTokenizer:
    """Tiny tokenizer whose decode/re-tokenize cycle adds deterministic overhead."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(range(len(str(text).split())))

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(f"t{index}" for index in token_ids)

    def num_special_tokens_to_add(self, pair: bool = True) -> int:
        assert pair is True
        return 3

    def __call__(
        self, premise: str, claim: str, *, add_special_tokens: bool,
        truncation: bool,
    ) -> dict[str, list[int]]:
        assert add_special_tokens is True and truncation is False
        # The extra five tokens emulate tokenizer-specific boundary/re-tokenization
        # overhead not captured by the initial arithmetic estimate.
        length = len(self.encode(premise)) + len(self.encode(claim)) + 3 + 5
        return {"input_ids": list(range(length))}


def test_balanced_renderer_trims_measured_retokenization_overhead() -> None:
    scorer = object.__new__(NLIScorer)
    scorer.model_id = "test-model"
    scorer.max_length = 32
    scorer.tokenizer = _RetokenizingBudgetTokenizer()
    unit = {
        "ranks": [1, 2, 3],
        "constituents": [" ".join(["a"] * 30), " ".join(["b"] * 30), " ".join(["c"] * 30)],
    }
    claim = "claim words"
    premise = scorer.render_premise(claim, unit)
    assert len(scorer.tokenizer(
        premise, claim, add_special_tokens=True, truncation=False,
    )["input_ids"]) <= scorer.max_length


def test_claim_budget_governance_is_general_and_does_not_truncate() -> None:
    scorer = object.__new__(NLIScorer)
    scorer.max_length = 8
    scorer.tokenizer = _RetokenizingBudgetTokenizer()
    claims = ["short claim", "one two three four five"]
    metadata = _claim_budget_metadata(scorer, claims)
    assert claims == ["short claim", "one two three four five"]
    assert metadata == {
        "claim_token_lengths": [2, 5],
        "exceeding_claim_indices": [2],
        "maximum_claim_token_length": 5,
        "model_max_sequence_length": 8,
        "model_pair_special_token_count": 3,
        "maximum_valid_claim_only_tokens": 4,
        "semantic_evaluable": False,
    }


class _CheckpointScorer:
    model_id = "candidate"
    revision = "revision"

    def __init__(self) -> None:
        self.scored_keys: list[str] = []

    def score(self, pairs: list[dict]) -> list[dict]:
        output = []
        for row in pairs:
            self.scored_keys.append(row["pair_key"])
            output.append({
                **{key: value for key, value in row.items() if key != "unit"},
                "unit_id": row["unit"]["unit_id"],
                "unit_type": row["unit"]["unit_type"],
                "chunk_ids": row["unit"]["chunk_ids"],
                "entailment": 0.8, "neutral": 0.1, "contradiction": 0.1,
            })
        return output


def _checkpoint_pair(key: str) -> dict:
    return {
        "pair_key": key, "condition_id": "condition", "question_id": "question",
        "split": "train", "retrieval_strategy": "bm25", "k": 1,
        "model_id": "candidate", "model_revision": "revision", "claim_index": 1,
        "claim_text": f"claim {key}", "claim_sha256": f"claim-sha-{key}",
        "unit": {
            "unit_id": f"unit-{key}", "unit_type": "claim_ranked_multi_chunk_window",
            "chunk_ids": [f"chunk-{key}"], "ranks": [1], "constituents": ["text"],
        },
    }


def test_checkpoint_bootstraps_legacy_rows_and_writes_atomic_resume_bitmap(tmp_path) -> None:
    path = tmp_path / "scores.jsonl"
    first = _checkpoint_pair("a")
    legacy = _CheckpointScorer().score([first])[0]
    path.write_text(json.dumps(legacy, sort_keys=True) + "\n", encoding="utf-8")
    scorer = _CheckpointScorer()
    rows = _score_pairs_checkpointed(
        scorer, [first, _checkpoint_pair("b")], path, 1,
        {"scope": "test", "model_id": "candidate", "model_revision": "revision"},
    )
    assert len(rows) == 2
    assert scorer.scored_keys == ["b"]
    manifest = json.loads(path.with_suffix(".jsonl.manifest.json").read_text(encoding="utf-8"))
    assert manifest["completed_bitmap"] == "11"
    assert manifest["completed_pair_count"] == 2
    assert manifest["legacy_jsonl_recovery_bootstrap"] is True
    assert not path.with_suffix(".jsonl.manifest.json.tmp").exists()
    with pytest.raises(ValueError, match="stale NLI checkpoint metadata rejected"):
        _score_pairs_checkpointed(
            _CheckpointScorer(), [first, _checkpoint_pair("b")], path, 1,
            {"scope": "changed", "model_id": "candidate", "model_revision": "revision"},
        )
