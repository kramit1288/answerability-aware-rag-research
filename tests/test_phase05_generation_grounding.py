from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from answerability_rag.generation.claims import segment_claims
from answerability_rag.generation.config import (
    PHASE05_CONFIG_SHA256,
    Phase05Config,
    assert_techqa_split_allowed,
    verify_upstream,
)
from answerability_rag.generation.context import (
    assemble_ranked_context,
    build_user_prompt,
)
from answerability_rag.generation.grounding import (
    aggregate_candidate_nli,
    parse_ragtruth_passages,
    response_grounding_metrics,
    select_candidate_chunks,
    select_support_threshold,
)
from answerability_rag.generation.policy import build_policy_view
from answerability_rag.hashing import canonical_json_sha256, sha256_file


ROOT = Path(__file__).resolve().parents[1]


class CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return "<user>" + messages[0]["content"] + "<assistant>"

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, values, skip_special_tokens=True):
        return "".join(chr(value) for value in values)


def _template() -> str:
    return "Question:\n{question}\n\nContext:\n{context}\n\nAnswer:"


def _chunks(count: int, size: int = 10):
    return [
        {"chunk_id": f"c{rank}", "rank": rank, "text": chr(64 + rank) * size}
        for rank in range(1, count + 1)
    ]


def test_phase05_upstream_hashes_and_pre_results_freeze_reproduce() -> None:
    config = Phase05Config.load(ROOT / "configs/phase05_generation_grounding.json", ROOT)
    assert config.canonical_sha256 == PHASE05_CONFIG_SHA256
    observed = verify_upstream(ROOT, config)
    assert observed["phase03_final_manifest_physical_sha256"] == "b02a0ca0e3352798c8d38ef4942fdca59e23d8d24c662493a1249d2864ca9879"
    assert observed["phase04_final_manifest_physical_sha256"] == "9afe8e988a14be4f8cf9eb032e71279b4154797ab16d6610e467f6da7cefccde"
    assert observed["phase04_model_binary_physical_sha256"] == "7d0f124cfea75d9b69e52ef83af066eae50afdf35b5d3f4a0d53f7785b36b9fd"


@pytest.mark.parametrize("field", [
    "benchmark_answer", "answer", "y_suff_final", "nli_scores", "span_coverage",
    "human_annotation", "model_probability", "gold_evidence", "retrieval_score", "policy_action",
])
def test_generation_prompt_rejects_evaluation_leakage(field: str) -> None:
    with pytest.raises(ValueError, match="forbidden"):
        build_user_prompt(_template(), question="q", context="c", metadata={field: "secret"})


def test_generation_prompt_has_only_question_and_context_placeholders() -> None:
    assert build_user_prompt(_template(), question="q", context="c") == "Question:\nq\n\nContext:\nc\n\nAnswer:"
    with pytest.raises(ValueError, match="only question/context"):
        build_user_prompt("{question} {answer}", question="q", context="c")


def test_context_assembly_preserves_rank_and_k10_adds_chunks() -> None:
    tokenizer = CharacterTokenizer()
    k5 = assemble_ranked_context(
        tokenizer=tokenizer, prompt_template=_template(), question="q",
        chunks=_chunks(5), maximum_input_tokens=1000,
    )
    k10 = assemble_ranked_context(
        tokenizer=tokenizer, prompt_template=_template(), question="q",
        chunks=_chunks(10), maximum_input_tokens=1000,
    )
    assert k5.prompt_visible_chunk_ids == ("c1", "c2", "c3", "c4", "c5")
    assert k10.prompt_visible_chunk_ids == tuple(f"c{i}" for i in range(1, 11))
    assert k10.context.startswith(k5.context)
    assert k5.assembled_context_sha256 != k10.assembled_context_sha256


def test_context_assembly_truncates_only_last_included_chunk_deterministically() -> None:
    tokenizer = CharacterTokenizer()
    base = assemble_ranked_context(
        tokenizer=tokenizer, prompt_template=_template(), question="q",
        chunks=_chunks(1, size=20), maximum_input_tokens=200,
    )
    budget = base.input_token_count + len("\n\n[CHUNK 2]\n") + 5
    first = assemble_ranked_context(
        tokenizer=tokenizer, prompt_template=_template(), question="q",
        chunks=_chunks(4, size=20), maximum_input_tokens=budget,
    )
    second = assemble_ranked_context(
        tokenizer=tokenizer, prompt_template=_template(), question="q",
        chunks=_chunks(4, size=20), maximum_input_tokens=budget,
    )
    assert first == second
    assert first.fully_included_chunk_count == 1
    assert first.final_truncated_chunk_id == "c2"
    assert first.prompt_visible_chunk_ids == ("c1", "c2")
    assert first.retrieved_chunks_not_included == 2


def test_frozen_generation_settings_and_synthetic_smoke_are_deterministic() -> None:
    values = json.loads((ROOT / "configs/phase05_generation_grounding.json").read_text(encoding="utf-8"))
    settings = values["generation"]
    assert settings["seed"] == 42
    assert settings["do_sample"] is False and settings["num_beams"] == 1
    assert settings["max_new_tokens"] == 128 and settings["repetition_penalty"] == 1.0
    freeze = json.loads((ROOT / "artifacts/results/phase05_pre_results_governance_freeze.json").read_text(encoding="utf-8"))
    smoke = freeze["generator_infrastructure_smoke_test"]
    assert smoke["status"] == "pass" and smoke["techqa_content_used"] is False
    assert smoke["output_sha256"] == "5de7efd99e2b6fa6d0af339b7148ba087fa0d2a5204d74aa9a65c34e1d4ddd47"


def test_claim_segmentation_sentence_bullet_and_offsets() -> None:
    response = "First claim. Second claim!\n- Third bullet claim\n2) Fourth claim?"
    claims = segment_claims(response)
    assert [claim.text for claim in claims] == [
        "First claim.", "Second claim!", "Third bullet claim", "Fourth claim?",
    ]
    assert all(response[claim.start:claim.end] == claim.text for claim in claims)
    assert [claim.claim_index for claim in claims] == [1, 2, 3, 4]
    assert segment_claims(" \n- !!!\n") == []


def test_ragtruth_passage_parser_preserves_order() -> None:
    parsed = parse_ragtruth_passages("passage 1:alpha\n\npassage 2:beta\n\npassage 3:gamma")
    assert [(row["chunk_id"], row["rank"], row["text"]) for row in parsed] == [
        ("passage-1", 1, "alpha"), ("passage-2", 2, "beta"), ("passage-3", 3, "gamma"),
    ]


def test_grounding_candidate_selection_and_aggregation() -> None:
    chunks = [
        {"chunk_id": "c1", "rank": 1}, {"chunk_id": "c2", "rank": 2},
        {"chunk_id": "c3", "rank": 3}, {"chunk_id": "c4", "rank": 4},
    ]
    selected = select_candidate_chunks([0.7, 0.9, 0.8, 0.1], chunks, 3)
    assert [row["chunk_id"] for row in selected] == ["c2", "c3", "c1"]
    aggregate = aggregate_candidate_nli([
        {"evaluation_status": "evaluable", "chunk_id": "c2", "rank": 2, "entailment": 0.4, "contradiction": 0.1},
        {"evaluation_status": "evaluable", "chunk_id": "c3", "rank": 3, "entailment": 0.8, "contradiction": 0.2},
        {"evaluation_status": "evaluable", "chunk_id": "c1", "rank": 1, "entailment": 0.8, "contradiction": 0.7},
    ])
    assert aggregate["claim_support_score"] == 0.8
    assert aggregate["supporting_chunk_id"] == "c1"
    assert aggregate["maximum_claim_contradiction"] == 0.7
    assert aggregate_candidate_nli([])["evaluation_status"] == "evaluator_unevaluable"


def test_grounding_zero_claim_and_unevaluable_rules() -> None:
    zero = response_grounding_metrics([], 0.5)
    assert zero["unsupported_claim_rate"] is None
    assert zero["fully_supported_response"] is None
    unevaluable = response_grounding_metrics([{"evaluation_status": "evaluator_unevaluable"}], 0.5)
    assert unevaluable["unsupported_claim_rate"] is None
    assert unevaluable["fully_supported_response"] is False
    mixed = response_grounding_metrics([
        {"evaluation_status": "evaluable", "claim_support_score": 0.8, "maximum_claim_contradiction": 0.1},
        {"evaluation_status": "evaluator_unevaluable"},
    ], 0.5)
    assert mixed["fully_supported_response"] is False


def test_ragtruth_manifest_preserves_official_source_grouping() -> None:
    manifest = pd.read_csv(ROOT / "artifacts/data/ragtruth_manifest.csv")
    qa = manifest[manifest["task_type"] == "QA"]
    assert len(qa) == 5934
    assert qa.groupby("source_id")["official_split"].nunique().max() == 1
    assert qa[qa["official_split"] == "train"]["source_id"].nunique() == 839
    assert qa[qa["official_split"] == "test"]["source_id"].nunique() == 150


def test_threshold_selection_is_train_only_and_test_cannot_affect_it() -> None:
    train = [
        {"official_split": "train", "evaluation_status": "evaluable", "human_unsupported": 0, "claim_support_score": 0.9},
        {"official_split": "train", "evaluation_status": "evaluable", "human_unsupported": 1, "claim_support_score": 0.2},
        {"official_split": "train", "evaluation_status": "evaluable", "human_unsupported": 1, "claim_support_score": 0.4},
    ]
    threshold, grid = select_support_threshold(train)
    assert len(grid) == 51 and sum(bool(row["selected"]) for row in grid) == 1
    assert threshold == 0.9
    with pytest.raises(ValueError, match="TRAIN"):
        select_support_threshold([*train, {**train[0], "official_split": "test", "human_unsupported": 1}])


def test_policy_views_select_correct_generation_and_null_abstention_quality() -> None:
    states = {
        ("q1", 5): {"response_id": "q1-k5", "rouge_l_f1": 0.5},
        ("q1", 10): {"response_id": "q1-k10", "rouge_l_f1": 1.0},
        ("q2", 5): {"response_id": "q2-k5", "rouge_l_f1": 0.4},
        ("q2", 10): {"response_id": "q2-k10", "rouge_l_f1": 0.8},
    }
    rows = build_policy_view(
        ["q1", "q2"], states, policy_id="G2",
        actions={"q1": "ANSWER_AT_K10", "q2": "ABSTAIN"},
    )
    assert rows[0]["response_id"] == "q1-k10"
    assert rows[1]["answered"] is False and rows[1]["rouge_l_f1"] is None
    config = json.loads((ROOT / "configs/phase05_generation_grounding.json").read_text(encoding="utf-8"))
    assert (config["policy_views"]["G2"]["t_low"], config["policy_views"]["G2"]["t_high"]) == (0.78, 0.82)
    assert (config["policy_views"]["G3"]["t_low"], config["policy_views"]["G3"]["t_high"]) == (0.56, 0.72)


@pytest.mark.parametrize("operation", [
    "assemble generation prompt", "generate answer", "calculate grounding",
    "calculate ROUGE-L", "calculate BERTScore", "construct policy view",
])
def test_techqa_test_seal_rejects_every_phase05_operation(operation: str) -> None:
    with pytest.raises(ValueError, match="TEST is sealed"):
        assert_techqa_split_allowed("test", operation)
