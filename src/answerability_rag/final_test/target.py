"""Mechanical construction of the frozen Phase 3 sufficiency target on TechQA TEST."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from answerability_rag.data.techqa import load_techqa_rows
from answerability_rag.hashing import canonical_json, canonical_json_sha256, sha256_file
from answerability_rag.retrieval.artifacts import write_canonical_parquet
from answerability_rag.sufficiency.rescue import final_prediction, rescue_mechanisms
from answerability_rag.sufficiency.semantic import (
    aggregate_claim_scores,
    aggregate_condition_claims,
    confirmation_stratum,
    segment_reference_answer,
    semantic_prediction,
)
from answerability_rag.sufficiency.semantic_nli import NLIScorer
from answerability_rag.sufficiency.semantic_pipeline import (
    _claim_budget_metadata,
    _pairs_for_condition,
    _score_pairs_checkpointed,
)

from .common import PHASE07_CONFIG_SHA256, RESULTS, Phase07Config, require_unsealed, write_csv, write_json


STRICT_FIELDS = (
    "schema_version", "example_id", "question_id", "split", "split_group_id",
    "retrieval_strategy", "k", "benchmark_is_impossible", "reference_status",
    "gold_alignment_status", "label_status", "y_suff_strict", "label_method",
    "label_provenance", "maximum_span_coverage_fraction", "exclusion_reason",
    "phase03_label_config_sha256",
)
CLAIM_FIELDS = (
    "schema_version", "scope", "condition_id", "question_id", "split",
    "retrieval_strategy", "k", "model_id", "model_revision", "claim_index",
    "claim_text", "claim_sha256", "entailment", "neutral", "contradiction",
    "best_unit_id", "best_unit_type", "best_chunk_ids_json",
    "maximum_unit_contradiction", "context_unit_count", "semantic_config_sha256",
)
SEMANTIC_FIELDS = (
    "schema_version", "example_id", "question_id", "split", "split_group_id",
    "retrieval_strategy", "k", "y_suff_strict", "claim_count",
    "claim_token_lengths_json", "maximum_claim_token_length", "model_max_sequence_length",
    "model_pair_special_token_count", "maximum_valid_claim_only_tokens",
    "semantic_model_id", "semantic_model_revision", "semantic_config_sha256",
    "semantic_label_governance_sha256", "primary_population_eligible",
    "semantic_label_status", "semantic_exclusion_reason", "y_suff_semantic",
    "semantic_support_score", "supported_claim_count", "minimum_claim_entailment",
    "mean_claim_entailment", "maximum_selected_premise_contradiction",
    "maximum_any_unit_contradiction", "contradiction_indicator",
    "semantic_label_method", "strict_semantic_agreement", "strict_semantic_disagreement",
    "strict_semantic_category", "label_provenance",
)
TARGET_FIELDS = (
    "schema_version", "example_id", "question_id", "split", "split_group_id",
    "retrieval_strategy", "k", "y_suff_strict", "y_suff_semantic", "y_suff_final",
    "maximum_span_coverage_fraction", "minimum_claim_entailment", "mean_claim_entailment",
    "maximum_selected_premise_contradiction", "coverage_rescue", "nli_rescue",
    "rescued_positive", "rescue_rule_status", "selected_rescue_family",
    "final_target_config_sha256", "phase07_config_sha256", "label_provenance",
)


def _load_test_inputs(root: Path) -> tuple[list[dict[str, Any]], dict[tuple[str, str, int], dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    labels = pq.read_table(
        root / "artifacts/data/context_sufficiency_labels.parquet",
        filters=[("split", "=", "test")],
    ).to_pylist()
    if len(labels) != 1632 or len({row["question_id"] for row in labels}) != 136:
        raise ValueError("frozen Phase 1 TEST condition census differs")
    conditions = pq.read_table(
        root / "artifacts/results/retrieval_query_level.parquet",
        filters=[("split", "=", "test")],
    ).to_pylist()
    condition_map = {
        (str(row["question_id"]), str(row["retrieval_strategy"]), int(row["k"])): row
        for row in conditions
    }
    if len(condition_map) != len(labels):
        raise ValueError("TEST retrieval conditions do not align one-to-one with strict labels")
    chunks = {
        str(row["chunk_id"]): row for row in pq.read_table(
            root / "artifacts/data/techqa_chunk_manifest.parquet",
            columns=["chunk_id", "doc_id", "filename", "char_start", "char_end", "text"],
        ).to_pylist()
    }
    phase01 = json.loads((root / "configs/phase01_data.json").read_text(encoding="utf-8"))
    raw_path = root / "data/raw/techqa" / phase01["techqa"]["revision"] / "train.json"
    questions = {row.question_id: row for row in load_techqa_rows(raw_path)}
    return labels, condition_map, chunks, questions


def _context(condition: dict[str, Any], chunks: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {**chunks[str(chunk_id)], "rank": rank}
        for rank, chunk_id in enumerate(json.loads(condition["ordered_chunk_ids_json"]), 1)
    ]


def _strict_rows(labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "schema_version": "phase07-test-strict-label-v1",
        "example_id": row["example_id"], "question_id": row["question_id"],
        "split": "test", "split_group_id": row["split_group_id"],
        "retrieval_strategy": row["retrieval_strategy"], "k": int(row["k"]),
        "benchmark_is_impossible": bool(row["benchmark_is_impossible"]),
        "reference_status": row["reference_status"],
        "gold_alignment_status": row["gold_alignment_status"],
        "label_status": row["label_status"],
        "y_suff_strict": None if row["y_suff"] is None else int(row["y_suff"]),
        "label_method": row["label_method"], "label_provenance": row["label_provenance"],
        "maximum_span_coverage_fraction": row["maximum_span_coverage_fraction"],
        "exclusion_reason": row["exclusion_reason"],
        "phase03_label_config_sha256": row["phase03_label_config_sha256"],
    } for row in labels]


def _selected_semantic(root: Path) -> dict[str, Any]:
    selected = json.loads(
        (root / "artifacts/results/phase03_semantic_selected_config.json").read_text(encoding="utf-8")
    )
    if selected["semantic_config_sha256"] != "98f7279821921d825470ee64efa810777e5b331d4c978e6234e7b689b6657fdf":
        raise ValueError("frozen Phase 3 semantic configuration differs")
    if selected["model_revision"] != "6f5cf0a2b59cabb106aca4c287eed12e357e90eb":
        raise ValueError("frozen Phase 3 semantic model revision differs")
    return selected


def _semantic_rows(
    root: Path, labels: list[dict[str, Any]], condition_map: dict[tuple[str, str, int], dict[str, Any]],
    chunks: dict[str, dict[str, Any]], questions: dict[str, Any], selected: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    eligible = [
        row for row in labels
        if row["benchmark_is_impossible"] is False and row["y_suff"] is not None
        and row["label_method"] == "gold_span_coverage"
        and str(row["gold_alignment_status"]).startswith("aligned_")
    ]
    scorer = NLIScorer(
        selected["model_id"], selected["model_revision"], root / ".cache/huggingface", 256, 16,
    )
    try:
        pair_map: dict[tuple[str, str], dict[str, Any]] = {}
        specs: dict[str, dict[str, Any]] = {}
        claims_by_question: dict[str, tuple[list[str], dict[str, Any]]] = {}
        unevaluable_questions: dict[str, dict[str, Any]] = {}
        for index, label in enumerate(eligible, 1):
            question_id = str(label["question_id"])
            if question_id not in claims_by_question:
                claims = segment_reference_answer(questions[question_id].answer)
                if not claims:
                    raise ValueError(f"eligible TEST question has no semantic claims: {question_id}")
                claims_by_question[question_id] = (claims, _claim_budget_metadata(scorer, claims))
            claims, budget = claims_by_question[question_id]
            if not budget["semantic_evaluable"]:
                specs[label["example_id"]] = {
                    "label": label, "claims": claims, "budget": budget, "keys": {},
                    "status": "unevaluable",
                }
                record = unevaluable_questions.setdefault(question_id, {
                    "question_id": question_id, "split": "test", "condition_count": 0,
                    "reason": "claim_exceeds_frozen_nli_pair_budget",
                    "claim_token_lengths_json": canonical_json(budget["claim_token_lengths"]),
                    "exceeding_claim_indices_json": canonical_json(budget["exceeding_claim_indices"]),
                })
                record["condition_count"] += 1
                continue
            condition = condition_map[(question_id, label["retrieval_strategy"], int(label["k"]))]
            pairs, keys = _pairs_for_condition(
                label["example_id"], question_id, "test", label["retrieval_strategy"],
                int(label["k"]), claims, _context(condition, chunks), selected["model_id"],
                selected["model_revision"],
            )
            for pair in pairs:
                pair_map.setdefault((pair["claim_sha256"], pair["unit"]["unit_id"]), pair)
            specs[label["example_id"]] = {
                "label": label, "claims": claims, "budget": budget, "keys": keys,
                "status": "evaluable",
            }
            if index % 250 == 0:
                print(f"Phase 7 target preparation {index}/{len(eligible)} conditions", flush=True)
        scored = _score_pairs_checkpointed(
            scorer, list(pair_map.values()),
            root / ".cache/phase07" / "test-semantic-pairs.jsonl", 16,
            {
                "scope": "phase07_immutable_test_target",
                "phase07_config_sha256": PHASE07_CONFIG_SHA256,
                "model_id": selected["model_id"], "model_revision": selected["model_revision"],
                "semantic_config_sha256": selected["semantic_config_sha256"],
                "semantic_label_governance_sha256": "233c61c45c7ad75b876e81deeb290d546a13f034f1d18c82787aea678ba0f533",
            },
        )
        score_map = {(row["claim_sha256"], row["unit_id"]): row for row in scored}
        semantic_rows: list[dict[str, Any]] = []
        claim_rows: list[dict[str, Any]] = []
        for example_id, spec in sorted(specs.items()):
            label, claims, budget = spec["label"], spec["claims"], spec["budget"]
            strict = int(label["y_suff"])
            common = {
                "schema_version": "phase07-test-semantic-label-v1", "example_id": example_id,
                "question_id": label["question_id"], "split": "test",
                "split_group_id": label["split_group_id"],
                "retrieval_strategy": label["retrieval_strategy"], "k": int(label["k"]),
                "y_suff_strict": strict, "claim_count": len(claims),
                "claim_token_lengths_json": canonical_json(budget["claim_token_lengths"]),
                "maximum_claim_token_length": budget["maximum_claim_token_length"],
                "model_max_sequence_length": budget["model_max_sequence_length"],
                "model_pair_special_token_count": budget["model_pair_special_token_count"],
                "maximum_valid_claim_only_tokens": budget["maximum_valid_claim_only_tokens"],
                "semantic_model_id": selected["model_id"],
                "semantic_model_revision": selected["model_revision"],
                "semantic_config_sha256": selected["semantic_config_sha256"],
                "semantic_label_governance_sha256": "233c61c45c7ad75b876e81deeb290d546a13f034f1d18c82787aea678ba0f533",
            }
            if spec["status"] == "unevaluable":
                semantic_rows.append({
                    **common, "primary_population_eligible": False,
                    "semantic_label_status": "unevaluable",
                    "semantic_exclusion_reason": "claim_exceeds_frozen_nli_pair_budget",
                    "y_suff_semantic": None, "semantic_support_score": None,
                    "supported_claim_count": None, "minimum_claim_entailment": None,
                    "mean_claim_entailment": None,
                    "maximum_selected_premise_contradiction": None,
                    "maximum_any_unit_contradiction": None, "contradiction_indicator": None,
                    "semantic_label_method": "not_applied_semantic_unevaluable",
                    "strict_semantic_agreement": None, "strict_semantic_disagreement": None,
                    "strict_semantic_category": "semantic_unevaluable",
                    "label_provenance": canonical_json({
                        "strict_example_id": example_id,
                        "semantic_exclusion_reason": "claim_exceeds_frozen_nli_pair_budget",
                        "semantic_config_sha256": selected["semantic_config_sha256"],
                    }),
                })
                continue
            aggregated_claims: list[dict[str, Any]] = []
            for claim_index, claim in enumerate(claims, 1):
                aggregate = aggregate_claim_scores([score_map[key] for key in spec["keys"][claim_index]])
                claim_row = {
                    "schema_version": "phase07-test-semantic-claim-score-v1",
                    "scope": "final_test", "condition_id": example_id,
                    "question_id": label["question_id"], "split": "test",
                    "retrieval_strategy": label["retrieval_strategy"], "k": int(label["k"]),
                    "model_id": selected["model_id"], "model_revision": selected["model_revision"],
                    "claim_index": claim_index, "claim_text": claim,
                    "claim_sha256": canonical_json_sha256({"claim": claim}), **aggregate,
                    "semantic_config_sha256": selected["semantic_config_sha256"],
                }
                claim_rows.append(claim_row); aggregated_claims.append(claim_row)
            aggregate = aggregate_condition_claims(aggregated_claims)
            semantic_label = semantic_prediction(aggregate, selected["thresholds"])
            supported = sum(
                float(row["entailment"]) >= float(selected["thresholds"]["minimum_claim_entailment"])
                and float(row["contradiction"]) < float(selected["thresholds"]["maximum_selected_premise_contradiction"])
                for row in aggregated_claims
            )
            semantic_rows.append({
                **common, "primary_population_eligible": True,
                "semantic_label_status": "evaluable", "semantic_exclusion_reason": None,
                "y_suff_semantic": semantic_label,
                "semantic_support_score": aggregate["minimum_claim_entailment"],
                "supported_claim_count": supported, **aggregate,
                "contradiction_indicator": (
                    aggregate["maximum_selected_premise_contradiction"]
                    >= float(selected["thresholds"]["maximum_selected_premise_contradiction"])
                ),
                "semantic_label_method": "local_nli_reference_claim_support",
                "strict_semantic_agreement": strict == semantic_label,
                "strict_semantic_disagreement": strict != semantic_label,
                "strict_semantic_category": confirmation_stratum(strict, semantic_label),
                "label_provenance": canonical_json({
                    "strict_example_id": example_id,
                    "semantic_method": "local_nli_reference_claim_support",
                    "semantic_config_sha256": selected["semantic_config_sha256"],
                }),
            })
        return semantic_rows, claim_rows, list(unevaluable_questions.values())
    finally:
        scorer.close()


def _target_rows(
    strict: list[dict[str, Any]], semantic: list[dict[str, Any]], config: Phase07Config,
) -> list[dict[str, Any]]:
    strict_by_id = {str(row["example_id"]): row for row in strict}
    candidate = {
        "family": "combined", "T_cov": 0.20, "T_mean": 0.35,
        "T_min": 0.05, "T_contradiction": 0.50,
    }
    output: list[dict[str, Any]] = []
    for row in semantic:
        if row["semantic_label_status"] != "evaluable":
            continue
        source = strict_by_id[str(row["example_id"])]
        combined = {
            "y_suff_strict": int(row["y_suff_strict"]),
            "maximum_span_coverage_fraction": source["maximum_span_coverage_fraction"],
            "minimum_claim_entailment": row["minimum_claim_entailment"],
            "mean_claim_entailment": row["mean_claim_entailment"],
            "maximum_selected_premise_contradiction": row["maximum_selected_premise_contradiction"],
        }
        coverage, nli = rescue_mechanisms(combined, candidate)
        final = final_prediction(combined, candidate)
        rescued = int(row["y_suff_strict"]) == 0 and final == 1
        output.append({
            "schema_version": "phase07-test-final-target-v1",
            "example_id": row["example_id"], "question_id": row["question_id"],
            "split": "test", "split_group_id": row["split_group_id"],
            "retrieval_strategy": row["retrieval_strategy"], "k": int(row["k"]),
            "y_suff_strict": int(row["y_suff_strict"]),
            "y_suff_semantic": int(row["y_suff_semantic"]), "y_suff_final": final,
            "maximum_span_coverage_fraction": source["maximum_span_coverage_fraction"],
            "minimum_claim_entailment": row["minimum_claim_entailment"],
            "mean_claim_entailment": row["mean_claim_entailment"],
            "maximum_selected_premise_contradiction": row["maximum_selected_premise_contradiction"],
            "coverage_rescue": bool(coverage), "nli_rescue": bool(nli),
            "rescued_positive": rescued,
            "rescue_rule_status": "strict_positive_retained" if int(row["y_suff_strict"]) else (
                "strict_negative_rescued" if rescued else "strict_negative_not_rescued"
            ),
            "selected_rescue_family": "combined",
            "final_target_config_sha256": config.values["upstream_freeze"]["phase03_final_target_config_canonical_sha256"],
            "phase07_config_sha256": config.canonical_sha256,
            "label_provenance": canonical_json({
                "strict_label_artifact": "phase07_test_strict_labels.parquet",
                "semantic_label_artifact": "phase07_test_semantic_labels.parquet",
                "rule": config.values["target"]["final_rule"],
            }),
        })
    return output


def _write_population(
    root: Path, labels: list[dict[str, Any]], semantic: list[dict[str, Any]],
    unevaluable: list[dict[str, Any]], target: list[dict[str, Any]],
) -> None:
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in labels:
        by_question[str(row["question_id"])].append(row)
    semantic_status = {
        str(row["question_id"]): str(row["semantic_label_status"]) for row in semantic
    }
    target_questions = {str(row["question_id"]) for row in target}
    exclusion_rows = []
    for question_id, rows in sorted(by_question.items()):
        first = rows[0]
        impossible = bool(first["benchmark_is_impossible"])
        unresolved = not impossible and first["y_suff"] is None
        sem_status = "not_applicable_benchmark_impossible" if impossible else (
            "not_applicable_unresolved" if unresolved else semantic_status.get(question_id, "missing")
        )
        if impossible:
            reason, role = "benchmark_impossible_primary_exclusion", "benchmark_impossible_sensitivity"
        elif unresolved:
            reason, role = str(first["exclusion_reason"]), "excluded_unresolved_evidence"
        elif sem_status == "unevaluable":
            reason, role = "claim_exceeds_frozen_nli_pair_budget", "excluded_semantic_unevaluable"
        else:
            reason, role = None, "primary"
        eligible = question_id in target_questions
        exclusion_rows.append({
            "question_id": question_id, "split": "test",
            "benchmark_is_impossible": impossible,
            "gold_alignment_status": first["gold_alignment_status"],
            "semantic_label_status": sem_status,
            "primary_eligible": eligible, "exclusion_reason": reason,
            "population_role": role, "retrieval_condition_count": len(rows),
        })
    fields = (
        "question_id", "split", "benchmark_is_impossible", "gold_alignment_status",
        "semantic_label_status", "primary_eligible", "exclusion_reason",
        "population_role", "retrieval_condition_count",
    )
    write_csv(root / RESULTS / "phase07_test_exclusion_manifest.csv", exclusion_rows, fields)
    reason_counts = Counter(row["exclusion_reason"] for row in exclusion_rows if row["exclusion_reason"])
    census = {
        "schema_version": "phase07-test-population-census-v1",
        "total_test_questions": len(exclusion_rows),
        "benchmark_answerable_test_questions": sum(not row["benchmark_is_impossible"] for row in exclusion_rows),
        "benchmark_impossible_test_questions": sum(row["benchmark_is_impossible"] for row in exclusion_rows),
        "primary_eligible_test_questions": sum(row["primary_eligible"] for row in exclusion_rows),
        "excluded_test_questions": sum(not row["primary_eligible"] for row in exclusion_rows),
        "exclusion_reason_counts": dict(sorted(reason_counts.items())),
        "primary_eligible_retrieval_conditions": len(target),
        "conditions_per_primary_question": 12,
        "benchmark_impossible_kept_separate": True,
        "split_membership_changed": False,
    }
    write_json(root / RESULTS / "phase07_test_population_census.json", census)


def construct_test_target(root: Path, config_path: Path) -> dict[str, Any]:
    config = Phase07Config.load(config_path)
    unseal = require_unsealed(root, config)
    final_path = root / RESULTS / "phase07_test_final_target.parquet"
    manifest_path = root / RESULTS / "phase07_test_target_manifest.json"
    if final_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["phase07_config_sha256"] != config.canonical_sha256:
            raise ValueError("existing immutable TEST target belongs to another configuration")
        return manifest
    labels, condition_map, chunks, questions = _load_test_inputs(root)
    strict = _strict_rows(labels)
    strict_path = root / RESULTS / "phase07_test_strict_labels.parquet"
    strict_artifact = write_canonical_parquet(
        strict_path, strict, STRICT_FIELDS, ("question_id", "retrieval_strategy", "k")
    )
    selected = _selected_semantic(root)
    semantic, claims, unevaluable = _semantic_rows(
        root, labels, condition_map, chunks, questions, selected,
    )
    semantic_artifact = write_canonical_parquet(
        root / RESULTS / "phase07_test_semantic_labels.parquet", semantic, SEMANTIC_FIELDS,
        ("question_id", "retrieval_strategy", "k"),
    )
    claims_artifact = write_canonical_parquet(
        root / RESULTS / "phase07_test_semantic_claim_scores.parquet", claims, CLAIM_FIELDS,
        ("question_id", "retrieval_strategy", "k", "claim_index"),
    )
    target = _target_rows(strict, semantic, config)
    target_artifact = write_canonical_parquet(
        final_path, target, TARGET_FIELDS, ("question_id", "retrieval_strategy", "k")
    )
    # Target class balance is calculated only after the complete target artifact is closed.
    distribution = Counter(int(row["y_suff_final"]) for row in target)
    _write_population(root, labels, semantic, unevaluable, target)
    manifest = {
        "schema_version": "phase07-test-target-manifest-v1",
        "phase07_config_sha256": config.canonical_sha256,
        "unseal_timestamp": unseal["unseal_timestamp"],
        "strict_artifact": strict_artifact, "semantic_artifact": semantic_artifact,
        "semantic_claim_artifact": claims_artifact, "final_target_artifact": target_artifact,
        "strict_positive_demotion_count": 0,
        "primary_question_count": len({row["question_id"] for row in target}),
        "primary_condition_count": len(target),
        "semantic_unevaluable_question_count": len(unevaluable),
        "semantic_unevaluable_condition_count": sum(int(row["condition_count"]) for row in unevaluable),
        "final_target_class_distribution": {
            "negative": distribution.get(0, 0), "positive": distribution.get(1, 0),
        },
        "aggregate_balance_accessed_only_after_target_closed": True,
        "immutable": True,
        "final_rule": config.values["target"]["final_rule"],
        "semantic_model_id": selected["model_id"],
        "semantic_model_revision": selected["model_revision"],
    }
    write_json(manifest_path, manifest, immutable=True)
    return manifest
