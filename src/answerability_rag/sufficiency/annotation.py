"""Deterministic, question-disjoint, blinded manual-validation material."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from answerability_rag.hashing import canonical_json, canonical_json_sha256, sha256_text


SAMPLE_FIELDS = (
    "schema_version", "sample_id", "blind_order", "question_id", "split",
    "retrieval_strategy", "k", "sampling_stratum", "sample_seed", "example_id",
    "annotation_guideline_version", "annotation_guideline_sha256",
)
BLINDED_FIELDS = (
    "schema_version", "sample_id", "blind_order", "question_id", "split",
    "retrieval_strategy", "k", "question", "benchmark_status", "benchmark_answer",
    "retrieved_context_json", "annotation_guideline_version", "annotation_guideline_sha256",
    "annotator_id", "manual_label", "rationale", "annotation_timestamp",
)
ANSWER_KEY_FIELDS = (
    "schema_version", "sample_id", "question_id", "example_id", "sampling_stratum",
    "automatic_y_suff", "label_status", "label_method", "gold_document_hit",
    "maximum_span_coverage_fraction", "partial_overlap", "exclusion_reason",
    "phase03_label_config_sha256",
)


def sampling_stratum(row: dict[str, Any]) -> str:
    if row["label_method"] == "benchmark_impossible":
        return "benchmark_impossible"
    if row["y_suff"] == 1:
        return "automatic_positive"
    if row["partial_overlap"]:
        return "partial_overlap"
    if row["gold_document_hit"]:
        return "correct_document_insufficient"
    return "wrong_document_retrieval"


def select_annotation_sample(label_rows: list[dict[str, Any]], annotation: dict[str, Any]) -> list[dict]:
    eligible = [
        row for row in label_rows
        if row["split"] in set(annotation["eligible_splits"]) and row["y_suff"] is not None
    ]
    by_stratum: dict[str, list[dict[str, Any]]] = {}
    for row in eligible:
        by_stratum.setdefault(sampling_stratum(row), []).append(row)
    seed = int(annotation["sample_seed"])
    used_questions: set[str] = set()
    selected: list[tuple[dict[str, Any], str]] = []
    strategy_counts: Counter[str] = Counter()
    depth_counts: Counter[int] = Counter()
    split_counts: Counter[str] = Counter()

    def stable(row: dict[str, Any], purpose: str) -> str:
        return canonical_json_sha256({
            "seed": seed, "purpose": purpose, "question_id": row["question_id"],
            "strategy": row["retrieval_strategy"], "k": row["k"],
        })

    def take(pool: list[dict[str, Any]], count: int, stratum: str) -> None:
        for _ in range(count):
            available = [row for row in pool if row["question_id"] not in used_questions]
            if not available:
                return
            row = min(available, key=lambda item: (
                strategy_counts[item["retrieval_strategy"]], depth_counts[int(item["k"])],
                split_counts[item["split"]], stable(item, stratum),
            ))
            selected.append((row, stratum))
            used_questions.add(row["question_id"])
            strategy_counts[row["retrieval_strategy"]] += 1
            depth_counts[int(row["k"])] += 1
            split_counts[row["split"]] += 1

    for stratum, target in annotation["target_strata"].items():
        take(by_stratum.get(stratum, []), int(target), stratum)
    target_size = int(annotation["target_size"])
    if len(selected) < target_size:
        take(eligible, target_size - len(selected), "availability_fill")
    if len(selected) != target_size:
        raise ValueError(f"manual sample has {len(selected)} rows; expected {target_size}")
    selected.sort(key=lambda item: stable(item[0], "blind_order"))
    output = []
    for blind_order, (row, stratum) in enumerate(selected, 1):
        sample_id = canonical_json_sha256({
            "example_id": row["example_id"], "sample_seed": seed,
            "annotation_version": annotation["version"],
        })
        output.append({**row, "sample_id": sample_id, "blind_order": blind_order,
                       "sampling_stratum": stratum, "sample_seed": seed})
    return output


def build_annotation_material(
    sample: list[dict[str, Any]], questions: dict[str, Any], conditions: list[dict[str, Any]],
    chunks: list[dict[str, Any]], *, guideline_version: str, guideline_text: str,
) -> tuple[list[dict], list[dict], list[dict]]:
    guideline_sha256 = sha256_text(guideline_text)
    condition_map = {
        (row["question_id"], row["retrieval_strategy"], int(row["k"])): row for row in conditions
    }
    chunk_map = {row["chunk_id"]: row for row in chunks}
    manifests: list[dict] = []
    blinded: list[dict] = []
    keys: list[dict] = []
    for row in sample:
        question = questions[row["question_id"]]
        condition = condition_map[(row["question_id"], row["retrieval_strategy"], int(row["k"]))]
        context = []
        for rank, chunk_id in enumerate(json.loads(condition["ordered_chunk_ids_json"]), 1):
            chunk = chunk_map[chunk_id]
            context.append({
                "rank": rank, "chunk_id": chunk_id, "document_id": chunk["doc_id"],
                "filename": chunk["filename"], "char_start": chunk["char_start"],
                "char_end": chunk["char_end"], "text": chunk["text"],
            })
        manifests.append({
            "schema_version": "phase03-annotation-sample-v1", "sample_id": row["sample_id"],
            "blind_order": row["blind_order"], "question_id": row["question_id"],
            "split": row["split"], "retrieval_strategy": row["retrieval_strategy"],
            "k": int(row["k"]), "sampling_stratum": row["sampling_stratum"],
            "sample_seed": row["sample_seed"], "example_id": row["example_id"],
            "annotation_guideline_version": guideline_version,
            "annotation_guideline_sha256": guideline_sha256,
        })
        blinded.append({
            "schema_version": "phase03-annotation-blinded-v1", "sample_id": row["sample_id"],
            "blind_order": row["blind_order"], "question_id": row["question_id"],
            "split": row["split"], "retrieval_strategy": row["retrieval_strategy"],
            "k": int(row["k"]), "question": question.question,
            "benchmark_status": (
                "benchmark_impossible_no_reference" if question.is_impossible
                else "answerable_with_reference"
            ), "benchmark_answer": None if question.is_impossible else question.answer,
            "retrieved_context_json": canonical_json(context),
            "annotation_guideline_version": guideline_version,
            "annotation_guideline_sha256": guideline_sha256, "annotator_id": "",
            "manual_label": "", "rationale": "", "annotation_timestamp": "",
        })
        keys.append({
            "schema_version": "phase03-annotation-key-v1", "sample_id": row["sample_id"],
            "question_id": row["question_id"], "example_id": row["example_id"],
            "sampling_stratum": row["sampling_stratum"], "automatic_y_suff": row["y_suff"],
            "label_status": row["label_status"], "label_method": row["label_method"],
            "gold_document_hit": row["gold_document_hit"],
            "maximum_span_coverage_fraction": row["maximum_span_coverage_fraction"],
            "partial_overlap": row["partial_overlap"], "exclusion_reason": row["exclusion_reason"],
            "phase03_label_config_sha256": row["phase03_label_config_sha256"],
        })
    return manifests, blinded, keys
