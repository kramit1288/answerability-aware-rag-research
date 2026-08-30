"""Deterministic offline semantic context-sufficiency construction for Phase 3.6."""

from __future__ import annotations

import html
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from answerability_rag.hashing import canonical_json_sha256


PRIMARY_STRATA = (
    "automatic_positive",
    "partial_overlap",
    "correct_document_insufficient",
    "wrong_document_retrieval",
)
DIAGNOSTIC_STRATA = PRIMARY_STRATA + ("benchmark_impossible",)
CONFIRMATION_STRATA = (
    "strict_negative_semantic_positive",
    "strict_positive_semantic_positive",
    "strict_negative_semantic_negative",
    "strict_positive_semantic_negative",
)
VALID_MANUAL_LABELS = frozenset({"sufficient", "insufficient", "ambiguous"})

_TAG = re.compile(r"<[^>]+>")
_BREAK_TAG = re.compile(r"(?i)<\s*(?:br\s*/?|/\s*(?:li|p|div)|li|p|div)\s*>")
_BULLET = re.compile(r"^\s*(?:[-*\u2022]+|\(?\d+[.)]|[A-Za-z][.)])\s+")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(<\[])" )
_WORD = re.compile(r"(?u)\b\w+(?:[./:-]+\w+)*\b")
_FINITE_VERB = re.compile(
    r"(?i)\b(?:is|are|was|were|be|been|has|have|had|can|could|must|should|will|would|"
    r"use|uses|set|sets|run|runs|select|selects|click|clicks|configure|configures|"
    r"install|installs|create|creates|remove|removes|add|adds|enable|enables|disable|"
    r"disables|specify|specifies|enter|enters|type|types|returns?|requires?|allows?)\b|"
    r"\b\w+(?:ed|ing)\b"
)


@dataclass(frozen=True)
class Phase03SemanticConfig:
    source_path: Path
    values: dict[str, Any]
    config_sha256: str

    @classmethod
    def load(cls, path: Path) -> "Phase03SemanticConfig":
        values = json.loads(path.read_text(encoding="utf-8"))
        required = {
            "development_annotations", "frozen_inputs", "primary_population",
            "candidate_models", "claim_segmentation", "context_aggregation",
            "threshold_search", "inference", "confirmation_sampling",
            "confirmation_gates", "test_seal",
        }
        missing = required - set(values)
        if missing:
            raise ValueError(f"Phase 3.6 config missing keys: {sorted(missing)}")
        if values["primary_population"]["splits"] != ["train", "validation"]:
            raise ValueError("Phase 3.6 primary population must remain TRAIN+VALIDATION only")
        if not values["primary_population"]["exclude_benchmark_impossible"]:
            raise ValueError("benchmark-impossible primary exclusion is frozen")
        if float(values["threshold_search"]["minimum_precision"]) != 0.90:
            raise ValueError("semantic development precision gate must remain 0.90")
        gates = values["confirmation_gates"]
        if float(gates["automatic_sufficient_precision_minimum"]) != 0.90:
            raise ValueError("confirmation precision gate must remain 0.90")
        if float(gates["prevalence_weighted_f1_minimum"]) != 0.85:
            raise ValueError("confirmation weighted-F1 gate must remain 0.85")
        if values["test_seal"] != {
            "calculate_test_semantic_scores": False,
            "calculate_test_semantic_aggregates": False,
        }:
            raise ValueError("Phase 3.6 TEST semantic outcomes must remain sealed")
        models = values["candidate_models"]
        if [int(row["candidate_order"]) for row in models] != list(range(1, len(models) + 1)):
            raise ValueError("candidate model order must be contiguous and frozen")
        return cls(path.resolve(), values, canonical_json_sha256(values))


def _semicolon_independent(parts: list[str]) -> bool:
    return len(parts) > 1 and all(
        len(_WORD.findall(part)) >= 4 and _FINITE_VERB.search(part) for part in parts
    )


def segment_reference_answer(answer: str) -> list[str]:
    """Split a reference answer without inventing or paraphrasing any claim text."""
    if not isinstance(answer, str) or not answer.strip() or answer.strip() == "-":
        return []
    text = html.unescape(answer).replace("\r\n", "\n").replace("\r", "\n")
    text = _BREAK_TAG.sub("\n", text)
    text = _TAG.sub(" ", text)
    lines: list[str] = []
    for raw_line in text.split("\n"):
        line = _BULLET.sub("", raw_line).strip()
        if line:
            lines.append(line)
    claims: list[str] = []
    for line in lines:
        for sentence in _SENTENCE_BOUNDARY.split(line):
            sentence = re.sub(r"\s+", " ", sentence).strip()
            if not sentence:
                continue
            semicolon_parts = [part.strip() for part in sentence.split(";") if part.strip()]
            claims.extend(semicolon_parts if _semicolon_independent(semicolon_parts) else [sentence])
    return claims


def claim_tokens(text: str) -> set[str]:
    return {token.casefold() for token in _WORD.findall(text)}


def _context_segments(text: str) -> list[str]:
    """Return source-preserving sentence/list snippets without semantic rewriting."""
    normalized = html.unescape(str(text)).replace("\r\n", "\n").replace("\r", "\n")
    normalized = _BREAK_TAG.sub("\n", normalized)
    normalized = _TAG.sub(" ", normalized)
    output: list[str] = []
    for raw_line in normalized.split("\n"):
        line = _BULLET.sub("", raw_line).strip()
        for sentence in _SENTENCE_BOUNDARY.split(line):
            sentence = re.sub(r"\s+", " ", sentence).strip()
            if sentence:
                output.append(sentence)
    return output or ([re.sub(r"\s+", " ", normalized).strip()] if normalized.strip() else [])


def build_context_units(claim: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build one deterministic claim-targeted premise from the whole retrieval bundle.

    Every chunk contributes candidate snippets to the ranking. Distinct chunks are
    preferred before a second snippet from an already selected chunk, enabling
    cross-chunk synthesis without truncating the bundle solely by retrieval order.
    """
    claim_terms = claim_tokens(claim)
    candidates: list[dict[str, Any]] = []
    for chunk in sorted(chunks, key=lambda row: (int(row["rank"]), str(row["chunk_id"]))):
        for segment_index, text in enumerate(_context_segments(str(chunk["text"])), 1):
            candidates.append({
                "chunk_id": str(chunk["chunk_id"]), "rank": int(chunk["rank"]),
                "segment_index": segment_index, "text": text,
                "overlap": len(claim_terms & claim_tokens(text)),
            })
    if not candidates:
        raise ValueError("retrieved context contains no scoreable text")
    ranked = sorted(candidates, key=lambda row: (
        -int(row["overlap"]), int(row["rank"]), int(row["segment_index"]),
        str(row["chunk_id"]),
    ))
    selected: list[dict[str, Any]] = []
    selected_chunks: set[str] = set()
    for row in ranked:
        if row["chunk_id"] not in selected_chunks:
            selected.append(row)
            selected_chunks.add(row["chunk_id"])
        if len(selected) == 3:
            break
    if len(selected) < 3:
        per_chunk = Counter(row["chunk_id"] for row in selected)
        for row in ranked:
            identity = (row["chunk_id"], row["segment_index"])
            if identity in {(item["chunk_id"], item["segment_index"]) for item in selected}:
                continue
            if per_chunk[row["chunk_id"]] >= 2:
                continue
            selected.append(row)
            per_chunk[row["chunk_id"]] += 1
            if len(selected) == 3:
                break
    selected.sort(key=lambda row: (
        int(row["rank"]), int(row["segment_index"]), str(row["chunk_id"])
    ))
    identity = {
        "unit_type": "claim_ranked_multi_chunk_window",
        "members": [
            {
                "chunk_id": row["chunk_id"], "rank": row["rank"],
                "segment_index": row["segment_index"],
                "text_sha256": canonical_json_sha256({"text": row["text"]}),
            }
            for row in selected
        ],
    }
    return [{
        "unit_id": canonical_json_sha256(identity),
        "unit_type": identity["unit_type"],
        "chunk_ids": [row["chunk_id"] for row in selected],
        "ranks": [row["rank"] for row in selected],
        "constituents": [row["text"] for row in selected],
    }]


def aggregate_claim_scores(unit_scores: Iterable[dict[str, Any]]) -> dict[str, Any]:
    scores = list(unit_scores)
    if not scores:
        raise ValueError("a claim must have at least one scored context unit")
    best = min(scores, key=lambda row: (-float(row["entailment"]), str(row["unit_id"])))
    return {
        "entailment": float(best["entailment"]),
        "neutral": float(best["neutral"]),
        "contradiction": float(best["contradiction"]),
        "best_unit_id": str(best["unit_id"]),
        "best_unit_type": str(best["unit_type"]),
        "best_chunk_ids_json": json.dumps(best["chunk_ids"], separators=(",", ":")),
        "maximum_unit_contradiction": max(float(row["contradiction"]) for row in scores),
        "context_unit_count": len(scores),
    }


def aggregate_condition_claims(claim_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not claim_rows:
        raise ValueError("a semantic-evaluable condition must contain reference claims")
    entailments = [float(row["entailment"]) for row in claim_rows]
    contradictions = [float(row["contradiction"]) for row in claim_rows]
    return {
        "claim_count": len(claim_rows),
        "minimum_claim_entailment": min(entailments),
        "mean_claim_entailment": sum(entailments) / len(entailments),
        "maximum_selected_premise_contradiction": max(contradictions),
        "maximum_any_unit_contradiction": max(
            float(row["maximum_unit_contradiction"]) for row in claim_rows
        ),
    }


def semantic_prediction(row: dict[str, Any], thresholds: dict[str, float]) -> int:
    return int(
        float(row["minimum_claim_entailment"]) >= thresholds["minimum_claim_entailment"]
        and float(row["mean_claim_entailment"]) >= thresholds["mean_claim_entailment"]
        and float(row["maximum_selected_premise_contradiction"])
        < thresholds["maximum_selected_premise_contradiction"]
    )


def threshold_grid(config: dict[str, Any]) -> list[dict[str, float]]:
    search = config["threshold_search"]
    output = []
    for minimum in search["minimum_claim_entailment"]:
        for mean in search["mean_claim_entailment"]:
            if search["mean_threshold_must_be_at_least_minimum_threshold"] and mean < minimum:
                continue
            for contradiction in search["maximum_selected_premise_contradiction"]:
                output.append({
                    "minimum_claim_entailment": float(minimum),
                    "mean_claim_entailment": float(mean),
                    "maximum_selected_premise_contradiction": float(contradiction),
                })
    return output


def confusion(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    cells = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    for row in rows:
        truth = 1 if row["manual_label"] == "sufficient" else 0
        pred = int(row["prediction"])
        cells["tp" if pred and truth else "fp" if pred else "fn" if truth else "tn"] += 1
    return cells


def metrics(cells: dict[str, float]) -> dict[str, float | None]:
    tn, fp, fn, tp = (cells[name] for name in ("tn", "fp", "fn", "tp"))
    total = tn + fp + fn + tp
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else 0.0 if precision == 0 or recall == 0 else None
    )
    return {
        "accuracy": (tp + tn) / total if total else None,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def weighted_metrics(
    rows: list[dict[str, Any]], population_counts: dict[str, int], strata: Iterable[str],
) -> dict[str, Any]:
    strata = tuple(strata)
    total = sum(int(population_counts[stratum]) for stratum in strata)
    weighted = {cell: 0.0 for cell in ("tn", "fp", "fn", "tp")}
    by_stratum = []
    complete = True
    for stratum in strata:
        selected = [
            row for row in rows
            if row["sampling_stratum"] == stratum and row["manual_label"] != "ambiguous"
        ]
        cells = confusion(selected)
        values = metrics(cells)
        prevalence = int(population_counts[stratum]) / total
        if not selected:
            complete = False
        else:
            for cell in weighted:
                weighted[cell] += prevalence * cells[cell] / len(selected)
        by_stratum.append({
            "sampling_stratum": stratum,
            "development_population_count": int(population_counts[stratum]),
            "completed_count": len(selected),
            "confusion_matrix": cells,
            **values,
        })
    values = metrics(weighted) if complete else {
        "accuracy": None, "precision": None, "recall": None, "f1": None,
    }
    return {
        "status": "available" if complete else "insufficient_stratum_coverage",
        "population_total": total,
        "estimated_confusion_proportions": weighted if complete else None,
        "metrics_by_stratum": by_stratum,
        **values,
    }


def confirmation_stratum(strict_label: int, semantic_label: int) -> str:
    return (
        "strict_positive_" if int(strict_label) else "strict_negative_"
    ) + ("semantic_positive" if int(semantic_label) else "semantic_negative")


def select_confirmation_sample(
    rows: list[dict[str, Any]], original_questions: set[str], config: dict[str, Any],
) -> list[dict[str, Any]]:
    sampling = config["confirmation_sampling"]
    seed = int(sampling["seed"])
    eligible = [
        row for row in rows
        if row["split"] in set(sampling["eligible_splits"])
        and row["question_id"] not in original_questions
        and row.get("semantic_label_status", "evaluable") == "evaluable"
        and row.get("y_suff_semantic") in {0, 1}
    ]
    for row in eligible:
        row["confirmation_sampling_stratum"] = confirmation_stratum(
            int(row["y_suff_strict"]), int(row["y_suff_semantic"])
        )
    selected: list[dict[str, Any]] = []
    used_questions: set[str] = set()
    strategy_counts: Counter[str] = Counter()
    depth_counts: Counter[int] = Counter()
    split_counts: Counter[str] = Counter()

    def stable(row: dict[str, Any], purpose: str) -> str:
        return canonical_json_sha256({
            "seed": seed, "purpose": purpose, "question_id": row["question_id"],
            "retrieval_strategy": row["retrieval_strategy"], "k": int(row["k"]),
            "semantic_config_sha256": row["semantic_config_sha256"],
        })

    def take(pool: list[dict[str, Any]], count: int, purpose: str) -> None:
        for _ in range(count):
            available = [row for row in pool if row["question_id"] not in used_questions]
            if not available:
                break
            row = min(available, key=lambda item: (
                strategy_counts[item["retrieval_strategy"]],
                depth_counts[int(item["k"])], split_counts[item["split"]],
                stable(item, purpose),
            ))
            selected.append(row)
            used_questions.add(row["question_id"])
            strategy_counts[row["retrieval_strategy"]] += 1
            depth_counts[int(row["k"])] += 1
            split_counts[row["split"]] += 1

    targets = sampling["mutually_exclusive_strata_targets"]
    for stratum in CONFIRMATION_STRATA:
        pool = [row for row in eligible if row["confirmation_sampling_stratum"] == stratum]
        take(pool, int(targets[stratum]), stratum)
    target_size = int(sampling["target_size"])
    if len(selected) < target_size:
        take(eligible, target_size - len(selected), "availability_fill")
    if len(selected) != target_size:
        raise ValueError(f"confirmation sample has {len(selected)} rows; expected {target_size}")
    selected.sort(key=lambda row: stable(row, "blind_order"))
    return selected
