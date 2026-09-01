"""Frozen MiniLM candidate selection, NLI aggregation, and RAGTruth thresholding."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from answerability_rag.sufficiency.semantic_nli import NLIScorer


_PASSAGE_MARKER = re.compile(r"(?im)^\s*passage\s+(\d+)\s*:")


def parse_ragtruth_passages(text: str) -> list[dict[str, Any]]:
    """Parse the released `passage N:` QA context while retaining source order."""
    value = str(text or "")
    markers = list(_PASSAGE_MARKER.finditer(value))
    if not markers:
        cleaned = value.strip()
        return ([{"chunk_id": "passage-1", "rank": 1, "text": cleaned}] if cleaned else [])
    output: list[dict[str, Any]] = []
    for index, marker in enumerate(markers):
        start = marker.end()
        end = markers[index + 1].start() if index + 1 < len(markers) else len(value)
        passage = value[start:end].strip()
        if passage:
            rank = int(marker.group(1))
            output.append({"chunk_id": f"passage-{rank}", "rank": rank, "text": passage})
    output.sort(key=lambda row: (int(row["rank"]), str(row["chunk_id"])))
    return output


def select_candidate_chunks(
    similarities: Iterable[float], chunks: list[dict[str, Any]], candidate_count: int = 3,
) -> list[dict[str, Any]]:
    values = list(similarities)
    if len(values) != len(chunks):
        raise ValueError("similarity/chunk lengths differ")
    ranked = sorted(
        ({**chunk, "similarity": float(score)} for score, chunk in zip(values, chunks)),
        key=lambda row: (-row["similarity"], int(row["rank"]), str(row["chunk_id"])),
    )
    return ranked[:candidate_count]


def aggregate_candidate_nli(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Use maximum entailment and maximum contradiction over independent candidates."""
    evaluable = [row for row in rows if row.get("evaluation_status") == "evaluable"]
    if not evaluable:
        return {
            "evaluation_status": "evaluator_unevaluable",
            "claim_support_score": None,
            "claim_unsupportedness_score": None,
            "maximum_claim_contradiction": None,
            "supporting_chunk_id": None,
        }
    supporting = min(
        evaluable,
        key=lambda row: (
            -float(row["entailment"]), int(row["rank"]), str(row["chunk_id"]),
        ),
    )
    support = float(supporting["entailment"])
    return {
        "evaluation_status": "evaluable",
        "claim_support_score": support,
        "claim_unsupportedness_score": 1.0 - support,
        "maximum_claim_contradiction": max(float(row["contradiction"]) for row in evaluable),
        "supporting_chunk_id": str(supporting["chunk_id"]),
    }


def binary_metrics(truth: Iterable[int], prediction: Iterable[int]) -> dict[str, float | int | None]:
    y = np.asarray(list(truth), dtype=int)
    p = np.asarray(list(prediction), dtype=int)
    if len(y) != len(p):
        raise ValueError("truth/prediction lengths differ")
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else 0.0 if precision == 0 or recall == 0 else None
    )
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "precision": precision, "recall": recall, "f1": f1}


def threshold_grid() -> list[float]:
    return [index / 50 for index in range(51)]


def select_support_threshold(train_claims: list[dict[str, Any]]) -> tuple[float, list[dict[str, Any]]]:
    """Select on official RAGTruth TRAIN only; TEST labels cannot enter this API."""
    if not train_claims:
        raise ValueError("RAGTruth TRAIN threshold population is empty")
    if {str(row["official_split"]).casefold() for row in train_claims} != {"train"}:
        raise ValueError("support threshold selection accepts RAGTruth TRAIN rows only")
    eligible = [
        row for row in train_claims
        if row.get("evaluation_status") == "evaluable"
        and row.get("human_unsupported") in {0, 1}
        and row.get("claim_support_score") is not None
    ]
    if not eligible:
        raise ValueError("RAGTruth TRAIN has no evaluable labelled claims")
    rows: list[dict[str, Any]] = []
    for threshold in threshold_grid():
        truth = [int(row["human_unsupported"]) for row in eligible]
        prediction = [int(float(row["claim_support_score"]) < threshold) for row in eligible]
        metric = binary_metrics(truth, prediction)
        rows.append({"t_support": threshold, "eligible_claim_count": len(eligible), **metric})
    selected = min(
        rows,
        key=lambda row: (
            -float(row["f1"] if row["f1"] is not None else -1.0),
            -float(row["precision"] if row["precision"] is not None else -1.0),
            -float(row["t_support"]),
        ),
    )
    for row in rows:
        row["selected"] = row is selected
    return float(selected["t_support"]), rows


def discrimination_metrics(
    truth: Iterable[int], prediction: Iterable[int], unsupportedness: Iterable[float],
) -> dict[str, float | int | None]:
    y = np.asarray(list(truth), dtype=int)
    p = np.asarray(list(prediction), dtype=int)
    scores = np.asarray(list(unsupportedness), dtype=float)
    result = binary_metrics(y, p)
    result["n"] = len(y)
    result["auroc"] = float(roc_auc_score(y, scores)) if len(set(y.tolist())) == 2 else None
    result["auprc"] = float(average_precision_score(y, scores)) if (y == 1).any() else None
    return result


def response_grounding_metrics(claims: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    claim_count = len(claims)
    evaluable = [row for row in claims if row.get("evaluation_status") == "evaluable"]
    unevaluable = claim_count - len(evaluable)
    if not evaluable:
        return {
            "claim_count": claim_count,
            "evaluable_claim_count": 0,
            "unevaluable_claim_count": unevaluable,
            "mean_claim_support_score": None,
            "minimum_claim_support_score": None,
            "unsupported_claim_count": 0,
            "unsupported_claim_rate": None,
            "maximum_claim_contradiction": None,
            "fully_supported_response": None if claim_count == 0 else False,
            "response_with_any_unsupported_claim": None,
            "response_grounding_status": "no_claims" if claim_count == 0 else "no_evaluable_claim",
        }
    supports = [float(row["claim_support_score"]) for row in evaluable]
    unsupported = sum(score < threshold for score in supports)
    return {
        "claim_count": claim_count,
        "evaluable_claim_count": len(evaluable),
        "unevaluable_claim_count": unevaluable,
        "mean_claim_support_score": sum(supports) / len(supports),
        "minimum_claim_support_score": min(supports),
        "unsupported_claim_count": unsupported,
        "unsupported_claim_rate": unsupported / len(supports),
        "maximum_claim_contradiction": max(float(row["maximum_claim_contradiction"]) for row in evaluable),
        "fully_supported_response": unsupported == 0 and unevaluable == 0,
        "response_with_any_unsupported_claim": unsupported > 0,
        "response_grounding_status": "evaluable" if unevaluable == 0 else "partially_evaluable",
    }


@dataclass
class GroundingModels:
    embedding_model_id: str
    embedding_revision: str
    nli_model_id: str
    nli_revision: str
    cache_dir: Path
    nli_max_length: int = 256
    nli_batch_size: int = 16
    nli_length_bucketing: bool = False

    def __post_init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.embedder = SentenceTransformer(
            self.embedding_model_id,
            revision=self.embedding_revision,
            cache_folder=str(self.cache_dir),
            device="cpu",
        )
        self.embedder.eval()
        self.nli = NLIScorer(
            self.nli_model_id, self.nli_revision, self.cache_dir,
            self.nli_max_length, self.nli_batch_size,
        )

    def close(self) -> None:
        self.nli.close()
        del self.embedder

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        values = self.embedder.encode(
            texts, batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=True,
        )
        return np.asarray(values, dtype=np.float32)

    def claim_fits(self, claim: str) -> tuple[bool, int, int]:
        special = int(self.nli.tokenizer.num_special_tokens_to_add(pair=True))
        length = len(self.nli.tokenizer.encode(claim, add_special_tokens=False))
        return length + special < self.nli_max_length, length, special

    def score_pairs(self, pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.nli_length_bucketing:
            return score_pairs_length_bucketed(self.nli, pairs)
        return self.nli.score(pairs)


def score_pairs_length_bucketed(scorer: Any, pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score exact pairs in length-sorted dynamic-padding batches.

    This is an execution-only variant of the frozen Phase 3 scorer. It renders
    and tokenizes the same premise/hypothesis strings, sorts only for padding
    efficiency, and restores the original pair order before returning.
    """
    import torch

    rows = list(pairs)
    prepared: list[tuple[int, int, dict[str, Any], str, list[int]]] = []
    for index, row in enumerate(rows):
        premise = scorer.render_premise(row["claim_text"], row["unit"])
        claim = row["claim_text"]
        single = scorer.tokenizer(premise, claim, add_special_tokens=True, truncation=False)
        prepared.append((len(single["input_ids"]), index, row, premise, list(single["input_ids"])))
    prepared.sort(key=lambda item: (item[0], item[1]))
    output: list[dict[str, Any] | None] = [None] * len(rows)
    for start in range(0, len(prepared), scorer.batch_size):
        batch = prepared[start:start + scorer.batch_size]
        encoded = scorer.tokenizer(
            [item[3] for item in batch], [item[2]["claim_text"] for item in batch],
            padding=True, truncation=False, return_tensors="pt",
        )
        for item_index, item in enumerate(batch):
            length = int(encoded["attention_mask"][item_index].sum().item())
            actual = encoded["input_ids"][item_index, :length].tolist()
            if actual != item[4]:
                raise ValueError("length-bucketed token IDs differ from unbucketed token IDs")
        with torch.inference_mode():
            logits = scorer.model(**encoded).logits
            probabilities = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        for item_index, item in enumerate(batch):
            row = item[2]
            values = probabilities[item_index]
            output[item[1]] = {
                **{key: value for key, value in row.items() if key != "unit"},
                "unit_id": row["unit"]["unit_id"],
                "unit_type": row["unit"]["unit_type"],
                "chunk_ids": row["unit"]["chunk_ids"],
                "entailment": float(values[scorer.label_indices["entailment"]]),
                "neutral": float(values[scorer.label_indices["neutral"]]),
                "contradiction": float(values[scorer.label_indices["contradiction"]]),
            }
    return [row for row in output if row is not None]


def candidate_json(rows: list[dict[str, Any]], fields: list[str]) -> str:
    return json.dumps(
        [{field: row.get(field) for field in fields} for row in rows],
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
