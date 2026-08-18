"""Machine-readable leakage boundaries for future classifier feature exports."""

from __future__ import annotations

from typing import Iterable

from .alignment import ALIGNMENT_FIELDS
from .labeling import LABEL_FIELDS


INFERENCE_LABEL_FIELDS = frozenset({"retrieval_strategy", "k"})
LABEL_ONLY_FIELDS = frozenset({
    "benchmark_is_impossible", "reference_status", "gold_alignment_status", "label_status",
    "y_suff", "label_method", "label_provenance", "accepted_gold_span_count",
    "covering_chunk_ids_json", "exclusion_reason",
})
EVALUATION_ONLY_FIELDS = frozenset({
    "gold_document_hit", "maximum_span_coverage_fraction", "fully_covered_span_count",
    "partial_overlap", "first_covering_rank",
})


def build_column_governance() -> dict:
    label_classes = {}
    for field in LABEL_FIELDS:
        if field in INFERENCE_LABEL_FIELDS:
            classification = "inference_available_feature"
        elif field in LABEL_ONLY_FIELDS:
            classification = "label_only"
        elif field in EVALUATION_ONLY_FIELDS:
            classification = "evaluation_only"
        else:
            classification = "provenance_only"
        label_classes[field] = classification
    alignment_classes = {
        field: ("provenance_only" if field in {
            "schema_version", "alignment_id", "question_id", "split", "phase01_split_sha256",
            "phase02_chunk_config_sha256", "phase03_config_sha256", "alignment_version",
            "normalization_version",
        } else "label_only")
        for field in ALIGNMENT_FIELDS
    }
    return {
        "schema_version": "phase03-column-governance-v1",
        "classifications": [
            "inference_available_feature", "label_only", "provenance_only", "evaluation_only"
        ],
        "artifacts": {
            "context_sufficiency_labels": label_classes,
            "techqa_evidence_alignments": alignment_classes,
        },
        "future_classifier_feature_allowlist_from_phase03_labels": sorted(INFERENCE_LABEL_FIELDS),
        "rule": "feature export must reject every non-inference_available_feature column",
    }


def validate_feature_selection(fields: Iterable[str], governance: dict) -> None:
    classes = governance["artifacts"]["context_sufficiency_labels"]
    rejected = sorted(field for field in fields if classes.get(field) != "inference_available_feature")
    if rejected:
        raise ValueError(f"forbidden Phase 3 classifier features selected: {rejected}")
