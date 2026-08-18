"""Phase 1 manifest schemas and exceptional manual-alignment records."""

from __future__ import annotations

from typing import Any


MANUAL_ALIGNMENT_FIELDS = (
    "schema_version", "question_id", "status", "manually_aligned_document", "evidence_start",
    "evidence_end", "annotator", "rationale", "timestamp", "provenance",
)


def unresolved_manual_alignment_rows(schema_version: str, timestamp: str) -> list[dict[str, Any]]:
    """Record the two frozen anomalies without fabricating document/span evidence."""
    rationale = (
        "Answerable row has an empty reference. Phase 1 records the anomaly but, under the frozen "
        "Phase 1 plan, does not infer a document or evidence span. Exclude from analyses requiring "
        "confirmed gold evidence until a later governed manual alignment resolves it."
    )
    return [
        {
            "schema_version": schema_version, "question_id": question_id, "status": "unresolved",
            "manually_aligned_document": None, "evidence_start": None, "evidence_end": None,
            "annotator": None, "rationale": rationale, "timestamp": timestamp,
            "provenance": "pinned_techqa_empty_answerable_reference",
        }
        for question_id in ("DEV_Q014", "DEV_Q094")
    ]
