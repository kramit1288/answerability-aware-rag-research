from __future__ import annotations

import unicodedata

from answerability_rag.sufficiency.alignment import align_answer_to_document
from answerability_rag.sufficiency.normalization import (
    normalize_alignment_text, normalize_alignment_text_with_mapping,
    normalize_retrieval_source_with_mapping,
)


def test_alignment_normalization_and_reversible_offsets() -> None:
    source = "Prefix ＡLPHA—Beta &amp; GAMMA suffix"
    mapped = normalize_alignment_text_with_mapping(source)
    assert mapped.text == "prefix alpha beta gamma suffix"
    start = mapped.text.index("alpha beta gamma")
    raw_start, raw_end = mapped.source_span(start, start + len("alpha beta gamma"))
    assert normalize_alignment_text(source[raw_start:raw_end]) == "alpha beta gamma"
    assert unicodedata.normalize("NFKC", source[raw_start:raw_end]).casefold()


def test_retrieval_source_mapping_matches_phase02_transform() -> None:
    raw = "  Alpha\r\n\tBeta  data:image/png;base64,AAAA  Gamma  "
    mapped = normalize_retrieval_source_with_mapping(raw)
    assert mapped.text == "Alpha Beta Gamma"
    alpha_start, alpha_end = mapped.source_span(0, 5)
    assert raw[alpha_start:alpha_end] == "Alpha"


def test_alignment_hierarchy_exact_normalized_multiple_and_unresolved() -> None:
    raw = "Alpha Beta; middle. Alpha Beta. A separate value: Café mode."
    canonical = normalize_retrieval_source_with_mapping(raw).text
    exact = align_answer_to_document(
        question_id="Q1", answer="Alpha Beta", raw_source=raw, canonical_source=canonical,
        doc_id="d", filename="d.txt",
    )
    assert exact.status == "aligned_exact"
    assert len(exact.candidates) == 2
    assert all(candidate.raw_char_start is not None for candidate in exact.candidates)
    normalized = align_answer_to_document(
        question_id="Q2", answer="CAFE—MODE", raw_source=raw, canonical_source=canonical,
        doc_id="d", filename="d.txt",
    )
    assert normalized.status == "unresolved"  # accent removal is deliberately not allowed
    punctuation = align_answer_to_document(
        question_id="Q3", answer="alpha—beta", raw_source=raw, canonical_source=canonical,
        doc_id="d", filename="d.txt",
    )
    assert punctuation.status == "aligned_normalized"
    assert len(punctuation.candidates) == 2
    empty = align_answer_to_document(
        question_id="Q4", answer="", raw_source=raw, canonical_source=canonical,
        doc_id="d", filename="d.txt",
    )
    assert empty.status == "unresolved"
    assert empty.exclusion_reason == "empty_answerable_reference"
