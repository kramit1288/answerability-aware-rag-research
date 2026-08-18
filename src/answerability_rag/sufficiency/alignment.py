"""Evidence-span alignment for answerable TechQA questions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from answerability_rag.data.schemas import TechQAQuestion
from answerability_rag.hashing import canonical_json_sha256, sha256_text

from .normalization import (
    find_all, normalize_alignment_text, normalize_alignment_text_with_mapping,
    normalize_retrieval_source_with_mapping,
)


ALIGNMENT_FIELDS = (
    "schema_version", "alignment_id", "question_id", "split", "reference_status",
    "gold_alignment_status", "alignment_method", "candidate_index", "candidate_span_count",
    "gold_doc_id", "filename", "source_coordinate_system", "source_char_start",
    "source_char_end", "raw_char_start", "raw_char_end", "reference_answer",
    "reference_answer_sha256", "evidence_text", "evidence_text_sha256", "character_length",
    "token_length", "raw_source_sha256", "normalized_source_sha256", "alignment_version",
    "normalization_version", "phase01_split_sha256", "phase02_chunk_config_sha256",
    "phase03_config_sha256", "exclusion_reason",
)


@dataclass(frozen=True)
class EvidenceCandidate:
    question_id: str
    doc_id: str
    filename: str
    source_char_start: int
    source_char_end: int
    raw_char_start: int | None
    raw_char_end: int | None
    evidence_text: str
    method: str


@dataclass(frozen=True)
class QuestionAlignment:
    question_id: str
    status: str
    method: str
    candidates: tuple[EvidenceCandidate, ...]
    exclusion_reason: str | None


def align_answer_to_document(
    *, question_id: str, answer: str, raw_source: str, canonical_source: str,
    doc_id: str, filename: str,
) -> QuestionAlignment:
    """Apply exact raw matching, then reversible normalized matching, retaining every span."""
    if not answer:
        return QuestionAlignment(question_id, "unresolved", "none", (), "empty_answerable_reference")
    mapped_source = normalize_retrieval_source_with_mapping(raw_source)
    if mapped_source.text != canonical_source:
        raise ValueError(f"canonical source mismatch for {doc_id}")
    exact = find_all(raw_source, answer)
    candidates: list[EvidenceCandidate] = []
    if exact:
        for raw_start, raw_end in exact:
            positions = [
                index for index, (start, end) in enumerate(
                    zip(mapped_source.source_starts, mapped_source.source_ends)
                ) if start < raw_end and end > raw_start
            ]
            if not positions:
                raise ValueError(f"exact raw span disappeared from canonical source: {question_id}")
            source_start, source_end = min(positions), max(positions) + 1
            candidates.append(EvidenceCandidate(
                question_id, doc_id, filename, source_start, source_end, raw_start, raw_end,
                canonical_source[source_start:source_end], "exact_source_text",
            ))
        method = "exact_source_text"
    else:
        normalized_answer = normalize_alignment_text(answer)
        if not normalized_answer:
            return QuestionAlignment(
                question_id, "unresolved", "none", (), "empty_after_alignment_normalization"
            )
        normalized_source = normalize_alignment_text_with_mapping(canonical_source)
        for match_start, match_end in find_all(normalized_source.text, normalized_answer):
            source_start, source_end = normalized_source.source_span(match_start, match_end)
            evidence_text = canonical_source[source_start:source_end]
            if normalize_alignment_text(evidence_text) != normalized_answer:
                raise ValueError(f"normalized match is not reversible for {question_id}")
            candidates.append(EvidenceCandidate(
                question_id, doc_id, filename, source_start, source_end, None, None,
                evidence_text, "normalized_text",
            ))
        method = "normalized_text"
    unique = {
        (candidate.doc_id, candidate.source_char_start, candidate.source_char_end): candidate
        for candidate in candidates
    }
    ordered = tuple(unique[key] for key in sorted(unique))
    if not ordered:
        return QuestionAlignment(question_id, "unresolved", "none", (), "no_defensible_text_match")
    status = "aligned_exact" if method == "exact_source_text" else "aligned_normalized"
    return QuestionAlignment(question_id, status, method, ordered, None)


def build_alignment_rows(
    questions: Iterable[TechQAQuestion], assignments: dict[str, dict[str, str]],
    corpus_by_filename: dict[str, dict[str, str]], corpus_root: Path, tokenizer,
    *, config: dict[str, Any], phase03_config_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, QuestionAlignment]]:
    expected = config["expected_inputs"]
    alignment_config = config["alignment"]
    rows: list[dict[str, Any]] = []
    alignments: dict[str, QuestionAlignment] = {}
    for question in sorted(questions, key=lambda item: item.question_id):
        if question.is_impossible:
            continue
        assignment = assignments[question.question_id]
        if not question.contexts:
            alignment = QuestionAlignment(
                question.question_id, "unresolved", "none", (), "missing_confirmed_gold_document"
            )
        else:
            candidates: list[EvidenceCandidate] = []
            methods: list[str] = []
            reasons: list[str] = []
            for context in question.contexts:
                document = corpus_by_filename[context.filename]
                raw_source = (corpus_root / document["archive_path"]).read_text(encoding="utf-8")
                canonical_source = normalize_retrieval_source_with_mapping(raw_source).text
                result = align_answer_to_document(
                    question_id=question.question_id, answer=question.answer, raw_source=raw_source,
                    canonical_source=canonical_source, doc_id=document["doc_id"],
                    filename=context.filename,
                )
                candidates.extend(result.candidates)
                if result.candidates:
                    methods.append(result.method)
                elif result.exclusion_reason:
                    reasons.append(result.exclusion_reason)
            unique = {
                (candidate.doc_id, candidate.source_char_start, candidate.source_char_end): candidate
                for candidate in candidates
            }
            accepted = tuple(unique[key] for key in sorted(unique))
            if accepted:
                method = "exact_source_text" if "exact_source_text" in methods else "normalized_text"
                status = "aligned_exact" if method == "exact_source_text" else "aligned_normalized"
                alignment = QuestionAlignment(question.question_id, status, method, accepted, None)
            else:
                reason = sorted(set(reasons))[0] if reasons else "no_defensible_text_match"
                alignment = QuestionAlignment(question.question_id, "unresolved", "none", (), reason)
        alignments[question.question_id] = alignment
        reference_status = assignment["reference_status"]
        if not alignment.candidates:
            identity = {"question_id": question.question_id, "status": "unresolved",
                        "phase03_config_sha256": phase03_config_sha256}
            rows.append({
                "schema_version": "phase03-alignments-v1",
                "alignment_id": canonical_json_sha256(identity), "question_id": question.question_id,
                "split": assignment["split"], "reference_status": reference_status,
                "gold_alignment_status": alignment.status, "alignment_method": "none",
                "candidate_index": None, "candidate_span_count": 0, "gold_doc_id": None,
                "filename": None, "source_coordinate_system": alignment_config["source_coordinate_system"],
                "source_char_start": None, "source_char_end": None, "raw_char_start": None,
                "raw_char_end": None, "reference_answer": question.answer,
                "reference_answer_sha256": sha256_text(question.answer), "evidence_text": None,
                "evidence_text_sha256": None, "character_length": None, "token_length": None,
                "raw_source_sha256": None, "normalized_source_sha256": None,
                "alignment_version": alignment_config["version"],
                "normalization_version": alignment_config["normalization_version"],
                "phase01_split_sha256": expected["phase01_split_sha256"],
                "phase02_chunk_config_sha256": expected["phase02_chunk_config_sha256"],
                "phase03_config_sha256": phase03_config_sha256,
                "exclusion_reason": alignment.exclusion_reason,
            })
            continue
        documents = {row["doc_id"]: row for row in corpus_by_filename.values()}
        for index, candidate in enumerate(alignment.candidates, 1):
            document = documents[candidate.doc_id]
            token_length = len(tokenizer(
                candidate.evidence_text, add_special_tokens=False, truncation=False, verbose=False
            )["input_ids"])
            identity = {
                "question_id": question.question_id, "doc_id": candidate.doc_id,
                "source_char_start": candidate.source_char_start,
                "source_char_end": candidate.source_char_end,
                "alignment_version": alignment_config["version"],
            }
            rows.append({
                "schema_version": "phase03-alignments-v1",
                "alignment_id": canonical_json_sha256(identity), "question_id": question.question_id,
                "split": assignment["split"], "reference_status": reference_status,
                "gold_alignment_status": alignment.status, "alignment_method": candidate.method,
                "candidate_index": index, "candidate_span_count": len(alignment.candidates),
                "gold_doc_id": candidate.doc_id, "filename": candidate.filename,
                "source_coordinate_system": alignment_config["source_coordinate_system"],
                "source_char_start": candidate.source_char_start,
                "source_char_end": candidate.source_char_end,
                "raw_char_start": candidate.raw_char_start, "raw_char_end": candidate.raw_char_end,
                "reference_answer": question.answer,
                "reference_answer_sha256": sha256_text(question.answer),
                "evidence_text": candidate.evidence_text,
                "evidence_text_sha256": sha256_text(candidate.evidence_text),
                "character_length": candidate.source_char_end - candidate.source_char_start,
                "token_length": token_length, "raw_source_sha256": document["raw_sha256"],
                "normalized_source_sha256": document["normalized_sha256"],
                "alignment_version": alignment_config["version"],
                "normalization_version": alignment_config["normalization_version"],
                "phase01_split_sha256": expected["phase01_split_sha256"],
                "phase02_chunk_config_sha256": expected["phase02_chunk_config_sha256"],
                "phase03_config_sha256": phase03_config_sha256, "exclusion_reason": None,
            })
    rows.sort(key=lambda row: (row["question_id"], row["alignment_id"]))
    return rows, alignments
