"""Pinned TechQA QA/corpus loading, schema checks, and official-corpus metadata."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
import zipfile

from ..hashing import sha256_bytes, sha256_text
from .schemas import TechQAContext, TechQAQuestion, ValidationReport


TECHQA_FIELDS = ("id", "question", "answer", "is_impossible", "contexts")
CORPUS_MANIFEST_FIELDS = (
    "schema_version", "dataset_revision", "corpus_archive_sha256", "doc_id", "archive_path",
    "filename", "raw_sha256", "normalized_sha256", "raw_bytes", "normalized_chars", "encoding",
    "cleaning_version", "removed_payload_count", "status",
)
EMPTY_ANSWERABLE_IDS = frozenset({"DEV_Q014", "DEV_Q094"})
CLEANING_VERSION = "strict-utf8-nfkc-whitespace-data-uri-v1"
_DATA_URI = re.compile(r"data:[^\s;,]+(?:;[^\s;,]+)*;base64,[A-Za-z0-9+/]+={0,2}", re.IGNORECASE)


def load_techqa_rows(path: Path) -> list[TechQAQuestion]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("TechQA train.json must contain a top-level list")
    rows: list[TechQAQuestion] = []
    for index, value in enumerate(raw):
        if not isinstance(value, dict) or set(value) != set(TECHQA_FIELDS):
            observed = sorted(value) if isinstance(value, dict) else type(value).__name__
            raise ValueError(f"TechQA row {index} fields differ: {observed}")
        for key in ("id", "question", "answer"):
            if not isinstance(value[key], str):
                raise ValueError(f"TechQA row {index} field {key} must be a string")
        if not isinstance(value["is_impossible"], bool) or not isinstance(value["contexts"], list):
            raise ValueError(f"TechQA row {index} has invalid label/contexts types")
        contexts: list[TechQAContext] = []
        for context_index, context in enumerate(value["contexts"]):
            if not isinstance(context, dict) or set(context) != {"filename", "text"}:
                raise ValueError(f"TechQA {value['id']} context {context_index} schema differs")
            if not all(isinstance(context[key], str) for key in ("filename", "text")):
                raise ValueError(f"TechQA {value['id']} context {context_index} fields must be strings")
            contexts.append(TechQAContext(context["filename"], context["text"]))
        rows.append(TechQAQuestion(
            value["id"], value["question"], value["answer"], value["is_impossible"], tuple(contexts)
        ))
    return rows


def validate_techqa_rows(rows: list[TechQAQuestion], expected_rows: int = 910,
                         expected_answerable: int = 610, expected_impossible: int = 300) -> ValidationReport:
    report = ValidationReport()
    ids = [row.question_id for row in rows]
    answerable = [row for row in rows if not row.is_impossible]
    impossible = [row for row in rows if row.is_impossible]
    report.add("techqa_row_count", "Pinned executable row count", expected_rows, len(rows), len(rows) == expected_rows)
    report.add("techqa_unique_ids", "Question IDs are unique", len(rows), len(set(ids)), len(ids) == len(set(ids)))
    report.add("techqa_nonempty_ids_questions", "IDs/questions are non-empty", True,
               all(row.question_id and row.question.strip() for row in rows),
               all(row.question_id and row.question.strip() for row in rows))
    report.add("techqa_answerable_count", "Answerable count", expected_answerable, len(answerable), len(answerable) == expected_answerable)
    report.add("techqa_impossible_count", "Impossible count", expected_impossible, len(impossible), len(impossible) == expected_impossible)
    context_rule = all(len(row.contexts) == (0 if row.is_impossible else 1) for row in rows)
    report.add("techqa_context_cardinality", "Answerable rows have one context; impossible rows none", True, context_rule, context_rule)
    impossible_answers = all(row.answer == "-" for row in impossible)
    report.add("techqa_impossible_answer_sentinel", "Impossible answers use '-'", True, impossible_answers, impossible_answers)
    empty_answerable = {row.question_id for row in answerable if row.answer == ""}
    report.add("techqa_empty_answerable_ids", "Only frozen empty-reference anomalies exist",
               sorted(EMPTY_ANSWERABLE_IDS), sorted(empty_answerable), empty_answerable == EMPTY_ANSWERABLE_IDS)
    provenance_ok = all(row.provenance_partition in {"TRAIN", "DEV"} for row in rows)
    report.add("techqa_provenance_prefixes", "Provenance prefixes are recognized", True, provenance_ok, provenance_ok)
    return report


def normalize_corpus_text(raw_text: str) -> tuple[str, int]:
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFKC", text)
    text, removed = _DATA_URI.subn(" ", text)
    return re.sub(r"\s+", " ", text).strip(), removed


def build_corpus_manifest(extracted_root: Path, *, archive: Path | None = None,
                          schema_version: str, dataset_revision: str,
                          corpus_archive_sha256: str, expected_documents: int = 28481
                          ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    basename_counts: Counter[str] = Counter()
    content_counts: Counter[str] = Counter()

    def append_record(archive_path: str, raw: bytes) -> None:
        try:
            decoded = raw.decode("utf-8", errors="strict")
            encoding, status = "utf-8", "ok"
        except UnicodeDecodeError as error:
            raise ValueError(f"official corpus file is not strict UTF-8: {archive_path}: {error}") from error
        normalized, removed = normalize_corpus_text(decoded)
        raw_hash = sha256_bytes(raw)
        filename = PurePosixPath(archive_path).name
        basename_counts[filename] += 1
        content_counts[raw_hash] += 1
        manifest.append({
            "schema_version": schema_version, "dataset_revision": dataset_revision,
            "corpus_archive_sha256": corpus_archive_sha256,
            "doc_id": "techqa-doc:" + archive_path, "archive_path": archive_path,
            "filename": filename, "raw_sha256": raw_hash,
            "normalized_sha256": sha256_text(normalized), "raw_bytes": len(raw),
            "normalized_chars": len(normalized), "encoding": encoding,
            "cleaning_version": CLEANING_VERSION, "removed_payload_count": removed, "status": status,
        })

    if archive is not None:
        # Sequential archive reads are materially faster than opening 28,481 small extracted files
        # on Windows, while hashing the exact immutable source bytes named in the manifest.
        with zipfile.ZipFile(archive) as bundle:
            members = sorted((member for member in bundle.infolist() if not member.is_dir()),
                             key=lambda member: member.filename)
            if len(members) != expected_documents:
                raise ValueError(f"official TechQA corpus has {len(members)} files; expected {expected_documents}")
            for member in members:
                append_record(PurePosixPath(member.filename).as_posix(), bundle.read(member))
    else:
        files = sorted((path for path in extracted_root.rglob("*") if path.is_file()),
                       key=lambda path: path.relative_to(extracted_root).as_posix())
        if len(files) != expected_documents:
            raise ValueError(f"official TechQA corpus has {len(files)} files; expected {expected_documents}")
        for path in files:
            append_record(path.relative_to(extracted_root).as_posix(), path.read_bytes())
    manifest.sort(key=lambda row: str(row["archive_path"]))
    diagnostics = {
        "document_count": len(manifest),
        "duplicate_basename_groups": sum(count > 1 for count in basename_counts.values()),
        "duplicate_content_groups": sum(count > 1 for count in content_counts.values()),
        "documents_with_removed_payloads": sum(int(row["removed_payload_count"]) > 0 for row in manifest),
    }
    return manifest, diagnostics


def validate_context_corpus_alignment(rows: Iterable[TechQAQuestion], corpus_manifest: list[dict[str, Any]],
                                      extracted_root: Path) -> ValidationReport:
    by_filename: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in corpus_manifest:
        by_filename[str(record["filename"])].append(record)
    attached = [context for row in rows for context in row.contexts]
    mismatches: list[str] = []
    ambiguous: list[str] = []
    for context in attached:
        matches = by_filename.get(context.filename, [])
        if len(matches) != 1:
            ambiguous.append(context.filename)
            continue
        corpus_text = (extracted_root / str(matches[0]["archive_path"])).read_text(encoding="utf-8")
        if corpus_text != context.text:
            mismatches.append(context.filename)
    report = ValidationReport()
    report.add("techqa_attached_context_count", "Attached gold contexts", 610, len(attached), len(attached) == 610)
    report.add("techqa_context_unique_resolution", "Every context filename resolves once", 0,
               len(ambiguous), not ambiguous, sorted(set(ambiguous)))
    report.add("techqa_context_exact_match", "Attached texts exactly match official corpus", 0,
               len(mismatches), not mismatches, sorted(set(mismatches)))
    linked = len({context.filename for context in attached})
    report.add("techqa_corpus_link_diagnostic", "Official corpus linked/unlinked counts",
               {"linked": linked, "unlinked": len(corpus_manifest) - linked},
               {"linked": linked, "unlinked": len(corpus_manifest) - linked}, True, severity="diagnostic")
    return report


def corpus_filename_to_doc_id(manifest: list[dict[str, Any]]) -> dict[str, str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in manifest:
        grouped[str(row["filename"])].append(str(row["doc_id"]))
    ambiguous = {name: ids for name, ids in grouped.items() if len(ids) != 1}
    if ambiguous:
        raise ValueError(f"corpus has ambiguous basenames: {sorted(ambiguous)[:5]}")
    return {name: ids[0] for name, ids in grouped.items()}
