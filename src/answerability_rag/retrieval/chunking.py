"""Tokenizer-bounded deterministic TechQA corpus construction."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from transformers import PreTrainedTokenizerFast

from answerability_rag.data.techqa import CLEANING_VERSION, normalize_corpus_text
from answerability_rag.hashing import canonical_json_sha256, sha256_text


CHUNK_FIELDS = (
    "schema_version", "chunk_id", "doc_id", "filename", "archive_path", "raw_source_sha256",
    "normalized_source_sha256", "retrieval_source_transform_version", "chunk_index",
    "token_start", "token_end", "char_start", "char_end", "token_length", "char_length",
    "text", "text_sha256", "tokenizer_name", "tokenizer_revision", "max_content_tokens",
    "overlap_tokens", "step_tokens", "chunking_version", "chunk_config_sha256",
    "corpus_manifest_sha256",
)


@dataclass(frozen=True)
class ChunkSpec:
    tokenizer_name: str
    tokenizer_revision: str
    max_content_tokens: int
    overlap_tokens: int
    step_tokens: int
    version: str

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(self.__dict__)


def chunk_document(
    source_text: str, *, document: dict[str, str], tokenizer: PreTrainedTokenizerFast,
    spec: ChunkSpec, corpus_manifest_sha256: str,
) -> list[dict]:
    """Chunk one canonical normalized source and preserve exact substring offsets."""
    if not tokenizer.is_fast:
        raise ValueError("chunking requires a fast tokenizer with offset mappings")
    encoded = tokenizer(
        source_text, add_special_tokens=False, return_offsets_mapping=True,
        truncation=False, verbose=False,
    )
    offsets = encoded["offset_mapping"]
    if source_text and not offsets:
        raise ValueError(f"non-empty tokenizable document produced no tokens: {document['doc_id']}")
    chunks = []
    start = 0
    chunk_index = 0
    while start < len(offsets):
        end = min(start + spec.max_content_tokens, len(offsets))
        char_start, char_end = int(offsets[start][0]), int(offsets[end - 1][1])
        text = source_text[char_start:char_end]
        provenance = {
            "chunking_version": spec.version,
            "doc_id": document["doc_id"],
            "normalized_source_sha256": document["normalized_sha256"],
            "chunk_index": chunk_index,
            "token_start": start,
            "token_end": end,
            "char_start": char_start,
            "char_end": char_end,
            "chunk_config_sha256": spec.sha256,
        }
        chunks.append({
            "schema_version": "phase02-chunks-v1", "chunk_id": canonical_json_sha256(provenance),
            "doc_id": document["doc_id"], "filename": document["filename"],
            "archive_path": document["archive_path"], "raw_source_sha256": document["raw_sha256"],
            "normalized_source_sha256": document["normalized_sha256"],
            "retrieval_source_transform_version": CLEANING_VERSION, "chunk_index": chunk_index,
            "token_start": start, "token_end": end, "char_start": char_start, "char_end": char_end,
            "token_length": end - start, "char_length": len(text), "text": text,
            "text_sha256": sha256_text(text), "tokenizer_name": spec.tokenizer_name,
            "tokenizer_revision": spec.tokenizer_revision,
            "max_content_tokens": spec.max_content_tokens, "overlap_tokens": spec.overlap_tokens,
            "step_tokens": spec.step_tokens, "chunking_version": spec.version,
            "chunk_config_sha256": spec.sha256, "corpus_manifest_sha256": corpus_manifest_sha256,
        })
        chunk_index += 1
        if end == len(offsets):
            break
        start += spec.step_tokens
    return chunks


def load_corpus_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_chunk_corpus(
    corpus_manifest: Iterable[dict[str, str]], corpus_root: Path,
    tokenizer: PreTrainedTokenizerFast, spec: ChunkSpec, corpus_manifest_sha256: str,
) -> tuple[list[dict], list[str]]:
    chunks: list[dict] = []
    zero_chunk_documents: list[str] = []
    for document in sorted(corpus_manifest, key=lambda row: row["doc_id"]):
        raw_text = (corpus_root / document["archive_path"]).read_text(encoding="utf-8")
        normalized, _ = normalize_corpus_text(raw_text)
        if sha256_text(normalized) != document["normalized_sha256"]:
            raise ValueError(f"normalized source hash drift: {document['doc_id']}")
        rows = chunk_document(
            normalized, document=document, tokenizer=tokenizer, spec=spec,
            corpus_manifest_sha256=corpus_manifest_sha256,
        )
        if not rows:
            zero_chunk_documents.append(document["doc_id"])
        chunks.extend(rows)
    chunks.sort(key=lambda row: row["chunk_id"])
    if len({row["chunk_id"] for row in chunks}) != len(chunks):
        raise ValueError("duplicate deterministic chunk IDs")
    return chunks, zero_chunk_documents
