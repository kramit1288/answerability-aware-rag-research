"""Canonical hashing and the frozen TechQA question normalization."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_json_sha256(value: Any) -> str:
    return sha256_text(canonical_json(value))


def normalize_question_for_grouping(text: str) -> str:
    """Apply the frozen NFKC/case/punctuation/whitespace grouping rule."""
    if not isinstance(text, str):
        raise TypeError("question normalization requires a string")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = "".join(
        " " if unicodedata.category(char).startswith("P") else char for char in normalized
    )
    return re.sub(r"\s+", " ", normalized).strip()


def normalized_question_sha256(text: str) -> str:
    return sha256_text(normalize_question_for_grouping(text))
