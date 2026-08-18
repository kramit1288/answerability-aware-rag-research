"""Content-addressed cache metadata and resumable embedding storage."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from answerability_rag.hashing import canonical_json_sha256


def cache_key(kind: str, inputs: dict[str, Any]) -> str:
    return canonical_json_sha256({"kind": kind, "inputs": inputs})


def validate_cache_metadata(path: Path, *, expected_key: str, expected_kind: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("kind") != expected_kind or metadata.get("cache_key") != expected_key:
        raise ValueError(f"stale/incompatible {expected_kind} cache: {path}")
    return metadata


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Windows can transiently deny replacement while antivirus/indexing or a
    # read-only monitor has the destination open. Retrying the same atomic
    # replace preserves checkpoint ordering and never marks an unflushed batch.
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(min(0.05 * (attempt + 1), 0.5))
