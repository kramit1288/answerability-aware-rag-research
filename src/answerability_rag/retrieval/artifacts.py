"""Canonical Phase 2 Parquet/JSON artifact serialization and dual hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from answerability_rag.hashing import canonical_json, sha256_file


def semantic_records_sha256(records: Iterable[dict], fields: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for record in records:
        logical = {field: record.get(field) for field in fields}
        digest.update(canonical_json(logical).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_canonical_parquet(
    path: Path, records: list[dict], fields: Sequence[str], sort_key: Sequence[str],
    *, compression_level: int = 9, row_group_size: int = 65536,
) -> dict:
    ordered = sorted(records, key=lambda row: tuple(row[key] for key in sort_key))
    logical = [{field: row.get(field) for field in fields} for row in ordered]
    table = pa.Table.from_pylist(logical)
    table = table.select(list(fields))
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table, path, version="2.6", compression="zstd", compression_level=compression_level,
        use_dictionary=False, write_statistics=True, data_page_version="1.0",
        row_group_size=row_group_size,
    )
    return {
        "path": path.as_posix(), "rows": len(ordered), "columns": list(fields),
        "physical_sha256": sha256_file(path),
        "semantic_sha256": semantic_records_sha256(ordered, fields),
        "bytes": path.stat().st_size,
    }


def read_parquet_records(path: Path) -> list[dict]:
    return pq.read_table(path).to_pylist()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
