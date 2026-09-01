"""Shared Phase 7 serialization, configuration, and immutable-boundary helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from answerability_rag.hashing import canonical_json_sha256, sha256_file
from answerability_rag.io import write_csv_atomic, write_json_atomic


PHASE07_CONFIG_SHA256 = "fa556227bf9a3f60dd6fc2bd5e19e1ed5a1d1a3e9cb72217d355511898ea9206"
RESULTS = Path("artifacts/results")
DERIVED = Path("data/derived/phase07")


@dataclass(frozen=True)
class Phase07Config:
    source_path: Path
    values: dict[str, Any]
    canonical_sha256: str

    @classmethod
    def load(cls, path: Path) -> "Phase07Config":
        values = json.loads(path.read_text(encoding="utf-8"))
        observed = canonical_json_sha256(values)
        if observed != PHASE07_CONFIG_SHA256:
            raise ValueError(f"Phase 7 config differs from pre-TEST freeze: {observed}")
        if values["governance_statement"] != "No post-TEST scientific choice is permitted.":
            raise ValueError("Phase 7 no-post-TEST-choice statement differs")
        if values["test_access"]["post_unseal_tuning_permitted"] is not False:
            raise ValueError("Phase 7 configuration permits post-unseal tuning")
        return cls(path.resolve(), values, observed)


def require_unsealed(root: Path, config: Phase07Config) -> dict[str, Any]:
    freeze_path = root / RESULTS / "phase07_pre_test_governance_freeze.json"
    integrity_path = root / RESULTS / "phase07_upstream_integrity_pre_unseal.json"
    unseal_path = root / RESULTS / "phase07_test_unseal_record.json"
    if not (freeze_path.exists() and integrity_path.exists() and unseal_path.exists()):
        raise PermissionError("Phase 7 TEST access requires governance, integrity, and unseal artifacts")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    unseal = json.loads(unseal_path.read_text(encoding="utf-8"))
    if freeze["phase07_config_canonical_sha256"] != config.canonical_sha256:
        raise ValueError("Phase 7 config differs from governance freeze")
    if integrity["status"] != "pass" or integrity["scientific_hash_mismatch_count"] != 0:
        raise ValueError("Phase 7 pre-unseal upstream integrity did not pass")
    if unseal["phase07_config_canonical_sha256"] != config.canonical_sha256:
        raise ValueError("Phase 7 unseal record config differs")
    return unseal


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> str:
    normalized: list[dict[str, Any]] = []
    for source in rows:
        row: dict[str, Any] = {}
        for field in fields:
            value = source.get(field)
            if isinstance(value, (dict, list, tuple)):
                value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            row[field] = value
        normalized.append(row)
    return write_csv_atomic(path, normalized, fields)


def finite_or_none(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def artifact_record(root: Path, relative: str, rows: int | None = None) -> dict[str, Any]:
    path = root / relative
    result: dict[str, Any] = {
        "path": relative,
        "physical_sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        result["rows"] = int(rows)
    return result


def write_json(path: Path, value: Any, *, immutable: bool = False) -> str:
    return write_json_atomic(path, value, immutable=immutable)
