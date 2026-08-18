"""Typed loading and hashing of the frozen Phase 2 configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from answerability_rag.hashing import canonical_json_sha256


@dataclass(frozen=True)
class Phase02Config:
    source_path: Path
    values: dict[str, Any]
    config_sha256: str

    @classmethod
    def load(cls, path: Path) -> "Phase02Config":
        values = json.loads(path.read_text(encoding="utf-8"))
        required = {"expected_phase01", "chunking", "bm25", "dense", "hybrid",
                    "retrieval_depths", "serialization", "phase02_aggregate_metric_splits"}
        missing = required - values.keys()
        if missing:
            raise ValueError(f"Phase 2 config missing keys: {sorted(missing)}")
        if values["retrieval_depths"] != [1, 3, 5, 10]:
            raise ValueError("retrieval depths differ from frozen [1,3,5,10]")
        if values["phase02_aggregate_metric_splits"] != ["train", "validation"]:
            raise ValueError("development aggregate splits must be train and validation only")
        return cls(path.resolve(), values, canonical_json_sha256(values))

    def section_hash(self, name: str) -> str:
        return canonical_json_sha256(self.values[name])
