"""Typed loading and hashing for the frozen Phase 3 configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from answerability_rag.hashing import canonical_json_sha256


@dataclass(frozen=True)
class Phase03Config:
    source_path: Path
    values: dict[str, Any]
    config_sha256: str

    @classmethod
    def load(cls, path: Path) -> "Phase03Config":
        values = json.loads(path.read_text(encoding="utf-8"))
        required = {
            "expected_inputs", "alignment", "label", "annotation", "serialization",
            "future_classifier_feature_allowlist_from_phase03_labels",
        }
        missing = required - values.keys()
        if missing:
            raise ValueError(f"Phase 3 config missing keys: {sorted(missing)}")
        if values["label"]["depths"] != [1, 3, 5, 10]:
            raise ValueError("Phase 3 depths differ from frozen [1,3,5,10]")
        if values["alignment"]["development_splits"] != ["train", "validation"]:
            raise ValueError("alignment development summaries must contain train/validation only")
        if values["annotation"]["eligible_splits"] != ["train", "validation"]:
            raise ValueError("manual validation material must exclude test")
        if float(values["alignment"]["minimum_development_alignment_coverage"]) != 0.90:
            raise ValueError("alignment feasibility gate must remain 0.90")
        return cls(path.resolve(), values, canonical_json_sha256(values))

    def section_hash(self, name: str) -> str:
        return canonical_json_sha256(self.values[name])
