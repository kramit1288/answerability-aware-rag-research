"""Typed Phase 1 configuration loading and validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .hashing import canonical_json_sha256


_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class SourceFileConfig:
    url: str
    sha256: str | None


@dataclass(frozen=True)
class DatasetConfig:
    repository: str
    revision: str
    files: dict[str, SourceFileConfig]
    values: dict[str, Any]


@dataclass(frozen=True)
class SplitConfig:
    seed: int
    ratios: dict[str, float]
    algorithm_version: str
    normalization_version: str
    tie_rule: str


@dataclass(frozen=True)
class Phase01Config:
    schema_version: str
    dataset_schema_version: str
    techqa: DatasetConfig
    ragtruth: DatasetConfig
    split: SplitConfig
    raw_root: Path
    derived_root: Path
    artifact_root: Path
    config_sha256: str
    source_path: Path


def _dataset_config(value: dict[str, Any], name: str) -> DatasetConfig:
    revision = value.get("revision")
    if not isinstance(revision, str) or not _REVISION.fullmatch(revision):
        raise ValueError(f"{name}.revision must be a 40-character lowercase Git commit")
    raw_files = value.get("files")
    if not isinstance(raw_files, dict) or not raw_files:
        raise ValueError(f"{name}.files must be a non-empty object")
    files: dict[str, SourceFileConfig] = {}
    for filename, record in raw_files.items():
        if not isinstance(record, dict) or not isinstance(record.get("url"), str):
            raise ValueError(f"{name}.files.{filename} must contain a URL")
        checksum = record.get("sha256")
        if checksum is not None and (not isinstance(checksum, str) or not _SHA256.fullmatch(checksum)):
            raise ValueError(f"{name}.files.{filename}.sha256 is invalid")
        files[filename] = SourceFileConfig(record["url"], checksum)
    return DatasetConfig(value["repository"], revision, files, dict(value))


def load_phase01_config(path: Path) -> Phase01Config:
    source_path = path.resolve()
    raw = json.loads(source_path.read_text(encoding="utf-8"))
    split = raw["split"]
    ratios = {key: float(value) for key, value in split["ratios"].items()}
    if tuple(ratios) != ("train", "validation", "test") or abs(sum(ratios.values()) - 1.0) > 1e-12:
        raise ValueError("split ratios must be ordered train/validation/test and sum to 1")
    if any(value <= 0 for value in ratios.values()):
        raise ValueError("split ratios must be positive")
    root = source_path.parent.parent
    paths = raw["paths"]
    return Phase01Config(
        schema_version=raw["schema_version"],
        dataset_schema_version=raw["dataset_schema_version"],
        techqa=_dataset_config(raw["techqa"], "techqa"),
        ragtruth=_dataset_config(raw["ragtruth"], "ragtruth"),
        split=SplitConfig(
            seed=int(split["seed"]), ratios=ratios,
            algorithm_version=split["algorithm_version"],
            normalization_version=split["normalization_version"], tie_rule=split["tie_rule"],
        ),
        raw_root=(root / paths["raw_root"]).resolve(),
        derived_root=(root / paths["derived_root"]).resolve(),
        artifact_root=(root / paths["artifact_root"]).resolve(),
        config_sha256=canonical_json_sha256(raw), source_path=source_path,
    )
