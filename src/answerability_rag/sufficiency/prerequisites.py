"""Fail-loud validation of frozen Phase 1 and Phase 2 dependencies."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from answerability_rag.hashing import sha256_file
from answerability_rag.retrieval.artifacts import read_parquet_records, semantic_records_sha256
from answerability_rag.retrieval.chunking import CHUNK_FIELDS
from answerability_rag.retrieval.config import Phase02Config
from answerability_rag.retrieval.pipeline import CONDITION_FIELDS, RANKED_FIELDS

from .config import Phase03Config


@dataclass(frozen=True)
class FrozenInputs:
    phase01_split_sha256: str
    chunk_config_sha256: str
    retrieval_config_sha256: str
    chunk_semantic_sha256: str
    ranked_hits_semantic_sha256: str
    conditions_semantic_sha256: str
    chunks: list[dict[str, Any]]
    ranked_hits: list[dict[str, Any]]
    conditions: list[dict[str, Any]]
    report: dict[str, Any]


def verify_frozen_inputs(root: Path, config: Phase03Config) -> FrozenInputs:
    expected = config.values["expected_inputs"]
    checks: list[dict[str, Any]] = []

    def require(name: str, observed: Any, expected_value: Any) -> None:
        passed = observed == expected_value
        checks.append({"check_id": name, "observed": observed, "expected": expected_value,
                       "status": "pass" if passed else "fail"})
        if not passed:
            raise ValueError(f"frozen prerequisite {name} differs: {observed!r} != {expected_value!r}")

    split_hash = (root / "artifacts/data/techqa_split.sha256").read_text(encoding="ascii").strip()
    require("phase01_split_sha256", split_hash, expected["phase01_split_sha256"])
    phase02 = Phase02Config.load(root / "configs/phase02_retrieval.json")
    require("phase02_retrieval_config_sha256", phase02.config_sha256,
            expected["phase02_retrieval_config_sha256"])
    chunk_summary = json.loads((root / "artifacts/data/phase02_chunk_summary.json").read_text())
    require("documents", chunk_summary["corpus_documents"], expected["documents"])
    require("chunks", chunk_summary["total_chunks"], expected["chunks"])
    require("phase02_chunk_config_sha256", chunk_summary["chunk_config_sha256"],
            expected["phase02_chunk_config_sha256"])
    hashes = json.loads((root / "artifacts/results/phase02_artifact_hashes.json").read_text())["artifacts"]
    paths_and_fields = {
        "chunks": (root / "artifacts/data/techqa_chunk_manifest.parquet", CHUNK_FIELDS),
        "ranked_hits": (root / "artifacts/results/retrieval_ranked_hits.parquet", RANKED_FIELDS),
        "conditions": (root / "artifacts/results/retrieval_query_level.parquet", CONDITION_FIELDS),
    }
    records: dict[str, list[dict[str, Any]]] = {}
    for name, (path, fields) in paths_and_fields.items():
        observed_physical = sha256_file(path)
        require(f"{name}_physical_sha256", observed_physical, hashes[name]["physical_sha256"])
        records[name] = read_parquet_records(path)
        observed_semantic = semantic_records_sha256(records[name], fields)
        require(f"{name}_semantic_sha256", observed_semantic, hashes[name]["semantic_sha256"])
        require(f"{name}_rows", len(records[name]), hashes[name]["rows"])
    require("chunk_semantic_frozen", hashes["chunks"]["semantic_sha256"],
            expected["phase02_chunk_semantic_sha256"])
    require("ranked_hits_semantic_frozen", hashes["ranked_hits"]["semantic_sha256"],
            expected["phase02_ranked_hits_semantic_sha256"])
    require("conditions_semantic_frozen", hashes["conditions"]["semantic_sha256"],
            expected["phase02_conditions_semantic_sha256"])
    require("ranked_hit_rows_frozen", len(records["ranked_hits"]), expected["ranked_hit_rows"])
    require("condition_rows_frozen", len(records["conditions"]), expected["condition_rows"])
    report = {"schema_version": "phase03-prerequisites-v1", "overall_status": "pass",
              "checks": checks}
    return FrozenInputs(
        split_hash, chunk_summary["chunk_config_sha256"], phase02.config_sha256,
        hashes["chunks"]["semantic_sha256"], hashes["ranked_hits"]["semantic_sha256"],
        hashes["conditions"]["semantic_sha256"], records["chunks"], records["ranked_hits"],
        records["conditions"], report,
    )
