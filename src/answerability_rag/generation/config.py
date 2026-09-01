"""Frozen Phase 5 configuration and upstream/test-seal guards."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from answerability_rag.hashing import canonical_json_sha256, sha256_file


PHASE05_CONFIG_SHA256 = "b35ca88eccd8a24194a2976a12b31f82cc9ea243856c8c379001a84250972dd3"
FREEZE_PATH = Path("artifacts/results/phase05_pre_results_governance_freeze.json")


def assert_techqa_split_allowed(split: str, operation: str) -> None:
    """Fail before any Phase 5 TechQA TEST row can be read or transformed."""
    normalized = str(split).strip().casefold()
    if normalized == "test":
        raise ValueError(
            f"TechQA TEST is sealed in Phase 5; refusing to {operation} for split={split!r}"
        )
    if normalized != "validation":
        raise ValueError(
            f"Phase 5 primary TechQA generation is VALIDATION-only; got split={split!r}"
        )


@dataclass(frozen=True)
class Phase05Config:
    source_path: Path
    values: dict[str, Any]
    canonical_sha256: str

    @classmethod
    def load(cls, path: Path, root: Path) -> "Phase05Config":
        values = json.loads(path.read_text(encoding="utf-8"))
        observed = canonical_json_sha256(values)
        if observed != PHASE05_CONFIG_SHA256:
            raise ValueError(
                f"Phase 5 configuration differs from the pre-results freeze: {observed}"
            )
        freeze_path = root / FREEZE_PATH
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if freeze["phase05_config_canonical_sha256"] != observed:
            raise ValueError("Phase 5 freeze does not reference the loaded configuration")
        if freeze["techqa_generation_started_before_freeze"]:
            raise ValueError("invalid governance: TechQA generation preceded the freeze")
        for item in freeze["governance_files"]:
            artifact = root / item["path"]
            if artifact.stat().st_size != int(item["bytes"]):
                raise ValueError(f"frozen governance byte count changed: {item['path']}")
            if sha256_file(artifact) != item["physical_sha256"]:
                raise ValueError(f"frozen governance SHA-256 changed: {item['path']}")
        assert_techqa_split_allowed(values["techqa_population"]["split"], "load config")
        if values["techqa_population"]["required_state_count"] != 178:
            raise ValueError("Phase 5 must retain exactly 178 TechQA generation states")
        if values["policy_views"]["G2"]["t_low"] != 0.78 or values["policy_views"]["G2"]["t_high"] != 0.82:
            raise ValueError("frozen primary Phase 4 policy thresholds changed")
        if values["policy_views"]["G3"]["t_low"] != 0.56 or values["policy_views"]["G3"]["t_high"] != 0.72:
            raise ValueError("frozen 20% Phase 4 policy thresholds changed")
        return cls(path.resolve(), values, observed)


def verify_upstream(root: Path, config: Phase05Config) -> dict[str, str]:
    """Reproduce immutable physical and canonical upstream hashes."""
    upstream = config.values["upstream_freeze"]
    physical = {
        "phase03_primary_target_physical_sha256": "artifacts/data/phase03_final_primary_target.parquet",
        "phase03_final_manifest_physical_sha256": "artifacts/results/phase03_final_artifact_manifest.json",
        "phase04_model_binary_physical_sha256": "artifacts/models/phase04_selected_model.joblib",
        "phase04_final_manifest_physical_sha256": "artifacts/results/phase04_artifact_manifest.json",
    }
    observed: dict[str, str] = {}
    for key, relative in physical.items():
        observed[key] = sha256_file(root / relative)
        if observed[key] != upstream[key]:
            raise ValueError(f"immutable upstream hash mismatch for {relative}")
    canonical = {
        "phase03_final_target_config_canonical_sha256": (
            "configs/phase03_final_target.json", "final_target_config_sha256"
        ),
        "phase04_selected_model_canonical_sha256": (
            "configs/phase04_selected_model.json", "selected_model_config_sha256"
        ),
        "phase04_selected_policy_canonical_sha256": (
            "configs/phase04_selected_policy.json", "selected_policy_config_sha256"
        ),
    }
    for key, (relative, embedded) in canonical.items():
        value = json.loads((root / relative).read_text(encoding="utf-8"))
        stored = value.pop(embedded)
        observed[key] = canonical_json_sha256(value)
        if stored != observed[key] or observed[key] != upstream[key]:
            raise ValueError(f"immutable canonical upstream hash mismatch for {relative}")
    for manifest_name in (
        "artifacts/results/phase03_final_artifact_manifest.json",
        "artifacts/results/phase04_artifact_manifest.json",
    ):
        manifest = json.loads((root / manifest_name).read_text(encoding="utf-8"))
        entries = manifest["artifacts"]
        iterable = entries.values() if isinstance(entries, dict) else entries
        for item in iterable:
            if sha256_file(root / item["path"]) != item["physical_sha256"]:
                raise ValueError(f"immutable manifest entry changed: {item['path']}")
    return observed
