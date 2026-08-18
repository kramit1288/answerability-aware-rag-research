from __future__ import annotations

import json
from pathlib import Path

import pytest

from answerability_rag.config import load_phase01_config
from answerability_rag.data.integrity import validate_persisted_phase01
from answerability_rag.hashing import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def test_persisted_manifest_integrity_and_stable_counts() -> None:
    config = load_phase01_config(ROOT / "configs/phase01_data.json")
    report = validate_persisted_phase01(config.artifact_root)
    failures = [check for check in report.checks if check.status == "fail"]
    assert failures == []
    metadata = json.loads((config.artifact_root / "dataset_metadata.json").read_text(encoding="utf-8"))
    assert metadata["techqa"]["observed_rows"] == 910
    assert metadata["techqa"]["corpus"]["document_count"] == 28481
    assert metadata["ragtruth"]["observed_responses"] == 17790
    assert metadata["ragtruth"]["observed_sources"] == 2965


@pytest.mark.network
def test_pinned_raw_release_checksums_match_contract() -> None:
    config = load_phase01_config(ROOT / "configs/phase01_data.json")
    for dataset_name, dataset in (("techqa", config.techqa), ("ragtruth", config.ragtruth)):
        for filename, source in dataset.files.items():
            if source.sha256 is None:
                continue
            path = config.raw_root / dataset_name / dataset.revision / filename
            assert path.is_file(), f"pinned raw input absent: {path}"
            assert sha256_file(path) == source.sha256
