from __future__ import annotations

import json
import zipfile

import pytest

from answerability_rag.data.techqa import build_corpus_manifest, load_techqa_rows
from answerability_rag.hashing import sha256_file
from answerability_rag.io import safe_extract_zip


def test_techqa_loader_preserves_valid_schema_and_rejects_coercion(tmp_path) -> None:
    valid = [{
        "id": "TRAIN_Q001", "question": "Question?", "answer": "Answer",
        "is_impossible": False, "contexts": [{"filename": "a.txt", "text": "Evidence"}],
    }]
    path = tmp_path / "tiny.json"
    path.write_text(json.dumps(valid), encoding="utf-8")
    rows = load_techqa_rows(path)
    assert rows[0].question_id == "TRAIN_Q001"
    assert rows[0].contexts[0].text == "Evidence"
    valid[0]["is_impossible"] = 0
    path.write_text(json.dumps(valid), encoding="utf-8")
    with pytest.raises(ValueError, match="label/contexts types"):
        load_techqa_rows(path)


def test_corpus_manifest_retains_nested_duplicate_content_documents(tmp_path) -> None:
    archive = tmp_path / "tiny.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("nested/a.txt", "same text")
        bundle.writestr("other/a.txt", "same text")
    root = tmp_path / "corpus"
    safe_extract_zip(archive, root)
    manifest, diagnostics = build_corpus_manifest(
        root, archive=archive, schema_version="test", dataset_revision="0" * 40,
        corpus_archive_sha256=sha256_file(archive), expected_documents=2,
    )
    assert [row["archive_path"] for row in manifest] == ["nested/a.txt", "other/a.txt"]
    assert len({row["doc_id"] for row in manifest}) == 2
    assert diagnostics["duplicate_basename_groups"] == 1
    assert diagnostics["duplicate_content_groups"] == 1


def test_safe_zip_rejects_path_traversal(tmp_path) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError, match="unsafe ZIP path"):
        safe_extract_zip(archive, tmp_path / "out")
