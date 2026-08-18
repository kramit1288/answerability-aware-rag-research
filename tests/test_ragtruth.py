from __future__ import annotations

from dataclasses import replace

from answerability_rag.data.ragtruth import build_ragtruth_manifest, validate_ragtruth
from answerability_rag.data.schemas import RAGTruthResponse, RAGTruthSource


def _fixture():
    sources = [
        RAGTruthSource("s1", "QA", "test", {"question": "q1", "passages": "p1"}, {"prompt": "prompt1"}),
        RAGTruthSource("s2", "QA", "test", {"question": "q2", "passages": "p2"}, {"prompt": "prompt2"}),
    ]
    responses = []
    for source_id, split in (("s1", "train"), ("s2", "test")):
        for index in range(6):
            text = "answer"
            label = ({"start": 0, "end": 6, "text": "answer", "meta": None,
                      "label_type": "Evident Baseless Info", "implicit_true": False,
                      "due_to_null": False},) if index == 0 else ()
            responses.append(RAGTruthResponse(
                f"{source_id}-{index}", source_id, "model", 0.0, label, split, "good", text,
                {"id": f"{source_id}-{index}"},
            ))
    expected = {
        "sources": 2, "responses": 12, "train_responses": 6, "test_responses": 6,
        "qa_sources": 2, "qa_responses": 12, "qa_train_sources": 1,
        "qa_test_sources": 1, "qa_train_responses": 6, "qa_test_responses": 6,
    }
    return sources, responses, expected


def test_ragtruth_official_split_and_source_grouping_are_preserved() -> None:
    sources, responses, expected = _fixture()
    report = validate_ragtruth(sources, responses, expected)
    assert report.passed
    manifest = build_ragtruth_manifest(sources, responses, schema_version="test", dataset_revision="0" * 40)
    assert len(manifest) == 12
    assert {row["official_split"] for row in manifest if row["source_id"] == "s1"} == {"train"}


def test_ragtruth_cross_source_split_fails() -> None:
    sources, responses, expected = _fixture()
    responses[-1] = replace(responses[-1], official_split="train")
    report = validate_ragtruth(sources, responses, expected)
    check = next(item for item in report.checks if item.check_id == "ragtruth_source_split_leakage")
    assert check.status == "fail"
    assert check.details == ["s2"]
