"""Pinned RAGTruth loading, official-split validation, and response manifest."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..hashing import canonical_json_sha256, sha256_text
from .schemas import RAGTruthResponse, RAGTruthSource, ValidationReport


SOURCE_FIELDS = ("source_id", "task_type", "source", "source_info", "prompt")
RESPONSE_FIELDS = ("id", "source_id", "model", "temperature", "labels", "split", "quality", "response")
LABEL_FIELDS = ("start", "end", "text", "meta", "label_type", "implicit_true", "due_to_null")
RAGTRUTH_MANIFEST_FIELDS = (
    "schema_version", "dataset_revision", "response_id", "source_id", "official_split",
    "task_type", "source_name", "model", "temperature", "quality", "response_sha256",
    "prompt_sha256", "source_info_sha256", "label_count", "has_unsupported_span",
    "implicit_true_count", "due_to_null_count", "label_types_json", "primary_grounding_eligible",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(value)
    return rows


def load_ragtruth_sources(path: Path) -> list[RAGTruthSource]:
    output: list[RAGTruthSource] = []
    for index, row in enumerate(_read_jsonl(path), start=1):
        if set(row) != set(SOURCE_FIELDS):
            raise ValueError(f"RAGTruth source line {index} fields differ: {sorted(row)}")
        if not all(isinstance(row[key], str) for key in ("source_id", "task_type", "source", "prompt")):
            raise ValueError(f"RAGTruth source line {index} has invalid string fields")
        if row["task_type"] == "QA":
            info = row["source_info"]
            if not isinstance(info, dict) or not isinstance(info.get("question"), str) or not isinstance(info.get("passages"), str):
                raise ValueError(f"RAGTruth QA source {row['source_id']} lacks string question/passages")
        output.append(RAGTruthSource(
            row["source_id"], row["task_type"], row["source"], row["source_info"], row
        ))
    return output


def load_ragtruth_responses(path: Path) -> list[RAGTruthResponse]:
    output: list[RAGTruthResponse] = []
    for index, row in enumerate(_read_jsonl(path), start=1):
        if set(row) != set(RESPONSE_FIELDS):
            raise ValueError(f"RAGTruth response line {index} fields differ: {sorted(row)}")
        if not all(isinstance(row[key], str) for key in ("id", "source_id", "model", "split", "response")):
            raise ValueError(f"RAGTruth response line {index} has invalid string fields")
        if isinstance(row["temperature"], bool) or not isinstance(row["temperature"], (int, float)):
            raise ValueError(f"RAGTruth response {row['id']} has invalid temperature")
        if row["quality"] is not None and not isinstance(row["quality"], str):
            raise ValueError(f"RAGTruth response {row['id']} has invalid quality")
        if not isinstance(row["labels"], list):
            raise ValueError(f"RAGTruth response {row['id']} labels must be a list")
        labels: list[dict[str, Any]] = []
        for label_index, label in enumerate(row["labels"]):
            if not isinstance(label, dict) or set(label) != set(LABEL_FIELDS):
                raise ValueError(f"RAGTruth response {row['id']} label {label_index} fields differ")
            if (isinstance(label["start"], bool) or not isinstance(label["start"], int) or
                    isinstance(label["end"], bool) or not isinstance(label["end"], int)):
                raise ValueError(f"RAGTruth response {row['id']} label offsets must be integers")
            if not all(isinstance(label[key], str) for key in ("text", "label_type")):
                raise ValueError(f"RAGTruth response {row['id']} label text/type fields are invalid")
            if label["meta"] is not None and not isinstance(label["meta"], str):
                raise ValueError(f"RAGTruth response {row['id']} label meta must be string or null")
            if not all(isinstance(label[key], bool) for key in ("implicit_true", "due_to_null")):
                raise ValueError(f"RAGTruth response {row['id']} label Boolean fields are invalid")
            labels.append(label)
        output.append(RAGTruthResponse(
            row["id"], row["source_id"], row["model"], row["temperature"], tuple(labels),
            row["split"], row["quality"], row["response"], row,
        ))
    return output


def validate_ragtruth(sources: list[RAGTruthSource], responses: list[RAGTruthResponse],
                      expected: dict[str, int] | None = None) -> ValidationReport:
    expected = expected or {
        "sources": 2965, "responses": 17790, "train_responses": 15090, "test_responses": 2700,
        "qa_sources": 989, "qa_responses": 5934, "qa_train_sources": 839,
        "qa_test_sources": 150, "qa_train_responses": 5034, "qa_test_responses": 900,
    }
    report = ValidationReport()
    source_ids = [source.source_id for source in sources]
    response_ids = [response.response_id for response in responses]
    report.add("ragtruth_source_count", "Pinned source count", expected["sources"], len(sources), len(sources) == expected["sources"])
    report.add("ragtruth_response_count", "Pinned response count", expected["responses"], len(responses), len(responses) == expected["responses"])
    report.add("ragtruth_unique_source_ids", "Source IDs are unique", len(sources), len(set(source_ids)), len(sources) == len(set(source_ids)))
    report.add("ragtruth_unique_response_ids", "Response IDs are unique", len(responses), len(set(response_ids)), len(responses) == len(set(response_ids)))
    per_source = Counter(response.source_id for response in responses)
    distribution = dict(sorted(Counter(per_source.values()).items()))
    report.add("ragtruth_responses_per_source", "Exactly six responses per source", {6: expected["sources"]}, distribution, distribution == {6: expected["sources"]})
    response_source_ids = set(per_source)
    join_differences = {
        "missing_response_group": sorted(set(source_ids) - response_source_ids),
        "orphan_response_group": sorted(response_source_ids - set(source_ids)),
    }
    report.add("ragtruth_source_join", "All and only source IDs join",
               {"joined_sources": len(sources), "missing_response_group": [], "orphan_response_group": []},
               {"joined_sources": len(response_source_ids), **join_differences},
               set(source_ids) == response_source_ids)
    split_counts = Counter(response.official_split for response in responses)
    expected_split_counts = {"train": expected["train_responses"], "test": expected["test_responses"]}
    report.add("ragtruth_official_split_counts", "Official response split counts", expected_split_counts,
               dict(split_counts), dict(split_counts) == expected_split_counts)
    source_splits: dict[str, set[str]] = defaultdict(set)
    for response in responses:
        source_splits[response.source_id].add(response.official_split)
    crossing = sorted(source_id for source_id, splits in source_splits.items() if len(splits) != 1)
    report.add("ragtruth_source_split_leakage", "No source_id crosses official train/test", 0, len(crossing), not crossing, crossing)
    source_by_id = {source.source_id: source for source in sources}
    qa_source_ids = {source.source_id for source in sources if source.task_type == "QA"}
    qa_responses = [response for response in responses if response.source_id in qa_source_ids]
    qa_source_split_counts = Counter(next(iter(source_splits[source_id])) for source_id in qa_source_ids)
    qa_response_split_counts = Counter(response.official_split for response in qa_responses)
    qa_observed = {
        "qa_sources": len(qa_source_ids), "qa_responses": len(qa_responses),
        "qa_train_sources": qa_source_split_counts["train"], "qa_test_sources": qa_source_split_counts["test"],
        "qa_train_responses": qa_response_split_counts["train"], "qa_test_responses": qa_response_split_counts["test"],
    }
    qa_expected = {key: expected[key] for key in qa_observed}
    report.add("ragtruth_qa_counts", "Pinned QA subset/source split counts", qa_expected, qa_observed, qa_observed == qa_expected)
    offset_mismatches: list[dict[str, Any]] = []
    for response in responses:
        for index, label in enumerate(response.labels):
            start, end = label["start"], label["end"]
            if start < 0 or end < start or end > len(response.response) or response.response[start:end] != label["text"]:
                offset_mismatches.append({"response_id": response.response_id, "label_index": index})
    report.add("ragtruth_label_offsets", "Label offsets/text match response", 0, len(offset_mismatches),
               not offset_mismatches, offset_mismatches[:50])
    task_join_ok = all(source_by_id[response.source_id].task_type for response in responses if response.source_id in source_by_id)
    report.add("ragtruth_task_join", "Every response has source task metadata", True, task_join_ok, task_join_ok)
    return report


def build_ragtruth_manifest(sources: list[RAGTruthSource], responses: list[RAGTruthResponse],
                            *, schema_version: str, dataset_revision: str) -> list[dict[str, Any]]:
    source_by_id = {source.source_id: source for source in sources}
    output: list[dict[str, Any]] = []
    for response in responses:
        source = source_by_id[response.source_id]
        labels = response.labels
        output.append({
            "schema_version": schema_version, "dataset_revision": dataset_revision,
            "response_id": response.response_id, "source_id": response.source_id,
            "official_split": response.official_split, "task_type": source.task_type,
            "source_name": source.source_name, "model": response.model,
            "temperature": response.temperature, "quality": response.quality,
            "response_sha256": sha256_text(response.response),
            "prompt_sha256": sha256_text(str(source.raw["prompt"])),
            "source_info_sha256": canonical_json_sha256(source.source_info),
            "label_count": len(labels), "has_unsupported_span": str(bool(labels)).lower(),
            "implicit_true_count": sum(bool(label["implicit_true"]) for label in labels),
            "due_to_null_count": sum(bool(label["due_to_null"]) for label in labels),
            "label_types_json": json.dumps(sorted({label["label_type"] for label in labels}), ensure_ascii=False, separators=(",", ":")),
            "primary_grounding_eligible": str(source.task_type == "QA" and response.quality == "good").lower(),
        })
    return sorted(output, key=lambda row: (len(str(row["response_id"])), str(row["response_id"])))
