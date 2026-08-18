"""Phase 1 orchestration; all scientific logic remains in focused modules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import Phase01Config
from ..hashing import sha256_file
from ..io import csv_bytes, download_verified, read_csv, safe_extract_zip, utc_now, write_bytes_atomic, write_csv_atomic, write_json_atomic
from ..reproducibility import capture_environment
from .integrity import validate_persisted_phase01
from .manifests import MANUAL_ALIGNMENT_FIELDS, unresolved_manual_alignment_rows
from .ragtruth import RAGTRUTH_MANIFEST_FIELDS, build_ragtruth_manifest, load_ragtruth_responses, load_ragtruth_sources, validate_ragtruth
from .schemas import ValidationReport
from .splits import ASSIGNMENT_FIELDS, COMPONENT_FIELDS, assignment_manifest_rows, assign_grouped_splits, build_question_components, component_manifest_rows, validate_split_assignments
from .techqa import CORPUS_MANIFEST_FIELDS, build_corpus_manifest, corpus_filename_to_doc_id, load_techqa_rows, validate_context_corpus_alignment, validate_techqa_rows


def _merge(target: ValidationReport, source: ValidationReport) -> None:
    target.checks.extend(source.checks)


def prepare_phase01_data(config: Phase01Config) -> dict[str, Any]:
    config.artifact_root.mkdir(parents=True, exist_ok=True)
    metadata_path = config.artifact_root / "dataset_metadata.json"
    existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    loading_timestamp = existing_metadata.get("loading_timestamp") or utc_now()
    existing_assignment_path = config.artifact_root / "techqa_split_assignments.csv"
    persisted_freeze = None
    if existing_assignment_path.exists():
        persisted_rows = read_csv(existing_assignment_path)
        persisted_freeze = persisted_rows[0].get("split_frozen_at") if persisted_rows else None
    split_frozen_at = (
        existing_metadata.get("techqa", {}).get("split", {}).get("split_frozen_at")
        or persisted_freeze or utc_now()
    )

    input_records: dict[str, Any] = {}
    raw_paths: dict[str, Path] = {}
    for dataset_name, dataset in (("techqa", config.techqa), ("ragtruth", config.ragtruth)):
        raw_dir = config.raw_root / dataset_name / dataset.revision
        for filename, source in dataset.files.items():
            path = raw_dir / filename
            record = download_verified(source.url, path, source.sha256)
            raw_paths[f"{dataset_name}/{filename}"] = path
            previous = existing_metadata.get("inputs", {}).get(f"{dataset_name}/{filename}", {})
            input_records[f"{dataset_name}/{filename}"] = {
                "path": path.relative_to(config.source_path.parent.parent).as_posix(),
                "url": source.url, "resolved_url": record.resolved_url,
                "revision": dataset.revision, "bytes": record.bytes, "sha256": record.sha256,
                "retrieved_at": previous.get("retrieved_at") or (loading_timestamp if record.retrieved_at == "cached" else record.retrieved_at),
            }

    corpus_root = config.derived_root / "techqa" / config.techqa.revision / "corpus"
    safe_extract_zip(raw_paths["techqa/corpus.zip"], corpus_root)
    rows = load_techqa_rows(raw_paths["techqa/train.json"])
    validation = ValidationReport()
    _merge(validation, validate_techqa_rows(
        rows, int(config.techqa.values["expected_rows"]), int(config.techqa.values["expected_answerable"]),
        int(config.techqa.values["expected_impossible"]),
    ))
    corpus_manifest, corpus_diagnostics = build_corpus_manifest(
        corpus_root, archive=raw_paths["techqa/corpus.zip"], schema_version=config.dataset_schema_version,
        dataset_revision=config.techqa.revision,
        corpus_archive_sha256=config.techqa.files["corpus.zip"].sha256 or "",
        expected_documents=int(config.techqa.values["expected_corpus_documents"]),
    )
    _merge(validation, validate_context_corpus_alignment(rows, corpus_manifest, corpus_root))
    components = build_question_components(rows)
    split_result = assign_grouped_splits(
        rows, components, ratios=config.split.ratios, seed=config.split.seed,
        algorithm_version=config.split.algorithm_version, tie_rule=config.split.tie_rule,
    )
    _merge(validation, validate_split_assignments(rows, components, split_result))

    sources = load_ragtruth_sources(raw_paths["ragtruth/source_info.jsonl"])
    responses = load_ragtruth_responses(raw_paths["ragtruth/response.jsonl"])
    rag_expected = {
        key: int(config.ragtruth.values[f"expected_{key}"])
        for key in ("sources", "responses", "train_responses", "test_responses", "qa_sources", "qa_responses",
                    "qa_train_sources", "qa_test_sources", "qa_train_responses", "qa_test_responses")
    }
    _merge(validation, validate_ragtruth(sources, responses, rag_expected))
    validation.require_pass()

    component_rows = component_manifest_rows(
        components, split_result, schema_version=config.dataset_schema_version,
        dataset_revision=config.techqa.revision, seed=config.split.seed,
        algorithm_version=config.split.algorithm_version,
    )
    component_path = config.artifact_root / "techqa_split_components.csv"
    component_content = csv_bytes(component_rows, COMPONENT_FIELDS)
    component_sha = write_bytes_atomic(component_path, component_content, immutable=True)
    assignment_rows = assignment_manifest_rows(
        rows, components, split_result, corpus_filename_to_doc_id(corpus_manifest),
        schema_version=config.dataset_schema_version, dataset_revision=config.techqa.revision,
        seed=config.split.seed, algorithm_version=config.split.algorithm_version,
        component_manifest_sha256=component_sha, split_frozen_at=split_frozen_at,
    )
    paths = {
        "techqa_components": component_path,
        "techqa_assignments": config.artifact_root / "techqa_split_assignments.csv",
        "techqa_corpus": config.artifact_root / "techqa_corpus_manifest.csv",
        "ragtruth": config.artifact_root / "ragtruth_manifest.csv",
        "manual_alignments": config.artifact_root / "manual_evidence_alignments.csv",
        "split_config": config.artifact_root / "phase01_split_config.json",
        "split_hash": config.artifact_root / "techqa_split.sha256",
        "integrity_report": config.artifact_root / "phase01_integrity_report.json",
    }
    write_csv_atomic(paths["techqa_assignments"], assignment_rows, ASSIGNMENT_FIELDS, immutable=True)
    write_csv_atomic(paths["techqa_corpus"], corpus_manifest, CORPUS_MANIFEST_FIELDS, immutable=True)
    ragtruth_manifest = build_ragtruth_manifest(
        sources, responses, schema_version=config.dataset_schema_version, dataset_revision=config.ragtruth.revision
    )
    write_csv_atomic(paths["ragtruth"], ragtruth_manifest, RAGTRUTH_MANIFEST_FIELDS, immutable=True)
    write_csv_atomic(paths["manual_alignments"], unresolved_manual_alignment_rows(config.dataset_schema_version, split_frozen_at),
                     MANUAL_ALIGNMENT_FIELDS, immutable=True)
    split_config = {
        "schema_version": config.dataset_schema_version, "seed": config.split.seed,
        "ratios": config.split.ratios, "algorithm_version": config.split.algorithm_version,
        "normalization_version": config.split.normalization_version, "tie_rule": config.split.tie_rule,
        "solver": split_result.diagnostics, "component_manifest_sha256": component_sha,
        "split_semantic_sha256": split_result.semantic_sha256, "split_frozen_at": split_frozen_at,
        "freeze_policy": "immutable; replacement requires an explicit pre-test contract decision",
    }
    write_json_atomic(paths["split_config"], split_config, immutable=True)
    write_bytes_atomic(paths["split_hash"], (split_result.semantic_sha256 + "\n").encode("ascii"), immutable=True)

    # Metadata excludes its own impossible-to-self-embed hash. All other research artifacts are hashed.
    artifact_records = {}
    for name, path in paths.items():
        if name == "integrity_report":
            continue
        artifact_records[name] = {
            "path": path.relative_to(config.source_path.parent.parent).as_posix(),
            "sha256": sha256_file(path),
            "rows": ({"techqa_components": len(component_rows), "techqa_assignments": len(assignment_rows),
                       "techqa_corpus": len(corpus_manifest), "ragtruth": len(ragtruth_manifest),
                       "manual_alignments": 2}.get(name)),
        }
    metadata = {
        "schema_version": config.dataset_schema_version, "phase": 1,
        "loading_timestamp": loading_timestamp, "config_sha256": config.config_sha256,
        "command": "python scripts/prepare_phase01_data.py --config configs/phase01_data.json",
        "environment": capture_environment(), "inputs": input_records,
        "techqa": {
            "repository": config.techqa.repository, "revision": config.techqa.revision,
            "config": config.techqa.values["config"], "exposed_split": config.techqa.values["split"],
            "observed_schema_fields": list(("id", "question", "answer", "is_impossible", "contexts")),
            "observed_rows": len(rows), "observed_answerable": sum(not row.is_impossible for row in rows),
            "observed_impossible": sum(row.is_impossible for row in rows),
            "readme_reported_rows": 908,
            "readme_executable_discrepancy": "README reports 908; pinned executable train.json contains 910 and is operational truth.",
            "corpus": corpus_diagnostics,
            "components": {
                "count": len(components.components),
                "size_distribution": dict(sorted(__import__("collections").Counter(int(row["component_size"]) for row in components.components).items())),
                "largest_size": max(int(row["component_size"]) for row in components.components),
            },
            "split": {**split_result.diagnostics, "split_semantic_sha256": split_result.semantic_sha256,
                      "component_manifest_sha256": component_sha, "split_frozen_at": split_frozen_at},
            "empty_answerable_references": ["DEV_Q014", "DEV_Q094"],
        },
        "ragtruth": {
            "repository": config.ragtruth.repository, "revision": config.ragtruth.revision,
            "observed_source_schema_fields": list(("source_id", "task_type", "source", "source_info", "prompt")),
            "observed_response_schema_fields": list(("id", "source_id", "model", "temperature", "labels", "split", "quality", "response")),
            "observed_label_schema_fields": list(("start", "end", "text", "meta", "label_type", "implicit_true", "due_to_null")),
            "observed_label_meta_nullable": True,
            "observed_sources": len(sources), "observed_responses": len(responses),
            "official_train_responses": sum(response.official_split == "train" for response in responses),
            "official_test_responses": sum(response.official_split == "test" for response in responses),
            "qa_sources": sum(source.task_type == "QA" for source in sources),
            "qa_responses": sum(next(source for source in sources if source.source_id == response.source_id).task_type == "QA" for response in responses),
            "responses_per_source_distribution": {"6": len(sources)},
        },
        "validation": validation.as_dict(), "artifacts": artifact_records,
        "warnings": ["TechQA README count 908 differs from executable pinned row count 910."],
        "deviations_from_contract": [],
    }
    write_json_atomic(metadata_path, metadata, immutable=bool(existing_metadata))
    # Now that metadata exists, validate solely from persisted artifacts and freeze that report.
    persisted = validate_persisted_phase01(config.artifact_root)
    persisted.require_pass()
    write_json_atomic(paths["integrity_report"], persisted.as_dict(), immutable=True)
    return metadata
