"""Integrity checks over persisted Phase 1 research artifacts only."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..hashing import canonical_json_sha256, sha256_file
from ..io import read_csv
from .manifests import MANUAL_ALIGNMENT_FIELDS
from .ragtruth import RAGTRUTH_MANIFEST_FIELDS
from .schemas import ValidationReport
from .splits import ASSIGNMENT_FIELDS, COMPONENT_FIELDS
from .techqa import CORPUS_MANIFEST_FIELDS


def _columns(path: Path) -> tuple[str, ...]:
    import csv
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        return tuple(next(reader))


def validate_persisted_phase01(artifact_root: Path) -> ValidationReport:
    report = ValidationReport()
    paths = {
        "assignments": artifact_root / "techqa_split_assignments.csv",
        "components": artifact_root / "techqa_split_components.csv",
        "corpus": artifact_root / "techqa_corpus_manifest.csv",
        "ragtruth": artifact_root / "ragtruth_manifest.csv",
        "manual": artifact_root / "manual_evidence_alignments.csv",
        "split_config": artifact_root / "phase01_split_config.json",
        "split_hash": artifact_root / "techqa_split.sha256",
        "metadata": artifact_root / "dataset_metadata.json",
    }
    missing = sorted(name for name, path in paths.items() if not path.is_file())
    report.add("persisted_required_artifacts", "All required persisted artifacts exist", [], missing, not missing)
    if missing:
        return report
    expected_columns = {
        "assignments": ASSIGNMENT_FIELDS, "components": COMPONENT_FIELDS,
        "corpus": CORPUS_MANIFEST_FIELDS, "ragtruth": RAGTRUTH_MANIFEST_FIELDS,
        "manual": MANUAL_ALIGNMENT_FIELDS,
    }
    for name, columns in expected_columns.items():
        observed = _columns(paths[name])
        report.add(f"persisted_{name}_schema", f"{name} exact column order", list(columns), list(observed), observed == columns)
    assignments, components = read_csv(paths["assignments"]), read_csv(paths["components"])
    corpus, ragtruth, manual = read_csv(paths["corpus"]), read_csv(paths["ragtruth"]), read_csv(paths["manual"])
    counts = {"assignments": len(assignments), "corpus": len(corpus), "ragtruth": len(ragtruth), "manual": len(manual)}
    expected_counts = {"assignments": 910, "corpus": 28481, "ragtruth": 17790, "manual": 2}
    report.add("persisted_manifest_counts", "Pinned manifest row counts", expected_counts, counts, counts == expected_counts)
    uniqueness = {
        "question_id": len({row["question_id"] for row in assignments}) == len(assignments),
        "split_group_id": len({row["split_group_id"] for row in components}) == len(components),
        "doc_id": len({row["doc_id"] for row in corpus}) == len(corpus),
        "archive_path": len({row["archive_path"] for row in corpus}) == len(corpus),
        "response_id": len({row["response_id"] for row in ragtruth}) == len(ragtruth),
    }
    report.add("persisted_unique_critical_ids", "Manifest critical IDs are unique",
               {key: True for key in uniqueness}, uniqueness, all(uniqueness.values()))
    critical_present = (
        all(row["question_id"] and row["split_group_id"] and row["split"] for row in assignments)
        and all(row["doc_id"] and row["archive_path"] and row["raw_sha256"] for row in corpus)
        and all(row["response_id"] and row["source_id"] and row["official_split"] for row in ragtruth)
    )
    report.add("persisted_critical_values", "No unexpected missing critical values", True, critical_present, critical_present)

    component_split = {row["split_group_id"]: row["assigned_split"] for row in components}
    component_crossing = sorted(group for group in {row["split_group_id"] for row in assignments}
                                if len({row["split"] for row in assignments if row["split_group_id"] == group}) > 1)
    filename_splits: dict[str, set[str]] = defaultdict(set)
    hash_splits: dict[str, set[str]] = defaultdict(set)
    for component in components:
        for filename in json.loads(component["shared_gold_filenames_json"]):
            filename_splits[filename].add(component["assigned_split"])
    for row in assignments:
        hash_splits[row["normalized_question_sha256"]].add(row["split"])
    filename_crossing = sorted(key for key, splits in filename_splits.items() if len(splits) > 1)
    hash_crossing = sorted(key for key, splits in hash_splits.items() if len(splits) > 1)
    report.add("persisted_component_leakage", "No persisted component crosses splits", 0, len(component_crossing), not component_crossing, component_crossing)
    report.add("persisted_filename_leakage", "No persisted gold filename crosses splits", 0, len(filename_crossing), not filename_crossing, filename_crossing)
    report.add("persisted_duplicate_leakage", "No persisted duplicate hash crosses splits", 0, len(hash_crossing), not hash_crossing, hash_crossing)
    assignment_component_match = all(component_split.get(row["split_group_id"]) == row["split"] for row in assignments)
    report.add("persisted_assignment_component_match", "Assignments agree with component splits", True,
               assignment_component_match, assignment_component_match)
    member_ids = [question_id for row in components for question_id in json.loads(row["member_question_ids_json"])]
    assignment_ids = {row["question_id"] for row in assignments}
    member_set = set(member_ids)
    member_differences = {
        "missing": sorted(assignment_ids - member_set), "unexpected": sorted(member_set - assignment_ids),
        "duplicates": len(member_ids) - len(member_set),
    }
    report.add("persisted_component_members", "Component members exactly cover assignments",
               {"members": len(assignments), "missing": [], "unexpected": [], "duplicates": 0},
               {"members": len(member_ids), **member_differences},
               member_set == assignment_ids and len(member_ids) == len(member_set))
    source_splits: dict[str, set[str]] = defaultdict(set)
    for row in ragtruth:
        source_splits[row["source_id"]].add(row["official_split"])
    rag_crossing = sorted(source_id for source_id, splits in source_splits.items() if len(splits) > 1)
    report.add("persisted_ragtruth_source_leakage", "No RAGTruth source crosses official split", 0, len(rag_crossing), not rag_crossing, rag_crossing)
    response_distribution = dict(sorted(Counter(Counter(row["source_id"] for row in ragtruth).values()).items()))
    report.add("persisted_ragtruth_response_distribution", "Six responses per source", {6: 2965},
               response_distribution, response_distribution == {6: 2965})
    manual_status = {row["question_id"]: row["status"] for row in manual}
    report.add("persisted_manual_anomalies", "Both empty-reference rows remain unresolved",
               {"DEV_Q014": "unresolved", "DEV_Q094": "unresolved"}, manual_status,
               manual_status == {"DEV_Q014": "unresolved", "DEV_Q094": "unresolved"})
    no_fabricated_spans = all(not row["manually_aligned_document"] and not row["evidence_start"] and not row["evidence_end"] for row in manual)
    report.add("persisted_no_fabricated_alignment", "Unresolved rows contain no fabricated location", True,
               no_fabricated_spans, no_fabricated_spans)

    semantic_rows = sorted(({
        "question_id": row["question_id"], "split_group_id": row["split_group_id"], "split": row["split"]
    } for row in assignments), key=lambda row: row["question_id"])
    semantic_hash = canonical_json_sha256(semantic_rows)
    persisted_hash = paths["split_hash"].read_text(encoding="ascii").strip()
    report.add("persisted_split_sha256", "Frozen semantic split SHA-256 matches assignments",
               persisted_hash, semantic_hash, persisted_hash == semantic_hash)
    split_config = json.loads(paths["split_config"].read_text(encoding="utf-8"))
    report.add("persisted_split_config_hash", "Split config records semantic hash", semantic_hash,
               split_config.get("split_semantic_sha256"), split_config.get("split_semantic_sha256") == semantic_hash)
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    hash_mismatches: dict[str, Any] = {}
    for name, record in metadata.get("artifacts", {}).items():
        artifact_path = artifact_root.parent.parent / record["path"]
        if not artifact_path.is_file() or sha256_file(artifact_path) != record["sha256"]:
            hash_mismatches[name] = record
    report.add("persisted_artifact_hashes", "Metadata artifact hashes match files", {}, hash_mismatches, not hash_mismatches)
    split_counts = Counter(row["split"] for row in assignments)
    class_counts = {
        split: {
            "answerable": sum(row["is_impossible"] == "false" for row in assignments if row["split"] == split),
            "impossible": sum(row["is_impossible"] == "true" for row in assignments if row["split"] == split),
        } for split in ("train", "validation", "test")
    }
    report.add("persisted_split_count_diagnostic", "Question/class counts by split",
               {"questions": dict(split_counts), "classes": class_counts},
               {"questions": dict(split_counts), "classes": class_counts}, True, severity="diagnostic")
    return report
