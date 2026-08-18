"""End-to-end Phase 3 evidence alignment, labeling, and annotation-pack pipeline."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from transformers import AutoTokenizer

from answerability_rag.data.techqa import load_techqa_rows
from answerability_rag.hashing import canonical_json_sha256, sha256_file
from answerability_rag.io import read_csv
from answerability_rag.retrieval.artifacts import (
    semantic_records_sha256, write_canonical_parquet, write_json,
)

from .alignment import ALIGNMENT_FIELDS, build_alignment_rows
from .annotation import (
    ANSWER_KEY_FIELDS, BLINDED_FIELDS, SAMPLE_FIELDS, build_annotation_material,
    select_annotation_sample,
)
from .config import Phase03Config
from .feasibility import development_alignment_report
from .governance import build_column_governance
from .integrity import validate_phase03
from .labeling import LABEL_FIELDS, build_label_rows
from .prerequisites import FrozenInputs, verify_frozen_inputs


DEVELOPMENT_SUMMARY_FIELDS = (
    "schema_version", "split", "retrieval_strategy", "k", "condition_rows",
    "positive_labels", "negative_labels", "na_labels", "answerable_wrong_document_negatives",
    "correct_document_incomplete_negatives", "partial_overlap_conditions",
    "benchmark_impossible_preliminary_negatives",
)


def _git_sha(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "HEAD"],
        cwd=root, text=True,
    ).strip()


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Sequence[str]) -> dict[str, Any]:
    ordered = [{field: row.get(field) for field in fields} for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(ordered)
    return {
        "path": path.as_posix(), "rows": len(ordered), "columns": list(fields),
        "physical_sha256": sha256_file(path),
        "semantic_sha256": semantic_records_sha256(ordered, fields), "bytes": path.stat().st_size,
    }


def _json_artifact(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    write_json(path, value)
    return {"path": path.as_posix(), "physical_sha256": sha256_file(path),
            "bytes": path.stat().st_size}


def _load_questions_and_manifests(root: Path) -> tuple[list, dict, dict, Path]:
    phase01 = json.loads((root / "configs/phase01_data.json").read_text(encoding="utf-8"))
    revision = phase01["techqa"]["revision"]
    questions = load_techqa_rows(root / f"data/raw/techqa/{revision}/train.json")
    assignments = {row["question_id"]: row for row in read_csv(
        root / "artifacts/data/techqa_split_assignments.csv"
    )}
    corpus_rows = read_csv(root / "artifacts/data/techqa_corpus_manifest.csv")
    corpus_by_filename = {row["filename"]: row for row in corpus_rows}
    if len(corpus_by_filename) != len(corpus_rows):
        raise ValueError("Phase 3 requires uniquely resolvable official corpus filenames")
    corpus_root = root / f"data/derived/techqa/{revision}/corpus"
    return questions, assignments, corpus_by_filename, corpus_root


def run_alignment_stage(config: Phase03Config, root: Path) -> dict[str, Any]:
    frozen = verify_frozen_inputs(root, config)
    questions, assignments, corpus_by_filename, corpus_root = _load_questions_and_manifests(root)
    align_config = config.values["alignment"]
    tokenizer = AutoTokenizer.from_pretrained(
        align_config["answer_tokenizer_name"],
        revision=align_config["answer_tokenizer_revision"], use_fast=True,
        cache_dir=root / "data/derived/phase02/model_cache",
    )
    rows, alignments = build_alignment_rows(
        questions, assignments, corpus_by_filename, corpus_root, tokenizer,
        config=config.values, phase03_config_sha256=config.config_sha256,
    )
    alignment_artifact = write_canonical_parquet(
        root / "artifacts/data/techqa_evidence_alignments.parquet", rows, ALIGNMENT_FIELDS,
        ("question_id", "alignment_id"),
    )
    feasibility = development_alignment_report(
        rows, minimum_coverage=float(align_config["minimum_development_alignment_coverage"]),
    )
    write_json(root / "artifacts/results/phase03_alignment_feasibility.json", feasibility)
    write_json(root / "artifacts/results/phase03_prerequisite_report.json", frozen.report)
    if feasibility["gate_status"] != "pass":
        request = (
            "# Phase 3 decision request\n\n"
            "Automatic train/validation evidence alignment did not reach the frozen 90% gate.\n\n"
            f"Observed coverage: `{feasibility['alignment_coverage']:.6f}`. "
            f"Unresolved questions: `{feasibility['unresolved_questions']}`.\n\n"
            "No retrieval-conditioned labels were constructed. Scientifically defensible options "
            "are targeted human source-span alignment, explicit exclusion with NA labels, or a "
            "pre-test amendment evaluated on a new development audit. Fuzzy matching, embeddings, "
            "NLI, and LLM-generated spans must not silently create gold evidence.\n"
        )
        (root / "docs/PHASE_03_DECISION_REQUEST.md").write_text(request, encoding="utf-8")
        raise RuntimeError("Phase 3 alignment coverage is below 90%; decision request created")
    return {
        "frozen": frozen, "questions": questions, "assignments": assignments,
        "alignment_rows": rows, "alignments": alignments,
        "alignment_artifact": alignment_artifact, "feasibility": feasibility,
    }


def _development_summary(labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in labels:
        if row["split"] in {"train", "validation"}:
            grouped[(row["split"], row["retrieval_strategy"], int(row["k"]))].append(row)
    output = []
    for (split, strategy, k), rows in sorted(grouped.items()):
        output.append({
            "schema_version": "phase03-development-summary-v1", "split": split,
            "retrieval_strategy": strategy, "k": k, "condition_rows": len(rows),
            "positive_labels": sum(row["y_suff"] == 1 for row in rows),
            "negative_labels": sum(row["y_suff"] == 0 for row in rows),
            "na_labels": sum(row["y_suff"] is None for row in rows),
            "answerable_wrong_document_negatives": sum(
                row["label_method"] == "gold_span_coverage" and row["y_suff"] == 0
                and row["gold_document_hit"] is False for row in rows
            ),
            "correct_document_incomplete_negatives": sum(
                row["label_method"] == "gold_span_coverage" and row["y_suff"] == 0
                and row["gold_document_hit"] is True for row in rows
            ),
            "partial_overlap_conditions": sum(
                row["label_method"] == "gold_span_coverage" and row["partial_overlap"] is True
                for row in rows
            ),
            "benchmark_impossible_preliminary_negatives": sum(
                row["label_method"] == "benchmark_impossible" for row in rows
            ),
        })
    return output


def run_phase03(config: Phase03Config, root: Path) -> dict[str, Any]:
    stage = run_alignment_stage(config, root)
    frozen: FrozenInputs = stage["frozen"]
    questions = stage["questions"]
    question_map = {row.question_id: row for row in questions}
    git_sha = _git_sha(root)
    run_id = "phase03-" + canonical_json_sha256({
        "phase03_config_sha256": config.config_sha256,
        "phase01_split_sha256": frozen.phase01_split_sha256,
        "phase02_conditions_semantic_sha256": frozen.conditions_semantic_sha256,
        "git_commit_sha": git_sha,
    })[:16]
    labels = build_label_rows(
        frozen.conditions, frozen.chunks, question_map, stage["assignments"],
        stage["alignment_rows"], run_id=run_id, config=config.values,
        phase03_config_sha256=config.config_sha256,
    )
    label_artifact = write_canonical_parquet(
        root / "artifacts/data/context_sufficiency_labels.parquet", labels, LABEL_FIELDS,
        ("question_id", "retrieval_strategy", "k"),
    )
    development = _development_summary(labels)
    development_path = root / "artifacts/results/phase03_development_label_summary.csv"
    development_artifact = _write_csv(development_path, development, DEVELOPMENT_SUMMARY_FIELDS)
    governance = build_column_governance()
    governance_path = root / "artifacts/data/phase03_column_governance.json"
    governance_artifact = _json_artifact(governance_path, governance)
    annotation = config.values["annotation"]
    guideline_path = root / "docs/PHASE_03_ANNOTATION_GUIDE.md"
    guideline_text = guideline_path.read_text(encoding="utf-8")
    sample = select_annotation_sample(labels, annotation)
    sample_manifest, blinded, answer_key = build_annotation_material(
        sample, question_map, frozen.conditions, frozen.chunks,
        guideline_version=annotation["version"], guideline_text=guideline_text,
    )
    sample_artifact = _write_csv(
        root / "artifacts/results/phase03_manual_sample_manifest.csv",
        sorted(sample_manifest, key=lambda row: row["sample_id"]), SAMPLE_FIELDS,
    )
    blinded_artifact = write_canonical_parquet(
        root / "artifacts/results/phase03_annotation_blinded.parquet", blinded, BLINDED_FIELDS,
        ("blind_order",),
    )
    template_artifact = _write_csv(
        root / "artifacts/results/phase03_annotation_template.csv",
        sorted(blinded, key=lambda row: row["blind_order"]), BLINDED_FIELDS,
    )
    key_artifact = write_canonical_parquet(
        root / "artifacts/results/phase03_annotation_answer_key.parquet", answer_key,
        ANSWER_KEY_FIELDS, ("sample_id",),
    )
    annotation_status = {
        "schema_version": "phase03-annotation-status-v1", "sample_rows": len(blinded),
        "first_human_annotation": "pending", "second_human_annotator": "pending",
        "raw_agreement": None, "cohens_kappa": None,
        "note": "No human judgement is inferred from Codex, an LLM, NLI, or embeddings.",
    }
    annotation_status_artifact = _json_artifact(
        root / "artifacts/results/phase03_annotation_status.json", annotation_status
    )
    integrity = validate_phase03(
        labels, frozen.conditions, development, sample_manifest, blinded, answer_key, governance,
        expected_sample_size=int(annotation["target_size"]),
    )
    if integrity["overall_status"] != "pass":
        raise ValueError(f"Phase 3 integrity failed: {integrity}")
    integrity_artifact = _json_artifact(
        root / "artifacts/results/phase03_label_integrity_report.json", integrity
    )
    feasibility_artifact = {
        "path": (root / "artifacts/results/phase03_alignment_feasibility.json").as_posix(),
        "physical_sha256": sha256_file(root / "artifacts/results/phase03_alignment_feasibility.json"),
    }
    prerequisites_artifact = {
        "path": (root / "artifacts/results/phase03_prerequisite_report.json").as_posix(),
        "physical_sha256": sha256_file(root / "artifacts/results/phase03_prerequisite_report.json"),
    }
    artifacts = {
        "evidence_alignments": stage["alignment_artifact"], "sufficiency_labels": label_artifact,
        "development_summary": development_artifact, "manual_sample_manifest": sample_artifact,
        "annotation_blinded": blinded_artifact, "annotation_template": template_artifact,
        "annotation_answer_key": key_artifact, "column_governance": governance_artifact,
        "alignment_feasibility": feasibility_artifact, "prerequisite_report": prerequisites_artifact,
        "annotation_status": annotation_status_artifact, "integrity_report": integrity_artifact,
    }
    hash_path = root / "artifacts/results/phase03_artifact_hashes.json"
    prior = json.loads(hash_path.read_text(encoding="utf-8")) if hash_path.exists() else None
    reproducibility = {"prior_completed_run_available": bool(prior)}
    if prior:
        for name in (
            "evidence_alignments", "sufficiency_labels", "manual_sample_manifest",
            "annotation_blinded", "annotation_answer_key",
        ):
            before, after = prior["artifacts"][name], artifacts[name]
            if before["semantic_sha256"] != after["semantic_sha256"]:
                raise ValueError(f"Phase 3 cached rerun changed {name} semantic hash")
            if before["physical_sha256"] != after["physical_sha256"]:
                raise ValueError(f"Phase 3 cached rerun changed {name} physical hash")
        reproducibility.update({"semantic_hashes_equal": True, "physical_hashes_equal": True})
    write_json(hash_path, {"schema_version": "phase03-artifact-hashes-v1",
                           "artifacts": artifacts, "reproducibility": reproducibility})
    manifest = {
        "schema_version": "phase03-label-manifest-v1", "run_id": run_id,
        "git_commit_sha": git_sha, "phase03_config_sha256": config.config_sha256,
        "phase01_split_sha256": frozen.phase01_split_sha256,
        "phase02_chunk_config_sha256": frozen.chunk_config_sha256,
        "phase02_retrieval_config_sha256": frozen.retrieval_config_sha256,
        "phase02_conditions_semantic_sha256": frozen.conditions_semantic_sha256,
        "alignment_rule_version": config.values["alignment"]["version"],
        "label_rule_version": config.values["label"]["version"],
        "annotation_guideline_version": annotation["version"],
        "annotation_guideline_sha256": sha256_file(guideline_path),
        "packages": {name: importlib.metadata.version(name) for name in ("pyarrow", "transformers")},
        "label_rows": len(labels), "manual_sample_rows": len(blinded),
        "alignment_gate": stage["feasibility"],
        "aggregate_label_splits": ["train", "validation"],
        "test_aggregate_label_statistics_calculated": False,
        "human_annotation_status": "pending", "second_annotator_status": "pending",
        "artifacts": artifacts, "reproducibility": reproducibility,
        "phase_boundary": "stopped before Phase 4; no classifier, calibration, policy, or generation logic",
    }
    write_json(root / "artifacts/results/phase03_label_manifest.json", manifest)
    return manifest
