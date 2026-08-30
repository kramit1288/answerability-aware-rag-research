"""Phase 3.6 semantic refinement, development selection, and blinded confirmation pack."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import subprocess
from datetime import datetime, timezone
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import pyarrow.parquet as pq

from answerability_rag.data.techqa import load_techqa_rows
from answerability_rag.hashing import canonical_json, canonical_json_sha256, sha256_file
from answerability_rag.io import read_csv
from answerability_rag.retrieval.artifacts import (
    semantic_records_sha256,
    write_canonical_parquet,
    write_json,
)

from .semantic import (
    CONFIRMATION_STRATA,
    DIAGNOSTIC_STRATA,
    PRIMARY_STRATA,
    Phase03SemanticConfig,
    aggregate_claim_scores,
    aggregate_condition_claims,
    build_context_units,
    confusion,
    confirmation_stratum,
    metrics,
    segment_reference_answer,
    select_confirmation_sample,
    semantic_prediction,
    threshold_grid,
    weighted_metrics,
)
from .semantic_nli import NLIScorer


DEVELOPMENT_CONDITION_FIELDS = (
    "schema_version", "sample_id", "question_id", "split", "retrieval_strategy", "k",
    "sampling_stratum", "manual_label", "evaluation_status", "model_id", "model_revision",
    "claim_count", "minimum_claim_entailment", "mean_claim_entailment",
    "maximum_selected_premise_contradiction", "maximum_any_unit_contradiction",
    "base_semantic_config_sha256",
)
CLAIM_SCORE_FIELDS = (
    "schema_version", "scope", "condition_id", "question_id", "split",
    "retrieval_strategy", "k", "model_id", "model_revision", "claim_index", "claim_text",
    "claim_sha256", "entailment", "neutral", "contradiction", "best_unit_id",
    "best_unit_type", "best_chunk_ids_json", "maximum_unit_contradiction",
    "context_unit_count", "semantic_config_sha256",
)
THRESHOLD_FIELDS = (
    "schema_version", "model_id", "model_revision", "candidate_order",
    "minimum_claim_entailment", "mean_claim_entailment",
    "maximum_selected_premise_contradiction", "precision_gate_minimum",
    "precision_gate_status", "predicted_positive_count", "eligible_count", "tn", "fp", "fn",
    "tp", "sample_precision", "sample_recall", "sample_f1", "weighted_accuracy",
    "weighted_precision", "weighted_recall", "weighted_f1",
    "weighted_confusion_proportions_json", "metrics_by_stratum_json",
)
SEMANTIC_LABEL_FIELDS = (
    "schema_version", "example_id", "question_id", "split", "split_group_id",
    "retrieval_strategy", "k", "primary_population_eligible", "semantic_label_status",
    "semantic_exclusion_reason", "y_suff_strict", "y_suff_semantic",
    "semantic_support_score", "claim_count", "claim_token_lengths_json",
    "maximum_claim_token_length", "model_max_sequence_length",
    "model_pair_special_token_count", "maximum_valid_claim_only_tokens",
    "supported_claim_count",
    "minimum_claim_entailment", "mean_claim_entailment",
    "maximum_selected_premise_contradiction", "maximum_any_unit_contradiction",
    "contradiction_indicator", "semantic_label_method", "semantic_model_id",
    "semantic_model_revision", "semantic_config_sha256",
    "semantic_label_governance_sha256", "strict_semantic_agreement",
    "strict_semantic_disagreement", "strict_semantic_category", "label_provenance",
)
SEMANTIC_UNEVALUABLE_FIELDS = (
    "schema_version", "question_id", "split", "semantic_label_status",
    "semantic_exclusion_reason", "retrieval_condition_count", "claim_count",
    "claim_token_lengths_json", "exceeding_claim_indices_json",
    "maximum_claim_token_length", "model_max_sequence_length",
    "model_pair_special_token_count", "maximum_valid_claim_only_tokens",
    "semantic_model_id", "semantic_model_revision", "semantic_config_sha256",
    "semantic_label_governance_sha256",
)
CONFIRMATION_MANIFEST_FIELDS = (
    "schema_version", "sample_id", "blind_order", "question_id", "split",
    "retrieval_strategy", "k", "confirmation_sampling_stratum", "sample_seed",
    "example_id", "semantic_config_sha256", "semantic_label_governance_sha256",
    "annotation_guideline_version",
    "annotation_guideline_sha256",
)
CONFIRMATION_BLINDED_FIELDS = (
    "schema_version", "sample_id", "blind_order", "question_id", "split",
    "retrieval_strategy", "k", "question", "benchmark_status", "benchmark_answer",
    "retrieved_context_json", "annotation_guideline_version", "annotation_guideline_sha256",
    "annotator_id", "manual_label", "rationale", "annotation_timestamp",
)
CONFIRMATION_KEY_FIELDS = (
    "schema_version", "sample_id", "question_id", "example_id",
    "confirmation_sampling_stratum", "y_suff_strict", "y_suff_semantic",
    "semantic_support_score", "minimum_claim_entailment", "mean_claim_entailment",
    "maximum_selected_premise_contradiction", "semantic_model_id",
    "semantic_model_revision", "semantic_config_sha256",
    "semantic_label_governance_sha256",
)

LABEL_GOVERNANCE_CONFIG = Path("configs/phase03_semantic_label_governance.json")


def _git_sha(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "HEAD"],
        cwd=root, text=True,
    ).strip()


def _write_csv(
    path: Path, rows: list[dict[str, Any]], fields: Sequence[str],
    sort_key: Sequence[str] = (),
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: tuple(row[key] for key in sort_key)) if sort_key else rows
    logical = [{field: row.get(field) for field in fields} for row in ordered]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(logical)
    return {
        "path": path.as_posix(), "rows": len(logical), "columns": list(fields),
        "physical_sha256": sha256_file(path),
        "semantic_sha256": semantic_records_sha256(logical, fields),
        "bytes": path.stat().st_size,
    }


def _json_artifact(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    write_json(path, value)
    return {"path": path.as_posix(), "physical_sha256": sha256_file(path), "bytes": path.stat().st_size}


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    """Durably replace checkpoint metadata without exposing a partial JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _checkpoint_cache_identity(
    *, scope: str, model_info: dict[str, Any], config: Phase03SemanticConfig,
    root: Path, dataset_or_sample_sha256: str, semantic_config_sha256: str,
) -> dict[str, Any]:
    dependency_names = (
        "torch", "transformers", "tokenizers", "safetensors", "huggingface-hub",
    )
    return {
        "scope": scope,
        "model_id": model_info["model_id"],
        "model_revision": model_info["revision"],
        "dataset_or_sample_sha256": dataset_or_sample_sha256,
        "claim_segmentation_sha256": canonical_json_sha256(
            config.values["claim_segmentation"]
        ),
        "context_aggregation_sha256": canonical_json_sha256(
            config.values["context_aggregation"]
        ),
        "semantic_config_sha256": semantic_config_sha256,
        "device": config.values["inference"]["device"],
        "dtype": config.values["inference"]["dtype"],
        "inference_batch_size": int(config.values["inference"]["batch_size"]),
        "checkpoint_pairs": int(config.values["inference"]["raw_score_checkpoint_pairs"]),
        "dependency_versions": {
            name: importlib.metadata.version(name) for name in dependency_names
        },
        "implementation_sha256": canonical_json_sha256({
            "semantic": sha256_file(root / "src/answerability_rag/sufficiency/semantic.py"),
            "semantic_nli": sha256_file(
                root / "src/answerability_rag/sufficiency/semantic_nli.py"
            ),
            "semantic_pipeline": sha256_file(
                root / "src/answerability_rag/sufficiency/semantic_pipeline.py"
            ),
        }),
    }


def _verify_frozen_inputs(config: Phase03SemanticConfig, root: Path) -> None:
    frozen = config.values["frozen_inputs"]
    checks = {
        "phase03_strict_labels_physical_sha256": root / "artifacts/data/context_sufficiency_labels.parquet",
        "original_prototype_sha256": root / "notebooks/original_prototype.ipynb",
    }
    for key, path in checks.items():
        observed = sha256_file(path)
        if observed != frozen[key]:
            raise ValueError(f"frozen input hash mismatch for {path}: {observed}")
    annotation = config.values["development_annotations"]
    observed = sha256_file(root / annotation["path"])
    if observed != annotation["physical_sha256"]:
        raise ValueError("primary human annotation file changed after Phase 3.6 config freeze")


def _load_label_governance(
    selected: dict[str, Any], root: Path,
) -> tuple[dict[str, Any], str, Path]:
    path = root / LABEL_GOVERNANCE_CONFIG
    values = json.loads(path.read_text(encoding="utf-8"))
    scoring = values["semantic_scoring_configuration"]
    if scoring["sha256"] != selected["semantic_config_sha256"]:
        raise ValueError("semantic label governance does not reference the frozen scoring hash")
    policy = values["semantic_unevaluable_policy"]
    if policy != {
        **policy,
        "semantic_label_status": "unevaluable",
        "exclusion_reason": "claim_exceeds_frozen_nli_pair_budget",
        "y_suff_semantic": None,
        "preserve_y_suff_strict": True,
        "claim_truncation": False,
        "post_hoc_claim_segmentation": False,
    }:
        raise ValueError("semantic-unevaluable policy differs from the frozen decision")
    if values["test_seal"] != {
        "calculate_test_semantic_scores": False,
        "calculate_test_semantic_aggregates": False,
    } or values["phase4_started"] is not False:
        raise ValueError("semantic label governance violates the TEST/Phase 4 seal")
    return values, canonical_json_sha256(values), path


def _claim_budget_metadata(scorer: NLIScorer, claims: list[str]) -> dict[str, Any]:
    """Apply the frozen model-tokenizer pair-budget guard without altering claim text."""
    special = int(scorer.tokenizer.num_special_tokens_to_add(pair=True))
    maximum = int(scorer.max_length)
    lengths = [
        len(scorer.tokenizer.encode(claim, add_special_tokens=False)) for claim in claims
    ]
    exceeding = [
        index for index, length in enumerate(lengths, 1)
        if length + special >= maximum
    ]
    return {
        "claim_token_lengths": lengths,
        "exceeding_claim_indices": exceeding,
        "maximum_claim_token_length": max(lengths) if lengths else None,
        "model_max_sequence_length": maximum,
        "model_pair_special_token_count": special,
        "maximum_valid_claim_only_tokens": maximum - special - 1,
        "semantic_evaluable": not exceeding,
    }


def _load_development(config: Phase03SemanticConfig, root: Path) -> list[dict[str, Any]]:
    annotations = read_csv(root / config.values["development_annotations"]["path"])
    blinded = {row["sample_id"]: row for row in read_csv(
        root / "artifacts/results/phase03_annotation_template.csv"
    )}
    keys = {str(row["sample_id"]): row for row in pq.read_table(
        root / "artifacts/results/phase03_annotation_answer_key.parquet"
    ).to_pylist()}
    expected = int(config.values["development_annotations"]["expected_rows"])
    if len(annotations) != expected or len({row["sample_id"] for row in annotations}) != expected:
        raise ValueError("development annotations are incomplete or duplicated")
    output = []
    for row in annotations:
        label = row["manual_label"]
        if label not in {"sufficient", "insufficient", "ambiguous"}:
            raise ValueError(f"invalid frozen manual label: {label!r}")
        if row["sample_id"] not in blinded or row["sample_id"] not in keys:
            raise ValueError(f"development sample identity is missing: {row['sample_id']}")
        blind, key = blinded[row["sample_id"]], keys[row["sample_id"]]
        output.append({
            **row,
            "question": blind["question"],
            "reference_answer": blind["benchmark_answer"],
            "retrieved_context_json": blind["retrieved_context_json"],
            "sampling_stratum": key["sampling_stratum"],
            "automatic_y_suff": key["automatic_y_suff"],
        })
    return output


def _pairs_for_condition(
    condition_id: str, question_id: str, split: str, strategy: str, k: int,
    claims: list[str], chunks: list[dict[str, Any]], model_id: str, revision: str,
) -> tuple[list[dict[str, Any]], dict[int, list[tuple[str, str]]]]:
    pairs: list[dict[str, Any]] = []
    keys_by_claim: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for claim_index, claim in enumerate(claims, 1):
        claim_sha = canonical_json_sha256({"claim": claim})
        for unit in build_context_units(claim, chunks):
            pair_key = (claim_sha, unit["unit_id"])
            keys_by_claim[claim_index].append(pair_key)
            pairs.append({
                "pair_key": "|".join(pair_key), "condition_id": condition_id,
                "question_id": question_id, "split": split, "retrieval_strategy": strategy,
                "k": int(k), "model_id": model_id, "model_revision": revision,
                "claim_index": claim_index, "claim_text": claim, "claim_sha256": claim_sha,
                "unit": unit,
            })
    return pairs, keys_by_claim


def _score_development_candidate(
    model_info: dict[str, Any], development: list[dict[str, Any]],
    config: Phase03SemanticConfig, root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    scorer = NLIScorer(
        model_info["model_id"], model_info["revision"],
        root / config.values["inference"]["cache_directory"],
        int(config.values["context_aggregation"]["max_sequence_length"]),
        int(config.values["inference"]["batch_size"]),
    )
    try:
        all_pairs: list[dict[str, Any]] = []
        condition_specs: dict[str, dict[str, Any]] = {}
        for row in development:
            claims = segment_reference_answer(str(row["reference_answer"] or ""))
            if not claims:
                condition_specs[row["sample_id"]] = {"row": row, "claims": [], "keys": {}}
                continue
            chunks = json.loads(row["retrieved_context_json"])
            pairs, keys = _pairs_for_condition(
                row["sample_id"], row["question_id"], row["split"],
                row["retrieval_strategy"], int(row["k"]), claims, chunks,
                model_info["model_id"], model_info["revision"],
            )
            all_pairs.extend(pairs)
            condition_specs[row["sample_id"]] = {"row": row, "claims": claims, "keys": keys}
        unique_pairs = {row["pair_key"]: row for row in all_pairs}
        scored = _score_pairs_checkpointed(
            scorer, list(unique_pairs.values()),
            root / ".cache/phase03_semantic" / (
                f"development-{config.config_sha256}-{model_info['revision']}.jsonl"
            ),
            int(config.values["inference"]["raw_score_checkpoint_pairs"]),
            _checkpoint_cache_identity(
                scope="development",
                model_info=model_info,
                config=config,
                root=root,
                dataset_or_sample_sha256=canonical_json_sha256({
                    "annotations_physical_sha256": config.values[
                        "development_annotations"
                    ]["physical_sha256"],
                    "original_sample_semantic_sha256": config.values["frozen_inputs"][
                        "phase03_original_sample_semantic_sha256"
                    ],
                }),
                semantic_config_sha256=config.config_sha256,
            ),
        )
        score_map = {
            (row["claim_sha256"], row["unit_id"]): row for row in scored
        }
        claim_rows: list[dict[str, Any]] = []
        condition_rows: list[dict[str, Any]] = []
        for sample_id, spec in condition_specs.items():
            row, claims = spec["row"], spec["claims"]
            if not claims:
                condition_rows.append({
                    "schema_version": "phase03-semantic-development-condition-v1",
                    "sample_id": sample_id, "question_id": row["question_id"],
                    "split": row["split"], "retrieval_strategy": row["retrieval_strategy"],
                    "k": int(row["k"]), "sampling_stratum": row["sampling_stratum"],
                    "manual_label": row["manual_label"],
                    "evaluation_status": "not_evaluable_no_reference_answer",
                    "model_id": model_info["model_id"], "model_revision": model_info["revision"],
                    "claim_count": 0, "base_semantic_config_sha256": config.config_sha256,
                })
                continue
            aggregated_claims = []
            for claim_index, claim in enumerate(claims, 1):
                units = [score_map[key] for key in spec["keys"][claim_index]]
                aggregate = aggregate_claim_scores(units)
                claim_row = {
                    "schema_version": "phase03-semantic-claim-score-v1", "scope": "development",
                    "condition_id": sample_id, "question_id": row["question_id"],
                    "split": row["split"], "retrieval_strategy": row["retrieval_strategy"],
                    "k": int(row["k"]), "model_id": model_info["model_id"],
                    "model_revision": model_info["revision"], "claim_index": claim_index,
                    "claim_text": claim, "claim_sha256": canonical_json_sha256({"claim": claim}),
                    **aggregate, "semantic_config_sha256": config.config_sha256,
                }
                claim_rows.append(claim_row)
                aggregated_claims.append(claim_row)
            condition_rows.append({
                "schema_version": "phase03-semantic-development-condition-v1",
                "sample_id": sample_id, "question_id": row["question_id"],
                "split": row["split"], "retrieval_strategy": row["retrieval_strategy"],
                "k": int(row["k"]), "sampling_stratum": row["sampling_stratum"],
                "manual_label": row["manual_label"], "evaluation_status": "evaluable",
                "model_id": model_info["model_id"], "model_revision": model_info["revision"],
                **aggregate_condition_claims(aggregated_claims),
                "base_semantic_config_sha256": config.config_sha256,
            })
        return condition_rows, claim_rows, scorer.compatibility
    finally:
        scorer.close()


def _score_pairs_checkpointed(
    scorer: NLIScorer, pairs: list[dict[str, Any]], path: Path, checkpoint_pairs: int,
    cache_identity: dict[str, Any],
) -> list[dict[str, Any]]:
    """Persist immutable pair scores with an atomic manifest and exact resume bitmap."""
    expected = {str(row["pair_key"]): row for row in pairs}
    if len(expected) != len(pairs):
        raise ValueError("NLI pair inputs contain duplicate pair keys")
    ordered = sorted(expected.values(), key=lambda row: (
        len(str(row["claim_text"]))
        + sum(len(str(text)) for text in row["unit"]["constituents"]),
        str(row["pair_key"]),
    ))
    ordered_keys = [str(row["pair_key"]) for row in ordered]
    pair_inputs_sha256 = canonical_json_sha256([
        {
            "pair_key": row["pair_key"],
            "condition_id": row["condition_id"],
            "question_id": row["question_id"],
            "split": row["split"],
            "retrieval_strategy": row["retrieval_strategy"],
            "k": int(row["k"]),
            "claim_index": int(row["claim_index"]),
            "claim_text": row["claim_text"],
            "claim_sha256": row["claim_sha256"],
            "unit": row["unit"],
        }
        for row in ordered
    ])
    effective_identity = {
        **cache_identity,
        "expected_pair_count": len(ordered),
        "expected_pair_order_sha256": canonical_json_sha256(ordered_keys),
        "pair_inputs_sha256": pair_inputs_sha256,
    }
    cache_key_sha256 = canonical_json_sha256(effective_identity)
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    existing_manifest = None
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("cache_key_sha256") != cache_key_sha256:
            raise ValueError(
                f"stale NLI checkpoint metadata rejected for {path}: cache key mismatch"
            )
    cached: dict[str, dict[str, Any]] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                key = str(row["pair_key"])
                if row["model_id"] != scorer.model_id or row["model_revision"] != scorer.revision:
                    raise ValueError(f"NLI checkpoint model mismatch at {path}:{line_number}")
                if key in cached and cached[key] != row:
                    raise ValueError(f"conflicting duplicate NLI checkpoint score: {key}")
                cached[key] = row
    unexpected = set(cached) - set(expected)
    if unexpected:
        raise ValueError(f"NLI checkpoint contains {len(unexpected)} unexpected pair keys")
    identity_fields = (
        "condition_id", "question_id", "split", "retrieval_strategy", "k", "model_id",
        "model_revision", "claim_index", "claim_text", "claim_sha256",
    )
    for key, row in cached.items():
        source = expected[key]
        expected_flat = {
            **{field: source[field] for field in identity_fields},
            "pair_key": source["pair_key"],
            "unit_id": source["unit"]["unit_id"],
            "unit_type": source["unit"]["unit_type"],
            "chunk_ids": source["unit"]["chunk_ids"],
        }
        if any(row.get(field) != value for field, value in expected_flat.items()):
            raise ValueError(f"NLI checkpoint input identity mismatch for pair: {key}")

    legacy_bootstrap = bool(cached) and existing_manifest is None

    def write_manifest(*, reconciled: bool) -> None:
        bitmap = "".join("1" if key in cached else "0" for key in ordered_keys)
        completed_batches = [
            index // checkpoint_pairs
            for index in range(0, len(ordered_keys), checkpoint_pairs)
            if all(key in cached for key in ordered_keys[index:index + checkpoint_pairs])
        ]
        _atomic_write_json(manifest_path, {
            "schema_version": "phase03-nli-checkpoint-manifest-v1",
            "checkpoint_path": path.as_posix(),
            "cache_key_sha256": cache_key_sha256,
            "cache_identity": effective_identity,
            "deterministic_order": "estimated_character_length_then_pair_key",
            "checkpoint_pairs": checkpoint_pairs,
            "completed_bitmap": bitmap,
            "completed_batch_indices": completed_batches,
            "completed_pair_count": len(cached),
            "completed_pair_keys_sha256": canonical_json_sha256(
                [key for key in ordered_keys if key in cached]
            ),
            "checkpoint_jsonl_bytes": path.stat().st_size if path.exists() else 0,
            "legacy_jsonl_recovery_bootstrap": legacy_bootstrap,
            "manifest_reconciled_from_jsonl": reconciled,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        })

    if existing_manifest is not None:
        recorded_bitmap = str(existing_manifest.get("completed_bitmap", ""))
        observed_bitmap = "".join("1" if key in cached else "0" for key in ordered_keys)
        reconciled = recorded_bitmap != observed_bitmap
    else:
        reconciled = legacy_bootstrap
    write_manifest(reconciled=reconciled)

    # Length-bucket inference deterministically to reduce CPU padding work. Rendering,
    # truncation, model inputs, probabilities, and pair identities are unchanged.
    remaining = [row for row in ordered if str(row["pair_key"]) not in cached]
    if cached:
        print(f"Resuming {path.name}: {len(cached)}/{len(expected)} pair scores cached", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(remaining), checkpoint_pairs):
        batch = remaining[start:start + checkpoint_pairs]
        scored = scorer.score(batch)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            for row in scored:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        cached.update({str(row["pair_key"]): row for row in scored})
        write_manifest(reconciled=False)
        print(
            f"Checkpointed {len(cached)}/{len(expected)} pair scores for {scorer.model_id}",
            flush=True,
        )
    if set(cached) != set(expected):
        raise ValueError("NLI checkpoint did not reach exact expected pair coverage")
    return [cached[str(row["pair_key"])] for row in pairs]


def _evaluate_thresholds(
    model_info: dict[str, Any], rows: list[dict[str, Any]], config: Phase03SemanticConfig,
) -> list[dict[str, Any]]:
    evaluable = [row for row in rows if row["evaluation_status"] == "evaluable"]
    search = config.values["threshold_search"]
    population_counts = {key: int(value) for key, value in search["primary_population_stratum_counts"].items()}
    output = []
    for thresholds in threshold_grid(config.values):
        predicted = [{**row, "prediction": semantic_prediction(row, thresholds)} for row in evaluable]
        primary = [row for row in predicted if row["sampling_stratum"] in PRIMARY_STRATA]
        cells = confusion(primary)
        sample = metrics(cells)
        weighted = weighted_metrics(primary, population_counts, PRIMARY_STRATA)
        # The frozen safety gate is ordinary precision among the non-ambiguous
        # answerable development judgments. Prevalence weighting is used for the
        # subsequent F1 objective, not to redefine the safety constraint.
        precision = sample["precision"]
        output.append({
            "schema_version": "phase03-semantic-threshold-search-v1",
            "model_id": model_info["model_id"], "model_revision": model_info["revision"],
            "candidate_order": int(model_info["candidate_order"]), **thresholds,
            "precision_gate_minimum": float(search["minimum_precision"]),
            "precision_gate_status": (
                "pass" if precision is not None and precision >= float(search["minimum_precision"])
                else "fail"
            ),
            "predicted_positive_count": sum(row["prediction"] for row in primary),
            "eligible_count": len(primary), **cells,
            "sample_precision": sample["precision"], "sample_recall": sample["recall"],
            "sample_f1": sample["f1"], "weighted_accuracy": weighted["accuracy"],
            "weighted_precision": weighted["precision"], "weighted_recall": weighted["recall"],
            "weighted_f1": weighted["f1"],
            "weighted_confusion_proportions_json": canonical_json(
                weighted["estimated_confusion_proportions"]
            ),
            "metrics_by_stratum_json": canonical_json(weighted["metrics_by_stratum"]),
        })
    return output


def _selection_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -float(row["weighted_f1"]), -float(row["weighted_recall"]),
        -float(row["weighted_precision"]), int(row["candidate_order"]),
        -float(row["minimum_claim_entailment"]), -float(row["mean_claim_entailment"]),
        float(row["maximum_selected_premise_contradiction"]),
    )


def _freeze_selection(
    threshold_rows: list[dict[str, Any]], compatibility: list[dict[str, Any]],
    config: Phase03SemanticConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    passing = [row for row in threshold_rows if row["precision_gate_status"] == "pass"]
    if not passing:
        raise RuntimeError("no predeclared semantic candidate satisfies the 0.90 precision gate")
    selected = min(passing, key=_selection_key)
    frozen_definition = {
        "schema_version": "phase03-semantic-selected-method-v1",
        "base_semantic_config_sha256": config.config_sha256,
        "model_id": selected["model_id"], "model_revision": selected["model_revision"],
        "claim_segmentation": config.values["claim_segmentation"],
        "context_aggregation": config.values["context_aggregation"],
        "thresholds": {
            "minimum_claim_entailment": selected["minimum_claim_entailment"],
            "mean_claim_entailment": selected["mean_claim_entailment"],
            "maximum_selected_premise_contradiction": selected[
                "maximum_selected_premise_contradiction"
            ],
        },
        "label_definition": (
            "sufficient iff every deterministic reference claim meets the minimum entailment "
            "threshold, mean claim entailment meets its threshold, and the largest contradiction "
            "score attached to each claim's selected maximum-entailment premise remains below "
            "the contradiction threshold"
        ),
        "selection_objective": config.values["threshold_search"]["selection_order"],
        "primary_population": config.values["primary_population"],
        "candidate_models": config.values["candidate_models"],
        "tie_breaking": "maximum entailment ties use lexicographically ascending context unit ID",
    }
    semantic_hash = canonical_json_sha256(frozen_definition)
    report = {
        **frozen_definition,
        "semantic_config_sha256": semantic_hash,
        "selected_development_objective": {
            key: selected[key] for key in (
                "weighted_precision", "weighted_recall", "weighted_f1", "weighted_accuracy",
                "sample_precision", "sample_recall", "sample_f1", "tn", "fp", "fn", "tp",
            )
        },
        "compatibility": compatibility,
        "frozen_before_confirmation_sampling": True,
    }
    return report, selected


def run_development_selection(
    config: Phase03SemanticConfig, root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _verify_frozen_inputs(config, root)
    development = _load_development(config, root)
    all_conditions: list[dict[str, Any]] = []
    all_claims: list[dict[str, Any]] = []
    all_thresholds: list[dict[str, Any]] = []
    compatibility: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for model_info in config.values["candidate_models"]:
        print(f"Scoring frozen development candidate: {model_info['model_id']}@{model_info['revision']}", flush=True)
        try:
            conditions, claims, compatible = _score_development_candidate(
                model_info, development, config, root
            )
        except Exception as error:
            failures.append({
                "model_id": model_info["model_id"], "revision": model_info["revision"],
                "status": "incompatible", "error_type": type(error).__name__,
                "error": str(error),
            })
            print(f"Candidate incompatible: {model_info['model_id']}: {type(error).__name__}: {error}", flush=True)
            continue
        compatibility.append(compatible)
        all_conditions.extend(conditions)
        all_claims.extend(claims)
        all_thresholds.extend(_evaluate_thresholds(model_info, conditions, config))
    if not all_thresholds:
        raise RuntimeError(f"all predeclared semantic candidates were incompatible: {failures}")
    selected_report, selected_row = _freeze_selection(all_thresholds, compatibility, config)
    selected_report["incompatible_candidates"] = failures
    artifacts: dict[str, Any] = {}
    artifacts["development_condition_scores"] = write_canonical_parquet(
        root / "artifacts/results/phase03_semantic_development_condition_scores.parquet",
        all_conditions, DEVELOPMENT_CONDITION_FIELDS, ("model_id", "sample_id"),
    )
    artifacts["development_claim_scores"] = write_canonical_parquet(
        root / "artifacts/results/phase03_semantic_development_claim_scores.parquet",
        all_claims, CLAIM_SCORE_FIELDS, ("model_id", "condition_id", "claim_index"),
    )
    artifacts["threshold_search"] = _write_csv(
        root / "artifacts/results/phase03_semantic_threshold_search.csv",
        all_thresholds, THRESHOLD_FIELDS,
        ("candidate_order", "minimum_claim_entailment", "mean_claim_entailment",
         "maximum_selected_premise_contradiction"),
    )
    artifacts["selected_config"] = _json_artifact(
        root / "artifacts/results/phase03_semantic_selected_config.json", selected_report
    )
    return selected_report, selected_row, {
        "artifacts": artifacts, "development": development,
        "conditions": all_conditions, "thresholds": all_thresholds,
    }


def _load_primary_population(
    root: Path,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, int], dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    labels = pq.read_table(
        root / "artifacts/data/context_sufficiency_labels.parquet",
        filters=[("split", "in", ["train", "validation"])],
    ).to_pylist()
    if any(row["split"] == "test" for row in labels):
        raise ValueError("TEST row entered semantic primary-population load")
    eligible = [
        row for row in labels
        if row["benchmark_is_impossible"] is False
        and row["y_suff"] is not None
        and row["label_method"] == "gold_span_coverage"
        and str(row["gold_alignment_status"]).startswith("aligned_")
    ]
    conditions = pq.read_table(
        root / "artifacts/results/retrieval_query_level.parquet",
        filters=[("split", "in", ["train", "validation"])],
    ).to_pylist()
    condition_map = {
        (row["question_id"], row["retrieval_strategy"], int(row["k"])): row
        for row in conditions
    }
    chunk_table = pq.read_table(
        root / "artifacts/data/techqa_chunk_manifest.parquet",
        columns=["chunk_id", "doc_id", "filename", "char_start", "char_end", "text"],
    )
    chunks = {str(row["chunk_id"]): row for row in chunk_table.to_pylist()}
    phase01 = json.loads((root / "configs/phase01_data.json").read_text(encoding="utf-8"))
    revision = phase01["techqa"]["revision"]
    questions = {
        row.question_id: row for row in load_techqa_rows(
            root / f"data/raw/techqa/{revision}/train.json"
        )
    }
    return eligible, condition_map, chunks, questions


def _chunks_for_condition(condition: dict[str, Any], chunks: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for rank, chunk_id in enumerate(json.loads(condition["ordered_chunk_ids_json"]), 1):
        output.append({**chunks[chunk_id], "rank": rank})
    return output


def apply_selected_semantic_method(
    selected: dict[str, Any], config: Phase03SemanticConfig,
    label_governance_sha256: str, root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    eligible, condition_map, chunks, questions = _load_primary_population(root)
    model_info = {"model_id": selected["model_id"], "revision": selected["model_revision"]}
    scorer = NLIScorer(
        model_info["model_id"], model_info["revision"],
        root / config.values["inference"]["cache_directory"],
        int(config.values["context_aggregation"]["max_sequence_length"]),
        int(config.values["inference"]["batch_size"]),
    )
    try:
        pair_map: dict[tuple[str, str], dict[str, Any]] = {}
        specs: dict[str, dict[str, Any]] = {}
        question_claims: dict[str, tuple[list[str], dict[str, Any]]] = {}
        unevaluable_questions: dict[str, dict[str, Any]] = {}
        for index, label in enumerate(eligible, 1):
            question = questions[label["question_id"]]
            if label["question_id"] not in question_claims:
                claims = segment_reference_answer(question.answer)
                question_claims[label["question_id"]] = (
                    claims, _claim_budget_metadata(scorer, claims)
                )
            claims, budget = question_claims[label["question_id"]]
            if not claims:
                raise ValueError(f"eligible answerable row has no semantic claims: {question.question_id}")
            if not budget["semantic_evaluable"]:
                specs[label["example_id"]] = {
                    "label": label, "claims": claims, "keys": {}, "budget": budget,
                    "semantic_label_status": "unevaluable",
                }
                record = unevaluable_questions.setdefault(label["question_id"], {
                    "schema_version": "phase03-semantic-unevaluable-question-v1",
                    "question_id": label["question_id"], "split": label["split"],
                    "semantic_label_status": "unevaluable",
                    "semantic_exclusion_reason": "claim_exceeds_frozen_nli_pair_budget",
                    "retrieval_condition_count": 0, "claim_count": len(claims),
                    "claim_token_lengths_json": canonical_json(
                        budget["claim_token_lengths"]
                    ),
                    "exceeding_claim_indices_json": canonical_json(
                        budget["exceeding_claim_indices"]
                    ),
                    "maximum_claim_token_length": budget["maximum_claim_token_length"],
                    "model_max_sequence_length": budget["model_max_sequence_length"],
                    "model_pair_special_token_count": budget[
                        "model_pair_special_token_count"
                    ],
                    "maximum_valid_claim_only_tokens": budget[
                        "maximum_valid_claim_only_tokens"
                    ],
                    "semantic_model_id": selected["model_id"],
                    "semantic_model_revision": selected["model_revision"],
                    "semantic_config_sha256": selected["semantic_config_sha256"],
                    "semantic_label_governance_sha256": label_governance_sha256,
                })
                record["retrieval_condition_count"] += 1
                continue
            condition = condition_map[
                (label["question_id"], label["retrieval_strategy"], int(label["k"]))
            ]
            context = _chunks_for_condition(condition, chunks)
            pairs, keys = _pairs_for_condition(
                label["example_id"], label["question_id"], label["split"],
                label["retrieval_strategy"], int(label["k"]), claims, context,
                model_info["model_id"], model_info["revision"],
            )
            for pair in pairs:
                key = (pair["claim_sha256"], pair["unit"]["unit_id"])
                pair_map.setdefault(key, pair)
            specs[label["example_id"]] = {
                "label": label, "claims": claims, "keys": keys, "budget": budget,
                "semantic_label_status": "evaluable",
            }
            if index % 1000 == 0:
                print(f"Prepared {index}/{len(eligible)} eligible conditions; {len(pair_map)} unique NLI pairs", flush=True)
        print(f"Scoring {len(pair_map)} unique selected-model NLI pairs", flush=True)
        scored = _score_pairs_checkpointed(
            scorer, list(pair_map.values()),
            root / ".cache/phase03_semantic" / f"primary-{selected['semantic_config_sha256']}.jsonl",
            int(config.values["inference"]["raw_score_checkpoint_pairs"]),
            _checkpoint_cache_identity(
                scope="primary_population",
                model_info=model_info,
                config=config,
                root=root,
                dataset_or_sample_sha256=canonical_json_sha256({
                    "phase02_conditions_semantic_sha256": config.values["frozen_inputs"][
                        "phase02_conditions_semantic_sha256"
                    ],
                    "phase03_strict_labels_semantic_sha256": config.values["frozen_inputs"][
                        "phase03_strict_labels_semantic_sha256"
                    ],
                    "population_definition": config.values["primary_population"],
                    "semantic_unevaluable_policy_sha256": label_governance_sha256,
                }),
                semantic_config_sha256=selected["semantic_config_sha256"],
            ),
        )
        score_map = {(row["claim_sha256"], row["unit_id"]): row for row in scored}
        claim_rows: list[dict[str, Any]] = []
        semantic_rows: list[dict[str, Any]] = []
        thresholds = selected["thresholds"]
        for example_id, spec in specs.items():
            label, claims, budget = spec["label"], spec["claims"], spec["budget"]
            strict = int(label["y_suff"])
            common = {
                "schema_version": "phase03-semantic-labels-v2", "example_id": example_id,
                "question_id": label["question_id"], "split": label["split"],
                "split_group_id": label["split_group_id"],
                "retrieval_strategy": label["retrieval_strategy"], "k": int(label["k"]),
                "y_suff_strict": strict, "claim_count": len(claims),
                "claim_token_lengths_json": canonical_json(budget["claim_token_lengths"]),
                "maximum_claim_token_length": budget["maximum_claim_token_length"],
                "model_max_sequence_length": budget["model_max_sequence_length"],
                "model_pair_special_token_count": budget["model_pair_special_token_count"],
                "maximum_valid_claim_only_tokens": budget[
                    "maximum_valid_claim_only_tokens"
                ],
                "semantic_model_id": selected["model_id"],
                "semantic_model_revision": selected["model_revision"],
                "semantic_config_sha256": selected["semantic_config_sha256"],
                "semantic_label_governance_sha256": label_governance_sha256,
            }
            if spec["semantic_label_status"] == "unevaluable":
                semantic_rows.append({
                    **common, "primary_population_eligible": False,
                    "semantic_label_status": "unevaluable",
                    "semantic_exclusion_reason": "claim_exceeds_frozen_nli_pair_budget",
                    "y_suff_semantic": None, "semantic_support_score": None,
                    "supported_claim_count": None, "minimum_claim_entailment": None,
                    "mean_claim_entailment": None,
                    "maximum_selected_premise_contradiction": None,
                    "maximum_any_unit_contradiction": None,
                    "contradiction_indicator": None,
                    "semantic_label_method": "not_applied_semantic_unevaluable",
                    "strict_semantic_agreement": None,
                    "strict_semantic_disagreement": None,
                    "strict_semantic_category": "semantic_unevaluable",
                    "label_provenance": canonical_json({
                        "strict_example_id": example_id,
                        "strict_conceptual_role": "strict_span_sufficiency",
                        "historical_source_label_method": "gold_span_coverage",
                        "semantic_method": "not_applied_semantic_unevaluable",
                        "semantic_exclusion_reason":
                            "claim_exceeds_frozen_nli_pair_budget",
                        "semantic_config_sha256": selected["semantic_config_sha256"],
                        "semantic_label_governance_sha256": label_governance_sha256,
                    }),
                })
                continue
            aggregated_claims = []
            for claim_index, claim in enumerate(claims, 1):
                units = [score_map[key] for key in spec["keys"][claim_index]]
                aggregate = aggregate_claim_scores(units)
                claim_row = {
                    "schema_version": "phase03-semantic-claim-score-v1", "scope": "primary_population",
                    "condition_id": example_id, "question_id": label["question_id"],
                    "split": label["split"], "retrieval_strategy": label["retrieval_strategy"],
                    "k": int(label["k"]), "model_id": selected["model_id"],
                    "model_revision": selected["model_revision"], "claim_index": claim_index,
                    "claim_text": claim, "claim_sha256": canonical_json_sha256({"claim": claim}),
                    **aggregate, "semantic_config_sha256": selected["semantic_config_sha256"],
                }
                claim_rows.append(claim_row)
                aggregated_claims.append(claim_row)
            aggregate = aggregate_condition_claims(aggregated_claims)
            semantic_label = semantic_prediction(aggregate, thresholds)
            supported = sum(
                float(row["entailment"]) >= float(thresholds["minimum_claim_entailment"])
                and float(row["contradiction"])
                < float(thresholds["maximum_selected_premise_contradiction"])
                for row in aggregated_claims
            )
            category = confirmation_stratum(strict, semantic_label)
            semantic_rows.append({
                **common, "primary_population_eligible": True,
                "semantic_label_status": "evaluable", "semantic_exclusion_reason": None,
                "y_suff_semantic": semantic_label,
                "semantic_support_score": aggregate["minimum_claim_entailment"],
                "supported_claim_count": supported,
                **{key: aggregate[key] for key in (
                    "minimum_claim_entailment", "mean_claim_entailment",
                    "maximum_selected_premise_contradiction", "maximum_any_unit_contradiction",
                )},
                "contradiction_indicator": (
                    aggregate["maximum_selected_premise_contradiction"]
                    >= float(thresholds["maximum_selected_premise_contradiction"])
                ),
                "semantic_label_method": "local_nli_reference_claim_support",
                "strict_semantic_agreement": strict == semantic_label,
                "strict_semantic_disagreement": strict != semantic_label,
                "strict_semantic_category": category,
                "label_provenance": canonical_json({
                    "strict_example_id": example_id,
                    "strict_conceptual_role": "strict_span_sufficiency",
                    "historical_source_label_method": "gold_span_coverage",
                    "semantic_method": "local_nli_reference_claim_support",
                    "semantic_config_sha256": selected["semantic_config_sha256"],
                    "semantic_label_governance_sha256": label_governance_sha256,
                }),
            })
        evaluable_rows = [
            row for row in semantic_rows if row["semantic_label_status"] == "evaluable"
        ]
        counts = {
            "condition_counts_by_split": dict(Counter(row["split"] for row in semantic_rows)),
            "evaluable_condition_counts_by_split": dict(Counter(
                row["split"] for row in evaluable_rows
            )),
            "question_counts_by_split": {
                split: len({row["question_id"] for row in semantic_rows if row["split"] == split})
                for split in ("train", "validation")
            },
            "condition_rows": len(semantic_rows),
            "question_rows": len({row["question_id"] for row in semantic_rows}),
            "semantic_evaluable_condition_rows": len(evaluable_rows),
            "semantic_evaluable_question_rows": len({
                row["question_id"] for row in evaluable_rows
            }),
            "semantic_unevaluable_condition_rows": len(semantic_rows) - len(evaluable_rows),
            "semantic_unevaluable_question_rows": len(unevaluable_questions),
            "semantic_unevaluable_exclusion_reasons": dict(Counter(
                row["semantic_exclusion_reason"] for row in semantic_rows
                if row["semantic_label_status"] == "unevaluable"
            )),
        }
        return semantic_rows, claim_rows, counts, {
            "condition_map": condition_map, "chunks": chunks, "questions": questions,
            "compatibility": scorer.compatibility,
            "unevaluable_questions": list(unevaluable_questions.values()),
        }
    finally:
        scorer.close()


def _development_selected_report(
    selected: dict[str, Any], development_rows: list[dict[str, Any]],
    config: Phase03SemanticConfig,
) -> dict[str, Any]:
    selected_rows = [
        row for row in development_rows
        if row["model_id"] == selected["model_id"] and row["evaluation_status"] == "evaluable"
    ]
    predicted = [
        {**row, "prediction": semantic_prediction(row, selected["thresholds"])}
        for row in selected_rows
    ]
    population = config.values["threshold_search"]["primary_population_stratum_counts"]
    primary = [row for row in predicted if row["sampling_stratum"] in PRIMARY_STRATA]
    primary_weighted = weighted_metrics(primary, population, PRIMARY_STRATA)
    primary_cells = confusion(primary)
    primary_sample = metrics(primary_cells)
    diagnostics = []
    for stratum in DIAGNOSTIC_STRATA:
        rows = [row for row in predicted if row["sampling_stratum"] == stratum]
        if stratum == "benchmark_impossible":
            diagnostics.append({
                "sampling_stratum": stratum, "status": "not_evaluable_no_reference_answer",
                "human_audit_count": 30, "human_sufficient_count": 8,
                "human_sufficient_rate": 8 / 30,
            })
        else:
            cells = confusion(rows)
            diagnostics.append({
                "sampling_stratum": stratum, "status": "available", "count": len(rows),
                "confusion_matrix": cells, **metrics(cells),
            })
    previous_false_negatives = [
        row for row in predicted
        if row["sampling_stratum"] != "automatic_positive"
        and row["manual_label"] == "sufficient"
    ]
    return {
        "answerable_only_primary": {
            "count": len(primary), "confusion_matrix": primary_cells,
            "sample_metrics": primary_sample, "prevalence_weighted_metrics": primary_weighted,
        },
        "all_original_strata_diagnostic": diagnostics,
        "previous_strict_false_negatives_answerable": {
            "count": len(previous_false_negatives),
            "recovered_semantic_positive": sum(row["prediction"] for row in previous_false_negatives),
            "retained_semantic_negative": sum(not row["prediction"] for row in previous_false_negatives),
            "by_stratum": dict(Counter(
                row["sampling_stratum"] for row in previous_false_negatives if row["prediction"]
            )),
        },
    }


def _build_confirmation_pack(
    semantic_rows: list[dict[str, Any]], selected: dict[str, Any], config: Phase03SemanticConfig,
    label_governance_sha256: str, root: Path, resources: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    original = read_csv(root / "artifacts/results/phase03_manual_sample_manifest.csv")
    original_questions = {row["question_id"] for row in original}
    sample = select_confirmation_sample(semantic_rows, original_questions, config.values)
    guideline_path = root / "docs/PHASE_03_ANNOTATION_GUIDE.md"
    guideline_sha = sha256_file(guideline_path)
    sampling = config.values["confirmation_sampling"]
    manifests, blinded, keys = [], [], []
    condition_map, chunks, questions = (
        resources["condition_map"], resources["chunks"], resources["questions"]
    )
    for blind_order, row in enumerate(sample, 1):
        condition = condition_map[(row["question_id"], row["retrieval_strategy"], int(row["k"]))]
        context = []
        for chunk in _chunks_for_condition(condition, chunks):
            context.append({
                "rank": int(chunk["rank"]), "chunk_id": chunk["chunk_id"],
                "document_id": chunk["doc_id"], "filename": chunk["filename"],
                "char_start": int(chunk["char_start"]), "char_end": int(chunk["char_end"]),
                "text": chunk["text"],
            })
        sample_id = canonical_json_sha256({
            "example_id": row["example_id"], "sample_seed": int(sampling["seed"]),
            "sampling_version": sampling["version"],
            "semantic_config_sha256": selected["semantic_config_sha256"],
            "semantic_label_governance_sha256": label_governance_sha256,
        })
        manifests.append({
            "schema_version": "phase03-semantic-confirmation-sample-v1", "sample_id": sample_id,
            "blind_order": blind_order, "question_id": row["question_id"],
            "split": row["split"], "retrieval_strategy": row["retrieval_strategy"],
            "k": int(row["k"]),
            "confirmation_sampling_stratum": row["confirmation_sampling_stratum"],
            "sample_seed": int(sampling["seed"]), "example_id": row["example_id"],
            "semantic_config_sha256": selected["semantic_config_sha256"],
            "semantic_label_governance_sha256": label_governance_sha256,
            "annotation_guideline_version": sampling["annotation_guideline_version"],
            "annotation_guideline_sha256": guideline_sha,
        })
        question = questions[row["question_id"]]
        blinded.append({
            "schema_version": "phase03-semantic-confirmation-blinded-v1", "sample_id": sample_id,
            "blind_order": blind_order, "question_id": row["question_id"],
            "split": row["split"], "retrieval_strategy": row["retrieval_strategy"],
            "k": int(row["k"]), "question": question.question,
            "benchmark_status": "answerable_with_reference", "benchmark_answer": question.answer,
            "retrieved_context_json": canonical_json(context),
            "annotation_guideline_version": sampling["annotation_guideline_version"],
            "annotation_guideline_sha256": guideline_sha, "annotator_id": "",
            "manual_label": "", "rationale": "", "annotation_timestamp": "",
        })
        keys.append({
            "schema_version": "phase03-semantic-confirmation-key-v1", "sample_id": sample_id,
            "question_id": row["question_id"], "example_id": row["example_id"],
            "confirmation_sampling_stratum": row["confirmation_sampling_stratum"],
            "y_suff_strict": row["y_suff_strict"], "y_suff_semantic": row["y_suff_semantic"],
            "semantic_support_score": row["semantic_support_score"],
            "minimum_claim_entailment": row["minimum_claim_entailment"],
            "mean_claim_entailment": row["mean_claim_entailment"],
            "maximum_selected_premise_contradiction": row[
                "maximum_selected_premise_contradiction"
            ],
            "semantic_model_id": selected["model_id"],
            "semantic_model_revision": selected["model_revision"],
            "semantic_config_sha256": selected["semantic_config_sha256"],
            "semantic_label_governance_sha256": label_governance_sha256,
        })
    original_conditions = {
        (row["question_id"], row["retrieval_strategy"], int(row["k"])) for row in original
    }
    new_conditions = {
        (row["question_id"], row["retrieval_strategy"], int(row["k"])) for row in manifests
    }
    if original_conditions & new_conditions:
        raise ValueError("semantic confirmation sample overlaps the original 150 conditions")
    if {row["question_id"] for row in manifests} & original_questions:
        raise ValueError("semantic confirmation sample violates stronger question-disjoint freeze")
    forbidden = {
        "y_suff_strict", "y_suff_semantic", "semantic_support_score", "automatic_y_suff",
        "entailment", "contradiction", "sampling_stratum", "confirmation_sampling_stratum",
    }
    if forbidden & set(blinded[0]):
        raise ValueError("confirmation blinded artifact exposes answer-key fields")
    artifacts = {
        "confirmation_manifest": _write_csv(
            root / "artifacts/results/phase03_semantic_confirmation_sample_manifest.csv",
            manifests, CONFIRMATION_MANIFEST_FIELDS, ("sample_id",),
        ),
        "confirmation_blinded": write_canonical_parquet(
            root / "artifacts/results/phase03_semantic_confirmation_blinded.parquet",
            blinded, CONFIRMATION_BLINDED_FIELDS, ("blind_order",),
        ),
        "confirmation_template": _write_csv(
            root / "artifacts/results/phase03_semantic_confirmation_template.csv",
            blinded, CONFIRMATION_BLINDED_FIELDS, ("blind_order",),
        ),
        "confirmation_answer_key": write_canonical_parquet(
            root / "artifacts/results/phase03_semantic_confirmation_answer_key.parquet",
            keys, CONFIRMATION_KEY_FIELDS, ("sample_id",),
        ),
    }
    population_strata = Counter(
        row["strict_semantic_category"] for row in semantic_rows
        if row["semantic_label_status"] == "evaluable"
    )
    sample_strata = Counter(row["confirmation_sampling_stratum"] for row in manifests)
    status = {
        "schema_version": "phase03-semantic-confirmation-status-v1",
        "sample_rows": len(manifests), "unique_questions": len({row["question_id"] for row in manifests}),
        "overlap_with_original_conditions": 0, "overlap_with_original_questions": 0,
        "sample_strata": dict(sample_strata), "primary_population_strata": dict(population_strata),
        "strategy_counts": dict(Counter(row["retrieval_strategy"] for row in manifests)),
        "depth_counts": {str(key): value for key, value in Counter(int(row["k"]) for row in manifests).items()},
        "split_counts": dict(Counter(row["split"] for row in manifests)),
        "primary_human_annotation": "pending", "second_human_annotation": "desirable_not_required",
        "automatic_sufficient_precision_gate": 0.90,
        "prevalence_weighted_f1_gate": 0.85,
        "answer_key_evaluation_status": "sealed_until_complete_genuine_human_confirmation",
        "semantic_unevaluable_rows_in_sample": 0,
        "semantic_label_governance_sha256": label_governance_sha256,
        "test_rows": 0,
    }
    artifacts["confirmation_status"] = _json_artifact(
        root / "artifacts/results/phase03_semantic_confirmation_status.json", status
    )
    return {"artifacts": artifacts, "status": status}, manifests


def _governance(label_governance_sha256: str) -> dict[str, Any]:
    classes = {}
    label_only = {
        "y_suff_strict", "y_suff_semantic", "semantic_support_score", "claim_count",
        "supported_claim_count", "minimum_claim_entailment", "mean_claim_entailment",
        "maximum_selected_premise_contradiction", "maximum_any_unit_contradiction",
        "contradiction_indicator", "semantic_label_method", "strict_semantic_agreement",
        "strict_semantic_disagreement", "strict_semantic_category", "label_provenance",
        "semantic_label_status", "semantic_exclusion_reason", "claim_token_lengths_json",
        "maximum_claim_token_length", "model_max_sequence_length",
        "model_pair_special_token_count", "maximum_valid_claim_only_tokens",
    }
    inference = {"retrieval_strategy", "k"}
    provenance = {
        "schema_version", "example_id", "question_id", "split", "split_group_id",
        "primary_population_eligible", "semantic_model_id", "semantic_model_revision",
        "semantic_config_sha256", "semantic_label_governance_sha256",
    }
    for field in SEMANTIC_LABEL_FIELDS:
        classes[field] = (
            "inference_available_feature" if field in inference
            else "label_only" if field in label_only else "provenance_only"
        )
    if set(classes) != set(SEMANTIC_LABEL_FIELDS) or set(classes) - label_only - inference - provenance:
        raise ValueError("semantic column governance is incomplete")
    claim_score_classes = {
        field: (
            "label_only" if field in {
                "claim_text", "entailment", "neutral", "contradiction",
                "maximum_unit_contradiction",
            }
            else "evaluation_only" if field in {
                "best_unit_id", "best_unit_type", "best_chunk_ids_json",
                "context_unit_count",
            }
            else "provenance_only"
        )
        for field in CLAIM_SCORE_FIELDS
    }
    confirmation_classes = {
        field: (
            "label_only" if field in {"benchmark_answer", "manual_label"}
            else "evaluation_only" if field in {
                "question", "retrieved_context_json", "rationale",
            }
            else "provenance_only"
        )
        for field in CONFIRMATION_BLINDED_FIELDS
    }
    confirmation_key_classes = {
        field: (
            "label_only" if field in {
                "y_suff_strict", "y_suff_semantic", "semantic_support_score",
                "minimum_claim_entailment", "mean_claim_entailment",
                "maximum_selected_premise_contradiction",
            }
            else "evaluation_only" if field == "confirmation_sampling_stratum"
            else "provenance_only"
        )
        for field in CONFIRMATION_KEY_FIELDS
    }
    return {
        "schema_version": "phase03-semantic-column-governance-v1",
        "semantic_label_governance_sha256": label_governance_sha256,
        "artifacts": {
            "context_sufficiency_semantic_labels": classes,
            "phase03_semantic_claim_scores": claim_score_classes,
            "phase03_semantic_confirmation_blinded": confirmation_classes,
            "phase03_semantic_confirmation_answer_key": confirmation_key_classes,
        },
        "future_classifier_feature_allowlist": sorted(inference),
        "blocked_gold_or_semantic_sources": [
            "reference answer", "claim text", "NLI scores", "semantic label", "strict label",
            "gold document identity", "span offsets", "benchmark answerability metadata",
        ],
        "rule": "Phase 4 feature export must reject every non-inference_available_feature field",
    }


def run_phase03_semantic(config: Phase03SemanticConfig, root: Path) -> dict[str, Any]:
    selected, selected_row, development = run_development_selection(config, root)
    label_governance, label_governance_sha256, label_governance_path = (
        _load_label_governance(selected, root)
    )
    print(
        f"Selected semantic method: {selected['model_id']}@{selected['model_revision']} "
        f"config={selected['semantic_config_sha256']}", flush=True,
    )
    semantic_rows, claim_rows, population_counts, resources = apply_selected_semantic_method(
        selected, config, label_governance_sha256, root
    )
    artifacts = dict(development["artifacts"])
    artifacts["semantic_labels"] = write_canonical_parquet(
        root / "artifacts/data/context_sufficiency_semantic_labels.parquet",
        semantic_rows, SEMANTIC_LABEL_FIELDS, ("question_id", "retrieval_strategy", "k"),
    )
    artifacts["primary_claim_scores"] = write_canonical_parquet(
        root / "artifacts/data/phase03_semantic_claim_scores.parquet",
        claim_rows, CLAIM_SCORE_FIELDS, ("question_id", "retrieval_strategy", "k", "claim_index"),
    )
    artifacts["semantic_unevaluable_questions"] = write_canonical_parquet(
        root / "artifacts/data/phase03_semantic_unevaluable_questions.parquet",
        resources["unevaluable_questions"], SEMANTIC_UNEVALUABLE_FIELDS, ("question_id",),
    )
    artifacts["semantic_label_governance_config"] = {
        "path": label_governance_path.as_posix(),
        "physical_sha256": sha256_file(label_governance_path),
        "semantic_sha256": label_governance_sha256,
        "bytes": label_governance_path.stat().st_size,
    }
    governance = _governance(label_governance_sha256)
    artifacts["semantic_column_governance"] = _json_artifact(
        root / "artifacts/data/phase03_semantic_column_governance.json", governance
    )
    confirmation, confirmation_manifest = _build_confirmation_pack(
        semantic_rows, selected, config, label_governance_sha256, root, resources
    )
    artifacts.update(confirmation["artifacts"])
    selected_development_rows = [
        row for row in development["conditions"] if row["model_id"] == selected["model_id"]
    ]
    development_report = _development_selected_report(
        selected, selected_development_rows, config
    )
    comparison = []
    for model in config.values["candidate_models"]:
        candidates = [
            row for row in development["thresholds"]
            if row["model_id"] == model["model_id"] and row["precision_gate_status"] == "pass"
        ]
        if candidates:
            best = min(candidates, key=_selection_key)
            comparison.append({
                "model_id": model["model_id"], "revision": model["revision"],
                "status": "evaluated", "selected_thresholds": {
                    key: best[key] for key in (
                        "minimum_claim_entailment", "mean_claim_entailment",
                        "maximum_selected_premise_contradiction",
                    )
                },
                "weighted_precision": best["weighted_precision"],
                "weighted_recall": best["weighted_recall"],
                "weighted_f1": best["weighted_f1"],
                "sample_precision": best["sample_precision"],
                "sample_recall": best["sample_recall"], "sample_f1": best["sample_f1"],
                "selected_overall": model["model_id"] == selected["model_id"],
            })
        else:
            failure = next((row for row in selected.get("incompatible_candidates", [])
                            if row["model_id"] == model["model_id"]), None)
            comparison.append(failure or {
                "model_id": model["model_id"], "revision": model["revision"],
                "status": "evaluated_no_configuration_passed_precision_gate",
                "selected_overall": False,
            })
    evaluable_semantic_rows = [
        row for row in semantic_rows if row["semantic_label_status"] == "evaluable"
    ]
    strict_positive = sum(row["y_suff_strict"] == 1 for row in evaluable_semantic_rows)
    semantic_positive = sum(row["y_suff_semantic"] == 1 for row in evaluable_semantic_rows)
    strict_semantic = {
        "comparison_scope": "semantic_evaluable_conditions_only",
        "all_conditions_strict_positive": sum(
            row["y_suff_strict"] == 1 for row in semantic_rows
        ),
        "strict_positive_conditions": strict_positive,
        "semantic_positive_conditions": semantic_positive,
        "agreement_conditions": sum(
            row["strict_semantic_agreement"] for row in evaluable_semantic_rows
        ),
        "disagreement_conditions": sum(
            row["strict_semantic_disagreement"] for row in evaluable_semantic_rows
        ),
        "strict_negative_semantic_positive_revisions": sum(
            row["y_suff_strict"] == 0 and row["y_suff_semantic"] == 1
            for row in evaluable_semantic_rows
        ),
        "strict_positive_semantic_negative_revisions": sum(
            row["y_suff_strict"] == 1 and row["y_suff_semantic"] == 0
            for row in evaluable_semantic_rows
        ),
        "by_strategy_and_k": [
            {
                "retrieval_strategy": strategy, "k": k,
                "conditions": len(group),
                "strict_positive": sum(row["y_suff_strict"] for row in group),
                "semantic_positive": sum(row["y_suff_semantic"] for row in group),
                "agreement": sum(row["strict_semantic_agreement"] for row in group),
            }
            for strategy, k in sorted({
                (row["retrieval_strategy"], int(row["k"]))
                for row in evaluable_semantic_rows
            })
            for group in [[row for row in evaluable_semantic_rows
                           if row["retrieval_strategy"] == strategy and int(row["k"]) == k]]
        ],
    }
    result = {
        "schema_version": "phase03-semantic-refinement-results-v2",
        "run_id": "phase03-semantic-" + selected["semantic_config_sha256"][:16],
        "git_commit_sha": _git_sha(root), "base_semantic_config_sha256": config.config_sha256,
        "semantic_config_sha256": selected["semantic_config_sha256"],
        "semantic_label_governance_sha256": label_governance_sha256,
        "semantic_configuration_relationship": {
            "scoring_configuration_sha256": selected["semantic_config_sha256"],
            "label_governance_configuration_sha256": label_governance_sha256,
            "scoring_configuration_changed_by_governance": False,
        },
        "candidate_comparison": comparison, "selected_method": selected,
        "primary_population": population_counts,
        "strict_vs_semantic": strict_semantic,
        "development_performance": development_report,
        "confirmation_sample": confirmation["status"],
        "test_semantic_scores_calculated": False,
        "test_semantic_aggregates_calculated": False,
        "phase4_started": False,
        "limitations": [
            "The original 150 annotations are development-only after method selection.",
            "Questions with any frozen-segmentation claim exceeding the selected model pair-input budget have y_suff_semantic missing and are excluded rather than described as insufficient.",
            "Benchmark-impossible examples have no reference answer and are not semantically NLI-evaluable; their original human contamination audit remains diagnostic.",
            "One primary human annotator was available, so inter-annotator agreement remains unavailable.",
            "The revised semantic rule remains provisional until the new blinded confirmation sample passes unchanged gates.",
        ],
    }
    artifacts["results"] = _json_artifact(
        root / "artifacts/results/phase03_semantic_refinement_results.json", result
    )
    hash_manifest = {
        "schema_version": "phase03-semantic-artifact-hashes-v2",
        "semantic_config_sha256": selected["semantic_config_sha256"],
        "semantic_label_governance_sha256": label_governance_sha256,
        "base_semantic_config_sha256": config.config_sha256,
        "git_commit_sha": result["git_commit_sha"], "artifacts": artifacts,
        "frozen_input_hashes": config.values["frozen_inputs"],
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "huggingface-hub", "pyarrow")
        },
        "test_outcomes_sealed": True, "phase4_started": False,
    }
    _json_artifact(root / "artifacts/results/phase03_semantic_artifact_hashes.json", hash_manifest)
    return result
