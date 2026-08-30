"""Phase 3.6b strict-preserving rescue selection and blinded confirmation export."""

from __future__ import annotations

import csv
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pyarrow.parquet as pq

from answerability_rag.hashing import canonical_json, canonical_json_sha256, sha256_file
from answerability_rag.io import read_csv
from answerability_rag.retrieval.artifacts import semantic_records_sha256, write_canonical_parquet

from .rescue import best_by_family, candidate_grid, evaluate_candidate, final_prediction, rescue_mechanisms, select_candidate
from .semantic_pipeline import _chunks_for_condition, _load_primary_population


GRID_PATH = Path("configs/phase03_rescue_grid.json")
GRID_FREEZE_PATH = Path("artifacts/results/phase03_rescue_candidate_grid_freeze.json")
SELECTED_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
SELECTED_REVISION = "6f5cf0a2b59cabb106aca4c287eed12e357e90eb"


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Sequence[str]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)
    return {"path": path.as_posix(), "rows": len(rows), "columns": list(fields),
            "physical_sha256": sha256_file(path),
            "semantic_sha256": semantic_records_sha256(rows, fields),
            "bytes": path.stat().st_size}


def _write_json(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"path": path.as_posix(), "physical_sha256": sha256_file(path),
            "bytes": path.stat().st_size}


def _load_and_verify_grid(root: Path) -> tuple[dict[str, Any], str]:
    path = root / GRID_PATH
    values = json.loads(path.read_text(encoding="utf-8"))
    physical = sha256_file(path)
    canonical = canonical_json_sha256(values)
    freeze = json.loads((root / GRID_FREEZE_PATH).read_text(encoding="utf-8"))
    if physical != freeze["candidate_grid_physical_sha256"]:
        raise ValueError("rescue grid changed after its pre-evaluation freeze")
    if canonical != freeze["candidate_grid_canonical_sha256"]:
        raise ValueError("canonical rescue-grid hash changed after freeze")
    if freeze["frozen_before_development_label_evaluation"] is not True:
        raise ValueError("rescue grid was not frozen before development evaluation")
    if values["test_seal"]["allow_test_load"] is not False:
        raise ValueError("rescue grid permits TEST access")
    candidate_grid(values)
    return values, canonical


def _verify_annotation_provenance(config: dict[str, Any], root: Path) -> None:
    dev = config["development"]
    for kind in ("selection", "transparency"):
        observed = sha256_file(root / dev[f"{kind}_annotation_path"])
        if observed != dev[f"{kind}_annotation_sha256"]:
            raise ValueError(f"{kind} annotation provenance hash mismatch")
    original = {row["sample_id"]: row for row in read_csv(root / dev["transparency_annotation_path"])}
    adjudicated = {row["sample_id"]: row for row in read_csv(root / dev["selection_annotation_path"])}
    if set(original) != set(adjudicated) or len(original) != 150:
        raise ValueError("original/adjudicated development samples do not match")
    changes = []
    for sample_id in original:
        if original[sample_id]["manual_label"] != adjudicated[sample_id]["manual_label"]:
            changes.append((original[sample_id]["question_id"], original[sample_id]["manual_label"],
                            adjudicated[sample_id]["manual_label"],
                            int(original[sample_id]["blind_order"])))
    if changes != [("DEV_Q217", "sufficient", "insufficient", 6)]:
        raise ValueError(f"unexpected adjudication label changes: {changes}")


def _load_development_rows(config: dict[str, Any], root: Path, annotation_key: str) -> list[dict[str, Any]]:
    annotations = {row["sample_id"]: row for row in read_csv(
        root / config["development"][annotation_key]
    )}
    manifest = {row["sample_id"]: row for row in read_csv(
        root / "artifacts/results/phase03_manual_sample_manifest.csv"
    )}
    keys = {row["sample_id"]: row for row in pq.read_table(
        root / "artifacts/results/phase03_annotation_answer_key.parquet"
    ).to_pylist()}
    selected_scores = {
        row["sample_id"]: row for row in pq.read_table(
            root / "artifacts/results/phase03_semantic_development_condition_scores.parquet"
        ).to_pylist()
        if row["model_id"] == SELECTED_MODEL and row["model_revision"] == SELECTED_REVISION
    }
    semantic = {row["example_id"]: row for row in pq.read_table(
        root / "artifacts/data/context_sufficiency_semantic_labels.parquet"
    ).to_pylist()}
    strata = set(config["development"]["primary_strata"])
    output = []
    for sample_id, annotation in annotations.items():
        man, key = manifest[sample_id], keys[sample_id]
        sem = semantic.get(man["example_id"])
        if man["split"] not in config["development"]["eligible_splits"]:
            continue
        if man["sampling_stratum"] not in strata or annotation["manual_label"] == "ambiguous":
            continue
        if sem is None or sem["semantic_label_status"] != "evaluable":
            continue
        score = selected_scores[sample_id]
        output.append({
            "sample_id": sample_id, "question_id": man["question_id"],
            "example_id": man["example_id"], "split": man["split"],
            "retrieval_strategy": man["retrieval_strategy"], "k": int(man["k"]),
            "sampling_stratum": man["sampling_stratum"],
            "manual_label": annotation["manual_label"],
            "y_suff_strict": int(key["automatic_y_suff"]),
            "maximum_span_coverage_fraction": key["maximum_span_coverage_fraction"],
            "minimum_claim_entailment": score["minimum_claim_entailment"],
            "mean_claim_entailment": score["mean_claim_entailment"],
            "maximum_selected_premise_contradiction": score[
                "maximum_selected_premise_contradiction"
            ],
        })
    if not output or any(row["split"] == "test" for row in output):
        raise ValueError("invalid or empty TRAIN+VALIDATION rescue development set")
    return output


def _evaluate(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    counts = config["development"]["frozen_population_stratum_counts"]
    strata = config["development"]["primary_strata"]
    return [evaluate_candidate(rows, candidate, counts, strata)
            for candidate in candidate_grid(config)]


def _serializable_result(row: dict[str, Any]) -> dict[str, Any]:
    return {**row,
            "weighted_confusion_proportions_json": canonical_json(
                row["weighted_confusion_proportions"]
            ),
            "metrics_by_stratum_json": canonical_json(row["metrics_by_stratum"])}


def _final_config(selected: dict[str, Any], grid_hash: str, grid: dict[str, Any]) -> tuple[dict[str, Any], str]:
    value = {
        "schema_version": "phase03-final-context-sufficiency-target-config-v1",
        "strict_preserving_rule": "if y_suff_strict == 1 then y_suff_final = 1 else apply frozen rescue rule",
        "strict_positive_demotion_permitted": False,
        "selected_rescue_family": selected["family"],
        "selected_thresholds": {key: selected[key] for key in (
            "T_cov", "T_mean", "T_min", "T_contradiction"
        )},
        "input_signal_definitions": {
            "y_suff_strict": "frozen Phase 3 exact accepted-gold-span coverage label",
            "maximum_span_coverage_fraction": "maximum normalized accepted-gold-span character coverage by retrieved chunks",
            "minimum_claim_entailment": "minimum selected-model entailment across frozen deterministic reference claims",
            "mean_claim_entailment": "mean selected-model entailment across frozen deterministic reference claims",
            "maximum_selected_premise_contradiction": "maximum contradiction score attached to each claim's selected maximum-entailment premise"
        },
        "semantic_unevaluable_exclusion_rule": "claim_exceeds_frozen_nli_pair_budget",
        "benchmark_impossible_primary_exclusion_rule": True,
        "unresolved_reference_or_evidence_primary_exclusion_rule": True,
        "candidate_grid_sha256": grid_hash,
        "upstream_configuration_hashes": grid["upstream"],
        "test_outcomes_sealed": True,
        "phase4_started": False,
    }
    return value, canonical_json_sha256(value)


def _apply_final_target(selected: dict[str, Any], final_hash: str, root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    semantic = pq.read_table(root / "artifacts/data/context_sufficiency_semantic_labels.parquet").to_pylist()
    strict = {row["example_id"]: row for row in pq.read_table(
        root / "artifacts/data/context_sufficiency_labels.parquet",
        filters=[("split", "in", ["train", "validation"])],
    ).to_pylist()}
    if any(row["split"] == "test" for row in semantic):
        raise ValueError("TEST row entered final-target construction")
    all_rows, primary = [], []
    mechanism_counts = Counter()
    exclusion_counts = Counter()
    for sem in semantic:
        source = strict[sem["example_id"]]
        evaluable = sem["semantic_label_status"] == "evaluable"
        row = {
            "schema_version": "phase03-final-context-sufficiency-label-v1",
            "example_id": sem["example_id"], "question_id": sem["question_id"],
            "split": sem["split"], "split_group_id": sem["split_group_id"],
            "retrieval_strategy": sem["retrieval_strategy"], "k": int(sem["k"]),
            "primary_phase4_target_eligible": evaluable,
            "primary_exclusion_reason": None if evaluable else sem["semantic_exclusion_reason"],
            "y_suff_strict": int(sem["y_suff_strict"]),
            "y_suff_semantic": sem["y_suff_semantic"], "y_suff_final": None,
            "maximum_span_coverage_fraction": source["maximum_span_coverage_fraction"],
            "minimum_claim_entailment": sem["minimum_claim_entailment"],
            "mean_claim_entailment": sem["mean_claim_entailment"],
            "maximum_selected_premise_contradiction": sem[
                "maximum_selected_premise_contradiction"
            ],
            "coverage_rescue": None, "nli_rescue": None, "rescue_applied": None,
            "selected_rescue_family": selected["family"],
            "semantic_config_sha256": sem["semantic_config_sha256"],
            "semantic_label_governance_sha256": sem["semantic_label_governance_sha256"],
            "final_target_config_sha256": final_hash,
        }
        if not evaluable:
            exclusion_counts[row["primary_exclusion_reason"]] += 1
            all_rows.append(row)
            continue
        row["y_suff_final"] = final_prediction(row, selected)
        coverage, nli = rescue_mechanisms(row, selected)
        row["coverage_rescue"] = bool(coverage and row["y_suff_strict"] == 0)
        row["nli_rescue"] = bool(nli and row["y_suff_strict"] == 0)
        row["rescue_applied"] = bool(row["y_suff_strict"] == 0 and row["y_suff_final"] == 1)
        if row["rescue_applied"]:
            mechanism_counts["coverage"] += int(row["coverage_rescue"])
            mechanism_counts["nli"] += int(row["nli_rescue"])
            mechanism_counts["overlap"] += int(row["coverage_rescue"] and row["nli_rescue"])
        all_rows.append(row)
        primary.append({key: row[key] for key in (
            "schema_version", "example_id", "question_id", "split", "split_group_id",
            "retrieval_strategy", "k", "y_suff_final", "final_target_config_sha256"
        )})
    if any(row["y_suff_strict"] == 1 and row["y_suff_final"] != 1 for row in all_rows if row["y_suff_final"] is not None):
        raise AssertionError("strict-positive demotion detected")
    counts = {
        "eligible_questions": len({row["question_id"] for row in primary}),
        "eligible_conditions": len(primary),
        "strict_positive_count": sum(row["y_suff_strict"] == 1 for row in all_rows if row["y_suff_final"] is not None),
        "rescued_positive_count": sum(bool(row["rescue_applied"]) for row in all_rows if row["y_suff_final"] is not None),
        "final_positive_count": sum(row["y_suff_final"] == 1 for row in all_rows),
        "final_negative_count": sum(row["y_suff_final"] == 0 for row in all_rows),
        "rescue_count_from_coverage": mechanism_counts["coverage"],
        "rescue_count_from_nli": mechanism_counts["nli"],
        "rescue_mechanism_overlap": mechanism_counts["overlap"],
        "semantic_unevaluable_questions": len({row["question_id"] for row in all_rows if row["y_suff_final"] is None}),
        "semantic_unevaluable_conditions": sum(row["y_suff_final"] is None for row in all_rows),
        "exclusion_reasons": dict(exclusion_counts),
    }
    return all_rows, primary, counts


def _select_confirmation(final_rows: list[dict[str, Any]], config: dict[str, Any], final_hash: str, root: Path) -> list[dict[str, Any]]:
    original_questions = {row["question_id"] for row in read_csv(
        root / "artifacts/results/phase03_manual_sample_manifest.csv"
    )}
    previous_questions = {row["question_id"] for row in read_csv(
        root / "artifacts/results/phase03_semantic_confirmation_sample_manifest.csv"
    )}
    eligible = [row for row in final_rows if row["primary_phase4_target_eligible"]
                and row["question_id"] not in original_questions
                and row["question_id"] not in previous_questions]
    for row in eligible:
        row["confirmation_sampling_stratum"] = (
            "strict_positive_retained" if row["y_suff_strict"] == 1 else
            "strict_negative_rescued" if row["y_suff_final"] == 1 else
            "strict_negative_not_rescued"
        )
    targets = {"strict_positive_retained": 17, "strict_negative_rescued": 17,
               "strict_negative_not_rescued": 16}
    seed = int(config["confirmation"]["seed"])
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    strategy, depth, split = Counter(), Counter(), Counter()

    def stable(row: dict[str, Any], purpose: str) -> str:
        return canonical_json_sha256({"seed": seed, "purpose": purpose,
            "question_id": row["question_id"], "example_id": row["example_id"],
            "final_target_config_sha256": final_hash})

    def take(pool: list[dict[str, Any]], count: int, purpose: str) -> None:
        for _ in range(count):
            available = [row for row in pool if row["question_id"] not in used]
            if not available:
                return
            row = min(available, key=lambda item: (
                strategy[item["retrieval_strategy"]], depth[int(item["k"])],
                split[item["split"]], stable(item, purpose)))
            selected.append(row); used.add(row["question_id"])
            strategy[row["retrieval_strategy"]] += 1
            depth[int(row["k"])] += 1; split[row["split"]] += 1

    for stratum, count in targets.items():
        take([row for row in eligible if row["confirmation_sampling_stratum"] == stratum], count, stratum)
    if len(selected) < 50:
        take(eligible, 50 - len(selected), "availability_fill")
    if len(selected) != 50 or len({row["question_id"] for row in selected}) != 50:
        raise ValueError("could not construct frozen 50-unique-question confirmation sample")
    selected.sort(key=lambda row: stable(row, "blind_order"))
    return selected


def _confirmation_pack(sample: list[dict[str, Any]], config: dict[str, Any], final_hash: str, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    _, condition_map, chunks, questions = _load_primary_population(root)
    guideline = root / "docs/PHASE_03_ANNOTATION_GUIDE.md"
    guideline_sha = sha256_file(guideline)
    manifests, blinded, keys = [], [], []
    for blind_order, row in enumerate(sample, 1):
        sample_id = canonical_json_sha256({"example_id": row["example_id"],
            "seed": config["confirmation"]["seed"],
            "version": config["confirmation"]["version"],
            "final_target_config_sha256": final_hash})
        condition = condition_map[(row["question_id"], row["retrieval_strategy"], int(row["k"]))]
        context = [{"rank": int(chunk["rank"]), "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["doc_id"], "filename": chunk["filename"],
                    "char_start": int(chunk["char_start"]), "char_end": int(chunk["char_end"]),
                    "text": chunk["text"]}
                   for chunk in _chunks_for_condition(condition, chunks)]
        common = {"sample_id": sample_id, "blind_order": blind_order,
                  "question_id": row["question_id"], "split": row["split"],
                  "retrieval_strategy": row["retrieval_strategy"], "k": int(row["k"])}
        manifests.append({"schema_version": "phase03-final-confirmation-sample-v1", **common,
            "confirmation_sampling_stratum": row["confirmation_sampling_stratum"],
            "sample_seed": config["confirmation"]["seed"], "example_id": row["example_id"],
            "final_target_config_sha256": final_hash,
            "annotation_guideline_version": "phase03-sufficiency-annotation-v1",
            "annotation_guideline_sha256": guideline_sha})
        q = questions[row["question_id"]]
        blinded.append({"schema_version": "phase03-final-confirmation-blinded-v1", **common,
            "question": q.question, "benchmark_status": "answerable_with_reference",
            "benchmark_answer": q.answer, "retrieved_context_json": canonical_json(context),
            "annotation_guideline_version": "phase03-sufficiency-annotation-v1",
            "annotation_guideline_sha256": guideline_sha, "annotator_id": "",
            "manual_label": "", "rationale": "", "annotation_timestamp": ""})
        keys.append({"schema_version": "phase03-final-confirmation-key-v1",
            "sample_id": sample_id, "question_id": row["question_id"],
            "example_id": row["example_id"],
            "confirmation_sampling_stratum": row["confirmation_sampling_stratum"],
            "y_suff_strict": row["y_suff_strict"], "y_suff_semantic": row["y_suff_semantic"],
            "y_suff_final": row["y_suff_final"],
            "maximum_span_coverage_fraction": row["maximum_span_coverage_fraction"],
            "minimum_claim_entailment": row["minimum_claim_entailment"],
            "mean_claim_entailment": row["mean_claim_entailment"],
            "maximum_selected_premise_contradiction": row["maximum_selected_premise_contradiction"],
            "final_target_config_sha256": final_hash})
    manifest_fields = tuple(manifests[0])
    blind_fields = tuple(blinded[0])
    key_fields = tuple(keys[0])
    forbidden = {"y_suff_strict", "y_suff_semantic", "y_suff_final",
        "maximum_span_coverage_fraction", "minimum_claim_entailment", "mean_claim_entailment",
        "maximum_selected_premise_contradiction", "model_prediction",
        "confirmation_sampling_stratum", "example_id", "final_target_config_sha256"}
    if forbidden & set(blind_fields):
        raise ValueError("final confirmation blinded view exposes answer-key fields")
    artifacts = {
        "manifest": _write_csv(root / "artifacts/results/phase03_final_confirmation_sample_manifest.csv", manifests, manifest_fields),
        "blinded": write_canonical_parquet(root / "artifacts/results/phase03_final_confirmation_blinded.parquet", blinded, blind_fields, ("blind_order",)),
        "template": _write_csv(root / "artifacts/results/phase03_final_confirmation_template.csv", blinded, blind_fields),
        "answer_key": write_canonical_parquet(root / "artifacts/results/phase03_final_confirmation_answer_key.parquet", keys, key_fields, ("sample_id",)),
    }
    status = {"schema_version": "phase03-final-confirmation-status-v1",
        "status": "pending_unannotated", "sample_rows": 50, "unique_questions": 50,
        "sample_strata": dict(Counter(row["confirmation_sampling_stratum"] for row in manifests)),
        "strategy_counts": dict(Counter(row["retrieval_strategy"] for row in manifests)),
        "depth_counts": {str(k): v for k, v in Counter(row["k"] for row in manifests).items()},
        "split_counts": dict(Counter(row["split"] for row in manifests)),
        "overlap_original_development_questions": 0,
        "overlap_superseded_confirmation_questions": 0,
        "benchmark_impossible_rows": 0, "semantic_unevaluable_rows": 0, "test_rows": 0,
        "automatic_sufficient_precision_gate": 0.90,
        "prevalence_weighted_f1_gate": 0.85,
        "answer_key_evaluation_status": "sealed_until_complete_genuine_human_confirmation",
        "second_human_annotator_required": False,
        "inter_annotator_agreement_limitation": "No second genuine human annotator currently exists.",
        "sample_size_limitation": config["confirmation"]["sample_size_limitation"],
        "final_target_config_sha256": final_hash}
    artifacts["status"] = _write_json(root / "artifacts/results/phase03_final_confirmation_status.json", status)
    return artifacts, status


def run_phase03_rescue(root: Path) -> dict[str, Any]:
    grid, grid_hash = _load_and_verify_grid(root)
    _verify_annotation_provenance(grid, root)
    adjudicated_rows = _load_development_rows(grid, root, "selection_annotation_path")
    original_rows = _load_development_rows(grid, root, "transparency_annotation_path")
    adjudicated_results = _evaluate(adjudicated_rows, grid)
    original_results = _evaluate(original_rows, grid)
    selected = select_candidate(adjudicated_results, grid)
    family_best = best_by_family(adjudicated_results, grid)
    if selected is None:
        gates_pass = False
    else:
        gates_pass = (float(selected["sample_precision"]) >= float(grid["development_gates"]["minimum_ordinary_precision"])
                      and float(selected["weighted_f1"]) >= float(grid["development_gates"]["minimum_prevalence_weighted_f1"]))
    candidate_fields = ("candidate_order", "family", "T_cov", "T_mean", "T_min", "T_contradiction",
        "eligible_count", "tn", "fp", "fn", "tp", "sample_accuracy", "sample_precision",
        "sample_recall", "sample_f1", "weighted_accuracy", "weighted_precision", "weighted_recall",
        "weighted_f1", "rescued_by_coverage", "rescued_by_nli", "rescue_mechanism_overlap",
        "weighted_confusion_proportions_json", "metrics_by_stratum_json")
    candidate_artifact = _write_csv(root / "artifacts/results/phase03_rescue_candidate_results.csv",
        [_serializable_result(row) for row in adjudicated_results], candidate_fields)
    original_by_order = {row["candidate_order"]: row for row in original_results}
    comparison = None if selected is None else {
        "adjudicated": selected,
        "original": original_by_order[selected["candidate_order"]],
        "changed_label": {"blind_order": 6, "question_id": "DEV_Q217",
                          "original": "sufficient", "adjudicated": "insufficient"},
    }
    report: dict[str, Any] = {
        "schema_version": "phase03-rescue-development-report-v1",
        "candidate_grid_sha256": grid_hash, "candidate_count": len(adjudicated_results),
        "adjudicated_development_rows": len(adjudicated_rows),
        "original_development_rows": len(original_rows),
        "best_by_family": family_best, "selected": selected,
        "original_vs_adjudicated_selected_rule": comparison,
        "development_gates": grid["development_gates"], "development_gates_passed": gates_pass,
        "test_outcomes_sealed": True, "phase4_started": False,
    }
    artifacts: dict[str, Any] = {"candidate_results": candidate_artifact}
    supersession = {"schema_version": "phase03-semantic-confirmation-supersession-v1",
        "prior_confirmation_status": "superseded_unannotated",
        "prior_confirmation_manifest_sha256": sha256_file(root / "artifacts/results/phase03_semantic_confirmation_sample_manifest.csv"),
        "prior_confirmation_template_sha256": sha256_file(root / "artifacts/results/phase03_semantic_confirmation_template.csv"),
        "prior_confirmation_blinded_sha256": sha256_file(root / "artifacts/results/phase03_semantic_confirmation_blinded.parquet"),
        "prior_confirmation_answer_key_sha256": sha256_file(root / "artifacts/results/phase03_semantic_confirmation_answer_key.parquet"),
        "answer_key_values_opened": False, "human_annotations_present": False,
        "reason": "semantic-only target not accepted; superseded before annotation by the Phase 3.6b decision",
        "test_outcomes_sealed": True, "phase4_started": False}
    artifacts["supersession"] = _write_json(
        root / "artifacts/results/phase03_semantic_confirmation_supersession.json", supersession)
    if not gates_pass:
        diagnostic_rule = selected or family_best["combined"]
        false_negatives, false_positives = [], []
        for row in adjudicated_rows:
            prediction = final_prediction(row, diagnostic_rule)
            truth = row["manual_label"] == "sufficient"
            if truth and prediction == 0:
                false_negatives.append(row)
            elif not truth and prediction == 1:
                false_positives.append(row)
        def coverage_band(row: dict[str, Any]) -> str:
            value = row["maximum_span_coverage_fraction"]
            if value is None:
                return "missing"
            return "below_0.25" if float(value) < 0.25 else "0.25_to_below_0.50" if float(value) < 0.50 else "at_least_0.50"
        def nli_failure(row: dict[str, Any]) -> str:
            reasons = []
            if float(row["mean_claim_entailment"]) < float(diagnostic_rule["T_mean"]):
                reasons.append("mean_below_threshold")
            if float(row["minimum_claim_entailment"]) < float(diagnostic_rule["T_min"]):
                reasons.append("minimum_below_threshold")
            if float(row["maximum_selected_premise_contradiction"]) >= float(diagnostic_rule["T_contradiction"]):
                reasons.append("contradiction_veto")
            return "+".join(reasons) or "nli_passed"
        patterns = {
            "diagnostic_rule_candidate_order": diagnostic_rule["candidate_order"],
            "false_positive_count": len(false_positives),
            "false_positive_question_ids": [row["question_id"] for row in false_positives],
            "false_negative_count": len(false_negatives),
            "false_negative_question_ids": [row["question_id"] for row in false_negatives],
            "false_negatives_by_stratum": dict(Counter(row["sampling_stratum"] for row in false_negatives)),
            "false_negatives_by_retrieval_strategy_diagnostic_only": dict(Counter(row["retrieval_strategy"] for row in false_negatives)),
            "false_negatives_by_k_diagnostic_only": {str(k): v for k, v in Counter(int(row["k"]) for row in false_negatives).items()},
            "false_negative_coverage_bands": dict(Counter(coverage_band(row) for row in false_negatives)),
            "false_negative_nli_failure_combinations": dict(Counter(nli_failure(row) for row in false_negatives)),
        }
        report["remaining_error_patterns"] = patterns
        lines = ["# Phase 3.6b Decision Request", "", "No frozen candidate passed both development gates.", "",
                 f"Candidate grid SHA-256: `{grid_hash}`", "",
                 f"The grid contained {len(adjudicated_results)} predeclared rules. Selection used {len(adjudicated_rows)} benchmark-answerable, semantic-evaluable, non-ambiguous TRAIN+VALIDATION development rows and the frozen population weights.", ""]
        for family, row in family_best.items():
            lines += [f"## Best {family}", "", f"Thresholds: T_cov={row['T_cov']}, T_mean={row['T_mean']}, T_min={row['T_min']}, contradiction<{row['T_contradiction']}.",
                      f"Precision={row['sample_precision']:.4f}, recall={row['sample_recall']:.4f}, F1={row['sample_f1']:.4f}.",
                      f"Weighted precision={row['weighted_precision']:.4f}, weighted recall={row['weighted_recall']:.4f}, weighted F1={row['weighted_f1']:.4f}.",
                      f"Confusion matrix: TN={row['tn']}, FP={row['fp']}, FN={row['fn']}, TP={row['tp']}.", ""]
        lines += ["### Selected combined rule by original development stratum", "",
                  "| Stratum | n | TN | FP | FN | TP | Precision | Recall | F1 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for stratum in diagnostic_rule["metrics_by_stratum"]:
            cells = stratum["confusion_matrix"]
            precision = "NA" if stratum["precision"] is None else f"{stratum['precision']:.4f}"
            lines.append(
                f"| {stratum['sampling_stratum']} | {stratum['completed_count']} | "
                f"{cells['tn']} | {cells['fp']} | {cells['fn']} | {cells['tp']} | "
                f"{precision} | {stratum['recall']:.4f} | {stratum['f1']:.4f} |"
            )
        lines += [""]
        lines += ["## Development-gate result", "",
                  f"The selected combined rule retained precision {diagnostic_rule['sample_precision']:.4f} but reached prevalence-weighted F1 {diagnostic_rule['weighted_f1']:.4f}, below the frozen 0.85 gate. It therefore cannot be frozen as the final target.", "",
                  "## Remaining error patterns", "",
                  f"False positives: {len(false_positives)}. No automatic-sufficient false-positive pattern was observed.",
                  f"False negatives: {len(false_negatives)}; by original stratum: {canonical_json(patterns['false_negatives_by_stratum'])}.",
                  f"Coverage bands among false negatives: {canonical_json(patterns['false_negative_coverage_bands'])}.",
                  f"NLI failure combinations among false negatives: {canonical_json(patterns['false_negative_nli_failure_combinations'])}.",
                  "Retrieval strategy and k summaries are retained only as diagnostics; neither was used as a rescue-boundary predictor.", "",
                  "## Original-versus-adjudicated transparency", "",
                  "Primary rescue metrics are identical for the original and adjudicated files because the sole changed row, DEV_Q217, belongs to the frozen benchmark-impossible stratum and is excluded from primary selection. The historical original file remains unchanged in content.", "",
                  f"For both files, the selected rule therefore has precision {diagnostic_rule['sample_precision']:.4f}, recall {diagnostic_rule['sample_recall']:.4f}, F1 {diagnostic_rule['sample_f1']:.4f}, weighted precision {diagnostic_rule['weighted_precision']:.4f}, weighted recall {diagnostic_rule['weighted_recall']:.4f}, weighted F1 {diagnostic_rule['weighted_f1']:.4f}, and confusion matrix TN={diagnostic_rule['tn']}, FP={diagnostic_rule['fp']}, FN={diagnostic_rule['fn']}, TP={diagnostic_rule['tp']} on the PRIMARY development population.", "",
                  "## Recommendation", "", "Stop Phase 3.6b at this decision point. Any further labeling method or any alteration of the frozen gates would require an explicit methodological decision; no post-hoc rule is recommended from these results.", "",
                  "No independent confirmation sample or Phase 4 artifact was created. TEST aggregate outcomes remained sealed.", ""]
        path = root / "docs/PHASE_03_6B_DECISION_REQUEST.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        artifacts["decision_request"] = {"path": path.as_posix(), "physical_sha256": sha256_file(path)}
        report["artifacts"] = artifacts
        report_artifact = _write_json(
            root / "artifacts/results/phase03_rescue_development_report.json", report
        )
        artifacts["development_report"] = report_artifact
        manifest = {"schema_version": "phase03-rescue-artifact-manifest-v1",
            "run_status": "development_gates_failed_decision_request",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "execution_timing": "resumed Phase 3.6b session; no NLI inference executed",
            "git_commit_sha": subprocess.check_output(
                ["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "HEAD"],
                cwd=root, text=True,
            ).strip(),
            "candidate_grid_sha256": grid_hash, "candidate_count": len(adjudicated_results),
            "artifacts": artifacts, "test_outcomes_sealed": True, "phase4_started": False}
        _write_json(root / "artifacts/results/phase03_rescue_artifact_manifest.json", manifest)
        return report
    final_config, final_hash = _final_config(selected, grid_hash, grid)
    final_config["final_target_config_sha256"] = final_hash
    artifacts["final_config"] = _write_json(root / "configs/phase03_final_target.json", final_config)
    all_rows, primary_rows, population = _apply_final_target(selected, final_hash, root)
    all_fields, primary_fields = tuple(all_rows[0]), tuple(primary_rows[0])
    artifacts["final_labels"] = write_canonical_parquet(
        root / "artifacts/data/context_sufficiency_final_labels.parquet", all_rows, all_fields, ("example_id",))
    artifacts["primary_target"] = write_canonical_parquet(
        root / "artifacts/data/phase03_final_primary_target.parquet", primary_rows, primary_fields, ("example_id",))
    sample = _select_confirmation(all_rows, grid, final_hash, root)
    confirmation_artifacts, confirmation_status = _confirmation_pack(sample, grid, final_hash, root)
    artifacts.update({f"confirmation_{k}": v for k, v in confirmation_artifacts.items()})
    report.update({"selected": selected, "final_target_config_sha256": final_hash,
                   "primary_population": population, "confirmation": confirmation_status})
    report["artifacts"] = artifacts
    report_artifact = _write_json(root / "artifacts/results/phase03_rescue_development_report.json", report)
    artifacts["development_report"] = report_artifact
    manifest = {"schema_version": "phase03-rescue-artifact-manifest-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": subprocess.check_output(
            ["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "HEAD"],
            cwd=root, text=True,
        ).strip(),
        "candidate_grid_sha256": grid_hash, "final_target_config_sha256": final_hash,
        "artifacts": artifacts, "test_outcomes_sealed": True, "phase4_started": False}
    _write_json(root / "artifacts/results/phase03_rescue_artifact_manifest.json", manifest)
    return report
