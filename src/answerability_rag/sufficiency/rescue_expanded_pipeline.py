"""Phase 3.6c expanded strict-preserving rescue development workflow."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from answerability_rag.hashing import canonical_json, canonical_json_sha256, sha256_file
from answerability_rag.retrieval.artifacts import write_canonical_parquet

from .rescue import best_by_family, evaluate_candidate, final_prediction, select_candidate
from .rescue_expanded import expanded_candidate_grid, wilson_interval
from .rescue_pipeline import (
    _apply_final_target,
    _confirmation_pack,
    _load_development_rows,
    _select_confirmation,
    _serializable_result,
    _write_csv,
    _write_json,
)


GRID_PATH = Path("configs/phase03_rescue_grid_expanded.json")
FREEZE_PATH = Path("artifacts/results/phase03_rescue_expanded_candidate_grid_freeze.json")


def _load_and_verify(root: Path) -> tuple[dict[str, Any], str]:
    grid_path = root / GRID_PATH
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    freeze = json.loads((root / FREEZE_PATH).read_text(encoding="utf-8"))
    physical, canonical = sha256_file(grid_path), canonical_json_sha256(grid)
    if physical != freeze["candidate_grid_physical_sha256"]:
        raise ValueError("expanded rescue grid physical hash changed after freeze")
    if canonical != freeze["candidate_grid_canonical_sha256"]:
        raise ValueError("expanded rescue grid canonical hash changed after freeze")
    if freeze["frozen_before_development_outcome_calculation"] is not True:
        raise ValueError("expanded grid was not frozen before development outcomes")
    if grid["iteration_governance"]["final_automatic_threshold_refinement_iteration"] is not True:
        raise ValueError("Phase 3.6c final-iteration hard stop is missing")
    if grid["test_seal"] != {"allow_test_load": False, "allow_test_aggregate_outcomes": False}:
        raise ValueError("Phase 3.6c TEST seal changed")
    expanded_candidate_grid(grid)

    frozen_b = grid["frozen_phase03_6b"]
    prior = {
        "candidate_grid_physical_sha256": root / "configs/phase03_rescue_grid.json",
        "candidate_results_physical_sha256": root / "artifacts/results/phase03_rescue_candidate_results.csv",
        "development_report_physical_sha256": root / "artifacts/results/phase03_rescue_development_report.json",
        "artifact_manifest_physical_sha256": root / "artifacts/results/phase03_rescue_artifact_manifest.json",
    }
    for key, path in prior.items():
        if sha256_file(path) != frozen_b[key]:
            raise ValueError(f"frozen Phase 3.6b artifact changed: {path}")
    prior_grid = json.loads((root / "configs/phase03_rescue_grid.json").read_text(encoding="utf-8"))
    if canonical_json_sha256(prior_grid) != frozen_b["candidate_grid_canonical_sha256"]:
        raise ValueError("frozen Phase 3.6b candidate-grid canonical hash changed")

    development = grid["development"]
    if sha256_file(root / development["selection_annotation_path"]) != development["selection_annotation_sha256"]:
        raise ValueError("adjudicated development annotation hash changed")
    if sha256_file(root / development["historical_annotation_path"]) != development["historical_annotation_sha256"]:
        raise ValueError("historical development annotation hash changed")
    upstream_files = {
        "phase03_strict_labels_physical_sha256": root / "artifacts/data/context_sufficiency_labels.parquet",
        "phase03_semantic_labels_physical_sha256": root / "artifacts/data/context_sufficiency_semantic_labels.parquet",
        "phase03_semantic_development_scores_physical_sha256": root / "artifacts/results/phase03_semantic_development_condition_scores.parquet",
    }
    for key, path in upstream_files.items():
        if sha256_file(path) != grid["upstream"][key]:
            raise ValueError(f"Phase 3.6c upstream hash mismatch: {path}")
    return grid, canonical


def _evaluate(rows: list[dict[str, Any]], grid: dict[str, Any]) -> list[dict[str, Any]]:
    counts = grid["development"]["frozen_population_stratum_counts"]
    strata = grid["development"]["primary_strata"]
    return [evaluate_candidate(rows, candidate, counts, strata)
            for candidate in expanded_candidate_grid(grid)]


def _error_patterns(rows: list[dict[str, Any]], selected: dict[str, Any]) -> dict[str, Any]:
    false_negatives, false_positives = [], []
    for row in rows:
        prediction = final_prediction(row, selected)
        truth = row["manual_label"] == "sufficient"
        if truth and prediction == 0:
            false_negatives.append(row)
        elif not truth and prediction == 1:
            false_positives.append(row)

    def coverage_band(row: dict[str, Any]) -> str:
        value = row["maximum_span_coverage_fraction"]
        if value is None:
            return "missing"
        thresholds = selected["T_cov"]
        return "below_selected_threshold" if thresholds is not None and float(value) < float(thresholds) else "at_or_above_selected_threshold"

    def nli_failure(row: dict[str, Any]) -> str:
        if selected["T_mean"] is None:
            return "not_used_by_selected_family"
        reasons = []
        if float(row["mean_claim_entailment"]) < float(selected["T_mean"]):
            reasons.append("mean_below_threshold")
        if float(row["minimum_claim_entailment"]) < float(selected["T_min"]):
            reasons.append("minimum_below_threshold")
        if float(row["maximum_selected_premise_contradiction"]) >= float(selected["T_contradiction"]):
            reasons.append("contradiction_veto")
        return "+".join(reasons) or "nli_passed"

    return {
        "false_positive_count": len(false_positives),
        "false_positive_question_ids": [row["question_id"] for row in false_positives],
        "false_positives_by_stratum": dict(Counter(row["sampling_stratum"] for row in false_positives)),
        "false_negative_count": len(false_negatives),
        "false_negative_question_ids": [row["question_id"] for row in false_negatives],
        "false_negatives_by_stratum": dict(Counter(row["sampling_stratum"] for row in false_negatives)),
        "false_negative_coverage_relation": dict(Counter(coverage_band(row) for row in false_negatives)),
        "false_negative_nli_failure_combinations": dict(Counter(nli_failure(row) for row in false_negatives)),
        "retrieval_strategy_diagnostic_only": {
            "false_positives": dict(Counter(row["retrieval_strategy"] for row in false_positives)),
            "false_negatives": dict(Counter(row["retrieval_strategy"] for row in false_negatives)),
        },
        "k_diagnostic_only": {
            "false_positives": {str(k): v for k, v in Counter(int(row["k"]) for row in false_positives).items()},
            "false_negatives": {str(k): v for k, v in Counter(int(row["k"]) for row in false_negatives).items()},
        },
    }


def _final_target_config(selected: dict[str, Any], grid_hash: str, grid: dict[str, Any]) -> tuple[dict[str, Any], str]:
    value = {
        "schema_version": "phase03-final-context-sufficiency-target-config-v1",
        "selection_iteration": "phase03_6c_final_automatic_threshold_refinement",
        "strict_preserving_rule": "if y_suff_strict == 1 then y_suff_final = 1 else apply the frozen selected rescue rule",
        "strict_positive_demotion_permitted": False,
        "selected_rescue_family": selected["family"],
        "selected_thresholds": {key: selected[key] for key in (
            "T_cov", "T_mean", "T_min", "T_contradiction"
        )},
        "input_signals": grid["rule_boundary_signals"],
        "semantic_unevaluable_exclusion_rule": "claim_exceeds_frozen_nli_pair_budget",
        "benchmark_impossible_primary_exclusion": True,
        "unresolved_reference_or_evidence_primary_exclusion": True,
        "expanded_candidate_grid_sha256": grid_hash,
        "frozen_phase03_6b_hashes": grid["frozen_phase03_6b"],
        "upstream_configuration_hashes": grid["upstream"],
        "confirmation_gates": {
            "minimum_ordinary_precision": grid["confirmation"]["human_gate_minimum_ordinary_precision"],
            "minimum_prevalence_weighted_f1": grid["confirmation"]["human_gate_minimum_prevalence_weighted_f1"],
        },
        "test_outcomes_sealed": True,
        "phase4_started": False,
    }
    return value, canonical_json_sha256(value)


def _candidate_csv(root: Path, results: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ("candidate_order", "family", "T_cov", "T_mean", "T_min", "T_contradiction",
        "historically_evaluated_in_phase03_6b", "new_phase03_6c_candidate", "eligible_count",
        "tn", "fp", "fn", "tp", "sample_accuracy", "sample_precision", "sample_recall",
        "sample_f1", "weighted_accuracy", "weighted_precision", "weighted_recall", "weighted_f1",
        "rescued_by_coverage", "rescued_by_nli", "rescue_mechanism_overlap",
        "weighted_confusion_proportions_json", "metrics_by_stratum_json")
    return _write_csv(root / "artifacts/results/phase03_rescue_expanded_candidate_results.csv",
                      [_serializable_result(row) for row in results], fields)


def _decision_request(root: Path, report: dict[str, Any]) -> dict[str, Any]:
    lines = ["# Phase 3.6c Decision Request", "",
        "No predeclared expanded-grid candidate passed both frozen development gates.", "",
        f"Expanded-grid SHA-256: `{report['expanded_grid_sha256']}`", "",
        f"Candidates: {report['candidate_count']} total, {report['new_candidate_count']} new, and {report['historically_evaluated_candidate_count']} historical comparison points.", ""]
    for family, row in report["best_by_family"].items():
        lines += [f"## Best {family}", "",
            f"Thresholds: T_cov={row['T_cov']}, T_mean={row['T_mean']}, T_min={row['T_min']}, contradiction<{row['T_contradiction']}.",
            f"Precision={row['sample_precision']:.4f}, recall={row['sample_recall']:.4f}, F1={row['sample_f1']:.4f}.",
            f"Weighted precision={row['weighted_precision']:.4f}, weighted recall={row['weighted_recall']:.4f}, weighted F1={row['weighted_f1']:.4f}.",
            f"Confusion matrix: TN={row['tn']}, FP={row['fp']}, FN={row['fn']}, TP={row['tp']}.", ""]
    selected = report["selected"]
    lines += ["## Selected eligible candidate", "",
        f"Precision 95% Wilson CI: [{report['precision_wilson_95']['lower']:.4f}, {report['precision_wilson_95']['upper']:.4f}].",
        f"Weighted-F1 gate gap: {report['weighted_f1_gate_gap']:.4f} ({report['gate_miss_description']}).", "",
        "### Performance by original development stratum", "",
        "| Stratum | n | TN | FP | FN | TP | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for stratum in selected["metrics_by_stratum"]:
        cells = stratum["confusion_matrix"]
        precision = "NA" if stratum["precision"] is None else f"{stratum['precision']:.4f}"
        lines.append(f"| {stratum['sampling_stratum']} | {stratum['completed_count']} | {cells['tn']} | {cells['fp']} | {cells['fn']} | {cells['tp']} | {precision} | {stratum['recall']:.4f} | {stratum['f1']:.4f} |")
    patterns = report["remaining_error_patterns"]
    lines += ["", "## Remaining error patterns", "",
        f"False positives: {patterns['false_positive_count']}; by stratum: {canonical_json(patterns['false_positives_by_stratum'])}.",
        f"False negatives: {patterns['false_negative_count']}; by stratum: {canonical_json(patterns['false_negatives_by_stratum'])}.",
        f"False-negative NLI failure combinations: {canonical_json(patterns['false_negative_nli_failure_combinations'])}.", "",
        "## Recommendation", "",
        "End automatic label refinement as predeclared. Simplify the thesis methodology by reporting the strict rule, semantic-only rule, Phase 3.6b rescue, and this final expanded search as transparent development evidence; any different target-construction method now requires a new explicit methodological decision.", "",
        "No confirmation sample, TEST aggregate, or Phase 4 artifact was created.", ""]
    path = root / "docs/PHASE_03_6C_DECISION_REQUEST.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return {"path": path.as_posix(), "physical_sha256": sha256_file(path), "bytes": path.stat().st_size}


def run_phase03_rescue_expanded(root: Path) -> dict[str, Any]:
    grid, grid_hash = _load_and_verify(root)
    adjudicated = _load_development_rows(grid, root, "selection_annotation_path")
    historical = _load_development_rows(grid, root, "historical_annotation_path")
    results = _evaluate(adjudicated, grid)
    historical_results = _evaluate(historical, grid)
    selected = select_candidate(results, grid)
    if selected is None:
        raise ValueError("expanded grid has no precision-eligible candidate")
    family_best = best_by_family(results, grid)
    precision_interval = wilson_interval(
        int(selected["tp"]), int(selected["tp"] + selected["fp"]),
        float(grid["precision_interval"]["z"]),
    )
    gates_pass = (
        float(selected["sample_precision"]) >= float(grid["development_gates"]["minimum_ordinary_precision"])
        and float(selected["weighted_f1"]) >= float(grid["development_gates"]["minimum_prevalence_weighted_f1"])
    )
    gap = max(0.0, float(grid["development_gates"]["minimum_prevalence_weighted_f1"]) - float(selected["weighted_f1"]))
    historical_by_order = {row["candidate_order"]: row for row in historical_results}
    artifacts: dict[str, Any] = {"candidate_results": _candidate_csv(root, results)}
    report: dict[str, Any] = {
        "schema_version": "phase03-expanded-rescue-development-report-v1",
        "expanded_grid_sha256": grid_hash,
        "candidate_count": len(results),
        "new_candidate_count": sum(row["new_phase03_6c_candidate"] for row in results),
        "historically_evaluated_candidate_count": sum(row["historically_evaluated_in_phase03_6b"] for row in results),
        "adjudicated_primary_development_rows": len(adjudicated),
        "historical_primary_development_rows": len(historical),
        "best_by_family": family_best,
        "selected": selected,
        "precision_wilson_95": precision_interval,
        "original_vs_adjudicated_selected_rule": {
            "adjudicated": selected,
            "historical_original": historical_by_order[selected["candidate_order"]],
            "primary_metrics_identical_because_changed_row_is_benchmark_impossible": True,
        },
        "remaining_error_patterns": _error_patterns(adjudicated, selected),
        "development_gates": grid["development_gates"],
        "development_gates_passed": gates_pass,
        "weighted_f1_gate_gap": gap,
        "gate_miss_description": "not_missed" if gates_pass else "narrowly_missed" if gap <= 0.02 else "materially_missed",
        "final_automatic_threshold_refinement_iteration": True,
        "test_outcomes_sealed": True,
        "phase4_started": False,
    }

    if not gates_pass:
        artifacts["decision_request"] = _decision_request(root, report)
        report["artifacts"] = artifacts
        report_artifact = _write_json(root / "artifacts/results/phase03_rescue_expanded_development_report.json", report)
        artifacts["development_report"] = report_artifact
        manifest = {
            "schema_version": "phase03-expanded-rescue-artifact-manifest-v1",
            "run_status": "development_gates_failed_final_automatic_refinement_stop",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "execution_timing": "development-only cached-aggregate evaluation; no NLI inference executed",
            "git_commit_sha": subprocess.check_output(
                ["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "expanded_grid_sha256": grid_hash, "candidate_count": len(results),
            "artifacts": artifacts, "test_outcomes_sealed": True, "phase4_started": False,
        }
        _write_json(root / "artifacts/results/phase03_rescue_expanded_artifact_manifest.json", manifest)
        return report

    final_config, final_hash = _final_target_config(selected, grid_hash, grid)
    final_config["final_target_config_sha256"] = final_hash
    artifacts["final_config"] = _write_json(root / "configs/phase03_final_target.json", final_config)
    all_rows, primary_rows, population = _apply_final_target(selected, final_hash, root)
    artifacts["final_labels"] = write_canonical_parquet(
        root / "artifacts/data/context_sufficiency_final_labels.parquet",
        all_rows, tuple(all_rows[0]), ("example_id",),
    )
    artifacts["primary_target"] = write_canonical_parquet(
        root / "artifacts/data/phase03_final_primary_target.parquet",
        primary_rows, tuple(primary_rows[0]), ("example_id",),
    )
    sample = _select_confirmation(all_rows, grid, final_hash, root)
    confirmation_artifacts, confirmation_status = _confirmation_pack(sample, grid, final_hash, root)
    artifacts.update({f"confirmation_{key}": value for key, value in confirmation_artifacts.items()})
    report.update({"final_target_config_sha256": final_hash,
                   "primary_population": population, "confirmation": confirmation_status})
    report["artifacts"] = artifacts
    report_artifact = _write_json(root / "artifacts/results/phase03_rescue_expanded_development_report.json", report)
    artifacts["development_report"] = report_artifact
    manifest = {
        "schema_version": "phase03-expanded-rescue-artifact-manifest-v1",
        "run_status": "development_gates_passed_pending_independent_human_confirmation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_timing": "development-only cached-aggregate evaluation and deterministic exports; no NLI inference executed",
        "git_commit_sha": subprocess.check_output(
            ["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "expanded_grid_sha256": grid_hash,
        "final_target_config_sha256": final_hash,
        "candidate_count": len(results), "artifacts": artifacts,
        "test_outcomes_sealed": True, "phase4_started": False,
    }
    _write_json(root / "artifacts/results/phase03_rescue_expanded_artifact_manifest.json", manifest)
    return report
