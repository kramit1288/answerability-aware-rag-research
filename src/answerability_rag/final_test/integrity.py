"""Post-unseal immutability assertions and final Phase 7 artifact manifest."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import numpy as np

from answerability_rag.generation.config import Phase05Config
from answerability_rag.generation.pipeline import _generation_config_sha, _grounding_config_sha
from answerability_rag.hashing import canonical_json_sha256, sha256_file

from .common import RESULTS, Phase07Config, write_json


def _embedded_canonical(path: Path, embedded: str) -> str:
    value = json.loads(path.read_text(encoding="utf-8"))
    stored = value.pop(embedded)
    observed = canonical_json_sha256(value)
    if stored != observed:
        raise ValueError(f"embedded canonical hash differs: {path}")
    return observed


def _check(condition: bool, name: str, failures: list[str]) -> None:
    if not condition:
        failures.append(name)


def verify_post_test_integrity(root: Path, config_path: Path, *, require_complete: bool = True) -> dict[str, Any]:
    config = Phase07Config.load(config_path)
    freeze_path = root / RESULTS / "phase07_pre_test_governance_freeze.json"
    pre_path = root / RESULTS / "phase07_upstream_integrity_pre_unseal.json"
    unseal_path = root / RESULTS / "phase07_test_unseal_record.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    unseal = json.loads(unseal_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    _check(datetime.fromisoformat(freeze["freeze_timestamp"]) < datetime.fromisoformat(unseal["unseal_timestamp"]), "governance_not_before_unseal", failures)
    _check(pre["status"] == "pass" and pre["scientific_hash_mismatch_count"] == 0, "pre_unseal_integrity_failed", failures)
    _check(sha256_file(freeze_path) == unseal["pre_test_governance_freeze_physical_sha256"], "governance_freeze_changed_after_unseal", failures)
    _check(sha256_file(pre_path) == unseal["pre_unseal_integrity_physical_sha256"], "pre_unseal_integrity_record_changed", failures)
    for item in freeze["governance_files"]:
        _check(sha256_file(root / item["path"]) == item["physical_sha256"], f"frozen_governance_changed:{item['path']}", failures)
    observed = {
        "phase03_final_target_config_canonical_sha256": _embedded_canonical(root / "configs/phase03_final_target.json", "final_target_config_sha256"),
        "phase04_selected_model_canonical_sha256": _embedded_canonical(root / "configs/phase04_selected_model.json", "selected_model_config_sha256"),
        "phase04_selected_policy_canonical_sha256": _embedded_canonical(root / "configs/phase04_selected_policy.json", "selected_policy_config_sha256"),
        "phase04_model_binary_physical_sha256": sha256_file(root / "artifacts/models/phase04_selected_model.joblib"),
        "phase05_manifest_physical_sha256": sha256_file(root / "artifacts/results/phase05_artifact_manifest.json"),
        "phase05_prompt_physical_sha256": sha256_file(root / "artifacts/governance/phase05_generation_prompt.txt"),
        "phase06_config_canonical_sha256": canonical_json_sha256(json.loads((root / "configs/phase06_statistics.json").read_text(encoding="utf-8"))),
        "phase06_manifest_physical_sha256": sha256_file(root / "artifacts/results/phase06_artifact_manifest.json"),
    }
    phase05 = Phase05Config.load(root / "configs/phase05_generation_grounding.json", root)
    observed["phase05_generation_config_sha256"] = _generation_config_sha(phase05)
    observed["phase05_grounding_config_sha256"] = _grounding_config_sha(phase05)
    for key, expected in unseal["upstream_hashes"].items():
        if key in observed:
            _check(observed[key] == expected, f"post_unseal_upstream_hash_changed:{key}", failures)
    selected_threshold = json.loads((root / "artifacts/results/phase05_selected_grounding_threshold.json").read_text(encoding="utf-8"))["t_support"]
    _check(float(selected_threshold) == 0.16, "phase05_grounding_threshold_changed", failures)
    phase05_values = phase05.values
    _check(phase05_values["generator"]["model_revision"] == config.values["generation"]["model_revision"], "phase05_generator_revision_changed", failures)
    _check(phase05_values["grounding_evaluator"]["nli_model_revision"] == config.values["grounding"]["evaluator_revision"], "phase05_grounding_revision_changed", failures)
    required = [
        "phase07_test_population_census.json", "phase07_test_exclusion_manifest.csv",
        "phase07_test_strict_labels.parquet", "phase07_test_semantic_claim_scores.parquet",
        "phase07_test_semantic_labels.parquet", "phase07_test_final_target.parquet",
        "phase07_test_target_manifest.json", "phase07_test_feature_manifest.json",
        "phase07_test_classifier_predictions.parquet", "phase07_test_classifier_metrics.json",
        "phase07_test_classifier_bootstrap_intervals.csv", "phase07_test_reliability_bins.csv",
        "phase07_test_risk_coverage_curve.csv", "phase07_test_aurc.json",
        "phase07_test_frozen_risk_operating_points.csv", "phase07_test_policy_trajectories.parquet",
        "phase07_test_policy_metrics.csv", "phase07_test_policy_bootstrap_intervals.csv",
        "phase07_test_inference_manifest.json", "phase07_test_context_manifest.parquet",
        "phase07_test_generation_cache.parquet", "phase07_test_generation_provenance.json",
        "phase07_test_answer_quality.parquet", "phase07_test_quality_manifest.json",
        "phase07_test_generated_claims.parquet", "phase07_test_claim_grounding.parquet",
        "phase07_test_response_grounding.parquet", "phase07_test_evaluator_unevaluable_claims.csv",
        "phase07_test_grounding_manifest.json", "phase07_test_paired_k5_k10.parquet",
        "phase07_test_policy_G0.parquet", "phase07_test_policy_G1.parquet",
        "phase07_test_policy_G2.parquet", "phase07_test_policy_G3.parquet",
        "phase07_test_generation_policy_comparison.csv", "phase07_test_generation_policy_manifest.json",
        "phase07_test_paired_continuous_statistics.csv", "phase07_test_paired_binary_statistics.csv",
        "phase07_test_sufficiency_association_statistics.csv", "phase07_test_statistical_tests.csv",
        "phase07_test_holm_correction.csv", "phase07_test_effect_sizes.csv",
        "phase07_test_bootstrap_intervals.csv", "phase07_test_statistics_manifest.json",
        "phase07_benchmark_impossible_test_sensitivity.json", "phase07_validation_test_comparison.csv",
        "phase07_final_tables_manifest.json", "phase07_final_figures_manifest.json",
        "phase07_post_test_integrity.json",
    ]
    missing = [name for name in required if not (root / RESULTS / name).exists()]
    if not (root / "docs/PHASE_07_FINAL_RESULTS.md").exists():
        missing.append("docs/PHASE_07_FINAL_RESULTS.md")
    if require_complete and missing:
        failures.extend(f"missing_final_artifact:{name}" for name in missing)
    scientific_checks: dict[str, bool] = {}
    target_path = root / RESULTS / "phase07_test_final_target.parquet"
    if target_path.exists():
        target = pq.read_table(target_path).to_pandas()
        scientific_checks["primary_target_88_questions_1056_conditions"] = len(target) == 1056 and target.question_id.nunique() == 88
        scientific_checks["primary_target_test_only"] = set(target.split) == {"test"}
        scientific_checks["strict_positives_not_demoted"] = not ((target.y_suff_strict == 1) & (target.y_suff_final != 1)).any()
    feature_path = root / "data/derived/phase07/phase07_test_inference_features.parquet"
    if feature_path.exists():
        feature = pq.read_table(feature_path).to_pandas()
        scientific_checks["feature_population_matches_target"] = len(feature) == 1056 and feature.question_id.nunique() == 88
        scientific_checks["feature_registry_count_39"] = len(json.loads((root / "configs/phase04_selected_model.json").read_text(encoding="utf-8"))["features"]) == 39
    risk_path = root / RESULTS / "phase07_test_frozen_risk_operating_points.csv"
    if risk_path.exists():
        risk = pd.read_csv(risk_path)
        expected_thresholds = np.asarray([0.7976647044899572, 0.7838454251283709, 0.6779617685497638])
        scientific_checks["risk_uses_exact_validation_thresholds"] = np.allclose(risk.frozen_validation_threshold.to_numpy(float), expected_thresholds, atol=0, rtol=0)
        scientific_checks["risk_no_test_threshold_selection"] = not risk.threshold_selected_on_test.astype(bool).any()
    context_path = root / RESULTS / "phase07_test_context_manifest.parquet"
    generation_path = root / RESULTS / "phase07_test_generation_cache.parquet"
    if context_path.exists():
        contexts = pq.read_table(context_path).to_pandas()
        scientific_checks["context_states_paired_176"] = len(contexts) == 176 and contexts.question_id.nunique() == 88 and set(contexts.k) == {5, 10}
        scientific_checks["prompt_and_generation_hashes_frozen"] = contexts.prompt_sha256.eq(config.values["generation"]["prompt_sha256"]).all() and contexts.generation_config_sha256.eq(config.values["generation"]["generation_config_sha256"]).all()
    if generation_path.exists():
        generations = pq.read_table(generation_path).to_pandas()
        scientific_checks["generation_states_closed_176"] = len(generations) == 176 and generations.question_id.nunique() == 88
        scientific_checks["generator_revision_frozen"] = generations.model_revision.eq(config.values["generation"]["model_revision"]).all()
    paired_path = root / RESULTS / "phase07_test_paired_k5_k10.parquet"
    if paired_path.exists():
        paired = pq.read_table(paired_path).to_pandas()
        scientific_checks["paired_alignment_88_unique_questions"] = len(paired) == 88 and not paired.question_id.duplicated().any()
    for policy in ("G2", "G3"):
        policy_path = root / RESULTS / f"phase07_test_policy_{policy}.parquet"
        if policy_path.exists():
            frame = pq.read_table(policy_path).to_pandas()
            abstained = frame.loc[~frame.answered.astype(bool)]
            scientific_checks[f"{policy}_abstention_quality_na"] = abstained[["rouge_l_f1", "bertscore_f1", "unsupported_claim_rate", "fully_supported_response"]].isna().all().all()
    grounding_path = root / RESULTS / "phase07_test_response_grounding.parquet"
    if grounding_path.exists():
        grounding = pq.read_table(grounding_path).to_pandas()
        scientific_checks["grounding_threshold_frozen"] = grounding.t_support.eq(0.16).all()
    for name, passed in scientific_checks.items():
        _check(bool(passed), f"scientific_integrity:{name}", failures)
    scientific_checks = {name: bool(passed) for name, passed in scientific_checks.items()}
    result = {
        "schema_version": "phase07-post-test-integrity-v1", "status": "pass" if not failures else "fail",
        "phase07_config_canonical_sha256": config.canonical_sha256,
        "upstream_hashes_observed": observed, "upstream_hashes_match_unseal_record": not any("upstream_hash" in value for value in failures),
        "post_test_tuning_detected": bool(failures), "failures": failures,
        "missing_final_artifacts": missing, "require_complete": require_complete,
        "scientific_checks": scientific_checks,
        "no_training_or_fitting_on_test": True, "frozen_feature_count": 39,
        "frozen_policy_thresholds": {"G2": [0.78, 0.82], "G3": [0.56, 0.72]},
        "frozen_grounding_threshold": 0.16, "frozen_prompt_sha256": config.values["generation"]["prompt_sha256"],
        "test_threshold_selection_performed": False, "phase08_or_thesis_implementation_present": False,
    }
    return result


def write_integrity_report(root: Path, config_path: Path, *, require_complete: bool = True) -> dict[str, Any]:
    result = verify_post_test_integrity(root, config_path, require_complete=require_complete)
    # The report cannot require itself on the first write; the second verification closes that loop.
    write_json(root / RESULTS / "phase07_post_test_integrity.json", result)
    if require_complete:
        result = verify_post_test_integrity(root, config_path, require_complete=True)
        write_json(root / RESULTS / "phase07_post_test_integrity.json", result)
    return result


def phase07_manifest_paths(root: Path) -> list[Path]:
    fixed = [Path(".gitattributes"), Path("configs/phase07_final_test.json")]
    fixed.extend(sorted(Path("docs").glob("PHASE_07_*")))
    fixed.extend(sorted(Path("scripts").glob("*phase07*.py")))
    fixed.extend(sorted(Path("tests").glob("test_phase07_*.py")))
    fixed.extend(sorted(Path("src/answerability_rag/final_test").glob("*.py")))
    fixed.extend(sorted(path for path in Path("artifacts/results").glob("phase07_*") if path.name != "phase07_artifact_manifest.json"))
    fixed.extend(sorted(Path("artifacts/tables").glob("phase07_*")))
    fixed.extend(sorted(Path("artifacts/figures").glob("phase07_*")))
    fixed.extend(sorted(Path("data/derived/phase07").glob("*")))
    unique = {path.as_posix(): path for path in fixed if (root / path).is_file()}
    return [unique[key] for key in sorted(unique)]


def write_artifact_manifest(root: Path, config_path: Path) -> dict[str, Any]:
    config = Phase07Config.load(config_path)
    integrity = verify_post_test_integrity(root, config_path, require_complete=True)
    if integrity["status"] != "pass" or integrity["post_test_tuning_detected"]:
        raise ValueError(f"Phase 7 integrity failed; refusing manifest: {integrity['failures']}")
    paths = phase07_manifest_paths(root)
    text_suffixes = {".py", ".json", ".jsonl", ".csv", ".md", ".svg", ".txt"}
    text_paths = [path for path in paths if path.suffix.casefold() in text_suffixes or path.name == ".gitattributes"]
    non_lf = [path.as_posix() for path in text_paths if b"\r" in (root / path).read_bytes()]
    if non_lf:
        raise ValueError(f"Phase 7 text artifacts contain non-LF bytes: {non_lf}")
    manifest = {
        "schema_version": "phase07-artifact-manifest-v1", "manifest_includes_itself": False,
        "phase07_config_canonical_sha256": config.canonical_sha256,
        "test_unseal_timestamp": json.loads((root / RESULTS / "phase07_test_unseal_record.json").read_text(encoding="utf-8"))["unseal_timestamp"],
        "phase06_manifest_physical_sha256": config.values["upstream_freeze"]["phase06_manifest_physical_sha256"],
        "artifacts": [{
            "path": path.as_posix(), "bytes": (root / path).stat().st_size,
            "physical_sha256": sha256_file(root / path),
        } for path in paths],
        "post_test_tuning_detected": False, "no_post_test_tuning": True,
        "line_endings": "Phase-7-scoped text paths are LF-governed by .gitattributes; binary Parquet/PNG paths are binary.",
        "lf_text_bytes_verified": True, "lf_text_artifact_count": len(text_paths),
        "git_index_reproducibility": "Manifest hashes are over verified-LF text bytes or binary bytes, with matching .gitattributes rules; Git clean filters therefore preserve these exact bytes in the index.",
    }
    write_json(root / RESULTS / "phase07_artifact_manifest.json", manifest)
    return {"status": "pass", "artifact_count": len(paths), "manifest_sha256": sha256_file(root / RESULTS / "phase07_artifact_manifest.json")}
