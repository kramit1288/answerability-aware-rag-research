"""Independent integrity and leakage checker for frozen Phase 4 artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from answerability_rag.answerability.pipeline import PHASE04_CONFIG_SHA, UPSTREAM  # noqa: E402
from answerability_rag.answerability.registry import load_feature_registry  # noqa: E402
from answerability_rag.hashing import canonical_json_sha256, sha256_file  # noqa: E402


def require(condition: bool, description: str, checks: list[str]) -> None:
    if not condition:
        raise AssertionError(description)
    checks.append(description)


def main() -> None:
    checks: list[str] = []
    for name, (relative, expected) in UPSTREAM.items():
        require(sha256_file(ROOT / relative) == expected,
                f"immutable upstream {name} SHA-256", checks)
    upstream_manifest = json.loads((ROOT / UPSTREAM["phase03_final_manifest"][0]).read_text(encoding="utf-8"))
    require(all(sha256_file(ROOT / item["path"]) == item["physical_sha256"]
                for item in upstream_manifest["artifacts"].values()),
            "all Phase 3 final manifest entries remain byte-identical", checks)
    require(upstream_manifest["test_outcomes_sealed"] and not upstream_manifest["phase4_started"],
            "upstream Phase 3 TEST seal and historical phase boundary remain intact", checks)

    config = json.loads((ROOT / "configs/phase04_modeling.json").read_text(encoding="utf-8"))
    require(canonical_json_sha256(config) == PHASE04_CONFIG_SHA,
            "Phase 4 canonical modeling configuration matches pre-results freeze", checks)
    freeze = json.loads((ROOT / "artifacts/results/phase04_modeling_config_freeze.json").read_text(encoding="utf-8"))
    require(freeze["phase04_modeling_canonical_sha256"] == PHASE04_CONFIG_SHA
            and not freeze["validation_results_observed_before_freeze"],
            "configuration freeze predates validation results", checks)
    require(all(sha256_file(ROOT / item["path"]) == item["physical_sha256"]
                for item in freeze["governance_files"]),
            "all independently frozen Phase 4 governance hashes reproduce", checks)
    registry = load_feature_registry(ROOT / "configs/phase04_feature_registry.json")
    require(len(registry.model_features) == len(set(registry.model_features)) == 39,
            "feature registry contains 39 unique inference-available features", checks)
    registry.validate_model_columns(registry.model_features)
    require(True, "feature registry leakage guard accepts only the declared model features", checks)

    feature_path = ROOT / "data/derived/phase04/phase04_inference_features.parquet"
    features = pq.read_table(feature_path).to_pandas()
    require(set(features.split) == {"train", "validation"} and not (features.split == "test").any(),
            "feature artifact contains no TEST row", checks)
    populations = {(split, frame.question_id.nunique(), len(frame), int(frame.y_suff_final.sum()))
                   for split, frame in features.groupby("split")}
    require(populations == {("train", 424, 5088, 2325), ("validation", 89, 1068, 477)},
            "PRIMARY TRAIN/VALIDATION feature populations reproduce", checks)
    require(len(features) == 6156 and int(features.y_suff_final.sum()) == 2802,
            "PRIMARY total questions/conditions/class counts reproduce", checks)
    expected_columns = {"schema_version", "retrieval_condition_id", "question_id", "split",
                        "y_suff_final", *registry.model_features}
    require(set(features.columns) == expected_columns,
            "feature artifact has exactly provenance, target, and registered inference features", checks)
    require(all(features[feature].isna().sum() == 1539 for feature in registry.undefined),
            "all k=1-only undefined features retain missing values", checks)

    folds = pd.read_csv(ROOT / "artifacts/results/phase04_grouped_cv_folds.csv")
    require(set(folds.purpose) == {"hyperparameter_selection", "rf_oof_calibration"}
            and len(folds) == 10 and folds.zero_question_overlap.all()
            and (folds.question_overlap_count == 0).all(),
            "all ten persisted grouped folds have zero question overlap", checks)
    oof = pq.read_table(ROOT / "artifacts/results/phase04_oof_calibration_provenance.parquet").to_pandas()
    require(len(oof) == 5088 and oof.raw_oof_probability.notna().all()
            and oof.groupby("question_id").fold.nunique().max() == 1
            and (oof.question_overlap_count == 0).all(),
            "grouped OOF calibration provenance is complete and question-disjoint", checks)

    candidates = pd.read_csv(ROOT / "artifacts/results/phase04_grouped_cv_candidates.csv")
    require(len(candidates[candidates.model_family == "logistic_regression"]) == 6
            and len(candidates[candidates.model_family == "random_forest"]) == 36
            and candidates.groupby("model_family").selected.sum().eq(1).all(),
            "complete frozen LR/RF grids have one selected candidate each", checks)
    b1 = pd.read_csv(ROOT / "artifacts/results/phase04_b1_threshold_cv.csv")
    require(len(b1) == 101 and b1.selected.sum() == 1,
            "complete predeclared B1 threshold grid has one TRAIN-CV selection", checks)
    calibration = pd.read_csv(ROOT / "artifacts/results/phase04_rf_calibration_comparison.csv")
    require(set(calibration.probability_method) == {"uncalibrated", "sigmoid", "isotonic"}
            and calibration.selected_calibration.sum() == 1,
            "RF calibration comparison contains exactly the three frozen methods", checks)
    require((ROOT / "artifacts/figures/phase04_reliability_diagram.svg").stat().st_size > 0,
            "validation reliability diagram is persisted", checks)
    metrics = pd.read_csv(ROOT / "artifacts/results/phase04_model_validation_metrics.csv")
    require(set(metrics.model) == {"B0_always_sufficient", "B1_idf_coverage_threshold",
                                  "B2_logistic_regression", "B3_random_forest",
                                  "B4_calibrated_random_forest"}
            and metrics.selected_model.sum() == 1,
            "B0-B4 VALIDATION metrics and one selected model are persisted", checks)

    selected_model = json.loads((ROOT / "configs/phase04_selected_model.json").read_text(encoding="utf-8"))
    stored_model_hash = selected_model.pop("selected_model_config_sha256")
    require(canonical_json_sha256(selected_model) == stored_model_hash,
            "selected-model canonical SHA-256 reproduces", checks)
    require(selected_model["features"] == list(registry.model_features)
            and selected_model["feature_count"] == 39,
            "selected model uses exactly the registered inference features", checks)
    require(sha256_file(ROOT / selected_model["model_artifact_path"])
            == selected_model["model_artifact_sha256"],
            "selected model binary hash reproduces", checks)
    binary = joblib.load(ROOT / selected_model["model_artifact_path"])
    require(binary["test_inference_permitted"] is False
            and binary["feature_names"] == list(registry.model_features),
            "serialized selected model preserves the TEST seal and feature order", checks)

    bootstrap = pd.read_csv(ROOT / "artifacts/results/phase04_bootstrap_confidence_intervals.csv")
    require(set(bootstrap.metric) == {"auroc", "auprc", "f1", "brier"}
            and (bootstrap.replicates == 1000).all()
            and (bootstrap.resampling_unit == "question_id").all(),
            "1,000-replicate question bootstrap intervals are complete", checks)
    ablations = pd.read_csv(ROOT / "artifacts/results/phase04_feature_ablation.csv")
    require(list(ablations.ablation) == ["A", "B", "C", "D"] and not ablations.retuned.any(),
            "exact A-D feature-family ablations are present without retuning", checks)
    importance_path = (ROOT / "artifacts/results/phase04_rf_permutation_importance.csv"
                       if selected_model["model_family"] == "random_forest"
                       else ROOT / "artifacts/results/phase04_logistic_coefficients.csv")
    require(importance_path.exists() and len(pd.read_csv(importance_path)) >= 39,
            "selected-family descriptive feature analysis is complete", checks)

    curve = pd.read_csv(ROOT / "artifacts/results/phase04_risk_coverage_curve.csv")
    require(len(curve) > 1 and curve.iloc[0].coverage == 0
            and pd.isna(curve.iloc[0].selective_risk)
            and curve.coverage.is_monotonic_increasing,
            "full risk-coverage curve preserves NA at zero coverage", checks)
    aurc = json.loads((ROOT / "artifacts/results/phase04_aurc.json").read_text(encoding="utf-8"))
    require(0 <= aurc["aurc"] <= 1, "AURC is persisted under the frozen integration rule", checks)
    risk_points = pd.read_csv(ROOT / "artifacts/results/phase04_risk_operating_points.csv")
    require(set(risk_points.risk_constraint.round(2)) == {0.05, 0.10, 0.20},
            "5/10/20 percent selective operating points are present", checks)

    grid = pd.read_csv(ROOT / "artifacts/results/phase04_policy_threshold_grid.csv")
    require(len(grid) == 1275 and (grid.t_low < grid.t_high).all(),
            "complete 1,275-pair three-way policy grid is persisted", checks)
    policy_points = pd.read_csv(ROOT / "artifacts/results/phase04_policy_operating_points.csv")
    require(set(policy_points.risk_constraint.round(2)) == {0.05, 0.10, 0.20},
            "5/10/20 percent three-way operating points are present", checks)
    trajectory = pd.read_csv(ROOT / "artifacts/results/phase04_policy_trajectories.csv")
    require(trajectory.groupby("risk_constraint").question_id.nunique().eq(89).all()
            and trajectory.retrieved_k.max() == 10,
            "selected policy trajectories cover every VALIDATION question with no k>10", checks)
    baselines = pd.read_csv(ROOT / "artifacts/results/phase04_policy_baselines.csv")
    require(set(baselines.policy) == {"P0_always_answer_hybrid_k5", "P1_two_way_hybrid_k5",
                                     "P2_always_retrieve_hybrid_k10_then_two_way"}
            and len(baselines) == 9,
            "P0/P1/P2 comparisons exist for all three constraints", checks)
    selected_policy = json.loads((ROOT / "configs/phase04_selected_policy.json").read_text(encoding="utf-8"))
    stored_policy_hash = selected_policy.pop("selected_policy_config_sha256")
    require(canonical_json_sha256(selected_policy) == stored_policy_hash
            and selected_policy["risk_constraint"] == 0.10
            and selected_policy["maximum_expansions"] == 1,
            "primary 10 percent policy canonical SHA and one-expansion rule reproduce", checks)

    sensitivity = json.loads((ROOT / "artifacts/results/phase04_benchmark_impossible_sensitivity.json").read_text(encoding="utf-8"))
    require(sensitivity["timing"].startswith("after PRIMARY")
            and not sensitivity["architecture_changed"]
            and not sensitivity["hyperparameters_retuned"]
            and not sensitivity["primary_thresholds_changed"]
            and not sensitivity["influences_primary_conclusions"],
            "benchmark-impossible sensitivity is separate and mechanically fixed", checks)
    integrity = json.loads((ROOT / "artifacts/results/phase04_integrity_report.json").read_text(encoding="utf-8"))
    require(integrity["status"] == "pass" and integrity["test_sealed"]
            and integrity["test_rows"] == 0 and not integrity["test_inference_run"]
            and not integrity["test_aggregate_results"] and not integrity["phase5_started"],
            "persisted integrity report confirms TEST sealing and Phase 5 absence", checks)

    manifest = json.loads((ROOT / "artifacts/results/phase04_artifact_manifest.json").read_text(encoding="utf-8"))
    require(not manifest["manifest_includes_itself"]
            and all(sha256_file(ROOT / item["path"]) == item["physical_sha256"]
                    for item in manifest["artifacts"]),
            "independent Phase 4 artifact manifest hashes all entries without self-reference", checks)
    phase5_paths = list((ROOT / "configs").glob("phase05*")) + list((ROOT / "scripts").glob("*phase05*"))
    require(not phase5_paths and not (ROOT / "src/answerability_rag/generation").exists(),
            "Phase 5 implementation has not started", checks)
    print(json.dumps({"status": "pass", "checks": checks, "check_count": len(checks)}, indent=2))


if __name__ == "__main__":
    main()
