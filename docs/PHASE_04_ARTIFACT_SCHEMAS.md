# Phase 4 Artifact Schemas

## Serialization rules

Phase 4 JSON is UTF-8, sorted by key, indented by two spaces, permits no NaN token, and ends with LF. CSV is UTF-8 with LF, stable declared column order, and empty fields for missing values. Parquet uses the Phase 2 deterministic PyArrow serialization configuration. Every persisted artifact is hashed by physical SHA-256 in the independent Phase 4 manifest.

## Governance and registry

`configs/phase04_feature_registry.json` classifies every model-relevant field into exactly one of `inference_available_feature`, `target_only`, `label_construction_only`, `provenance_only`, or `evaluation_only`. It contains feature families, numeric/categorical lists, undefined features, forbidden aliases, and the identifier regex.

`artifacts/results/phase04_modeling_config_freeze.json` records the canonical configuration SHA-256, physical hashes of all Phase 4 governance files, upstream immutable hashes, freeze time, and the rule that it predates VALIDATION performance.

## Feature artifacts

`data/derived/phase04/phase04_inference_features.parquet` has one row per eligible TRAIN or VALIDATION retrieval condition. Required provenance columns are `question_id`, `split`, `retrieval_strategy`, `k`, and `retrieval_condition_id`; the target column is `y_suff_final`; the remaining model columns are exactly the registered 39 inference features. Mathematically undefined continuous values are null. No question text, context text, benchmark answer, gold evidence, or Phase 3 label-construction field is exported.

`artifacts/results/phase04_feature_manifest.json` records row/question/class counts by split, column schema, registry and configuration hashes, source hashes, missing counts, semantic model revision, and physical feature-artifact SHA-256.

## Model-development artifacts

- `phase04_grouped_cv_folds.csv`: model/fold identifiers; training and held-out question/condition/class counts; overlap count and pass flag.
- `phase04_grouped_cv_candidates.csv`: model family, complete parameter values, mean fold AUPRC/AUROC, and selected flag.
- `phase04_b1_threshold_cv.csv`: every threshold and fold/mean F1 and precision, with selected flag.
- `phase04_model_validation_metrics.csv`: B0–B4 validation metrics, probability method, decision threshold, and selection flags; unsupported constant-score metrics are empty.
- `phase04_rf_calibration_comparison.csv`: uncalibrated/sigmoid/isotonic RF metrics with Brier/ECE selection evidence.
- `phase04_oof_calibration_provenance.parquet`: TRAIN row identifiers, fold, raw OOF probability, target, and zero-group-overlap provenance.
- `phase04_reliability_bins.csv`: model/method/bin bounds, count, mean probability, positive fraction, and ECE contribution.
- `phase04_bootstrap_confidence_intervals.csv`: selected model metric, point estimate, replicate count, lower and upper percentile bounds, seed, and resampling unit.
- `phase04_selected_model.joblib`: fitted preprocessing, predictor, and optional calibration mapping; it contains no VALIDATION target.
- `configs/phase04_selected_model.json`: frozen selected family, features, parameters, calibration, preprocessing, seed, source hashes, binary hash, and canonical selected-model configuration SHA-256.

## Interpretation and ablation artifacts

Exactly one primary feature-analysis table is populated: `phase04_logistic_coefficients.csv` for Logistic Regression or `phase04_rf_permutation_importance.csv` for Random Forest. The unused alternative is absent. `phase04_feature_ablation.csv` contains A–D, declared families/features, fixed selected hyperparameters, and VALIDATION AUROC/AUPRC/F1.

## Selective prediction artifacts

`phase04_risk_coverage_curve.csv` stores threshold, answered count, total count, coverage, unsafe answered count, and selective risk for every full-curve point. `phase04_aurc.json` records the integration convention and value. `phase04_risk_operating_points.csv` stores each risk constraint, feasibility, selected threshold, coverage, and observed risk.

## Three-way policy artifacts

`phase04_policy_threshold_grid.csv` stores every valid low/high pair for every risk-constraint assessment, including counts and all policy metrics. `phase04_policy_operating_points.csv` contains the lexicographically selected pair for 5%, 10%, and 20%. `phase04_policy_trajectories.csv` contains one VALIDATION question per selected operating point, p5/p10 where used, frozen k5/k10 targets, expansion flag, final action, safety, and retrieved depth. `phase04_policy_baselines.csv` contains P0/P1/P2 metrics under the same constraints.

`configs/phase04_selected_policy.json` freezes the 10% illustrative policy, selected-model configuration SHA, feature/configuration hashes, calibration, low/high thresholds, exact decision rule, cost convention, and canonical policy-configuration SHA-256.

## Sensitivity and integrity artifacts

`phase04_benchmark_impossible_sensitivity.json` records the mechanically retrained fixed architecture, added TRAIN population, relevant VALIDATION populations, metrics, and explicit non-influence on PRIMARY selection.

`artifacts/results/phase04_artifact_manifest.json` is an independent manifest of Phase 4 files and immutable upstream dependency hashes. It does not include itself and does not alter the Phase 3 manifest. `phase04_integrity_report.json` records checker assertions, artifact counts/hashes, population checks, zero leakage, zero grouped-CV overlap, TEST sealing, and Phase 5 absence.
