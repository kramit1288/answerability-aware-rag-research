# Phase 7 Artifact Schemas

## Serialization

JSON is sorted-key UTF-8, two-space indented, rejects NaN, and ends with LF. CSV uses stable
columns, empty missing fields, and LF. Parquet uses the frozen deterministic Phase 2 writer.
Every final artifact has a physical SHA-256 in the independent Phase 7 manifest. Configuration-
like objects also record canonical JSON SHA-256. The manifest never contains itself.

## Governance and unseal

- `phase07_pre_test_governance_freeze.json`: Phase 7 canonical config SHA, physical hashes of the
  four governance files, declared upstream hashes, freeze timestamp, explicit absence of TEST
  scientific access, and the no-post-TEST-choice statement.
- `phase07_upstream_integrity_pre_unseal.json`: every expected/observed physical or canonical hash,
  model/prompt/evaluator revision, tag/commit/branch assertion, pass status, and checked timestamp.
- `phase07_test_unseal_record.json`: immutable unseal timestamp, current Git commit/branch,
  upstream tag, Phase 7 SHA, verified hashes, prior-use declaration, and no-tuning declaration.

## Population and target

- `phase07_test_population_census.json`: total TEST questions; benchmark-answerable/impossible;
  primary eligible/excluded questions; eligible conditions; reason counts.
- `phase07_test_exclusion_manifest.csv`: one TEST question with benchmark status, alignment status,
  semantic status, primary eligibility, exclusion reason, condition count, and sensitivity role.
- `phase07_test_strict_labels.parquet`: all TEST condition identities and frozen Phase 3 strict
  label/provenance/coverage diagnostics, including null unresolved rows.
- `phase07_test_semantic_claim_scores.parquet`: each evaluable reference claim and its frozen
  selected-premise entailment/neutral/contradiction evidence.
- `phase07_test_semantic_labels.parquet`: each benchmark-answerable aligned TEST condition with
  semantic status, budget metadata, semantic aggregates, semantic label, and provenance.
- `phase07_test_final_target.parquet`: only PRIMARY eligible TEST conditions; exactly one immutable
  `y_suff_final` per question/strategy/k plus strict/semantic/rescue provenance and config hashes.
- `phase07_test_target_manifest.json`: hashes, row/question counts, exclusions, immutable flag,
  frozen rule/revisions, and class distribution calculated only after target closure.

## Features, predictions, classifier, risk, and policy

- `phase07_test_inference_features.parquet`: provenance, target, and exactly the registered 39
  inference features; no forbidden gold/answer/semantic construction fields enter the model
  matrix.
- `phase07_test_feature_manifest.json`: schema, feature registry, source hashes, count/missingness,
  feature physical/semantic hashes, and explicit no-fit assertions.
- `phase07_test_classifier_predictions.parquet`: condition identity, target, frozen probability,
  0.5 prediction/correctness, selected-model/config/binary hashes.
- `phase07_test_classifier_metrics.json`: all requested point estimates and denominators.
- `phase07_test_classifier_bootstrap_intervals.csv`: AUROC/AUPRC/F1/Brier point, bounds,
  requested/valid replicates, question unit, seed.
- `phase07_test_reliability_bins.csv`: ten bin bounds/counts, mean probability, positive fraction,
  gap and ECE contribution.
- `phase07_test_risk_coverage_curve.csv`: every tied-threshold curve point with counts, coverage,
  risk, and AURC convention.
- `phase07_test_aurc.json`: TEST AURC and integration convention.
- `phase07_test_frozen_risk_operating_points.csv`: exact frozen validation threshold, TEST answered
  count/coverage/unsafe count/risk for each 5/10/20 declaration; no TEST-selected threshold.
- `phase07_test_policy_trajectories.parquet`: one primary TEST question per frozen policy with
  p5/p10, y5/y10, actions, expansion, final k/action/target, safe/unsafe, and thresholds.
- `phase07_test_policy_metrics.csv`: counts, coverage, selective risk, false abstention, expansion,
  mean depth, and retrieval cost for the 10% primary and 20% sensitivity.
- `phase07_test_policy_bootstrap_intervals.csv`: important policy proportions with question-
  bootstrap bounds and numerator/denominator definitions.

## Contexts, generation, quality, and grounding

- `phase07_test_context_manifest.parquet`: one primary question/k state with ordered retrieval and
  prompt-visible chunks, exact context/prompt hashes, input budget diagnostics, and frozen config
  hashes.
- `phase07_test_generation_cache.parquet`: one state/cache identity with raw/normalized output,
  output hash/tokens, status, attempts, runtime, and failure provenance.
- `phase07_test_generation_provenance.json`: closed-generation hash, state/status counts, exact
  model/prompt/config/context revisions, and reference-not-yet-accessed declaration.
- `phase07_test_answer_quality.parquet`: per generated state reference status, ROUGE-L F1,
  BERTScore F1, metric status, and closed generation hash.
- `phase07_test_generated_claims.parquet`: deterministic claims and source offsets.
- `phase07_test_claim_grounding.parquet`: candidate identities/similarities, candidate NLI scores,
  support/contradiction, frozen threshold, predicted unsupported, and evaluator status.
- `phase07_test_response_grounding.parquet`: claim/evaluable/unevaluable counts, unsupported rate,
  fully-supported status, mean/minimum support, maximum contradiction, and frozen target metadata.
- `phase07_test_evaluator_unevaluable_claims.csv`: every unevaluable claim and reason.
- `phase07_test_paired_k5_k10.parquet`: aligned k5/k10 quality, grounding, token outcomes and
  predeclared differences.
- `phase07_test_policy_G0.parquet` through `phase07_test_policy_G3.parquet`: one eligible question,
  selected state/action, coverage denominator, answer fields, and NA content fields for abstention.
- `phase07_test_generation_policy_comparison.csv`: G0-G3 coverage, conditional quality/grounding,
  grounded yield, unsupported population exposure, and denominators.

## Inferential, sensitivity, and generalization outputs

- `phase07_test_paired_continuous_statistics.csv`, `phase07_test_paired_binary_statistics.csv`,
  `phase07_test_sufficiency_association_statistics.csv`: TEST equivalents of the Phase 6 fixed
  schemas and estimands.
- `phase07_test_statistical_tests.csv`, `phase07_test_holm_correction.csv`,
  `phase07_test_effect_sizes.csv`, and `phase07_test_bootstrap_intervals.csv`: normalized TEST
  inference, family-wise Holm results, numerical effects, and intervals.
- `phase07_benchmark_impossible_test_sensitivity.json`: separate frozen-sensitivity census,
  predicted-sufficient rate, allowed metrics, and contamination limitation.
- `phase07_validation_test_comparison.csv`: frozen VALIDATION and TEST estimates, intervals,
  descriptive generalization classification, and no-tuning flag.

## Final evidence, tables, figures, and integrity

- `docs/PHASE_07_FINAL_RESULTS.md`: generated/post-results RQ1-RQ4 evidence with validation and
  TEST effects/CIs, directional replication, and limitations; not a thesis chapter rewrite.
- `artifacts/tables/phase07_table*.csv/.md`: final classifier, risk/coverage, policy,
  quality/grounding, sufficiency-association, and inferential tables generated from artifacts.
- `artifacts/figures/phase07_figure*_data.csv/.svg/.png`: only material TEST-aware figures, each
  rendered from its persisted data CSV.
- `phase07_post_test_integrity.json`: upstream re-verification, no-fit/no-tuning assertions,
  `post_test_tuning_detected`, population/NA/pairing checks, and status.
- `phase07_artifact_manifest.json`: every Phase 7 governance, code, test, scientific result,
  table, and figure artifact except itself, hashed from final Git-index bytes.
