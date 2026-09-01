# Phase 5 Artifact Schemas

## Serialization

JSON is UTF-8, sorted-key, two-space indented, rejects NaN, and ends with LF. CSV is UTF-8 with LF, stable declared columns, and empty fields for missing values. Parquet follows the frozen Phase 2 deterministic PyArrow settings. Physical SHA-256 is recorded for every Phase 5 artifact; configuration-like JSON also records canonical JSON SHA-256.

## Governance and generator

- `phase05_pre_results_governance_freeze.json`: canonical configuration SHA, physical hashes of all pre-results governance files, upstream hashes, environment, and proof that TechQA results were not observed before freeze.
- `phase05_generation_model_manifest.json`: model/tokenizer identifiers and revisions, local critical-file hashes, class, device/dtype, context window, package versions, and synthetic smoke result.
- `phase05_generation_config_manifest.json`: prompt/config hashes, decoding parameters, cache-key definition, and output-normalization rule.
- `phase05_generation_prompt.txt`: exact two-placeholder prompt text.

## TechQA contexts and generation

`phase05_techqa_context_manifest.parquet` has one row per question/k state. Core fields are `response_id, question_id, split, retrieval_strategy, k, ordered_retrieved_chunk_ids_json, prompt_visible_chunk_ids_json, assembled_context_sha256, prompt_sha256, input_token_count, fully_included_chunk_count, final_truncated_chunk_id, final_truncated_chunk_original_tokens, final_truncated_chunk_included_tokens, retrieved_chunks_not_included, k5_context_prefix_sha256, exposes_context_beyond_k5, generation_config_sha256`.

`phase05_generation_cache.parquet` has one row per state: all cache-key fields plus `raw_generated_text, normalized_generated_text, input_token_count, output_token_count, generation_status, attempt_count, runtime_seconds, runtime_metadata_json, output_sha256`. Failures store an error class/message hash and null text, never invented empty text.

`phase05_generation_provenance.json` records counts/statuses, resumptions/retries, context-utilization distributions, exact input hashes, and the closed-generation artifact SHA before benchmark-answer evaluation.

## Answer quality

`phase05_answer_quality.parquet` has one row per generated state: `response_id, question_id, k, generation_status, reference_status, rouge_l_f1, bertscore_f1, metric_status, generation_cache_sha256`. Reference metrics are null when undefined. `phase05_quality_manifest.json` freezes implementations, revisions, settings, counts, and hashes.

## RAGTruth grounding validation

- `phase05_grounding_evaluator_config.json`: exact MiniLM/NLI revisions, candidate count, token budget, segmentation, aggregation, and threshold rules.
- `phase05_ragtruth_schema_alignment_audit.json`: actual fields, census, official split/source isolation, quality counts, label counts, and offset-alignment result.
- `phase05_ragtruth_claim_scores.parquet`: response/source/split metadata; claim offsets/text; human overlap label; candidate passage IDs; entailment/contradiction per candidate; maximum support; unsupportedness; evaluator status.
- `phase05_ragtruth_threshold_search.csv`: all 51 TRAIN-only thresholds with confusion counts, unsupported precision/recall/F1, selection keys, and selected flag.
- `phase05_selected_grounding_threshold.json`: selected threshold, TRAIN input hash, selection rule, and explicit absence of TEST influence.
- `phase05_ragtruth_test_metrics.csv`: claim/response metric point estimates for primary and sensitivity populations.
- `phase05_ragtruth_bootstrap_intervals.csv`: metric, level, point estimate, percentile bounds, source resampling, seed, requested/valid replicates.
- `phase05_grounding_validation_manifest.json`: input/output hashes, populations, exclusions, evaluator validity interpretation, and no-retuning assertion.

## TechQA grounding

`phase05_techqa_generated_claims.parquet` stores `response_id, question_id, k, claim_id, claim_index, claim_text, claim_start, claim_end, segmentation_version`.

`phase05_techqa_claim_grounding.parquet` adds candidate chunk IDs/similarities, per-candidate NLI scores, maximum entailment/contradiction, supporting chunk ID, frozen threshold, predicted unsupported flag, and evaluator status.

`phase05_techqa_response_grounding.parquet` has one row per state with `claim_count, evaluable_claim_count, unevaluable_claim_count, mean_claim_support_score, minimum_claim_support_score, unsupported_claim_count, unsupported_claim_rate, maximum_claim_contradiction, fully_supported_response, response_grounding_status, y_suff_final`. Undefined rates are null.

`phase05_evaluator_unevaluable_claims.csv` contains every unevaluable claim and deterministic reason.

## Policy and paired views

Each `phase05_policy_G*.parquet` file has one row per eligible question with selected k/state, frozen action/threshold provenance, answered flag, coverage denominator, generation/quality/grounding fields, and null quality fields for abstentions.

`phase05_policy_generation_comparison.csv` stores conditional quality with answer denominators and the two population-level yield/exposure metrics over all 89 trajectories. `phase05_context_sufficiency_grounding_comparison.csv` groups k5/k10 states by the unchanged `y_suff_final`. `phase05_paired_k5_k10.parquet` stores both state values and predeclared descriptive differences for every question; no Phase 6 p-value appears.

## Integrity

`phase05_artifact_manifest.json` inventories Phase 5 files without including itself. `phase05_integrity_report.json` records upstream immutability, exact populations, leakage guards, RAGTruth grouping/threshold boundary, context utilization, policy reproduction, missing-value rules, TEST sealing, line endings, test results, and confirmation that Phase 6 was not started.
