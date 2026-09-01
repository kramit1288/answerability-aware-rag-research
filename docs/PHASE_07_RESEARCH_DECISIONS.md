# Phase 7 Research Decisions

## Governance boundary

Phase 7 begins from tag `phase-06-statistical-analysis-v1`, peeled commit
`a6dc70ee13c0ddc14e7d5f7622110343d5befe8e`. Phases 1-6 are immutable scientific inputs.
The authoritative pre-TEST machine-readable protocol is `configs/phase07_final_test.json`.
This document, the configuration, execution protocol, and schemas must be hashed in
`phase07_pre_test_governance_freeze.json` before any TechQA TEST question, context, strict label,
semantic score, final target, feature, probability, generation, aggregate, or outcome is read or
created.

**No post-TEST scientific choice is permitted.**

## Decision P7-01: one-time unseal

The TEST split is unsealed once, only after the pre-TEST governance artifact exists and every
frozen upstream scientific hash passes. The unseal record stores the timestamp, branch, commit,
upstream tag, Phase 7 configuration hash, and all key upstream hashes. Once written, TEST remains
conceptually unsealed even if execution later fails. A failure cannot justify retuning or a fresh
selective unseal.

## Decision P7-02: population and target

The complete frozen Phase 1 TEST census is reported before exclusions and split membership never
changes. PRIMARY contains only benchmark-answerable, defensibly aligned, semantic-evaluable TEST
questions. Benchmark-impossible questions, unresolved evidence, and frozen-rule semantic-
unevaluable questions remain explicit exclusions. Benchmark-impossible cases are reported only in
the already frozen separate sensitivity and cannot influence PRIMARY.

The TEST target is constructed mechanically. Strict positives cannot be demoted. A strict
negative is rescued exactly when span coverage is at least 0.20, or when mean claim entailment is
at least 0.35, minimum claim entailment is at least 0.05, and selected-premise contradiction is
strictly below 0.50. The exact Phase 3 NLI revision, deterministic claim segmentation,
claim-targeted multi-chunk premise, float32 CPU inference, and semantic-unevaluable budget rule
are reused. Aggregate target balance is not inspected until every eligible condition has a final
target or explicit exclusion. The completed TEST target is immutable.

## Decision P7-03: inference and classifier evaluation

Exactly the 39 registered Phase 4 inference features are constructed from frozen retrieval and
context resources. TEST fits no normalization, imputation, categorical mapping, scaler, model, or
calibrator. The serialized TRAIN-fitted uncalibrated Random Forest is loaded unchanged.
Classification uses threshold 0.5 and reports AUROC, AUPRC, accuracy, precision, recall, F1,
Brier score, and ten-bin equal-width ECE.

AUROC, AUPRC, F1, and Brier receive 5,000-replicate percentile confidence intervals using seed 42
and question-level resampling; each sampled question occurrence carries all 12 retrieval
conditions. Undefined replicates are discarded and reported, never zero-filled.

## Decision P7-04: risk-coverage and frozen policy thresholds

The complete TEST risk-coverage curve uses frozen probabilities and the Phase 4 tied-threshold,
right-endpoint AURC convention. The only reported operating-threshold checks are the exact
VALIDATION-derived thresholds 0.7976647044899572, 0.7838454251283709, and 0.6779617685497638 for
the declared 5%, 10%, and 20% points. No TEST threshold search is permitted.

The primary adaptive policy remains `t_low=0.78`, `t_high=0.82`; the sensitivity remains
`t_low=0.56`, `t_high=0.72`. Each starts at hybrid k5 and may expand once to hybrid k10. All
coverage, risk, false-abstention, expansion, safety, and retrieval-cost metrics are evaluated
mechanically with question-bootstrap intervals for important proportions. An observed TEST risk
above the validation constraint is reported unchanged.

## Decision P7-05: frozen generation, quality, and grounding

Every PRIMARY TEST question receives one deterministic hybrid-k5 generation and one deterministic
hybrid-k10 generation from Qwen/Qwen2.5-1.5B-Instruct revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, CPU float32, greedy one-beam decoding, and 128 maximum
new tokens. Prompt SHA-256 is `2db430ca27b5dec058b00815d07b21bcf9924f17b0d12006c1a07963ca17e8d1`.
Context assembly and budgeting are exactly Phase 5. Benchmark answers are unavailable until the
complete generation artifact is closed and hashed.

ROUGE-L and BERTScore F1 reuse the frozen Phase 5 definitions. Grounding reuses the frozen
three-candidate MiniLM/NLI evaluator and support threshold 0.16. Grounding is an imperfect
retrieved-context support proxy, not authoritative hallucination truth. Failures, empty outputs,
unevaluable claims, and abstentions preserve their frozen missing-value semantics. No quality-
motivated retry or model/prompt/evaluator substitution is allowed.

## Decision P7-06: inference families and policy views

The TEST analyses reproduce the Phase 6 families without expansion: Family A contains five paired
k10-minus-k5 continuous outcomes; Family B contains eight k-specific sufficiency associations;
Family C contains two paired binary outcomes. Tests, effect sizes, 5,000 question bootstraps,
directions, missingness, and within-family Holm adjustment are identical to Phase 6.

G0 always selects k5, G1 always selects k10, G2 selects the frozen primary adaptive state, and G3
selects the frozen 20% sensitivity state. Content quality is conditional on answers and remains NA
for abstentions. Grounded-answer yield and unsupported-answer population rate retain all eligible
questions in their denominators.

## Decision P7-07: validation comparison, sensitivity, and reporting

VALIDATION estimates are shown alongside TEST only for generalization discussion. Descriptions are
limited to directionally consistent, weaker on TEST, stronger on TEST, or uncertain because of CI
overlap/sample size. No post-hoc pass/fail rule is introduced. Worse TEST results, missed
hypotheses, constraint violations, and generalization failures are reported without changing the
system.

The benchmark-impossible TEST sensitivity mechanically applies the frozen Phase 4 sensitivity
definition and remains separate. Its preliminary negatives have known contamination and are not
reinterpreted as perfect truth.

## Decision P7-08: stopping and integrity

Any pre-unseal scientific hash mismatch stops Phase 7 before TEST access. After unseal, a target,
feature, model, generation, or evaluator failure is recorded against the frozen procedure and
cannot authorize tuning. A post-TEST integrity assertion compares every upstream scientific hash
to the unseal record and must persist `post_test_tuning_detected=false` for completion.

Phase 7 ends with final TEST artifacts, RQ evidence, tables, limited material figures, an
independent non-self-referential manifest, and passing checks. It does not rewrite thesis chapters,
commit, tag, or begin Phase 8.
