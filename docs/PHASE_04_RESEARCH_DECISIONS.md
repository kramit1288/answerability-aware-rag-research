# Phase 4 Research Decisions

## Governance boundary

Phase 4 starts from the frozen Phase 3 v3 state (`phase-03-context-sufficiency-v3`, commit `04e901e68937f0f149ab12e2d2f44055b00b8fad`). `docs/RESEARCH_DECISIONS.md` is immutable because it is included in the Phase 3 final artifact manifest. All new Phase 4 methodological decisions are therefore recorded in this append-only Phase 4 document. This is a governance separation, not a methodology change.

The immutable upstream dependencies are:

- Phase 3 final manifest SHA-256: `b02a0ca0e3352798c8d38ef4942fdca59e23d8d24c662493a1249d2864ca9879`
- Phase 3 final target configuration SHA-256: `5b9a394e5e97776844054f3282af815462148adc62b528c77245d61707406977`
- PRIMARY target SHA-256: `b775276ace9113286a8e35fd29acaee5623d83be60400537e6902027b8cc07c1`

Phase 4 verifies the Phase 3 manifest as an immutable upstream dependency. It does not extend or mutate that manifest. Phase 4 governance and results have their own manifest and checker.

## Verified development population

The PRIMARY Phase 4 development population is the frozen Phase 3 PRIMARY target:

| Partition | Questions | Retrieval-condition rows |
|---|---:|---:|
| TRAIN | 424 | 5,088 |
| VALIDATION | 89 | 1,068 |
| Total | 513 | 6,156 |

The total contains 2,802 positive and 3,354 negative retrieval-conditioned context-sufficiency targets. Benchmark-impossible and unresolved-evidence questions are excluded. `DEV_Q066` and `TRAIN_Q526` remain excluded under the frozen semantic input-budget rule, and their excluded conditions remain NA rather than negatives. TEST remains sealed.

## Decision P4-01: prediction target and experimental unit

The target is the frozen `y_suff_final` operational target for retrieval-conditioned context sufficiency, not global question answerability or universal objective answerability. One row is one question/retrieval-strategy/k condition. `question_id` is used only for provenance, grouping, bootstrap resampling, and leakage checks; it is never a predictor.

All inner cross-validation and out-of-fold calibration prediction uses five-fold `GroupKFold` with `question_id` as the group. TRAIN fits transformations, selects hyperparameters, creates out-of-fold probabilities, and fits calibration mappings. VALIDATION compares families and calibration methods, describes ablations and feature importance, evaluates risk–coverage, and selects policy thresholds. TEST is not read for Phase 4 feature generation or inference.

## Decision P4-02: feature boundary

The machine-readable registry is `configs/phase04_feature_registry.json`. Only fields classified exactly as `inference_available_feature` may enter a model matrix. The registry contains 39 base features: eight query features, eight retrieval-score features, five lexical query/context features, five MiniLM query/context features, nine context-composition features, two BM25/dense agreement features, and two retrieval-condition metadata features.

The experiment contract already made `retrieval_strategy` and `k` eligible predictors. Phase 4 therefore uses a fixed one-hot representation for strategy and numeric `k`; the declared ablation makes their contribution transparent. Reciprocal-rank agreement is excluded. No LLM judge or additional pretrained semantic model is introduced.

The identifier-like token regex, frozen BM25 lexical tokenizer, exact cached Phase 2 BM25 IDF values (including the library's epsilon adjustment), deterministic OOV IDF equation, frozen MiniLM revision, overlap denominators, and aggregation conventions are specified in `configs/phase04_modeling.json` before model results.

## Decision P4-03: retrieval-score scales and missing values

BM25, dense cosine, and hybrid RRF scores are not treated as a shared raw scale. Each retrieval-score feature is normalized within retrieval strategy using means and population standard deviations learned from the applicable TRAIN fold. This transform is refit within grouped cross-validation and is reused without refitting on VALIDATION.

Mathematically undefined values, including top-two gaps at k=1 and pairwise redundancy for a single chunk, are stored as missing. Continuous values use TRAIN-fitted median imputation with missing indicators. Logistic Regression then uses TRAIN-fitted standardization; Random Forest uses the same normalization/imputation semantics without the final scaler. Strategy levels are fixed to BM25, dense, and hybrid.

## Decision P4-04: model search and calibration

B0 always predicts sufficient and is an operational reference; constant-score ranking metrics are NA. B1 thresholds IDF-weighted query-token context coverage on a predeclared 0.00–1.00 grid with step 0.01 and selects by grouped-TRAIN mean F1, then precision, then the higher threshold.

Logistic Regression searches `C` in `{0.1, 1.0, 10.0}` and `class_weight` in `{None, balanced}`. Random Forest uses 500 trees and searches the exact grid in the Phase 4 configuration. Both select by grouped-TRAIN mean AUPRC, then AUROC, then the frozen simplicity order.

The selected Random Forest produces grouped out-of-fold TRAIN probabilities. Sigmoid and isotonic mappings are fit only to those OOF probabilities and TRAIN targets. The forest is refit on all TRAIN rows before producing raw VALIDATION probabilities. Calibration is selected on VALIDATION by Brier score and then ten-bin ECE; uncalibrated output may remain selected.

Learned model-family selection is highest VALIDATION AUPRC, then AUROC, then Brier score where meaningful, then the frozen simpler-family order. The selected-model configuration is hashed before policy tuning.

## Decision P4-05: metrics, uncertainty, interpretation, and ablations

Probability-to-class metrics use threshold 0.5. ECE uses ten equal-width bins and the weighted absolute difference between bin positive fraction and mean predicted probability. Confidence intervals use 1,000 fixed-seed question-level bootstrap replicates and percentile bounds; every sampled question occurrence brings all its condition rows.

The selected Logistic Regression is described using signed standardized coefficients. If Random Forest is selected, its primary descriptive importance is fixed-seed VALIDATION permutation importance under AUPRC, not impurity importance. Importance is descriptive and not causal.

Ablations A–D are exactly score-only, query/context matching-only, their union, and the full set. They reuse the selected family and frozen hyperparameters without independent tuning. Metadata appears only in the full ablation.

## Decision P4-06: selective prediction and policy

The full condition-level risk–coverage curve uses all distinct selected-model VALIDATION probabilities with tied rows entering together. Selective risk is NA at zero coverage. AURC uses the frozen right-endpoint rectangular convention recorded in the configuration. Operating constraints are 5%, 10%, and 20%; maximum coverage is selected among feasible thresholds.

The runtime research policy is hybrid k=5 followed by at most one retrieval expansion to hybrid k=10. It never generates a new query, requests user clarification, loops, retrieves beyond k=10, or generates an answer in Phase 4. The complete 0.02 low/high threshold grid is persisted. Pair selection follows risk feasibility, maximum answer coverage, minimum expansion rate, larger high threshold, then deterministic larger low threshold. The 10% constraint is illustrative, not a production safety guarantee.

## Decision P4-07: benchmark-impossible sensitivity and sealing

Only after the PRIMARY model, calibration, and policy are frozen, benchmark-impossible preliminary negatives are added to TRAIN for a mechanically retrained sensitivity analysis. Features, family, hyperparameters, calibration architecture, and primary thresholds remain fixed. The sensitivity cannot alter PRIMARY conclusions.

No Phase 4 TEST features, probabilities, summaries, metrics, calibration, importance, risk–coverage, policy search, or error analysis are permitted. Phase 5 generation is outside this phase and is not started.

## Decision P4-08: relation to earlier planning documents

The explicit approved Phase 4 specification supersedes older Phase 4 search-grid defaults in the implementation plan and experiment contract where they differ. It also includes freezing the non-generative three-way policy during Phase 4; this is a phase-governance naming adjustment relative to an older plan that numbered policy implementation as Phase 5. No generation or generation-quality evaluation is included.
