# Phase 7 Final Results

## Boundary and population

TechQA TEST was unsealed once at `2026-09-01T12:07:27.0173627+05:30` after governance and upstream integrity passed. No post-TEST scientific choice was permitted or made.

The frozen TEST census was 136 questions: 91 benchmark-answerable and 45 benchmark-impossible. PRIMARY contains 88 questions and 1056 conditions; 3 unresolved-evidence questions and 45 benchmark-impossible questions were excluded from PRIMARY. The immutable final target contains 392 sufficient and 664 insufficient conditions.

## RQ1 — Retrieval and retrieved-context sufficiency

Validation evidence established strategy/depth-dependent retrieval and sufficiency patterns under the frozen Phase 2/3 design. On final TEST, the exact same 12 retrieval conditions per PRIMARY question were labelled with the frozen Phase 3 rule. TEST sufficiency rates by condition were: bm25 k=1 0.2500; bm25 k=3 0.3636; bm25 k=5 0.3636; bm25 k=10 0.4659; dense k=1 0.2500; dense k=3 0.3409; dense k=5 0.3750; dense k=10 0.4659; hybrid k=1 0.2159; hybrid k=3 0.3864; hybrid k=5 0.4432; hybrid k=10 0.5341. Absolute strategy/depth differences are the predeclared descriptive effects; no new aggregate RQ1 interval or post-hoc test was invented. This is a retrieval-conditioned target, not global question answerability.

## RQ2 — Predicting retrieved-context sufficiency

The selected uncalibrated Random Forest changed from VALIDATION to TEST as follows: auroc 0.7188 to 0.7588 (TEST 95% CI [0.6821, 0.8294]); auprc 0.6746 to 0.5788 (TEST 95% CI [0.4388, 0.7520]); f1 0.5077 to 0.5863 (TEST 95% CI [0.4613, 0.6891]); brier 0.2138 to 0.1939 (TEST 95% CI [0.1649, 0.2254]). TEST used all 39 frozen inference features and the TRAIN-fitted preprocessing/model without fitting or recalibration.

## RQ3 — Frozen answer/request-more-evidence/abstain policy

TEST AURC is 0.4801. The primary G2 policy (t_low=0.78, t_high=0.82) achieved coverage 0.0341 (95% CI [0.0000, 0.0795]), selective risk 1.0000 (CI [1.0000, 1.0000]), false-abstention 1.0000, and expansion 0.0114 with 0 safe and 3 unsafe answers. The G3 sensitivity policy (0.56, 0.72) achieved coverage 0.1591 and selective risk 0.2857. Thresholds were not reselected even when TEST risk exceeded its nominal validation constraint.

## RQ4 — Gating and unsupported-claim exposure

The final k5/k10 quality and grounding table reports ROUGE-L, BERTScore F1, the automatic unsupported-claim-rate proxy, fully-supported response rate, claim support, and output length. TEST paired k10−k5 effects were: rouge_l_f1 mean difference 0.0074 (95% CI [-0.0083, 0.0237]), rank-biserial 0.1184, Holm p=1.0000; bertscore_f1 mean difference 0.0018 (95% CI [-0.0009, 0.0045]), rank-biserial 0.1864, Holm p=0.6518; unsupported_claim_rate mean difference 0.0268 (95% CI [-0.0562, 0.1123]), rank-biserial -0.0197, Holm p=1.0000; mean_claim_support_score mean difference -0.0334 (95% CI [-0.1003, 0.0311]), rank-biserial -0.0365, Holm p=1.0000; output_token_count mean difference 0.9432 (95% CI [-4.2500, 6.1250]), rank-biserial 0.0407, Holm p=1.0000. Sufficiency associations, binary paired effects, and their CIs are in Tables 5–6. The grounding outputs are an automatic retrieved-context support proxy, not authoritative hallucination labels.

TEST Holm-surviving hypotheses: `B_k5_rouge_l_f1_sufficient_minus_insufficient`, `B_k5_bertscore_f1_sufficient_minus_insufficient`, `B_k10_rouge_l_f1_sufficient_minus_insufficient`, `B_k10_bertscore_f1_sufficient_minus_insufficient`, `B_k10_unsupported_claim_rate_sufficient_minus_insufficient`, `B_k10_fully_supported_response_sufficient_minus_insufficient`.

## Validation-to-TEST consistency

- classifier/auroc: VALIDATION 0.7188, TEST 0.7588 — uncertain due to CI overlap/sample size.
- classifier/auprc: VALIDATION 0.6746, TEST 0.5788 — uncertain due to CI overlap/sample size.
- classifier/f1: VALIDATION 0.5077, TEST 0.5863 — uncertain due to CI overlap/sample size.
- classifier/brier: VALIDATION 0.2138, TEST 0.1939 — uncertain due to CI overlap/sample size.
- risk_coverage/aurc: VALIDATION 0.3785, TEST 0.4801 — weaker on TEST.
- policy/G2_coverage: VALIDATION 0.0225, TEST 0.0341 — directionally consistent.
- policy/G2_selective_risk: VALIDATION 0.0000, TEST 1.0000 — weaker on TEST.
- policy/G3_coverage: VALIDATION 0.1798, TEST 0.1591 — weaker on TEST.
- policy/G3_selective_risk: VALIDATION 0.1875, TEST 0.2857 — weaker on TEST.
- generation/k5_mean_rouge_l: VALIDATION 0.1920, TEST 0.1977 — stronger on TEST.
- generation/k5_mean_bertscore_f1: VALIDATION 0.8875, TEST 0.8832 — weaker on TEST.
- generation/k5_mean_unsupported_claim_rate: VALIDATION 0.3013, TEST 0.3100 — weaker on TEST.
- generation/k5_fully_supported_response_rate: VALIDATION 0.4607, TEST 0.4659 — stronger on TEST.
- generation/k10_mean_rouge_l: VALIDATION 0.1991, TEST 0.2051 — stronger on TEST.
- generation/k10_mean_bertscore_f1: VALIDATION 0.8879, TEST 0.8850 — weaker on TEST.
- generation/k10_mean_unsupported_claim_rate: VALIDATION 0.2771, TEST 0.3368 — weaker on TEST.
- generation/k10_fully_supported_response_rate: VALIDATION 0.4719, TEST 0.4205 — weaker on TEST.
- sufficiency_association/B_k10_bertscore_f1_sufficient_minus_insufficient: VALIDATION 0.0169, TEST 0.0216 — directionally consistent.
- sufficiency_association/B_k10_fully_supported_response_sufficient_minus_insufficient: VALIDATION 0.0461, TEST 0.2849 — directionally consistent.
- sufficiency_association/B_k10_rouge_l_f1_sufficient_minus_insufficient: VALIDATION 0.0571, TEST 0.1523 — directionally consistent.
- sufficiency_association/B_k10_unsupported_claim_rate_sufficient_minus_insufficient: VALIDATION -0.0976, TEST -0.2123 — directionally consistent.
- sufficiency_association/B_k5_bertscore_f1_sufficient_minus_insufficient: VALIDATION 0.0155, TEST 0.0247 — directionally consistent.
- sufficiency_association/B_k5_fully_supported_response_sufficient_minus_insufficient: VALIDATION 0.1510, TEST 0.1303 — directionally consistent.
- sufficiency_association/B_k5_rouge_l_f1_sufficient_minus_insufficient: VALIDATION 0.0506, TEST 0.1668 — directionally consistent.
- sufficiency_association/B_k5_unsupported_claim_rate_sufficient_minus_insufficient: VALIDATION -0.1982, TEST -0.1208 — directionally consistent.

## Benchmark-impossible sensitivity

The separate 45-question/540-condition benchmark-impossible sensitivity produced a predicted-sufficient rate of 0.1204 at 0.5. These preliminary negatives are benchmark-relative and contaminated; they are not perfect context-sufficiency ground truth and did not influence PRIMARY results.

## Limitations

- TEST has 88 PRIMARY questions, so uncertainty can remain wide and subgroup results may be sparse.
- The final target is operational and inherits evidence-alignment, claim-segmentation, NLI-proxy, and single-annotator limitations from Phase 3.
- ROUGE-L and BERTScore measure reference similarity, not grounding; the NLI grounding measure is a validated but imperfect proxy.
- Selective risk is an observed held-out estimate, not a production safety guarantee.
- Benchmark-impossible sensitivity has known label contamination and remains separate.
- No TEST result was used to tune or reselect any component; worse generalization is retained unchanged.

This document is the Phase 7 empirical source for later thesis Chapters 4–6. It is not a thesis rewrite.
