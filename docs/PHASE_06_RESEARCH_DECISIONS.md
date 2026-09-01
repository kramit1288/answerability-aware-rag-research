# Phase 6 Research Decisions

## Governance boundary

Phase 6 begins from tag `phase-05-generation-grounding-v1`, peeled commit
`2850efa76b376a6640f6b6636bae3c531c6f4064`, and the Phase 5 final artifact-manifest
SHA-256 `2f87445752de5013afe20961116fd024ee78dbb5a298fc2fe5fa8f2b2b57138e`.
Phases 1-5 are immutable inputs. Phase 6 performs no retrieval change, label change, model fit,
calibration fit, policy tuning, answer generation, NLI inference, or grounding-threshold search.
TechQA TEST remains sealed; the already completed RAGTruth supporting TEST evaluation is only
consolidated.

The authoritative pre-results machine-readable decisions are in
`configs/phase06_statistics.json`. This document and the configuration are frozen before any
Phase 6 significance test is calculated.

## Decision P6-01: research-question mapping

The four frozen contract questions remain authoritative. The proposal's broader question-level
wording is preserved as intent but operationalized through the actual retrieval-conditioned unit
`(question, retrieval strategy, k, retrieved context, y_suff_final)`. Global benchmark
answerability and retrieved-context sufficiency are not interchangeable.

RQ1 is consolidated from the frozen controlled retrieval and context-sufficiency artifacts. RQ2
is consolidated from frozen Phase 4 discrimination, calibration, ablation, and feature-importance
artifacts; no post-hoc classifier-family test is introduced. RQ3 is evaluated through frozen
risk-coverage and policy artifacts, with question-bootstrap uncertainty for generation-policy
proportions but no test between deterministic operating thresholds. RQ4 receives the planned
paired and associational generation/grounding inference and the frozen RAGTruth evaluator
validation.

## Decision P6-02: units, uncertainty, and missingness

TechQA resampling uses `question_id`. A sampled question occurrence carries the complete row or
paired state needed for the named statistic. Retrieval-condition rows and claims are never
bootstrapped independently. RAGTruth intervals retain the already frozen `source_id` grouping;
no response-level replacement interval is created.

All Phase 6 TechQA intervals use 5,000 percentile bootstrap replicates, seed 42, and 95%
confidence. This explicit Phase 6 instruction supersedes the older contract's general 10,000-
replicate default without changing any frozen upstream method or result. Missing outcome values
are excluded only for the metric requiring them, with aligned complete pairs for paired tests.
NA is never converted to false or zero. Policy abstentions retain NA answer-content quality.

## Decision P6-03: hypothesis families

Holm correction is applied separately to three logical families:

- Family A: five paired k10-minus-k5 continuous outcomes (ROUGE-L, BERTScore F1,
  unsupported-claim rate, mean claim support, and output-token count), using two-sided Wilcoxon
  signed-rank tests with Pratt zero handling.
- Family B: sufficient-minus-insufficient associations at k=5 and k=10 for three continuous
  outcomes and one binary outcome per depth. Continuous tests use two-sided Mann-Whitney U;
  binary tests use two-sided Fisher exact.
- Family C: paired k5/k10 fully-supported-response and response-contains-unsupported-claim
  outcomes, using two-sided exact McNemar tests.

Every p-value is stored with a numerical effect size, confidence interval, family ID, raw value,
Holm-adjusted value, family size, and rejection decision.

## Decision P6-04: effect sizes

Paired continuous outcomes use mean and median k10-minus-k5 differences plus matched-pairs
rank-biserial correlation. Independent continuous associations use sufficient-minus-insufficient
mean/median differences and Cliff's delta. Binary paired outcomes use absolute paired risk
difference. Binary sufficiency associations report risk difference, risk ratio where its
denominator permits, and odds ratio where defined. Effect sizes remain numerical; arbitrary
small/medium/large labels are not applied.

## Decision P6-05: policy uncertainty and tiny samples

G0-G3 receive 95% question-bootstrap intervals for answer coverage, grounded-answer yield, and
unsupported-answer population rate. Quality intervals are produced only for policies with at
least 10 answered responses. G2 has two answers and cannot support a quality comparison or
quality bootstrap claim. Its observed zero unsupported responses is explicitly not treated as
evidence that its true risk is zero.

## Decision P6-06: tables, figures, and interpretation

Eleven predeclared tables and eight figure concepts (with AUROC and AUPRC rendered separately for
clarity) are generated mechanically from canonical frozen inputs and Phase 6 result artifacts.
Every figure has a CSV data artifact, SVG, 300-DPI PNG, caption candidate, and SHA-256. Matplotlib
is deterministic and seaborn is not used.

All sufficiency/generation comparisons are associational. Model confidence is not grounding;
grounding is not authoritative hallucination truth; validation risk is not a production safety
guarantee. The required non-significance wording is “did not provide evidence of a statistically
detectable difference.” Phase 6 stops before Phase 7 and creates neither a commit nor a tag.
