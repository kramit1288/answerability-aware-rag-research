# Phase 4 Results Interpretation

## Status and provenance

This is a **post-results interpretation artifact**. Phase 4 methodology was frozen before result evaluation, and this document was created only after the Phase 4 empirical results were complete. It does not modify the pre-results methodology. It records interpretation, limitations, and implications of the frozen results.

This document must not be used as evidence that any model, feature, calibration, or threshold decision was predeclared. No model tuning or threshold modification followed these interpretations. The immutable pre-results governance remains in `configs/phase04_modeling.json`, `docs/PHASE_04_EXECUTION.md`, `docs/PHASE_04_RESEARCH_DECISIONS.md`, `docs/PHASE_04_ARTIFACT_SCHEMAS.md`, and `artifacts/results/phase04_modeling_config_freeze.json`.

Relevant frozen identifiers are:

- Phase 4 modeling configuration canonical SHA-256: `f0a0377e4001e6401e4cce77ad15e0904ec13e87719d56660a40171341a6801e`
- Selected model configuration SHA-256: `efb07f98b8e73a5f277bd89591badf1d64f486354de8e4caccdad7672463a019`
- Selected policy configuration canonical SHA-256: `455b11f36c2bcef93d6a2ff9a2323ebc72ebdafe34c7a0b5c0ae82cd1e70b0d5`
- Phase 4 artifact manifest SHA-256 before this interpretation was added: `646908b85462963a96e8fdaaff630904444b5f64ed14f7c332d3263aadec3edc`

The last value is the **pre-interpretation manifest hash**. The final post-results artifact manifest supersedes it only as the inventory of the completed Phase 4 state; it does not supersede or rewrite the historical pre-results governance freeze.

## Model-selection interpretation

The frozen VALIDATION results were:

| Predictor | AUPRC | AUROC | F1 |
|---|---:|---:|---:|
| B1 simple threshold | 0.600657 | 0.680320 | 0.585519 |
| Logistic Regression | 0.625554 | 0.705811 | 0.574394 |
| Random Forest | 0.674558 | 0.718833 | 0.507692 |

Random Forest was selected because the frozen primary model-selection criterion was VALIDATION AUPRC. This does not imply that Random Forest dominated every metric. In particular, its F1 was lower at the default 0.5 classification threshold. That lower F1 is compatible with its selection because F1 at 0.5 was not the predeclared primary model-selection criterion.

## Calibration interpretation

| Random Forest probability method | Brier | ECE |
|---|---:|---:|
| Uncalibrated | 0.213815 | 0.063370 |
| Sigmoid | 0.217307 | 0.079029 |
| Isotonic | 0.217786 | 0.071414 |

Both calibration methods worsened the frozen calibration-selection metrics. The selected predictor therefore uses **uncalibrated Random Forest probabilities**. It is not a calibrated Random Forest, and its probability must not be interpreted as objective or epistemic certainty.

## Feature interpretation

The frozen ablations were:

| Feature family | AUPRC | AUROC |
|---|---:|---:|
| Retrieval scores only | 0.578596 | 0.667678 |
| Lexical and semantic query-context matching | 0.673557 | 0.710334 |
| Retrieval scores plus matching | 0.668344 | 0.708602 |
| Full feature set | 0.674558 | 0.718833 |

In this experiment, query-context evidence-alignment features carried substantially more predictive signal than retrieval-score features alone. This is a descriptive association, not a causal conclusion. Semantic similarity does not establish factual support.

`retrieval_strategy` was also one of the stronger permutation-importance features. This may indicate that prediction behavior varies across retrieval regimes. Retrieval strategy itself is not direct evidence that a retrieved context is sufficient.

## Discrimination and uncertainty

The selected Random Forest produced:

| Metric | VALIDATION estimate | 95% question-bootstrap CI |
|---|---:|---:|
| AUROC | 0.718833 | [0.645963, 0.787351] |
| AUPRC | 0.674558 | [0.539924, 0.785402] |
| F1 | 0.507692 | [0.394183, 0.608769] |
| Brier | 0.213815 | [0.189549, 0.238542] |

The lightweight classifier exhibits moderate discrimination rather than near-perfect prediction. The relatively wide confidence intervals reflect the 89-question VALIDATION population. The model is not production-ready.

## Safety-coverage trade-off

The frozen condition-level selective operating points were:

| Observed risk constraint | Threshold | Coverage | Observed selective risk |
|---|---:|---:|---:|
| 5% | 0.797665 | 0.027154 | 0.034483 |
| 10% | 0.783845 | 0.043071 | 0.086957 |
| 20% | 0.677962 | 0.112360 | 0.200000 |

AURC was `0.3784805395` under the frozen integration convention.

More conservative observed selective-risk constraints produced sharply lower answer coverage. This demonstrates a safety-utility trade-off rather than a free safety gain. None of these observed risks is a real-world safety guarantee.

## Primary three-way policy

The primary illustrative constraint was 10%, with `t_low = 0.78` and `t_high = 0.82`. On VALIDATION:

- answer coverage: 2/89 = 0.022472
- selective risk: 0
- safe answers: 2
- unsafe answers: 0
- abstentions: 87
- retrieval expansions: 3/89 = 0.033708
- false-abstention rate: 51/53 = 0.962264
- mean retrieval depth: 5.168539
- retrieval-depth cost proxy: 1.033708

The primary policy produced zero observed unsafe answers in this particular VALIDATION sample, but it did so with extremely low answer coverage and very high false abstention. It must not be described as generally safe, and its low coverage is a central empirical result rather than a defect to conceal or tune away post hoc.

## Adaptive retrieval interpretation

At the 10% constraint:

| Policy | Coverage | Observed selective risk | Mean retrieval depth |
|---|---:|---:|---:|
| Adaptive k5 to optional k10 | 0.022472 | 0 | 5.168539 |
| Always k10 | 0.022472 | 0 | 10 |

Adaptive expansion did **not** improve answer coverage at the central 10% operating point. Its measured advantage was retrieval efficiency: it matched the observed coverage and risk of always-k10 while retrieving substantially less context on average.

At the less conservative 20% validation operating point, adaptive coverage was `0.179775`, compared with `0.123596` for always-k10, while remaining within the frozen risk constraint. This is only a validation-set observation, not a general or production guarantee.

## Two-way and always-answer policies

No nonempty P1 two-way k5 operating point satisfied any of the frozen 5%, 10%, or 20% risk constraints on the predeclared 0.02 probability grid. Zero-answer candidates used selective risk = NA and were not treated as valid safe operating points. Within this experiment, this is evidence that retrieval expansion can create operational value relative to immediate answer-or-abstain decisions; it must not be overgeneralized.

For P0 always-answer at hybrid k5:

- coverage: 1.0
- selective risk: 0.471910
- safe answers: 47
- unsafe answers: 42

P0 violated all three frozen selective-risk constraints. This indicates that unconditional answering carries substantial risk under the frozen operational sufficiency target. The 42 unsafe cases are not established hallucinations; they are retrieval contexts labeled insufficient for a materially complete answer.

## Benchmark-impossible sensitivity

PRIMARY Random Forest AUPRC was `0.674558`. The separately frozen benchmark-impossible sensitivity produced:

- AUROC: 0.697316
- AUPRC: 0.466677
- F1: 0.297935
- Brier: 0.192374
- ECE: 0.065439
- impossible-row predicted-sufficient rate at 0.5: 0.122222

The sensitivity analysis produced substantially lower AUPRC. This requires caution because the benchmark-impossible examples had already shown answerability contamination during Phase 3 human validation. The sensitivity result does not modify PRIMARY Phase 4 model or policy selection.

## Post-results implication for Phase 5

This section records a post-results study interpretation, not a methodology rewrite or authorization to begin Phase 5.

Phase 5 must preserve the frozen primary 10% policy. It may additionally report generation and grounding outcomes under the already persisted 20% risk operating point as a sensitivity and utility analysis. The 20% setting must not replace the 10% primary policy, and no new Phase 4 operating point may be created or tuned.

The reason for reporting the already frozen sensitivity is that the primary 10% policy answers only two VALIDATION questions, making standalone generation-quality analysis statistically sparse. Phase 5 remains unstarted and requires separate authorization.

## Thesis-safe interpretation

The lightweight predictor showed moderate ability to distinguish sufficient from insufficient retrieved contexts. Selective answering reduced observed unsafe answering, but conservative risk constraints substantially reduced coverage. Adaptive retrieval expansion improved retrieval efficiency at the central operating point and improved validation coverage at the less conservative 20% operating point.

These results do not establish that hallucination was solved, that objective universal answerability was predicted, that probabilities represent epistemic certainty, that feature importance is causal, that zero observed unsafe answers implies production safety, or that the 10% threshold is universally acceptable.
