# Research Implementation Plan

## 1. Research contract

The study evaluates whether a RAG system for technical documentation has **sufficient retrieved evidence to answer a question**, and whether an answerability-aware decision policy reduces unsafe answering without causing excessive abstention.

The central object is not only a question-level label. It is a retrieval-conditioned example:

\[
(q_i, r, k, C_{i,r,k}, y^{suff}_{i,r,k})
\]

where:

- \(q_i\): query
- \(r\): retrieval strategy
- \(k\): retrieval depth
- \(C_{i,r,k}\): context actually retrieved under that condition
- \(y^{suff}_{i,r,k}\): whether that retrieved context contains sufficient evidence to answer

This distinction must be preserved throughout the implementation.

## 2. Revised research questions

### RQ1
How do retrieval strategy and retrieval depth affect retrieved-context sufficiency for technical-documentation QA?

### RQ2
Which retrieval and context characteristics are most predictive of context sufficiency, and does a calibrated multivariate classifier outperform simpler score-based baselines?

### RQ3
How does an answerability-aware three-way decision policy affect unsafe-answer rate, false-abstention rate, coverage and selective risk compared with an always-answer RAG baseline?

### RQ4
Does answerability-aware gating reduce unsupported claims among generated answers, and how reliably can unsupported claims be detected?

## 3. Phase 0 — Audit and experiment contract

### Tasks

- Inspect every cell of `notebooks/original_prototype.ipynb`.
- Create `docs/NOTEBOOK_AUDIT.md` mapping current code to the revised experiment.
- Identify:
  - reusable code
  - scientifically invalid or ambiguous logic
  - data leakage risks
  - hard-coded thresholds
  - metrics that do not match their names
  - any unsupported assumptions
- Inspect the actual TechQA-RAG-Eval and RAGTruth schemas before writing data logic.
- Produce `docs/EXPERIMENT_CONTRACT.md` containing:
  - exact dataset fields used
  - definition of experimental unit
  - split strategy
  - label construction
  - models
  - retrieval conditions
  - metrics
  - statistical tests
  - artifact schemas

### Acceptance criteria

No implementation of new model logic begins until the experiment contract is explicit and internally consistent.

## 4. Phase 1 — Data foundation

### TechQA

- Load the official TechQA-RAG-Eval dataset.
- Use the official/fixed technical-document corpus exposed by the benchmark rather than reconstructing the only corpus from QA-row contexts, if supported by the current dataset schema.
- Persist dataset version/fingerprint.
- Create group-aware train/validation/test partitions using source/document grouping where identifiers allow it.
- If an official split should supersede custom splitting, document and use it.

### RAGTruth

- Respect the dataset's official train/test split.
- Preserve source-level grouping such as `source_id` where applicable.
- Do not randomly split generated responses from the same source across train and test.

### Required outputs

- `artifacts/data/techqa_split_assignments.csv`
- `artifacts/data/techqa_corpus_manifest.csv`
- `artifacts/data/ragtruth_manifest.csv`
- `artifacts/data/dataset_metadata.json`

### Tests

- no train/validation/test group overlap
- stable split under the same seed
- unique IDs
- expected label distributions
- no accidental null/empty critical fields

## 5. Phase 2 — Controlled retrieval experiment

Implement retrieval modules:

- BM25
- dense sentence embedding retrieval
- hybrid retrieval

Retrieval depth:

\[
k \in \{1,3,5,10\}
\]

Use the same corpus and query split for all retrieval strategies.

Persist query-level rankings/scores rather than only aggregate means.

### Features

At minimum:

- top BM25 score
- top dense score
- hybrid/fusion score
- top1-top2 score gap
- score statistics across top-k
- lexical overlap
- context count
- context length
- query length

Add better evidence features only if their definitions are fixed before test evaluation.

### Metrics

- Recall@1
- Recall@3
- Recall@5
- Recall@10
- MRR where ground-truth relevance permits it

### Output

`artifacts/results/retrieval_query_level.csv`

One row per query × retrieval strategy × k.

## 6. Phase 3 — Retrieved-context sufficiency labels

This is the most important methodological correction.

Do not train the classifier directly against a global question-answerability label if that label does not reflect whether the **retrieved context for the current condition** contains the answer.

Create:

\[
y^{suff}_{i,r,k}
\]

based on whether the retrieved context supports the ground-truth answer/evidence.

### Preferred label hierarchy

1. gold evidence/document identity where available
2. answer/evidence containment with defensible normalization
3. semantic/NLI evidence-support rule validated manually
4. manual adjudication for ambiguous cases

Do not silently choose one. Document the actual label rule and validate it.

### Validation

Draw a stratified manual sample from positive/negative/ambiguous labels and record human judgement.

### Outputs

- `artifacts/data/context_sufficiency_examples.csv`
- `artifacts/results/context_sufficiency_label_validation.csv`

## 7. Phase 4 — Answerability prediction and calibration

### Baselines/models

B0. Always answer  
B1. Simple score threshold  
B2. Logistic Regression  
B3. Random Forest, uncalibrated  
B4. Random Forest, calibrated

Optional additional model only if it adds research value without obscuring the study.

### Training discipline

- training set: fit models
- validation set: feature/model decisions, calibration and threshold tuning
- test set: untouched until final evaluation

### Classification metrics

- accuracy
- precision
- recall
- F1
- AUROC
- optionally AUPRC when class imbalance warrants it

### Calibration metrics

- Brier score
- Expected Calibration Error (ECE)
- reliability diagram

Persist trained models and calibration objects.

## 8. Phase 5 — Three-way decision policy

For calibrated sufficiency probability \(p\):

\[
a(p)=
\begin{cases}
\text{abstain}, & p<t_{low} \\
\text{request evidence}, & t_{low}\le p<t_{high} \\
\text{answer}, & p\ge t_{high}
\end{cases}
\]

Do not use a fixed arbitrary relation such as `t_low = t_high - 0.20` unless justified before evaluation.

Tune \(t_{low}\) and \(t_{high}\) jointly on validation data.

A defensible optimization is:

\[
\max Coverage
\]

subject to a predeclared upper bound on unsafe-answer rate, or compare several predeclared safety constraints.

### Metrics

- unsafe answer rate
- false abstention rate
- request-more-evidence rate
- answer coverage
- useful coverage
- selective risk
- risk-coverage curve
- AURC

Always interpret quality metrics alongside coverage.

## 9. Phase 6 — Generation and unsupported-claim evaluation

Generate answers only for the policy conditions that choose `answer`.

Keep the generation model/configuration fixed across policy comparisons.

For non-answer outcomes, generation-quality metrics must be `NA`, not zero.

### Unsupported claims

The old cosine-similarity threshold may be retained as an explicitly named **embedding support proxy**, not as definitive hallucination detection.

Add a stronger support/faithfulness method using NLI or a reproducible evaluation framework.

Validate automatic support judgements using:

- RAGTruth official labels where mapping is meaningful
- and/or a manually reviewed stratified sample

Report automatic-vs-human agreement.

### Answer metrics

Use only metrics suitable for the dataset and explicitly define them. ROUGE-L/BERTScore may be secondary; grounding and safety are the primary outcomes.

## 10. Phase 7 — Statistical analysis

At minimum:

- descriptive statistics with 95% confidence intervals
- class-preserving or paired bootstrap confidence intervals where appropriate
- paired tests when the same questions are evaluated by multiple methods
- McNemar for paired binary decisions/outcomes where applicable
- Wilcoxon signed-rank for paired continuous/non-normal metrics where applicable
- effect sizes
- Holm correction for families of multiple comparisons

Do not choose a test simply because another thesis used it. Match each test to this experiment's unit of observation and dependency structure.

### Required outputs

- `artifacts/results/statistical_tests.csv`
- `artifacts/results/bootstrap_intervals.csv`
- `artifacts/results/model_comparison.csv`

## 11. Thesis figures

Generate reproducible thesis figures from saved result files, including:

1. Recall@k by retrieval strategy
2. classifier ROC or PR curve
3. calibration reliability diagram
4. risk-coverage curve
5. unsafe-answer vs false-abstention/coverage trade-off
6. confusion matrix for final sufficiency classifier
7. policy outcome distribution
8. unsupported-claim rate among answered responses

Figures must be created by scripts and written to `artifacts/figures/`.

## 12. Final reproducibility bundle

A final command or documented sequence should rebuild all results from the frozen data/artifacts.

Freeze:

- Python version
- package versions
- random seed(s)
- dataset revision/fingerprint
- embedding model
- generation model
- prompts
- retrieval configuration
- threshold selection rule
- statistical-test definitions

Produce:

- `artifacts/results/final_metrics.csv`
- `artifacts/results/query_level_final_results.csv`
- `artifacts/results/final_statistical_tests.csv`
- `artifacts/results/run_manifest.json`
- all final figures

## 13. What not to do

- Do not optimize on the test set.
- Do not reconstruct different corpora for methods being compared.
- Do not report aggregate results without retaining query-level records.
- Do not call an embedding similarity threshold "hallucination detection".
- Do not score abstentions as grounded answers.
- Do not claim novelty as the invention of context sufficiency or abstention.
- Do not copy another thesis's factorial design merely to make the study look larger.
- Do not expand scope unnecessarily after the experiment contract is frozen.
