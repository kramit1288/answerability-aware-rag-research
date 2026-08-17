# AGENTS.md — Research Engineering Instructions

## Purpose

This is an MSc research repository, not a product prototype. Correct experimental design, reproducibility, traceability and defensible claims are more important than code cleverness.

## Required reading before modifying research logic

Read these files first:

1. `docs/RESEARCH_IMPLEMENTATION_PLAN.md`
2. `docs/MARKER_FEEDBACK.md`
3. `notebooks/original_prototype.ipynb`
4. `references/research_proposal.pdf` when proposal wording or scope needs to be checked

## Scientific integrity rules

- Never invent, estimate, interpolate or backfill experimental results.
- Never write a result into code/docs unless it came from a completed run and is persisted in an artifact.
- Preserve raw downloaded datasets; derived datasets must have a generation script and metadata.
- Fix and record random seeds.
- Persist split assignments so train/validation/test membership cannot change between runs.
- Do not use the test set for feature engineering, threshold tuning, model selection or calibration.
- Avoid query/response-level leakage across source documents. Use group-aware splitting where source/document identifiers permit it.
- Respect official dataset train/test partitions where available, especially RAGTruth.
- Distinguish:
  - global/question answerability
  - retrieved-context sufficiency for a specific retrieval condition
  These are not interchangeable labels.
- The answerability classifier must predict **retrieved-context sufficiency**, not merely the dataset's global question-answerability label.
- Every metric must have a precise mathematical or programmatic definition.
- Content-quality metrics for cases where the system does not answer should normally be missing/NA, not forced to zero.
- Always report quality together with coverage for selective/abstaining systems.
- Treat embedding-similarity unsupported-claim scoring as a proxy unless validated.
- Do not claim "hallucination detection" from cosine similarity alone.
- Record library/model versions and final configuration.

## Experimental requirements

The final study should include, at minimum:

- Retrieval strategies: BM25, dense, hybrid.
- Retrieval depths: k in {1, 3, 5, 10}.
- Retrieval metrics including Recall@k and preferably MRR.
- A context-sufficiency label tied to the actual retrieved context for each condition.
- Baselines:
  1. always-answer
  2. simple retrieval-score/heuristic threshold
  3. Logistic Regression
  4. uncalibrated Random Forest
  5. calibrated Random Forest
- Calibration metrics: Brier score and ECE, plus a reliability diagram.
- Three-way policy: answer / request-more-evidence / abstain.
- Joint tuning of low/high thresholds on validation data.
- Policy metrics: unsafe answer rate, false abstention rate, coverage, selective risk, and AURC/risk-coverage curve.
- Unsupported-claim/faithfulness analysis with validation against RAGTruth and/or a manually reviewed sample.
- 95% confidence intervals.
- Paired statistical tests when comparing systems on the same questions.
- Effect sizes and multiple-comparison correction when relevant.

## Engineering expectations

- Move reusable logic from the notebook into `src/answerability_rag/`.
- Use thin scripts in `scripts/` to run each experiment.
- Add unit tests for deterministic logic and data-integrity checks.
- Functions should be small, typed where practical, and documented where scientific interpretation matters.
- Do not hide important experimental constants inside notebook cells.
- Prefer config objects/files that are saved with each run.
- All result tables should be reproducible from raw/derived artifacts by script.

## Before considering a milestone complete

Run the relevant tests and data-integrity checks. Report:

1. files changed
2. commands run
3. tests/checks passed or failed
4. artifacts generated
5. unresolved methodological issues
6. any deviation from the research plan and why

Do not proceed to the next research milestone merely because code executes. The milestone is complete only when its acceptance criteria in the implementation plan are satisfied.
