# Codex Prompts

Run Codex from the repository root. Use one phase at a time.

## Phase 0 — first prompt to paste into Codex

Read `AGENTS.md`, `docs/RESEARCH_IMPLEMENTATION_PLAN.md`, `docs/MARKER_FEEDBACK.md`, and inspect `notebooks/original_prototype.ipynb` completely.

Do not modify the implementation yet.

Audit the current notebook against the implementation plan and marker feedback. Create:

1. `docs/NOTEBOOK_AUDIT.md`
2. `docs/EXPERIMENT_CONTRACT.md`
3. `docs/PHASE_01_PLAN.md`

The audit must identify exact notebook cells/functions that can be reused, experimental-validity problems, leakage risks, label-definition problems, thresholds or metrics that are not scientifically justified, and anything that would make the final thesis claims unsupported.

For the experiment contract, inspect the live/current schemas of TechQA-RAG-Eval and RAGTruth using executable code if network access is available. Do not assume field names from memory. Define the exact experimental unit, corpus source, split/grouping strategy, context-sufficiency label rule, retrieval conditions, models, metrics, calibration method, policy threshold selection, statistical tests, and persisted artifact schemas.

Do not fabricate results. Do not use the test set for tuning. Stop after writing the three documents and report unresolved methodological decisions that genuinely require my input.

Run any lightweight checks necessary to validate notebook parsing and dataset schemas.

## Phase 1

Implement Phase 1 from `docs/RESEARCH_IMPLEMENTATION_PLAN.md` and the approved `docs/EXPERIMENT_CONTRACT.md`.

Move reusable data logic into `src/answerability_rag/`, add tests, and create a reproducible script under `scripts/`.

Do not implement retrieval/model training yet.

Generate the required dataset manifests and split artifacts. Verify no group leakage and stable splits. Report commands, tests, artifact row counts, label distributions, and any deviation from the contract.

## Phase 2

Implement only the controlled retrieval experiment from Phase 2 of the plan using the frozen Phase 1 data contract.

Implement BM25, dense and hybrid retrieval with k in {1,3,5,10}. Keep corpus, queries and preprocessing controlled across methods. Persist query-level rankings, scores and retrieval metrics. Add tests for ranking shape, stable IDs, duplicate handling and metric correctness.

Do not train the answerability classifier yet.

## Phase 3

Implement the retrieved-context sufficiency label pipeline from Phase 3.

The target must describe whether the actual retrieved context for a query/retrieval-condition is sufficient to support the ground-truth answer. It must not simply reuse a global question-answerability label.

Implement provenance for every label and a manual-validation sample. Stop and report if the dataset does not support a defensible automatic label rule.

## Phase 4

Implement the baselines, Logistic Regression, uncalibrated Random Forest, calibrated Random Forest, calibration metrics and three-way decision policy from Phase 4/5.

Tune only on validation data. Keep the test set untouched until the final evaluation command. Jointly tune low/high decision thresholds using the criterion defined in `docs/EXPERIMENT_CONTRACT.md`.

Generate reliability and risk-coverage artifacts.

## Phase 5

Implement answer generation and support/faithfulness evaluation.

Do not assign generation-quality scores of zero to abstain/request-more-evidence outcomes. Those should be NA and analyzed through coverage/policy metrics instead.

Keep the embedding-similarity method only as an explicitly named proxy. Add and validate the stronger support evaluator specified in the experiment contract.

## Phase 6

Implement the pre-specified statistical analysis and thesis figure generation.

Every statistical test must operate at the correct unit of observation and account for paired/dependent observations where appropriate. Generate 95% confidence intervals, effect sizes and corrected p-values where specified.

Create all tables/figures from persisted query-level result artifacts, not hard-coded values.

## Final freeze

Run the final reproducibility pipeline from clean inputs. Freeze configs, package versions, seeds, model names/revisions, dataset fingerprints and prompts.

Produce `artifacts/results/run_manifest.json` and the final thesis-ready result bundle.

Then write `docs/FINAL_EXPERIMENT_SUMMARY.md` containing only results that are present in generated artifacts, with exact artifact references for every number.
