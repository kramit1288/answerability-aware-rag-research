# Phase 0 Notebook and Prototype Audit

## Status and scope

This document audits `notebooks/original_prototype.ipynb` and its faithful export,
`notebooks/original_prototype.py`, against the research proposal, marker feedback,
`docs/RESEARCH_IMPLEMENTATION_PLAN.md`, and the dataset schemas inspected on 2026-08-17.
It is an implementation audit, not an experimental result. No model was trained and no final
experiment was run during this phase.

The notebook contains 40 cells: 37 code cells and 3 Markdown cells. Every code-cell source is
present verbatim in the Python export. All notebook `execution_count` fields are null and every
cell has an empty `outputs` list. The notebook therefore contains no persisted result that can be
used as thesis evidence. Its large file size is mainly Jupyter widget metadata.

The Python export is a reading aid, not an executable script as written: notebook shell magics
such as `!pip install` remain at Python lines 10-11. Neither original file is to be modified.

## Severity scale

- **P0 — must fix before the final experiment:** invalidates the target, split, comparison,
  metric, or traceability of headline claims.
- **P1 — important for distinction-quality research:** materially weakens interpretation,
  calibration, validation, or robustness.
- **P2 — reproducibility/presentation improvement:** should be corrected, but is not by itself a
  reason to reject the central experiment.

## Executive finding

The current classifier does **not** predict retrieved-context sufficiency. It predicts

\[
y_i = \mathbb{1}[\neg\texttt{is\_impossible}_i],
\]

a global question-level label, from features computed for one hybrid retrieval condition at
`TOP_K = 5`. Consequently, an answerable question whose current retrieval misses its evidence
is still labelled positive. This breaks the central claim that the gate determines whether the
retrieved context for the current condition is sufficient. All downstream “unsafe answer,” false
abstention, calibration, threshold, and policy analyses inherit this target mismatch.

The other decisive P0 issue is corpus construction. The release includes an official fixed corpus
of 28,481 documents. The prototype instead reconstructs a 496-filename candidate collection from
the gold contexts attached to QA rows, including rows later assigned to the test set. This creates
an artificially restricted, query-derived retrieval universe and makes the corpus note in notebook
cell 9 factually outdated.

## Verified dataset facts relevant to the audit

These are schema observations from completed, read-only Python inspections, not estimated values.

### NVIDIA TechQA-RAG-Eval

- Inspected repository revision:
  `0b5bbc84b7f07d6d09d063130e90b716d8d4a32a`.
- Hugging Face exposes one configuration, `default`, and one split, `train`.
- `train.json` SHA-256:
  `69d97231509482ed6bd5ec1c4bc0607acb82a88d11169eb8383592d0ca8b93c7`.
- The loaded file contains 910 rows, although the README says 908.
- Fields are exactly `id`, `question`, `answer`, `is_impossible`, and `contexts`.
- There are 610 rows with `is_impossible == false`, each with exactly one context object, and
  300 impossible rows, each with an empty context list and answer `"-"`.
- A context object contains only `filename` and `text`. There is no chunk ID, passage offset,
  source ID beyond filename, or explicit evidence span.
- IDs retain provenance prefixes: 600 `TRAIN_Q*` and 310 `DEV_Q*`. These are not exposed as
  separate Hugging Face splits.
- There are 496 unique gold filenames; 88 filenames are linked to more than one question.
- There are 29 exact normalized duplicate-question groups; 22 cross the `TRAIN_Q`/`DEV_Q`
  provenance boundary.
- The two provenance groups share 40 gold filenames. Therefore the prefixes alone do not form a
  document-disjoint split.
- Two non-impossible rows have an empty reference answer (`DEV_Q014`, `DEV_Q094`), and one
  (`TRAIN_Q034`) has a URL-only answer that becomes empty under URL-removing normalization.
- `corpus.zip` SHA-256:
  `c06aa287dcc1abf8db6b49b8495df095db73342d729f6451ac330785245d10be`.
  It contains 28,481 uniquely named files under `corpus/`. All 496 QA-linked filenames are
  present, and all 610 attached context texts exactly equal their corresponding corpus files.
  Thus, 27,985 official corpus documents are omitted by the prototype.

Primary source: [NVIDIA TechQA-RAG-Eval](https://huggingface.co/datasets/nvidia/TechQA-RAG-Eval/tree/0b5bbc84b7f07d6d09d063130e90b716d8d4a32a).

### RAGTruth

- Inspected repository revision:
  `c103204b9ce28d6bbad859304bf30de72b8ed8fe`.
- `response.jsonl` SHA-256:
  `e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073`.
- `source_info.jsonl` SHA-256:
  `0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b`.
- There are 17,790 responses and 2,965 sources. Every source has exactly six responses.
- Response fields are `id`, `source_id`, `model`, `temperature`, `labels`, `split`, `quality`,
  and `response`.
- Source fields are `source_id`, `task_type`, `source`, `source_info`, and `prompt`.
- The official response split contains 15,090 train and 2,700 test responses. No `source_id`
  occurs in both splits.
- The QA subset contains 989 sources and 5,934 responses: 839/150 train/test sources and
  5,034/900 train/test responses. QA `source_info` contains `question` and `passages`.
- Every hallucination label contains `start`, `end`, `text`, `meta`, `label_type`,
  `implicit_true`, and `due_to_null`. The last two fields affect whether the claim is factually
  false versus unsupported by the supplied context.

Primary source: [RAGTruth repository](https://github.com/ParticleMedia/RAGTruth/tree/c103204b9ce28d6bbad859304bf30de72b8ed8fe/dataset).

## P0 issue register

| ID | P0 finding | Consequence | Required correction |
|---|---|---|---|
| P0-01 | `answerable = ~is_impossible` is the model target (cells 5, 14-18). | The classifier learns global answerability, not `y_suff(question, retrieved_context)`. | Build one labelled example per query × retriever × k and derive sufficiency from evidence in that exact bundle. |
| P0-02 | QA rows are randomly split without source or duplicate grouping (cell 7). | Same document and duplicate questions can occur across train/validation/test. | Persist a deterministic connected-component group split based on gold filename and normalized-question identity. |
| P0-03 | Corpus is rebuilt only from QA-linked contexts (cell 8). | Test-linked gold documents define the candidate collection; 27,985 official documents are omitted. | Load and checksum the released `corpus.zip`; use one fixed corpus for every split and retriever. |
| P0-04 | Only hybrid retrieval at `k=5` is evaluated (cells 12-14, 33). | RQ1 cannot compare BM25, dense, hybrid, or depth. | Evaluate the controlled 3 × 4 matrix with `k in {1,3,5,10}`. |
| P0-05 | “Recall@k” is a filename hit at one k, with no MRR or query-level ranking artifact (cells 13, 33). | Retrieval claims are incomplete and not reproducible. | Define document Recall@k and MRR@10 using the current one-gold-document schema; persist rankings and denominators. |
| P0-06 | `t_low = t_high - 0.20` and the high-threshold objective contains arbitrary weights (cell 17). | The three-way policy is not jointly or scientifically tuned. | Predeclare a safety constraint and jointly enumerate independent threshold pairs on validation data. |
| P0-07 | Primary end-to-end code collapses request-more-evidence into answer or abstain (cell 23), while the ablation uses a different policy (cell 39). | There is no consistent three-way policy to evaluate. | Define request as an observable third action; if it represents expansion from k=5 to k=10, evaluate that intervention explicitly. |
| P0-08 | Global answerability is used in unsafe-answer and false-abstention denominators (cells 24, 33, 39-40). | Policy metrics do not answer whether the current context was sufficient. | Use `y_suff` and publish numerator, denominator, coverage, and selective risk together. |
| P0-09 | RAGTruth is randomly split at response level (cell 29). | Responses from the same source and the official train/test partitions leak. | Preserve official `split`; group every operation and resample by `source_id`. |
| P0-10 | The RAGTruth classifier is a separate shallow response model, not validation of the TechQA cosine support proxy (cells 29-31). | The notebook cannot claim external validation of its unsupported-claim measure. | Apply the same detector to RAGTruth train/test and evaluate agreement with its human spans/response labels. |
| P0-11 | Cosine similarity threshold 0.42 is arbitrary and cannot test entailment, negation, numbers, or contradiction (cell 21). | “Hallucination” or “unsupported claim” conclusions would be overclaimed. | Retain it only as an embedding proxy; add a pinned NLI detector, RAGTruth validation, and manual validation. |
| P0-12 | Non-answers receive support value zero in ablation rows (cell 39). | Abstention mechanically improves mean unsupported-claim rate. | Store answer-content metrics as `NA`; separately define query-level exposure/harm incidence. |
| P0-13 | The LLM judge omits the question, retrieved context, explicit action, and non-answer text (cell 37). | It cannot evaluate decision appropriateness, context-grounded factuality, or completeness as claimed. | Remove it from headline evidence or redesign the prompt/input and validate the judge independently. |
| P0-14 | Test data are repeatedly evaluated, plotted, sampled, and used for arbitrary ablations throughout cells 15, 18, 24-26, 32-40. | The notebook workflow encourages test-set adaptation. | Create a new immutable grouped split; seal test outcomes until all methods, thresholds, and tests are frozen. |
| P0-15 | No query-level result, split, configuration, model, prompt, or manifest is written to disk. | No thesis number can be reproduced or audited. | Make persisted artifacts the only input to aggregate tables and figures. |
| P0-16 | The retrieval bundle used for labelling/features can differ from the text reaching the generator because of per-chunk, character-budget, and tokenizer truncation (cell 20). | “Sufficient retrieved context” can be present but absent from the actual generation prompt. | Persist both retrieved-bundle and rendered-prompt context IDs/hashes and never conflate them. |

## Cell-by-cell audit

Line references are to `notebooks/original_prototype.py`.

| Notebook cell / Python location | Current behaviour | Reuse disposition | Risks and required change | Priority |
|---|---|---|---|---|
| 1, lines 8-11 — package installation | Installs unpinned latest packages inside the notebook. | Remove from experiment logic; replace with environment installation from project metadata/lock. | Mutable dependencies; export is invalid Python due to `!`; no version capture. | P2 |
| 2, lines 14-53 — imports and seeds | Imports all notebook dependencies, seeds Python/NumPy/Torch, and prints GPU details. | Reuse the intent in a reproducibility utility. | Does not pin versions/revisions, seed CUDA workers, request deterministic algorithms, or save runtime metadata; suppresses all warnings. | P1 |
| 3, lines 56-84 — constants | Defines one `TOP_K=5`, hybrid alpha, chunking, model IDs, thresholds, and safety limit. | Move declared constants into a versioned configuration saved with each run. | No `{1,3,5,10}` matrix; alpha 0.60 and safety 0.20 lack predeclared justification; model revisions are unpinned. | P0/P1 |
| 4, lines 86-99 — revision notes | Claims a dissertation-ready design and full test evaluation. | Keep only as historical commentary. | It does not acknowledge the invalid target or official corpus. No stored output supports its claims. | P1 |
| 5, lines 101-112 — TechQA load | Loads `nvidia/TechQA-RAG-Eval`, implicit default config, only HF `train`, then creates global `answerable`. | Reuse dataset loading only after pinning revision and validating schema. | Central label error; no fingerprint/checksum; actual 910-row schema disagrees with card’s 908 count; provenance prefixes ignored. | P0 |
| 6, lines 115-140 — context normalization | Converts context-like values to lists and counts attached contexts. | Reuse normalization helpers only with strict validation and explicit failure records. | Malformed JSON/string/list values silently become empty lists, converting data errors into apparent unanswerability. `context_count` is attached gold context count, not retrieved context count. | P1 |
| 7, lines 143-161 — split | Performs 70/15/15 row-level stratified random splits. | Remove and replace with persisted group-aware assignment. | Gold filenames, 29 duplicate-question groups, and original provenance can cross splits. Assignments are not persisted. | P0 |
| 8, lines 164-239 — corpus/chunking | Builds and chunks a corpus from attached contexts across all rows; prefix-based deduplication. | Reuse only carefully tested text-cleaning/chunk-loop concepts. | Omits most official corpus; uses test-linked gold contexts to define candidate documents; prefix hashes can collide; no stable IDs, source offsets, manifest, or transform metadata. The regex deleting any line containing “base64” may remove valid prose. | P0 |
| 9, lines 241-247 — corpus note | Calls the query-derived corpus a necessary simplification and an upper-bound limitation. | Remove/replace. | The release actually ships `corpus.zip`. The assertion that gate metrics are not affected is unsupported because retrieval features and distributions depend on this reduced corpus. | P0 |
| 10, lines 249-260 — BM25 | Defines a technical-token regex and builds BM25 over chunks. | Reimplement as a deterministic retriever with stable tie-breaking and saved config. | No standalone BM25 experiment or ranking artifact; tokenizer definition is not versioned/tested. | P1 |
| 11, lines 263-277 — dense index | Embeds the reduced corpus with normalized MiniLM embeddings. | Reimplement with pinned model/revision, cache manifest, and deterministic IDs. | Model revision/package/runtime not recorded; only the invalid corpus is indexed. | P1 |
| 12, lines 280-319 — hybrid retrieval | Min-max normalizes full-corpus BM25/dense scores, mixes them with alpha 0.60, and returns hybrid top-k. | Retain score extraction ideas; replace fusion with predeclared equal-weight RRF for the primary experiment. | No BM25/dense arms; query-wise min-max and alpha are arbitrary; tie order is not explicit; no complete rankings persisted. | P0 |
| 13, lines 322-355 — retrieval evaluation | Reports whether any retrieved chunk has a gold filename for answerable questions. | Reuse gold-filename extraction after schema validation. | At current schema it is document Success/Recall@k because there is one gold doc, but it is not chunk evidence sufficiency. It evaluates only hybrid k=5, lacks MRR, prints aggregates, and touches test early. | P0 |
| 14, lines 358-410 — features and labels | Computes hybrid-ranked score/overlap/length features and labels each question with global `answerable`. | Several feature formulas are reusable after renaming and condition-aware implementation. | Wrong target and experimental unit. `top_dense`/`top_bm25` are scores of the hybrid-top chunk, not necessarily the top dense/BM25 scores. No strategy/k/context identity. Corpus leakage remains. | P0 |
| 15, lines 413-455 — RF and isotonic calibration | Fits one Random Forest, calibrates on all validation rows, reports validation and test classification at 0.5. | Reimplement RF and calibration behind saved pipelines. | Missing required baselines. Validation calibration performance is in-sample; threshold tuning later reuses the same values. Test is inspected before design freeze. Target is global answerability. | P0 |
| 16, lines 458-488 — impurity importance | Plots Gini feature importances and claims to address RQ3. | Replace headline interpretation with permutation importance and uncertainty. | Gini importance is biased toward continuous/high-cardinality variables; RQ mapping is outdated; no artifact. | P1 |
| 17, lines 491-586 — threshold tuning | Searches one answer threshold under an unsafe-rate constraint and weighted score; derives lower threshold by subtracting 0.20. | Retain validation-only enumeration concept. | Arbitrary weighted objective and lower-threshold relation; not a joint three-action optimization; global labels; safety tolerance not justified. | P0 |
| 18, lines 589-659 — test probability plot | Visualizes test probabilities, tuned thresholds, and decision-region counts. | Recreate from final persisted predictions only. | Direct test inspection during development; still global label. | P0 |
| 19, lines 662-682 — generator load | Loads FLAN-T5-large with environment-dependent device placement. | May be retained as a compute-feasible fixed generator after revision pinning. | Model revision, tokenizer, dtype, device map, and package versions are not saved. | P1 |
| 20, lines 685-783 — prompt/generation | Builds a grounded prompt, applies character budgets and tokenizer truncation, and decodes deterministically. | Prompt discipline and deterministic decoding are reusable. | Retrieved `C` differs from rendered prompt after clipping/truncation; no prompt/version/context hash; filenames/scores and fallback behaviour are not controlled across comparisons. | P0 |
| 21, lines 786-860 — support proxy | Splits answers into sentences, compares embeddings with at most 30 context chunks, and marks scores below 0.42 unsupported. | Keep only as explicitly named exploratory embedding-support proxy. | Sentence is not necessarily an atomic claim; arbitrary threshold; similarity is not entailment; first-30 truncation can hide support; no-claim output is forced to 0 rather than NA. | P0 |
| 22, lines 862-866 — interpretation note | Correctly warns that similarity is a proxy, not a verified hallucination label. | Retain this scientific caveat. | The surrounding plots and variable names still risk stronger interpretation than the note permits. | P1 |
| 23, lines 868-1059 — full policy | Scores k=5, expands middle-band cases to k=10, rescoring with the same model, and converts failures to abstentions; may use extractive fallback. | Reuse only small deterministic helpers after redesign. | k=10 features are out of the k=5 training regime; final output erases the request action; fallback confounds generator/policy comparison; non-answer support is zero; duplicate feature code can diverge. | P0 |
| 24, lines 1062-1191 — end-to-end evaluation | Generates on the test split, reports decision metrics, ROUGE-L, retrieval hit, and unsupported rate in memory. | Row assembly is a useful prototype for the final query-level schema. | UAR/FAR use global answerability; context and retrieval condition are missing; filename hit is not sufficiency; no file is written. Content metrics are correctly NA for most non-answers here, but this is inconsistent elsewhere. | P0 |
| 25, lines 1194-1256 — error inspection | Displays unsafe, false-abstention, low-grounding, and wrong-retrieval candidates. | Recreate from frozen final artifacts. | Categories inherit invalid labels and arbitrary proxy cutoffs; repeated test inspection; display only. | P0/P1 |
| 26, lines 1259-1326 — manual sample | Takes up to three first rows per derived category and leaves notes blank. | Keep the idea, replace with seeded stratified sampling and a saved rubric. | `head()` is not a random or representative sample; labels are not blind/adjudicated; sample and notes are not persisted. | P1 |
| 27, lines 1329-1365 — RAGTruth load | Downloads unpinned `main` JSONL files, merges sources/responses, derives any-label flag, and selects QA. | Reimplement against pinned files and strict schemas. | Broad exception silently makes validation optional; official split is loaded but not enforced; provenance/hashes absent. | P0 |
| 28, lines 1368-1380 — label summary | Counts RAGTruth label types. | Reuse as descriptive manifest reporting. | No official-split/source-aware breakdown and no connection to the TechQA detector. | P1 |
| 29, lines 1383-1452 — RAGTruth RF | Builds shallow response/context features, random-splits responses 75/25, and predicts any-label status. | Remove from the primary grounding method; may remain a separately named baseline only. | Official split and `source_id` grouping are violated; six responses per source leak; it does not validate the cosine detector; span labels and `implicit_true` semantics are discarded. | P0 |
| 30, lines 1455-1586 — RAGTruth plots | Scores the full dataset with the just-trained model and plots label/risk summaries. | Rebuild only from official held-out predictions if retained. | Predictions mix train and test/in-sample rows. Hallucination rate by dominant type is partly tautological. No uncertainty or saved rows. | P0/P1 |
| 31, lines 1589-1613 — transfer proxy | Applies the RAGTruth RF to TechQA answers. | Remove from headline analysis. | Unvalidated MARCO-to-TechQA transfer; risk is also computed for refusal/non-answer text rather than set to NA. | P0 |
| 32, lines 1616-1676 — demos | Selects the highest-confidence “clean” test answer and runs a hand-written failure probe. | Use only as a clearly labelled qualitative illustration after final evaluation. | Outcome-based test cherry-picking; failure probe has no gold label or persisted audit trail. | P1 |
| 33, lines 1679-1758 — metric matrix | Recomputes hybrid Recall@5, global-label classifier metrics, and end-to-end aggregates. | Replace with a script that aggregates immutable query-level artifacts. | Wrong target/conditions; no k matrix, MRR, Brier/ECE, coverage-risk curve, uncertainty, or persistence. | P0 |
| 34, lines 1761-1881 — figures | Plots retrieval, classifier/system confusion matrices, ROC, and key rates from test. | Recreate from saved final results. | Test repeatedly inspected; coverage is absent from the compact safety chart; calibrated ROC does not measure calibration. | P0/P1 |
| 35, lines 1884-1983 — BERTScore | Computes BERTScore only for answerable and answered cases. | Retain as a secondary answer-quality metric with a pinned evaluator. | Correctly avoids zeroing non-answers, but must always report denominator/coverage; evaluator revision and package are unpinned and `bert-score` is not in project dependencies. | P1 |
| 36, lines 1986-2015 — confidence/quality correlation | Correlates gate probability with BERTScore among selected answered cases. | Remove as a headline analysis or label exploratory. | Conditioning on the gate decision causes range restriction/selection bias; Pearson assumptions and uncertainty are not assessed; BERTScore is called “actual” quality. | P1 |
| 37, lines 2018-2245 — LLM judge | Samples 100 test rows and asks the same FLAN-T5 model for A-E grades on four dimensions. | Remove unless completely redesigned and validated. | Prompt omits question, context, action, and non-answer reason; non-answers appear as `nan`; retry omits criterion/reference; parser can accept incidental letters; self-judging is non-independent. | P0 |
| 38, lines 2248-2312 — calibration | Refits isotonic calibration on validation and reports test Brier/ECE/reliability bins. | Reuse formula definitions after creating cross-fitted validation predictions and saved bins. | Duplicates earlier calibrator; threshold selection reused the fitted validation points; ECE binning has no declared small-sample policy; no artifact/CI. | P1 |
| 39, lines 2315-2436 — ablation | Compares tuned, manually conservative/aggressive, binary, and always-answer variants on full test. | Replace with baselines and ablations frozen before test. | Multiple arbitrary policies are tried on test; policy semantics differ from cell 23; non-answer support is zero and averaged, rewarding abstention; labels remain global. | P0 |
| 40, lines 2439-2615 — bootstrap | Row-resamples each metric 1,000 times and reports percentile intervals. | Reuse only the general bootstrap intent. | “Paired” helper does not compare systems; no query/source clustering across repeated conditions; class-denominator resamples can be undefined; no paired tests, effects, or multiplicity correction; global RNG makes results order-dependent. | P0/P1 |

## Metrics and research-question alignment

| Research question | What the prototype currently supplies | Why it is insufficient |
|---|---|---|
| RQ1: retriever/depth effect on sufficiency | Hybrid filename hit at k=5. | No BM25/dense arms, depth manipulation, MRR, evidence-level sufficiency, paired comparisons, or query-level artifact. |
| RQ2: predictive features and calibrated model vs baselines | One RF trained on global labels; Gini importance. | Wrong target, no score/logistic/uncalibrated baselines, no condition-aware observations, and biased importance. |
| RQ3: three-way safety/coverage policy | One high threshold and derived lower threshold; inconsistent request semantics. | Thresholds are not jointly tuned; request disappears in primary evaluation; UAR/FAR use global labels; no risk-coverage/AURC. |
| RQ4: unsupported claims and detector reliability | Arbitrary cosine proxy plus a separate leaking RAGTruth RF. | No validation of the same detector, no NLI/entailment, no source-held-out results, and non-answer zeros bias one analysis. |

## Reusable elements

The prototype is useful as a behavioural sketch, not as final research logic. The following ideas
can be reimplemented with tests and configuration:

- deterministic tokenization, overlap, and score-summary helpers;
- BM25 and normalized dense embedding retrieval mechanics;
- explicit prompt instruction to use only supplied evidence;
- deterministic generation rather than sampling for the primary comparison;
- conditional `NA` handling for ROUGE-L/BERTScore in cell 24/35;
- query-level row assembly begun in `evaluate_system`;
- the explicit caveat that cosine support is only a proxy;
- validation-only threshold-search intent and bootstrap-CI intent.

No aggregate value printed by the notebook should be copied into the thesis. There are no stored
outputs, and the present target/corpus/split make any rerun of this exact notebook unsuitable as
the final experiment.

## Proposal, marker, plan, and implementation discrepancies

1. The proposal says the classifier predicts adequacy of retrieved information, but its stated
   70/15/15 row-stratified implementation and the prototype use global `is_impossible`.
2. The proposal describes BM25 and dense retrieval, whereas the prototype exposes only their
   fixed-weight hybrid and never makes controlled standalone comparisons.
3. The proposal says thresholds are validation-tuned, but the lower threshold is manually tied to
   the upper threshold and several extra policies are evaluated directly on test.
4. The marker request for mathematical parameters, baselines, validation, statistics, and
   measurable contribution is only partly addressed. Definitions, dependency-aware tests,
   effects, multiplicity control, and artifact traceability are absent.
5. The implementation plan correctly requires the official fixed corpus, but the prototype’s
   statement that no separate corpus is available is false for the inspected release.
6. The plan requires official RAGTruth split/source grouping, whereas the prototype discards both.
7. The plan requires quality with coverage and `NA` for non-answer content metrics. Cell 24/35
   partly comply; cell 39 does not.
8. The intended contribution must remain the empirical combination of technical-document QA,
   retrieval-conditioned sufficiency, a lightweight calibrated gate, controlled retrieval
   variation, and safety/coverage/grounding validation. Neither context sufficiency nor
   abstention is a defensible novelty claim by itself.

## Audit conclusion

The original implementation is not suitable for a final experimental run. The three blocking
corrections are: use the complete fixed corpus, create leakage-resistant persisted question groups,
and replace the global target with a condition-specific sufficiency label. The final methodology
that resolves these issues is specified in `docs/EXPERIMENT_CONTRACT.md`. Phase 1 must stop at
the data foundation and split manifests described in `docs/PHASE_01_PLAN.md`; it must not begin
retrieval or modelling.
