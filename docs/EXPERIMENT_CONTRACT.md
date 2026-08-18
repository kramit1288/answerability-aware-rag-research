# Experiment Contract

## 1. Purpose, status, and change control

This contract defines the final experimental methodology before implementation. It replaces the
methodologically invalid parts of the prototype while retaining the approved research scope. It
does not contain experimental results.

The study evaluates a **retrieval-conditioned context-sufficiency gate** for technical-document
question answering. It does not claim to invent context sufficiency, selective prediction,
abstention, calibration, or hallucination evaluation. Its intended contribution is the controlled
combination of:

- technical-documentation QA;
- a context-sufficiency target tied to each retrieved bundle;
- lightweight retrieval/context-derived prediction;
- BM25, dense, and hybrid retrieval-quality variation;
- a calibrated two-stage policy whose first-stage actions are answer, retrieval expansion, or
  abstain and whose final user-visible outcome is answer or abstain;
- explicit safety-versus-coverage evaluation; and
- externally and manually validated grounding analysis.

After approval, every result-changing change to this contract must be added to
`docs/RESEARCH_DECISIONS.md` before test evaluation. A post-test change must be identified as
post hoc and cannot silently replace the preregistered primary analysis.

## 2. Research questions

### RQ1 — Retrieval and sufficiency

How do retrieval strategy and retrieval depth affect whether the retrieved context is sufficient
for technical-documentation questions?

Primary outcomes: document Recall@k, MRR@10, context-sufficient rate, and paired differences
across BM25, dense, and hybrid retrieval at `k in {1,3,5,10}`.

### RQ2 — Prediction and calibration

Which retrieval and context characteristics predict retrieved-context sufficiency, and does a
calibrated multivariate classifier improve on always-sufficient, single-score, linear, and
uncalibrated tree baselines?

Primary outcomes: AUROC, AUPRC, F1 at the validation-selected operating point, Brier score,
ECE, reliability plots, and permutation importance. “Improve” must be supported by paired effect
estimates and confidence intervals, not only different point estimates.

### RQ3 — Retrieval-expansion decision policy

How does a calibrated policy that can answer, request additional retrieval evidence, or abstain at
`k=5`—and that resolves expansion cases once at `k=10`—change unsafe answering, false abstention,
retrieval expansion, coverage, and selective risk relative to an always-answer RAG system and
simpler gates?

Primary outcomes: the full risk-coverage relationship, AURC, unsafe-answer-versus-coverage
trade-offs, and final false abstention. Predeclared 5%, 10%, and 20% validation selective-risk
constraints define secondary operating points; 10% is the central illustration, not a universal
acceptable safety threshold.

### RQ4 — Grounding among generated answers

Does gating reduce exposure to unsupported generated claims, and how reliably does the automatic
grounding measure agree with RAGTruth and blinded manual judgement?

Primary outcomes: unsupported-claim rate conditional on answering, unsupported-answer exposure
per query, RAGTruth response/claim-level discrimination, manual agreement, and answer quality
reported with coverage.

## 3. Notation and experimental unit

Let:

- `q_i` be TechQA question `i` with stable `question_id`;
- `r` be retrieval strategy in `{bm25, dense, hybrid_rrf}`;
- `k` be requested retrieval depth in `{1,3,5,10}`;
- `R_(i,r)` be the complete deterministic ranking for question `i` under strategy `r`;
- `B_(i,r,k)` be the ordered top-k chunk bundle, a prefix of `R_(i,r)`;
- `C_(i,r,k)` be the full serialized retrieved context represented by those chunk IDs and stored
  chunk texts;
- `D_(i,r,k)` be document retrieval success: at least one retrieved chunk originates from the
  gold filename;
- `E_i = {E_i1, ..., E_im}` be the manually or automatically aligned alternative gold evidence
  sets, where each evidence set may require one or more source spans;
- `G_(i,r,k)` be the exact rendered context that reaches the generator after any deterministic
  prompt-budget handling; and
- `y_suff(i,r,k)` be 1 only if `C_(i,r,k)` contains enough evidence to recover the reference
  answer, and 0 otherwise.

The primary classifier observation is

\[
e_{i,r,k} = (q_i, r, k, C_{i,r,k}, x_{i,r,k}, y^{suff}_{i,r,k}),
\]

where `x` contains only information available from the query, retrieval scores/ranks, and
retrieved context. There are up to 12 observations per question, but all observations from a
question share one train/validation/test assignment. Repeated conditions are dependent and must
remain together in cross-validation, bootstrap resampling, and paired tests.

Operationally, `C_(i,r,k)` is the ordered set of the actual retrieved chunk IDs and their complete
stored texts under that condition. For a gold evidence set `E_ij`, let `covered(E_ij,C)` mean that
the union of source-word intervals represented in the retrieved chunks contains every required
span in `E_ij` (or that an adjudicated equivalent evidence span is present). For benchmark-
answerable questions:

\[
y^{suff}_{i,r,k}=
\mathbb{1}\left[\exists E_{ij}\in E_i:\;covered(E_{ij},C_{i,r,k})\right],
\]

subject to the explicit alternate-evidence/manual rules in Section 6. In contrast,

\[
D_{i,r,k}=\mathbb{1}\left[\exists c\in B_{i,r,k}:doc(c)=gold\_doc_i\right].
\]

`D=1` does **not** imply `y_suff=1`: the correct document may be retrieved through a chunk that
does not contain the required answer evidence. A document hit is a retrieval result; sufficiency
is a property of the actual chunk bundle.

`C` and `G` must not be conflated. Retrieval/sufficiency claims concern `C`. Generation and
grounding claims concern the exact prompt-visible `G`. Both receive stable hashes. If generator
budgeting removes evidence that exists in `C`, that is recorded as a prompt-assembly failure, not
silently attributed to retrieval or the gate.

For retrieval rankings, top-k bundles are strict prefixes of one top-10-or-deeper ranking. This
ensures that increasing k does not replace higher-ranked chunks and makes evidence sufficiency
monotone unless a manual label inconsistency is discovered.

For the primary policy, the evaluation unit is one question-level trajectory beginning with
`e_(i,hybrid,5)`. A medium-confidence first-stage action adds `e_(i,hybrid,10)`, recomputes the same
feature schema and calibrated sufficiency score, and then ends in answer or abstain. The trajectory
stores both condition-level observations; it is not treated as two independent questions.

## 4. Datasets, fixed corpus, and leakage controls

### 4.1 TechQA-RAG-Eval primary dataset

The primary release is:

- dataset: `nvidia/TechQA-RAG-Eval`;
- configuration: `default`;
- exposed split: `train`;
- pinned revision: `0b5bbc84b7f07d6d09d063130e90b716d8d4a32a`;
- `train.json` SHA-256:
  `69d97231509482ed6bd5ec1c4bc0607acb82a88d11169eb8383592d0ca8b93c7`;
- `corpus.zip` SHA-256:
  `c06aa287dcc1abf8db6b49b8495df095db73342d729f6451ac330785245d10be`.

The current executable inspection found one 910-row split with fields:

| Field | Type and role |
|---|---|
| `id` | Unique string question ID; retains `TRAIN_Q`/`DEV_Q` provenance prefix. |
| `question` | Non-empty question string. |
| `answer` | Reference answer string; `"-"` for all impossible rows; `DEV_Q014` and `DEV_Q094` are non-impossible rows with empty references and must be explicitly flagged. |
| `is_impossible` | Global benchmark answerability indicator. It is not the classifier target. |
| `contexts` | List of `{filename, text}`; exactly one for each of 610 non-impossible rows and empty for 300 impossible rows in the pinned revision. |

The README’s stated 908 records must not override the observed 910 records. Schema/count checks
are pinned to the inspected file and must fail loudly if a later revision differs.

### 4.2 Official corpus

The searchable collection is the 28,481-file `corpus.zip`, not a collection reconstructed from QA
contexts. All documents are available to every split because they represent the fixed external
documentation collection, not learned labels. Corpus visibility is not permission for test labels,
answers, evidence mappings, or outcomes to enter model selection.

Each document receives a stable ID derived from its unique archive path and records raw and
normalized SHA-256 hashes. The 610 attached gold context texts must match their corpus files, as
they did in the Phase 0 inspection. Cleaning is deterministic and audit-preserving:

1. decode UTF-8 strictly;
2. normalize line endings and Unicode to NFKC;
3. collapse only repeated whitespace for the retrieval representation;
4. remove only syntactically recognized data-URI/base64 payload tokens, recording every removal;
5. never delete an entire line merely because it contains the word `base64`.

The original raw archive and JSON remain immutable. Normalized documents are derived data with a
transform version and input/output hashes.

Before any retrieval ranking or test result was produced, the prototype's 360-word/80-word-overlap
scheme was replaced because it can exceed the pinned MiniLM model's 256-word-piece maximum and
silently make lexical and dense retrieval represent different portions of a nominal chunk. The
primary chunker uses the fast tokenizer belonging to the pinned
`sentence-transformers/all-MiniLM-L6-v2` revision, with `add_special_tokens=False`, no truncation
while determining boundaries, a maximum content length of 224 tokenizer tokens, overlap 48, and
step 176. One final partial chunk is retained; processing stops when its exclusive token end equals
the document token count, so no redundant overlap-only tail is emitted.

Tokenizer offset mappings define exclusive `token_start`, `token_end`, `char_start`, and
`char_end` positions in the canonical normalized retrieval document. The materialized chunk text
must equal `normalized_document[char_start:char_end]` exactly. The manifest retains the official
filename/document ID, raw source SHA-256, normalized source SHA-256, retrieval-source transform
version, chunk index, token and character offsets, token and character lengths, source-exact chunk
text, chunk-text SHA-256, and chunk-configuration SHA-256. This explicitly distinguishes raw
archive provenance from offsets into the normalized retrieval representation when normalization
prevents raw-string offset equivalence.

Stable chunk IDs are SHA-256 hashes of a canonical JSON object containing the schema/chunker
version, document ID, normalized source SHA-256, chunk index, token and character offsets, and
chunk-configuration SHA-256. They never depend on questions, answers, gold labels, split
membership, or processing order. Every non-empty document containing at least one tokenizer token
must produce a chunk. Chunking sensitivity is not a primary factor; any later sensitivity run is
secondary and must not choose the test-optimal setting.

### 4.3 TechQA split construction

The approved primary split is a custom, persisted group-aware 70/15/15 split over all 910 rows.
The current Hugging Face release exposes only one split. Although IDs contain `TRAIN_Q` and
`DEV_Q` provenance, those prefixes are not a leakage-safe boundary: the groups share gold
documents and exact normalized duplicate questions. They are retained only as provenance.

Groups are connected components of a graph in which two questions are connected if they share:

- a gold `contexts.filename`; or
- the SHA-256 of a normalized question (Unicode NFKC, casefold, punctuation-to-space, whitespace
  collapse).

Impossible rows have no gold filename and are grouped by normalized-question hash. The original
ID prefix is retained as `provenance_partition` but is not used as the primary split label.
Component IDs and optimizer tie keys are derived from a canonical signature of sorted normalized-
question hashes and sorted gold filenames, not from `provenance_partition`; question IDs are
retained as component members for traceability.

Components are assigned with a deterministic constrained optimizer to approximate 70% train, 15%
validation, and 15% test. It first minimizes the maximum normalized absolute deviation across each
split’s target total, answerable count, and impossible count; it then minimizes their summed
normalized absolute deviation while holding the first objective at its optimum. Equivalent optimal
solutions use a seed-42, provenance-neutral component-signature tie rule. No component may be
split, and a non-optimal solver status stops the phase. The objective, solver/version, component
manifest, and assignments are persisted. All 12 retrieval-condition examples inherit their
question’s split.

Once integrity checks pass, the component manifest and split-assignment file are frozen by
SHA-256 and reused unchanged by every later phase, including feature extraction, calibration,
policy evaluation, generation, and statistical analysis. A later reassignment requires an explicit
pre-test contract change; no row may be moved to improve class balance or results after inspection.

The global label may be used to stratify split assignment; it may not be used as the learned
sufficiency target. The test assignment is sealed before retrieval features, labels, models,
calibration, thresholds, or statistical choices are adjusted. Data-integrity checks may count and
hash test rows but may not inspect model outcomes.

### 4.4 RAGTruth external validation dataset

The external grounding-validation release is:

- repository revision: `c103204b9ce28d6bbad859304bf30de72b8ed8fe`;
- `response.jsonl` SHA-256:
  `e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073`;
- `source_info.jsonl` SHA-256:
  `0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b`.

The full release contains 17,790 responses for 2,965 sources, exactly six responses per
`source_id`. The QA subset has 989 sources and 5,934 responses. QA source information contains
`question` and `passages`. The primary grounding validation uses only `task_type == "QA"`.

The released response `split` is authoritative: QA has 839 train/150 test sources and
5,034 train/900 test responses. No `source_id` crosses the official split. Detector thresholds and
all design decisions use official train only, with group cross-validation by `source_id` where
needed. Official test is evaluated once. Resampling and uncertainty use `source_id` as the cluster.

The primary detector-validation population uses `quality == "good"`; `incorrect_refusal` and
`truncated` rows remain in the manifest and are reported in a predeclared all-quality sensitivity
analysis. Labels with `implicit_true == true` are still unsupported by the supplied context and
count as ungrounded for the documentation-grounding task. A separate sensitivity excludes them
when discussing factual hallucination. `due_to_null` and label type are retained rather than
collapsed invisibly.

## 5. Controlled retrieval experiment

### 5.1 Shared controls

Every strategy uses exactly the same normalized chunks, questions, split assignments, and maximum
ranking depth. Stable ascending `chunk_id` breaks exact score ties. Library and model revisions,
tokenizer, chunking, serialization, and hardware are recorded. No retrieval parameter is selected
from validation or test performance.

### 5.2 Strategies

1. **BM25:** `rank_bm25.BM25Okapi` with `k1=1.5`, `b=0.75`, and `epsilon=0.25`.
   Corpus and query text use the identical deterministic preprocessing: Unicode NFKC,
   Unicode casefold, then Python-regex pattern `(?u)\w+(?:[./:-]+\w+)*`. This preserves
   Unicode alphanumeric/underscore identifier segments and internal `.`, `/`, `:`, and `-`
   separators while excluding separators at token boundaries. There is no stop-word removal,
   stemming, or lemmatization. The exact `rank-bm25` version is pinned in the Phase 2 lock and
   run manifest. Parameters are not tuned from validation or test performance.
2. **Dense:** symmetric direct encoding of original question text and source-exact normalized
   chunk text using `sentence-transformers/all-MiniLM-L6-v2` at pinned revision
   `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Queries and passages use the same encode semantics
   with no instruction/prompt prefixes and `normalize_embeddings=True`. Float32 normalized
   embeddings are persisted. Exact float32 dot product (mathematically cosine similarity after
   normalization) uses deterministic blocked NumPy matrix multiplication on CPU. The loaded
   model's embedding dimension is recorded rather than assumed. ANN, quantization, and
   device/backend substitution are forbidden without a documented pre-test amendment supported
   by measured feasibility evidence.
3. **Hybrid RRF:** equal-weight Reciprocal Rank Fusion over the union of the BM25 top 100 and dense
   top 100 constituent rankings:

\[
s_{RRF}(d\mid q) = \mathbb{1}[d\in BM25_{100}]\frac{1}{60 + rank_{BM25}(d)} +
                    \mathbb{1}[d\in dense_{100}]\frac{1}{60 + rank_{dense}(d)}.
\]

The RRF constant 60 and equal weights are fixed a priori; they are not tuned on test. This replaces
the prototype’s query-wise min-max score mixture and arbitrary alpha.

An absent constituent contributes zero. Sort descending by RRF score and break exact ties by
ascending `chunk_id`; retain the final top 10 while persisting both constituent ranks and scores.
Candidate depth 100, constant 60, and equal weights are fixed a priori and are not tuned on
validation or test. This candidate-pool definition supersedes any earlier implication that RRF is
computed over full-corpus constituent rankings.

### 5.3 Depths and retrieval metrics

Each strategy is evaluated at `k = 1, 3, 5, 10`. The pinned TechQA schema has one gold document
for each non-impossible question, so:

\[
Recall@k_i = \mathbb{1}[\text{any top-k chunk has the gold filename}],
\]

and mean Recall@k is the average over non-impossible questions with a valid gold filename. In this
specific schema it is numerically a document hit/success rate; the implementation and thesis must
state that denominator and must not imply multiple graded relevance labels.

Let `rank_i` be the first rank at which a chunk from the gold document occurs. Then

\[
RR@10_i = \begin{cases}1/rank_i,& rank_i\le10\\0,&otherwise\end{cases},
\qquad MRR@10 = \frac{1}{N_{gold}}\sum_i RR@10_i.
\]

Evidence sufficiency is reported separately through `y_suff`; a correct filename hit is not called
sufficient evidence. One deterministic top-10 ranking is persisted per query/strategy and
`k={1,3,5,10}` conditions are nested prefixes of it. Phase 2 aggregate development metrics are
calculated and reported only for train and validation. Test rankings are persisted for later
phases and checked only for non-performance integrity; aggregate test Recall, MRR, strategy
ordering, and performance differences are neither calculated nor exposed until the final frozen
evaluation workflow.

## 6. Retrieved-context sufficiency target

### 6.1 Chunk-level definition

Context sufficiency is evaluated on the chunks actually returned for one retrieval condition, not
on the question in isolation and not on filename retrieval alone:

\[
y^{suff}(q_i,C_{i,r,k}) = 1
\]

if and only if at least one complete admissible evidence set for the answer is covered by the
actual retrieved chunk texts in `C_(i,r,k)`. Otherwise it is 0. A high score, topical relevance, or
a chunk from the correct document is insufficient unless the required evidence is present.

For example, if a gold document contains the answer in words 2,000-2,080 but the retriever returns
a chunk from words 1-360 of that same document, `D_(i,r,k)=1` while `y_suff(i,r,k)=0`. Document
Recall@k and context sufficiency are therefore reported as different outcomes.

### 6.2 Planned labelling hierarchy

Reference answers and gold evidence are used only to construct labels. Reference-answer text or
embeddings, answer length/hash, gold filename/rank, global `is_impossible`, evidence spans, and
manual decisions are forbidden prediction features.

The frozen Phase 3 hierarchy is applied as follows.

1. **Confirmed-gold alignment.** TechQA's source annotation selected a shortest contiguous span
   sufficient to answer the question and excluded questions requiring multiple separate spans or
   multiple Technotes. For every benchmark-answerable row with a confirmed source document, first
   retain all literal source-text occurrences of the reference answer. If none exists, apply the
   deterministic normalization (HTML entity decoding, Unicode NFKC, casefolding,
   punctuation-to-space while retaining canonical URL text, and whitespace collapse) with a
   reversible map to source character offsets. Retain every defensible normalized occurrence,
   including occurrences across multiple confirmed gold documents; never select one duplicate
   arbitrarily. Semantic similarity, fuzzy matching, embeddings, NLI, and LLM output cannot create
   a gold span.
2. **Primary automatic condition label.** For each accepted document/span alternative, merge
   overlapping or adjacent source-character intervals belonging to retrieved chunks from that
   document. Set `y_suff=1` exactly when this union fully covers at least one complete accepted
   span, otherwise set `y_suff=0` for an aligned answerable question. The span may be covered by
   more than one retrieved chunk. A correct-document hit, topical chunk, or partial overlap is not
   sufficient. Persist coverage diagnostics as label-construction metadata that are forbidden
   classifier features.
3. **Benchmark-impossible preliminary negatives.** For `is_impossible == true`, store `y_suff=0`,
   `label_method=benchmark_impossible`, and
   `label_status=preliminary_benchmark_negative`. This is benchmark-relative provenance: the
   original annotation found no answer in its candidate Technotes, but the complete 28,481-file
   research corpus could contain related or supporting material. Phase 4 must report a sensitivity
   excluding these preliminary negatives without rebuilding Phase 3.
4. **Unresolved status.** If an answerable row has no defensible alignment, store `y_suff=NA`, an
   explicit exclusion reason, and `label_status=unresolved`; never force it to zero or drop its
   retrieval-condition rows. `DEV_Q014` and `DEV_Q094` remain governed by this rule unless a
   separate unambiguous manual alignment is persisted.
5. **Optional disagreement screening only.** The already pinned
   `cross-encoder/nli-deberta-v3-base` revision may later flag development disagreements, but it
   cannot replace or automatically promote the gold-span label. Because its threshold requires
   human development annotations, Phase 3 automatic construction does not enable the screen when
   those annotations are absent and does not inspect test disagreements.

The two known answerable rows with empty references, `DEV_Q014` and `DEV_Q094`, are explicitly
flagged as `reference_status=empty_answerable`. If manual review unambiguously aligns the required
evidence to official-corpus spans, those spans and provenance are persisted. If not, their
condition rows remain `y_suff=NA` and are excluded only from gold-evidence-dependent retrieval and
context-sufficiency analyses. The questions remain in the frozen split and descriptive dataset
manifest; no evidence location or answer is fabricated.

Every resolved label has provenance from:
`gold_span_covered`, `alternate_exact_evidence`, `benchmark_impossible_initialization`,
`manual_evidence_alignment`, `manual_override_positive`, or `manual_override_negative`.

### 6.3 Annotation guideline and validation protocol

Before any manual labels are created, an explicit versioned annotation guideline must define:

- the annotation unit `(question, ordered retrieved chunks, retrieval strategy, k)`;
- labels `sufficient`, `insufficient`, and `ambiguous`;
- that all material answer evidence—not merely the correct topic/document—must be in the chunks;
- handling of partial procedures, prerequisites, negation, numerical/version constraints,
  conflicting chunks, duplicate documents, and benchmark-impossible cases;
- how to record required evidence spans, reason codes, confidence, and exclusion status;
- blinded presentation order and prohibited access to classifier probability/policy action; and
- pilot, independent annotation, adjudication, and guideline-version procedures.

The guideline is frozen before the validation sample and its version/hash is stored with every
annotation.

Before classifier development, create a seeded, approximately 150-condition, question-disjoint
manual validation sample from train and validation only. Stratify across automatic positives,
answerable automatic negatives, partial overlaps, correct-document/insufficient-chunk cases,
retrieval failures, benchmark-impossible preliminary negatives, retrieval strategies, and
`k={1,3,5,10}`. Present the question, actual retrieved text, benchmark answer where applicable, and
necessary provenance in a blinded artifact that excludes automatic labels and future model/policy
outputs. Keep the automatic answer key programmatically separate. Target independent annotation by
two humans using `sufficient`, `insufficient`, or `ambiguous`; compute raw agreement and Cohen's
kappa only when two genuine human files exist. If human annotations are unavailable during Phase 3,
produce the complete pack and evaluation tooling but leave validation accuracy/agreement pending.
No manual test sample is drawn or inspected during Phase 3 development.

The frozen 150-condition sample deliberately allocates 30 conditions to each of five mutually
exclusive strata: `automatic_positive`, `partial_overlap`, `correct_document_insufficient`,
`wrong_document_retrieval`, and `benchmark_impossible`. It is therefore not self-weighting.
Unweighted overall sample accuracy may be shown only as a descriptive audit-sample statistic with
an explicit warning; it must not be presented as an estimate of natural TRAIN+VALIDATION label
accuracy. The validation analysis reports metrics separately by sampling stratum, automatic-label
precision/recall/F1 against non-ambiguous human decisions, the ambiguous/unjudgeable rate, and
named disagreement categories. Its population aggregate is constructed from stratum-specific
non-ambiguous confusion rates weighted by the actual resolved TRAIN+VALIDATION condition
frequencies in those five strata. Precision, recall, and F1 are derived from the resulting weighted
confusion proportions rather than averaged naively across strata.

Before any human results are examined, the following internal research-validity gates are frozen;
they are not claimed as universal benchmark thresholds:

- automatic `sufficient` precision against non-ambiguous primary-human judgements targets at least
  `0.90`;
- prevalence-weighted F1 targets at least `0.85`;
- if strictly more than `10%` of non-ambiguous audited `benchmark_impossible` conditions are judged
  context-sufficient, benchmark-impossible negatives are excluded from PRIMARY Phase 4 classifier
  training and retained only for a separately reported sensitivity analysis; and
- human-human Cohen's kappa of at least `0.80` is desirable when two genuine annotators are
  available. A lower value requires reported disagreement analysis and adjudication; it cannot be
  hidden or replaced by a machine annotator.

The first genuine human annotator covers all 150 conditions. If available, a second genuine human
annotator independently covers approximately 100 of the same blinded conditions. A smaller overlap
is permissible only when its exact size and limitation are reported. Raw agreement and Cohen's
kappa use the original overlapping categorical decisions before adjudication. An LLM, Codex, NLI
model, embedding model, or other automatic evaluator is never represented as a human annotator.
The separate automatic answer key is opened by the evaluation workflow only after the complete
first-annotator file passes blinded-schema and completion checks.

## 7. Initial prediction features

All features are computed from `q_i`, retrieval output, and `C_(i,r,k)` before generation. The
initial frozen families are:

- retrieval condition: one-hot strategy, k, actual context count;
- BM25: top score, second score, top1-top2 gap, mean, minimum, maximum, standard deviation over
  retrieved chunks;
- dense: the analogous raw cosine-score summaries;
- hybrid: RRF top score, second score, gap, and summaries when the condition is hybrid;
- score shape: coefficient of variation where defined and entropy of a softmax-normalized top-k
  score distribution;
- retriever agreement: BM25/dense top-k chunk Jaccard overlap and rank correlation over their
  intersection;
- lexical evidence: fraction of unique normalized query tokens present anywhere in `C`, top-chunk
  overlap, mean chunk overlap, and technical-token coverage for error codes/identifiers;
- context composition: retrieved chunk count, unique document count, total word/character count,
  mean/max chunk length, and duplicate/redundancy rate; and
- query descriptors: word count, character count, and technical-token count.

Score names refer to the true maximum/top rank under the named retriever, not merely the BM25 or
dense score attached to the hybrid-top chunk. Missing/undefined features use pipeline imputation
fit on training data only and add a missingness indicator where relevant.

Forbidden features include reference answer text/embedding/length, global `is_impossible`, gold
filename/document rank, evidence spans, `y_suff`, manual labels, generated answers, judge scores,
and any aggregate computed with validation/test labels.

## 8. Baselines and models

| ID | System | Scientific purpose |
|---|---|---|
| B0 | Always-sufficient / always-answer | Measures standard ungated RAG and the maximum-coverage safety cost. For classifier metrics it predicts sufficiency probability 1 for every example. |
| B1 | Single retrieval-score threshold | Tests whether one conventional confidence score is enough. Thresholds are strategy-specific and tuned only on validation after train-only score scaling. |
| B2 | Logistic Regression | Interpretable linear multivariate baseline; tests whether combining weak features helps without nonlinear trees. |
| B3 | Uncalibrated Random Forest | Tests nonlinear discrimination without claiming raw tree vote fractions are calibrated probabilities. |
| B4 | Calibrated Random Forest | Primary lightweight gate; tests whether calibration adds operational value to B3. |

Logistic Regression uses standardized numeric features, one-hot condition features, L2 penalty,
`class_weight="balanced"`, and `C` selected from `{0.01, 0.1, 1, 10}` by grouped five-fold
cross-validation inside training. Random Forest uses 500 trees, seed 42,
`class_weight="balanced_subsample"`; `max_depth in {8,16,None}`,
`min_samples_leaf in {1,5,10}`, and `max_features in {"sqrt",0.5}` are selected by the same
train-only grouped cross-validation. Mean AUROC is primary selection score, mean AUPRC breaks
ties, and the less complex setting breaks any remaining tie. All conditions for one question/group
stay in one fold.

No optional model may replace these baselines after test inspection. Feature ablations and
permutation importance are defined on development data first and run on test only after freezing.

## 9. Probability calibration

The primary RF calibrator is sigmoid/Platt calibration. It is preferred to the prototype’s
isotonic mapping because the effective validation sample size is the number of grouped questions,
not the much larger number of correlated condition rows.

Calibration uses validation data only and avoids evaluating each validation point on a calibrator
that fitted that point:

1. fit the selected base RF on training data;
2. obtain raw RF probabilities for validation;
3. create grouped five-fold validation folds by question/split group;
4. fit the sigmoid calibrator on four folds and predict the held-out fold, producing cross-fitted
   calibrated validation probabilities;
5. tune policy thresholds on those cross-fitted probabilities; and
6. refit one sigmoid calibrator on all validation raw probabilities for the single final test
   transformation.

Isotonic calibration is a preregistered secondary sensitivity, evaluated with the same cross-fit
procedure, but it does not replace the primary sigmoid result based on test performance.

For labels `y_j` and probabilities `p_j`, Brier score is

\[
Brier = \frac{1}{n}\sum_{j=1}^{n}(p_j-y_j)^2.
\]

Expected Calibration Error (ECE) uses ten fixed equal-width bins over `[0,1]`:

\[
ECE = \sum_{b=1}^{10}\frac{|I_b|}{n}
\left|\operatorname{mean}_{j\in I_b}(p_j)-
\operatorname{mean}_{j\in I_b}(y_j)\right|.
\]

Empty bins contribute zero and bin counts are published. Brier, ECE, and a reliability diagram
are reported for raw and calibrated RF probabilities on cross-fitted validation and final test.
ECE is treated as a bin-dependent diagnostic, not a proper scoring rule.

## 10. Three-way policy

### 10.1 Primary operating condition

The primary policy starts with hybrid RRF at `k=5`. BM25 and dense policy results are secondary,
frozen comparisons; the controlled retrieval experiment still evaluates all three retrievers at
every `k` in `{1,3,5,10}` independently of this policy.

Let `p_(i,5)` be the calibrated prediction for `C_(i,hybrid,5)`. For independently selected
`t_low < t_high`, the initial action is:

- `p_(i,5) >= t_high`: answer using the `k=5` context;
- `t_low <= p_(i,5) < t_high`: request additional retrieval evidence; and
- `p_(i,5) < t_low`: abstain.

The middle action is an operational retrieval expansion, not a request to the user. It expands the
same retrieval strategy from `k=5` to `k=10`, constructs the actual expanded context
`C_(i,hybrid,10)`, recomputes the complete frozen retrieval/context feature vector, and obtains
`p_(i,10)` from the same frozen calibrated model. After expansion, the system answers using the
`k=10` context only if `p_(i,10) >= t_high`; otherwise it abstains.

There is no further iteration beyond `k=10`. Retrieval expansion is a recorded intermediate policy
action; the final user-visible action is only `answer` or `abstain`. The system never describes the
middle action as asking the user for clarification or evidence. For evaluation, the final context
depth is `k_i^*=5` for a direct answer or initial abstention and `k_i^*=10` after expansion, and the
corresponding target is `y_i^*=y_suff(q_i,C_(i,hybrid,k_i^*))`. Because the top-5 chunks are a
prefix of top-10 under a fixed ranking, `y_(i,5)=1, y_(i,10)=0` is a label/integrity error requiring
adjudication.

### 10.2 Joint validation optimization

Enumerate every ordered pair from the unique cross-fitted validation probabilities plus endpoints.
For each pair, simulate the complete two-stage policy, including `k=10` feature recomputation for
middle-band examples, and calculate final-action selective risk, coverage, unsafe-answer rate,
false abstention rate, and expansion rate. This produces the full validation safety-coverage
frontier. The primary scientific evaluation is the full risk-coverage relationship and AURC,
together with the unsafe-answer/coverage trade-off; it is not a single risk constraint.

Three operating-point analyses are predeclared with validation selective-risk constraints
`delta in {0.05, 0.10, 0.20}`. For each `delta`, retain threshold pairs whose non-empty answer set
has final selective risk at or below `delta`, then select lexicographically:

1. maximum final answer coverage;
2. minimum retrieval-expansion rate;
3. lower final selective risk; and
4. higher `t_high`, then higher `t_low`, as deterministic ties.

If no non-empty pair satisfies a constraint, report the operating point as infeasible rather than
selecting an answer-empty policy or relaxing the constraint. The 10% constraint is the central
illustrative operating point for detailed policy and generation tables, not a claim of a universal
acceptable safety threshold and not the primary scientific criterion. This is a joint
two-dimensional search; `t_low` and `t_high` are independently tuned and no fixed offset is
allowed. The entire validation grid, frontier membership, constraint-specific feasibility, and
selected pairs are persisted before test is unsealed.

## 11. Policy metrics and denominators

For a test population of `N` primary-policy questions, let `A_i` indicate a final answer, `S_i`
indicate a final abstention, `X_i` indicate that retrieval expansion occurred, and `y_i^*` be
sufficiency of the actual final context used for that decision. Thus `A_i+S_i=1`; `X_i` is an
intermediate action and is not a third final outcome.

| Metric | Definition | Denominator interpretation |
|---|---|---|
| Selective risk | `sum(A_i and y_i^*=0) / sum(A_i)` | Fraction of issued answers whose final answer context is insufficient; NA if no answers. |
| Unsafe answer rate (UAR) | `sum(A_i and y_i^*=0) / sum(y_i^*=0)` | Fraction of all final-context-insufficient cases that are nevertheless answered. |
| Answer coverage | `sum(A_i) / N` | Fraction of all eligible questions receiving a final answer. |
| False abstention rate (FAR) | `sum(S_i and y_i^*=1) / sum(y_i^*=1)` | Fraction of final-context-sufficient cases that receive a final abstention. |
| Retrieval-expansion rate | `sum(X_i) / N` | Fraction for which the policy requests additional retrieval evidence. |
| Expansion answer yield | `sum(X_i and A_i) / sum(X_i)` | Fraction of expanded cases that become final answers; NA if no expansion. |
| Evidence-recovery rate among initial misses | `sum(X_i and y_(i,5)=0 and y_(i,10)=1) / sum(X_i and y_(i,5)=0)` | Among expanded initial evidence misses, fraction made sufficient by `k=10`. |
| Useful answer coverage | `sum(A_i and y_i^*=1) / N` | Fraction receiving an answer backed by sufficient retrieved context, before generation quality. |

Every table stores numerator and denominator. Undefined rates are NA, not zero. Selective risk,
UAR, coverage, and FAR have different denominators and must never be used as synonyms.

For the conventional risk-coverage curve of a frozen classifier at a fixed retrieval condition,
sort questions by decreasing calibrated probability with stable question-ID tie-breaking fixed
independently of labels. At prefix `m`, coverage is `m/N` and risk is the mean of `1-y` among the
top `m`. The discrete AURC is

\[
AURC = \frac{1}{N}\sum_{m=1}^{N} Risk(m).
\]

Lower is better. AURC and the full curve are primary, with per-condition results and the primary
hybrid-`k=5` result reported. Separately, the two-threshold policy frontier persists final risk,
coverage, UAR, FAR, and expansion rate for every threshold pair; it is not relabelled as the
conventional AURC unless an area definition is preregistered. Calibration and policy operating
points are interpreted together; a lower risk does not by itself imply a better system if coverage
collapses.

## 12. Generation and grounding evaluation

### 12.1 When generation occurs

The generator is fixed as `google/flan-t5-large` at revision
`0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a` unless a pre-test resource decision is logged.
Prompt, tokenizer revision, maximum input/output tokens, decoding, dtype, and context rendering
are frozen and hashed.

The always-answer baseline generates once from hybrid `k=5` for every primary-condition question.
Generation is deterministic and cached by `(question_id, context_id, prompt_hash)`. A direct policy
answer exposes the same cached `k=5` answer. If retrieval expansion leads to a final answer, the
generator runs from the expanded `k=10` context and that deterministic context-specific output is
cached and shared by every system using the same context. Retrieval expansion itself does not emit
an answer. This keeps the generator fixed while allowing the policy to use the evidence it actually
retrieved.

For final abstentions, including failed retrieval expansions, generated answer, ROUGE-L, BERTScore,
claim counts, unsupported-claim rate, and judge scores are `NA`. A separate binary
**unsupported-answer exposure** can validly be zero for a non-answer because it measures whether
the system exposed a user to an unsupported answer, not the quality of nonexistent content.

An extractive fallback is not part of the primary system. If studied, it is a named generation
ablation with its own outputs, not a hidden recovery step.

### 12.2 Answer quality

ROUGE-L and BERTScore are secondary reference-answer metrics among answered, benchmark-
answerable rows with usable references. They are reported with answer coverage and denominator.
They are not grounding metrics and do not receive zero for non-answers.

### 12.3 Primary grounding detector

The prototype cosine method is retained only as `embedding_support_proxy`. Its threshold, if used,
is tuned on RAGTruth official train rather than fixed at 0.42.

The primary detector is sentence/claim-level NLI with the pinned
`cross-encoder/nli-deberta-v3-base` model. Deterministic sentence units are paired with each
prompt-visible context chunk as premise/hypothesis; maximum entailment and contradiction scores
are retained. The support threshold is selected on RAGTruth official QA train with grouping by
`source_id`. A response/claim is unsupported when its best evidence does not meet the frozen
support rule. Because a sentence is not always atomic, the thesis calls these “sentence/claim
units” and reports this limitation.

On RAGTruth QA test:

- a response-level gold ungrounded label is `len(labels) > 0` for the primary documentation-
  support task;
- a sentence/claim unit is gold ungrounded if its character interval overlaps any annotated span;
- AUROC, AUPRC, precision, recall, F1, calibration where applicable, and source-clustered 95% CIs
  are reported;
- `implicit_true`-excluded factual-hallucination and all-quality analyses are sensitivities; and
- the same detector implementation and threshold are then applied to TechQA answers.

A separate explicit, versioned grounding guideline defines the sentence/claim unit and the
categories `supported`, `unsupported`, and `cannot_determine`, including rules for negation,
numbers, versions, partial evidence, contradictory documents, and non-atomic sentences. A blinded,
seeded TechQA sample targets 120 answered sentence/claim units, stratified by detector
decision/confidence and policy. All sampled units are targeted for independent annotation by two
annotators, keeping the planned double-annotated sample within approximately 100-150. Raw
percentage agreement and Cohen's kappa are calculated on unadjudicated categorical labels before
adjudication; class metrics and error categories are then reported against the adjudicated labels.
If the final eligible manual sample differs materially, the sample size and reason are documented.
If a second annotator is unavailable, one annotator follows the frozen guideline, raw agreement
and kappa remain `NA`, and the limitation is reported rather than estimated. Without adequate
manual validity, automatic unsupported-claim results remain exploratory.

## 13. Statistical analysis plan

### 13.1 Estimands and resampling unit

The estimand is normally the mean outcome over questions, not over the 12 correlated condition
rows. Confidence intervals and comparisons resample the persisted `split_group_id`, retaining all
questions and all condition/system rows within a sampled group. This accounts for repeated
conditions and shared gold documents. RAGTruth uses `source_id` clusters.

Primary intervals use 10,000 percentile cluster-bootstrap resamples with a metric-specific seed
derived from master seed 42 and the metric name. Undefined replicates are discarded and their
count reported; at least 9,900 valid replicates are required. Simple binomial/Wilson intervals may
be added as transparent secondary intervals but do not replace clustered intervals where document
dependency exists.

### 13.2 Predeclared comparisons

| Analysis | Unit and pairing | Primary test/interval | Effect size |
|---|---|---|---|
| Retriever Recall@k or sufficiency at fixed k | Same question under two retrievers | McNemar on paired binary outcomes plus paired cluster-bootstrap difference | Absolute risk difference and discordant-pair odds ratio |
| Retriever reciprocal rank | Same answerable question | Wilcoxon signed-rank (`zero_method="pratt"`) plus paired cluster-bootstrap mean/median difference | Matched-pairs rank-biserial correlation |
| Classifier error over all conditions | Per-question mean loss, with all conditions retained | Paired cluster bootstrap; Wilcoxon on per-question loss as secondary | Mean loss/F1/AUROC difference with CI |
| Primary-condition classifier correctness | Same hybrid-k5 question | McNemar where binary thresholded correctness is meaningful | Risk difference and discordant odds ratio |
| Policy selective risk/UAR/FAR/coverage/expansion/exposure | Same question-level trajectories starting at hybrid k=5 | Paired cluster-bootstrap differences; McNemar for paired binary per-query harm/correctness | Absolute percentage-point difference |
| AURC | Same questions and scores | Paired cluster-bootstrap AURC difference | Absolute AURC difference |
| Answer/grounding continuous scores | Intersection of questions answered by both systems, explicitly reported | Wilcoxon signed-rank plus paired bootstrap | Rank-biserial correlation and mean/median difference |

McNemar is not applied to the full 12-condition table as if rows were independent. Likewise,
ordinary response-level tests are not used on RAGTruth’s six responses per source.

Two-sided alpha is 0.05. Holm correction is applied separately to declared families: (a) the 12
retriever-pair-by-k primary comparisons, (b) classifier-baseline comparisons, (c) policy-baseline
comparisons, and (d) grounding-detector comparisons. Raw and adjusted p-values, family ID, number
of comparisons, confidence intervals, and effects are stored. Practical magnitude and uncertainty
take precedence over a binary significance claim.

## 14. Persisted artifacts and schemas

CSV list-valued fields use canonical JSON with sorted keys. Research-critical Parquet artifacts
use the exact pinned PyArrow version, a stable declared schema/column order, canonical-key row
sorting, Parquet version 2.6, ZSTD compression level 9, dictionary encoding disabled, statistics
enabled, data-page version 1.0, and row groups of at most 65,536 rows. Every such artifact records
both a physical file SHA-256 and a semantic SHA-256 computed from canonical JSON serialization of
the ordered logical records. Semantic equality is the primary cross-environment reproducibility
test; within the pinned environment physical equality is also required. Every artifact includes
`schema_version` and relevant config/input hashes; run-produced result artifacts also include
`run_id`, while immutable data-foundation artifacts use their pinned dataset revision and manifest
hashes.

### 14.1 Data foundation

`artifacts/data/techqa_split_assignments.csv` — one row per question:

`schema_version, dataset_revision, question_id, provenance_partition, question_sha256,
normalized_question_sha256, is_impossible, reference_status, gold_doc_ids_json,
gold_evidence_analysis_status, split_group_id, split, split_seed, split_algorithm_version,
component_manifest_sha256, split_frozen_at`

`artifacts/data/techqa_split_components.csv` — one row per connected component:

`schema_version, dataset_revision, split_group_id, component_size, member_question_ids_json,
shared_gold_filenames_json, normalized_question_hashes_json, answerable_count, impossible_count,
assigned_split, split_seed, split_algorithm_version, component_sha256`

`artifacts/data/techqa_corpus_manifest.csv` — one row per official document:

`schema_version, dataset_revision, corpus_archive_sha256, doc_id, archive_path, filename,
raw_sha256, normalized_sha256, raw_bytes, normalized_chars, encoding, cleaning_version,
removed_payload_count, status`

`artifacts/data/techqa_chunk_manifest.parquet` — one row per derived chunk:

`schema_version, chunk_id, doc_id, filename, archive_path, raw_source_sha256,
normalized_source_sha256, retrieval_source_transform_version, chunk_index, token_start, token_end,
char_start, char_end, token_length, char_length, text, text_sha256, tokenizer_name,
tokenizer_revision, max_content_tokens, overlap_tokens, step_tokens, chunking_version,
chunk_config_sha256, corpus_manifest_sha256`

`artifacts/data/ragtruth_manifest.csv` — one row per response joined to its source:

`schema_version, dataset_revision, response_id, source_id, official_split, task_type, source_name,
model, temperature, quality, response_sha256, prompt_sha256, source_info_sha256, label_count,
has_unsupported_span, implicit_true_count, due_to_null_count, label_types_json,
primary_grounding_eligible`

`artifacts/data/dataset_metadata.json` contains URLs, revisions, file hashes, observed schemas,
counts, retrieval timestamp, licenses, loader versions, component/split freeze timestamp, and
manifest hashes. The frozen component and assignment hashes are immutable inputs to every later
run manifest.

### 14.2 Retrieval and sufficiency

`artifacts/results/retrieval_query_level.csv` — one row per query × strategy × k:

`schema_version, run_id, split, split_group_id, question_id, retrieval_strategy, k, context_id,
ordered_chunk_ids_json, ordered_doc_ids_json, bm25_scores_json, dense_scores_json, fusion_scores_json,
gold_doc_id, first_gold_rank, doc_recall_at_k, reciprocal_rank_at_10, context_words, context_chars,
retrieval_config_sha256`

**Phase 3 implemented-schema amendment (frozen before label construction).** The planned CSV
schemas below are retained as historical design notes but are superseded for Phase 3 by the
canonically sorted, dual-hashed artifacts in `docs/PHASE_03_ARTIFACT_SCHEMAS.md`:

- `artifacts/data/techqa_evidence_alignments.parquet` stores every accepted candidate span plus an
  explicit row for each unresolved answerable question. Literal matches retain raw source offsets;
  every accepted span uses the canonical normalized-source character coordinates shared with
  Phase 2 chunks.
- `artifacts/data/context_sufficiency_labels.parquet` stores exactly one row per frozen condition,
  with distinct `gold_span_coverage`, `benchmark_impossible`, and unresolved/NA provenance.
- the manual pack is separated into a seeded sample manifest, blinded Parquet/CSV material, and a
  separate automatic answer key. Human agreement outputs are created only from genuine completed
  annotation files.
- `artifacts/data/phase03_column_governance.json` classifies every evidence/label column and limits
  future Phase 4 feature selection from the Phase 3 label table to `retrieval_strategy` and `k`.

`artifacts/data/techqa_evidence_alignments.csv` — superseded planned schema:

`schema_version, alignment_id, question_id, reference_status, evidence_set_id, gold_doc_id,
start_word, end_word, evidence_text_sha256, alignment_method, alignment_provenance, annotator_id,
rationale, alignment_status, guideline_version, created_at`

`artifacts/data/context_sufficiency_examples.csv` — one row per condition:

`schema_version, example_id, question_id, split, split_group_id, retrieval_strategy, k, context_id,
global_is_impossible, reference_status, reference_answer_sha256, gold_doc_id,
gold_evidence_spans_json, gold_evidence_chunk_ids_json, evidence_analysis_eligible,
evidence_exclusion_reason, evidence_span_covered, alternate_exact_evidence, nli_screen_score,
auto_label, final_y_suff, label_status, label_provenance, label_rule_version, manual_override,
manual_alignment_id`

`artifacts/results/context_sufficiency_label_validation.csv` — one row per annotation:

`schema_version, sample_id, example_id, sampling_stage, sampling_stratum, sample_seed, blind_order,
annotator_id, annotation_round, annotation_guideline_version, annotation_guideline_sha256,
manual_y_suff, confidence, reason_codes_json, notes, adjudicated_y_suff,
used_for_rule_development, annotation_timestamp`

`artifacts/results/annotation_agreement.csv` — one row per annotation task and sample:

`schema_version, annotation_task, sample_id, guideline_version, n_double_annotated,
n_exact_agreements, raw_percentage_agreement, cohens_kappa, label_distribution_json,
second_annotator_available, adjudication_complete, status, limitation_note`

`artifacts/data/sufficiency_features.parquet` — one row per labelled condition containing
`example_id`, split/group IDs, the frozen feature columns, feature-schema hash, and no forbidden
gold fields.

### 14.3 Models, calibration, and policy

`artifacts/results/classifier_predictions.csv` — one row per example × model:

`schema_version, run_id, example_id, question_id, split, split_group_id, model_id, model_variant,
fold_id, y_suff, raw_probability, calibrated_probability, operating_threshold, predicted_label,
correct, feature_schema_sha256, model_artifact_sha256`

`artifacts/results/calibration_results.csv` — one row per model/probability/split summary:

`schema_version, run_id, model_id, calibration_method, calibration_fit_split, evaluation_split,
probability_kind, n_questions, n_examples, brier, ece, n_bins, binning_method, calibrator_sha256`

`artifacts/results/calibration_bins.csv` adds:
`model_id, evaluation_split, probability_kind, bin_index, lower, upper, n, mean_probability,
empirical_sufficiency, absolute_gap`.

`artifacts/results/policy_threshold_search.csv` — every validation pair:

`schema_version, run_id, policy_id, t_low, t_high, risk_constraint,
selective_risk_numerator, selective_risk_denominator, selective_risk,
unsafe_answer_numerator, unsafe_answer_denominator, unsafe_answer_rate,
coverage_numerator, coverage_denominator, answer_coverage, false_abstention_numerator,
false_abstention_denominator, false_abstention_rate, expansion_numerator, expansion_denominator,
retrieval_expansion_rate, expansion_answer_yield, frontier_member, feasible, selection_rank,
selected_operating_point`

`artifacts/results/risk_coverage_curve.csv` — one row per fixed-condition coverage prefix:

`schema_version, run_id, model_id, split, retrieval_strategy, k, prefix_rank, included_question_id,
included_probability, included_y_suff, risk_numerator, risk_denominator, selective_risk,
coverage_numerator, coverage_denominator, coverage, aurc`

`artifacts/results/policy_outcomes.csv` — one row per primary-policy question × policy/model:

`schema_version, run_id, policy_id, risk_constraint, question_id, split_group_id, base_strategy,
initial_k, expanded_k, initial_context_id, expanded_context_id, final_context_id,
raw_probability_k5, calibrated_probability_k5, raw_probability_k10,
calibrated_probability_k10, t_low, t_high, initial_action, expansion_triggered,
post_expansion_action, final_action, final_k, y_suff_k5, y_suff_k10, y_suff_final, unsafe_answer,
false_abstention, answered, useful_answer, answer_id`

### 14.4 Generation, grounding, statistics, and final join

`artifacts/results/grounding_evaluation.parquet` — one row per sentence/claim unit:

`schema_version, run_id, dataset, response_id, source_id_or_question_id, official_or_techqa_split,
policy_id, initial_action, expansion_triggered, final_action, answer_id, claim_id, claim_text,
claim_start, claim_end, final_context_id, detector_id,
entailment_probability, contradiction_probability, embedding_proxy_score, detector_threshold,
predicted_supported, gold_supported, ragtruth_label_types_json, implicit_true, due_to_null,
adjudicated_manual_label, evaluation_eligible`

`artifacts/results/grounding_manual_annotations.csv` — one row per annotator and sampled claim:

`schema_version, sample_id, claim_id, sampling_stratum, sample_seed, blind_order, annotator_id,
annotation_round, grounding_guideline_version, grounding_guideline_sha256, manual_grounding_label,
confidence, reason_codes_json, notes, adjudicated_grounding_label, annotation_timestamp`

`artifacts/results/statistical_tests.csv`:

`schema_version, run_id, family_id, comparison_id, metric, system_a, system_b, experimental_unit,
test_name, alternative, n_pairs_or_clusters, statistic, p_raw, p_holm, effect_name, effect_value,
ci_low, ci_high, status`

`artifacts/results/bootstrap_intervals.csv`:

`schema_version, run_id, metric, system_or_contrast, point_estimate, confidence_level, ci_low,
ci_high, method, resampling_unit, n_resamples, valid_resamples, seed`

`artifacts/results/query_level_final_results.csv` — one row per final example × system:

`schema_version, run_id, question_id, split_group_id, retrieval_strategy, initial_k, initial_context_id,
expanded_k, expanded_context_id, final_k, final_context_id, prompt_context_id, y_suff_k5,
y_suff_k10, y_suff_final, model_id, raw_probability_k5, calibrated_probability_k5,
raw_probability_k10, calibrated_probability_k10, policy_id, risk_constraint, initial_action,
expansion_triggered, final_action, answer_id, generated_answer, rouge_l, bertscore_f1,
unsupported_claim_rate, claim_count, unsupported_answer_exposure, content_metric_eligible`

Non-answer `generated_answer`, answer-quality, and claim-quality fields are null. Aggregate tables
must be generated from this file or a named upstream artifact, never entered manually.

`artifacts/results/run_manifest.json` contains at least:

- schema/run ID, start/end time, exact commands, Git commit and dirty state;
- OS, Python, package, CPU/GPU/CUDA versions;
- seed registry and determinism settings;
- dataset URLs/revisions/file hashes and all data-manifest hashes;
- frozen TechQA component/split hashes and freeze timestamp;
- model/tokenizer IDs, revisions, local file hashes, dtype/device;
- full retrieval/chunk/feature/model/calibration/policy/generation/statistics configurations;
- prompt text or canonical prompt hash and annotation/grounding guideline versions and hashes;
- test-unseal timestamp and decision-log revision;
- every produced artifact path, row count, schema version, and SHA-256; and
- deviations, failed/partial stages, and resumed-run provenance.

## 15. Test-set discipline and stopping rules

- Training fits feature transforms and models.
- Validation supplies cross-fitted calibration probabilities, operating thresholds, label-rule
  validation, and permitted design selection.
- Test is untouched by feature engineering, model/hyperparameter choice, calibration fitting,
  threshold selection, label-rule changes, metric definition, and statistical-test choice.
- Test evaluation runs only after configs, prompts, code revision, artifact schemas, and the
  decision log are frozen.
- A failed context-label validation blocks modelling. A failed leakage/schema check blocks all
  later phases. Compute success alone does not waive an acceptance criterion.
- Any unplanned test rerun is recorded with reason and does not silently replace the first valid
  final run.

This contract deliberately changes the proposal’s row-stratified split, the prototype’s reduced
corpus, global target, fixed hybrid/k=5 evaluation, isotonic-only calibration, arbitrary threshold
offset, and response-random RAGTruth split. Those changes are required by the observed dataset
structure and the marker’s reproducibility and methodological-rigor requirements.
