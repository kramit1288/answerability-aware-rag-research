# Phase 3 evidence alignment and sufficiency construction

## Phase 3.6b strict-preserving rescue checkpoint (2026-08-30)

Before evaluation, the annotation paths were corrected with an atomic filename swap. The
historical original bytes are now at `phase03_annotation_annotator_1.csv` (86 sufficient, 64
insufficient, DEV_Q217 sufficient; SHA-256
`04c0f33db4ca3ac9d2a58322f6399c1688ef66a48855bbb7c7546ca7d5851959`). The reviewed bytes are
now at `phase03_annotation_annotator_1_adjudicated.csv` (85 sufficient, 65 insufficient, DEV_Q217
insufficient; SHA-256
`3e7a87f2da694cbb40930c142549cd70dde86c0d89429885711549cf91b99cc4`). This was a filename and
provenance correction only. Neither CSV was rewritten and no annotation value was changed by the
swap.

The original 150-row set remains the historical frozen Phase 3.5/3.6 development annotation. It
was later reviewed against the frozen material-completeness definition, and one borderline label
(blind order 6 / DEV_Q217) was changed by the same human workflow after AI-assisted evidence
review. This is not a second human annotator and is not inter-annotator validation. Only the
separate adjudicated 85/65 file is the Phase 3.6b development input. DEV_Q217 is benchmark-
impossible, so the original and adjudicated PRIMARY rescue metrics are identical.

Phase 1, Phase 2, Phase 3, and Phase 3.6 integrity checkers passed before evaluation. Existing
scores and aggregates were reused; no NLI model ran. The selected model/revision, scoring SHA-256
`98f7279821921d825470ee64efa810777e5b331d4c978e6234e7b689b6657fdf`, and semantic-governance
SHA-256 `233c61c45c7ad75b876e81deeb290d546a13f034f1d18c82787aea678ba0f533`
remain unchanged.

The exact 59-rule grid (5 coverage-only, 9 NLI-only, 45 combined) was persisted before label
evaluation at canonical SHA-256
`4112de919a857992a52c077d54d7c64594778c1c9732eb37be69990c17709f78`. Selection used 120
benchmark-answerable, semantic-evaluable, non-ambiguous TRAIN+VALIDATION judgments and the frozen
population weights.

- Best coverage-only (`T_cov=0.50`): precision 1.0000, recall 0.6667, F1 0.8000, weighted F1
  0.8230; `TN=42, FP=0, FN=26, TP=52`.
- Best NLI-only (`T_mean=0.45`, `T_min=0.10`, contradiction `<0.50`): precision 1.0000, recall
  0.4359, F1 0.6071, weighted F1 0.6859; `TN=42, FP=0, FN=44, TP=34`.
- Best combined (the same thresholds): precision 1.0000, recall 0.6795, F1 0.8092, weighted
  precision 1.0000, weighted recall 0.7068, weighted F1 0.8282; `TN=42, FP=0, FN=25, TP=53`.

No candidate met the unchanged 0.85 weighted-F1 gate. The run therefore stopped with
`docs/PHASE_03_6B_DECISION_REQUEST.md`. The prior semantic-only confirmation is marked
`superseded_unannotated`, remained unannotated, and its answer-key values were not opened. No
`y_suff_final`, new 50-question confirmation sample, TEST aggregate, or Phase 4 artifact was
created.

Run from the repository root in the frozen environment:

```powershell
.\.venv\Scripts\python scripts\check_phase01_data.py --config configs/phase01_data.json
.\.venv\Scripts\python scripts\check_phase02_retrieval.py
.\.venv\Scripts\python scripts\align_phase03_evidence.py
.\.venv\Scripts\python scripts\construct_phase03_labels.py
.\.venv\Scripts\python scripts\check_phase03_labels.py
.\.venv\Scripts\python -m pytest -q
```

The alignment command stops before label construction when train/validation alignment is below
90%. The construction command never regenerates retrieval: it validates the frozen Parquet hashes,
uses the persisted chunk IDs and offsets, creates one condition label per Phase 2 row, writes only
train/validation summaries, and prepares the blinded human pack. Repeating construction must
reproduce semantic and physical hashes.

After genuine human annotation files exist:

```powershell
.\.venv\Scripts\python scripts\evaluate_phase03_annotations.py `
  --annotations artifacts/results/phase03_annotation_annotator_1.csv `
  --annotations artifacts/results/phase03_annotation_annotator_2.csv `
  --output artifacts/results/phase03_human_validation_results.json `
  --disagreements-output artifacts/results/phase03_annotation_disagreements.csv
```

Create each annotator file as an independent copy of
`artifacts/results/phase03_annotation_template.csv`. The first annotator completes all 150 rows. If
available, the second annotator independently completes approximately 100 overlapping rows (the
frozen default is `blind_order` 1 through 100) and may leave the other rows blank. If no second
human is available, omit the second `--annotations` argument. The exact overlap and any shortfall
from 100 are reported.

The command rejects automatic-label/coverage columns in human inputs and refuses to open the
separate answer key until the 150-row first-human file is complete. It reads only resolved
TRAIN+VALIDATION rows when deriving the real five-stratum population frequencies. It reports
stratum metrics, prevalence-weighted confusion-derived metrics, ambiguity, disagreements, raw
human-human agreement, Cohen's kappa, and all frozen research-validity gates. A failure of the
precision or weighted-F1 gate is written to the report and returns exit status 2. A
non-evaluable/pending core gate (for example, an all-ambiguous stratum) returns exit status 3. A
benchmark-impossible contamination rate strictly above 10% records the mandatory Phase 4 exclusion
action rather than silently relabelling Phase 3. No TEST aggregate is read or emitted.

For the frozen Phase 3 artifact, those exclusive resolved TRAIN+VALIDATION frequencies are:
`automatic_positive=1686`, `partial_overlap=1014`,
`correct_document_insufficient=787`, `wrong_document_retrieval=2693`, and
`benchmark_impossible=3060` (total `9240`). The evaluator recomputes and records this census from
the frozen label artifact rather than trusting the documentation.

Do not enter automatic labels in the blinded template. No classifier, calibration, threshold,
policy, generation, grounding, or final test evaluation belongs to this phase.

## Completed run

Run `phase03-e2562c50a697a675` passed the alignment gate with 515/519 (99.2293%)
train/validation answerable questions aligned: 511 literal-source, four normalized-only, two with
multiple defensible occurrences, and four unresolved. The unresolved questions are `DEV_Q014`,
`DEV_Q094`, `DEV_Q156`, and `DEV_Q291`; their condition labels remain null.

The run produced all 10,920 condition rows, 3,600 provenance-distinct benchmark-impossible
preliminary negatives, zero k-monotonicity violations, and a 150-question blinded development
sample with 30 rows in each frozen sampling stratum. The checker passed all 14 Phase 3 integrity
checks and the repository test suite passed 41 tests. A second construction reproduced the
physical and semantic hashes of the alignment, label, sample, blinded, and answer-key artifacts.

Only train/validation label summaries were generated. No aggregate test label prevalence,
strategy comparison, disagreement pattern, or human test adjudication was calculated or exposed.
## Phase 3.5 human validation outcome

The primary genuine human annotator completed all 150 frozen TRAIN+VALIDATION examples: 86
`sufficient`, 64 `insufficient`, and zero `ambiguous`. No genuine second-annotator file was
available, so raw human-human agreement and Cohen's kappa remain unavailable.

The strict gold-span-coverage rule obtained precision 1.0000, recall 0.3488, F1 0.5172, and
prevalence-weighted F1 0.5642 (`TN=64`, `FP=0`, `FN=56`, `TP=30`). It therefore passed the frozen
0.90 precision gate but failed the unchanged 0.85 weighted-F1 gate. Its role is frozen as
`strict_span_sufficiency`: a high-precision, low-recall proxy retained for verified positives,
diagnostics, and comparison rather than primary context-sufficiency ground truth.

The benchmark-impossible audit found 8/30 (26.67%) human-sufficient contexts. Because this exceeds
the predeclared strict >10% trigger, benchmark-impossible rows are excluded from the PRIMARY Phase
4 classifier-training population and may appear only in sensitivity, separate audit, and
robustness analyses.

## Phase 3.6 boundary

The original 150 annotations are development data for a frozen local NLI semantic-support
procedure. They cannot be presented as independent confirmation of that procedure. The eligible
primary label population contains only benchmark-answerable, defensibly aligned, non-NA
TRAIN+VALIDATION retrieval conditions. Existing strict labels are preserved as `y_suff_strict`;
the new offline result is stored separately as `y_suff_semantic` and is forbidden as a runtime
feature.

Before any candidate performance metric was calculated or inspected, a results-blind feasibility
count found that an initial all-single/all-pair formulation required 110,041 unique NLI forwards
for the 6,180-condition primary population and Candidate 2 throughput projected to weeks on the
frozen two-thread CPU environment. The frozen feasible formulation uses one claim-ranked
multi-chunk premise per deterministic reference claim: snippets from every retrieved chunk enter
the ranking, at most three source snippets are selected with distinct-chunk priority, and the
256-token pair uses equal per-snippet head/tail budgets. Superseded raw score checkpoints are
excluded by the amended base semantic configuration hash and were never converted into
human-linked performance metrics.

After semantic method selection and configuration hashing, Phase 3.6 must generate a new,
non-overlapping, exactly 100-row blinded TRAIN+VALIDATION confirmation pack. The unchanged
confirmation gates are automatic-sufficient precision >=0.90 and prevalence-weighted F1 >=0.85.
Phase 3.6 stops for genuine human annotation of that pack. Aggregate TEST semantic outcomes and all
Phase 4 model, calibration, threshold, and policy work remain sealed.

### Semantic-unevaluable input-budget governance

Before any semantic label, independent confirmation sample, Phase 4 model, or TEST semantic
evaluation was produced, resumed primary inference encountered two frozen-segmentation claims
that cannot fit the selected model's 256-token pair input: `DEV_Q066` has 343 claim tokens and
`TRAIN_Q526` has 434. Across the 12 retrieval conditions per question this affects 24 conditions.

The selected scoring procedure and SHA-256
`98f7279821921d825470ee64efa810777e5b331d4c978e6234e7b689b6657fdf` are unchanged. A separately
hashed label-governance configuration now applies a question-independent rule: if any claim's
token count plus the selected tokenizer's pair-special-token count is at least the frozen maximum
sequence length, all conditions for that question retain `y_suff_strict` but receive
`y_suff_semantic=NA`, `semantic_label_status=unevaluable`, and
`semantic_exclusion_reason=claim_exceeds_frozen_nli_pair_budget`. They are retained only for
transparent exclusion accounting and strict-label diagnostics and are excluded from primary
semantic-target training, calibration, threshold selection, evaluation, and confirmation
sampling. They are never described as insufficient.

No claim is truncated and no post-selection segmentation is introduced: truncation could remove
evidence-bearing material, and a new segmentation rule would change the frozen semantic method.
The explicit NA policy preserves the selected procedure and reports the small exclusion directly.
