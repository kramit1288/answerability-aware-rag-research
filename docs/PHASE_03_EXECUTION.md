# Phase 3 evidence alignment and sufficiency construction

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
First and second genuine human annotations remain pending.
