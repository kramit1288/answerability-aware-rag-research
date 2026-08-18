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
  --annotations path/to/annotator_1.csv `
  --annotations path/to/annotator_2.csv
```

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
