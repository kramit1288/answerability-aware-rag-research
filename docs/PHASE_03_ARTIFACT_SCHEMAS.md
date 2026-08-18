# Phase 3 artifact schemas

All research-critical Parquet files use the frozen Phase 2 PyArrow writer settings and canonical
row ordering. They receive physical and canonical-logical semantic SHA-256 hashes. Gold alignment
metadata is deliberately separate from future inference features.

## Evidence alignments

`artifacts/data/techqa_evidence_alignments.parquet` has one row per accepted candidate span and one
explicit row per unresolved answerable question. Offsets are exclusive character offsets into the
same canonical normalized source used by Phase 2 chunks. Literal source matches additionally retain
raw official-source offsets. It contains reference answers and is entirely label/provenance data.

## Condition labels

`artifacts/data/context_sufficiency_labels.parquet` has exactly one row per frozen Phase 2
`(question_id, retrieval_strategy, k)` condition. Its stable column order is `LABEL_FIELDS` in
`sufficiency/labeling.py`. Aligned answerable rows use `gold_span_coverage`; benchmark-impossible
rows use the distinct preliminary method `benchmark_impossible`; unresolved answerable rows use a
null `y_suff` and an explicit exclusion reason.

Coverage fractions, gold-document hits, offsets, covering chunk IDs, answer text, and provenance
are label-construction/evaluation metadata. They are forbidden classifier features.

## Manual validation pack

- `phase03_manual_sample_manifest.csv`: seeded allocation and strata, separate from the blinded UI.
- `phase03_annotation_blinded.parquet`: questions, benchmark answers where applicable, actual
  context text, and blank human-entry fields; no automatic label or coverage diagnostic.
- `phase03_annotation_template.csv`: lightweight CSV view of the same blinded material.
- `phase03_annotation_answer_key.parquet`: programmatically separate automatic labels and
  diagnostics used only after human annotation.
- `phase03_annotation_status.json`: records that genuine first/second human annotation and
  agreement are pending until supplied.

## Reports and governance

- `phase03_alignment_feasibility.json`: train/validation-only alignment gate and span summaries.
- `phase03_development_label_summary.csv`: train/validation-only automatic label counts.
- `phase03_prerequisite_report.json`: frozen Phase 1/2 hash and count checks.
- `phase03_label_integrity_report.json`: row coverage, monotonicity, benchmark provenance, NA,
  blinding, governance, and test-seal checks.
- `phase03_column_governance.json`: classifies every evidence/label column as
  `inference_available_feature`, `label_only`, `provenance_only`, or `evaluation_only`.
- `phase03_label_manifest.json`: run/config/input revisions and explicit Phase 4 boundary.
- `phase03_artifact_hashes.json`: physical and semantic artifact hashes plus rerun equality.

**A correct-document retrieval hit is not proof of context sufficiency.**
