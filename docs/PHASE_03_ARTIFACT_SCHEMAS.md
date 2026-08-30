# Phase 3 artifact schemas

## Phase 3.6c final rescue and confirmation artifacts

- `configs/phase03_rescue_grid_expanded.json`: final-iteration governance, the exact 77-rule grid,
  historical/new-candidate flags, fixed gates, Wilson interval definition, exclusions, and all
  upstream hashes.
- `phase03_rescue_expanded_candidate_grid_freeze.json`: physical/canonical grid hashes and proof
  of pre-outcome freezing.
- `phase03_rescue_expanded_candidate_results.csv`: 77 rows with thresholds, historical/new status,
  confusion matrices, ordinary/weighted metrics, rescue-mechanism counts, and per-stratum JSON.
- `phase03_rescue_expanded_development_report.json`: selected rule, family leaders, Wilson precision
  interval, per-stratum metrics, error patterns, original/adjudicated transparency, gate result,
  final population counts, and confirmation status.
- `configs/phase03_final_target.json`: strict-preserving final rule and SHA-256, exact thresholds,
  signal definitions, exclusions, confirmation gates, and upstream Phase 1/2/3/3.6/3.6b hashes.
- `artifacts/data/context_sufficiency_final_labels.parquet`: all 6,180 initially eligible
  TRAIN+VALIDATION conditions with preserved `y_suff_strict`, preserved/NA `y_suff_semantic`,
  `y_suff_final`, eligibility/exclusion status, rescue diagnostics, and configuration hashes.
- `artifacts/data/phase03_final_primary_target.parquet`: only the 6,156 eligible primary target
  rows. Semantic-unevaluable, benchmark-impossible, and unresolved-evidence rows cannot enter it.
- `phase03_final_confirmation_sample_manifest.csv`: 50 frozen sample identities and allocation
  strata, separate from the blinded annotator view.
- `phase03_final_confirmation_blinded.parquet` and
  `phase03_final_confirmation_template.csv`: identical 50-row blinded material with blank human
  fields and no strict/semantic/final label, coverage/NLI score, prediction, or answer-key field.
- `phase03_final_confirmation_answer_key.parquet`: separate sealed automatic labels and diagnostic
  values; its values are not read before valid human annotation is complete.
- `phase03_final_confirmation_status.json`: pending/unannotated state, unchanged confirmation
  gates, balance counts, overlap/exclusion checks, and study limitations.
- `phase03_rescue_expanded_artifact_manifest.json`: hashes and run state for the completed
  DEVELOPMENT pass checkpoint. It records that TEST remained sealed and Phase 4 unstarted.

## Phase 3.6b rescue decision artifacts

- `configs/phase03_rescue_grid.json` is the pre-evaluation machine-readable declaration of all
  59 strict-preserving rescue candidates, input signals, forbidden boundary inputs, frozen
  population weights, selection order, gates, confirmation exclusions, and upstream hashes.
- `artifacts/results/phase03_rescue_candidate_grid_freeze.json` records its physical and canonical
  SHA-256 values and confirms that the grid preceded development-label evaluation.
- `artifacts/results/phase03_rescue_candidate_results.csv` contains one row per frozen candidate:
  family/thresholds, ordinary and weighted metrics, confusion matrix, rescue-mechanism counts,
  and JSON-encoded per-stratum metrics.
- `artifacts/results/phase03_rescue_development_report.json` records the best member of each
  family, the selected eligible rule, original/adjudicated transparency comparison, remaining
  error patterns, frozen-gate outcome, and TEST/Phase 4 seals.
- `artifacts/results/phase03_semantic_confirmation_supersession.json` marks the existing semantic-
  only confirmation pack `superseded_unannotated` without modifying or reading its answer-key
  values.
- `docs/PHASE_03_6B_DECISION_REQUEST.md` is required when the development gate fails. In that
  state, `context_sufficiency_final_labels.parquet`, the primary final-target export, and all
  `phase03_final_confirmation_*` artifacts must not exist.

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

The sample has exactly 30 rows in each of five mutually exclusive strata and is not self-weighting.
Human annotators edit independent CSV copies of `phase03_annotation_template.csv`; automatic-label
and coverage columns are forbidden in those inputs. The complete first-human file has 150 rows. A
second-human file, when available, should overlap on approximately 100 rows; any smaller overlap is
reported exactly.

## Human validation outputs (created only after genuine annotation)

- `phase03_human_validation_results.json` records the primary annotator, actual resolved
  TRAIN+VALIDATION stratum census, metrics and ambiguous rate by stratum, descriptive-only
  unweighted sample metrics, prevalence-weighted metrics derived from weighted confusion rates,
  benchmark-impossible audit rate, human-human agreement, and the frozen gate report.
- `phase03_annotation_disagreements.csv` records automatic-vs-primary-human disagreements,
  ambiguous decisions, and original human-human disagreements for review. Its canonical key is
  `(disagreement_scope, sample_id, annotator_id)`.

The development-population weighting strata are `automatic_positive`, `partial_overlap`,
`correct_document_insufficient`, `wrong_document_retrieval`, and `benchmark_impossible`. Resolved
TRAIN+VALIDATION condition counts are read mechanically from
`context_sufficiency_labels.parquet`; TEST rows and unresolved/null rows are excluded. The physical
answer key remains separate from all blinded annotator inputs and is joined only by the evaluation
script after the first-human file is complete.

## Phase 3.6 semantic refinement artifacts

The original 150-row human-validation sample is development-only for Phase 3.6. The semantic
procedure does not replace or mutate those annotations, and the historical exact-span output is
retained as `y_suff_strict`.

- `artifacts/results/phase03_semantic_development_condition_scores.parquet` records one
  condition-level support summary per compatible predeclared candidate and original development
  condition. Benchmark-impossible rows have the explicit status
  `not_evaluable_no_reference_answer`; no reference answer is invented for them.
- `artifacts/results/phase03_semantic_development_claim_scores.parquet` persists deterministic
  reference-answer claim units and their selected NLI support/contradiction scores.
- `artifacts/results/phase03_semantic_threshold_search.csv` records every point in the finite,
  predeclared threshold grid for every compatible candidate.
- `artifacts/results/phase03_semantic_selected_config.json` freezes the selected model revision,
  claim segmentation, context-unit aggregation, thresholds, tie-breaking, population definition,
  and semantic configuration SHA-256 before confirmation sampling.
- `artifacts/data/context_sufficiency_semantic_labels.parquet` has one row per initially eligible
  benchmark-answerable TRAIN or VALIDATION retrieval condition. It retains `y_suff_strict` for
  every row. Semantically evaluable rows have provisional binary `y_suff_semantic`; a question
  containing any claim beyond the frozen pair-input budget instead has `y_suff_semantic=NA`,
  `semantic_label_status=unevaluable`, exclusion reason
  `claim_exceeds_frozen_nli_pair_budget`, and persisted claim-length/model-budget metadata on all
  of its conditions. These NA rows are not insufficient examples and are not Phase 4 primary
  semantic-target rows.
- `artifacts/data/phase03_semantic_claim_scores.parquet` contains the persisted claim units and
  NLI support evidence used to construct the provisional primary-population labels.
- `artifacts/data/phase03_semantic_unevaluable_questions.parquet` contains one row per excluded
  question, its split, condition count, unchanged claim-token lengths, exceeding claim indices,
  model pair-budget metadata, explicit reason, scoring hash, and label-governance hash.
- `configs/phase03_semantic_label_governance.json` freezes the mechanical input-budget exclusion
  policy separately from the already selected semantic scoring configuration. Its canonical hash
  is the final Phase 3.6 label-governance/configuration hash; it references, but does not redefine,
  scoring SHA-256 `98f7279821921d825470ee64efa810777e5b331d4c978e6234e7b689b6657fdf`.
- `artifacts/data/phase03_semantic_column_governance.json` marks reference-, answer-, strict-label-,
  semantic-label-, score-, and provenance-derived fields as `label_only` or `evaluation_only` and
  therefore unavailable to Phase 4 feature export. Its coverage includes the revised condition
  labels, claim-score table, blinded confirmation view, and separate confirmation answer key.
- `artifacts/results/phase03_semantic_refinement_results.json` contains TRAIN+VALIDATION-only
  candidate comparison, selected development performance, primary-population counts, and strict
  versus semantic diagnostics. It contains no aggregate TEST semantic result.
- `artifacts/results/phase03_semantic_artifact_hashes.json` records physical hashes and semantic
  hashes where row-oriented canonicalization is defined, plus frozen input and package versions.

## Independent semantic confirmation pack

The confirmation sample is created only after `phase03_semantic_selected_config.json` is frozen.
It contains exactly 100 benchmark-answerable TRAIN+VALIDATION conditions, uses at most one
condition per question, and excludes every question represented in the original 150-row
development sample. It also excludes all semantic-unevaluable questions.

- `phase03_semantic_confirmation_sample_manifest.csv`: seeded sample identities, allocation
  strata, and frozen semantic configuration hash; separate from the annotator view.
- `phase03_semantic_confirmation_blinded.parquet`: canonical blinded material with blank human
  fields and no strict label, semantic label, NLI score, automatic prediction, or answer-key field.
- `phase03_semantic_confirmation_template.csv`: human-editable CSV view of the same blinded rows.
- `phase03_semantic_confirmation_answer_key.parquet`: separate automatic labels and semantic
  diagnostics. It must remain unopened for evaluation until the human template is complete and
  valid.
- `phase03_semantic_confirmation_status.json`: sample size, stratum counts, seed, frozen gates,
  blinding state, and the explicit pending-human-confirmation status.

The confirmation gates are frozen at automatic-sufficient precision at least 0.90 and
prevalence-weighted F1 at least 0.85. The template is not an evaluated result and must not be
filled by an LLM or by the semantic procedure.

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
