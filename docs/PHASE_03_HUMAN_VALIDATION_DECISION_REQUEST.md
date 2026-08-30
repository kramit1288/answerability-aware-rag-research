# Phase 3 human-validation decision request

Date: 2026-08-29  
Status: **review required; Phase 4 blocked**

## Reason for this request

The frozen Phase 3 human-validation evaluator completed on the 150 genuine primary-human
annotations and returned `automatic_label_quality_status = fail`. The automatic-sufficient
precision gate passed, but the prevalence-weighted F1 gate failed. The independent
benchmark-impossible contamination rule also requires exclusion of benchmark-impossible negatives
from the PRIMARY Phase 4 training population.

No human annotation was changed, no automatic label was changed, no consensus label was created,
and no Phase 4 model, calibration, threshold, or policy work was started.

## Frozen gates and observed results

| Gate | Frozen criterion | Observed | Status |
|---|---:|---:|---|
| Automatic `sufficient` precision | >= 0.90 | 1.000000 | Pass |
| Prevalence-weighted F1 | >= 0.85 | 0.564188 | **Fail** |
| Benchmark-impossible contamination action | Exclude if strictly > 0.10 | 0.266667 (8/30) | **Exclusion required** |
| Human-human Cohen's kappa | >= 0.80 desirable when available | NA | Second genuine annotator unavailable |

The prevalence-weighted confusion proportions were `TN=0.535635`, `FP=0.000000`,
`FN=0.281898`, and `TP=0.182468`. The corresponding prevalence-weighted precision was 1.000000,
recall was 0.392940, F1 was 0.564188, and accuracy was 0.718102. These estimates use the frozen
TRAIN+VALIDATION stratum census of 9,240 conditions and are not the unweighted audit-sample
accuracy.

## Human annotations and disagreements

The primary annotator completed all 150 rows: 86 `sufficient`, 64 `insufficient`, and zero
`ambiguous` (ambiguous rate 0.000000). The non-ambiguous sample confusion matrix was `TN=64`,
`FP=0`, `FN=56`, and `TP=30`.

All 56 recorded disagreements have the frozen category
`automatic_insufficient_human_sufficient`. There were no
`automatic_sufficient_human_insufficient` disagreements and no ambiguous decisions.

| Frozen sampling stratum | Population count | Sample confusion `(TN, FP, FN, TP)` | Disagreements | Stratum accuracy | Human-sufficient rate among automatic negatives |
|---|---:|---:|---:|---:|---:|
| `automatic_positive` | 1,686 | (0, 0, 0, 30) | 0 | 1.000000 | Not applicable |
| `partial_overlap` | 1,014 | (2, 0, 28, 0) | 28 | 0.066667 | 0.933333 |
| `correct_document_insufficient` | 787 | (15, 0, 15, 0) | 15 | 0.500000 | 0.500000 |
| `wrong_document_retrieval` | 2,693 | (25, 0, 5, 0) | 5 | 0.833333 | 0.166667 |
| `benchmark_impossible` | 3,060 | (22, 0, 8, 0) | 8 | 0.733333 | 0.266667 |

The benchmark-impossible audit contained 30 non-ambiguous examples, of which eight were judged
context-sufficient by the primary human. Its 26.67% contamination rate exceeds the frozen 10%
trigger.

## Are the failures systematic or isolated?

The failure appears systematic rather than isolated. Disagreements occur in every automatic-negative
stratum. They dominate the `partial_overlap` audit (28/30) and affect half of the
`correct_document_insufficient` audit (15/30). The same direction of error also occurs in
`wrong_document_retrieval` (5/30) and `benchmark_impossible` (8/30). Conversely, every sampled
automatic positive agrees with the human judgement. This pattern indicates a strongly conservative
automatic rule or a systematic construct/interpretation mismatch for negative conditions, not a
small number of symmetric annotation errors.

The current evidence does not identify the cause conclusively. Plausible causes requiring review
include over-broad aligned reference spans, retrieved alternative evidence that is sufficient but
not captured by literal span coverage, or a difference between the annotation guide's notion of a
defensible answer and the exact benchmark-span rule. A single annotator means human judgement
reliability cannot yet be separated from those possible label-construction causes.

## Scientifically defensible options

1. **Obtain independent human replication and adjudicate.** Have a genuine second annotator work
   independently on the frozen blinded material, prioritising the planned approximately 100-row
   overlap and ensuring coverage of the 56 disagreements. Report original raw agreement and
   Cohen's kappa before any adjudication. Preserve both original human files and persist any
   adjudication separately.
2. **Perform a traceable root-cause audit before changing methodology.** Review accepted evidence
   spans, retrieved bundles, and rationales for the disagreement cases, grouped by the frozen
   strata. Distinguish alternative valid evidence, over-inclusive gold spans, partial-but-adequate
   evidence, and human-guide interpretation. This audit must not rewrite original human labels.
3. **If the rule is amended, treat it as an explicit pre-test methodological change.** Record the
   decision and rationale, regenerate derived automatic labels reproducibly, and validate the
   amended rule on a new untouched, blinded sample. Reusing the present 150 labels as the sole gate
   for a rule designed after seeing them would give an optimistically biased validation estimate.
4. **Retain the frozen automatic rule only for a clearly bounded sensitivity analysis.** It may be
   scientifically useful as a high-precision, low-recall operational definition, but it does not
   satisfy the frozen validity gate for the primary classifier target.
5. **Do not spot-correct only the 56 sampled conditions or silently remove difficult strata.** Such
   changes would not establish valid labels for the unsampled development population and would
   compromise the frozen sampling design.

## Recommendation

Pause Phase 4. First obtain independent second-human coverage if feasible and conduct the
stratum-structured root-cause/adjudication audit. Then make an explicit reviewed decision between
retaining the current rule as a sensitivity definition and amending the primary label construction.
If the primary rule changes, require fresh blinded validation on an untouched sample before model
training.

## Consequences for the Phase 4 training population

- Phase 4 primary classifier training is currently **not approved**, because the core
  prevalence-weighted F1 gate failed.
- Benchmark-impossible negatives are ineligible for the PRIMARY Phase 4 training population under
  the frozen contamination rule. They may be retained only for a separately reported sensitivity
  analysis.
- The `partial_overlap` and `correct_document_insufficient` strata cannot presently be treated as
  reliable automatic-negative populations for primary training without reviewed methodological
  resolution and revalidation.
- `wrong_document_retrieval` also contains observed false negatives and must be included in the
  root-cause analysis rather than assumed error-free.
- The perfect sampled precision of automatic positives supports their precision in this audit but
  does not, by itself, validate the complete binary target or permit Phase 4 to begin.

## Preserved outputs

- `artifacts/results/phase03_human_validation_results.json`
- `artifacts/results/phase03_annotation_disagreements.csv`
- `artifacts/results/phase03_annotation_annotator_1.csv` (unchanged by evaluation)

Inter-annotator overlap, raw agreement, and Cohen's kappa are unavailable because
`artifacts/results/phase03_annotation_annotator_2.csv` does not exist.
