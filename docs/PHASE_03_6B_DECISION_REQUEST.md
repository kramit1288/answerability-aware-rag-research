# Phase 3.6b Decision Request

No frozen candidate passed both development gates.

Candidate grid SHA-256: `4112de919a857992a52c077d54d7c64594778c1c9732eb37be69990c17709f78`

The grid contained 59 predeclared rules. Selection used 120 benchmark-answerable, semantic-evaluable, non-ambiguous TRAIN+VALIDATION development rows and the frozen population weights.

## Best coverage_only

Thresholds: T_cov=0.5, T_mean=None, T_min=None, contradiction<None.
Precision=1.0000, recall=0.6667, F1=0.8000.
Weighted precision=1.0000, weighted recall=0.6992, weighted F1=0.8230.
Confusion matrix: TN=42, FP=0, FN=26, TP=52.

## Best nli_only

Thresholds: T_cov=None, T_mean=0.45, T_min=0.1, contradiction<0.5.
Precision=1.0000, recall=0.4359, F1=0.6071.
Weighted precision=1.0000, weighted recall=0.5219, weighted F1=0.6859.
Confusion matrix: TN=42, FP=0, FN=44, TP=34.

## Best combined

Thresholds: T_cov=0.5, T_mean=0.45, T_min=0.1, contradiction<0.5.
Precision=1.0000, recall=0.6795, F1=0.8092.
Weighted precision=1.0000, weighted recall=0.7068, weighted F1=0.8282.
Confusion matrix: TN=42, FP=0, FN=25, TP=53.

### Selected combined rule by original development stratum

| Stratum | n | TN | FP | FN | TP | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| automatic_positive | 30 | 0 | 0 | 0 | 30 | 1.0000 | 1.0000 | 1.0000 |
| partial_overlap | 30 | 2 | 0 | 6 | 22 | 1.0000 | 0.7857 | 0.8800 |
| correct_document_insufficient | 30 | 15 | 0 | 14 | 1 | 1.0000 | 0.0667 | 0.1250 |
| wrong_document_retrieval | 30 | 25 | 0 | 5 | 0 | NA | 0.0000 | 0.0000 |

## Development-gate result

The selected combined rule retained precision 1.0000 but reached prevalence-weighted F1 0.8282, below the frozen 0.85 gate. It therefore cannot be frozen as the final target.

## Remaining error patterns

False positives: 0. No automatic-sufficient false-positive pattern was observed.
False negatives: 25; by original stratum: {"correct_document_insufficient":14,"partial_overlap":6,"wrong_document_retrieval":5}.
Coverage bands among false negatives: {"0.25_to_below_0.50":6,"below_0.25":19}.
NLI failure combinations among false negatives: {"mean_below_threshold+minimum_below_threshold":15,"mean_below_threshold+minimum_below_threshold+contradiction_veto":9,"minimum_below_threshold+contradiction_veto":1}.
Retrieval strategy and k summaries are retained only as diagnostics; neither was used as a rescue-boundary predictor.

## Original-versus-adjudicated transparency

Primary rescue metrics are identical for the original and adjudicated files because the sole changed row, DEV_Q217, belongs to the frozen benchmark-impossible stratum and is excluded from primary selection. The historical original file remains unchanged in content.

For both files, the selected rule therefore has precision 1.0000, recall 0.6795, F1 0.8092, weighted precision 1.0000, weighted recall 0.7068, weighted F1 0.8282, and confusion matrix TN=42, FP=0, FN=25, TP=53 on the PRIMARY development population.

## Recommendation

Stop Phase 3.6b at this decision point. Any further labeling method or any alteration of the frozen gates would require an explicit methodological decision; no post-hoc rule is recommended from these results.

No independent confirmation sample or Phase 4 artifact was created. TEST aggregate outcomes remained sealed.
