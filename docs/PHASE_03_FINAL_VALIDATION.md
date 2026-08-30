# Phase 3 final independent human confirmation

Date: 2026-08-30

## Frozen procedure

The final target remained unchanged throughout confirmation. Every strict positive is retained;
a strict negative is rescued only when coverage is at least 0.20, or when mean claim entailment is
at least 0.35, minimum claim entailment is at least 0.05, and maximum selected-premise
contradiction is below 0.50. The canonical final-target configuration SHA-256 is
`5b9a394e5e97776844054f3282af815462148adc62b528c77245d61707406977`.

The completed human CSV passed the pre-key boundary before the separate answer key was opened. It
contained exactly the frozen 50 sample IDs and questions once each, all labels and annotator
metadata were complete and valid, and its non-annotation fields matched the frozen blinded
template. It contained no automatic label, score, prediction, answer-key, TEST,
benchmark-impossible, or semantic-unevaluable field/row. Question overlap was zero against both
the original 150-row development set and the superseded unannotated 100-question semantic-only
confirmation set. The completed CSV SHA-256 is
`1e643fd2e6a9e746f2c1f81033000c638a9008d6b6931b3f2fcb8bbd80141c4a`.

The 50 rows were initially annotated manually by one human annotator. Before any automatic
answer-key access, those annotations were reviewed with AI assistance against the already frozen
material-completeness definition. During that pre-key review, one annotation (`DEV_Q012`) changed
from sufficient to insufficient. No automatic target label, NLI score, span-coverage score,
prediction, or answer-key value had been exposed. This provenance is therefore a single-human
annotation with AI-assisted pre-key adjudication, not two independent human annotations.

## Independent confirmation result

The human labels were 34 sufficient, 16 insufficient, and zero ambiguous. Ambiguous rate was
therefore 0/50 = 0.0000; no row was removed from the binary denominators.

| Metric | Result |
|---|---:|
| Automatic-sufficient precision | 0.9118 |
| Precision 95% Wilson CI | 0.7704-0.9695 |
| Recall | 0.9118 |
| Raw F1 | 0.9118 |
| Accuracy | 0.8800 |
| Prevalence-weighted precision | 0.9297 |
| Prevalence-weighted recall | 0.8055 |
| Prevalence-weighted F1 | 0.8632 |
| Prevalence-weighted accuracy | 0.8659 |

The raw confusion matrix is `TN=13, FP=3, FN=3, TP=31`. Population-weighted metrics use the
frozen PRIMARY TRAIN+VALIDATION condition frequencies within the three confirmation strata; they
must not be interpreted as an unstratified natural-prevalence estimate from the 50 rows.

| Frozen confirmation stratum | Sample n | PRIMARY conditions | TN | FP | FN | TP | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| strict positive retained | 17 | 1,686 | 0 | 0 | 0 | 17 | 1.0000 | 1.0000 | 1.0000 |
| strict negative rescued | 17 | 1,116 | 0 | 3 | 0 | 14 | 0.8235 | 1.0000 | 0.9032 |
| strict negative not rescued | 16 | 3,354 | 13 | 0 | 3 | 0 | NA | 0.0000 | 0.0000 |

Both predeclared confirmation gates passed without reinterpretation: ordinary automatic-
sufficient precision 0.9118 is at least 0.90, and prevalence-weighted F1 0.8632 is at least 0.85.
The Wilson interval is descriptive and was not used as an additional post-hoc gate.

## Disagreements

There were six disagreements. The three false positives were rescued strict negatives whose
retrieved text was related but omitted a material command, supported-environment qualifier, or
minimum-operating-system detail (`DEV_Q025`, `DEV_Q012`, and `TRAIN_Q064`). The three false
negatives were unrescued strict negatives that the human judged sufficient: one supported an
alternative valid yes-answer using other security bulletins (`DEV_Q162`), and two contained the
core resolution despite omitted linked or secondary details (`DEV_Q016` and `TRAIN_Q141`). These
are descriptive confirmation errors only; no threshold, segmentation, model, or label was changed
after seeing them.

## Frozen final population

The PRIMARY target contains 6,156 eligible TRAIN+VALIDATION conditions from 513 questions:
1,686 strict positives, 1,116 rescued positives, 2,802 final positives, and 3,354 final negatives.
Benchmark-impossible rows remain excluded from primary model training and retained for sensitivity
analysis. Four unresolved-evidence questions remain excluded. The two semantic-unevaluable
questions (`DEV_Q066` and `TRAIN_Q526`; 24 conditions) remain NA with reason
`claim_exceeds_frozen_nli_pair_budget`.

## Limitations and boundary

Only one genuine human annotator was available; the AI-assisted pre-key adjudication was not a
second independent human annotation. Inter-annotator agreement was therefore not measured and
remains a study limitation.
The independent sample contains 50 rather than 100 questions as frozen before its outcomes. All
50 came from remaining TRAIN questions because the original-development and superseded-sample
question exclusions exhausted eligible VALIDATION questions. Benchmark-impossible cases are
excluded from primary training and retained only for sensitivity analysis, and the two
semantic-unevaluable questions remain excluded under the frozen input-budget policy.

Phase 3 is complete. Aggregate TEST outcomes remain sealed. Phase 4 was not started and requires
separate review before work begins.
