# Phase 7 Selective-Tail Integrity Audit

## Scope and conclusion

This post-results audit examined only persisted frozen Phase 4 and Phase 7 artifacts. It did not retrain a model, refit preprocessing, recompute features, alter probabilities or labels, select a TEST threshold, rerun generation or NLI, or modify any Phase 1–6 artifact.

Root-cause category: **C — genuine frozen-model high-confidence-tail generalization failure**.

Global classifier discrimination remained moderate on TEST (AUROC 0.7588; AUPRC 0.5788), but the frozen validation-derived high-confidence operating points did not preserve their validation selective-risk constraints. The failure reproduces directly from persisted TEST probability and target rows. No TEST operating point was selected and no post-TEST tuning occurred.

## Positive-class and probability semantics

- The classifier target in Phase 4 and Phase 7 is `y_suff_final`.
- `y_suff_final == 1` means sufficient retrieved context; `0` means insufficient retrieved context.
- The serialized pipeline and Random Forest both have `classes_ == [0, 1]`.
- Phase 7 used `pipeline.predict_proba(X)[:, 1]`, followed by the frozen uncalibrated identity transform. Therefore persisted `probability` is `p_suff = P(y_suff_final == 1)`.
- The exact ordered persisted probability-vector audit SHA-256 was `241b932073fe29445a51773f2294d0944399ab86ef714d7e000b69dbc897c554`.

## Frozen preprocessing audit

- Raw feature count and order: 39, exactly matching the serialized bundle and Phase 4 registry.
- Transformed width: 46, exactly matching `RandomForestClassifier.n_features_in_`.
- TRAIN-fitted score normalization was reused for the three frozen strategies.
- The only numeric preprocessing step was the TRAIN-fitted median `SimpleImputer(add_indicator=True)`; no scaler was present.
- Expected and observed missing-indicator features matched exactly: `retrieval_top2_score`, `retrieval_top1_top2_gap`, `query_chunk_similarity_top1_top2_gap`, `chunk_pairwise_similarity_mean`, and `chunk_pairwise_similarity_max`.
- Frozen one-hot categories and observed TEST values were both `bm25`, `dense`, and `hybrid`.
- k remained the unchanged numeric feature with TEST values 1, 3, 5, and 10.
- TEST preprocessing/model/calibrator fit calls were zero.

Frozen identities used in this check:

- Feature registry SHA-256: `850a29b2493e481f93fabdbc9374a2ee64a971cd6e6f045699f046f7424decfb`.
- Selected-model canonical SHA-256: `efb07f98b8e73a5f277bd89591badf1d64f486354de8e4caccdad7672463a019`.
- Model artifact SHA-256: `7d0f124cfea75d9b69e52ef83af066eae50afdf35b5d3f4a0d53f7785b36b9fd`.
- TEST feature artifact SHA-256: `0964f7b5d229ac395cf2a899cfa09192fe1403521bc5e6f70ba8709f38130d38`.
- TEST prediction artifact SHA-256: `8aad1cbdd099f91d7dddfa3518458ab898ceb61d82600495600af2f25a8dc258`.
- TEST target artifact SHA-256: `d6ed77f1162846e8d71b5d5dfb9bab11bb0b1c714e1241c485900db59e1be258`.

## Row alignment and operating-point reconstruction

Target, feature, and prediction artifacts each contained 1,056 rows. Each had zero duplicate `(question_id, retrieval_strategy, k)` keys. Pairwise missing and extra key counts were all zero. Target-label mismatches, condition-ID mismatches, and policy probability/label mismatches were all zero. The complete persisted risk-coverage curve reconstructed to 1,016 rows with no omitted eligible conditions and only floating serialization error (`1.11e-16`) relative to the stored curve.

Using `answer = p_suff >= threshold`:

| Validation constraint | Frozen threshold | Eligible | Answered | Sufficient | Insufficient | Coverage | Selective risk | Min answered p | Max answered p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5% | 0.7976647044899572 | 1,056 | 29 | 2 | 27 | 0.027462 | 0.931034 | 0.798700 | 0.955199 |
| 10% | 0.7838454251283709 | 1,056 | 36 | 5 | 31 | 0.034091 | 0.861111 | 0.788540 | 0.955199 |
| 20% | 0.6779617685497638 | 1,056 | 131 | 83 | 48 | 0.124053 | 0.366412 | 0.681881 | 0.955199 |

At the frozen default classifier threshold 0.5, the reconstructed confusion matrix was TN 581, FP 83, FN 195, TP 197. This reproduces precision 0.703571, recall 0.502551, and F1 0.586310 under the same positive-class semantics.

## Policy trajectory audit

G2 (`t_low=0.78`, `t_high=0.82`) answered three questions:

| Question | Answer context | p5 | p10 | Actual-context target | Final action |
|---|---:|---:|---:|---:|---|
| `DEV_Q055` | k5 | 0.937800 | 0.946372 | 0 | `ANSWER_AT_K5` |
| `DEV_Q262` | k10 after expansion | 0.801253 | 0.860083 | 0 | `ANSWER_AT_K10` |
| `TRAIN_Q458` | k5 | 0.901846 | 0.894005 | 0 | `ANSWER_AT_K5` |

All three answers were therefore unsafe under the frozen retrieved-context sufficiency target. G3 mechanically reproduced 14 answers, 10 safe answers, 4 unsafe answers, and selective risk `4/14 = 0.285714`.

For every G2 and G3 trajectory, k5 answers used `y_suff_final(hybrid, k=5)` and answers after expansion used `y_suff_final(hybrid, k=10)`. Actual-context target mismatches were zero.

## Grounding versus sufficiency

The three G2 answered responses reproduced as follows:

| Question | `y_suff_final` | Fully supported response | Unsupported-claim rate |
|---|---:|---:|---:|
| `DEV_Q055` | 0 | true | 0.000000 |
| `DEV_Q262` | 0 | false | 0.333333 |
| `TRAIN_Q458` | 0 | true | 0.000000 |

Thus Phase 3 sufficiency classified 0/3 as safe, while the frozen grounding proxy classified 2/3 generated responses as fully supported. This is not an integrity contradiction. Context sufficiency evaluates whether retrieved context supports a materially complete answer to the question; grounding evaluates whether the claims the model actually produced are supported by that context. A partial or incomplete answer can be fully grounded while originating from context labelled insufficient. Neither construct was redefined during this audit.

## Integrity interpretation

The high-confidence TEST selective-risk result is numerically and semantically correct under the frozen definitions. Moderate global rank discrimination does not guarantee that a small extreme-probability tail will retain validation calibration or risk constraints under distribution shift. The frozen tail contained many high-confidence false positives, and the result is retained unchanged.

No scientific artifact, threshold, probability, feature, target, generator, grounding result, or Phase 1–6 file was changed by this audit. This document is an interpretation/integrity artifact only.
