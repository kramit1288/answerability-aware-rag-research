# Phase 5 Results Interpretation

## Status and boundary

This is a post-results interpretation artifact. The immutable pre-results configuration has canonical SHA-256 `b35ca88eccd8a24194a2976a12b31f82cc9ea243856c8c379001a84250972dd3`. No Phase 4 model or threshold was changed, TechQA TEST remained sealed, and Phase 6 inferential testing was not started.

## Grounding-evaluator validation

The frozen RAGTruth TRAIN support threshold is `0.16`. On the primary
good-quality RAGTruth TEST population, the persisted metrics were:

| Level | Precision | Recall | F1 | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|
| Claim | 0.2280 | 0.5753 | 0.3266 | 0.8030 | 0.2286 |
| Response | 0.2756 | 0.7750 | 0.4066 | 0.7282 | 0.4056 |

The predeclared response-level AUROC >= 0.60 interpretation gate passed.
However, binary unsupported classifications have relatively low precision.
The evaluator is therefore treated as an imperfect automatic grounding proxy,
not authoritative hallucination ground truth. Continuous support metrics and
binary unsupported metrics must be interpreted accordingly.

## TechQA k5 and k10

| Context | Mean ROUGE-L | Mean BERTScore F1 | Mean unsupported-claim rate | Fully-supported response rate |
|---|---:|---:|---:|---:|
| k=5 | 0.1920 | 0.8875 | 0.3013 | 0.4607 |
| k=10 | 0.1991 | 0.8879 | 0.2771 | 0.4719 |

## Frozen policy views

| Policy | Answers | Coverage | Mean ROUGE-L | Mean BERTScore | Mean unsupported rate | Grounded-answer yield | Unsupported-answer population rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| G0 | 89 | 1.0000 | 0.1920 | 0.8875 | 0.3013 | 0.4607 | 0.5393 |
| G1 | 89 | 1.0000 | 0.1991 | 0.8879 | 0.2771 | 0.4719 | 0.5281 |
| G2 | 2 | 0.0225 | 0.2821 | 0.9309 | 0.0000 | 0.0225 | 0.0000 |
| G3 | 16 | 0.1798 | 0.2212 | 0.8917 | 0.2760 | 0.0787 | 0.1011 |


## Paired descriptive comparison

Mean k10-minus-k5 differences were ROUGE-L 0.0071, BERTScore F1 0.0004, unsupported-claim rate -0.0242, mean claim support -0.0343, and output tokens -1.9101. These are descriptive only; Phase 6 will perform the predeclared paired inference.

## Interpretation limits

`y_suff_final` remains an operational retrieval-conditioned context-sufficiency target and was not redefined from generation. NLI support is an externally validated automatic proxy, not hallucination probability. ROUGE-L and BERTScore measure reference similarity, not grounding. Policy quality is conditional on answering and is always reported with coverage; abstentions remain NA for content quality. The primary G2 estimate is especially sparse because only two of 89 trajectories are answered.
