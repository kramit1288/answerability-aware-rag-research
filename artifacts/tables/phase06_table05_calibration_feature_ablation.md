# Table 5 — Calibration and feature ablation

| section | item | auroc | auprc | f1 | brier | ece | selected | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| calibration | B3_random_forest:uncalibrated | 0.718833 | 0.674558 | 0.507692 | 0.2138151724561436 | 0.06336962935842 | yes | frozen validation comparison |
| calibration | B4_calibrated_random_forest:sigmoid | 0.718833 | 0.674558 | 0.502577 | 0.2173071070299721 | 0.0790289808945361 | no | frozen validation comparison |
| calibration | B4_calibrated_random_forest:isotonic | 0.718774 | 0.662285 | 0.476060 | 0.2177861910489285 | 0.0714143198562065 | no | frozen validation comparison |
| feature_ablation | A | 0.667678 | 0.578596 | 0.550661 |  |  | no | ["retrieval_score"] |
| feature_ablation | B | 0.710334 | 0.673557 | 0.538368 |  |  | no | ["query_context_lexical","query_context_semantic"] |
| feature_ablation | C | 0.708602 | 0.668344 | 0.561151 |  |  | no | ["retrieval_score","query_context_lexical","query_context_semantic"] |
| feature_ablation | D | 0.718833 | 0.674558 | 0.507692 |  |  | yes | ["query","retrieval_score","query_context_lexical","query_context_semantic","context_composition","cross_retriever_agreement","retrieval_condition_metadata"] |
