# Phase 6 Results Summary

This is a post-analysis evidence artifact generated from frozen Phase 1-5 inputs under the pre-analysis Phase 6 configuration. TechQA TEST remained sealed and Phase 7 was not started.

## RQ1 — Retrieval and retrieved-context sufficiency

Frozen Phase 2 validation retrieval results are reported in Table 2 and Figure 1; Phase 3 target validation is reported in Table 3. Document retrieval and retrieved-context sufficiency remain distinct constructs. No new aggregate-only interval or post-hoc retriever test was fabricated in Phase 6.

## RQ2 — Prediction and calibration

The selected uncalibrated Random Forest had AUROC 0.718833 (95% frozen question-bootstrap CI [0.645963, 0.787351]), AUPRC 0.674558 [0.539924, 0.785402], F1 0.507692 [0.394183, 0.608769], Brier score 0.213815 [0.189549, 0.238542], and ECE 0.063370. Tables 4-5 and Figures 2A-4 consolidate the classifier, calibration, ablation, reliability, and importance evidence. No post-hoc classifier-family p-value was introduced.

## RQ3 — Policy safety and coverage

The frozen fixed-condition operating points were: 5% constraint, coverage 0.027154 and observed risk 0.034483; 10%, coverage 0.043071 and risk 0.086957; 20%, coverage 0.112360 and risk 0.200000. AURC was 0.3784805395. Table 6 and Figure 5 show the full trade-off.

For G0-G3, question-bootstrap policy intervals are in Table 10 and the policy CI artifact. G2 answered only 2/89 questions; its quality is descriptive only and zero observed unsupported responses does not establish zero true risk. Under the frozen validation protocol, selective answering reduced the observed rate of unsupported answers at substantial cost to coverage.

## RQ4 — Generation grounding and evaluator reliability

### Paired k10 minus k5 outcomes

- `rouge_l_f1`: mean difference 0.007130 (95% CI [-0.009736, 0.024316]); median difference 0.000000; Wilcoxon W=1890.000000, raw p=0.744723, Holm p=1.000000; rank-biserial=0.040366. This did not provide evidence of a statistically detectable difference after Holm correction.
- `bertscore_f1`: mean difference 0.000404 (95% CI [-0.003127, 0.004015]); median difference 0.000000; Wilcoxon W=1961.000000, raw p=0.938002, Holm p=1.000000; rank-biserial=-0.009596. This did not provide evidence of a statistically detectable difference after Holm correction.
- `unsupported_claim_rate`: mean difference -0.024179 (95% CI [-0.098723, 0.051114]); median difference 0.000000; Wilcoxon W=1274.000000, raw p=0.266386, Holm p=1.000000; rank-biserial=-0.167048. This did not provide evidence of a statistically detectable difference after Holm correction.
- `mean_claim_support_score`: mean difference -0.034269 (95% CI [-0.087416, 0.017998]); median difference -0.006963; Wilcoxon W=1714.000000, raw p=0.250229, Holm p=1.000000; rank-biserial=-0.140852. This did not provide evidence of a statistically detectable difference after Holm correction.
- `output_token_count`: mean difference -1.910112 (95% CI [-7.168539, 3.539326]); median difference 0.000000; Wilcoxon W=1512.000000, raw p=0.145522, Holm p=0.727610; rank-biserial=-0.189059. This did not provide evidence of a statistically detectable difference after Holm correction.

### Paired binary outcomes

- `fully_supported_response`: k5 rate 0.460674, k10 rate 0.471910, paired risk difference 0.011236 (95% CI [-0.089888, 0.112360]); exact McNemar statistic=9.000000, raw p=1.000000, Holm p=1.000000. This did not provide evidence of a statistically detectable difference after Holm correction.
- `response_contains_unsupported_claim`: k5 rate 0.539326, k10 rate 0.528090, paired risk difference -0.011236 (95% CI [-0.112360, 0.089888]); exact McNemar statistic=9.000000, raw p=1.000000, Holm p=1.000000. This did not provide evidence of a statistically detectable difference after Holm correction.

### Context-sufficiency associations

- k=5 `rouge_l_f1`: sufficient-minus-insufficient mean difference 0.050590 (95% CI [0.012580, 0.088745]); Cliff's delta 0.343972 [0.112486, 0.561032]; raw p=0.005333, Holm p=0.037332.
- k=5 `bertscore_f1`: sufficient-minus-insufficient mean difference 0.015478 (95% CI [0.005621, 0.025226]); Cliff's delta 0.366768 [0.138804, 0.586283]; raw p=0.002968, Holm p=0.023744.
- k=5 `unsupported_claim_rate`: sufficient-minus-insufficient mean difference -0.198227 (95% CI [-0.336730, -0.060971]); Cliff's delta -0.279129 [-0.497972, -0.056079]; raw p=0.017041, Holm p=0.085207.
- k=5 fully-supported response: sufficient rate 0.531915, insufficient rate 0.380952, risk difference 0.150963 [-0.057071, 0.353063], risk ratio 1.396277, odds ratio 1.846591, raw p=0.201972, Holm p=0.605915.
- k=10 `rouge_l_f1`: sufficient-minus-insufficient mean difference 0.057092 (95% CI [0.015942, 0.100538]); Cliff's delta 0.239518 [0.002671, 0.470791]; raw p=0.056648, Holm p=0.226593.
- k=10 `bertscore_f1`: sufficient-minus-insufficient mean difference 0.016863 (95% CI [0.006361, 0.027133]); Cliff's delta 0.338574 [0.103062, 0.556604]; raw p=0.007019, Holm p=0.042114.
- k=10 `unsupported_claim_rate`: sufficient-minus-insufficient mean difference -0.097554 (95% CI [-0.248375, 0.049540]); Cliff's delta -0.118449 [-0.355811, 0.117858]; raw p=0.318722, Holm p=0.637443.
- k=10 fully-supported response: sufficient rate 0.490566, insufficient rate 0.444444, risk difference 0.046122 [-0.169557, 0.256566], risk ratio 1.103774, odds ratio 1.203704, raw p=0.828935, Holm p=0.828935.

These sufficient-versus-insufficient comparisons are associational: the frozen Phase 3 target and generation outcomes are measurements on the same retrieval-conditioned context states. They do not establish causation.

The binary unsupported-claim rate and continuous mean support can move in apparently contradictory directions because threshold crossings and average score shifts are different estimands. The result is retained rather than reconciled through post-hoc retuning.

### Grounding evaluator

On good-quality RAGTruth TEST, claim-level precision/recall/F1/AUROC/AUPRC were 0.2280/0.5753/0.3266/0.8030/0.2286; response-level values were 0.2756/0.7750/0.4066/0.7282/0.4056. Frozen source-bootstrap intervals appear in Table 7. Ranking discrimination is meaningful, but binary precision is low; the evaluator is an imperfect grounding proxy and every TechQA grounding result inherits that limitation.

## Holm-adjusted conclusions

Adjusted-significant hypotheses: `B_k5_rouge_l_f1_sufficient_minus_insufficient`, `B_k5_bertscore_f1_sufficient_minus_insufficient`, `B_k10_bertscore_f1_sufficient_minus_insufficient`.

## Limitations

- The TechQA analysis is VALIDATION-only with 89 questions; confidence intervals can remain wide.
- Phase 3 final confirmation used one human annotator with AI-assisted pre-key adjudication, not two independent annotators.
- RAGTruth validation supports ranking discrimination but exposes low precision for binary unsupported classifications.
- ROUGE-L and BERTScore measure reference similarity, not grounding.
- The G2 quality population contains only two answered questions.
- Context sufficiency, model confidence, grounding, and authoritative hallucination truth remain distinct.
- Observed validation risk is not a production safety guarantee.

Tables 1-11 and Figures 1-8 are the evidence package for later thesis writing; no full thesis chapter is drafted here.
