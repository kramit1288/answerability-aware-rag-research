# Table 6 — Risk-coverage and policy comparison

| section | system | risk_constraint | t_low | t_high_or_threshold | coverage | observed_selective_risk | retrieval_expansion_rate | false_abstention_rate | mean_retrieved_k | aurc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed-condition risk-coverage | selected Random Forest | 0.050000 |  | 0.7976647044899571 | 0.027154 | 0.034483 |  |  |  | 0.3784805394600604 |
| fixed-condition risk-coverage | selected Random Forest | 0.100000 |  | 0.7838454251283709 | 0.043071 | 0.086957 |  |  |  | 0.3784805394600604 |
| fixed-condition risk-coverage | selected Random Forest | 0.200000 |  | 0.6779617685497638 | 0.112360 | 0.200000 |  |  |  | 0.3784805394600604 |
| adaptive policy | P/G adaptive k5-to-k10 | 0.050000 | 0.78 | 0.82 | 0.022472 | 0.000000 | 0.0337078651685393 | 0.9622641509433962 | 5.168539325842697 |  |
| adaptive policy | P/G adaptive k5-to-k10 | 0.100000 | 0.78 | 0.82 | 0.022472 | 0.000000 | 0.0337078651685393 | 0.9622641509433962 | 5.168539325842697 |  |
| adaptive policy | P/G adaptive k5-to-k10 | 0.200000 | 0.56 | 0.7199999999999999 | 0.179775 | 0.187500 | 0.146067415730337 | 0.7547169811320755 | 5.730337078651686 |  |
| policy baseline | P0_always_answer_hybrid_k5 | 0.050000 |  |  | 1.000000 | 0.471910 | 0.0 | 0.0 | 5.0 |  |
| policy baseline | P2_always_retrieve_hybrid_k10_then_two_way | 0.050000 |  | 0.82 | 0.022472 | 0.000000 | 1.0 | 0.9622641509433962 | 10.0 |  |
| policy baseline | P0_always_answer_hybrid_k5 | 0.100000 |  |  | 1.000000 | 0.471910 | 0.0 | 0.0 | 5.0 |  |
| policy baseline | P2_always_retrieve_hybrid_k10_then_two_way | 0.100000 |  | 0.82 | 0.022472 | 0.000000 | 1.0 | 0.9622641509433962 | 10.0 |  |
| policy baseline | P0_always_answer_hybrid_k5 | 0.200000 |  |  | 1.000000 | 0.471910 | 0.0 | 0.0 | 5.0 |  |
| policy baseline | P2_always_retrieve_hybrid_k10_then_two_way | 0.200000 |  | 0.78 | 0.123596 | 0.181818 | 1.0 | 0.830188679245283 | 10.0 |  |
