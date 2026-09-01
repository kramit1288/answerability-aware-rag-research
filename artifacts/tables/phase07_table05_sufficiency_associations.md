# Context sufficiency associations

| split | comparison_id | k | metric | outcome_type | sufficient_n | insufficient_n | mean_difference | risk_difference | cliffs_delta | p_raw | p_holm | reject_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| VALIDATION | B_k5_rouge_l_f1_sufficient_minus_insufficient | 5 | rouge_l_f1 | continuous | 47 | 42 | 0.0505900203743147 | NA | 0.3439716312056737 | 0.0053332076101334 | 0.037332453270934 | True |
| VALIDATION | B_k5_bertscore_f1_sufficient_minus_insufficient | 5 | bertscore_f1 | continuous | 47 | 42 | 0.0154778253222912 | NA | 0.3667679837892604 | 0.0029679779161112 | 0.02374382332889 | True |
| VALIDATION | B_k5_unsupported_claim_rate_sufficient_minus_insufficient | 5 | unsupported_claim_rate | continuous | 47 | 42 | -0.1982274470071716 | NA | -0.279128672745694 | 0.0170413651136337 | 0.0852068255681686 | False |
| VALIDATION | B_k5_fully_supported_response_sufficient_minus_insufficient | 5 | fully_supported_response | binary | 47 | 42 | NA | 0.1509625126646403 | NA | 0.2019716909247501 | 0.6059150727742504 | False |
| VALIDATION | B_k10_rouge_l_f1_sufficient_minus_insufficient | 10 | rouge_l_f1 | continuous | 53 | 36 | 0.0570915258000818 | NA | 0.2395178197064989 | 0.0566483697184712 | 0.226593478873885 | False |
| VALIDATION | B_k10_bertscore_f1_sufficient_minus_insufficient | 10 | bertscore_f1 | continuous | 53 | 36 | 0.0168634682144008 | NA | 0.3385744234800839 | 0.0070190727568749 | 0.0421144365412495 | True |
| VALIDATION | B_k10_unsupported_claim_rate_sufficient_minus_insufficient | 10 | unsupported_claim_rate | continuous | 53 | 36 | -0.0975541579315164 | NA | -0.1184486373165618 | 0.3187215929181816 | 0.6374431858363632 | False |
| VALIDATION | B_k10_fully_supported_response_sufficient_minus_insufficient | 10 | fully_supported_response | binary | 53 | 36 | NA | 0.0461215932914046 | NA | 0.8289345678525519 | 0.8289345678525519 | False |
| TEST | B_k5_rouge_l_f1_sufficient_minus_insufficient | 5 | rouge_l_f1 | continuous | 39 | 49 | 0.1668380345625089 | NA | 0.5264259549973835 | 2.430269940290947e-05 | 0.0001701188958203 | True |
| TEST | B_k5_bertscore_f1_sufficient_minus_insufficient | 5 | bertscore_f1 | continuous | 39 | 49 | 0.0247134878521873 | NA | 0.4306645735217164 | 0.0005557901579397 | 0.0033347409476384 | True |
| TEST | B_k5_unsupported_claim_rate_sufficient_minus_insufficient | 5 | unsupported_claim_rate | continuous | 39 | 49 | -0.1208484984800773 | NA | -0.1815803244374673 | 0.124073382229541 | 0.2481467644590821 | False |
| TEST | B_k5_fully_supported_response_sufficient_minus_insufficient | 5 | fully_supported_response | binary | 39 | 49 | NA | 0.1302982731554159 | NA | 0.283387973604151 | 0.283387973604151 | False |
| TEST | B_k10_rouge_l_f1_sufficient_minus_insufficient | 10 | rouge_l_f1 | continuous | 47 | 41 | 0.152298199844779 | NA | 0.5609756097560976 | 6.26972967702327e-06 | 5.015783741618616e-05 | True |
| TEST | B_k10_bertscore_f1_sufficient_minus_insufficient | 10 | bertscore_f1 | continuous | 47 | 41 | 0.0216328647697163 | NA | 0.3876491956408925 | 0.0018079448195556 | 0.0090397240977781 | True |
| TEST | B_k10_unsupported_claim_rate_sufficient_minus_insufficient | 10 | unsupported_claim_rate | continuous | 47 | 41 | -0.2123330397331435 | NA | -0.3326414115204982 | 0.0051728546842015 | 0.020691418736806 | True |
| TEST | B_k10_fully_supported_response_sufficient_minus_insufficient | 10 | fully_supported_response | binary | 47 | 41 | NA | 0.2848988064348728 | NA | 0.0093892855707425 | 0.0281678567122275 | True |
