# Phase 6 Artifact Schemas

## Serialization

JSON is sorted-key UTF-8, rejects NaN, uses two-space indentation, and ends with LF. CSV uses
UTF-8, stable declared columns, empty fields for missing values, and LF. SVG is deterministic
text; PNG is binary at 300 DPI. Every Phase 6 artifact is physically SHA-256 hashed in the final
manifest.

## Governance

- `configs/phase06_statistics.json`: complete pre-analysis statistical configuration and RQ
  matrix.
- `artifacts/results/phase06_pre_analysis_governance_freeze.json`: canonical configuration SHA,
  physical governance-file hashes, upstream hashes, and explicit proof that Phase 6 inferential
  results did not predate the freeze.
- `artifacts/results/phase06_rq_analysis_matrix.csv`: one row per frozen RQ with proposal intent,
  operational mapping, dataset, unit, outcomes, comparison, test, effect, interval, tables, and
  figures.

## Statistical results

`phase06_paired_continuous_statistics.csv` stores one row per k10-minus-k5 outcome:
`metric, n_pairs, excluded_pairs, k5_mean, k10_mean, mean_difference, median_difference,
mean_ci_low, mean_ci_high, median_ci_low, median_ci_high, wilcoxon_statistic, p_raw, p_holm,
reject_holm, rank_biserial, family_id, direction, bootstrap_replicates, valid_replicates, seed`.

`phase06_paired_binary_statistics.csv` stores paired 2x2 counts, exact McNemar result, k5/k10
rates, k10-minus-k5 risk difference and bootstrap interval, raw/adjusted p-values, and family.
Rows missing either binary outcome are excluded and counted, never set false.

`phase06_sufficiency_association_statistics.csv` stores depth, metric, group counts, group
mean/median or rate, sufficient-minus-insufficient contrasts, bootstrap intervals, Mann-Whitney
or Fisher statistic/p-value, Cliff's delta and interval for continuous outcomes, and risk
difference/risk ratio/odds ratio for binary outcomes.

`phase06_statistical_tests.csv` is the normalized inferential summary:
`schema_version, family_id, family_size, comparison_id, metric, system_a, system_b,
experimental_unit, test_name, alternative, n_a, n_b, n_pairs, statistic, p_raw, p_holm,
reject_holm, effect_name, effect_value, ci_low, ci_high, status`.

`phase06_holm_correction.csv` stores family, comparison, rank, family size, raw p-value,
Holm-adjusted p-value, and rejection. `phase06_effect_sizes.csv` stores every numerical effect and
its interval where available. `phase06_bootstrap_intervals.csv` stores point, interval, method,
unit, requested/valid replicates, and seed for every Phase 6 interval.

`phase06_policy_confidence_intervals.csv` stores G0-G3 numerator, denominator, point estimate,
and question-bootstrap interval for coverage, grounded-answer yield, and unsupported-answer
population rate. `phase06_policy_quality_intervals.csv` stores conditional answered-response
quality intervals only when answered n is at least 10; G2 rows carry guard status and no interval.

`phase06_statistical_summary.json` records source hashes, family completion, significant adjusted
hypotheses, exact frozen descriptive reproduction, primary policy counts, limitations, and seal
status.

## Tables

Each `artifacts/tables/phase06_tableNN_*.csv` is the machine-readable form of one predeclared
thesis table. The matching `.md` is generated from that CSV, not hand-entered. Table 11 is a
rendered view of `phase06_statistical_tests.csv`.

## Figures

Each `artifacts/figures/phase06_figureNN*_data.csv` is the sole plotted data source for matching
SVG and PNG files. `phase06_figure_captions.json` maps figure IDs to title and thesis-safe caption
candidates. `phase06_figure_manifest.json` records paths, row counts, dimensions/DPI, captions,
source artifact hashes, and physical hashes of data/SVG/PNG.

## Interpretation and integrity

- `docs/PHASE_06_RESULTS_SUMMARY.md`: post-analysis evidence summary organized by frozen RQ.
- `phase06_artifact_manifest.json`: independent Phase 6 inventory excluding itself.
- `phase06_integrity_report.json`: checker assertions, upstream immutability, configuration hash,
  family/table/figure completion, TEST seal, and Phase 7 absence.
- `scripts/check_phase06.py`: independent recomputation of hashes and scientific invariants.
