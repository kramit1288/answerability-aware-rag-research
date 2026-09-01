"""Independent Phase 6 integrity, grouping, correction, figure, and seal checker."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from answerability_rag.hashing import sha256_file  # noqa: E402
from answerability_rag.statistics.core import holm_adjust  # noqa: E402
from answerability_rag.statistics.pipeline import (  # noqa: E402
    EXPECTED_CONFIG_SHA256,
    EXPECTED_MEAN_DIFFERENCES,
    Phase06Config,
    verify_upstream,
)


def require(condition: bool, message: str, checks: list[str]) -> None:
    if not condition:
        raise AssertionError(message)
    checks.append(message)


def main() -> None:
    checks: list[str] = []
    config = Phase06Config.load(ROOT / "configs/phase06_statistics.json", ROOT)
    require(config.canonical_sha256 == EXPECTED_CONFIG_SHA256,
            "Phase 6 canonical configuration reproduces", checks)
    upstream = verify_upstream(ROOT, config)
    require(upstream["techqa_test_sealed"],
            "all Phase 3-5 scientific hashes and invariants reproduce", checks)
    freeze = json.loads((ROOT / "artifacts/results/phase06_pre_analysis_governance_freeze.json").read_text(encoding="utf-8"))
    require(not freeze["inferential_results_observed_before_freeze"]
            and freeze["research_question_mapping_frozen_in_config"],
            "Phase 6 governance and RQ mapping predate inference", checks)

    paired_source = pq.read_table(ROOT / "artifacts/results/phase05_paired_k5_k10.parquet").to_pandas()
    paired = pd.read_csv(ROOT / "artifacts/results/phase06_paired_continuous_statistics.csv")
    require(len(paired_source) == 89 and paired_source.question_id.nunique() == 89
            and len(paired) == 5 and (paired.n_pairs == 89).all(),
            "all five paired outcomes use exactly 89 aligned questions", checks)
    observed = dict(zip(paired.metric, paired.mean_difference, strict=True))
    require(all(np.isclose(observed[key], value, atol=1e-14, rtol=0)
                for key, value in EXPECTED_MEAN_DIFFERENCES.items()),
            "all frozen k10-minus-k5 descriptive means reproduce", checks)
    require((paired.direction == "k10_minus_k5").all(),
            "paired difference direction is consistently k10 minus k5", checks)

    binary = pd.read_csv(ROOT / "artifacts/results/phase06_paired_binary_statistics.csv")
    transition_columns = ["first_false_second_false", "first_false_second_true",
                          "first_true_second_false", "first_true_second_true"]
    require(len(binary) == 2 and binary[transition_columns].sum(axis=1).eq(binary.n_pairs).all()
            and (binary.excluded_pairs == 0).all(),
            "paired binary transition tables are complete without NA coercion", checks)

    association = pd.read_csv(ROOT / "artifacts/results/phase06_sufficiency_association_statistics.csv")
    require(len(association) == 8
            and set(map(tuple, association[["k", "sufficient_n", "insufficient_n"]].drop_duplicates().to_numpy()))
            == {(5, 47, 42), (10, 53, 36)},
            "k5 and k10 sufficiency-association populations reproduce", checks)
    require((association.status == "associational").all(),
            "sufficiency-generation comparisons remain explicitly associational", checks)

    tests = pd.read_csv(ROOT / "artifacts/results/phase06_statistical_tests.csv")
    expected_sizes = {"A_paired_k5_k10_continuous": 5,
                      "B_sufficiency_association": 8,
                      "C_paired_k5_k10_binary": 2}
    require(tests.groupby("family_id").size().to_dict() == expected_sizes,
            "all three predeclared hypothesis families are complete", checks)
    for family, group in tests.groupby("family_id"):
        adjusted, rejected = holm_adjust(group.p_raw.to_numpy(), config.values["alpha"])
        require(np.allclose(adjusted, group.p_holm.to_numpy())
                and np.array_equal(rejected, group.reject_holm.astype(bool).to_numpy()),
                f"Holm correction reproduces for {family}", checks)
    require(tests.effect_value.notna().all() and tests.ci_low.notna().all()
            and tests.ci_high.notna().all(),
            "every hypothesis test has a numerical effect size and matching interval", checks)

    bootstrap = pd.read_csv(ROOT / "artifacts/results/phase06_bootstrap_intervals.csv")
    require((bootstrap.resampling_unit == "question_id").all()
            and (bootstrap.requested_replicates == 5000).all()
            and (bootstrap.valid_replicates > 0).all(),
            "all new TechQA intervals use 5,000 question-level bootstrap replicates", checks)
    ragtruth_ci = pd.read_csv(ROOT / "artifacts/results/phase05_ragtruth_bootstrap_intervals.csv")
    require((ragtruth_ci.resampling_unit == "source_id").all(),
            "all consolidated RAGTruth intervals retain source_id grouping", checks)

    policy_ci = pd.read_csv(ROOT / "artifacts/results/phase06_policy_confidence_intervals.csv")
    require(len(policy_ci) == 12 and set(policy_ci.policy_id) == {"G0", "G1", "G2", "G3"}
            and set(policy_ci.metric) == {"answer_coverage", "grounded_answer_yield",
                                         "unsupported_answer_population_rate"},
            "G0-G3 policy proportion uncertainty is complete", checks)
    quality_ci = pd.read_csv(ROOT / "artifacts/results/phase06_policy_quality_intervals.csv")
    require((quality_ci.loc[quality_ci.policy_id == "G2", "answered_eligible_n"] == 2).all()
            and quality_ci.loc[quality_ci.policy_id == "G2", "status"].eq(
                "tiny_sample_guard_no_quality_inference").all()
            and quality_ci.loc[quality_ci.policy_id == "G2", ["mean_ci_low", "mean_ci_high"]].isna().all().all(),
            "G2 tiny-sample guard blocks quality inference", checks)

    table_manifest = json.loads((ROOT / "artifacts/tables/phase06_table_manifest.json").read_text(encoding="utf-8"))
    require(len(table_manifest["tables"]) == 11
            and all(sha256_file(ROOT / row["csv_path"]) == row["csv_sha256"]
                    and sha256_file(ROOT / row["markdown_path"]) == row["markdown_sha256"]
                    for row in table_manifest["tables"]),
            "all eleven generated thesis tables reproduce", checks)
    figure_manifest = json.loads((ROOT / "artifacts/figures/phase06_figure_manifest.json").read_text(encoding="utf-8"))
    require(len(figure_manifest["figures"]) == 9 and not figure_manifest["seaborn_used"]
            and all(sha256_file(ROOT / row["data_path"]) == row["data_sha256"]
                    and sha256_file(ROOT / row["svg_path"]) == row["svg_sha256"]
                    and sha256_file(ROOT / row["png_path"]) == row["png_sha256"]
                    and row["png_dpi"] >= 300 for row in figure_manifest["figures"]),
            "all figure data, standalone SVGs, and 300-DPI PNGs reproduce", checks)
    fig6 = pd.read_csv(ROOT / "artifacts/figures/phase06_figure06_paired_unsupported_difference_data.csv")
    require(np.allclose(fig6.difference_unsupported_claim_rate_k10_minus_k5,
                        paired_source.difference_unsupported_claim_rate_k10_minus_k5),
            "paired-difference figure data match the canonical Phase 5 artifact", checks)
    fig7 = pd.read_csv(ROOT / "artifacts/figures/phase06_figure07_sufficiency_unsupported_data.csv")
    require(len(fig7) == 4 and set(map(tuple, fig7[["k", "context_group"]].to_numpy()))
            == {(5, "sufficient"), (5, "insufficient"),
                (10, "sufficient"), (10, "insufficient")},
            "sufficiency figure data match the four frozen groups", checks)

    manifest_path = ROOT / "artifacts/results/phase06_artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(all(sha256_file(ROOT / item["path"]) == item["physical_sha256"]
                for item in manifest["artifacts"]),
            "every Phase 6 manifest entry reproduces byte-for-byte", checks)
    require(manifest["techqa_test_sealed"] and not manifest["phase07_started"],
            "TechQA TEST remains sealed and Phase 7 remains unstarted", checks)
    require(not list(ROOT.glob("artifacts/**/*phase07*"))
            and not list(ROOT.glob("configs/*phase07*"))
            and not list(ROOT.glob("docs/PHASE_07*")),
            "no Phase 7 artifact exists", checks)

    print(json.dumps({
        "status": "pass", "check_count": len(checks), "checks": checks,
        "phase06_config_canonical_sha256": config.canonical_sha256,
        "phase06_manifest_sha256": sha256_file(manifest_path),
        "holm_rejections": tests.loc[tests.reject_holm.astype(bool), "comparison_id"].tolist(),
        "techqa_test_sealed": True, "phase07_started": False,
    }, indent=2))


if __name__ == "__main__":
    main()
