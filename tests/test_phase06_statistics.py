"""Phase 6 statistical fixtures, grouping guards, integrity, and seal tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from scipy import stats

from answerability_rag.hashing import sha256_file
from answerability_rag.statistics.core import (
    aligned_complete_pairs,
    cliffs_delta,
    cluster_bootstrap_samples,
    exact_mcnemar,
    holm_adjust,
    independent_complete_groups,
    matched_rank_biserial,
    paired_transition_counts,
)
from answerability_rag.statistics.pipeline import Phase06Config, verify_upstream


ROOT = Path(__file__).resolve().parents[1]


def test_upstream_immutability() -> None:
    config = Phase06Config.load(ROOT / "configs/phase06_statistics.json", ROOT)
    observed = verify_upstream(ROOT, config)
    assert observed["phase05_manifest_sha256"] == "2f87445752de5013afe20961116fd024ee78dbb5a298fc2fe5fa8f2b2b57138e"
    manifest = json.loads((ROOT / "artifacts/results/phase05_artifact_manifest.json").read_text(encoding="utf-8"))
    assert all(sha256_file(ROOT / row["path"]) == row["physical_sha256"] for row in manifest["artifacts"])


def test_question_bootstrap_retains_cluster_rows() -> None:
    frame = pd.DataFrame({"question_id": ["q1", "q1", "q2", "q2"], "k": [5, 10, 5, 10]})
    for sample in cluster_bootstrap_samples(frame, "question_id", replicates=5, seed=42):
        assert len(sample) == 4
        assert sample.groupby("_bootstrap_occurrence").size().eq(2).all()
        assert sample.groupby("_bootstrap_occurrence").k.apply(set).eq({5, 10}).all()


def test_ragtruth_source_bootstrap_retains_source_responses() -> None:
    frame = pd.DataFrame({"source_id": ["s1"] * 3 + ["s2"] * 3, "response": range(6)})
    for sample in cluster_bootstrap_samples(frame, "source_id", replicates=5, seed=42):
        assert len(sample) == 6
        assert sample.groupby("_bootstrap_occurrence").size().eq(3).all()


def test_paired_alignment_and_missing_exclusion_without_zero_imputation() -> None:
    frame = pd.DataFrame({"question_id": ["q1", "q2", "q3"], "a": [1.0, np.nan, 4.0], "b": [2.0, 9.0, np.nan]})
    complete = aligned_complete_pairs(frame, "a", "b")
    assert complete.question_id.tolist() == ["q1"]
    assert complete.a.tolist() == [1.0]
    with pytest.raises(ValueError):
        aligned_complete_pairs(pd.concat([frame, frame.iloc[[0]]]), "a", "b")


def test_paired_difference_direction_and_wilcoxon_alignment() -> None:
    frame = pd.DataFrame({"question_id": ["q1", "q2", "q3"], "k5": [1.0, 2.0, 3.0], "k10": [2.0, 1.0, 5.0]})
    complete = aligned_complete_pairs(frame, "k5", "k10")
    differences = complete.k10 - complete.k5
    assert differences.tolist() == [1.0, -1.0, 2.0]
    observed = stats.wilcoxon(differences, zero_method="pratt", alternative="two-sided", method="auto")
    direct = stats.wilcoxon(complete.k10, complete.k5, zero_method="pratt", alternative="two-sided", method="auto")
    assert observed.statistic == direct.statistic
    assert observed.pvalue == direct.pvalue


def test_mcnemar_contingency_and_exact_result() -> None:
    counts = paired_transition_counts([False, False, True, True], [False, True, False, True])
    assert counts == {
        "first_false_second_false": 1, "first_false_second_true": 1,
        "first_true_second_false": 1, "first_true_second_true": 1,
        "n_pairs": 4, "excluded_pairs": 0,
    }
    statistic, p_value = exact_mcnemar(counts)
    assert statistic == 1
    assert p_value == 1


def test_fisher_and_mann_whitney_group_construction() -> None:
    frame = pd.DataFrame({"group": [1, 1, 0, 0, 0], "continuous": [3.0, 4.0, 1.0, 2.0, np.nan], "binary": [1, 1, 0, 1, np.nan]})
    sufficient, insufficient, excluded = independent_complete_groups(frame, "group", "continuous", 1, 0)
    assert sufficient.tolist() == [3.0, 4.0]
    assert insufficient.tolist() == [1.0, 2.0]
    assert excluded == 1
    assert stats.mannwhitneyu(sufficient, insufficient).statistic == 4
    binary_s, binary_i, binary_excluded = independent_complete_groups(frame, "group", "binary", 1, 0)
    assert binary_excluded == 1
    table = [[int(binary_s.sum()), len(binary_s) - int(binary_s.sum())],
             [int(binary_i.sum()), len(binary_i) - int(binary_i.sum())]]
    assert table == [[2, 0], [1, 1]]
    assert 0 <= stats.fisher_exact(table).pvalue <= 1


def test_cliffs_delta_fixture() -> None:
    assert cliffs_delta([3, 4], [1, 2]) == 1.0
    assert cliffs_delta([1, 2], [3, 4]) == -1.0


def test_rank_biserial_fixture() -> None:
    assert np.isclose(matched_rank_biserial([1.0, 2.0, -3.0]), 0.0)
    assert matched_rank_biserial([1.0, 2.0, 3.0]) == 1.0


def test_holm_fixture() -> None:
    adjusted, rejected = holm_adjust([0.01, 0.04, 0.03], alpha=0.05)
    assert np.allclose(adjusted, [0.03, 0.06, 0.06])
    assert rejected.tolist() == [True, False, False]


def test_g2_tiny_sample_guard_and_abstention_na() -> None:
    config = json.loads((ROOT / "configs/phase06_statistics.json").read_text(encoding="utf-8"))
    g2 = pq.read_table(ROOT / "artifacts/results/phase05_policy_G2.parquet").to_pandas()
    assert g2.answered.sum() == 2
    assert int(g2.answered.sum()) < config["tiny_sample_guard"]["minimum_answered_n_for_quality_bootstrap"]
    assert g2.loc[~g2.answered.astype(bool), ["rouge_l_f1", "bertscore_f1", "unsupported_claim_rate"]].isna().all().all()


def test_figure_data_results_consistency() -> None:
    paired = pq.read_table(ROOT / "artifacts/results/phase05_paired_k5_k10.parquet").to_pandas()
    figure = pd.read_csv(ROOT / "artifacts/figures/phase06_figure06_paired_unsupported_difference_data.csv")
    assert figure.question_id.tolist() == paired.question_id.tolist()
    assert np.allclose(figure.difference_unsupported_claim_rate_k10_minus_k5,
                       paired.difference_unsupported_claim_rate_k10_minus_k5)


def test_techqa_test_seal_and_phase7_absence() -> None:
    summary = json.loads((ROOT / "artifacts/results/phase06_statistical_summary.json").read_text(encoding="utf-8"))
    assert summary["techqa_test_accessed"] is False
    assert summary["techqa_test_sealed"] is True
    assert summary["phase07_started"] is False
    assert not list(ROOT.glob("artifacts/**/*phase07*"))
    assert not list(ROOT.glob("configs/*phase07*"))
    assert not list(ROOT.glob("docs/PHASE_07*"))
