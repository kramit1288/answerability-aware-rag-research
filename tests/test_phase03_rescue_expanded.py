"""Phase 3.6c final-iteration grid and confidence-interval tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from answerability_rag.sufficiency.rescue import final_prediction
from answerability_rag.sufficiency.rescue_expanded import expanded_candidate_grid, wilson_interval


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return json.loads((ROOT / "configs/phase03_rescue_grid_expanded.json").read_text(encoding="utf-8"))


def test_expanded_grid_has_exact_frozen_family_counts() -> None:
    rows = expanded_candidate_grid(_config())
    assert len(rows) == 77
    assert sum(row["family"] == "coverage_only" for row in rows) == 5
    assert sum(row["family"] == "nli_only" for row in rows) == 12
    assert sum(row["family"] == "combined" for row in rows) == 60
    assert sum(row["historically_evaluated_in_phase03_6b"] for row in rows) == 3
    assert sum(row["new_phase03_6c_candidate"] for row in rows) == 74


def test_expanded_grid_contains_only_declared_thresholds() -> None:
    rows = expanded_candidate_grid(_config())
    assert {row["T_cov"] for row in rows if row["T_cov"] is not None} == {0.2, 0.3, 0.4, 0.45, 0.5}
    assert {row["T_mean"] for row in rows if row["T_mean"] is not None} == {0.3, 0.35, 0.4, 0.45}
    assert {row["T_min"] for row in rows if row["T_min"] is not None} == {0.0, 0.05, 0.1}
    assert {row["T_contradiction"] for row in rows if row["T_contradiction"] is not None} == {0.5}


def test_every_expanded_candidate_preserves_strict_positive() -> None:
    row = {"y_suff_strict": 1, "maximum_span_coverage_fraction": 0.0,
           "mean_claim_entailment": 0.0, "minimum_claim_entailment": 0.0,
           "maximum_selected_premise_contradiction": 1.0}
    assert all(final_prediction(row, candidate) == 1 for candidate in expanded_candidate_grid(_config()))


def test_wilson_interval_matches_selected_precision_counts() -> None:
    interval = wilson_interval(59, 60)
    assert interval["lower"] == pytest.approx(0.9114487027240993)
    assert interval["upper"] == pytest.approx(0.9970518402052136)
    assert interval["lower"] < 59 / 60 < interval["upper"]


def test_final_iteration_and_test_seals_are_predeclared() -> None:
    config = _config()
    assert config["iteration_governance"]["final_automatic_threshold_refinement_iteration"] is True
    assert config["iteration_governance"]["allow_further_autonomous_threshold_search"] is False
    assert config["test_seal"] == {"allow_test_load": False,
                                   "allow_test_aggregate_outcomes": False}
    assert config["development_gates"] == {
        "minimum_ordinary_precision": 0.9,
        "minimum_prevalence_weighted_f1": 0.85,
    }
