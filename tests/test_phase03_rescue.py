"""Phase 3.6b fixed-grid and strict-preservation tests."""

from __future__ import annotations

import json
from pathlib import Path

from answerability_rag.sufficiency.rescue import candidate_grid, final_prediction


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return json.loads((ROOT / "configs/phase03_rescue_grid.json").read_text(encoding="utf-8"))


def test_rescue_grid_is_exactly_the_predeclared_59_rules() -> None:
    rows = candidate_grid(_config())
    assert len(rows) == 59
    assert sum(row["family"] == "coverage_only" for row in rows) == 5
    assert sum(row["family"] == "nli_only" for row in rows) == 9
    assert sum(row["family"] == "combined" for row in rows) == 45
    assert {row["T_cov"] for row in rows if row["family"] == "coverage_only"} == {
        0.5, 0.6, 0.7, 0.8, 0.9
    }


def test_strict_positive_can_never_be_demoted() -> None:
    row = {"y_suff_strict": 1, "maximum_span_coverage_fraction": 0.0,
           "mean_claim_entailment": 0.0, "minimum_claim_entailment": 0.0,
           "maximum_selected_premise_contradiction": 1.0}
    for candidate in candidate_grid(_config()):
        assert final_prediction(row, candidate) == 1


def test_test_and_forbidden_boundary_inputs_are_frozen_out() -> None:
    config = _config()
    assert config["test_seal"] == {"allow_test_load": False,
                                   "allow_test_aggregate_outcomes": False}
    assert {"retrieval_strategy", "k", "question_id", "example_id", "sample_id",
            "split", "benchmark_is_impossible"} <= set(config["forbidden_boundary_inputs"])
    assert "semantic_unevaluable" in config["primary_export_exclusions"]
    assert "benchmark_impossible" in config["primary_export_exclusions"]


def test_confirmation_exclusions_cover_every_prior_human_review() -> None:
    confirmation = _config()["confirmation"]
    assert confirmation["exclude_original_development_questions"] is True
    assert confirmation["exclude_superseded_semantic_confirmation_questions"] is True
    assert confirmation["exclude_benchmark_impossible"] is True
    assert confirmation["exclude_semantic_unevaluable"] is True
