from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from answerability_rag.answerability.registry import assert_test_sealed
from answerability_rag.answerability.selective import (
    aurc, policy_action, risk_coverage_curve, simulate_policy,
)


def test_risk_coverage_and_aurc_known_fixture() -> None:
    curve = risk_coverage_curve(np.array([1, 0, 1]), np.array([0.9, 0.8, 0.2]))
    observed = [(row["coverage"], row["selective_risk"]) for row in curve]
    assert observed[0] == (0.0, None)
    assert observed[1] == (pytest.approx(1 / 3), 0.0)
    assert observed[2] == (pytest.approx(2 / 3), 0.5)
    assert observed[3] == (1.0, pytest.approx(1 / 3))
    assert aurc(curve) == pytest.approx((0 + 0.5 + 1 / 3) / 3)
    assert [row["threshold"] for row in curve] == sorted(
        [row["threshold"] for row in curve], reverse=True
    )


@pytest.mark.parametrize(("p5", "p10", "low", "high", "expected"), [
    (0.8, 0.0, 0.2, 0.7, ("ANSWER_AT_K5", False, 5)),
    (0.1, 1.0, 0.2, 0.7, ("ABSTAIN", False, 5)),
    (0.5, 0.8, 0.2, 0.7, ("ANSWER_AT_K10", True, 10)),
    (0.5, 0.6, 0.2, 0.7, ("ABSTAIN", True, 10)),
    (0.2, 0.8, 0.2, 0.7, ("ANSWER_AT_K10", True, 10)),
    (0.7, 0.0, 0.2, 0.7, ("ANSWER_AT_K5", False, 5)),
])
def test_all_policy_transitions_and_exact_boundaries(p5, p10, low, high, expected) -> None:
    assert policy_action(p5, p10, low, high) == expected


def test_policy_denominators_and_no_second_expansion() -> None:
    frame = pd.DataFrame({
        "question_id": ["a", "b", "c", "d"],
        "p5": [0.9, 0.1, 0.5, 0.5], "p10": [0.0, 1.0, 0.9, 0.4],
        "y5": [0, 0, 0, 0], "y10": [1, 1, 1, 0],
    })
    metrics, rows = simulate_policy(frame, 0.2, 0.7)
    assert metrics["unsafe_answer_count"] == 1
    assert metrics["safe_answer_count"] == 1
    assert metrics["final_selective_risk"] == 0.5  # unsafe / all answered
    assert metrics["false_abstention_count"] == 1
    assert metrics["false_abstention_denominator"] == 3
    assert metrics["false_abstention_rate"] == pytest.approx(1 / 3)
    assert all(row["retrieved_k"] in (5, 10) for row in rows)
    assert max(row["retrieved_k"] for row in rows) == 10


def test_phase04_test_seal_rejects_aggregate_and_inference_operations() -> None:
    for operation in ("feature generation", "aggregate metrics", "policy search", "importance"):
        with pytest.raises(PermissionError, match="TEST seal"):
            assert_test_sealed("TEST", operation)
