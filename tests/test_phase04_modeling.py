from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from answerability_rag.answerability.modeling import (
    ProbabilityCalibrator, StrategyScoreNormalizer, grouped_oof_probabilities,
    grouped_splits, make_pipeline, search_candidates,
)
from answerability_rag.answerability.registry import load_feature_registry


ROOT = Path(__file__).resolve().parents[1]


def _fixture() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    registry = load_feature_registry(ROOT / "configs/phase04_feature_registry.json")
    rows = []
    for group in range(10):
        for strategy_index, strategy in enumerate(("bm25", "dense", "hybrid")):
            row = {feature: float(group + strategy_index + 1) for feature in registry.numeric}
            row["retrieval_strategy"] = strategy
            row["k"] = (1, 3, 5)[strategy_index]
            if row["k"] == 1:
                for feature in registry.undefined:
                    row[feature] = np.nan
            rows.append(row)
    X = pd.DataFrame(rows, columns=registry.model_features)
    y = np.asarray([(group % 2) for group in range(10) for _ in range(3)])
    groups = np.asarray([f"q{group}" for group in range(10) for _ in range(3)])
    return X, y, groups


def test_grouped_cv_has_no_question_overlap() -> None:
    _, _, groups = _fixture()
    for train, heldout in grouped_splits(groups):
        assert set(groups[train]).isdisjoint(groups[heldout])


def test_score_normalizer_fits_train_only_and_validation_does_not_mutate() -> None:
    X, _, _ = _fixture()
    fitted = StrategyScoreNormalizer().fit(X.iloc[:24])
    before = dict(fitted.statistics_)
    validation = X.iloc[24:].copy()
    validation.loc[:, "retrieval_top1_score"] = 1_000_000.0
    fitted.transform(validation)
    assert fitted.statistics_ == before
    assert before[("bm25", "retrieval_top1_score")][0] != 1_000_000.0


def test_pipeline_is_seed_reproducible_and_validation_not_fitted() -> None:
    registry = load_feature_registry(ROOT / "configs/phase04_feature_registry.json")
    X, y, _ = _fixture()
    params = {"C": 1.0, "class_weight": None}
    first = make_pipeline("logistic_regression", registry, registry.model_features, params, 42).fit(X, y)
    second = make_pipeline("logistic_regression", registry, registry.model_features, params, 42).fit(X, y)
    np.testing.assert_allclose(first.predict_proba(X), second.predict_proba(X), atol=1e-12, rtol=0)
    imputer = first.named_steps["columns"].named_transformers_["numeric"].named_steps["imputer"]
    scaler = first.named_steps["columns"].named_transformers_["numeric"].named_steps["scaler"]
    statistics_before = imputer.statistics_.copy()
    mean_before = scaler.mean_.copy()
    extreme_validation = X.iloc[:3].copy()
    extreme_validation.loc[:, registry.numeric] = 1_000_000.0
    first.predict_proba(extreme_validation)
    np.testing.assert_array_equal(imputer.statistics_, statistics_before)
    np.testing.assert_array_equal(scaler.mean_, mean_before)


def test_candidate_selection_and_metric_table_are_reproducible() -> None:
    registry = load_feature_registry(ROOT / "configs/phase04_feature_registry.json")
    X, y, groups = _fixture()
    candidates = [{"C": 0.1, "class_weight": None}, {"C": 1.0, "class_weight": None}]
    first_selection, first_records = search_candidates(
        "logistic_regression", X, y, groups, registry, registry.model_features, candidates, 42,
    )
    second_selection, second_records = search_candidates(
        "logistic_regression", X, y, groups, registry, registry.model_features, candidates, 42,
    )
    assert first_selection == second_selection
    assert first_records == second_records


def test_grouped_oof_predictions_cover_each_row_without_group_contamination() -> None:
    registry = load_feature_registry(ROOT / "configs/phase04_feature_registry.json")
    X, y, groups = _fixture()
    pipeline = make_pipeline(
        "logistic_regression", registry, registry.model_features,
        {"C": 1.0, "class_weight": None}, 42,
    )
    probability, fold_ids, manifests = grouped_oof_probabilities(pipeline, X, y, groups)
    assert np.isfinite(probability).all()
    assert set(fold_ids) == set(range(5))
    assert all(row["zero_question_overlap"] for row in manifests)


def test_calibration_is_deterministic_and_has_no_validation_fit_argument() -> None:
    probability = np.array([0.1, 0.2, 0.7, 0.8])
    y = np.array([0, 0, 1, 1])
    first = ProbabilityCalibrator("sigmoid").fit(probability, y)
    second = ProbabilityCalibrator("sigmoid").fit(probability, y)
    validation = np.array([0.3, 0.9])
    np.testing.assert_allclose(first.transform(validation), second.transform(validation), atol=1e-12)
