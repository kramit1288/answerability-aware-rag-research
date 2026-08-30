"""Grouped model development, calibration, and classifier evaluation."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .registry import FeatureRegistry


SCORE_FEATURES = (
    "retrieval_top1_score", "retrieval_top2_score", "retrieval_top1_top2_gap",
    "retrieval_score_mean", "retrieval_score_std", "retrieval_score_min",
    "retrieval_score_max", "retrieval_score_range",
)


class StrategyScoreNormalizer(BaseEstimator, TransformerMixin):
    """TRAIN-fitted within-strategy score normalization preserving missingness."""

    def __init__(self, score_features: Sequence[str] = SCORE_FEATURES):
        self.score_features = tuple(score_features)

    def fit(self, X: pd.DataFrame, y: Any = None) -> "StrategyScoreNormalizer":
        frame = self._frame(X)
        self.statistics_: dict[tuple[str, str], tuple[float, float]] = {}
        for strategy in ("bm25", "dense", "hybrid"):
            subset = frame.loc[frame["retrieval_strategy"] == strategy]
            if subset.empty:
                raise ValueError(f"TRAIN fit lacks retrieval strategy {strategy}")
            for feature in self.score_features:
                values = pd.to_numeric(subset[feature], errors="coerce").dropna().to_numpy(float)
                if not len(values):
                    self.statistics_[(strategy, feature)] = (math.nan, math.nan)
                else:
                    self.statistics_[(strategy, feature)] = (
                        float(values.mean()), float(values.std(ddof=0)),
                    )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "statistics_"):
            raise RuntimeError("StrategyScoreNormalizer is not fitted")
        frame = self._frame(X).copy()
        unexpected = sorted(set(frame["retrieval_strategy"]) - {"bm25", "dense", "hybrid"})
        if unexpected:
            raise ValueError(f"unknown retrieval strategies: {unexpected}")
        for strategy in ("bm25", "dense", "hybrid"):
            mask = frame["retrieval_strategy"] == strategy
            for feature in self.score_features:
                mean, std = self.statistics_[(strategy, feature)]
                values = pd.to_numeric(frame.loc[mask, feature], errors="coerce")
                if math.isnan(mean):
                    frame.loc[mask, feature] = np.nan
                elif not std:
                    frame.loc[mask & frame[feature].notna(), feature] = 0.0
                else:
                    frame.loc[mask, feature] = (values - mean) / std
        return frame

    @staticmethod
    def _frame(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Phase 4 preprocessing requires a named pandas DataFrame")
        return X


def make_pipeline(
    family: str, registry: FeatureRegistry, features: Sequence[str], params: dict[str, Any], seed: int,
) -> Pipeline:
    registry.validate_model_columns(features)
    numeric = [feature for feature in features if feature in registry.numeric]
    categorical = [feature for feature in features if feature in registry.categorical]
    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ]
    if family == "logistic_regression":
        numeric_steps.append(("scaler", StandardScaler()))
        estimator = LogisticRegression(
            C=params["C"], class_weight=params["class_weight"], solver="lbfgs",
            max_iter=5000, random_state=seed,
        )
    elif family == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=500, max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"], max_features=params["max_features"],
            class_weight=params["class_weight"], random_state=seed, n_jobs=-1,
        )
    else:
        raise ValueError(f"unknown model family: {family}")
    transformers: list[tuple[str, Any, list[str]]] = [
        ("numeric", Pipeline(numeric_steps), numeric),
    ]
    if categorical:
        transformers.append((
            "categorical",
            OneHotEncoder(categories=[["bm25", "dense", "hybrid"]],
                          handle_unknown="error", sparse_output=False),
            categorical,
        ))
    column_transformer = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    return Pipeline([
        ("score_normalizer", StrategyScoreNormalizer()),
        ("columns", column_transformer),
        ("model", estimator),
    ])


def grouped_splits(groups: Sequence[str], n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    groups_array = np.asarray(groups)
    splitter = GroupKFold(n_splits=n_splits)
    dummy = np.zeros(len(groups_array))
    folds = list(splitter.split(dummy, dummy, groups_array))
    for train, heldout in folds:
        overlap = set(groups_array[train]).intersection(groups_array[heldout])
        if overlap:
            raise AssertionError(f"question leakage in grouped fold: {sorted(overlap)[:3]}")
    return folds


def fold_manifest(
    y: Sequence[int], groups: Sequence[str], folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *, purpose: str,
) -> list[dict[str, Any]]:
    labels, qids = np.asarray(y, dtype=int), np.asarray(groups)
    records = []
    for fold_id, (train, heldout) in enumerate(folds):
        overlap = set(qids[train]).intersection(qids[heldout])
        records.append({
            "purpose": purpose, "fold": fold_id,
            "train_question_count": len(set(qids[train])),
            "train_condition_count": len(train),
            "train_positive_count": int(labels[train].sum()),
            "train_negative_count": int(len(train) - labels[train].sum()),
            "heldout_question_count": len(set(qids[heldout])),
            "heldout_condition_count": len(heldout),
            "heldout_positive_count": int(labels[heldout].sum()),
            "heldout_negative_count": int(len(heldout) - labels[heldout].sum()),
            "question_overlap_count": len(overlap),
            "zero_question_overlap": not overlap,
        })
    return records


def _rank_metrics(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    return float(average_precision_score(y, probability)), float(roc_auc_score(y, probability))


def search_candidates(
    family: str, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
    registry: FeatureRegistry, features: Sequence[str], candidates: Sequence[dict[str, Any]], seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    folds = grouped_splits(groups)
    records: list[dict[str, Any]] = []
    for order, params in enumerate(candidates):
        fold_ap, fold_auc = [], []
        for train, heldout in folds:
            model = make_pipeline(family, registry, features, params, seed)
            model.fit(X.iloc[train], y[train])
            probability = model.predict_proba(X.iloc[heldout])[:, 1]
            ap, auc = _rank_metrics(y[heldout], probability)
            fold_ap.append(ap); fold_auc.append(auc)
        records.append({
            "model_family": family, "candidate_order": order, **params,
            "mean_auprc": float(np.mean(fold_ap)), "mean_auroc": float(np.mean(fold_auc)),
            "fold_auprc_json": fold_ap, "fold_auroc_json": fold_auc,
        })
    selected = max(records, key=lambda row: (row["mean_auprc"], row["mean_auroc"], -row["candidate_order"]))
    for row in records:
        row["selected"] = row is selected
    params = {key: selected[key] for key in candidates[0]}
    return params, records


def search_b1_threshold(
    feature: Sequence[float], y: np.ndarray, groups: np.ndarray,
    thresholds: Iterable[float],
) -> tuple[float, list[dict[str, Any]]]:
    values = np.asarray(feature, dtype=float)
    folds = grouped_splits(groups)
    records = []
    for threshold in thresholds:
        fold_f1, fold_precision = [], []
        for _, heldout in folds:
            predicted = (values[heldout] >= threshold).astype(int)
            fold_f1.append(float(f1_score(y[heldout], predicted, zero_division=0)))
            fold_precision.append(float(precision_score(y[heldout], predicted, zero_division=0)))
        records.append({
            "threshold": float(threshold), "mean_f1": float(np.mean(fold_f1)),
            "mean_precision": float(np.mean(fold_precision)),
            "fold_f1_json": fold_f1, "fold_precision_json": fold_precision,
        })
    selected = max(records, key=lambda row: (row["mean_f1"], row["mean_precision"], row["threshold"]))
    for row in records:
        row["selected"] = row is selected
    return selected["threshold"], records


@dataclass
class ProbabilityCalibrator:
    method: str
    mapping: Any = None

    def fit(self, probability: np.ndarray, y: np.ndarray) -> "ProbabilityCalibrator":
        if self.method == "uncalibrated":
            self.mapping = None
        elif self.method == "sigmoid":
            self.mapping = LogisticRegression(
                C=1_000_000, solver="lbfgs", max_iter=5000, random_state=42,
            ).fit(np.asarray(probability).reshape(-1, 1), y)
        elif self.method == "isotonic":
            self.mapping = IsotonicRegression(out_of_bounds="clip").fit(probability, y)
        else:
            raise ValueError(f"unknown calibration method: {self.method}")
        return self

    def transform(self, probability: np.ndarray) -> np.ndarray:
        values = np.asarray(probability, dtype=float)
        if self.method == "uncalibrated":
            return values.copy()
        if self.method == "sigmoid":
            return self.mapping.predict_proba(values.reshape(-1, 1))[:, 1]
        return np.asarray(self.mapping.predict(values), dtype=float)


def grouped_oof_probabilities(
    pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    folds = grouped_splits(groups)
    oof = np.full(len(y), np.nan, dtype=float)
    fold_ids = np.full(len(y), -1, dtype=int)
    for fold_id, (train, heldout) in enumerate(folds):
        fitted = clone(pipeline).fit(X.iloc[train], y[train])
        oof[heldout] = fitted.predict_proba(X.iloc[heldout])[:, 1]
        fold_ids[heldout] = fold_id
    if np.isnan(oof).any() or (fold_ids < 0).any():
        raise AssertionError("grouped OOF predictions are incomplete")
    return oof, fold_ids, fold_manifest(y, groups, folds, purpose="rf_oof_calibration")


def reliability_bins(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> tuple[float, list[dict[str, Any]]]:
    y = np.asarray(y, dtype=int); probability = np.asarray(probability, dtype=float)
    indexes = np.minimum((probability * bins).astype(int), bins - 1)
    records, ece = [], 0.0
    for index in range(bins):
        mask = indexes == index
        count = int(mask.sum())
        confidence = float(probability[mask].mean()) if count else None
        accuracy = float(y[mask].mean()) if count else None
        contribution = count / len(y) * abs(accuracy - confidence) if count else 0.0
        ece += contribution
        records.append({
            "bin": index, "lower": index / bins, "upper": (index + 1) / bins,
            "upper_inclusive": index == bins - 1, "count": count,
            "mean_probability": confidence, "positive_fraction": accuracy,
            "ece_contribution": contribution,
        })
    return float(ece), records


def classifier_metrics(y: np.ndarray, probability: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y = np.asarray(y, dtype=int); probability = np.asarray(probability, dtype=float)
    predicted = (probability >= threshold).astype(int)
    ece, _ = reliability_bins(y, probability)
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "auroc": float(roc_auc_score(y, probability)),
        "auprc": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "ece": ece,
    }


def grouped_bootstrap_intervals(
    y: np.ndarray, probability: np.ndarray, groups: np.ndarray, *, replicates: int = 1000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    y = np.asarray(y, dtype=int); probability = np.asarray(probability, dtype=float)
    groups = np.asarray(groups); unique = np.asarray(sorted(set(groups)))
    indexes = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in ("auroc", "auprc", "f1", "brier")}
    for _ in range(replicates):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        selected = np.concatenate([indexes[group] for group in drawn])
        metrics = classifier_metrics(y[selected], probability[selected])
        for name in samples:
            samples[name].append(metrics[name])
    point = classifier_metrics(y, probability)
    return [{
        "metric": name, "point_estimate": point[name], "replicates": replicates,
        "lower_95": float(np.percentile(values, 2.5)),
        "upper_95": float(np.percentile(values, 97.5)), "seed": seed,
        "resampling_unit": "question_id",
    } for name, values in samples.items()]


def logistic_candidates() -> list[dict[str, Any]]:
    return [{"C": c, "class_weight": weight}
            for c, weight in itertools.product((0.1, 1.0, 10.0), (None, "balanced"))]


def random_forest_candidates() -> list[dict[str, Any]]:
    # Explicit ordering implements the frozen simplicity tie-break.
    return [{"max_depth": depth, "min_samples_leaf": leaf,
             "max_features": features, "class_weight": weight}
            for depth, leaf, features, weight in itertools.product(
                (4, 8, None), (10, 5, 2), ("sqrt", 0.5), (None, "balanced")
            )]
