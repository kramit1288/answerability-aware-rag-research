"""Small statistical primitives with explicit dependency and missingness semantics."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class BootstrapInterval:
    """A percentile interval and the resampling audit needed to reproduce it."""

    point_estimate: float
    ci_low: float
    ci_high: float
    requested_replicates: int
    valid_replicates: int
    seed: int
    resampling_unit: str


def percentile_bounds(values: Sequence[float], confidence_level: float) -> tuple[float, float]:
    """Return deterministic linear percentile bounds from finite replicates."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.nan, np.nan
    tail = (1.0 - confidence_level) / 2.0
    return float(np.quantile(finite, tail)), float(np.quantile(finite, 1.0 - tail))


def cluster_bootstrap_samples(
    frame: pd.DataFrame,
    cluster_column: str,
    *,
    replicates: int,
    seed: int,
) -> Iterator[pd.DataFrame]:
    """Yield cluster bootstrap samples, retaining every row in each sampled cluster.

    Duplicate sampled clusters receive a bootstrap-occurrence column so downstream grouping does
    not accidentally collapse repeated occurrences. This is the central guard against treating
    TechQA retrieval conditions or RAGTruth responses as independent rows.
    """
    if cluster_column not in frame.columns:
        raise KeyError(f"missing bootstrap cluster column: {cluster_column}")
    if frame[cluster_column].isna().any():
        raise ValueError(f"bootstrap cluster column contains missing values: {cluster_column}")
    clusters = sorted(frame[cluster_column].unique().tolist(), key=str)
    if not clusters:
        raise ValueError("cannot bootstrap an empty cluster population")
    grouped = {cluster: frame.loc[frame[cluster_column] == cluster].copy() for cluster in clusters}
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        selected = rng.choice(np.asarray(clusters, dtype=object), size=len(clusters), replace=True)
        pieces: list[pd.DataFrame] = []
        for occurrence, cluster in enumerate(selected):
            piece = grouped[cluster].copy()
            piece["_bootstrap_occurrence"] = occurrence
            pieces.append(piece)
        yield pd.concat(pieces, ignore_index=True)


def cluster_bootstrap(
    frame: pd.DataFrame,
    cluster_column: str,
    statistic: Callable[[pd.DataFrame], float],
    *,
    replicates: int,
    seed: int,
    confidence_level: float,
) -> BootstrapInterval:
    """Calculate a percentile cluster-bootstrap interval without zero-filling undefined draws."""
    point = float(statistic(frame))
    sampled: list[float] = []
    for sample in cluster_bootstrap_samples(
        frame, cluster_column, replicates=replicates, seed=seed
    ):
        try:
            value = float(statistic(sample))
        except (ValueError, ZeroDivisionError, FloatingPointError):
            continue
        if np.isfinite(value):
            sampled.append(value)
    low, high = percentile_bounds(sampled, confidence_level)
    return BootstrapInterval(
        point_estimate=point,
        ci_low=low,
        ci_high=high,
        requested_replicates=replicates,
        valid_replicates=len(sampled),
        seed=seed,
        resampling_unit=cluster_column,
    )


def holm_adjust(p_values: Sequence[float], alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Apply Holm's step-down correction, preserving input order."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Holm correction requires a non-empty one-dimensional p-value vector")
    if np.isnan(values).any() or (values < 0).any() or (values > 1).any():
        raise ValueError("Holm correction requires finite p-values in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted_sorted = np.empty(values.size, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (values.size - rank) * values[index])
        running = max(running, candidate)
        adjusted_sorted[rank] = running
    adjusted = np.empty(values.size, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted, adjusted <= alpha


def matched_rank_biserial(differences: Sequence[float]) -> float:
    """Matched-pairs rank-biserial correlation using Pratt ranks and nonzero denominator."""
    diff = np.asarray(differences, dtype=float)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return np.nan
    ranks = stats.rankdata(np.abs(diff), method="average")
    positive = float(ranks[diff > 0].sum())
    negative = float(ranks[diff < 0].sum())
    denominator = positive + negative
    if denominator == 0:
        return 0.0
    return (positive - negative) / denominator


def cliffs_delta(group_a: Sequence[float], group_b: Sequence[float]) -> float:
    """Return P(A>B)-P(A<B), excluding missing observations without imputation."""
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan
    comparisons = a[:, None] - b[None, :]
    return float((np.sum(comparisons > 0) - np.sum(comparisons < 0)) / comparisons.size)


def paired_transition_counts(
    first: Sequence[object], second: Sequence[object]
) -> dict[str, int]:
    """Construct a paired 2x2 table after pairwise NA exclusion, never NA-to-false coercion."""
    left = pd.Series(first, dtype="object")
    right = pd.Series(second, dtype="object")
    if len(left) != len(right):
        raise ValueError("paired binary inputs must have equal length")
    complete = left.notna() & right.notna()
    left = left.loc[complete].astype(bool)
    right = right.loc[complete].astype(bool)
    return {
        "first_false_second_false": int((~left & ~right).sum()),
        "first_false_second_true": int((~left & right).sum()),
        "first_true_second_false": int((left & ~right).sum()),
        "first_true_second_true": int((left & right).sum()),
        "n_pairs": int(complete.sum()),
        "excluded_pairs": int((~complete).sum()),
    }


def exact_mcnemar(counts: dict[str, int]) -> tuple[float, float]:
    """Return the conventional exact McNemar statistic and two-sided binomial p-value."""
    b = int(counts["first_false_second_true"])
    c = int(counts["first_true_second_false"])
    discordant = b + c
    if discordant == 0:
        return 0.0, 1.0
    result = stats.binomtest(b, discordant, p=0.5, alternative="two-sided")
    return float(min(b, c)), float(result.pvalue)


def aligned_complete_pairs(
    frame: pd.DataFrame, first_column: str, second_column: str
) -> pd.DataFrame:
    """Return pairwise-complete aligned rows and fail on duplicate question identities."""
    required = {"question_id", first_column, second_column}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"missing paired columns: {sorted(missing)}")
    if frame.question_id.duplicated().any():
        raise ValueError("paired artifact must contain exactly one row per question_id")
    return frame.loc[frame[first_column].notna() & frame[second_column].notna()].copy()


def independent_complete_groups(
    frame: pd.DataFrame,
    group_column: str,
    outcome_column: str,
    group_a: object,
    group_b: object,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Construct named independent groups with metric-specific NA exclusion and audit count."""
    required = {group_column, outcome_column}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"missing independent-group columns: {sorted(missing)}")
    selected = frame.loc[frame[group_column].isin([group_a, group_b])].copy()
    complete = selected.loc[selected[outcome_column].notna()]
    a = complete.loc[complete[group_column] == group_a, outcome_column].to_numpy(dtype=float)
    b = complete.loc[complete[group_column] == group_b, outcome_column].to_numpy(dtype=float)
    if a.size == 0 or b.size == 0:
        raise ValueError("both independent groups require at least one complete observation")
    return a, b, int(len(selected) - len(complete))
