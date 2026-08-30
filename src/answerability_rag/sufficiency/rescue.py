"""Frozen Phase 3.6b strict-preserving rescue-rule definitions and evaluation."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from .semantic import confusion, metrics, weighted_metrics


FAMILY_ORDER = {"coverage_only": 0, "nli_only": 1, "combined": 2}


def candidate_grid(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand exactly the predeclared 59 rescue candidates in deterministic order."""
    families = config["candidate_families"]
    rows: list[dict[str, Any]] = []
    order = 0
    for threshold in families["coverage_only"]["coverage_thresholds"]:
        order += 1
        rows.append({"candidate_order": order, "family": "coverage_only",
                     "T_cov": float(threshold), "T_mean": None, "T_min": None,
                     "T_contradiction": None})
    nli = families["nli_only"]
    for mean in nli["mean_entailment_thresholds"]:
        for minimum in nli["minimum_entailment_thresholds"]:
            order += 1
            rows.append({"candidate_order": order, "family": "nli_only",
                         "T_cov": None, "T_mean": float(mean),
                         "T_min": float(minimum),
                         "T_contradiction": float(nli["maximum_selected_premise_contradiction"])})
    combined = families["combined"]
    for coverage in combined["coverage_thresholds"]:
        for mean in combined["mean_entailment_thresholds"]:
            for minimum in combined["minimum_entailment_thresholds"]:
                order += 1
                rows.append({"candidate_order": order, "family": "combined",
                             "T_cov": float(coverage), "T_mean": float(mean),
                             "T_min": float(minimum),
                             "T_contradiction": float(
                                 combined["maximum_selected_premise_contradiction"]
                             )})
    if len(rows) != 59 or len(rows) != int(config["candidate_count"]):
        raise ValueError(f"frozen rescue grid must contain 59 candidates, got {len(rows)}")
    return rows


def rescue_mechanisms(row: dict[str, Any], candidate: dict[str, Any]) -> tuple[bool, bool]:
    """Return coverage and NLI rescue signals for one strict-negative row."""
    coverage = (
        candidate["T_cov"] is not None
        and row.get("maximum_span_coverage_fraction") is not None
        and float(row["maximum_span_coverage_fraction"]) >= float(candidate["T_cov"])
    )
    nli = (
        candidate["T_mean"] is not None
        and row.get("mean_claim_entailment") is not None
        and row.get("minimum_claim_entailment") is not None
        and row.get("maximum_selected_premise_contradiction") is not None
        and float(row["mean_claim_entailment"]) >= float(candidate["T_mean"])
        and float(row["minimum_claim_entailment"]) >= float(candidate["T_min"])
        and float(row["maximum_selected_premise_contradiction"])
        < float(candidate["T_contradiction"])
    )
    return coverage, nli


def final_prediction(row: dict[str, Any], candidate: dict[str, Any]) -> int:
    """Apply the strict-preserving rule; a strict positive can never be demoted."""
    if int(row["y_suff_strict"]) == 1:
        return 1
    coverage, nli = rescue_mechanisms(row, candidate)
    family = candidate["family"]
    return int(coverage if family == "coverage_only" else nli if family == "nli_only"
               else coverage or nli)


def evaluate_candidate(
    rows: list[dict[str, Any]], candidate: dict[str, Any],
    population_counts: dict[str, int], strata: Iterable[str],
) -> dict[str, Any]:
    evaluated = [{**row, "prediction": final_prediction(row, candidate)} for row in rows]
    cells = confusion(evaluated)
    sample = metrics(cells)
    weighted = weighted_metrics(evaluated, population_counts, strata)
    mechanisms = Counter()
    for row in evaluated:
        if int(row["y_suff_strict"]) == 0 and row["prediction"] == 1:
            cov, nli = rescue_mechanisms(row, candidate)
            mechanisms["coverage"] += int(cov)
            mechanisms["nli"] += int(nli)
            mechanisms["overlap"] += int(cov and nli)
    return {
        **candidate,
        "eligible_count": len(evaluated),
        **cells,
        "sample_accuracy": sample["accuracy"],
        "sample_precision": sample["precision"],
        "sample_recall": sample["recall"],
        "sample_f1": sample["f1"],
        "weighted_accuracy": weighted["accuracy"],
        "weighted_precision": weighted["precision"],
        "weighted_recall": weighted["recall"],
        "weighted_f1": weighted["f1"],
        "weighted_confusion_proportions": weighted["estimated_confusion_proportions"],
        "metrics_by_stratum": weighted["metrics_by_stratum"],
        "rescued_by_coverage": mechanisms["coverage"],
        "rescued_by_nli": mechanisms["nli"],
        "rescue_mechanism_overlap": mechanisms["overlap"],
    }


def select_candidate(results: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any] | None:
    """Select by frozen precision gate, weighted F1, recall, then simplicity."""
    gate = float(config["selection"]["minimum_ordinary_precision"])
    eligible = [row for row in results if row["sample_precision"] is not None
                and float(row["sample_precision"]) >= gate]
    if not eligible:
        return None
    tolerance = float(config["selection"]["effective_tie_absolute_tolerance"])
    best_f1 = max(float(row["weighted_f1"]) for row in eligible)
    tied = [row for row in eligible if abs(float(row["weighted_f1"]) - best_f1) <= tolerance]
    best_recall = max(float(row["sample_recall"]) for row in tied)
    tied = [row for row in tied if abs(float(row["sample_recall"]) - best_recall) <= tolerance]
    return min(tied, key=lambda row: (FAMILY_ORDER[row["family"]], row["candidate_order"]))


def best_by_family(results: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output = {}
    for family in FAMILY_ORDER:
        family_rows = [row for row in results if row["family"] == family]
        output[family] = select_candidate(family_rows, config) or max(
            family_rows,
            key=lambda row: (
                -1.0 if row["sample_precision"] is None else float(row["sample_precision"]),
                -1.0 if row["weighted_f1"] is None else float(row["weighted_f1"]),
                -1.0 if row["sample_recall"] is None else float(row["sample_recall"]),
                -int(row["candidate_order"]),
            ),
        )
    return output
