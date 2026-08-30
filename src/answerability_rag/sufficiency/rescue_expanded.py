"""Phase 3.6c expanded-grid definitions and descriptive precision interval."""

from __future__ import annotations

import math
from typing import Any


def expanded_candidate_grid(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand the exact predeclared 77-candidate Phase 3.6c grid."""
    families = config["candidate_families"]
    historical = {
        (row["family"], row["T_cov"], row["T_mean"], row["T_min"], row["T_contradiction"])
        for row in config["historical_phase03_6b_overlaps"]
    }
    rows: list[dict[str, Any]] = []
    order = 0

    def add(family: str, coverage: float | None, mean: float | None,
            minimum: float | None, contradiction: float | None) -> None:
        nonlocal order
        order += 1
        key = (family, coverage, mean, minimum, contradiction)
        rows.append({
            "candidate_order": order,
            "family": family,
            "T_cov": coverage,
            "T_mean": mean,
            "T_min": minimum,
            "T_contradiction": contradiction,
            "historically_evaluated_in_phase03_6b": key in historical,
            "new_phase03_6c_candidate": key not in historical,
        })

    for coverage in families["coverage_only"]["coverage_thresholds"]:
        add("coverage_only", float(coverage), None, None, None)
    nli = families["nli_only"]
    contradiction = float(nli["maximum_selected_premise_contradiction"])
    for mean in nli["mean_entailment_thresholds"]:
        for minimum in nli["minimum_entailment_thresholds"]:
            add("nli_only", None, float(mean), float(minimum), contradiction)
    combined = families["combined"]
    contradiction = float(combined["maximum_selected_premise_contradiction"])
    for coverage in combined["coverage_thresholds"]:
        for mean in combined["mean_entailment_thresholds"]:
            for minimum in combined["minimum_entailment_thresholds"]:
                add("combined", float(coverage), float(mean), float(minimum), contradiction)

    if len(rows) != 77 or len(rows) != int(config["candidate_count"]):
        raise ValueError(f"expanded rescue grid must contain 77 candidates, got {len(rows)}")
    historical_count = sum(row["historically_evaluated_in_phase03_6b"] for row in rows)
    new_count = sum(row["new_phase03_6c_candidate"] for row in rows)
    if historical_count != int(config["historically_evaluated_candidate_count"]):
        raise ValueError("expanded-grid historical-overlap count changed")
    if new_count != int(config["new_candidate_count"]):
        raise ValueError("expanded-grid new-candidate count changed")
    return rows


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> dict[str, float | int | None]:
    """Return the two-sided Wilson score interval without continuity correction."""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("invalid Wilson interval counts")
    if trials == 0:
        return {"successes": successes, "trials": trials, "confidence_level": 0.95,
                "lower": None, "upper": None}
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (proportion + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    return {"successes": successes, "trials": trials, "confidence_level": 0.95,
            "lower": max(0.0, centre - half), "upper": min(1.0, centre + half)}
