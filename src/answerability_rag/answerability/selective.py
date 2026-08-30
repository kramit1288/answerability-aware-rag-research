"""Risk--coverage and non-generative three-way retrieval policy logic."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import pandas as pd


def risk_coverage_curve(y: np.ndarray, probability: np.ndarray) -> list[dict[str, Any]]:
    y = np.asarray(y, dtype=int); probability = np.asarray(probability, dtype=float)
    if len(y) != len(probability) or not len(y):
        raise ValueError("risk-coverage inputs must be nonempty and aligned")
    thresholds = [float(np.nextafter(probability.max(), np.inf))]
    thresholds.extend(float(value) for value in sorted(set(probability), reverse=True))
    if 0.0 not in thresholds:
        thresholds.append(0.0)
    records = []
    for threshold in thresholds:
        answered = probability >= threshold
        count = int(answered.sum())
        unsafe = int(((y == 0) & answered).sum())
        records.append({
            "threshold": threshold, "answered_count": count, "total_count": len(y),
            "coverage": count / len(y), "unsafe_answered_count": unsafe,
            "selective_risk": unsafe / count if count else None,
        })
    return records


def aurc(curve: list[dict[str, Any]]) -> float:
    """Right-endpoint rectangle AURC over increasing observed coverage."""
    area, previous = 0.0, 0.0
    for row in sorted(curve, key=lambda item: item["coverage"]):
        coverage, risk = float(row["coverage"]), row["selective_risk"]
        if risk is not None and coverage > previous:
            area += (coverage - previous) * float(risk)
        previous = max(previous, coverage)
    return float(area)


def select_risk_operating_point(
    curve: list[dict[str, Any]], risk_constraint: float,
) -> dict[str, Any]:
    feasible = [row for row in curve if row["selective_risk"] is not None
                and row["selective_risk"] <= risk_constraint]
    if not feasible:
        return {"risk_constraint": risk_constraint, "feasible": False,
                "threshold": None, "coverage": None, "selective_risk": None}
    selected = max(feasible, key=lambda row: (row["coverage"], row["threshold"]))
    return {"risk_constraint": risk_constraint, "feasible": True,
            "threshold": selected["threshold"], "coverage": selected["coverage"],
            "selective_risk": selected["selective_risk"]}


def policy_action(p5: float, p10: float, t_low: float, t_high: float) -> tuple[str, bool, int]:
    """Return final action, whether expansion occurred, and final retrieval depth."""
    if not 0 <= t_low < t_high <= 1:
        raise ValueError("policy requires 0 <= t_low < t_high <= 1")
    if p5 >= t_high:
        return "ANSWER_AT_K5", False, 5
    if p5 < t_low:
        return "ABSTAIN", False, 5
    if p10 >= t_high:
        return "ANSWER_AT_K10", True, 10
    return "ABSTAIN", True, 10


def simulate_policy(
    trajectories: pd.DataFrame, t_low: float, t_high: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    required = {"question_id", "p5", "p10", "y5", "y10"}
    if set(trajectories.columns) != required:
        missing = sorted(required - set(trajectories.columns))
        extra = sorted(set(trajectories.columns) - required)
        raise ValueError(f"policy trajectory columns differ: missing={missing}, extra={extra}")
    rows = []
    for row in trajectories.sort_values("question_id").itertuples(index=False):
        action, expanded, depth = policy_action(row.p5, row.p10, t_low, t_high)
        answered = action.startswith("ANSWER")
        safe = bool(row.y5) if action == "ANSWER_AT_K5" else (
            bool(row.y10) if action == "ANSWER_AT_K10" else None
        )
        rows.append({
            "question_id": row.question_id, "p5": float(row.p5),
            "p10": float(row.p10) if expanded else None, "y5": int(row.y5),
            "y10": int(row.y10), "t_low": t_low, "t_high": t_high,
            "expanded": expanded, "final_action": action, "answered": answered,
            "safe_answer": safe, "retrieved_k": depth, "retrieved_chunk_count": depth,
        })
    n = len(rows)
    answered_rows = [row for row in rows if row["answered"]]
    unsafe = sum(row["safe_answer"] is False for row in answered_rows)
    safe = sum(row["safe_answer"] is True for row in answered_rows)
    expanded = sum(row["expanded"] for row in rows)
    possible = sum(row["y10"] == 1 for row in rows)
    false_abstentions = sum(row["final_action"] == "ABSTAIN" and row["y10"] == 1 for row in rows)
    mean_k = float(np.mean([row["retrieved_k"] for row in rows]))
    metrics = {
        "t_low": t_low, "t_high": t_high, "eligible_trajectory_count": n,
        "final_answer_coverage": len(answered_rows) / n,
        "final_selective_risk": unsafe / len(answered_rows) if answered_rows else None,
        "retrieval_expansion_rate": expanded / n,
        "false_abstention_rate": false_abstentions / possible if possible else None,
        "false_abstention_count": false_abstentions,
        "false_abstention_denominator": possible,
        "unsafe_answer_count": unsafe, "safe_answer_count": safe,
        "abstention_count": n - len(answered_rows), "mean_retrieved_k": mean_k,
        "mean_retrieved_chunk_count": mean_k,
        "relative_retrieval_depth_cost_proxy": mean_k / 5.0,
    }
    return metrics, rows


def policy_grid(trajectories: pd.DataFrame, thresholds: Iterable[float]) -> list[dict[str, Any]]:
    values = list(thresholds)
    return [simulate_policy(trajectories, low, high)[0]
            for low in values for high in values if low < high]


def select_policy(grid: list[dict[str, Any]], risk_constraint: float) -> dict[str, Any]:
    feasible = [row for row in grid if row["final_selective_risk"] is not None
                and row["final_selective_risk"] <= risk_constraint]
    if not feasible:
        return {"risk_constraint": risk_constraint, "feasible": False}
    selected = max(feasible, key=lambda row: (
        row["final_answer_coverage"], -row["retrieval_expansion_rate"],
        row["t_high"], row["t_low"],
    ))
    return {"risk_constraint": risk_constraint, "feasible": True, **selected}


def two_way_baseline(
    trajectories: pd.DataFrame, *, at_k: int, threshold: float,
) -> dict[str, Any]:
    if at_k not in (5, 10):
        raise ValueError("two-way baseline supports only k=5 or k=10")
    probability = trajectories[f"p{at_k}"].to_numpy(float)
    target = trajectories[f"y{at_k}"].to_numpy(int)
    answered = probability >= threshold
    count = int(answered.sum()); unsafe = int(((target == 0) & answered).sum())
    possible = int(trajectories["y10"].sum())
    abstained = ~answered
    false_abstentions = int((abstained & (trajectories["y10"].to_numpy(int) == 1)).sum())
    mean_k = float(at_k)
    return {
        "threshold": threshold, "final_answer_coverage": count / len(target),
        "final_selective_risk": unsafe / count if count else None,
        "retrieval_expansion_rate": 0.0 if at_k == 5 else 1.0,
        "false_abstention_rate": false_abstentions / possible if possible else None,
        "unsafe_answer_count": unsafe, "safe_answer_count": count - unsafe,
        "abstention_count": len(target) - count, "mean_retrieved_k": mean_k,
        "mean_retrieved_chunk_count": mean_k,
        "relative_retrieval_depth_cost_proxy": mean_k / 5.0,
    }


def select_two_way(
    trajectories: pd.DataFrame, *, at_k: int, thresholds: Iterable[float],
    risk_constraint: float,
) -> dict[str, Any]:
    candidates = [two_way_baseline(trajectories, at_k=at_k, threshold=value)
                  for value in thresholds]
    feasible = [row for row in candidates if row["final_selective_risk"] is not None
                and row["final_selective_risk"] <= risk_constraint]
    if not feasible:
        return {"risk_constraint": risk_constraint, "feasible": False, "at_k": at_k}
    selected = max(feasible, key=lambda row: (row["final_answer_coverage"], row["threshold"]))
    return {"risk_constraint": risk_constraint, "feasible": True, "at_k": at_k, **selected}


def always_answer_k5(trajectories: pd.DataFrame) -> dict[str, Any]:
    y5 = trajectories["y5"].to_numpy(int); count = len(y5); unsafe = int((y5 == 0).sum())
    possible = int(trajectories["y10"].sum())
    return {
        "final_answer_coverage": 1.0, "final_selective_risk": unsafe / count,
        "retrieval_expansion_rate": 0.0, "false_abstention_rate": 0.0 if possible else None,
        "unsafe_answer_count": unsafe, "safe_answer_count": count - unsafe,
        "abstention_count": 0, "mean_retrieved_k": 5.0,
        "mean_retrieved_chunk_count": 5.0, "relative_retrieval_depth_cost_proxy": 1.0,
    }
