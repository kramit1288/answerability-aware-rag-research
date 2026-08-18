"""Human-vs-automatic validation and genuine second-annotator agreement."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from sklearn.metrics import cohen_kappa_score


VALID_MANUAL_LABELS = frozenset({"sufficient", "insufficient", "ambiguous"})


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if total == 0:
        return None, None
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def evaluate_human_annotations(
    annotation_rows: list[dict[str, Any]], answer_key_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    key = {row["sample_id"]: int(row["automatic_y_suff"]) for row in answer_key_rows}
    completed = []
    for row in annotation_rows:
        label = str(row.get("manual_label", "")).strip().lower()
        if not label:
            continue
        if label not in VALID_MANUAL_LABELS:
            raise ValueError(f"invalid human label {label!r}")
        annotator = str(row.get("annotator_id", "")).strip()
        if not annotator:
            raise ValueError("completed human annotation lacks annotator_id")
        if row["sample_id"] not in key:
            raise ValueError(f"annotation sample is absent from answer key: {row['sample_id']}")
        completed.append({**row, "manual_label": label, "annotator_id": annotator})
    by_annotator: dict[str, list[dict[str, Any]]] = {}
    for row in completed:
        by_annotator.setdefault(row["annotator_id"], []).append(row)
    annotator_results = []
    for annotator, rows in sorted(by_annotator.items()):
        eligible = [row for row in rows if row["manual_label"] != "ambiguous"]
        true = [1 if row["manual_label"] == "sufficient" else 0 for row in eligible]
        automatic = [key[row["sample_id"]] for row in eligible]
        tp = sum(a == 1 and y == 1 for a, y in zip(automatic, true))
        tn = sum(a == 0 and y == 0 for a, y in zip(automatic, true))
        fp = sum(a == 1 and y == 0 for a, y in zip(automatic, true))
        fn = sum(a == 0 and y == 1 for a, y in zip(automatic, true))
        accuracy = (tp + tn) / len(eligible) if eligible else None
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 2 * precision * recall / (precision + recall) if precision and recall else (
            0.0 if precision == 0 or recall == 0 else None
        )
        low, high = _wilson(tp + tn, len(eligible))
        annotator_results.append({
            "annotator_id": annotator, "completed_count": len(rows),
            "eligible_non_ambiguous_count": len(eligible), "accuracy": accuracy,
            "accuracy_ci_low_95": low, "accuracy_ci_high_95": high,
            "precision": precision, "recall": recall, "f1": f1,
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "disagreement_categories": dict(Counter(
                "auto_positive_human_negative" if a == 1 and y == 0
                else "auto_negative_human_positive" for a, y in zip(automatic, true) if a != y
            )),
        })
    agreement: dict[str, Any]
    if len(by_annotator) >= 2:
        first, second = sorted(by_annotator)[:2]
        first_map = {row["sample_id"]: row["manual_label"] for row in by_annotator[first]}
        second_map = {row["sample_id"]: row["manual_label"] for row in by_annotator[second]}
        common = sorted(first_map.keys() & second_map.keys())
        first_labels = [first_map[item] for item in common]
        second_labels = [second_map[item] for item in common]
        exact = sum(left == right for left, right in zip(first_labels, second_labels))
        agreement = {
            "status": "available", "annotators": [first, second], "double_annotated": len(common),
            "raw_agreement": exact / len(common) if common else None,
            "cohens_kappa": float(cohen_kappa_score(first_labels, second_labels)) if common else None,
        }
    else:
        agreement = {
            "status": "pending_second_human_annotator", "annotators": sorted(by_annotator),
            "double_annotated": 0, "raw_agreement": None, "cohens_kappa": None,
        }
    return {
        "schema_version": "phase03-human-validation-v1",
        "completed_human_annotations": len(completed), "annotator_results": annotator_results,
        "inter_annotator_agreement": agreement,
    }
