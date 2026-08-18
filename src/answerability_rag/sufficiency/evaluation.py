"""Frozen human validation analysis for Phase 3 automatic sufficiency labels."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable

from sklearn.metrics import cohen_kappa_score

from answerability_rag.sufficiency.annotation import sampling_stratum


VALID_MANUAL_LABELS = frozenset({"sufficient", "insufficient", "ambiguous"})
SAMPLING_STRATA = (
    "automatic_positive",
    "partial_overlap",
    "correct_document_insufficient",
    "wrong_document_retrieval",
    "benchmark_impossible",
)
FORBIDDEN_BLINDED_COLUMNS = frozenset({
    "automatic_y_suff",
    "y_suff",
    "label_status",
    "label_method",
    "gold_document_hit",
    "maximum_span_coverage_fraction",
    "partial_overlap",
    "exclusion_reason",
})
SUFFICIENT_PRECISION_GATE = 0.90
PREVALENCE_WEIGHTED_F1_GATE = 0.85
BENCHMARK_IMPOSSIBLE_CONTAMINATION_GATE = 0.10
DESIRABLE_COHENS_KAPPA = 0.80
TARGET_SECOND_ANNOTATOR_OVERLAP = 100


def _wilson(
    successes: int, total: int, z: float = 1.959963984540054,
) -> tuple[float | None, float | None]:
    if total == 0:
        return None, None
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def development_stratum_frequencies(label_rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Count the exclusive sampling strata in resolved TRAIN+VALIDATION conditions only."""
    counts: Counter[str] = Counter()
    for row in label_rows:
        if row["split"] not in {"train", "validation"} or row["y_suff"] is None:
            continue
        stratum = sampling_stratum(row)
        if stratum not in SAMPLING_STRATA:
            raise ValueError(f"unexpected Phase 3 sampling stratum: {stratum}")
        counts[stratum] += 1
    missing = set(SAMPLING_STRATA) - set(counts)
    if missing:
        raise ValueError(f"development population has empty sampling strata: {sorted(missing)}")
    return {stratum: counts[stratum] for stratum in SAMPLING_STRATA}


def _confusion(
    rows: Iterable[dict[str, Any]], key: dict[str, dict[str, Any]],
) -> dict[str, int]:
    counts = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    for row in rows:
        automatic = int(key[row["sample_id"]]["automatic_y_suff"])
        human = 1 if row["manual_label"] == "sufficient" else 0
        cell = (
            "tp" if automatic == 1 and human == 1
            else "tn" if automatic == 0 and human == 0
            else "fp" if automatic == 1
            else "fn"
        )
        counts[cell] += 1
    return counts


def _metrics(confusion: dict[str, float]) -> dict[str, float | None]:
    tn, fp, fn, tp = (confusion[name] for name in ("tn", "fp", "fn", "tp"))
    total = tn + fp + fn + tp
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else 0.0 if precision == 0 or recall == 0
        else None
    )
    return {
        "accuracy": (tp + tn) / total if total else None,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _prepare_completed(
    annotation_rows: list[dict[str, Any]], key: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    completed: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in annotation_rows:
        exposed = FORBIDDEN_BLINDED_COLUMNS & set(row)
        if exposed:
            raise ValueError(
                "human annotation input violates blinding; forbidden columns present: "
                f"{sorted(exposed)}"
            )
        label = str(row.get("manual_label", "")).strip().lower()
        if not label:
            continue
        if label not in VALID_MANUAL_LABELS:
            raise ValueError(f"invalid human label {label!r}")
        annotator = str(row.get("annotator_id", "")).strip()
        if not annotator:
            raise ValueError("completed human annotation lacks annotator_id")
        timestamp = str(row.get("annotation_timestamp", "")).strip()
        if not timestamp:
            raise ValueError("completed human annotation lacks annotation_timestamp")
        sample_id = str(row.get("sample_id", "")).strip()
        if sample_id not in key:
            raise ValueError(f"annotation sample is absent from answer key: {sample_id}")
        identity = (sample_id, annotator)
        if identity in seen:
            raise ValueError(f"duplicate annotation for sample/annotator: {identity}")
        seen.add(identity)
        completed.append({
            **row,
            "sample_id": sample_id,
            "manual_label": label,
            "annotator_id": annotator,
            "annotation_timestamp": timestamp,
        })
    return completed


def _annotator_result(
    annotator: str,
    rows: list[dict[str, Any]],
    key: dict[str, dict[str, Any]],
    population_counts: dict[str, int],
) -> dict[str, Any]:
    population_total = sum(population_counts.values())
    stratum_results: list[dict[str, Any]] = []
    weighted_confusion = {name: 0.0 for name in ("tn", "fp", "fn", "tp")}
    weighted_ambiguous_rate = 0.0
    complete_for_weighting = True

    for stratum in SAMPLING_STRATA:
        stratum_rows = [row for row in rows if key[row["sample_id"]]["sampling_stratum"] == stratum]
        eligible = [row for row in stratum_rows if row["manual_label"] != "ambiguous"]
        ambiguous_count = len(stratum_rows) - len(eligible)
        confusion = _confusion(eligible, key)
        metrics = _metrics(confusion)
        low, high = _wilson(confusion["tp"] + confusion["tn"], len(eligible))
        population_count = population_counts[stratum]
        prevalence = population_count / population_total
        if not stratum_rows or not eligible:
            complete_for_weighting = False
        else:
            for cell in weighted_confusion:
                weighted_confusion[cell] += prevalence * confusion[cell] / len(eligible)
            weighted_ambiguous_rate += prevalence * ambiguous_count / len(stratum_rows)
        stratum_results.append({
            "sampling_stratum": stratum,
            "development_population_count": population_count,
            "development_population_prevalence": prevalence,
            "completed_count": len(stratum_rows),
            "ambiguous_count": ambiguous_count,
            "ambiguous_rate": ambiguous_count / len(stratum_rows) if stratum_rows else None,
            "eligible_non_ambiguous_count": len(eligible),
            **metrics,
            "accuracy_ci_low_95": low,
            "accuracy_ci_high_95": high,
            "confusion_matrix": confusion,
        })

    eligible = [row for row in rows if row["manual_label"] != "ambiguous"]
    sample_confusion = _confusion(eligible, key)
    sample_metrics = _metrics(sample_confusion)
    sample_low, sample_high = _wilson(
        sample_confusion["tp"] + sample_confusion["tn"], len(eligible)
    )
    weighted = _metrics(weighted_confusion) if complete_for_weighting else {
        "accuracy": None, "precision": None, "recall": None, "f1": None,
    }
    automatic_positive_rows = [
        row for row in eligible if int(key[row["sample_id"]]["automatic_y_suff"]) == 1
    ]
    sufficient_true = sum(row["manual_label"] == "sufficient" for row in automatic_positive_rows)
    precision_low, precision_high = _wilson(sufficient_true, len(automatic_positive_rows))
    impossible_rows = [
        row for row in eligible
        if key[row["sample_id"]]["sampling_stratum"] == "benchmark_impossible"
    ]
    impossible_sufficient = sum(row["manual_label"] == "sufficient" for row in impossible_rows)

    return {
        "annotator_id": annotator,
        "completed_count": len(rows),
        "ambiguous_count": len(rows) - len(eligible),
        "ambiguous_rate": (len(rows) - len(eligible)) / len(rows) if rows else None,
        "eligible_non_ambiguous_count": len(eligible),
        "metrics_by_sampling_stratum": stratum_results,
        "sample_unweighted_metrics": {
            **sample_metrics,
            "accuracy_ci_low_95": sample_low,
            "accuracy_ci_high_95": sample_high,
            "confusion_matrix": sample_confusion,
            "interpretation": (
                "descriptive for the deliberately equal-allocation audit sample only; "
                "not an estimate of natural TRAIN+VALIDATION population performance"
            ),
        },
        "prevalence_weighted_metrics": {
            "status": "available" if complete_for_weighting else "insufficient_stratum_coverage",
            "method": "stratum-prevalence-weighted confusion-rate aggregation",
            "development_population_total": population_total,
            "estimated_confusion_proportions": (
                weighted_confusion if complete_for_weighting else None
            ),
            **weighted,
            "ambiguous_rate": weighted_ambiguous_rate if complete_for_weighting else None,
        },
        "automatic_sufficient_precision": (
            sufficient_true / len(automatic_positive_rows) if automatic_positive_rows else None
        ),
        "automatic_sufficient_precision_ci_low_95": precision_low,
        "automatic_sufficient_precision_ci_high_95": precision_high,
        "benchmark_impossible_audit": {
            "eligible_non_ambiguous_count": len(impossible_rows),
            "human_context_sufficient_count": impossible_sufficient,
            "human_context_sufficient_rate": (
                impossible_sufficient / len(impossible_rows) if impossible_rows else None
            ),
        },
    }


def _inter_annotator_agreement(
    by_annotator: dict[str, list[dict[str, Any]]], primary_annotator_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    others = sorted(set(by_annotator) - {primary_annotator_id})
    if not others:
        return ({
            "status": "pending_second_human_annotator",
            "annotators": [primary_annotator_id],
            "double_annotated": 0,
            "target_double_annotated": TARGET_SECOND_ANNOTATOR_OVERLAP,
            "overlap_limitation": "second genuine human annotation is unavailable",
            "raw_agreement": None,
            "cohens_kappa": None,
        }, [])
    if len(others) > 1:
        raise ValueError("Phase 3 validation supports one primary and one second human annotator")
    second = others[0]
    first_map = {
        row["sample_id"]: row["manual_label"]
        for row in by_annotator[primary_annotator_id]
    }
    second_map = {row["sample_id"]: row["manual_label"] for row in by_annotator[second]}
    common = sorted(first_map.keys() & second_map.keys())
    first_labels = [first_map[item] for item in common]
    second_labels = [second_map[item] for item in common]
    exact = sum(left == right for left, right in zip(first_labels, second_labels))
    kappa = float(cohen_kappa_score(first_labels, second_labels)) if common else None
    if kappa is not None and math.isnan(kappa):
        kappa = None
    disagreements = [{
        "disagreement_scope": "human_human",
        "sample_id": sample_id,
        "annotator_id": primary_annotator_id,
        "second_annotator_id": second,
        "automatic_label": "",
        "human_label": first_map[sample_id],
        "second_human_label": second_map[sample_id],
        "sampling_stratum": "",
        "disagreement_category": "human_label_disagreement",
    } for sample_id in common if first_map[sample_id] != second_map[sample_id]]
    return ({
        "status": "available",
        "annotators": [primary_annotator_id, second],
        "double_annotated": len(common),
        "target_double_annotated": TARGET_SECOND_ANNOTATOR_OVERLAP,
        "overlap_limitation": (
            None if len(common) >= TARGET_SECOND_ANNOTATOR_OVERLAP
            else f"overlap is {len(common)} rather than the target of approximately 100"
        ),
        "raw_agreement": exact / len(common) if common else None,
        "cohens_kappa": kappa,
    }, disagreements)


def _gate_report(primary: dict[str, Any], agreement: dict[str, Any]) -> dict[str, Any]:
    precision = primary["automatic_sufficient_precision"]
    weighted_f1 = primary["prevalence_weighted_metrics"]["f1"]
    impossible_rate = primary["benchmark_impossible_audit"]["human_context_sufficient_rate"]
    kappa = agreement["cohens_kappa"]

    precision_status = "pending" if precision is None else (
        "pass" if precision >= SUFFICIENT_PRECISION_GATE else "fail"
    )
    f1_status = "pending" if weighted_f1 is None else (
        "pass" if weighted_f1 >= PREVALENCE_WEIGHTED_F1_GATE else "fail"
    )
    if impossible_rate is None:
        impossible_status = "pending"
        impossible_action = "pending_human_audit"
    elif impossible_rate > BENCHMARK_IMPOSSIBLE_CONTAMINATION_GATE:
        impossible_status = "exclusion_required"
        impossible_action = (
            "exclude benchmark-impossible negatives from PRIMARY Phase 4 classifier training; "
            "retain only for separately reported sensitivity analysis"
        )
    else:
        impossible_status = "within_predeclared_gate"
        impossible_action = (
            "benchmark-impossible negatives may remain in primary training, with the separately "
            "reported exclusion sensitivity analysis still required"
        )
    kappa_status = "pending_second_human_annotator" if kappa is None else (
        "desirable_gate_met" if kappa >= DESIRABLE_COHENS_KAPPA
        else "disagreement_analysis_and_adjudication_required"
    )
    core = (precision_status, f1_status)
    overall = "fail" if "fail" in core else "pending" if "pending" in core else "pass"
    return {
        "purpose": "internal research-validity gates; not universal benchmark thresholds",
        "automatic_label_quality_status": overall,
        "automatic_sufficient_precision": {
            "target_minimum": SUFFICIENT_PRECISION_GATE,
            "observed": precision,
            "status": precision_status,
        },
        "prevalence_weighted_f1": {
            "target_minimum": PREVALENCE_WEIGHTED_F1_GATE,
            "observed": weighted_f1,
            "status": f1_status,
        },
        "benchmark_impossible_human_sufficient_rate": {
            "exclusion_trigger_strictly_greater_than": BENCHMARK_IMPOSSIBLE_CONTAMINATION_GATE,
            "observed": impossible_rate,
            "status": impossible_status,
            "required_phase4_action": impossible_action,
        },
        "human_human_cohens_kappa": {
            "desirable_minimum": DESIRABLE_COHENS_KAPPA,
            "observed": kappa,
            "status": kappa_status,
        },
    }


def evaluate_human_annotations(
    annotation_rows: list[dict[str, Any]],
    answer_key_rows: list[dict[str, Any]],
    development_population_counts: dict[str, int] | None = None,
    *,
    primary_annotator_id: str | None = None,
) -> dict[str, Any]:
    """Evaluate genuine human files after blinding, using frozen stratified-sample rules."""
    sample_ids = [str(row["sample_id"]) for row in answer_key_rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("answer key contains duplicate sample_id values")
    key = {str(row["sample_id"]): row for row in answer_key_rows}
    missing_strata = [
        sample_id for sample_id, row in key.items() if not row.get("sampling_stratum")
    ]
    if missing_strata:
        raise ValueError("answer key lacks sampling_stratum")
    completed = _prepare_completed(annotation_rows, key)
    by_annotator: dict[str, list[dict[str, Any]]] = {}
    for row in completed:
        by_annotator.setdefault(row["annotator_id"], []).append(row)
    if not by_annotator:
        raise ValueError("no completed genuine-human annotations were supplied")
    if primary_annotator_id is None:
        primary_annotator_id = sorted(by_annotator)[0]
    if primary_annotator_id not in by_annotator:
        raise ValueError(f"primary annotator is absent: {primary_annotator_id}")
    if development_population_counts is None:
        raise ValueError("TRAIN+VALIDATION development stratum frequencies are required")
    if set(development_population_counts) != set(SAMPLING_STRATA):
        raise ValueError("development stratum frequencies do not match the frozen five strata")
    if any(int(value) <= 0 for value in development_population_counts.values()):
        raise ValueError("development stratum frequencies must be positive")

    annotator_results = [
        _annotator_result(annotator, rows, key, development_population_counts)
        for annotator, rows in sorted(by_annotator.items())
    ]
    primary = next(
        result for result in annotator_results if result["annotator_id"] == primary_annotator_id
    )
    agreement, human_disagreements = _inter_annotator_agreement(
        by_annotator, primary_annotator_id
    )
    automatic_disagreements: list[dict[str, Any]] = []
    for row in by_annotator[primary_annotator_id]:
        if row["manual_label"] == "ambiguous":
            category = "automatic_vs_human_ambiguous"
        else:
            automatic = int(key[row["sample_id"]]["automatic_y_suff"])
            human = 1 if row["manual_label"] == "sufficient" else 0
            if automatic == human:
                continue
            category = (
                "automatic_sufficient_human_insufficient" if automatic == 1
                else "automatic_insufficient_human_sufficient"
            )
        automatic_disagreements.append({
            "disagreement_scope": "automatic_human",
            "sample_id": row["sample_id"],
            "annotator_id": primary_annotator_id,
            "second_annotator_id": "",
            "automatic_label": (
                "sufficient" if int(key[row["sample_id"]]["automatic_y_suff"]) == 1
                else "insufficient"
            ),
            "human_label": row["manual_label"],
            "second_human_label": "",
            "sampling_stratum": key[row["sample_id"]]["sampling_stratum"],
            "disagreement_category": category,
        })
    disagreement_records = sorted(
        automatic_disagreements + human_disagreements,
        key=lambda row: (row["disagreement_scope"], row["sample_id"], row["annotator_id"]),
    )
    return {
        "schema_version": "phase03-human-validation-v2",
        "primary_annotator_id": primary_annotator_id,
        "completed_human_annotations": len(completed),
        "development_population_stratum_counts": {
            stratum: int(development_population_counts[stratum]) for stratum in SAMPLING_STRATA
        },
        "stratified_sample_interpretation": (
            "The 150-example equal-allocation sample is not self-weighting. Its unweighted overall "
            "accuracy is descriptive only and is not a natural development-population estimate."
        ),
        "annotator_results": annotator_results,
        "inter_annotator_agreement": agreement,
        "disagreement_category_counts": dict(Counter(
            row["disagreement_category"] for row in disagreement_records
        )),
        "disagreement_records": disagreement_records,
        "predeclared_gate_report": _gate_report(primary, agreement),
    }
