"""Leakage-safe TechQA components and deterministic lexicographic MILP splitting."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from ..hashing import canonical_json, canonical_json_sha256, normalized_question_sha256, sha256_text
from .schemas import TechQAQuestion, ValidationReport


SPLIT_NAMES = ("train", "validation", "test")
COMPONENT_FIELDS = (
    "schema_version", "dataset_revision", "split_group_id", "component_size",
    "member_question_ids_json", "shared_gold_filenames_json", "normalized_question_hashes_json",
    "answerable_count", "impossible_count", "assigned_split", "split_seed",
    "split_algorithm_version", "component_sha256",
)
ASSIGNMENT_FIELDS = (
    "schema_version", "dataset_revision", "question_id", "provenance_partition",
    "question_sha256", "normalized_question_sha256", "is_impossible", "reference_status",
    "gold_doc_ids_json", "gold_evidence_analysis_status", "split_group_id", "split", "split_seed",
    "split_algorithm_version", "component_manifest_sha256", "split_frozen_at",
)


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            first, second = sorted((left_root, right_root))
            self.parent[second] = first


@dataclass(frozen=True)
class ComponentBuildResult:
    memberships: dict[str, str]
    normalized_hashes: dict[str, str]
    components: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class SplitAssignmentResult:
    component_splits: dict[str, str]
    question_splits: dict[str, str]
    diagnostics: dict[str, Any]
    semantic_sha256: str


def build_question_components(rows: Iterable[TechQAQuestion]) -> ComponentBuildResult:
    ordered = sorted(rows, key=lambda row: row.question_id)
    if len({row.question_id for row in ordered}) != len(ordered):
        raise ValueError("question IDs must be unique before component construction")
    union = _UnionFind(row.question_id for row in ordered)
    normalized = {row.question_id: normalized_question_sha256(row.question) for row in ordered}
    by_filename: dict[str, list[str]] = defaultdict(list)
    by_question: dict[str, list[str]] = defaultdict(list)
    for row in ordered:
        for filename in set(row.gold_filenames):
            by_filename[filename].append(row.question_id)
        by_question[normalized[row.question_id]].append(row.question_id)
    for groups in (by_filename.values(), by_question.values()):
        for members in groups:
            anchor = min(members)
            for member in sorted(members):
                union.union(anchor, member)
    grouped: dict[str, list[TechQAQuestion]] = defaultdict(list)
    for row in ordered:
        grouped[union.find(row.question_id)].append(row)
    memberships: dict[str, str] = {}
    components: list[dict[str, Any]] = []
    for members in grouped.values():
        member_ids = sorted(row.question_id for row in members)
        hashes = sorted({normalized[row.question_id] for row in members})
        filenames = sorted({name for row in members for name in row.gold_filenames})
        signature = {"normalized_question_hashes": hashes, "gold_filenames": filenames}
        signature_json = canonical_json(signature)
        group_id = "techqa-group:" + sha256_text(signature_json)
        for member in member_ids:
            memberships[member] = group_id
        components.append({
            "split_group_id": group_id,
            "component_size": len(members),
            "member_question_ids": member_ids,
            "shared_gold_filenames": filenames,
            "normalized_question_hashes": hashes,
            "answerable_count": sum(not row.is_impossible for row in members),
            "impossible_count": sum(row.is_impossible for row in members),
            "component_signature": signature_json,
        })
    components.sort(key=lambda row: row["split_group_id"])
    if len(memberships) != len(ordered):
        raise AssertionError("component construction did not assign every question exactly once")
    return ComponentBuildResult(memberships, normalized, tuple(components))


def _tie_coefficient(seed: int, signature: str, split: str) -> float:
    digest = hashlib.sha256(f"{seed}\0{signature}\0{split}".encode("utf-8")).digest()
    return (int.from_bytes(digest[:8], "big") + 1) / (2**64 + 1)


def assign_grouped_splits(rows: Iterable[TechQAQuestion], component_result: ComponentBuildResult,
                          ratios: dict[str, float] | None = None, seed: int = 42,
                          algorithm_version: str = "connected-component-lexicographic-milp-v1",
                          tie_rule: str = "sha256(seed,component_signature,split)-weighted-linear-objective-v1",
                          time_limit_seconds: float = 300.0) -> SplitAssignmentResult:
    """Prove an optimal two-stage balance, then apply a stable neutral tie objective."""
    try:
        import numpy as np
        import scipy
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix, vstack
    except ImportError as error:
        raise RuntimeError("SciPy is required for the Phase 1 optimal split proof") from error

    ratios = ratios or {"train": 0.70, "validation": 0.15, "test": 0.15}
    if tuple(ratios) != SPLIT_NAMES:
        raise ValueError(f"split ratios must be ordered as {SPLIT_NAMES}")
    components = list(component_result.components)
    # Components with the same (total, answerable, impossible) vector are exchangeable for the
    # two balance objectives. Aggregating them removes hundreds of symmetric binary variables
    # while preserving the exact feasible count/class region and therefore the optimality proof.
    bucket_members: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for component in components:
        key = (int(component["component_size"]), int(component["answerable_count"]),
               int(component["impossible_count"]))
        bucket_members[key].append(component)
    optimization_units = [
        {"key": key, "count": len(bucket_members[key]), "component_size": key[0],
         "answerable_count": key[1], "impossible_count": key[2],
         "component_signature": canonical_json({"component_vector": key})}
        for key in sorted(bucket_members)
    ]
    questions = list(rows)
    totals = {
        "total": len(questions),
        "answerable": sum(not row.is_impossible for row in questions),
        "impossible": sum(row.is_impossible for row in questions),
    }
    metric_keys = tuple((split, metric) for split in SPLIT_NAMES for metric in ("total", "answerable", "impossible"))
    targets = {(split, metric): ratios[split] * totals[metric] for split, metric in metric_keys}
    n_units, n_splits, n_metrics = len(optimization_units), len(SPLIT_NAMES), len(metric_keys)
    x_count, d_start, t_index = n_units * n_splits, n_units * n_splits, n_units * n_splits + n_metrics
    variable_count = t_index + 1
    rows_a: list[Any] = []
    lower: list[float] = []
    upper: list[float] = []

    def component_value(component: dict[str, Any], metric: str) -> int:
        return int(component["component_size"] if metric == "total" else component[f"{metric}_count"])

    for component_index, unit in enumerate(optimization_units):
        vector = lil_matrix((1, variable_count), dtype=float)
        for split_index in range(n_splits):
            vector[0, component_index * n_splits + split_index] = 1.0
        rows_a.append(vector); lower.append(float(unit["count"])); upper.append(float(unit["count"]))

    for metric_index, (split, metric) in enumerate(metric_keys):
        split_index = SPLIT_NAMES.index(split)
        target = targets[(split, metric)]
        actual = lil_matrix((1, variable_count), dtype=float)
        for component_index, component in enumerate(optimization_units):
            actual[0, component_index * n_splits + split_index] = component_value(component, metric)
        deviation_index = d_start + metric_index
        positive = actual.copy(); positive[0, deviation_index] = -1.0
        negative = actual.copy(); negative[0, deviation_index] = 1.0
        normalized = lil_matrix((1, variable_count), dtype=float)
        normalized[0, deviation_index] = 1.0
        normalized[0, t_index] = -target
        rows_a.extend((positive, negative, normalized))
        lower.extend((-np.inf, target, -np.inf))
        upper.extend((target, np.inf, 0.0))
        # Every final split must contain both global classes.
        if metric in {"answerable", "impossible"}:
            rows_a.append(actual); lower.append(1.0); upper.append(np.inf)

    matrix = vstack(rows_a, format="csr")
    base_constraint = LinearConstraint(matrix, np.asarray(lower), np.asarray(upper))
    lb = np.zeros(variable_count)
    ub = np.full(variable_count, np.inf)
    for component_index, unit in enumerate(optimization_units):
        for split_index in range(n_splits):
            ub[component_index * n_splits + split_index] = float(unit["count"])
    integrality = np.zeros(variable_count, dtype=int)
    integrality[:x_count] = 1
    options = {"time_limit": float(time_limit_seconds), "presolve": True}

    first_objective = np.zeros(variable_count); first_objective[t_index] = 1.0
    first = milp(first_objective, integrality=integrality, bounds=Bounds(lb, ub),
                 constraints=base_constraint, options=options)
    if first.status != 0 or first.x is None:
        raise RuntimeError(f"Phase 1 primary split MILP was not optimal: status={first.status}, {first.message}")
    primary_optimum = float(first.fun)

    ub_second = ub.copy(); ub_second[t_index] = primary_optimum + 1e-9
    second_objective = np.zeros(variable_count)
    for metric_index, key in enumerate(metric_keys):
        second_objective[d_start + metric_index] = 1.0 / targets[key]
    second = milp(second_objective, integrality=integrality, bounds=Bounds(lb, ub_second),
                  constraints=base_constraint, options=options)
    if second.status != 0 or second.x is None:
        raise RuntimeError(f"Phase 1 secondary split MILP was not optimal: status={second.status}, {second.message}")
    secondary_optimum = float(second.fun)

    secondary_row = lil_matrix((1, variable_count), dtype=float)
    secondary_row[0, :] = second_objective
    final_matrix = vstack((matrix, secondary_row), format="csr")
    final_constraint = LinearConstraint(
        final_matrix, np.append(np.asarray(lower), -np.inf),
        np.append(np.asarray(upper), secondary_optimum + 1e-8),
    )
    tie_objective = np.zeros(variable_count)
    for component_index, component in enumerate(optimization_units):
        for split_index, split in enumerate(SPLIT_NAMES):
            tie_objective[component_index * n_splits + split_index] = _tie_coefficient(
                seed, str(component["component_signature"]), split
            )
    third = milp(tie_objective, integrality=integrality, bounds=Bounds(lb, ub_second),
                 constraints=final_constraint, options=options)
    if third.status != 0 or third.x is None:
        raise RuntimeError(f"Phase 1 tie-break split MILP was not optimal: status={third.status}, {third.message}")

    component_splits: dict[str, str] = {}
    aggregate_allocations: dict[str, dict[str, int]] = {}
    for index, unit in enumerate(optimization_units):
        values = third.x[index * n_splits:(index + 1) * n_splits]
        allocations = [int(round(value)) for value in values]
        if any(abs(value - rounded) > 1e-6 for value, rounded in zip(values, allocations)):
            raise RuntimeError("MILP returned a non-integral aggregate component assignment")
        key = tuple(unit["key"])
        if sum(allocations) != int(unit["count"]):
            raise RuntimeError("MILP aggregate allocation does not preserve component count")
        aggregate_allocations[str(key)] = dict(zip(SPLIT_NAMES, allocations))
        ordered_members = sorted(
            bucket_members[key],
            key=lambda component: hashlib.sha256(
                f"{seed}\0{component['component_signature']}".encode("utf-8")
            ).hexdigest(),
        )
        split_order = sorted(
            range(n_splits),
            key=lambda split_index: hashlib.sha256(
                f"{seed}\0{key}\0{SPLIT_NAMES[split_index]}".encode("utf-8")
            ).hexdigest(),
        )
        cursor = 0
        for split_index in split_order:
            count = allocations[split_index]
            for component in ordered_members[cursor:cursor + count]:
                component_splits[str(component["split_group_id"])] = SPLIT_NAMES[split_index]
            cursor += count
    question_splits = {
        question_id: component_splits[group_id]
        for question_id, group_id in component_result.memberships.items()
    }
    achieved: dict[str, dict[str, int]] = {}
    normalized_deviations: list[float] = []
    for split in SPLIT_NAMES:
        selected_components = [component for component in components if component_splits[str(component["split_group_id"])] == split]
        achieved[split] = {
            "total": sum(int(component["component_size"]) for component in selected_components),
            "answerable": sum(int(component["answerable_count"]) for component in selected_components),
            "impossible": sum(int(component["impossible_count"]) for component in selected_components),
            "components": len(selected_components),
        }
        for metric in ("total", "answerable", "impossible"):
            normalized_deviations.append(abs(achieved[split][metric] - targets[(split, metric)]) / targets[(split, metric)])
    semantic_rows = sorted(
        ({"question_id": question_id, "split_group_id": component_result.memberships[question_id], "split": split}
         for question_id, split in question_splits.items()), key=lambda row: row["question_id"]
    )
    diagnostics = {
        "solver": "scipy.optimize.milp/HiGHS", "solver_version": scipy.__version__,
        "solver_status": "optimal", "seed": seed, "algorithm_version": algorithm_version,
        "tie_rule": tie_rule, "primary_objective_max_normalized_deviation": max(normalized_deviations),
        "secondary_objective_sum_normalized_deviation": sum(normalized_deviations),
        "solver_primary_objective": primary_optimum, "solver_secondary_objective": secondary_optimum,
        "tie_objective": float(third.fun), "aggregate_component_vectors": len(optimization_units),
        "aggregate_allocations": aggregate_allocations, "targets": {
            split: {metric: targets[(split, metric)] for metric in ("total", "answerable", "impossible")}
            for split in SPLIT_NAMES
        }, "achieved": achieved,
    }
    return SplitAssignmentResult(
        component_splits, question_splits, diagnostics, canonical_json_sha256(semantic_rows)
    )


def component_manifest_rows(component_result: ComponentBuildResult, split_result: SplitAssignmentResult,
                            *, schema_version: str, dataset_revision: str, seed: int,
                            algorithm_version: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for component in component_result.components:
        group_id = str(component["split_group_id"])
        core = {
            "split_group_id": group_id, "component_size": int(component["component_size"]),
            "member_question_ids": component["member_question_ids"],
            "shared_gold_filenames": component["shared_gold_filenames"],
            "normalized_question_hashes": component["normalized_question_hashes"],
            "answerable_count": int(component["answerable_count"]),
            "impossible_count": int(component["impossible_count"]),
            "assigned_split": split_result.component_splits[group_id],
        }
        rows.append({
            "schema_version": schema_version, "dataset_revision": dataset_revision,
            "split_group_id": group_id, "component_size": core["component_size"],
            "member_question_ids_json": canonical_json(core["member_question_ids"]),
            "shared_gold_filenames_json": canonical_json(core["shared_gold_filenames"]),
            "normalized_question_hashes_json": canonical_json(core["normalized_question_hashes"]),
            "answerable_count": core["answerable_count"], "impossible_count": core["impossible_count"],
            "assigned_split": core["assigned_split"], "split_seed": seed,
            "split_algorithm_version": algorithm_version, "component_sha256": canonical_json_sha256(core),
        })
    return sorted(rows, key=lambda row: str(row["split_group_id"]))


def assignment_manifest_rows(rows: Iterable[TechQAQuestion], component_result: ComponentBuildResult,
                             split_result: SplitAssignmentResult, filename_to_doc_id: dict[str, str],
                             *, schema_version: str, dataset_revision: str, seed: int,
                             algorithm_version: str, component_manifest_sha256: str,
                             split_frozen_at: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: item.question_id):
        if row.is_impossible:
            reference_status, evidence_status = "not_applicable_impossible", "not_applicable_impossible"
        elif row.question_id in {"DEV_Q014", "DEV_Q094"}:
            reference_status, evidence_status = "empty_answerable", "pending_manual_alignment"
        else:
            reference_status, evidence_status = "available", "confirmed_attached_context"
        output.append({
            "schema_version": schema_version, "dataset_revision": dataset_revision,
            "question_id": row.question_id, "provenance_partition": row.provenance_partition,
            "question_sha256": sha256_text(row.question),
            "normalized_question_sha256": component_result.normalized_hashes[row.question_id],
            "is_impossible": str(row.is_impossible).lower(), "reference_status": reference_status,
            "gold_doc_ids_json": canonical_json([filename_to_doc_id[name] for name in row.gold_filenames]),
            "gold_evidence_analysis_status": evidence_status,
            "split_group_id": component_result.memberships[row.question_id],
            "split": split_result.question_splits[row.question_id], "split_seed": seed,
            "split_algorithm_version": algorithm_version,
            "component_manifest_sha256": component_manifest_sha256, "split_frozen_at": split_frozen_at,
        })
    return output


def validate_split_assignments(rows: Iterable[TechQAQuestion], component_result: ComponentBuildResult,
                               split_result: SplitAssignmentResult) -> ValidationReport:
    questions = list(rows)
    report = ValidationReport()
    ids = [row.question_id for row in questions]
    report.add("split_unique_question_ids", "No question ID appears twice", len(ids), len(set(ids)), len(ids) == len(set(ids)))
    assigned_ids = set(split_result.question_splits)
    coverage_differences = {
        "missing": sorted(set(ids) - assigned_ids), "unexpected": sorted(assigned_ids - set(ids)),
    }
    report.add("split_complete_question_coverage", "All eligible TechQA rows are assigned",
               {"assigned": len(ids), "missing": [], "unexpected": []},
               {"assigned": len(assigned_ids), **coverage_differences},
               assigned_ids == set(ids))

    def crossing(key_by_question: dict[str, Iterable[str]]) -> list[str]:
        splits_by_key: dict[str, set[str]] = defaultdict(set)
        for question_id, keys in key_by_question.items():
            for key in keys:
                splits_by_key[key].add(split_result.question_splits[question_id])
        return sorted(key for key, splits in splits_by_key.items() if len(splits) > 1)

    component_crossings = crossing({qid: (gid,) for qid, gid in component_result.memberships.items()})
    filename_crossings = crossing({row.question_id: row.gold_filenames for row in questions})
    duplicate_crossings = crossing({qid: (value,) for qid, value in component_result.normalized_hashes.items()})
    report.add("split_component_leakage", "No component crosses splits", 0, len(component_crossings), not component_crossings, component_crossings)
    report.add("split_gold_filename_leakage", "No gold/source filename crosses splits", 0, len(filename_crossings), not filename_crossings, filename_crossings)
    report.add("split_duplicate_question_leakage", "No normalized duplicate crosses splits", 0, len(duplicate_crossings), not duplicate_crossings, duplicate_crossings)
    component_total = sum(int(component["component_size"]) for component in component_result.components)
    report.add("split_component_question_sum", "Component sizes sum to all questions", len(ids), component_total, component_total == len(ids))
    answerable = sum(not row.is_impossible for row in questions)
    impossible = sum(row.is_impossible for row in questions)
    achieved = split_result.diagnostics["achieved"]
    report.add("split_class_count_sum", "Split class counts sum correctly",
               {"answerable": answerable, "impossible": impossible},
               {"answerable": sum(value["answerable"] for value in achieved.values()),
                "impossible": sum(value["impossible"] for value in achieved.values())},
               sum(value["answerable"] for value in achieved.values()) == answerable and
               sum(value["impossible"] for value in achieved.values()) == impossible)
    report.add("split_component_count_sum", "Split component counts sum correctly", len(component_result.components),
               sum(value["components"] for value in achieved.values()),
               sum(value["components"] for value in achieved.values()) == len(component_result.components))
    report.add("split_solver_optimal", "Lexicographic optimizer proved optimal", "optimal",
               split_result.diagnostics["solver_status"], split_result.diagnostics["solver_status"] == "optimal")
    return report
