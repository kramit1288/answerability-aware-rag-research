from __future__ import annotations

from answerability_rag.data.schemas import TechQAContext, TechQAQuestion
from answerability_rag.data.splits import (
    assign_grouped_splits,
    build_question_components,
    validate_split_assignments,
)


def q(question_id: str, text: str, impossible: bool, filename: str | None = None) -> TechQAQuestion:
    contexts = () if filename is None else (TechQAContext(filename, "evidence"),)
    return TechQAQuestion(question_id, text, "-" if impossible else "answer", impossible, contexts)


def test_components_include_direct_duplicate_and_transitive_links() -> None:
    rows = [
        q("TRAIN_Q001", "How to reset?", False, "a.txt"),
        q("DEV_Q001", "Different wording", False, "a.txt"),
        q("TRAIN_Q002", "different—wording", False, "b.txt"),
        q("DEV_Q002", "Unrelated", True),
    ]
    result = build_question_components(rows)
    assert result.memberships["TRAIN_Q001"] == result.memberships["DEV_Q001"]
    assert result.memberships["DEV_Q001"] == result.memberships["TRAIN_Q002"]
    assert result.memberships["DEV_Q002"] != result.memberships["TRAIN_Q001"]
    assert sum(component["component_size"] for component in result.components) == len(rows)


def _split_fixture() -> list[TechQAQuestion]:
    rows = [q(f"TRAIN_Q{i:03d}", f"answerable {i}", False, f"doc-{i}.txt") for i in range(12)]
    rows += [q(f"DEV_Q{i:03d}", f"impossible {i}", True) for i in range(12)]
    # Two direct document pairs and a duplicate pair make non-singleton components.
    rows[1] = q("TRAIN_Q001", "answerable 1", False, "doc-0.txt")
    rows[3] = q("TRAIN_Q003", "Answerable—two", False, "doc-3.txt")
    rows[2] = q("TRAIN_Q002", "answerable two", False, "doc-2.txt")
    return rows


def test_grouped_split_is_deterministic_and_leakage_safe() -> None:
    rows = _split_fixture()
    components = build_question_components(rows)
    ratios = {"train": 0.50, "validation": 0.25, "test": 0.25}
    first = assign_grouped_splits(rows, components, ratios=ratios, seed=42, time_limit_seconds=30)
    reversed_rows = list(reversed(rows))
    second_components = build_question_components(reversed_rows)
    second = assign_grouped_splits(reversed_rows, second_components, ratios=ratios, seed=42, time_limit_seconds=30)
    assert first.question_splits == second.question_splits
    assert first.semantic_sha256 == second.semantic_sha256
    assert first.diagnostics["solver_status"] == "optimal"
    report = validate_split_assignments(rows, components, first)
    assert report.passed
    checks = {check.check_id: check for check in report.checks}
    assert checks["split_component_leakage"].observed == 0
    assert checks["split_gold_filename_leakage"].observed == 0
    assert checks["split_duplicate_question_leakage"].observed == 0


def test_component_identity_does_not_contain_provenance_partition() -> None:
    train = build_question_components([q("TRAIN_Q001", "Same question", True)])
    dev = build_question_components([q("DEV_Q999", "Same question", True)])
    assert train.components[0]["split_group_id"] == dev.components[0]["split_group_id"]
