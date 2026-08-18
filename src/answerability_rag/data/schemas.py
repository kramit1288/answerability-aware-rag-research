"""Typed records shared by Phase 1 loaders and validators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TechQAContext:
    filename: str
    text: str


@dataclass(frozen=True)
class TechQAQuestion:
    question_id: str
    question: str
    answer: str
    is_impossible: bool
    contexts: tuple[TechQAContext, ...]

    @property
    def provenance_partition(self) -> str:
        return self.question_id.split("_Q", 1)[0]

    @property
    def gold_filenames(self) -> tuple[str, ...]:
        return tuple(context.filename for context in self.contexts)


@dataclass(frozen=True)
class RAGTruthSource:
    source_id: str
    task_type: str
    source_name: str
    source_info: Any
    raw: dict[str, Any]


@dataclass(frozen=True)
class RAGTruthResponse:
    response_id: str
    source_id: str
    model: str
    temperature: int | float
    labels: tuple[dict[str, Any], ...]
    official_split: str
    quality: str | None
    response: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class ValidationCheck:
    check_id: str
    description: str
    expected: Any
    observed: Any
    status: str
    severity: str = "P0"
    details: Any = None


@dataclass
class ValidationReport:
    checks: list[ValidationCheck] = field(default_factory=list)

    def add(self, check_id: str, description: str, expected: Any, observed: Any, passed: bool,
            details: Any = None, severity: str = "P0") -> None:
        self.checks.append(ValidationCheck(
            check_id, description, expected, observed, "pass" if passed else "fail", severity, details
        ))

    @property
    def passed(self) -> bool:
        return all(check.status == "pass" for check in self.checks if check.severity == "P0")

    def require_pass(self) -> None:
        failures = [check for check in self.checks if check.severity == "P0" and check.status != "pass"]
        if failures:
            messages = "; ".join(f"{check.check_id}: {check.observed!r}" for check in failures)
            raise ValueError(f"Phase 1 integrity validation failed: {messages}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "overall_status": "pass" if self.passed else "fail",
            "checks": [check.__dict__ for check in self.checks],
        }
