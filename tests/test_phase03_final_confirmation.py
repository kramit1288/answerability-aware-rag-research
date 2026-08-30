"""Tests for the frozen final Phase 3 confirmation boundary and metrics."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from answerability_rag.sufficiency.final_confirmation import (
    _confusion,
    _metrics,
    validate_confirmation,
    wilson_interval,
)


ROOT = Path(__file__).resolve().parents[1]


def test_completed_confirmation_passes_without_opening_answer_key() -> None:
    result, rows = validate_confirmation(ROOT)
    assert result["status"] == "pass"
    assert result["answer_key_opened"] is False
    assert len(rows) == result["unique_questions"] == 50
    assert result["manual_label_counts"] == {"sufficient": 34, "insufficient": 16}
    assert result["test_rows"] == result["benchmark_impossible_rows"] == 0
    assert result["semantic_unevaluable_rows"] == 0


def test_binary_metrics_and_wilson_interval() -> None:
    rows = (
        [{"manual_label": "sufficient", "prediction": 1}] * 31
        + [{"manual_label": "insufficient", "prediction": 1}] * 3
        + [{"manual_label": "sufficient", "prediction": 0}] * 3
        + [{"manual_label": "insufficient", "prediction": 0}] * 13
    )
    cells = _confusion(rows)
    assert cells == {"tn": 13, "fp": 3, "fn": 3, "tp": 31}
    values = _metrics(cells)
    assert values["precision"] == pytest.approx(31 / 34)
    assert values["recall"] == pytest.approx(31 / 34)
    assert values["f1"] == pytest.approx(31 / 34)
    assert values["accuracy"] == pytest.approx(44 / 50)
    interval = wilson_interval(31, 34)
    assert interval["lower"] == pytest.approx(0.7703951343719113)
    assert interval["upper"] == pytest.approx(0.9695340563253514)


def test_human_file_has_exact_blinded_schema() -> None:
    human = ROOT / "artifacts/results/phase03_final_confirmation_annotator_1.csv"
    template = ROOT / "artifacts/results/phase03_final_confirmation_template.csv"
    with human.open(encoding="utf-8-sig", newline="") as handle:
        human_fields = csv.DictReader(handle).fieldnames
    with template.open(encoding="utf-8-sig", newline="") as handle:
        template_fields = csv.DictReader(handle).fieldnames
    assert human_fields == template_fields
    assert "y_suff_final" not in human_fields
    assert "confirmation_sampling_stratum" not in human_fields
