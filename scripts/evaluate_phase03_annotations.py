"""Evaluate completed genuine-human Phase 3 annotations after blinding is complete."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from answerability_rag.retrieval.artifacts import read_parquet_records
from answerability_rag.sufficiency.evaluation import (
    FORBIDDEN_BLINDED_COLUMNS,
    VALID_MANUAL_LABELS,
    development_stratum_frequencies,
    evaluate_human_annotations,
)


DISAGREEMENT_FIELDS = (
    "disagreement_scope", "sample_id", "annotator_id", "second_annotator_id",
    "automatic_label", "human_label", "second_human_label", "sampling_stratum",
    "disagreement_category",
)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        exposed = FORBIDDEN_BLINDED_COLUMNS & set(reader.fieldnames or ())
        if exposed:
            raise ValueError(
                f"{path} violates annotation blinding; forbidden columns: {sorted(exposed)}"
            )
        return list(reader)


def _completed_identity(rows: list[dict[str, str]], *, require_all: bool) -> str:
    completed = [row for row in rows if str(row.get("manual_label", "")).strip()]
    if require_all and len(completed) != len(rows):
        raise ValueError(
            "the first-annotator file is incomplete; the automatic answer key remains sealed"
        )
    for row in completed:
        label = str(row.get("manual_label", "")).strip().lower()
        if label not in VALID_MANUAL_LABELS:
            raise ValueError(
                f"invalid human label {label!r}; the automatic answer key remains sealed"
            )
        if not str(row.get("annotator_id", "")).strip():
            raise ValueError(
                "a completed row lacks annotator_id; the automatic answer key remains sealed"
            )
        if not str(row.get("annotation_timestamp", "")).strip():
            raise ValueError(
                "a completed row lacks annotation_timestamp; the automatic answer key "
                "remains sealed"
            )
    annotators = {str(row["annotator_id"]).strip() for row in completed}
    if len(annotators) != 1:
        raise ValueError("each annotation file must contain exactly one genuine annotator_id")
    return next(iter(annotators))


def _write_disagreements(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DISAGREEMENT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in DISAGREEMENT_FIELDS} for row in rows
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations", type=Path, action="append", required=True,
        help="First path is the complete primary-human file; optional second path is independent.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/results/phase03_human_validation_results.json"),
    )
    parser.add_argument(
        "--disagreements-output", type=Path,
        default=Path("artifacts/results/phase03_annotation_disagreements.csv"),
    )
    arguments = parser.parse_args()
    if len(arguments.annotations) > 2:
        parser.error("at most two genuine-human annotation files are supported")

    files = [ROOT / path for path in arguments.annotations]
    file_rows = [_read(path) for path in files]
    primary_annotator = _completed_identity(file_rows[0], require_all=True)
    if len(file_rows[0]) != 150:
        raise ValueError(
            f"the frozen first-annotator sample must contain 150 rows, got {len(file_rows[0])}"
        )
    if len(file_rows) == 2:
        second_annotator = _completed_identity(file_rows[1], require_all=False)
        if second_annotator == primary_annotator:
            raise ValueError(
                "the second annotation file must contain an independent human annotator_id"
            )

    # The automatic answer key is first opened only after the primary human file is complete.
    keys = read_parquet_records(ROOT / "artifacts/results/phase03_annotation_answer_key.parquet")
    development_table = pq.read_table(
        ROOT / "artifacts/data/context_sufficiency_labels.parquet",
        columns=["split", "y_suff", "label_method", "partial_overlap", "gold_document_hit"],
        filters=[("split", "in", ["train", "validation"])],
    )
    population_counts = development_stratum_frequencies(development_table.to_pylist())
    rows = [row for group in file_rows for row in group]
    report = evaluate_human_annotations(
        rows, keys, population_counts, primary_annotator_id=primary_annotator,
    )
    disagreements = report.pop("disagreement_records")
    disagreement_output = ROOT / arguments.disagreements_output
    _write_disagreements(disagreement_output, disagreements)
    report["disagreement_records"] = {
        "path": disagreement_output.relative_to(ROOT).as_posix(),
        "row_count": len(disagreements),
    }
    output = ROOT / arguments.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))

    quality_status = report["predeclared_gate_report"]["automatic_label_quality_status"]
    if quality_status == "fail":
        raise SystemExit(2)
    if quality_status == "pending":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
