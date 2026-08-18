"""Evaluate completed genuine-human Phase 3 annotations against the separate answer key."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from answerability_rag.retrieval.artifacts import read_parquet_records
from answerability_rag.sufficiency.evaluation import evaluate_human_annotations


def _read(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, action="append", required=True)
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/results/phase03_human_validation_results.json"),
    )
    arguments = parser.parse_args()
    rows = [row for path in arguments.annotations for row in _read(ROOT / path)]
    keys = read_parquet_records(ROOT / "artifacts/results/phase03_annotation_answer_key.parquet")
    report = evaluate_human_annotations(rows, keys)
    output = ROOT / arguments.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
