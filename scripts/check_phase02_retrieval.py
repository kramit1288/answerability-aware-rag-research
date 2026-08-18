"""Validate persisted Phase 2 outputs without calculating test performance."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from answerability_rag.retrieval.artifacts import read_parquet_records
from answerability_rag.retrieval.integrity import validate_retrieval_outputs


def main() -> None:
    root = ROOT
    ranked = read_parquet_records(root / "artifacts/results/retrieval_ranked_hits.parquet")
    conditions = read_parquet_records(root / "artifacts/results/retrieval_query_level.parquet")
    with (root / "artifacts/results/phase02_metrics_train_validation.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        metrics = list(csv.DictReader(handle))
    report = validate_retrieval_outputs(ranked, conditions, metrics)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["overall_status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
