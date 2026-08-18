"""Validate Phase 3 labels without emitting aggregate test label performance."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from answerability_rag.retrieval.artifacts import read_parquet_records
from answerability_rag.sufficiency.config import Phase03Config
from answerability_rag.sufficiency.integrity import validate_phase03
from answerability_rag.sufficiency.prerequisites import verify_frozen_inputs


def _csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/phase03_sufficiency.json"))
    arguments = parser.parse_args()
    config = Phase03Config.load(ROOT / arguments.config)
    frozen = verify_frozen_inputs(ROOT, config)
    labels = read_parquet_records(ROOT / "artifacts/data/context_sufficiency_labels.parquet")
    sample = _csv(ROOT / "artifacts/results/phase03_manual_sample_manifest.csv")
    for row in sample:
        row["k"] = int(row["k"])
        row["blind_order"] = int(row["blind_order"])
    report = validate_phase03(
        labels, frozen.conditions,
        _csv(ROOT / "artifacts/results/phase03_development_label_summary.csv"), sample,
        read_parquet_records(ROOT / "artifacts/results/phase03_annotation_blinded.parquet"),
        read_parquet_records(ROOT / "artifacts/results/phase03_annotation_answer_key.parquet"),
        json.loads((ROOT / "artifacts/data/phase03_column_governance.json").read_text()),
        expected_sample_size=int(config.values["annotation"]["target_size"]),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["overall_status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
