"""Prepare and freeze Phase 1 data artifacts from immutable pinned releases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from answerability_rag.config import load_phase01_config  # noqa: E402
from answerability_rag.data.prepare import prepare_phase01_data  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs/phase01_data.json")
    args = parser.parse_args()
    metadata = prepare_phase01_data(load_phase01_config(args.config))
    summary = {
        "status": "pass", "techqa_rows": metadata["techqa"]["observed_rows"],
        "techqa_corpus_documents": metadata["techqa"]["corpus"]["document_count"],
        "techqa_components": metadata["techqa"]["components"]["count"],
        "split_counts": metadata["techqa"]["split"]["achieved"],
        "split_sha256": metadata["techqa"]["split"]["split_semantic_sha256"],
        "ragtruth_sources": metadata["ragtruth"]["observed_sources"],
        "ragtruth_responses": metadata["ragtruth"]["observed_responses"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
