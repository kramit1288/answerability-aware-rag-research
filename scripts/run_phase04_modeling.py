"""Run the frozen Phase 4 modeling and policy experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from answerability_rag.answerability.pipeline import run_phase04  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase04_modeling.json")
    args = parser.parse_args()
    if Path(args.config).as_posix() != "configs/phase04_modeling.json":
        raise ValueError("Phase 4 accepts only the frozen canonical modeling config")
    print(json.dumps(run_phase04(ROOT), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
