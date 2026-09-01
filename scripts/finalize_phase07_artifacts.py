"""Write the non-self-referential Phase 7 artifact manifest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from answerability_rag.final_test.integrity import write_artifact_manifest  # noqa: E402


if __name__ == "__main__":
    print(json.dumps(write_artifact_manifest(ROOT, ROOT / "configs/phase07_final_test.json"), indent=2, sort_keys=True))
