"""Validate and evaluate the frozen final Phase 3 human confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from answerability_rag.sufficiency.final_confirmation import (
    build_final_artifact_manifest,
    run_final_confirmation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--build-manifest", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    result = run_final_confirmation(root, validate_only=args.validate_only)
    if args.build_manifest and not args.validate_only:
        result["final_artifact_manifest"] = build_final_artifact_manifest(root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
