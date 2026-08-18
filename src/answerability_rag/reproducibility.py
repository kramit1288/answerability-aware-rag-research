"""Environment capture limited to Phase 1 data loading and optimization."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from typing import Any


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def capture_environment() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": {
            "answerability-aware-rag": package_version("answerability-aware-rag"),
            "scipy": package_version("scipy"),
            "numpy": package_version("numpy"),
            "datasets": package_version("datasets"),
            "huggingface-hub": package_version("huggingface-hub"),
        },
        "data_loader": "python-standard-library-json-csv-urllib-v1",
    }
