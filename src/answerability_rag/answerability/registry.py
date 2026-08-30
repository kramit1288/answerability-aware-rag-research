"""Machine-enforced Phase 4 feature boundary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class FeatureRegistry:
    """Resolved registry with exactly one category per known field."""

    families: dict[str, tuple[str, ...]]
    categories: dict[str, str]
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    undefined: tuple[str, ...]
    identifier_regex: str

    @property
    def model_features(self) -> tuple[str, ...]:
        return tuple(feature for values in self.families.values() for feature in values)

    def features_for_families(self, names: Iterable[str]) -> tuple[str, ...]:
        return tuple(feature for name in names for feature in self.families[name])

    def validate_model_columns(self, columns: Iterable[str]) -> None:
        observed = tuple(columns)
        duplicates = sorted({name for name in observed if observed.count(name) > 1})
        unknown = sorted(set(observed) - set(self.categories))
        forbidden = sorted(
            name for name in set(observed) & set(self.categories)
            if self.categories[name] != "inference_available_feature"
        )
        if duplicates or unknown or forbidden:
            raise ValueError(
                "Phase 4 feature leakage guard failed: "
                f"duplicates={duplicates}, unknown={unknown}, forbidden={forbidden}"
            )


def load_feature_registry(path: Path) -> FeatureRegistry:
    values = json.loads(path.read_text(encoding="utf-8"))
    allowed = values["allowed_model_category"]
    if allowed != "inference_available_feature":
        raise ValueError(f"unexpected allowed feature category: {allowed}")
    declared_categories = tuple(values["categories"])
    if declared_categories != (
        "inference_available_feature", "target_only", "label_construction_only",
        "provenance_only", "evaluation_only",
    ):
        raise ValueError("feature registry category order/content differs")

    categories: dict[str, str] = {}
    for category, fields in values["field_categories"].items():
        if category not in declared_categories:
            raise ValueError(f"unknown feature category: {category}")
        for field in fields:
            if field in categories:
                raise ValueError(f"field classified more than once: {field}")
            categories[field] = category

    families = {name: tuple(fields) for name, fields in values["feature_families"].items()}
    model_features = tuple(feature for fields in families.values() for feature in fields)
    if len(model_features) != len(set(model_features)):
        raise ValueError("inference feature occurs in multiple feature families")
    if set(model_features) != {
        name for name, category in categories.items() if category == allowed
    }:
        raise ValueError("feature families and inference-available classifications differ")
    numeric = tuple(values["numeric_features"])
    categorical = tuple(values["categorical_features"])
    if set(numeric).intersection(categorical) or set(numeric).union(categorical) != set(model_features):
        raise ValueError("numeric/categorical partition does not cover the feature set exactly")
    undefined = tuple(values["mathematically_undefined_features"])
    if not set(undefined).issubset(numeric):
        raise ValueError("undefined features must be numeric model features")
    return FeatureRegistry(
        families, categories, numeric, categorical, undefined,
        values["identifier_like_token_regex"],
    )


def assert_test_sealed(split: str, operation: str) -> None:
    """Fail before any Phase 4 TEST feature, inference, or aggregate operation."""
    if split.casefold() == "test":
        raise PermissionError(f"Phase 4 TEST seal forbids {operation}")
