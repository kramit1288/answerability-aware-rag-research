"""Finalize descriptive family summaries and refresh the independent Phase 4 manifest."""

from __future__ import annotations

import argparse
import json
import math
import sys
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from answerability_rag.answerability.pipeline import (  # noqa: E402
    PHASE04_CONFIG_SHA, UPSTREAM, _write_json, phase04_manifest_paths,
)
from answerability_rag.hashing import sha256_file  # noqa: E402


def clean(value: Any) -> Any:
    if isinstance(value, dict): return {key: clean(item) for key, item in value.items()}
    if isinstance(value, list): return [clean(item) for item in value]
    if isinstance(value, float) and math.isnan(value): return None
    if hasattr(value, "item"): return clean(value.item())
    return value


def write_reliability_diagram(results: Path) -> None:
    bins = pd.read_csv(results / "phase04_reliability_bins.csv")
    width, height, left, top, plot = 1000, 650, 90, 60, 520
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.axis{stroke:#222;stroke-width:2}.grid{stroke:#bbb;stroke-width:1;opacity:.45}</style>',
        '<text x="500" y="30" text-anchor="middle" font-size="22">Phase 4 validation reliability</text>',
    ]
    for tick in range(11):
        value = tick / 10
        x = left + value * plot; y = top + (1 - value) * plot
        lines.extend([
            f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot}"/>',
            f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot}" y2="{y:.1f}"/>',
            f'<text x="{x:.1f}" y="{top + plot + 22}" text-anchor="middle" font-size="11">{value:.1f}</text>',
            f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-size="11">{value:.1f}</text>',
        ])
    lines.extend([
        f'<line class="axis" x1="{left}" y1="{top + plot}" x2="{left + plot}" y2="{top + plot}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot}"/>',
        f'<line x1="{left}" y1="{top + plot}" x2="{left + plot}" y2="{top}" stroke="#000" stroke-dasharray="7,5"/>',
        f'<text x="{left + plot / 2}" y="{top + plot + 55}" text-anchor="middle" font-size="16">Mean predicted sufficiency probability</text>',
        f'<text x="24" y="{top + plot / 2}" text-anchor="middle" font-size="16" transform="rotate(-90 24 {top + plot / 2})">Observed positive fraction</text>',
    ])
    for index, ((model, method), group) in enumerate(bins.groupby(["model", "probability_method"], sort=True)):
        observed = group[group["count"] > 0]
        points = " ".join(
            f"{left + row.mean_probability * plot:.1f},{top + (1 - row.positive_fraction) * plot:.1f}"
            for row in observed.itertuples(index=False)
        )
        color = colors[index % len(colors)]
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for row in observed.itertuples(index=False):
            x = left + row.mean_probability * plot; y = top + (1 - row.positive_fraction) * plot
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        legend_y = 95 + index * 34
        lines.extend([
            f'<line x1="660" y1="{legend_y}" x2="700" y2="{legend_y}" stroke="{color}" stroke-width="3"/>',
            f'<text x="712" y="{legend_y + 5}" font-size="12">{escape(str(model))}: {escape(str(method))}</text>',
        ])
    lines.append('</svg>')
    output = ROOT / "artifacts/figures/phase04_reliability_diagram.svg"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-only", action="store_true",
        help="Refresh only the final artifact manifest without rewriting result artifacts.",
    )
    args = parser.parse_args()
    results = ROOT / "artifacts/results"
    candidates = pd.read_csv(results / "phase04_grouped_cv_candidates.csv")
    metrics = pd.read_csv(results / "phase04_model_validation_metrics.csv")
    selected_model = json.loads((ROOT / "configs/phase04_selected_model.json").read_text(encoding="utf-8"))
    selected_family = selected_model["model_family"]
    calibration = selected_model["calibration_method"]
    if not args.manifest_only:
        write_reliability_diagram(results)
        for family, filename, validation_name in (
            ("logistic_regression", "phase04_logistic_selected_results.json", "B2_logistic_regression"),
            ("random_forest", "phase04_random_forest_selected_results.json", None),
        ):
            candidate = candidates[(candidates.model_family == family) & candidates.selected].iloc[0].to_dict()
            if family == "logistic_regression":
                validation = metrics[metrics.model == validation_name].iloc[0].to_dict()
            else:
                validation = metrics[
                    metrics.model.isin(["B3_random_forest", "B4_calibrated_random_forest"])
                    & (metrics.probability_method == calibration)
                ].iloc[0].to_dict()
            hyperparameters = ({"C": float(candidate["C"]),
                                "class_weight": clean(candidate["class_weight"])}
                               if family == "logistic_regression" else
                               {"max_depth": int(candidate["max_depth"]) if not pd.isna(candidate["max_depth"]) else None,
                                "min_samples_leaf": int(candidate["min_samples_leaf"]),
                                "max_features": candidate["max_features"],
                                "class_weight": clean(candidate["class_weight"])})
            value = {
                "schema_version": "phase04-family-selected-result-v1",
                "model_family": family, "selected_hyperparameters": hyperparameters,
                "grouped_cv_selection": clean(candidate), "validation_metrics": clean(validation),
                "selected_primary_family": family == selected_family,
            }
            if family == "random_forest": value["selected_calibration"] = calibration
            _write_json(results / filename, value)

    paths = phase04_manifest_paths(selected_family)
    missing = [path for path in paths if not (ROOT / path).exists()]
    if missing:
        raise FileNotFoundError(f"Phase 4 manifest inputs missing: {missing}")
    manifest = {
        "schema_version": "phase04-artifact-manifest-v1",
        "phase03_upstream_manifest_sha256": UPSTREAM["phase03_final_manifest"][1],
        "phase04_modeling_config_sha256": PHASE04_CONFIG_SHA,
        "artifacts": [{"path": path, "physical_sha256": sha256_file(ROOT / path),
                       "bytes": (ROOT / path).stat().st_size} for path in paths],
        "manifest_includes_itself": False, "test_sealed": True, "phase5_started": False,
    }
    _write_json(results / "phase04_artifact_manifest.json", manifest)
    print(json.dumps({"status": "complete", "artifact_count": len(paths),
                      "manifest_sha256": sha256_file(results / "phase04_artifact_manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
