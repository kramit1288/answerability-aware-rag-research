"""Deterministic Phase 6 thesis tables and standalone Matplotlib figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from answerability_rag.hashing import sha256_file
from answerability_rag.io import write_bytes_atomic, write_json_atomic


COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#666666",
}


def _display(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    return str(value).replace("|", "\\|").replace("\n", " ")


def dataframe_markdown(frame: pd.DataFrame, title: str) -> bytes:
    """Render a dependency-free GitHub Markdown table from the canonical CSV frame."""
    headers = [str(column) for column in frame.columns]
    lines = [f"# {title}", "", "| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_display(value) for value in row) + " |")
    lines.append("")
    return "\n".join(lines).encode("utf-8")


def write_table(csv_path: Path, markdown_path: Path, frame: pd.DataFrame, title: str) -> None:
    """Write exact CSV and a mechanically rendered Markdown view."""
    csv_content = frame.to_csv(index=False, lineterminator="\n", na_rep="").encode("utf-8")
    write_bytes_atomic(csv_path, csv_content)
    reread = pd.read_csv(csv_path, keep_default_na=False)
    write_bytes_atomic(markdown_path, dataframe_markdown(reread, title))


def _figure_style() -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        "svg.fonttype": "none",
        "svg.hashsalt": "answerability-aware-rag-phase06",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _save(fig: plt.Figure, base_path: Path) -> tuple[Path, Path]:
    svg = base_path.with_suffix(".svg")
    png = base_path.with_suffix(".png")
    fig.savefig(
        svg,
        format="svg",
        bbox_inches="tight",
        metadata={"Date": None, "Creator": "answerability-aware-rag Phase 6"},
    )
    write_bytes_atomic(svg, svg.read_bytes().replace(b"\r\n", b"\n"))
    fig.savefig(
        png,
        format="png",
        dpi=300,
        bbox_inches="tight",
        metadata={"Software": "answerability-aware-rag Phase 6"},
    )
    plt.close(fig)
    return svg, png


def _line_figure(data: pd.DataFrame, base: Path) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    colors = {"bm25": COLORS["blue"], "dense": COLORS["orange"], "hybrid": COLORS["green"]}
    labels = {"bm25": "BM25", "dense": "Dense", "hybrid": "Hybrid RRF"}
    for strategy in ["bm25", "dense", "hybrid"]:
        part = data.loc[data.retrieval_strategy == strategy].sort_values("k")
        ax.plot(part.k, part.document_recall_at_k, marker="o", linewidth=2,
                color=colors[strategy], label=labels[strategy])
    ax.set(title="Validation document Recall@k", xlabel="Retrieval depth k",
           ylabel="Document Recall@k", xticks=[1, 3, 5, 10], ylim=(0, 1))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, base)


def _model_dot(data: pd.DataFrame, metric: str, title: str, base: Path) -> tuple[Path, Path]:
    labels = {
        "B1_idf_coverage_threshold": "B1 score threshold",
        "B2_logistic_regression": "Logistic Regression",
        "B3_random_forest": "Random Forest",
    }
    ordered = data.set_index("model").loc[list(labels)].reset_index()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    y = np.arange(len(ordered))
    ax.scatter(ordered[metric], y, s=65, color=COLORS["blue"], zorder=3)
    for yi, value in zip(y, ordered[metric], strict=True):
        ax.text(float(value) + 0.006, yi, f"{float(value):.3f}", va="center")
    ax.set_yticks(y, [labels[value] for value in ordered.model])
    ax.set(title=title, xlabel=metric.upper(), xlim=(0.5, 0.8))
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()
    fig.tight_layout()
    return _save(fig, base)


def _reliability(data: pd.DataFrame, base: Path) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    observed = data.loc[data["count"] > 0].copy()
    ax.plot([0, 1], [0, 1], linestyle="--", color=COLORS["gray"], label="Perfect calibration")
    ax.plot(observed.mean_probability, observed.positive_fraction, marker="o", linewidth=2,
            color=COLORS["blue"], label="Selected Random Forest")
    ax.set(title="Selected-model reliability on validation conditions",
           xlabel="Mean predicted sufficiency probability",
           ylabel="Observed sufficient-context fraction", xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    return _save(fig, base)


def _importance(data: pd.DataFrame, base: Path) -> tuple[Path, Path]:
    shown = data.sort_values("rank").head(15).sort_values("permutation_auprc_decrease_mean")
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    ax.errorbar(
        shown.permutation_auprc_decrease_mean,
        np.arange(len(shown)),
        xerr=shown.permutation_auprc_decrease_std,
        fmt="o",
        color=COLORS["blue"],
        ecolor=COLORS["gray"],
        capsize=3,
    )
    ax.set_yticks(np.arange(len(shown)), shown.feature)
    ax.axvline(0, color="#999999", linewidth=1)
    ax.set(title="Random Forest permutation importance",
           xlabel="Mean decrease in validation AUPRC")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return _save(fig, base)


def _risk_coverage(data: pd.DataFrame, base: Path) -> tuple[Path, Path]:
    curve = data.loc[data.record_type == "curve"].sort_values("coverage")
    points = data.loc[data.record_type == "operating_point"]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.plot(curve.coverage, curve.selective_risk, color=COLORS["blue"], linewidth=2)
    ax.scatter(points.coverage, points.selective_risk, color=COLORS["red"], s=50, zorder=3)
    for row in points.itertuples(index=False):
        ax.annotate(f"{float(row.risk_constraint):.0%} constraint",
                    (row.coverage, row.selective_risk), xytext=(6, 6),
                    textcoords="offset points")
    ax.set(title="Selected-model validation risk-coverage curve",
           xlabel="Answer coverage", ylabel="Observed selective risk", xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return _save(fig, base)


def _paired_distribution(data: pd.DataFrame, base: Path) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    values = data.difference_unsupported_claim_rate_k10_minus_k5.to_numpy(dtype=float)
    bins = np.linspace(min(-1.0, values.min()), max(1.0, values.max()), 21)
    ax.hist(values, bins=bins, color=COLORS["blue"], edgecolor="white", alpha=0.9)
    ax.axvline(0, color=COLORS["gray"], linestyle="--", linewidth=1.5)
    ax.axvline(values.mean(), color=COLORS["red"], linewidth=1.8,
               label=f"Mean difference = {values.mean():.3f}")
    ax.set(title="Paired unsupported-claim-rate differences",
           xlabel="Unsupported-claim rate difference (k10 - k5)", ylabel="Questions")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return _save(fig, base)


def _sufficiency_interval(data: pd.DataFrame, base: Path) -> tuple[Path, Path]:
    order = [(5, "insufficient"), (5, "sufficient"), (10, "insufficient"), (10, "sufficient")]
    indexed = data.set_index(["k", "context_group"]).loc[order].reset_index()
    labels = [f"k={int(row.k)}, {row.context_group}" for row in indexed.itertuples(index=False)]
    y = np.arange(len(indexed))
    point = indexed.mean_unsupported_claim_rate.to_numpy(dtype=float)
    low = indexed.ci_low.to_numpy(dtype=float)
    high = indexed.ci_high.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.errorbar(point, y, xerr=np.vstack([point - low, high - point]), fmt="o",
                capsize=4, color=COLORS["blue"], ecolor=COLORS["gray"])
    ax.set_yticks(y, labels)
    ax.set(title="Context sufficiency and unsupported generation",
           xlabel="Mean unsupported-claim rate (95% question-bootstrap CI)", xlim=(0, 1))
    ax.grid(axis="x", alpha=0.2)
    ax.invert_yaxis()
    fig.tight_layout()
    return _save(fig, base)


def _policy_utility(data: pd.DataFrame, base: Path) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    colors = [COLORS["gray"], COLORS["blue"], COLORS["orange"], COLORS["green"]]
    for color, row in zip(colors, data.sort_values("policy_id").itertuples(index=False), strict=True):
        ax.scatter(row.answer_coverage, row.unsupported_answer_population_rate,
                   s=75, color=color, zorder=3)
        ax.annotate(
            f"{row.policy_id}\ngrounded yield={float(row.grounded_answer_yield):.3f}",
            (row.answer_coverage, row.unsupported_answer_population_rate),
            xytext=(7, 6), textcoords="offset points",
        )
    ax.set(title="Validation policy safety-utility observations",
           xlabel="Answer coverage", ylabel="Unsupported-answer population rate",
           xlim=(-0.03, 1.08), ylim=(-0.03, 0.62))
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return _save(fig, base)


def generate_figures(root: Path, figure_data: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Write figure-data CSVs, render standalone SVG/PNG files, and hash all outputs."""
    _figure_style()
    output = root / "artifacts/figures"
    output.mkdir(parents=True, exist_ok=True)
    captions = {
        "F01": {"title": "Validation document Recall@k",
                "caption": "Document Recall@k on the frozen TechQA VALIDATION answerable population. A correct-document hit is not equivalent to retrieved-context sufficiency."},
        "F02A": {"title": "Validation AUPRC by model",
                 "caption": "Validation AUPRC for the simple score threshold, Logistic Regression, and uncalibrated Random Forest. The Random Forest was selected by the frozen AUPRC-first rule."},
        "F02B": {"title": "Validation AUROC by model",
                 "caption": "Validation AUROC for the same frozen classifier comparison. Separate panels avoid a dual-axis display."},
        "F03": {"title": "Selected-model reliability",
                "caption": "Ten-bin reliability diagram for the selected uncalibrated Random Forest on frozen validation retrieval conditions; ECE is a bin-dependent diagnostic."},
        "F04": {"title": "Random Forest permutation importance",
                "caption": "Top inference-time features ranked by fixed-seed validation AUPRC permutation decrease. Importance is descriptive and not causal."},
        "F05": {"title": "Selected-model risk-coverage curve",
                "caption": "Frozen validation risk-coverage curve with the 5%, 10%, and 20% observed-risk operating points. Observed validation risk is not a production guarantee."},
        "F06": {"title": "Paired unsupported-claim-rate differences",
                "caption": "Distribution across 89 paired questions of unsupported-claim rate at k=10 minus k=5. Negative values favor lower binary unsupported exposure at k=10 under the frozen proxy."},
        "F07": {"title": "Context sufficiency and unsupported generation",
                "caption": "Mean unsupported-claim rate by frozen context-sufficiency group, analyzed separately at k=5 and k=10, with 95% question-bootstrap intervals. Comparisons are associational."},
        "F08": {"title": "Policy safety-utility observations",
                "caption": "Answer coverage versus unsupported-answer population rate for G0-G3, annotated with grounded-answer yield. No Pareto-optimality or production-safety claim is made."},
    }
    specs = {
        "F01": ("phase06_figure01_recall_at_k", _line_figure),
        "F02A": ("phase06_figure02a_model_auprc", lambda data, base: _model_dot(data, "auprc", "Validation AUPRC by model", base)),
        "F02B": ("phase06_figure02b_model_auroc", lambda data, base: _model_dot(data, "auroc", "Validation AUROC by model", base)),
        "F03": ("phase06_figure03_reliability", _reliability),
        "F04": ("phase06_figure04_permutation_importance", _importance),
        "F05": ("phase06_figure05_risk_coverage", _risk_coverage),
        "F06": ("phase06_figure06_paired_unsupported_difference", _paired_distribution),
        "F07": ("phase06_figure07_sufficiency_unsupported", _sufficiency_interval),
        "F08": ("phase06_figure08_policy_safety_utility", _policy_utility),
    }
    manifest_rows: list[dict[str, Any]] = []
    for figure_id, (stem, renderer) in specs.items():
        data_path = output / f"{stem}_data.csv"
        write_bytes_atomic(
            data_path,
            figure_data[figure_id].to_csv(index=False, lineterminator="\n", na_rep="").encode("utf-8"),
        )
        plotted = pd.read_csv(data_path)
        svg, png = renderer(plotted, output / stem)
        manifest_rows.append({
            "figure_id": figure_id,
            "title": captions[figure_id]["title"],
            "caption": captions[figure_id]["caption"],
            "data_path": data_path.relative_to(root).as_posix(),
            "data_rows": len(plotted),
            "data_sha256": sha256_file(data_path),
            "svg_path": svg.relative_to(root).as_posix(),
            "svg_sha256": sha256_file(svg),
            "png_path": png.relative_to(root).as_posix(),
            "png_sha256": sha256_file(png),
            "png_dpi": 300,
        })
    write_json_atomic(output / "phase06_figure_captions.json", captions)
    manifest = {
        "schema_version": "phase06-figure-manifest-v1",
        "deterministic_matplotlib": True,
        "seaborn_used": False,
        "figures": manifest_rows,
    }
    write_json_atomic(output / "phase06_figure_manifest.json", manifest)
    return manifest
