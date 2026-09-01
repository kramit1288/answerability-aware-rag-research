"""Validation-to-TEST comparison, final tables/figures, and RQ evidence summary."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from answerability_rag.hashing import sha256_file
from answerability_rag.io import write_bytes_atomic

from .common import RESULTS, Phase07Config, require_unsealed, write_csv, write_json


TABLES = Path("artifacts/tables")
FIGURES = Path("artifacts/figures")
matplotlib.rcParams["svg.hashsalt"] = "answerability-aware-rag-phase07"


def _fmt(value: Any, digits: int = 4) -> str:
    return "NA" if value is None or pd.isna(value) else f"{float(value):.{digits}f}"


def _markdown_table(frame: pd.DataFrame, title: str) -> bytes:
    shown = frame.copy()
    for column in shown.columns:
        shown[column] = shown[column].map(lambda value: "NA" if pd.isna(value) else value)
    lines = [f"# {title}", "", "| " + " | ".join(map(str, shown.columns)) + " |", "|" + "|".join("---" for _ in shown.columns) + "|"]
    for row in shown.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _write_table(root: Path, name: str, title: str, frame: pd.DataFrame) -> dict[str, Any]:
    csv_path = root / TABLES / f"{name}.csv"
    md_path = root / TABLES / f"{name}.md"
    write_csv(csv_path, frame.to_dict("records"), tuple(frame.columns))
    write_bytes_atomic(md_path, _markdown_table(frame, title))
    return {
        "table_id": name, "title": title,
        "csv": {"path": csv_path.relative_to(root).as_posix(), "physical_sha256": sha256_file(csv_path)},
        "markdown": {"path": md_path.relative_to(root).as_posix(), "physical_sha256": sha256_file(md_path)},
        "rows": len(frame),
    }


def _classifier_table(root: Path) -> pd.DataFrame:
    validation = pd.read_csv(root / RESULTS / "phase04_model_validation_metrics.csv")
    validation = validation.loc[validation.selected_model.astype(bool)].iloc[0]
    val_ci = pd.read_csv(root / RESULTS / "phase04_bootstrap_confidence_intervals.csv").set_index("metric")
    test = json.loads((root / RESULTS / "phase07_test_classifier_metrics.json").read_text(encoding="utf-8"))
    test_ci = pd.read_csv(root / RESULTS / "phase07_test_classifier_bootstrap_intervals.csv").set_index("metric")
    rows = []
    for metric in ("auroc", "auprc", "accuracy", "precision", "recall", "f1", "brier", "ece"):
        for split, point, intervals in (("VALIDATION", validation[metric], val_ci), ("TEST", test[metric], test_ci)):
            ci = intervals.loc[metric] if metric in intervals.index else None
            rows.append({
                "split": split, "metric": metric, "point_estimate": point,
                "ci_low": None if ci is None else ci.get("lower_95", ci.get("ci_low")),
                "ci_high": None if ci is None else ci.get("upper_95", ci.get("ci_high")),
                "bootstrap_replicates": None if ci is None else ci.get("replicates", ci.get("requested_replicates")),
            })
    return pd.DataFrame(rows)


def _risk_table(root: Path) -> pd.DataFrame:
    val_aurc = json.loads((root / RESULTS / "phase04_aurc.json").read_text(encoding="utf-8"))["aurc"]
    test_aurc = json.loads((root / RESULTS / "phase07_test_aurc.json").read_text(encoding="utf-8"))["aurc"]
    val = pd.read_csv(root / RESULTS / "phase04_risk_operating_points.csv")
    test = pd.read_csv(root / RESULTS / "phase07_test_frozen_risk_operating_points.csv")
    rows = [{"split": "VALIDATION", "operating_point": "AURC", "threshold": None, "coverage": None, "selective_risk": val_aurc},
            {"split": "TEST", "operating_point": "AURC", "threshold": None, "coverage": None, "selective_risk": test_aurc}]
    for row in val.itertuples(index=False):
        rows.append({"split": "VALIDATION", "operating_point": f"{float(row.risk_constraint):.0%}", "threshold": row.threshold, "coverage": row.coverage, "selective_risk": row.selective_risk})
    for row in test.itertuples(index=False):
        rows.append({"split": "TEST", "operating_point": f"{float(row.validation_risk_constraint):.0%}", "threshold": row.frozen_validation_threshold, "coverage": row.test_coverage, "selective_risk": row.test_selective_risk})
    return pd.DataFrame(rows)


def _policy_table(root: Path) -> pd.DataFrame:
    val = pd.read_csv(root / RESULTS / "phase04_policy_operating_points.csv")
    test = pd.read_csv(root / RESULTS / "phase07_test_policy_metrics.csv")
    rows = []
    for risk, policy in ((0.1, "G2"), (0.2, "G3")):
        v = val.loc[np.isclose(val.risk_constraint, risk)].iloc[0]
        t = test.loc[test.policy_id == policy].iloc[0]
        rows.extend([
            {"split": "VALIDATION", "policy_id": policy, "risk_constraint": risk, "coverage": v.final_answer_coverage, "selective_risk": v.final_selective_risk, "false_abstention_rate": v.false_abstention_rate, "expansion_rate": v.retrieval_expansion_rate, "safe_answers": int(v.safe_answer_count), "unsafe_answers": int(v.unsafe_answer_count)},
            {"split": "TEST", "policy_id": policy, "risk_constraint": risk, "coverage": t.answer_coverage, "selective_risk": t.selective_risk, "false_abstention_rate": t.false_abstention_rate, "expansion_rate": t.retrieval_expansion_rate, "safe_answers": int(t.safe_answer_count), "unsafe_answers": int(t.unsafe_answer_count)},
        ])
    return pd.DataFrame(rows)


def _quality_table(root: Path) -> pd.DataFrame:
    validation = pd.read_csv(root / RESULTS / "phase05_policy_generation_comparison.csv")
    test = pd.read_csv(root / RESULTS / "phase07_test_generation_policy_comparison.csv")
    rows = []
    for split, frame in (("VALIDATION", validation), ("TEST", test)):
        for policy, k in (("G0", 5), ("G1", 10)):
            row = frame.loc[frame.policy_id == policy].iloc[0]
            rows.append({
                "split": split, "k": k, "answer_count": int(row.policy_answer_count),
                "coverage": row.policy_answer_coverage, "mean_rouge_l": row.mean_rouge_l,
                "mean_bertscore_f1": row.mean_bertscore_f1,
                "mean_unsupported_claim_rate": row.mean_unsupported_claim_rate,
                "fully_supported_response_rate": row.fully_supported_response_rate,
                "mean_claim_support_score": row.mean_claim_support_score,
                "mean_output_tokens": row.mean_output_tokens,
            })
    return pd.DataFrame(rows)


def _association_table(root: Path) -> pd.DataFrame:
    val = pd.read_csv(root / RESULTS / "phase06_sufficiency_association_statistics.csv")
    test = pd.read_csv(root / RESULTS / "phase07_test_sufficiency_association_statistics.csv")
    columns = ["comparison_id", "k", "metric", "outcome_type", "sufficient_n", "insufficient_n", "mean_difference", "risk_difference", "cliffs_delta", "p_raw", "p_holm", "reject_holm"]
    return pd.concat([val[columns].assign(split="VALIDATION"), test[columns].assign(split="TEST")], ignore_index=True)[["split", *columns]]


def _tests_table(root: Path) -> pd.DataFrame:
    val = pd.read_csv(root / RESULTS / "phase06_statistical_tests.csv")
    test = pd.read_csv(root / RESULTS / "phase07_test_statistical_tests.csv")
    columns = ["family_id", "comparison_id", "metric", "test_name", "statistic", "p_raw", "p_holm", "reject_holm", "effect_name", "effect_value", "ci_low", "ci_high"]
    return pd.concat([val[columns].assign(split="VALIDATION"), test[columns].assign(split="TEST")], ignore_index=True)[["split", *columns]]


def _target_condition_table(root: Path) -> pd.DataFrame:
    validation = pq.read_table(root / "artifacts/data/phase03_final_primary_target.parquet").to_pandas()
    validation = validation.loc[validation.split == "validation"]
    test = pq.read_table(root / RESULTS / "phase07_test_final_target.parquet").to_pandas()
    rows = []
    for split, frame in (("VALIDATION", validation), ("TEST", test)):
        for (strategy, k), group in frame.groupby(["retrieval_strategy", "k"], sort=True):
            rows.append({
                "split": split, "retrieval_strategy": strategy, "k": int(k),
                "question_count": int(group.question_id.nunique()), "condition_count": len(group),
                "sufficient_count": int(group.y_suff_final.astype(int).sum()),
                "sufficiency_rate": float(group.y_suff_final.astype(int).mean()),
            })
    return pd.DataFrame(rows)


def _comparison_rows(classifier: pd.DataFrame, risk: pd.DataFrame, policy: pd.DataFrame, quality: pd.DataFrame, association: pd.DataFrame) -> pd.DataFrame:
    rows = []
    higher_better = {"auroc", "auprc", "f1"}
    for metric in ("auroc", "auprc", "f1", "brier"):
        v = classifier[(classifier.split == "VALIDATION") & (classifier.metric == metric)].iloc[0]
        t = classifier[(classifier.split == "TEST") & (classifier.metric == metric)].iloc[0]
        overlap = not (pd.isna(v.ci_low) or pd.isna(t.ci_low)) and max(v.ci_low, t.ci_low) <= min(v.ci_high, t.ci_high)
        if overlap:
            label = "uncertain due to CI overlap/sample size"
        else:
            stronger = t.point_estimate > v.point_estimate if metric in higher_better else t.point_estimate < v.point_estimate
            label = "stronger on TEST" if stronger else "weaker on TEST"
        rows.append({"domain": "classifier", "metric": metric, "validation": v.point_estimate, "test": t.point_estimate, "classification": label})
    for metric in ("AURC",):
        v = risk[(risk.split == "VALIDATION") & (risk.operating_point == metric)].iloc[0]
        t = risk[(risk.split == "TEST") & (risk.operating_point == metric)].iloc[0]
        rows.append({"domain": "risk_coverage", "metric": "aurc", "validation": v.selective_risk, "test": t.selective_risk, "classification": "stronger on TEST" if t.selective_risk < v.selective_risk else "weaker on TEST"})
    for policy_id in ("G2", "G3"):
        for metric in ("coverage", "selective_risk"):
            v = policy[(policy.split == "VALIDATION") & (policy.policy_id == policy_id)].iloc[0][metric]
            t = policy[(policy.split == "TEST") & (policy.policy_id == policy_id)].iloc[0][metric]
            direction = (t >= v) if metric == "coverage" else (t <= v)
            rows.append({"domain": "policy", "metric": f"{policy_id}_{metric}", "validation": v, "test": t, "classification": "directionally consistent" if direction else "weaker on TEST"})
    for k in (5, 10):
        for metric in ("mean_rouge_l", "mean_bertscore_f1", "mean_unsupported_claim_rate", "fully_supported_response_rate"):
            v = quality[(quality.split == "VALIDATION") & (quality.k == k)].iloc[0][metric]
            t = quality[(quality.split == "TEST") & (quality.k == k)].iloc[0][metric]
            higher = metric not in {"mean_unsupported_claim_rate"}
            stronger = (t >= v) if higher else (t <= v)
            rows.append({"domain": "generation", "metric": f"k{k}_{metric}", "validation": v, "test": t, "classification": "stronger on TEST" if stronger else "weaker on TEST"})
    for comparison_id in sorted(association.comparison_id.unique()):
        v = association[(association.split == "VALIDATION") & (association.comparison_id == comparison_id)].iloc[0]
        t = association[(association.split == "TEST") & (association.comparison_id == comparison_id)].iloc[0]
        column = "risk_difference" if v.outcome_type == "binary" else "mean_difference"
        vv, tt = v[column], t[column]
        classification = "directionally consistent" if np.sign(vv) == np.sign(tt) else "weaker on TEST"
        rows.append({"domain": "sufficiency_association", "metric": comparison_id, "validation": vv, "test": tt, "classification": classification})
    return pd.DataFrame(rows)


def _save_figure(fig: Any, root: Path, stem: str, data: pd.DataFrame) -> dict[str, Any]:
    png = root / FIGURES / f"{stem}.png"; svg = root / FIGURES / f"{stem}.svg"; csv = root / FIGURES / f"{stem}_data.csv"
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=180, bbox_inches="tight", metadata={"Software": "answerability-aware-rag Phase 7"})
    fig.savefig(svg, bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    svg_bytes = svg.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    write_bytes_atomic(svg, svg_bytes)
    write_csv(csv, data.to_dict("records"), tuple(data.columns))
    return {"figure_id": stem, "png": {"path": png.relative_to(root).as_posix(), "physical_sha256": sha256_file(png)}, "svg": {"path": svg.relative_to(root).as_posix(), "physical_sha256": sha256_file(svg)}, "data": {"path": csv.relative_to(root).as_posix(), "physical_sha256": sha256_file(csv)}}


def _figures(root: Path, association: pd.DataFrame, policy: pd.DataFrame) -> list[dict[str, Any]]:
    validation = pd.read_csv(root / RESULTS / "phase04_risk_coverage_curve.csv").assign(split="VALIDATION")
    test = pd.read_csv(root / RESULTS / "phase07_test_risk_coverage_curve.csv").assign(split="TEST")
    risk = pd.concat([validation[["coverage", "selective_risk", "split"]], test[["coverage", "selective_risk", "split"]]])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for split, frame in risk.groupby("split"):
        ax.plot(frame.coverage, frame.selective_risk, label=split)
    ax.set(xlabel="Coverage", ylabel="Selective risk", title="Frozen validation and final TEST risk–coverage")
    ax.legend(); ax.grid(alpha=.25)
    figures = [_save_figure(fig, root, "phase07_figure01_validation_test_risk_coverage", risk)]
    assoc = association.loc[association.metric == "unsupported_claim_rate", ["split", "k", "mean_difference"]]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for index, split in enumerate(("VALIDATION", "TEST")):
        frame = assoc.loc[assoc.split == split]
        ax.bar(frame.k + (index-.5)*0.8, frame.mean_difference, width=.35, label=split)
    ax.axhline(0, color="black", linewidth=.8); ax.set(xlabel="Retrieval depth k", ylabel="Sufficient − insufficient mean unsupported rate", title="Context sufficiency and unsupported-generation proxy")
    ax.set_xticks([5, 10]); ax.legend(); ax.grid(axis="y", alpha=.25)
    figures.append(_save_figure(fig, root, "phase07_figure02_sufficiency_unsupported_association", assoc))
    generation = pd.concat([
        pd.read_csv(root / RESULTS / "phase05_policy_generation_comparison.csv").assign(split="VALIDATION"),
        pd.read_csv(root / RESULTS / "phase07_test_generation_policy_comparison.csv").assign(split="TEST"),
    ])
    plot = generation[["split", "policy_id", "policy_answer_coverage", "unsupported_answer_population_rate"]]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for split, frame in plot.groupby("split"):
        ax.scatter(frame.policy_answer_coverage, frame.unsupported_answer_population_rate, label=split)
        for row in frame.itertuples(index=False):
            ax.annotate(row.policy_id, (row.policy_answer_coverage, row.unsupported_answer_population_rate), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set(xlabel="Answer coverage", ylabel="Unsupported-answer population rate", title="Policy coverage and unsupported-answer exposure")
    ax.legend(); ax.grid(alpha=.25)
    figures.append(_save_figure(fig, root, "phase07_figure03_policy_coverage_unsupported_exposure", plot))
    return figures


def _final_results(root: Path, classifier: pd.DataFrame, risk: pd.DataFrame, policy: pd.DataFrame, quality: pd.DataFrame, association: pd.DataFrame, tests: pd.DataFrame, comparison: pd.DataFrame) -> bytes:
    target = json.loads((root / RESULTS / "phase07_test_target_manifest.json").read_text(encoding="utf-8"))
    census = json.loads((root / RESULTS / "phase07_test_population_census.json").read_text(encoding="utf-8"))
    sensitivity = json.loads((root / RESULTS / "phase07_benchmark_impossible_test_sensitivity.json").read_text(encoding="utf-8"))
    rejections = tests.loc[(tests.split == "TEST") & tests.reject_holm.astype(bool), "comparison_id"].tolist()
    target_conditions = _target_condition_table(root)
    test_conditions = target_conditions.loc[target_conditions.split == "TEST"]
    paired_test = pd.read_csv(root / RESULTS / "phase07_test_paired_continuous_statistics.csv")
    policy_ci = pd.read_csv(root / RESULTS / "phase07_test_policy_bootstrap_intervals.csv")
    def metric(split: str, name: str) -> Any:
        return classifier[(classifier.split == split) & (classifier.metric == name)].iloc[0].point_estimate
    g2 = policy[(policy.split == "TEST") & (policy.policy_id == "G2")].iloc[0]
    g3 = policy[(policy.split == "TEST") & (policy.policy_id == "G3")].iloc[0]
    lines = [
        "# Phase 7 Final Results", "", "## Boundary and population", "",
        f"TechQA TEST was unsealed once at `{json.loads((root / RESULTS / 'phase07_test_unseal_record.json').read_text(encoding='utf-8'))['unseal_timestamp']}` after governance and upstream integrity passed. No post-TEST scientific choice was permitted or made.", "",
        f"The frozen TEST census was {census['total_test_questions']} questions: {census['benchmark_answerable_test_questions']} benchmark-answerable and {census['benchmark_impossible_test_questions']} benchmark-impossible. PRIMARY contains {census['primary_eligible_test_questions']} questions and {census['primary_eligible_retrieval_conditions']} conditions; 3 unresolved-evidence questions and 45 benchmark-impossible questions were excluded from PRIMARY. The immutable final target contains {target['final_target_class_distribution']['positive']} sufficient and {target['final_target_class_distribution']['negative']} insufficient conditions.", "",
        "## RQ1 — Retrieval and retrieved-context sufficiency", "",
        "Validation evidence established strategy/depth-dependent retrieval and sufficiency patterns under the frozen Phase 2/3 design. On final TEST, the exact same 12 retrieval conditions per PRIMARY question were labelled with the frozen Phase 3 rule. TEST sufficiency rates by condition were: " + "; ".join(f"{row.retrieval_strategy} k={int(row.k)} {_fmt(row.sufficiency_rate)}" for row in test_conditions.itertuples(index=False)) + ". Absolute strategy/depth differences are the predeclared descriptive effects; no new aggregate RQ1 interval or post-hoc test was invented. This is a retrieval-conditioned target, not global question answerability.", "",
        "## RQ2 — Predicting retrieved-context sufficiency", "",
        "The selected uncalibrated Random Forest changed from VALIDATION to TEST as follows: " + "; ".join(
            f"{name} {_fmt(classifier[(classifier.split=='VALIDATION') & (classifier.metric==name)].iloc[0].point_estimate)} to {_fmt(classifier[(classifier.split=='TEST') & (classifier.metric==name)].iloc[0].point_estimate)} (TEST 95% CI [{_fmt(classifier[(classifier.split=='TEST') & (classifier.metric==name)].iloc[0].ci_low)}, {_fmt(classifier[(classifier.split=='TEST') & (classifier.metric==name)].iloc[0].ci_high)}])"
            for name in ("auroc", "auprc", "f1", "brier")
        ) + ". TEST used all 39 frozen inference features and the TRAIN-fitted preprocessing/model without fitting or recalibration.", "",
        "## RQ3 — Frozen answer/request-more-evidence/abstain policy", "",
        f"TEST AURC is {_fmt(risk[(risk.split=='TEST') & (risk.operating_point=='AURC')].iloc[0].selective_risk)}. The primary G2 policy (t_low=0.78, t_high=0.82) achieved coverage {_fmt(g2.coverage)} (95% CI [{_fmt(policy_ci[(policy_ci.policy_id=='G2') & (policy_ci.metric=='answer_coverage')].iloc[0].ci_low)}, {_fmt(policy_ci[(policy_ci.policy_id=='G2') & (policy_ci.metric=='answer_coverage')].iloc[0].ci_high)}]), selective risk {_fmt(g2.selective_risk)} (CI [{_fmt(policy_ci[(policy_ci.policy_id=='G2') & (policy_ci.metric=='selective_risk')].iloc[0].ci_low)}, {_fmt(policy_ci[(policy_ci.policy_id=='G2') & (policy_ci.metric=='selective_risk')].iloc[0].ci_high)}]), false-abstention {_fmt(g2.false_abstention_rate)}, and expansion {_fmt(g2.expansion_rate)} with {int(g2.safe_answers)} safe and {int(g2.unsafe_answers)} unsafe answers. The G3 sensitivity policy (0.56, 0.72) achieved coverage {_fmt(g3.coverage)} and selective risk {_fmt(g3.selective_risk)}. Thresholds were not reselected even when TEST risk exceeded its nominal validation constraint.", "",
        "## RQ4 — Gating and unsupported-claim exposure", "",
        "The final k5/k10 quality and grounding table reports ROUGE-L, BERTScore F1, the automatic unsupported-claim-rate proxy, fully-supported response rate, claim support, and output length. TEST paired k10−k5 effects were: " + "; ".join(f"{row.metric} mean difference {_fmt(row.mean_difference)} (95% CI [{_fmt(row.mean_difference_ci_low)}, {_fmt(row.mean_difference_ci_high)}]), rank-biserial {_fmt(row.rank_biserial)}, Holm p={_fmt(row.p_holm)}" for row in paired_test.itertuples(index=False)) + ". Sufficiency associations, binary paired effects, and their CIs are in Tables 5–6. The grounding outputs are an automatic retrieved-context support proxy, not authoritative hallucination labels.", "",
        ("TEST Holm-surviving hypotheses: " + ", ".join(f"`{value}`" for value in rejections) + ".") if rejections else "No TEST hypothesis survived the predeclared within-family Holm correction.", "",
        "## Validation-to-TEST consistency", "",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(f"- {row.domain}/{row.metric}: VALIDATION {_fmt(row.validation)}, TEST {_fmt(row.test)} — {row.classification}.")
    lines += ["", "## Benchmark-impossible sensitivity", "",
        f"The separate 45-question/540-condition benchmark-impossible sensitivity produced a predicted-sufficient rate of {_fmt(sensitivity['predicted_sufficient_rate_at_0_5'])} at 0.5. These preliminary negatives are benchmark-relative and contaminated; they are not perfect context-sufficiency ground truth and did not influence PRIMARY results.", "",
        "## Limitations", "",
        "- TEST has 88 PRIMARY questions, so uncertainty can remain wide and subgroup results may be sparse.",
        "- The final target is operational and inherits evidence-alignment, claim-segmentation, NLI-proxy, and single-annotator limitations from Phase 3.",
        "- ROUGE-L and BERTScore measure reference similarity, not grounding; the NLI grounding measure is a validated but imperfect proxy.",
        "- Selective risk is an observed held-out estimate, not a production safety guarantee.",
        "- Benchmark-impossible sensitivity has known label contamination and remains separate.",
        "- No TEST result was used to tune or reselect any component; worse generalization is retained unchanged.", "",
        "This document is the Phase 7 empirical source for later thesis Chapters 4–6. It is not a thesis rewrite.", "",
    ]
    return "\n".join(lines).encode("utf-8")


def run_final_reporting(root: Path, config_path: Path) -> dict[str, Any]:
    config = Phase07Config.load(config_path); require_unsealed(root, config)
    classifier = _classifier_table(root); risk = _risk_table(root); policy = _policy_table(root)
    quality = _quality_table(root); association = _association_table(root); tests = _tests_table(root)
    target_conditions = _target_condition_table(root)
    tables = [
        _write_table(root, "phase07_table00_target_by_retrieval_condition", "Retrieved-context sufficiency by strategy and depth", target_conditions),
        _write_table(root, "phase07_table01_classifier_validation_test", "Final classifier: VALIDATION vs TEST", classifier),
        _write_table(root, "phase07_table02_risk_coverage_validation_test", "Frozen risk/coverage: VALIDATION vs TEST", risk),
        _write_table(root, "phase07_table03_policy_validation_test", "Frozen adaptive policies: VALIDATION vs TEST", policy),
        _write_table(root, "phase07_table04_quality_grounding_validation_test", "k5/k10 answer quality and grounding", quality),
        _write_table(root, "phase07_table05_sufficiency_associations", "Context sufficiency associations", association),
        _write_table(root, "phase07_table06_inferential_tests", "Predeclared inferential tests", tests),
    ]
    comparison = _comparison_rows(classifier, risk, policy, quality, association)
    write_csv(root / RESULTS / "phase07_validation_test_comparison.csv", comparison.to_dict("records"), tuple(comparison.columns))
    figures = _figures(root, association, policy)
    write_json(root / RESULTS / "phase07_final_tables_manifest.json", {"schema_version": "phase07-final-tables-manifest-v1", "tables": tables})
    write_json(root / RESULTS / "phase07_final_figures_manifest.json", {"schema_version": "phase07-final-figures-manifest-v1", "figures": figures})
    write_bytes_atomic(root / "docs/PHASE_07_FINAL_RESULTS.md", _final_results(root, classifier, risk, policy, quality, association, tests, comparison))
    return {"status": "pass", "table_count": len(tables), "figure_count": len(figures), "comparison_rows": len(comparison)}
