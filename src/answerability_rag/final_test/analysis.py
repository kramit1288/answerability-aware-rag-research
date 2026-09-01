"""Final TEST generation-policy views and predeclared Phase 6 statistical analyses."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

from answerability_rag.generation.policy import build_policy_view
from answerability_rag.retrieval.artifacts import write_canonical_parquet
from answerability_rag.statistics.core import (
    cliffs_delta,
    cluster_bootstrap,
    exact_mcnemar,
    holm_adjust,
    matched_rank_biserial,
    paired_transition_counts,
)

from .common import RESULTS, Phase07Config, require_unsealed, write_csv, write_json


POLICY_FIELDS = (
    "policy_id", "question_id", "final_action", "answered", "selected_k", "response_id",
    "generation_status", "rouge_l_f1", "bertscore_f1", "unsupported_claim_rate",
    "fully_supported_response", "response_with_any_unsupported_claim",
    "mean_claim_support_score", "minimum_claim_support_score", "maximum_claim_contradiction",
    "output_token_count", "y_suff_final",
)


def _clean(values: Any) -> list[float]:
    return [float(value) for value in values if value is not None and not pd.isna(value)]


def _mean(values: Any) -> float | None:
    clean = _clean(values)
    return float(np.mean(clean)) if clean else None


def _median(values: Any) -> float | None:
    clean = _clean(values)
    return float(np.median(clean)) if clean else None


def _ci(frame: pd.DataFrame, statistic: Callable[[pd.DataFrame], float]) -> dict[str, Any]:
    interval = cluster_bootstrap(
        frame, "question_id", statistic, replicates=5000, seed=42, confidence_level=0.95,
    )
    return {
        "point": interval.point_estimate, "ci_low": interval.ci_low, "ci_high": interval.ci_high,
        "requested_replicates": interval.requested_replicates,
        "valid_replicates": interval.valid_replicates,
    }


def _states(root: Path) -> tuple[list[str], dict[tuple[str, int], dict[str, Any]]]:
    generations = pq.read_table(root / RESULTS / "phase07_test_generation_cache.parquet").to_pylist()
    quality = {str(row["response_id"]): row for row in pq.read_table(root / RESULTS / "phase07_test_answer_quality.parquet").to_pylist()}
    grounding = {str(row["response_id"]): row for row in pq.read_table(root / RESULTS / "phase07_test_response_grounding.parquet").to_pylist()}
    states: dict[tuple[str, int], dict[str, Any]] = {}
    for row in generations:
        response_id = str(row["response_id"])
        q = quality[response_id]; g = grounding[response_id]
        states[(str(row["question_id"]), int(row["k"]))] = {
            "response_id": response_id, "generation_status": row["generation_status"],
            "rouge_l_f1": q["rouge_l_f1"], "bertscore_f1": q["bertscore_f1"],
            "unsupported_claim_rate": g["unsupported_claim_rate"],
            "fully_supported_response": g["fully_supported_response"],
            "response_with_any_unsupported_claim": g["response_with_any_unsupported_claim"],
            "mean_claim_support_score": g["mean_claim_support_score"],
            "minimum_claim_support_score": g["minimum_claim_support_score"],
            "maximum_claim_contradiction": g["maximum_claim_contradiction"],
            "output_token_count": row["output_token_count"], "y_suff_final": int(g["y_suff_final"]),
        }
    ids = sorted({key[0] for key in states})
    if any((qid, k) not in states for qid in ids for k in (5, 10)):
        raise ValueError("TEST k5/k10 generation state alignment is incomplete")
    return ids, states


def _policy_summary(policy_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    answered = [row for row in rows if bool(row["answered"])]
    grounding = [row for row in answered if row.get("unsupported_claim_rate") is not None]
    fully = [row for row in answered if row.get("fully_supported_response") is not None]
    unsupported = [row for row in answered if row.get("response_with_any_unsupported_claim") is not None]
    fully_count = sum(row.get("fully_supported_response") is True for row in answered)
    unsupported_count = sum(row.get("response_with_any_unsupported_claim") is True for row in answered)
    return {
        "policy_id": policy_id, "eligible_trajectory_count": len(rows),
        "policy_answer_count": len(answered), "policy_answer_coverage": len(answered) / len(rows),
        "abstention_count": len(rows) - len(answered),
        "successful_generated_answer_count": sum(row.get("generation_status") == "generated" for row in answered),
        "generation_failure_or_empty_count": sum(row.get("generation_status") != "generated" for row in answered),
        "mean_rouge_l": _mean(row.get("rouge_l_f1") for row in answered),
        "median_rouge_l": _median(row.get("rouge_l_f1") for row in answered),
        "mean_bertscore_f1": _mean(row.get("bertscore_f1") for row in answered),
        "median_bertscore_f1": _median(row.get("bertscore_f1") for row in answered),
        "mean_unsupported_claim_rate": _mean(row.get("unsupported_claim_rate") for row in grounding),
        "unsupported_claim_rate_denominator": len(grounding),
        "fully_supported_response_rate": fully_count / len(fully) if fully else None,
        "fully_supported_response_rate_denominator": len(fully),
        "response_with_any_unsupported_claim_rate": unsupported_count / len(unsupported) if unsupported else None,
        "response_with_any_unsupported_claim_rate_denominator": len(unsupported),
        "mean_claim_support_score": _mean(row.get("mean_claim_support_score") for row in answered),
        "mean_output_tokens": _mean(row.get("output_token_count") for row in answered),
        "grounded_answer_yield": fully_count / len(rows),
        "unsupported_answer_population_rate": unsupported_count / len(rows),
        "abstention_quality_metrics": None,
    }


def build_test_policy_views(root: Path, config_path: Path) -> dict[str, Any]:
    config = Phase07Config.load(config_path); require_unsealed(root, config)
    ids, states = _states(root)
    trajectory = pq.read_table(root / RESULTS / "phase07_test_policy_trajectories.parquet").to_pandas()
    actions = {
        policy: dict(zip(frame.question_id.astype(str), frame.final_action.astype(str)))
        for policy, frame in trajectory.groupby("policy_id")
    }
    if set(actions) != {"G2", "G3"} or any(set(value) != set(ids) for value in actions.values()):
        raise ValueError("frozen TEST adaptive policy identities are incomplete")
    policies = {
        "G0": build_policy_view(ids, states, policy_id="G0", fixed_k=5),
        "G1": build_policy_view(ids, states, policy_id="G1", fixed_k=10),
        "G2": build_policy_view(ids, states, policy_id="G2", actions=actions["G2"]),
        "G3": build_policy_view(ids, states, policy_id="G3", actions=actions["G3"]),
    }
    artifacts = {}
    for policy, rows in policies.items():
        artifacts[policy] = write_canonical_parquet(
            root / RESULTS / f"phase07_test_policy_{policy}.parquet", rows,
            POLICY_FIELDS, ("question_id",),
        )
    summaries = [_policy_summary(policy, rows) for policy, rows in policies.items()]
    summary_fields = tuple(summaries[0])
    write_csv(root / RESULTS / "phase07_test_generation_policy_comparison.csv", summaries, summary_fields)
    manifest = {
        "schema_version": "phase07-test-generation-policy-manifest-v1",
        "question_count": len(ids), "policy_artifacts": artifacts,
        "policy_summaries": summaries, "abstention_quality_zero_imputed": False,
    }
    write_json(root / RESULTS / "phase07_test_generation_policy_manifest.json", manifest)
    return manifest


def _paired_frame(root: Path) -> pd.DataFrame:
    g0 = pq.read_table(root / RESULTS / "phase07_test_policy_G0.parquet").to_pandas()
    g1 = pq.read_table(root / RESULTS / "phase07_test_policy_G1.parquet").to_pandas()
    fields = [
        "rouge_l_f1", "bertscore_f1", "unsupported_claim_rate", "mean_claim_support_score",
        "output_token_count", "fully_supported_response", "response_with_any_unsupported_claim",
    ]
    paired = g0[["question_id", *fields]].merge(
        g1[["question_id", *fields]], on="question_id", suffixes=("_k5", "_k10"), validate="one_to_one",
    )
    return paired


def _paired_continuous(paired: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics = ["rouge_l_f1", "bertscore_f1", "unsupported_claim_rate", "mean_claim_support_score", "output_token_count"]
    rows = []; tests = []
    for metric in metrics:
        left, right = f"{metric}_k5", f"{metric}_k10"
        frame = paired[["question_id", left, right]].dropna().copy()
        frame["difference"] = frame[right].astype(float) - frame[left].astype(float)
        mean_ci = _ci(frame, lambda x: float(x.difference.mean()))
        effect_ci = _ci(frame, lambda x: matched_rank_biserial(x.difference))
        differences = frame.difference.to_numpy(float)
        if len(differences) == 0:
            statistic = p_raw = math.nan; status = "no_complete_pairs"
        elif np.allclose(differences, 0):
            statistic, p_raw, status = 0.0, 1.0, "all_zero_differences"
        else:
            result = stats.wilcoxon(differences, zero_method="pratt", alternative="two-sided", method="auto")
            statistic, p_raw, status = float(result.statistic), float(result.pvalue), "ok"
        comparison = f"A_{metric}_k10_minus_k5"
        row = {
            "schema_version": "phase07-test-paired-continuous-v1", "comparison_id": comparison,
            "family_id": "A_paired_k5_k10_continuous", "metric": metric, "direction": "k10_minus_k5",
            "n_pairs": len(frame), "excluded_pairs": len(paired) - len(frame),
            "k5_mean": _mean(frame[left]), "k10_mean": _mean(frame[right]),
            "mean_difference": mean_ci["point"], "mean_difference_ci_low": mean_ci["ci_low"],
            "mean_difference_ci_high": mean_ci["ci_high"], "median_difference": _median(frame.difference),
            "wilcoxon_statistic": statistic, "p_raw": p_raw, "p_holm": None, "reject_holm": False,
            "rank_biserial": effect_ci["point"], "rank_biserial_ci_low": effect_ci["ci_low"],
            "rank_biserial_ci_high": effect_ci["ci_high"], "bootstrap_replicates": 5000,
            "valid_replicates": min(mean_ci["valid_replicates"], effect_ci["valid_replicates"]),
            "seed": 42, "status": status,
        }
        rows.append(row)
        tests.append({
            "family_id": row["family_id"], "comparison_id": comparison, "metric": metric,
            "test_name": "Wilcoxon signed-rank (Pratt)", "n": len(frame), "statistic": statistic,
            "p_raw": p_raw, "p_holm": None, "reject_holm": False,
            "effect_name": "matched_pairs_rank_biserial", "effect_value": effect_ci["point"],
            "ci_low": effect_ci["ci_low"], "ci_high": effect_ci["ci_high"], "status": status,
        })
    return rows, tests


def _paired_binary(paired: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    specs = [
        ("fully_supported_response", "fully_supported_response_k5", "fully_supported_response_k10"),
        ("response_contains_unsupported_claim", "response_with_any_unsupported_claim_k5", "response_with_any_unsupported_claim_k10"),
    ]
    rows = []; tests = []
    for metric, left, right in specs:
        counts = paired_transition_counts(paired[left], paired[right])
        frame = paired.loc[paired[left].notna() & paired[right].notna(), ["question_id", left, right]].copy()
        frame["difference"] = frame[right].astype(bool).astype(int) - frame[left].astype(bool).astype(int)
        interval = _ci(frame, lambda x: float(x.difference.mean()))
        statistic, p_raw = exact_mcnemar(counts)
        comparison = f"C_{metric}_k10_minus_k5"
        row = {
            "schema_version": "phase07-test-paired-binary-v1", "comparison_id": comparison,
            "family_id": "C_paired_k5_k10_binary", "metric": metric, "direction": "k10_minus_k5",
            **counts, "k5_rate": float(frame[left].astype(bool).mean()) if len(frame) else None,
            "k10_rate": float(frame[right].astype(bool).mean()) if len(frame) else None,
            "risk_difference": interval["point"], "risk_difference_ci_low": interval["ci_low"],
            "risk_difference_ci_high": interval["ci_high"], "mcnemar_statistic": statistic,
            "p_raw": p_raw, "p_holm": None, "reject_holm": False,
            "bootstrap_replicates": 5000, "valid_replicates": interval["valid_replicates"], "seed": 42,
        }
        rows.append(row)
        tests.append({
            "family_id": row["family_id"], "comparison_id": comparison, "metric": metric,
            "test_name": "McNemar exact", "n": len(frame), "statistic": statistic,
            "p_raw": p_raw, "p_holm": None, "reject_holm": False,
            "effect_name": "paired_risk_difference", "effect_value": interval["point"],
            "ci_low": interval["ci_low"], "ci_high": interval["ci_high"], "status": "ok",
        })
    return rows, tests


def _association(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output = []; tests = []
    continuous = ["rouge_l_f1", "bertscore_f1", "unsupported_claim_rate"]
    for k, policy in ((5, "G0"), (10, "G1")):
        frame = pq.read_table(root / RESULTS / f"phase07_test_policy_{policy}.parquet").to_pandas()
        for metric in continuous:
            analysis = frame[["question_id", "y_suff_final", metric]].dropna(subset=[metric]).copy()
            sufficient = analysis.loc[analysis.y_suff_final == 1, metric].astype(float)
            insufficient = analysis.loc[analysis.y_suff_final == 0, metric].astype(float)
            if not len(sufficient) or not len(insufficient):
                raise ValueError(f"TEST association group missing for k={k} metric={metric}")
            diff_ci = _ci(analysis, lambda x: float(x.loc[x.y_suff_final == 1, metric].mean() - x.loc[x.y_suff_final == 0, metric].mean()))
            cliff_ci = _ci(analysis, lambda x: cliffs_delta(x.loc[x.y_suff_final == 1, metric], x.loc[x.y_suff_final == 0, metric]))
            result = stats.mannwhitneyu(sufficient, insufficient, alternative="two-sided", method="auto")
            comparison = f"B_k{k}_{metric}_sufficient_minus_insufficient"
            row = {
                "schema_version": "phase07-test-sufficiency-association-v1", "comparison_id": comparison,
                "family_id": "B_sufficiency_association", "k": k, "metric": metric,
                "outcome_type": "continuous", "direction": "sufficient_minus_insufficient",
                "sufficient_n": len(sufficient), "insufficient_n": len(insufficient),
                "excluded_n": len(frame) - len(analysis), "sufficient_mean": float(sufficient.mean()),
                "insufficient_mean": float(insufficient.mean()), "mean_difference": diff_ci["point"],
                "mean_difference_ci_low": diff_ci["ci_low"], "mean_difference_ci_high": diff_ci["ci_high"],
                "statistic": float(result.statistic), "p_raw": float(result.pvalue),
                "p_holm": None, "reject_holm": False, "cliffs_delta": cliff_ci["point"],
                "cliffs_delta_ci_low": cliff_ci["ci_low"], "cliffs_delta_ci_high": cliff_ci["ci_high"],
                "sufficient_rate": None, "insufficient_rate": None, "risk_difference": None,
                "risk_difference_ci_low": None, "risk_difference_ci_high": None,
                "risk_ratio": None, "odds_ratio": None, "bootstrap_replicates": 5000,
                "valid_replicates": min(diff_ci["valid_replicates"], cliff_ci["valid_replicates"]), "seed": 42,
            }
            output.append(row)
            tests.append({
                "family_id": row["family_id"], "comparison_id": comparison, "metric": metric,
                "test_name": "Mann-Whitney U", "n": len(analysis), "statistic": row["statistic"],
                "p_raw": row["p_raw"], "p_holm": None, "reject_holm": False,
                "effect_name": "cliffs_delta", "effect_value": cliff_ci["point"],
                "ci_low": cliff_ci["ci_low"], "ci_high": cliff_ci["ci_high"], "status": "associational",
            })
        metric = "fully_supported_response"
        analysis = frame[["question_id", "y_suff_final", metric]].dropna(subset=[metric]).copy()
        analysis[metric] = analysis[metric].astype(bool).astype(int)
        sufficient = analysis.loc[analysis.y_suff_final == 1, metric]
        insufficient = analysis.loc[analysis.y_suff_final == 0, metric]
        if not len(sufficient) or not len(insufficient):
            raise ValueError(f"TEST binary association group missing for k={k}")
        table = np.asarray([[int(sufficient.sum()), int(len(sufficient)-sufficient.sum())], [int(insufficient.sum()), int(len(insufficient)-insufficient.sum())]])
        fisher = stats.fisher_exact(table, alternative="two-sided")
        interval = _ci(analysis, lambda x: float(x.loc[x.y_suff_final == 1, metric].mean() - x.loc[x.y_suff_final == 0, metric].mean()))
        s_rate, i_rate = float(sufficient.mean()), float(insufficient.mean())
        comparison = f"B_k{k}_{metric}_sufficient_minus_insufficient"
        row = {
            "schema_version": "phase07-test-sufficiency-association-v1", "comparison_id": comparison,
            "family_id": "B_sufficiency_association", "k": k, "metric": metric,
            "outcome_type": "binary", "direction": "sufficient_minus_insufficient",
            "sufficient_n": len(sufficient), "insufficient_n": len(insufficient),
            "excluded_n": len(frame)-len(analysis), "sufficient_mean": None, "insufficient_mean": None,
            "mean_difference": None, "mean_difference_ci_low": None, "mean_difference_ci_high": None,
            "statistic": float(fisher.statistic), "p_raw": float(fisher.pvalue),
            "p_holm": None, "reject_holm": False, "cliffs_delta": None,
            "cliffs_delta_ci_low": None, "cliffs_delta_ci_high": None,
            "sufficient_rate": s_rate, "insufficient_rate": i_rate,
            "risk_difference": interval["point"], "risk_difference_ci_low": interval["ci_low"],
            "risk_difference_ci_high": interval["ci_high"],
            "risk_ratio": s_rate/i_rate if i_rate else None,
            "odds_ratio": float(fisher.statistic) if (table > 0).all() else None,
            "bootstrap_replicates": 5000, "valid_replicates": interval["valid_replicates"], "seed": 42,
        }
        output.append(row)
        tests.append({
            "family_id": row["family_id"], "comparison_id": comparison, "metric": metric,
            "test_name": "Fisher exact", "n": len(analysis), "statistic": row["statistic"],
            "p_raw": row["p_raw"], "p_holm": None, "reject_holm": False,
            "effect_name": "risk_difference", "effect_value": interval["point"],
            "ci_low": interval["ci_low"], "ci_high": interval["ci_high"], "status": "associational",
        })
    return output, tests


def _apply_holm(tests: list[dict[str, Any]], details: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    for family in sorted({row["family_id"] for row in tests}):
        selected = [row for row in tests if row["family_id"] == family]
        adjusted, rejected = holm_adjust([float(row["p_raw"]) for row in selected], alpha=0.05)
        for row, value, reject in zip(selected, adjusted, rejected):
            row["p_holm"] = float(value); row["reject_holm"] = bool(reject)
    lookup = {row["comparison_id"]: row for row in tests}
    for group in details:
        for row in group:
            row["p_holm"] = lookup[row["comparison_id"]]["p_holm"]
            row["reject_holm"] = lookup[row["comparison_id"]]["reject_holm"]
    return tests


def run_test_statistics(root: Path, config_path: Path) -> dict[str, Any]:
    config = Phase07Config.load(config_path); require_unsealed(root, config)
    paired = _paired_frame(root)
    paired["rouge_l_f1_difference"] = paired["rouge_l_f1_k10"] - paired["rouge_l_f1_k5"]
    paired["bertscore_f1_difference"] = paired["bertscore_f1_k10"] - paired["bertscore_f1_k5"]
    paired["unsupported_claim_rate_difference"] = paired["unsupported_claim_rate_k10"] - paired["unsupported_claim_rate_k5"]
    paired["mean_claim_support_score_difference"] = paired["mean_claim_support_score_k10"] - paired["mean_claim_support_score_k5"]
    paired["output_token_count_difference"] = paired["output_token_count_k10"] - paired["output_token_count_k5"]
    paired_rows = paired.where(pd.notna(paired), None).to_dict("records")
    write_canonical_parquet(
        root / RESULTS / "phase07_test_paired_k5_k10.parquet", paired_rows,
        tuple(paired.columns), ("question_id",),
    )
    continuous, test_a = _paired_continuous(paired)
    binary, test_c = _paired_binary(paired)
    association, test_b = _association(root)
    tests = _apply_holm(test_a + test_b + test_c, [continuous, association, binary])
    write_csv(root / RESULTS / "phase07_test_paired_continuous_statistics.csv", continuous, tuple(continuous[0]))
    write_csv(root / RESULTS / "phase07_test_paired_binary_statistics.csv", binary, tuple(binary[0]))
    write_csv(root / RESULTS / "phase07_test_sufficiency_association_statistics.csv", association, tuple(association[0]))
    write_csv(root / RESULTS / "phase07_test_statistical_tests.csv", tests, tuple(tests[0]))
    holm = [{
        "family_id": family, "comparison_id": row["comparison_id"], "rank": rank,
        "family_size": len(selected), "p_raw": row["p_raw"], "p_holm": row["p_holm"],
        "alpha": 0.05, "reject_holm": row["reject_holm"],
    } for family in sorted({row["family_id"] for row in tests})
       for selected in [[x for x in tests if x["family_id"] == family]]
       for rank, row in enumerate(sorted(selected, key=lambda x: (x["p_raw"], x["comparison_id"])), 1)]
    write_csv(root / RESULTS / "phase07_test_holm_correction.csv", holm, tuple(holm[0]))
    effects = [{
        "family_id": row["family_id"], "comparison_id": row["comparison_id"],
        "metric": row["metric"], "effect_name": row["effect_name"],
        "effect_value": row["effect_value"], "ci_low": row["ci_low"], "ci_high": row["ci_high"],
        "experimental_unit": "question_id",
    } for row in tests]
    write_csv(root / RESULTS / "phase07_test_effect_sizes.csv", effects, tuple(effects[0]))
    bootstrap = []
    for row in continuous:
        bootstrap.extend([
            {"comparison_id": row["comparison_id"], "metric": row["metric"], "estimand": "mean_difference", "point_estimate": row["mean_difference"], "ci_low": row["mean_difference_ci_low"], "ci_high": row["mean_difference_ci_high"], "requested_replicates": 5000, "valid_replicates": row["valid_replicates"], "seed": 42, "resampling_unit": "question_id"},
            {"comparison_id": row["comparison_id"], "metric": row["metric"], "estimand": "rank_biserial", "point_estimate": row["rank_biserial"], "ci_low": row["rank_biserial_ci_low"], "ci_high": row["rank_biserial_ci_high"], "requested_replicates": 5000, "valid_replicates": row["valid_replicates"], "seed": 42, "resampling_unit": "question_id"},
        ])
    for row in binary:
        bootstrap.append({"comparison_id": row["comparison_id"], "metric": row["metric"], "estimand": "paired_risk_difference", "point_estimate": row["risk_difference"], "ci_low": row["risk_difference_ci_low"], "ci_high": row["risk_difference_ci_high"], "requested_replicates": 5000, "valid_replicates": row["valid_replicates"], "seed": 42, "resampling_unit": "question_id"})
    for row in association:
        if row["outcome_type"] == "continuous":
            bootstrap.append({"comparison_id": row["comparison_id"], "metric": row["metric"], "estimand": "mean_difference", "point_estimate": row["mean_difference"], "ci_low": row["mean_difference_ci_low"], "ci_high": row["mean_difference_ci_high"], "requested_replicates": 5000, "valid_replicates": row["valid_replicates"], "seed": 42, "resampling_unit": "question_id"})
            bootstrap.append({"comparison_id": row["comparison_id"], "metric": row["metric"], "estimand": "cliffs_delta", "point_estimate": row["cliffs_delta"], "ci_low": row["cliffs_delta_ci_low"], "ci_high": row["cliffs_delta_ci_high"], "requested_replicates": 5000, "valid_replicates": row["valid_replicates"], "seed": 42, "resampling_unit": "question_id"})
        else:
            bootstrap.append({"comparison_id": row["comparison_id"], "metric": row["metric"], "estimand": "risk_difference", "point_estimate": row["risk_difference"], "ci_low": row["risk_difference_ci_low"], "ci_high": row["risk_difference_ci_high"], "requested_replicates": 5000, "valid_replicates": row["valid_replicates"], "seed": 42, "resampling_unit": "question_id"})
    write_csv(root / RESULTS / "phase07_test_bootstrap_intervals.csv", bootstrap, tuple(bootstrap[0]))
    summary = {
        "schema_version": "phase07-test-statistical-analysis-manifest-v1",
        "families": {family: sum(row["family_id"] == family for row in tests) for family in sorted({row["family_id"] for row in tests})},
        "holm_rejections": [row["comparison_id"] for row in tests if row["reject_holm"]],
        "test_count": len(tests), "bootstrap_replicates": 5000, "seed": 42,
        "validation_and_test_inference_separated": True, "test_results_used_for_tuning": False,
    }
    write_json(root / RESULTS / "phase07_test_statistics_manifest.json", summary)
    return summary
