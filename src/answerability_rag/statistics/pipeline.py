"""Frozen Phase 6 validation-only statistical analysis and thesis artifact pipeline."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy
from scipy import stats

from answerability_rag.hashing import canonical_json_sha256, sha256_file
from answerability_rag.io import write_bytes_atomic, write_json_atomic
from answerability_rag.statistics.core import (
    aligned_complete_pairs,
    cliffs_delta,
    exact_mcnemar,
    holm_adjust,
    independent_complete_groups,
    matched_rank_biserial,
    paired_transition_counts,
    percentile_bounds,
)
from answerability_rag.statistics.reporting import generate_figures, write_table


EXPECTED_CONFIG_SHA256 = "f0aba2d25163010e81d2dbfee0cd98dbe5a81857839085cd41b048e2e56a6826"
EXPECTED_MEAN_DIFFERENCES = {
    "rouge_l_f1": 0.007129825807797376,
    "bertscore_f1": 0.00040371431393569775,
    "unsupported_claim_rate": -0.024179334655210393,
    "mean_claim_support_score": -0.03426896506473279,
    "output_token_count": -1.9101123595505618,
}


@dataclass(frozen=True)
class Phase06Config:
    path: Path
    values: dict[str, Any]
    canonical_sha256: str

    @classmethod
    def load(cls, path: Path, root: Path) -> "Phase06Config":
        values = json.loads(path.read_text(encoding="utf-8"))
        observed = canonical_json_sha256(values)
        if observed != EXPECTED_CONFIG_SHA256:
            raise ValueError(f"Phase 6 configuration drift: {observed}")
        freeze_path = root / "artifacts/results/phase06_pre_analysis_governance_freeze.json"
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if freeze["phase06_config_canonical_sha256"] != observed:
            raise ValueError("Phase 6 freeze/config canonical hash mismatch")
        for item in freeze["governance_files"]:
            if sha256_file(root / item["path"]) != item["physical_sha256"]:
                raise ValueError(f"frozen Phase 6 governance drift: {item['path']}")
        if freeze["inferential_results_observed_before_freeze"]:
            raise ValueError("invalid governance freeze: results predate methodology")
        return cls(path.resolve(), values, observed)


def _canonical_embedded_hash(path: Path, key: str) -> str:
    value = json.loads(path.read_text(encoding="utf-8"))
    stored = value.pop(key)
    observed = canonical_json_sha256(value)
    if stored != observed:
        raise ValueError(f"canonical embedded hash mismatch: {path}")
    return stored


def verify_upstream(root: Path, config: Phase06Config) -> dict[str, Any]:
    """Verify frozen scientific hashes and invariants without opening TechQA TEST content."""
    upstream = config.values["upstream_freeze"]
    phase3 = _canonical_embedded_hash(
        root / "configs/phase03_final_target.json", "final_target_config_sha256"
    )
    phase4_model = _canonical_embedded_hash(
        root / "configs/phase04_selected_model.json", "selected_model_config_sha256"
    )
    phase4_policy = _canonical_embedded_hash(
        root / "configs/phase04_selected_policy.json", "selected_policy_config_sha256"
    )
    expected = {
        "phase03": upstream["phase03_final_target_config_canonical_sha256"],
        "phase4_model": upstream["phase04_selected_model_canonical_sha256"],
        "phase4_policy": upstream["phase04_selected_policy_canonical_sha256"],
    }
    observed = {"phase03": phase3, "phase4_model": phase4_model, "phase4_policy": phase4_policy}
    if observed != expected:
        raise ValueError(f"frozen upstream canonical hash mismatch: {observed}")
    manifest_path = root / "artifacts/results/phase05_artifact_manifest.json"
    if sha256_file(manifest_path) != upstream["phase05_manifest_physical_sha256"]:
        raise ValueError("Phase 5 final manifest physical hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest["artifacts"]:
        path = root / item["path"]
        if sha256_file(path) != item["physical_sha256"]:
            raise ValueError(f"Phase 5 manifest entry drift: {item['path']}")
    if not manifest["techqa_test_sealed"]:
        raise ValueError("frozen Phase 5 manifest does not preserve the TechQA TEST seal")

    paired = pq.read_table(root / "artifacts/results/phase05_paired_k5_k10.parquet").to_pandas()
    responses = pq.read_table(
        root / "artifacts/results/phase05_techqa_response_grounding.parquet"
    ).to_pandas()
    ragtruth = pq.read_table(root / "artifacts/results/phase05_ragtruth_claim_scores.parquet").to_pandas()
    threshold = json.loads(
        (root / "artifacts/results/phase05_selected_grounding_threshold.json").read_text(
            encoding="utf-8"
        )
    )
    if len(paired) != 89 or paired.question_id.nunique() != 89:
        raise ValueError("frozen paired TechQA population is not exactly 89 questions")
    if len(responses) != 178 or responses.question_id.nunique() != 89 or set(responses.k) != {5, 10}:
        raise ValueError("frozen TechQA generation-state population is not 178 paired states")
    if len(ragtruth) != upstream["required_ragtruth_claims"]:
        raise ValueError("frozen RAGTruth claim population is not 42,296")
    if not np.isclose(float(threshold["t_support"]), upstream["required_support_threshold"]):
        raise ValueError("frozen RAGTruth support threshold is not 0.16")
    return {
        "phase03_final_target_config_sha256": phase3,
        "phase04_selected_model_sha256": phase4_model,
        "phase04_selected_policy_sha256": phase4_policy,
        "phase05_manifest_sha256": sha256_file(manifest_path),
        "techqa_validation_questions": paired.question_id.nunique(),
        "generation_states": len(responses),
        "ragtruth_claims": len(ragtruth),
        "t_support": float(threshold["t_support"]),
        "techqa_test_sealed": True,
    }


def _bootstrap_one_row_per_question(
    frame: pd.DataFrame,
    statistics: dict[str, Callable[[pd.DataFrame], float]],
    *,
    replicates: int,
    seed: int,
    confidence_level: float,
) -> dict[str, dict[str, float | int]]:
    if frame.question_id.duplicated().any() or frame.question_id.isna().any():
        raise ValueError("question bootstrap requires exactly one non-missing row per question_id")
    ordered = frame.sort_values("question_id").reset_index(drop=True)
    n = len(ordered)
    if n == 0:
        raise ValueError("cannot bootstrap an empty question population")
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {name: [] for name in statistics}
    for _ in range(replicates):
        sample = ordered.iloc[rng.integers(0, n, size=n)]
        for name, function in statistics.items():
            try:
                value = float(function(sample))
            except (ValueError, ZeroDivisionError, FloatingPointError):
                continue
            if np.isfinite(value):
                draws[name].append(value)
    results: dict[str, dict[str, float | int]] = {}
    for name, function in statistics.items():
        point = float(function(ordered))
        low, high = percentile_bounds(draws[name], confidence_level)
        results[name] = {
            "point": point,
            "ci_low": low,
            "ci_high": high,
            "valid_replicates": len(draws[name]),
        }
    return results


def _bootstrap_row(
    *, metric: str, contrast: str, kind: str, interval: dict[str, Any], config: Phase06Config
) -> dict[str, Any]:
    return {
        "schema_version": "phase06-bootstrap-interval-v1",
        "metric": metric,
        "system_or_contrast": contrast,
        "statistic": kind,
        "point_estimate": interval["point"],
        "confidence_level": config.values["confidence_level"],
        "ci_low": interval["ci_low"],
        "ci_high": interval["ci_high"],
        "method": "question_id_cluster_percentile_bootstrap",
        "resampling_unit": "question_id",
        "requested_replicates": config.values["bootstrap"]["replicates"],
        "valid_replicates": interval["valid_replicates"],
        "seed": config.values["random_seed"],
    }


def _paired_continuous(
    paired: pd.DataFrame, config: Phase06Config, bootstrap_rows: list[dict[str, Any]]
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    metric_columns = {
        "rouge_l_f1": ("k5_rouge_l_f1", "k10_rouge_l_f1"),
        "bertscore_f1": ("k5_bertscore_f1", "k10_bertscore_f1"),
        "unsupported_claim_rate": ("k5_unsupported_claim_rate", "k10_unsupported_claim_rate"),
        "mean_claim_support_score": ("k5_mean_claim_support_score", "k10_mean_claim_support_score"),
        "output_token_count": ("k5_output_tokens", "k10_output_tokens"),
    }
    rows: list[dict[str, Any]] = []
    tests: list[dict[str, Any]] = []
    effects: list[dict[str, Any]] = []
    reps = config.values["bootstrap"]["replicates"]
    seed = config.values["random_seed"]
    confidence = config.values["confidence_level"]
    for metric, (k5_column, k10_column) in metric_columns.items():
        complete = aligned_complete_pairs(paired, k5_column, k10_column)
        analysis = complete[["question_id", k5_column, k10_column]].copy()
        analysis["difference"] = analysis[k10_column].astype(float) - analysis[k5_column].astype(float)
        observed_mean = float(analysis.difference.mean())
        if not np.isclose(observed_mean, EXPECTED_MEAN_DIFFERENCES[metric], atol=1e-14, rtol=0):
            raise ValueError(
                f"frozen descriptive difference failed for {metric}: {observed_mean}"
            )
        intervals = _bootstrap_one_row_per_question(
            analysis,
            {
                "mean_difference": lambda x: float(x.difference.mean()),
                "median_difference": lambda x: float(x.difference.median()),
                "rank_biserial": lambda x: matched_rank_biserial(x.difference),
            },
            replicates=reps,
            seed=seed,
            confidence_level=confidence,
        )
        for name, interval in intervals.items():
            bootstrap_rows.append(_bootstrap_row(
                metric=metric, contrast="k10_minus_k5", kind=name,
                interval=interval, config=config,
            ))
        differences = analysis.difference.to_numpy(dtype=float)
        if np.allclose(differences, 0):
            statistic, p_raw, status = 0.0, 1.0, "all_zero_differences"
        else:
            result = stats.wilcoxon(
                differences, zero_method="pratt", alternative="two-sided", method="auto"
            )
            statistic, p_raw, status = float(result.statistic), float(result.pvalue), "ok"
        comparison_id = f"A_{metric}_k10_minus_k5"
        row = {
            "schema_version": "phase06-paired-continuous-v1",
            "comparison_id": comparison_id,
            "family_id": "A_paired_k5_k10_continuous",
            "metric": metric,
            "direction": "k10_minus_k5",
            "n_pairs": len(analysis),
            "excluded_pairs": len(paired) - len(analysis),
            "k5_mean": float(analysis[k5_column].mean()),
            "k10_mean": float(analysis[k10_column].mean()),
            "mean_difference": observed_mean,
            "mean_difference_ci_low": intervals["mean_difference"]["ci_low"],
            "mean_difference_ci_high": intervals["mean_difference"]["ci_high"],
            "median_difference": float(analysis.difference.median()),
            "median_difference_ci_low": intervals["median_difference"]["ci_low"],
            "median_difference_ci_high": intervals["median_difference"]["ci_high"],
            "wilcoxon_statistic": statistic,
            "p_raw": p_raw,
            "p_holm": np.nan,
            "reject_holm": False,
            "rank_biserial": intervals["rank_biserial"]["point"],
            "rank_biserial_ci_low": intervals["rank_biserial"]["ci_low"],
            "rank_biserial_ci_high": intervals["rank_biserial"]["ci_high"],
            "bootstrap_replicates": reps,
            "valid_replicates": min(v["valid_replicates"] for v in intervals.values()),
            "seed": seed,
            "status": status,
        }
        rows.append(row)
        tests.append({
            "schema_version": "phase06-statistical-test-v1",
            "family_id": row["family_id"], "family_size": 5,
            "comparison_id": comparison_id, "metric": metric,
            "system_a": "k5", "system_b": "k10", "experimental_unit": "question_id",
            "test_name": "Wilcoxon signed-rank (Pratt)", "alternative": "two-sided",
            "n_a": len(analysis), "n_b": len(analysis), "n_pairs": len(analysis),
            "statistic": statistic, "p_raw": p_raw, "p_holm": np.nan,
            "reject_holm": False, "effect_name": "matched_pairs_rank_biserial",
            "effect_value": row["rank_biserial"],
            "ci_low": row["rank_biserial_ci_low"], "ci_high": row["rank_biserial_ci_high"],
            "status": status,
        })
        for effect_name, point_key, low_key, high_key in [
            ("mean_difference", "mean_difference", "mean_difference_ci_low", "mean_difference_ci_high"),
            ("median_difference", "median_difference", "median_difference_ci_low", "median_difference_ci_high"),
            ("matched_pairs_rank_biserial", "rank_biserial", "rank_biserial_ci_low", "rank_biserial_ci_high"),
        ]:
            effects.append({
                "comparison_id": comparison_id, "metric": metric, "effect_name": effect_name,
                "effect_value": row[point_key], "ci_low": row[low_key], "ci_high": row[high_key],
                "direction": "k10_minus_k5", "experimental_unit": "question_id",
            })
    return pd.DataFrame(rows), tests, effects


def _paired_binary(
    g0: pd.DataFrame, g1: pd.DataFrame, config: Phase06Config,
    bootstrap_rows: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    merged = g0[["question_id", "fully_supported_response", "response_with_any_unsupported_claim"]].merge(
        g1[["question_id", "fully_supported_response", "response_with_any_unsupported_claim"]],
        on="question_id", suffixes=("_k5", "_k10"), validate="one_to_one",
    )
    specifications = [
        ("fully_supported_response", "fully_supported_response_k5", "fully_supported_response_k10"),
        ("response_contains_unsupported_claim", "response_with_any_unsupported_claim_k5", "response_with_any_unsupported_claim_k10"),
    ]
    rows: list[dict[str, Any]] = []
    tests: list[dict[str, Any]] = []
    effects: list[dict[str, Any]] = []
    for metric, k5_column, k10_column in specifications:
        counts = paired_transition_counts(merged[k5_column], merged[k10_column])
        complete = merged.loc[merged[k5_column].notna() & merged[k10_column].notna()].copy()
        complete["first"] = complete[k5_column].astype(bool).astype(int)
        complete["second"] = complete[k10_column].astype(bool).astype(int)
        interval = _bootstrap_one_row_per_question(
            complete[["question_id", "first", "second"]],
            {"risk_difference": lambda x: float((x.second - x.first).mean())},
            replicates=config.values["bootstrap"]["replicates"],
            seed=config.values["random_seed"],
            confidence_level=config.values["confidence_level"],
        )["risk_difference"]
        bootstrap_rows.append(_bootstrap_row(
            metric=metric, contrast="k10_minus_k5", kind="paired_risk_difference",
            interval=interval, config=config,
        ))
        statistic, p_raw = exact_mcnemar(counts)
        comparison_id = f"C_{metric}_k10_minus_k5"
        row = {
            "schema_version": "phase06-paired-binary-v1",
            "comparison_id": comparison_id,
            "family_id": "C_paired_k5_k10_binary",
            "metric": metric, "direction": "k10_minus_k5",
            **counts,
            "k5_rate": float(complete["first"].mean()),
            "k10_rate": float(complete["second"].mean()),
            "risk_difference": interval["point"],
            "risk_difference_ci_low": interval["ci_low"],
            "risk_difference_ci_high": interval["ci_high"],
            "mcnemar_statistic": statistic, "p_raw": p_raw,
            "p_holm": np.nan, "reject_holm": False,
            "bootstrap_replicates": config.values["bootstrap"]["replicates"],
            "valid_replicates": interval["valid_replicates"],
            "seed": config.values["random_seed"], "status": "ok",
        }
        rows.append(row)
        tests.append({
            "schema_version": "phase06-statistical-test-v1",
            "family_id": row["family_id"], "family_size": 2,
            "comparison_id": comparison_id, "metric": metric,
            "system_a": "k5", "system_b": "k10", "experimental_unit": "question_id",
            "test_name": "McNemar exact", "alternative": "two-sided",
            "n_a": len(complete), "n_b": len(complete), "n_pairs": len(complete),
            "statistic": statistic, "p_raw": p_raw, "p_holm": np.nan,
            "reject_holm": False, "effect_name": "paired_risk_difference",
            "effect_value": interval["point"], "ci_low": interval["ci_low"],
            "ci_high": interval["ci_high"], "status": "ok",
        })
        effects.append({
            "comparison_id": comparison_id, "metric": metric,
            "effect_name": "paired_risk_difference", "effect_value": interval["point"],
            "ci_low": interval["ci_low"], "ci_high": interval["ci_high"],
            "direction": "k10_minus_k5", "experimental_unit": "question_id",
        })
    return pd.DataFrame(rows), tests, effects


def _sufficiency_associations(
    policies: dict[int, pd.DataFrame], config: Phase06Config,
    bootstrap_rows: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    continuous = ["rouge_l_f1", "bertscore_f1", "unsupported_claim_rate"]
    rows: list[dict[str, Any]] = []
    tests: list[dict[str, Any]] = []
    effects: list[dict[str, Any]] = []
    for k, frame in sorted(policies.items()):
        if frame.question_id.duplicated().any():
            raise ValueError(f"duplicate question in k={k} sufficiency population")
        counts = frame.groupby("y_suff_final").size().to_dict()
        expected = {5: {0: 42, 1: 47}, 10: {0: 36, 1: 53}}[k]
        if counts != expected:
            raise ValueError(f"frozen k={k} sufficiency census mismatch: {counts}")
        for metric in continuous:
            analysis = frame[["question_id", "y_suff_final", metric]].dropna(subset=[metric]).copy()
            sufficient, insufficient, excluded = independent_complete_groups(
                frame, "y_suff_final", metric, 1, 0
            )

            statistics = {
                "sufficient_mean": lambda x: float(x.loc[x.y_suff_final == 1, metric].mean()),
                "insufficient_mean": lambda x: float(x.loc[x.y_suff_final == 0, metric].mean()),
                "mean_difference": lambda x: float(
                    x.loc[x.y_suff_final == 1, metric].mean()
                    - x.loc[x.y_suff_final == 0, metric].mean()
                ),
                "sufficient_median": lambda x: float(x.loc[x.y_suff_final == 1, metric].median()),
                "insufficient_median": lambda x: float(x.loc[x.y_suff_final == 0, metric].median()),
                "median_difference": lambda x: float(
                    x.loc[x.y_suff_final == 1, metric].median()
                    - x.loc[x.y_suff_final == 0, metric].median()
                ),
                "cliffs_delta": lambda x: cliffs_delta(
                    x.loc[x.y_suff_final == 1, metric], x.loc[x.y_suff_final == 0, metric]
                ),
            }
            intervals = _bootstrap_one_row_per_question(
                analysis, statistics,
                replicates=config.values["bootstrap"]["replicates"],
                seed=config.values["random_seed"],
                confidence_level=config.values["confidence_level"],
            )
            for name, interval in intervals.items():
                bootstrap_rows.append(_bootstrap_row(
                    metric=metric, contrast=f"k{k}_sufficient_minus_insufficient", kind=name,
                    interval=interval, config=config,
                ))
            mw = stats.mannwhitneyu(sufficient, insufficient, alternative="two-sided", method="auto")
            comparison_id = f"B_k{k}_{metric}_sufficient_minus_insufficient"
            row = {
                "schema_version": "phase06-sufficiency-association-v1",
                "comparison_id": comparison_id, "family_id": "B_sufficiency_association",
                "k": k, "metric": metric, "outcome_type": "continuous",
                "direction": "sufficient_minus_insufficient",
                "sufficient_n": len(sufficient), "insufficient_n": len(insufficient),
                "excluded_n": excluded,
                "sufficient_mean": intervals["sufficient_mean"]["point"],
                "sufficient_mean_ci_low": intervals["sufficient_mean"]["ci_low"],
                "sufficient_mean_ci_high": intervals["sufficient_mean"]["ci_high"],
                "insufficient_mean": intervals["insufficient_mean"]["point"],
                "insufficient_mean_ci_low": intervals["insufficient_mean"]["ci_low"],
                "insufficient_mean_ci_high": intervals["insufficient_mean"]["ci_high"],
                "mean_difference": intervals["mean_difference"]["point"],
                "mean_difference_ci_low": intervals["mean_difference"]["ci_low"],
                "mean_difference_ci_high": intervals["mean_difference"]["ci_high"],
                "sufficient_median": intervals["sufficient_median"]["point"],
                "insufficient_median": intervals["insufficient_median"]["point"],
                "median_difference": intervals["median_difference"]["point"],
                "median_difference_ci_low": intervals["median_difference"]["ci_low"],
                "median_difference_ci_high": intervals["median_difference"]["ci_high"],
                "test_name": "Mann-Whitney U", "statistic": float(mw.statistic),
                "p_raw": float(mw.pvalue), "p_holm": np.nan, "reject_holm": False,
                "cliffs_delta": intervals["cliffs_delta"]["point"],
                "cliffs_delta_ci_low": intervals["cliffs_delta"]["ci_low"],
                "cliffs_delta_ci_high": intervals["cliffs_delta"]["ci_high"],
                "sufficient_rate": np.nan, "insufficient_rate": np.nan,
                "risk_difference": np.nan, "risk_difference_ci_low": np.nan,
                "risk_difference_ci_high": np.nan, "risk_ratio": np.nan, "odds_ratio": np.nan,
                "bootstrap_replicates": config.values["bootstrap"]["replicates"],
                "valid_replicates": min(v["valid_replicates"] for v in intervals.values()),
                "seed": config.values["random_seed"], "status": "associational",
            }
            rows.append(row)
            tests.append({
                "schema_version": "phase06-statistical-test-v1",
                "family_id": row["family_id"], "family_size": 8,
                "comparison_id": comparison_id, "metric": metric,
                "system_a": f"k{k}_insufficient", "system_b": f"k{k}_sufficient",
                "experimental_unit": "question_id", "test_name": "Mann-Whitney U",
                "alternative": "two-sided", "n_a": len(insufficient), "n_b": len(sufficient),
                "n_pairs": np.nan, "statistic": row["statistic"], "p_raw": row["p_raw"],
                "p_holm": np.nan, "reject_holm": False, "effect_name": "cliffs_delta",
                "effect_value": row["cliffs_delta"], "ci_low": row["cliffs_delta_ci_low"],
                "ci_high": row["cliffs_delta_ci_high"], "status": "associational",
            })
            for effect_name, point, low, high in [
                ("mean_difference", row["mean_difference"], row["mean_difference_ci_low"], row["mean_difference_ci_high"]),
                ("median_difference", row["median_difference"], row["median_difference_ci_low"], row["median_difference_ci_high"]),
                ("cliffs_delta", row["cliffs_delta"], row["cliffs_delta_ci_low"], row["cliffs_delta_ci_high"]),
            ]:
                effects.append({
                    "comparison_id": comparison_id, "metric": metric, "effect_name": effect_name,
                    "effect_value": point, "ci_low": low, "ci_high": high,
                    "direction": "sufficient_minus_insufficient",
                    "experimental_unit": "question_id",
                })

        metric = "fully_supported_response"
        analysis = frame[["question_id", "y_suff_final", metric]].dropna(subset=[metric]).copy()
        analysis[metric] = analysis[metric].astype(bool).astype(int)
        sufficient = analysis.loc[analysis.y_suff_final == 1, metric]
        insufficient = analysis.loc[analysis.y_suff_final == 0, metric]
        table = np.asarray([
            [int(sufficient.sum()), int(len(sufficient) - sufficient.sum())],
            [int(insufficient.sum()), int(len(insufficient) - insufficient.sum())],
        ])
        fisher = stats.fisher_exact(table, alternative="two-sided")
        interval = _bootstrap_one_row_per_question(
            analysis,
            {"risk_difference": lambda x: float(
                x.loc[x.y_suff_final == 1, metric].mean()
                - x.loc[x.y_suff_final == 0, metric].mean()
            )},
            replicates=config.values["bootstrap"]["replicates"],
            seed=config.values["random_seed"],
            confidence_level=config.values["confidence_level"],
        )["risk_difference"]
        bootstrap_rows.append(_bootstrap_row(
            metric=metric, contrast=f"k{k}_sufficient_minus_insufficient",
            kind="risk_difference", interval=interval, config=config,
        ))
        sufficient_rate = float(sufficient.mean())
        insufficient_rate = float(insufficient.mean())
        risk_ratio = sufficient_rate / insufficient_rate if insufficient_rate > 0 else np.nan
        odds_ratio = float(fisher.statistic) if (table > 0).all() else np.nan
        comparison_id = f"B_k{k}_{metric}_sufficient_minus_insufficient"
        row = {
            "schema_version": "phase06-sufficiency-association-v1",
            "comparison_id": comparison_id, "family_id": "B_sufficiency_association",
            "k": k, "metric": metric, "outcome_type": "binary",
            "direction": "sufficient_minus_insufficient",
            "sufficient_n": len(sufficient), "insufficient_n": len(insufficient),
            "excluded_n": len(frame) - len(analysis),
            "sufficient_mean": np.nan, "sufficient_mean_ci_low": np.nan,
            "sufficient_mean_ci_high": np.nan, "insufficient_mean": np.nan,
            "insufficient_mean_ci_low": np.nan, "insufficient_mean_ci_high": np.nan,
            "mean_difference": np.nan, "mean_difference_ci_low": np.nan,
            "mean_difference_ci_high": np.nan, "sufficient_median": np.nan,
            "insufficient_median": np.nan, "median_difference": np.nan,
            "median_difference_ci_low": np.nan, "median_difference_ci_high": np.nan,
            "test_name": "Fisher exact", "statistic": float(fisher.statistic),
            "p_raw": float(fisher.pvalue), "p_holm": np.nan, "reject_holm": False,
            "cliffs_delta": np.nan, "cliffs_delta_ci_low": np.nan,
            "cliffs_delta_ci_high": np.nan,
            "sufficient_rate": sufficient_rate, "insufficient_rate": insufficient_rate,
            "risk_difference": interval["point"],
            "risk_difference_ci_low": interval["ci_low"],
            "risk_difference_ci_high": interval["ci_high"],
            "risk_ratio": risk_ratio, "odds_ratio": odds_ratio,
            "bootstrap_replicates": config.values["bootstrap"]["replicates"],
            "valid_replicates": interval["valid_replicates"],
            "seed": config.values["random_seed"], "status": "associational",
        }
        rows.append(row)
        tests.append({
            "schema_version": "phase06-statistical-test-v1",
            "family_id": row["family_id"], "family_size": 8,
            "comparison_id": comparison_id, "metric": metric,
            "system_a": f"k{k}_insufficient", "system_b": f"k{k}_sufficient",
            "experimental_unit": "question_id", "test_name": "Fisher exact",
            "alternative": "two-sided", "n_a": len(insufficient), "n_b": len(sufficient),
            "n_pairs": np.nan, "statistic": row["statistic"], "p_raw": row["p_raw"],
            "p_holm": np.nan, "reject_holm": False, "effect_name": "risk_difference",
            "effect_value": row["risk_difference"], "ci_low": row["risk_difference_ci_low"],
            "ci_high": row["risk_difference_ci_high"], "status": "associational",
        })
        for effect_name, value, low, high in [
            ("risk_difference", interval["point"], interval["ci_low"], interval["ci_high"]),
            ("risk_ratio", risk_ratio, np.nan, np.nan),
            ("odds_ratio", odds_ratio, np.nan, np.nan),
        ]:
            effects.append({
                "comparison_id": comparison_id, "metric": metric, "effect_name": effect_name,
                "effect_value": value, "ci_low": low, "ci_high": high,
                "direction": "sufficient_minus_insufficient",
                "experimental_unit": "question_id",
            })
    return pd.DataFrame(rows), tests, effects


def _apply_holm(
    tests: pd.DataFrame,
    detail_frames: list[pd.DataFrame],
    *,
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.DataFrame]]:
    tests = tests.copy()
    correction_rows: list[dict[str, Any]] = []
    for family, indices in tests.groupby("family_id", sort=True).groups.items():
        family_indices = list(indices)
        adjusted, rejected = holm_adjust(tests.loc[family_indices, "p_raw"].to_numpy(), alpha)
        tests.loc[family_indices, "p_holm"] = adjusted
        tests.loc[family_indices, "reject_holm"] = rejected
        ordered = tests.loc[family_indices].sort_values(["p_raw", "comparison_id"], kind="stable")
        for rank, (_, row) in enumerate(ordered.iterrows(), start=1):
            corrected = tests.loc[tests.comparison_id == row.comparison_id].iloc[0]
            correction_rows.append({
                "schema_version": "phase06-holm-v1", "family_id": family,
                "comparison_id": row.comparison_id, "rank": rank,
                "family_size": len(family_indices), "p_raw": float(row.p_raw),
                "p_holm": float(corrected.p_holm), "alpha": alpha,
                "reject_holm": bool(corrected.reject_holm),
            })
    updated: list[pd.DataFrame] = []
    lookup = tests.set_index("comparison_id")[["p_holm", "reject_holm"]]
    for detail in detail_frames:
        detail = detail.copy()
        detail["p_holm"] = detail.comparison_id.map(lookup.p_holm)
        detail["reject_holm"] = detail.comparison_id.map(lookup.reject_holm).astype(bool)
        updated.append(detail)
    return tests, pd.DataFrame(correction_rows), updated


def _policy_intervals(
    policy_frames: dict[str, pd.DataFrame], config: Phase06Config,
    bootstrap_rows: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    proportion_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    reps = config.values["bootstrap"]["replicates"]
    seed = config.values["random_seed"]
    confidence = config.values["confidence_level"]
    minimum = config.values["tiny_sample_guard"]["minimum_answered_n_for_quality_bootstrap"]
    for policy_id, frame in sorted(policy_frames.items()):
        if len(frame) != 89 or frame.question_id.nunique() != 89:
            raise ValueError(f"policy {policy_id} does not contain 89 unique questions")
        analysis = frame.copy()
        analysis["answered_indicator"] = analysis.answered.astype(bool).astype(int)
        answered = analysis.answered.astype(bool)
        if analysis.loc[answered, "fully_supported_response"].isna().any():
            raise ValueError(f"answered {policy_id} rows have missing fully-supported outcome")
        if analysis.loc[answered, "response_with_any_unsupported_claim"].isna().any():
            raise ValueError(f"answered {policy_id} rows have missing unsupported outcome")
        analysis["grounded_indicator"] = 0
        analysis.loc[answered, "grounded_indicator"] = (
            analysis.loc[answered, "fully_supported_response"].astype(bool).astype(int)
        )
        analysis["unsupported_exposure_indicator"] = 0
        analysis.loc[answered, "unsupported_exposure_indicator"] = (
            analysis.loc[answered, "response_with_any_unsupported_claim"].astype(bool).astype(int)
        )
        names = {
            "answer_coverage": "answered_indicator",
            "grounded_answer_yield": "grounded_indicator",
            "unsupported_answer_population_rate": "unsupported_exposure_indicator",
        }
        intervals = _bootstrap_one_row_per_question(
            analysis[["question_id", *names.values()]],
            {name: (lambda x, column=column: float(x[column].mean()))
             for name, column in names.items()},
            replicates=reps, seed=seed, confidence_level=confidence,
        )
        for metric, column in names.items():
            interval = intervals[metric]
            numerator = int(analysis[column].sum())
            proportion_rows.append({
                "schema_version": "phase06-policy-ci-v1", "policy_id": policy_id,
                "metric": metric, "numerator": numerator, "denominator": len(analysis),
                "point_estimate": interval["point"], "ci_low": interval["ci_low"],
                "ci_high": interval["ci_high"], "confidence_level": confidence,
                "method": "question_id_cluster_percentile_bootstrap",
                "requested_replicates": reps, "valid_replicates": interval["valid_replicates"],
                "seed": seed, "status": "ok",
            })
            bootstrap_rows.append(_bootstrap_row(
                metric=metric, contrast=policy_id, kind="policy_proportion",
                interval=interval, config=config,
            ))

        answered_frame = analysis.loc[answered].copy()
        quality_metrics = [
            "rouge_l_f1", "bertscore_f1", "unsupported_claim_rate",
            "mean_claim_support_score", "output_token_count",
        ]
        for metric in quality_metrics:
            eligible = answered_frame[["question_id", metric]].dropna(subset=[metric]).copy()
            if len(eligible) < minimum:
                quality_rows.append({
                    "schema_version": "phase06-policy-quality-ci-v1", "policy_id": policy_id,
                    "metric": metric, "answered_eligible_n": len(eligible),
                    "mean": float(eligible[metric].mean()) if len(eligible) else np.nan,
                    "mean_ci_low": np.nan, "mean_ci_high": np.nan,
                    "median": float(eligible[metric].median()) if len(eligible) else np.nan,
                    "median_ci_low": np.nan, "median_ci_high": np.nan,
                    "minimum_n": minimum, "status": "tiny_sample_guard_no_quality_inference",
                })
                continue
            quality_intervals = _bootstrap_one_row_per_question(
                eligible,
                {
                    "mean": lambda x: float(x[metric].mean()),
                    "median": lambda x: float(x[metric].median()),
                },
                replicates=reps, seed=seed, confidence_level=confidence,
            )
            quality_rows.append({
                "schema_version": "phase06-policy-quality-ci-v1", "policy_id": policy_id,
                "metric": metric, "answered_eligible_n": len(eligible),
                "mean": quality_intervals["mean"]["point"],
                "mean_ci_low": quality_intervals["mean"]["ci_low"],
                "mean_ci_high": quality_intervals["mean"]["ci_high"],
                "median": quality_intervals["median"]["point"],
                "median_ci_low": quality_intervals["median"]["ci_low"],
                "median_ci_high": quality_intervals["median"]["ci_high"],
                "minimum_n": minimum, "status": "ok",
            })
            for kind, interval in quality_intervals.items():
                bootstrap_rows.append(_bootstrap_row(
                    metric=metric, contrast=policy_id, kind=f"policy_quality_{kind}",
                    interval=interval, config=config,
                ))
    return pd.DataFrame(proportion_rows), pd.DataFrame(quality_rows)


def _rq_matrix(config: Phase06Config) -> pd.DataFrame:
    rows = []
    for rq in config.values["research_questions"]:
        rows.append({
            "rq_id": rq["rq_id"], "frozen_wording": rq["frozen_wording"],
            "proposal_intent": rq["proposal_intent"],
            "operational_mapping": rq["operational_mapping"],
            "datasets_json": json.dumps(rq["datasets"], ensure_ascii=False),
            "experimental_unit": rq["experimental_unit"],
            "outcomes_json": json.dumps(rq["outcomes"], ensure_ascii=False),
            "comparison": rq["comparisons"], "statistical_test": rq["statistical_test"],
            "effect_size": rq["effect_size"], "confidence_interval": rq["confidence_interval"],
            "table_ids_json": json.dumps(rq["table_ids"]),
            "figure_ids_json": json.dumps(rq["figure_ids"]),
        })
    return pd.DataFrame(rows)


def _table_artifacts(
    root: Path, paired_continuous: pd.DataFrame, paired_binary: pd.DataFrame,
    association: pd.DataFrame, tests: pd.DataFrame, policy_ci: pd.DataFrame,
    policy_quality_ci: pd.DataFrame,
) -> list[dict[str, Any]]:
    tables_dir = root / "artifacts/tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((root / "artifacts/data/dataset_metadata.json").read_text(encoding="utf-8"))
    feature_manifest = json.loads((root / "artifacts/results/phase04_feature_manifest.json").read_text(encoding="utf-8"))
    rag_audit = json.loads((root / "artifacts/results/phase05_ragtruth_schema_alignment_audit.json").read_text(encoding="utf-8"))
    split = metadata["techqa"]["split"]["achieved"]
    table1_rows = []
    for split_name in ["train", "validation", "test"]:
        values = split[split_name]
        table1_rows.append({
            "dataset": "TechQA", "population": "frozen benchmark census",
            "split": split_name if split_name != "test" else "test (sealed; census only)",
            "questions_or_sources": values["total"], "observations": values["total"],
            "positive_or_answerable": values["answerable"],
            "negative_or_impossible": values["impossible"],
            "notes": "No TechQA TEST outcomes accessed" if split_name == "test" else "group-aware split",
        })
    for split_name in ["train", "validation"]:
        values = feature_manifest["splits"][split_name]
        table1_rows.append({
            "dataset": "TechQA", "population": "Phase 3/4 PRIMARY context-sufficiency",
            "split": split_name, "questions_or_sources": values["questions"],
            "observations": values["conditions"], "positive_or_answerable": values["positive"],
            "negative_or_impossible": values["negative"],
            "notes": "retrieval-conditioned y_suff_final",
        })
    table1_rows.extend([
        {"dataset": "TechQA", "population": "Phase 5 generation states", "split": "validation",
         "questions_or_sources": 89, "observations": 178, "positive_or_answerable": np.nan,
         "negative_or_impossible": np.nan, "notes": "hybrid k=5 and k=10"},
        {"dataset": "RAGTruth QA", "population": "supporting evaluator validation", "split": "official train",
         "questions_or_sources": rag_audit["qa_train_sources"], "observations": rag_audit["qa_train_responses"],
         "positive_or_answerable": np.nan, "negative_or_impossible": np.nan,
         "notes": "threshold selection only"},
        {"dataset": "RAGTruth QA", "population": "supporting evaluator validation", "split": "official test",
         "questions_or_sources": rag_audit["qa_test_sources"], "observations": rag_audit["qa_test_responses"],
         "positive_or_answerable": np.nan, "negative_or_impossible": np.nan,
         "notes": "not the sealed TechQA final TEST"},
    ])
    table1 = pd.DataFrame(table1_rows)

    table2 = pd.read_csv(root / "artifacts/results/phase02_metrics_train_validation.csv")
    table2 = table2.loc[table2.split == "validation"].reset_index(drop=True)

    initial = json.loads((root / "artifacts/results/phase03_human_validation_results.json").read_text(encoding="utf-8"))
    final = json.loads((root / "artifacts/results/phase03_final_confirmation_evaluation.json").read_text(encoding="utf-8"))
    population = json.loads((root / "artifacts/results/phase03_final_target_population_manifest.json").read_text(encoding="utf-8"))
    first = initial["annotator_results"][0]
    table3 = pd.DataFrame([
        {"section": "strict proxy development audit", "metric": "sample size", "estimate": first["completed_count"], "ci_low": np.nan, "ci_high": np.nan, "notes": "equal-allocation stratified audit"},
        {"section": "strict proxy development audit", "metric": "precision", "estimate": first["automatic_sufficient_precision"], "ci_low": first["automatic_sufficient_precision_ci_low_95"], "ci_high": first["automatic_sufficient_precision_ci_high_95"], "notes": "strict positives"},
        {"section": "strict proxy development audit", "metric": "raw F1", "estimate": first["sample_unweighted_metrics"]["f1"], "ci_low": np.nan, "ci_high": np.nan, "notes": "descriptive sample only"},
        {"section": "strict proxy development audit", "metric": "weighted F1", "estimate": first["prevalence_weighted_metrics"]["f1"], "ci_low": np.nan, "ci_high": np.nan, "notes": "failed original 0.85 gate"},
        {"section": "final independent confirmation", "metric": "sample size", "estimate": final["sample_size"], "ci_low": np.nan, "ci_high": np.nan, "notes": "one human with AI-assisted pre-key adjudication"},
        {"section": "final independent confirmation", "metric": "precision", "estimate": final["precision"], "ci_low": final["precision_wilson_95"]["lower"], "ci_high": final["precision_wilson_95"]["upper"], "notes": "Wilson CI"},
        {"section": "final independent confirmation", "metric": "recall", "estimate": final["recall"], "ci_low": np.nan, "ci_high": np.nan, "notes": ""},
        {"section": "final independent confirmation", "metric": "raw F1", "estimate": final["f1"], "ci_low": np.nan, "ci_high": np.nan, "notes": ""},
        {"section": "final independent confirmation", "metric": "accuracy", "estimate": final["accuracy"], "ci_low": np.nan, "ci_high": np.nan, "notes": ""},
        {"section": "final independent confirmation", "metric": "weighted F1", "estimate": final["weighted_f1"], "ci_low": np.nan, "ci_high": np.nan, "notes": "passed frozen 0.85 gate"},
        {"section": "final target population", "metric": "eligible questions", "estimate": population["eligible_questions"], "ci_low": np.nan, "ci_high": np.nan, "notes": "TRAIN+VALIDATION"},
        {"section": "final target population", "metric": "eligible conditions", "estimate": population["eligible_conditions"], "ci_low": np.nan, "ci_high": np.nan, "notes": "PRIMARY"},
        {"section": "final target population", "metric": "positive conditions", "estimate": population["final_positive_conditions"], "ci_low": np.nan, "ci_high": np.nan, "notes": ""},
        {"section": "final target population", "metric": "negative conditions", "estimate": population["final_negative_conditions"], "ci_low": np.nan, "ci_high": np.nan, "notes": ""},
    ])

    table4 = pd.read_csv(root / "artifacts/results/phase04_model_validation_metrics.csv")
    calibration = pd.read_csv(root / "artifacts/results/phase04_rf_calibration_comparison.csv")
    calibration["section"] = "calibration"
    calibration["item"] = calibration.model + ":" + calibration.probability_method
    ablation = pd.read_csv(root / "artifacts/results/phase04_feature_ablation.csv")
    ablation_table = pd.DataFrame({
        "section": "feature_ablation", "item": ablation.ablation,
        "auroc": ablation.auroc, "auprc": ablation.auprc, "f1": ablation.f1,
        "brier": np.nan, "ece": np.nan, "selected": ablation.ablation.eq("D"),
        "notes": ablation.families,
    })
    calibration_table = pd.DataFrame({
        "section": calibration.section, "item": calibration.item,
        "auroc": calibration.auroc, "auprc": calibration.auprc, "f1": calibration.f1,
        "brier": calibration.brier, "ece": calibration.ece,
        "selected": calibration.selected_calibration, "notes": "frozen validation comparison",
    })
    table5 = pd.concat([calibration_table, ablation_table], ignore_index=True)

    risk = pd.read_csv(root / "artifacts/results/phase04_risk_operating_points.csv")
    adaptive = pd.read_csv(root / "artifacts/results/phase04_policy_operating_points.csv")
    baselines = pd.read_csv(root / "artifacts/results/phase04_policy_baselines.csv")
    aurc = json.loads((root / "artifacts/results/phase04_aurc.json").read_text(encoding="utf-8"))["aurc"]
    table6_rows = []
    for row in risk.itertuples(index=False):
        table6_rows.append({
            "section": "fixed-condition risk-coverage", "system": "selected Random Forest",
            "risk_constraint": row.risk_constraint, "t_low": np.nan, "t_high_or_threshold": row.threshold,
            "coverage": row.coverage, "observed_selective_risk": row.selective_risk,
            "retrieval_expansion_rate": np.nan, "false_abstention_rate": np.nan,
            "mean_retrieved_k": np.nan, "aurc": aurc,
        })
    for row in adaptive.itertuples(index=False):
        table6_rows.append({
            "section": "adaptive policy", "system": "P/G adaptive k5-to-k10",
            "risk_constraint": row.risk_constraint, "t_low": row.t_low,
            "t_high_or_threshold": row.t_high, "coverage": row.final_answer_coverage,
            "observed_selective_risk": row.final_selective_risk,
            "retrieval_expansion_rate": row.retrieval_expansion_rate,
            "false_abstention_rate": row.false_abstention_rate,
            "mean_retrieved_k": row.mean_retrieved_k, "aurc": np.nan,
        })
    for row in baselines.itertuples(index=False):
        if row.policy not in {"P0_always_answer_hybrid_k5", "P2_always_retrieve_hybrid_k10_then_two_way"}:
            continue
        table6_rows.append({
            "section": "policy baseline", "system": row.policy,
            "risk_constraint": row.risk_constraint, "t_low": np.nan,
            "t_high_or_threshold": row.threshold, "coverage": row.final_answer_coverage,
            "observed_selective_risk": row.final_selective_risk,
            "retrieval_expansion_rate": row.retrieval_expansion_rate,
            "false_abstention_rate": row.false_abstention_rate,
            "mean_retrieved_k": row.mean_retrieved_k, "aurc": np.nan,
        })
    table6 = pd.DataFrame(table6_rows)

    rag_metrics = pd.read_csv(root / "artifacts/results/phase05_ragtruth_test_metrics.csv")
    rag_ci = pd.read_csv(root / "artifacts/results/phase05_ragtruth_bootstrap_intervals.csv")
    table7 = rag_ci.merge(
        rag_metrics[["population", "level", "eligible_count", "n"]],
        on=["population", "level"], validate="many_to_one",
    )
    table7 = table7[["population", "level", "metric", "eligible_count", "n", "point_estimate", "ci_low", "ci_high", "resampling_unit", "requested_replicates"]]

    table8_cont = paired_continuous.copy()
    table8_cont.insert(0, "analysis_type", "paired_continuous")
    table8_bin = paired_binary.copy()
    table8_bin.insert(0, "analysis_type", "paired_binary")
    common = sorted(set(table8_cont.columns).union(table8_bin.columns))
    table8 = pd.concat([table8_cont.reindex(columns=common), table8_bin.reindex(columns=common)], ignore_index=True)
    table9 = association.copy()

    policy_summary = pd.read_csv(root / "artifacts/results/phase05_policy_generation_comparison.csv")
    ci_wide = policy_ci.pivot(index="policy_id", columns="metric", values=["ci_low", "ci_high"])
    ci_wide.columns = [f"{metric}_{bound}" for bound, metric in ci_wide.columns]
    ci_wide = ci_wide.reset_index()
    table10 = policy_summary.merge(ci_wide, on="policy_id", validate="one_to_one")
    g2_guard = policy_quality_ci.loc[policy_quality_ci.policy_id == "G2", "status"].unique().tolist()
    table10["quality_interval_status"] = np.where(
        table10.policy_id.eq("G2"), g2_guard[0], "reported separately in policy quality interval artifact"
    )
    table11 = tests.copy()

    specs = [
        (1, "dataset_population_split_census", "Table 1 — Dataset, population, and split census", table1),
        (2, "controlled_retrieval_results", "Table 2 — Controlled retrieval results", table2),
        (3, "phase03_human_target_validation", "Table 3 — Phase 3 human and final-target validation", table3),
        (4, "phase04_classifier_comparison", "Table 4 — Phase 4 classifier comparison", table4),
        (5, "calibration_feature_ablation", "Table 5 — Calibration and feature ablation", table5),
        (6, "risk_coverage_policy_comparison", "Table 6 — Risk-coverage and policy comparison", table6),
        (7, "ragtruth_evaluator_validation", "Table 7 — RAGTruth evaluator validation", table7),
        (8, "paired_k5_k10_generation_grounding", "Table 8 — Paired k5/k10 generation and grounding", table8),
        (9, "sufficiency_association_outcomes", "Table 9 — Sufficiency-association outcomes", table9),
        (10, "generation_policy_comparison", "Table 10 — G0-G3 safety, quality, and coverage", table10),
        (11, "inferential_test_summary", "Table 11 — Inferential statistical test summary", table11),
    ]
    manifest: list[dict[str, Any]] = []
    for number, stem, title, frame in specs:
        base = tables_dir / f"phase06_table{number:02d}_{stem}"
        write_table(base.with_suffix(".csv"), base.with_suffix(".md"), frame, title)
        manifest.append({
            "table_id": f"T{number:02d}", "title": title, "rows": len(frame),
            "csv_path": base.with_suffix(".csv").relative_to(root).as_posix(),
            "csv_sha256": sha256_file(base.with_suffix(".csv")),
            "markdown_path": base.with_suffix(".md").relative_to(root).as_posix(),
            "markdown_sha256": sha256_file(base.with_suffix(".md")),
        })
    write_json_atomic(tables_dir / "phase06_table_manifest.json", {
        "schema_version": "phase06-table-manifest-v1", "tables": manifest,
        "generated_from_canonical_artifacts": True,
    })
    return manifest


def _result_summary_markdown(
    tests: pd.DataFrame, paired: pd.DataFrame, binary: pd.DataFrame,
    association: pd.DataFrame, policy_ci: pd.DataFrame,
) -> str:
    def fmt(value: Any, digits: int = 6) -> str:
        return "NA" if pd.isna(value) else f"{float(value):.{digits}f}"

    lines = [
        "# Phase 6 Results Summary", "",
        "This is a post-analysis evidence artifact generated from frozen Phase 1-5 inputs under the pre-analysis Phase 6 configuration. TechQA TEST remained sealed and Phase 7 was not started.", "",
        "## RQ1 — Retrieval and retrieved-context sufficiency", "",
        "Frozen Phase 2 validation retrieval results are reported in Table 2 and Figure 1; Phase 3 target validation is reported in Table 3. Document retrieval and retrieved-context sufficiency remain distinct constructs. No new aggregate-only interval or post-hoc retriever test was fabricated in Phase 6.", "",
        "## RQ2 — Prediction and calibration", "",
        "The selected uncalibrated Random Forest had AUROC 0.718833 (95% frozen question-bootstrap CI [0.645963, 0.787351]), AUPRC 0.674558 [0.539924, 0.785402], F1 0.507692 [0.394183, 0.608769], Brier score 0.213815 [0.189549, 0.238542], and ECE 0.063370. Tables 4-5 and Figures 2A-4 consolidate the classifier, calibration, ablation, reliability, and importance evidence. No post-hoc classifier-family p-value was introduced.", "",
        "## RQ3 — Policy safety and coverage", "",
        "The frozen fixed-condition operating points were: 5% constraint, coverage 0.027154 and observed risk 0.034483; 10%, coverage 0.043071 and risk 0.086957; 20%, coverage 0.112360 and risk 0.200000. AURC was 0.3784805395. Table 6 and Figure 5 show the full trade-off.", "",
        "For G0-G3, question-bootstrap policy intervals are in Table 10 and the policy CI artifact. G2 answered only 2/89 questions; its quality is descriptive only and zero observed unsupported responses does not establish zero true risk. Under the frozen validation protocol, selective answering reduced the observed rate of unsupported answers at substantial cost to coverage.", "",
        "## RQ4 — Generation grounding and evaluator reliability", "",
        "### Paired k10 minus k5 outcomes", "",
    ]
    for row in paired.itertuples(index=False):
        wording = "statistically detectable after Holm correction" if row.reject_holm else "did not provide evidence of a statistically detectable difference after Holm correction"
        lines.append(
            f"- `{row.metric}`: mean difference {fmt(row.mean_difference)} (95% CI [{fmt(row.mean_difference_ci_low)}, {fmt(row.mean_difference_ci_high)}]); median difference {fmt(row.median_difference)}; Wilcoxon W={fmt(row.wilcoxon_statistic)}, raw p={fmt(row.p_raw)}, Holm p={fmt(row.p_holm)}; rank-biserial={fmt(row.rank_biserial)}. This {wording}."
        )
    lines.extend(["", "### Paired binary outcomes", ""])
    for row in binary.itertuples(index=False):
        wording = "statistically detectable after Holm correction" if row.reject_holm else "did not provide evidence of a statistically detectable difference after Holm correction"
        lines.append(
            f"- `{row.metric}`: k5 rate {fmt(row.k5_rate)}, k10 rate {fmt(row.k10_rate)}, paired risk difference {fmt(row.risk_difference)} (95% CI [{fmt(row.risk_difference_ci_low)}, {fmt(row.risk_difference_ci_high)}]); exact McNemar statistic={fmt(row.mcnemar_statistic)}, raw p={fmt(row.p_raw)}, Holm p={fmt(row.p_holm)}. This {wording}."
        )
    lines.extend(["", "### Context-sufficiency associations", ""])
    for row in association.itertuples(index=False):
        if row.outcome_type == "continuous":
            lines.append(
                f"- k={int(row.k)} `{row.metric}`: sufficient-minus-insufficient mean difference {fmt(row.mean_difference)} (95% CI [{fmt(row.mean_difference_ci_low)}, {fmt(row.mean_difference_ci_high)}]); Cliff's delta {fmt(row.cliffs_delta)} [{fmt(row.cliffs_delta_ci_low)}, {fmt(row.cliffs_delta_ci_high)}]; raw p={fmt(row.p_raw)}, Holm p={fmt(row.p_holm)}."
            )
        else:
            lines.append(
                f"- k={int(row.k)} fully-supported response: sufficient rate {fmt(row.sufficient_rate)}, insufficient rate {fmt(row.insufficient_rate)}, risk difference {fmt(row.risk_difference)} [{fmt(row.risk_difference_ci_low)}, {fmt(row.risk_difference_ci_high)}], risk ratio {fmt(row.risk_ratio)}, odds ratio {fmt(row.odds_ratio)}, raw p={fmt(row.p_raw)}, Holm p={fmt(row.p_holm)}."
            )
    significant = tests.loc[tests.reject_holm.astype(bool), "comparison_id"].tolist()
    lines.extend([
        "", "These sufficient-versus-insufficient comparisons are associational: the frozen Phase 3 target and generation outcomes are measurements on the same retrieval-conditioned context states. They do not establish causation.", "",
        "The binary unsupported-claim rate and continuous mean support can move in apparently contradictory directions because threshold crossings and average score shifts are different estimands. The result is retained rather than reconciled through post-hoc retuning.", "",
        "### Grounding evaluator", "",
        "On good-quality RAGTruth TEST, claim-level precision/recall/F1/AUROC/AUPRC were 0.2280/0.5753/0.3266/0.8030/0.2286; response-level values were 0.2756/0.7750/0.4066/0.7282/0.4056. Frozen source-bootstrap intervals appear in Table 7. Ranking discrimination is meaningful, but binary precision is low; the evaluator is an imperfect grounding proxy and every TechQA grounding result inherits that limitation.", "",
        "## Holm-adjusted conclusions", "",
        ("Adjusted-significant hypotheses: " + ", ".join(f"`{value}`" for value in significant) + ".") if significant else "No predeclared hypothesis remained significant after within-family Holm correction.", "",
        "## Limitations", "",
        "- The TechQA analysis is VALIDATION-only with 89 questions; confidence intervals can remain wide.",
        "- Phase 3 final confirmation used one human annotator with AI-assisted pre-key adjudication, not two independent annotators.",
        "- RAGTruth validation supports ranking discrimination but exposes low precision for binary unsupported classifications.",
        "- ROUGE-L and BERTScore measure reference similarity, not grounding.",
        "- The G2 quality population contains only two answered questions.",
        "- Context sufficiency, model confidence, grounding, and authoritative hallucination truth remain distinct.",
        "- Observed validation risk is not a production safety guarantee.", "",
        "Tables 1-11 and Figures 1-8 are the evidence package for later thesis writing; no full thesis chapter is drafted here.", "",
    ])
    return "\n".join(lines)


def _phase06_manifest_paths(root: Path) -> list[Path]:
    fixed = [
        Path(".gitattributes"), Path("configs/phase06_statistics.json"),
        Path("artifacts/results/.gitignore"), Path("artifacts/figures/.gitignore"),
        Path("docs/PHASE_06_RESEARCH_DECISIONS.md"), Path("docs/PHASE_06_EXECUTION.md"),
        Path("docs/PHASE_06_ARTIFACT_SCHEMAS.md"), Path("docs/PHASE_06_RESULTS_SUMMARY.md"),
        Path("scripts/run_phase06_statistics.py"), Path("scripts/check_phase06.py"),
        Path("tests/test_phase06_statistics.py"),
    ]
    fixed.extend(sorted(Path("src/answerability_rag/statistics").glob("*.py")))
    fixed.extend(sorted(
        path for path in Path("artifacts/results").glob("phase06_*")
        if path.name != "phase06_artifact_manifest.json"
    ))
    fixed.extend(sorted(Path("artifacts/tables").glob("phase06_*")))
    fixed.extend(sorted(Path("artifacts/figures").glob("phase06_*")))
    unique = {path.as_posix(): path for path in fixed if (root / path).is_file()}
    return [unique[key] for key in sorted(unique)]


def run(root: Path, config_path: Path) -> dict[str, Any]:
    config = Phase06Config.load(config_path, root)
    upstream = verify_upstream(root, config)
    results = root / "artifacts/results"
    paired = pq.read_table(results / "phase05_paired_k5_k10.parquet").to_pandas()
    policy_frames = {
        policy: pq.read_table(results / f"phase05_policy_{policy}.parquet").to_pandas()
        for policy in ["G0", "G1", "G2", "G3"]
    }
    bootstrap_rows: list[dict[str, Any]] = []
    paired_continuous, tests_a, effects_a = _paired_continuous(paired, config, bootstrap_rows)
    paired_binary, tests_c, effects_c = _paired_binary(
        policy_frames["G0"], policy_frames["G1"], config, bootstrap_rows
    )
    association, tests_b, effects_b = _sufficiency_associations(
        {5: policy_frames["G0"], 10: policy_frames["G1"]}, config, bootstrap_rows
    )
    tests = pd.DataFrame(tests_a + tests_b + tests_c)
    tests, holm, detail = _apply_holm(
        tests, [paired_continuous, association, paired_binary], alpha=config.values["alpha"]
    )
    paired_continuous, association, paired_binary = detail
    effects = pd.DataFrame(effects_a + effects_b + effects_c)
    policy_ci, policy_quality_ci = _policy_intervals(policy_frames, config, bootstrap_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    rq_matrix = _rq_matrix(config)

    outputs = {
        "phase06_rq_analysis_matrix.csv": rq_matrix,
        "phase06_paired_continuous_statistics.csv": paired_continuous,
        "phase06_paired_binary_statistics.csv": paired_binary,
        "phase06_sufficiency_association_statistics.csv": association,
        "phase06_statistical_tests.csv": tests,
        "phase06_holm_correction.csv": holm,
        "phase06_effect_sizes.csv": effects,
        "phase06_bootstrap_intervals.csv": bootstrap,
        "phase06_policy_confidence_intervals.csv": policy_ci,
        "phase06_policy_quality_intervals.csv": policy_quality_ci,
    }
    for name, frame in outputs.items():
        write_bytes_atomic(
            results / name,
            frame.to_csv(index=False, lineterminator="\n", na_rep="").encode("utf-8"),
        )

    table_manifest = _table_artifacts(
        root, paired_continuous, paired_binary, association, tests, policy_ci, policy_quality_ci
    )
    phase2 = pd.read_csv(results / "phase02_metrics_train_validation.csv")
    model_metrics = pd.read_csv(results / "phase04_model_validation_metrics.csv")
    reliability = pd.read_csv(results / "phase04_reliability_bins.csv")
    importance = pd.read_csv(results / "phase04_rf_permutation_importance.csv")
    curve = pd.read_csv(results / "phase04_risk_coverage_curve.csv")
    curve["record_type"] = "curve"
    curve["risk_constraint"] = np.nan
    operating = pd.read_csv(results / "phase04_risk_operating_points.csv")
    operating_data = pd.DataFrame({
        "threshold": operating.threshold, "answered_count": np.nan, "total_count": np.nan,
        "coverage": operating.coverage, "unsafe_answered_count": np.nan,
        "selective_risk": operating.selective_risk, "record_type": "operating_point",
        "risk_constraint": operating.risk_constraint,
    })
    unsupported_assoc = association.loc[
        (association.metric == "unsupported_claim_rate") & (association.outcome_type == "continuous")
    ]
    figure7_rows = []
    for row in unsupported_assoc.itertuples(index=False):
        figure7_rows.extend([
            {"k": row.k, "context_group": "insufficient", "n": row.insufficient_n,
             "mean_unsupported_claim_rate": row.insufficient_mean,
             "ci_low": row.insufficient_mean_ci_low, "ci_high": row.insufficient_mean_ci_high},
            {"k": row.k, "context_group": "sufficient", "n": row.sufficient_n,
             "mean_unsupported_claim_rate": row.sufficient_mean,
             "ci_low": row.sufficient_mean_ci_low, "ci_high": row.sufficient_mean_ci_high},
        ])
    policy_summary = pd.read_csv(results / "phase05_policy_generation_comparison.csv")
    figure_manifest = generate_figures(root, {
        "F01": phase2.loc[phase2.split == "validation", ["retrieval_strategy", "k", "eligible_questions", "document_recall_at_k", "mrr_at_10"]],
        "F02A": model_metrics.loc[model_metrics.model.isin(["B1_idf_coverage_threshold", "B2_logistic_regression", "B3_random_forest"]), ["model", "auprc", "auroc"]],
        "F02B": model_metrics.loc[model_metrics.model.isin(["B1_idf_coverage_threshold", "B2_logistic_regression", "B3_random_forest"]), ["model", "auprc", "auroc"]],
        "F03": reliability.loc[(reliability.model == "B3_random_forest") & (reliability.probability_method == "uncalibrated")],
        "F04": importance,
        "F05": pd.concat([curve, operating_data], ignore_index=True),
        "F06": paired[["question_id", "difference_unsupported_claim_rate_k10_minus_k5"]],
        "F07": pd.DataFrame(figure7_rows),
        "F08": policy_summary[["policy_id", "policy_answer_coverage", "unsupported_answer_population_rate", "grounded_answer_yield"]].rename(columns={"policy_answer_coverage": "answer_coverage"}),
    })

    summary = {
        "schema_version": "phase06-statistical-summary-v1",
        "phase06_config_canonical_sha256": config.canonical_sha256,
        "upstream": upstream,
        "families": {
            family: {
                "size": len(group),
                "holm_rejections": int(group.reject_holm.astype(bool).sum()),
                "complete": True,
            }
            for family, group in tests.groupby("family_id", sort=True)
        },
        "significant_after_holm": tests.loc[tests.reject_holm.astype(bool), "comparison_id"].tolist(),
        "paired_mean_differences_reproduced": {
            row.metric: row.mean_difference for row in paired_continuous.itertuples(index=False)
        },
        "techqa_test_accessed": False,
        "techqa_test_sealed": True,
        "phase04_model_retrained": False,
        "phase04_thresholds_changed": False,
        "phase05_answers_regenerated": False,
        "ragtruth_nli_rerun": False,
        "phase07_started": False,
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "pyarrow": importlib.metadata.version("pyarrow"),
        },
        "limitations": [
            "TechQA inference is validation-only over 89 questions.",
            "Sufficiency-generation comparisons are associational.",
            "Automatic grounding is an imperfect proxy with low binary precision on RAGTruth.",
            "G2 quality has only two answered cases and is not inferentially compared.",
        ],
    }
    write_json_atomic(results / "phase06_statistical_summary.json", summary)
    write_bytes_atomic(
        root / "docs/PHASE_06_RESULTS_SUMMARY.md",
        _result_summary_markdown(tests, paired_continuous, paired_binary, association, policy_ci).encode("utf-8"),
    )
    integrity = {
        "schema_version": "phase06-integrity-report-v1", "status": "pass",
        "phase06_config_canonical_sha256": config.canonical_sha256,
        "upstream_hashes_reproduced": True, "pre_analysis_governance_frozen_before_results": True,
        "techqa_bootstrap_unit": "question_id", "ragtruth_bootstrap_unit": "source_id",
        "bootstrap_replicates": config.values["bootstrap"]["replicates"],
        "holm_family_sizes": tests.groupby("family_id").size().sort_index().to_dict(),
        "paired_descriptive_differences_reproduced": True,
        "sufficiency_counts": {"k5": {"sufficient": 47, "insufficient": 42}, "k10": {"sufficient": 53, "insufficient": 36}},
        "g2_tiny_sample_guard_applied": True, "table_count": len(table_manifest),
        "figure_count": len(figure_manifest["figures"]), "techqa_test_rows": 0,
        "techqa_test_accessed": False, "techqa_test_sealed": True,
        "phase04_model_retrained": False, "phase04_thresholds_changed": False,
        "phase05_answers_regenerated": False, "ragtruth_nli_rerun": False,
        "phase07_started": False,
    }
    write_json_atomic(results / "phase06_integrity_report.json", integrity)
    manifest_paths = _phase06_manifest_paths(root)
    manifest = {
        "schema_version": "phase06-artifact-manifest-v1",
        "manifest_includes_itself": False,
        "phase06_config_canonical_sha256": config.canonical_sha256,
        "phase05_manifest_physical_sha256": upstream["phase05_manifest_sha256"],
        "artifacts": [{
            "path": path.as_posix(), "bytes": (root / path).stat().st_size,
            "physical_sha256": sha256_file(root / path),
        } for path in manifest_paths],
        "techqa_test_sealed": True, "phase07_started": False,
    }
    write_json_atomic(results / "phase06_artifact_manifest.json", manifest)
    return {
        "status": "pass", "phase06_config_canonical_sha256": config.canonical_sha256,
        "phase06_manifest_sha256": sha256_file(results / "phase06_artifact_manifest.json"),
        "hypothesis_count": len(tests),
        "holm_rejection_count": int(tests.reject_holm.astype(bool).sum()),
        "table_count": len(table_manifest), "figure_count": len(figure_manifest["figures"]),
        "techqa_test_sealed": True, "phase07_started": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase06_statistics.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    print(json.dumps(run(root, root / args.config), indent=2))


if __name__ == "__main__":
    main()
