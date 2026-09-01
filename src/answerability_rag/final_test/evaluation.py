"""Frozen feature inference, classifier, risk-coverage, and adaptive policy evaluation."""

from __future__ import annotations

import json
import math
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer

from answerability_rag.answerability.features import condition_features
from answerability_rag.answerability.modeling import classifier_metrics, reliability_bins
from answerability_rag.answerability.pipeline import _fit_with_calibration
from answerability_rag.answerability.registry import FeatureRegistry, load_feature_registry
from answerability_rag.answerability.selective import aurc, risk_coverage_curve
from answerability_rag.data.techqa import load_techqa_rows
from answerability_rag.hashing import canonical_json_sha256, sha256_file
from answerability_rag.retrieval.artifacts import write_canonical_parquet

from .common import RESULTS, Phase07Config, require_unsealed, write_csv, write_json


FEATURE_PREFIX = (
    "schema_version", "retrieval_condition_id", "question_id", "split",
    "retrieval_strategy", "k", "y_suff_final",
)
PREDICTION_FIELDS = (
    "schema_version", "retrieval_condition_id", "question_id", "split",
    "retrieval_strategy", "k", "y_suff_final", "probability", "decision_threshold",
    "predicted_label", "correct", "feature_registry_sha256",
    "selected_model_config_sha256", "model_artifact_sha256", "phase07_config_sha256",
)
POLICY_TRAJECTORY_FIELDS = (
    "schema_version", "policy_id", "risk_constraint", "question_id", "split_group_id",
    "p5", "p10", "y5", "y10", "t_low", "t_high", "initial_action",
    "expansion_triggered", "post_expansion_action", "final_action", "final_k",
    "y_suff_final_context", "answered", "safe_answer", "unsafe_answer",
    "false_abstention", "retrieved_depth", "retrieval_cost_proxy",
)


@dataclass
class TestFeatureResources:
    root: Path
    registry: FeatureRegistry
    chunks: pd.DataFrame
    chunk_index: dict[str, int]
    embeddings: np.memmap
    conditions: pd.DataFrame
    questions: dict[str, str]
    query_embeddings: dict[str, np.ndarray]
    idf: dict[str, float]
    oov_idf: float

    @classmethod
    def load(cls, root: Path, registry: FeatureRegistry, question_ids: set[str]) -> "TestFeatureResources":
        chunks = pq.read_table(
            root / "artifacts/data/techqa_chunk_manifest.parquet",
            columns=["chunk_id", "doc_id", "token_length", "text"],
        ).to_pandas()
        chunk_index = {str(chunk_id): index for index, chunk_id in enumerate(chunks["chunk_id"])}
        dense_metadata_path = next((root / "data/derived/phase02/cache").glob("dense-*.json"))
        dense_metadata = json.loads(dense_metadata_path.read_text(encoding="utf-8"))
        dense_path = dense_metadata_path.with_suffix(".f32")
        shape = tuple(dense_metadata["shape"])
        if shape != (len(chunks), 384) or dense_path.stat().st_size != int(np.prod(shape)) * 4:
            raise ValueError("frozen dense cache shape or bytes differ")
        embeddings = np.memmap(dense_path, dtype=np.float32, mode="r", shape=shape)
        conditions = pq.read_table(
            root / "artifacts/results/retrieval_query_level.parquet",
            filters=[("split", "=", "test")],
            columns=["question_id", "split", "retrieval_strategy", "k",
                     "ordered_chunk_ids_json", "ordered_doc_ids_json", "ordered_raw_scores_json"],
        ).to_pandas()
        conditions = conditions.loc[conditions.question_id.isin(question_ids)].reset_index(drop=True)
        if set(conditions.split) != {"test"} or conditions.question_id.nunique() != len(question_ids):
            raise ValueError("TEST feature retrieval population differs")
        phase01 = json.loads((root / "configs/phase01_data.json").read_text(encoding="utf-8"))
        raw_path = root / "data/raw/techqa" / phase01["techqa"]["revision"] / "train.json"
        questions = {
            row.question_id: row.question for row in load_techqa_rows(raw_path)
            if row.question_id in question_ids
        }
        if set(questions) != question_ids:
            raise ValueError("PRIMARY TEST questions do not resolve exactly")
        snapshot = (
            root / "data/derived/phase02/model_cache/models--sentence-transformers--all-MiniLM-L6-v2"
            / "snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
        )
        model = SentenceTransformer(str(snapshot), device="cpu")
        ordered_ids = sorted(questions)
        encoded = model.encode(
            [questions[qid] for qid in ordered_ids], batch_size=64, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        ).astype(np.float32, copy=False)
        query_embeddings = dict(zip(ordered_ids, encoded, strict=True))
        del model
        bm25_path = next((root / "data/derived/phase02/cache").glob("bm25-*.pkl"))
        with bm25_path.open("rb") as handle:
            bm25 = pickle.load(handle)
        idf = dict(bm25.idf)
        corpus_size = int(bm25.corpus_size)
        oov_idf = math.log((corpus_size + 0.5) / 0.5)
        return cls(
            root, registry, chunks, chunk_index, embeddings, conditions, questions,
            query_embeddings, idf, oov_idf,
        )

    def make_rows(self, targets: pd.DataFrame) -> list[dict[str, Any]]:
        condition_map = {
            (row.question_id, row.retrieval_strategy, int(row.k)): row
            for row in self.conditions.itertuples(index=False)
        }
        agreement: dict[tuple[str, int, str], list[str]] = {}
        for row in self.conditions.itertuples(index=False):
            if row.retrieval_strategy in {"bm25", "dense"}:
                agreement[(row.question_id, int(row.k), row.retrieval_strategy)] = json.loads(
                    row.ordered_chunk_ids_json
                )
        rows: list[dict[str, Any]] = []
        for target in targets.sort_values(["question_id", "retrieval_strategy", "k"]).itertuples(index=False):
            key = (target.question_id, target.retrieval_strategy, int(target.k))
            condition = condition_map.get(key)
            if condition is None:
                raise ValueError(f"missing TEST retrieval condition: {key}")
            chunk_ids = json.loads(condition.ordered_chunk_ids_json)
            indexes = [self.chunk_index[str(chunk_id)] for chunk_id in chunk_ids]
            chunks = self.chunks.iloc[indexes]
            features = condition_features(
                question=self.questions[target.question_id],
                scores=json.loads(condition.ordered_raw_scores_json),
                chunk_texts=chunks.text.tolist(), token_lengths=chunks.token_length.tolist(),
                doc_ids=chunks.doc_id.tolist(), query_embedding=self.query_embeddings[target.question_id],
                chunk_embeddings=np.asarray(self.embeddings[indexes]),
                bm25_chunk_ids=agreement[(target.question_id, int(target.k), "bm25")],
                dense_chunk_ids=agreement[(target.question_id, int(target.k), "dense")],
                idf=self.idf, oov_idf=self.oov_idf,
                identifier_regex=self.registry.identifier_regex,
                retrieval_strategy=target.retrieval_strategy, k=int(target.k),
            )
            self.registry.validate_model_columns(features.keys())
            rows.append({
                "schema_version": "phase07-test-features-v1",
                "retrieval_condition_id": target.example_id, "question_id": target.question_id,
                "split": "test", "retrieval_strategy": target.retrieval_strategy,
                "k": int(target.k), "y_suff_final": int(target.y_suff_final), **features,
            })
        if len(rows) != len(targets):
            raise AssertionError("TEST feature rows and target rows differ")
        return rows


def _classifier_bootstrap(frame: pd.DataFrame, replicates: int = 5000, seed: int = 42) -> list[dict[str, Any]]:
    groups = sorted(frame.question_id.unique())
    pieces = {group: frame.loc[frame.question_id == group] for group in groups}
    rng = np.random.default_rng(seed)
    samples = {metric: [] for metric in ("auroc", "auprc", "f1", "brier")}
    for _ in range(replicates):
        drawn = rng.choice(np.asarray(groups, dtype=object), size=len(groups), replace=True)
        sample = pd.concat([pieces[str(group)] for group in drawn], ignore_index=True)
        try:
            metrics = classifier_metrics(
                sample.y_suff_final.to_numpy(int), sample.probability.to_numpy(float), 0.5,
            )
        except ValueError:
            continue
        for metric in samples:
            value = float(metrics[metric])
            if np.isfinite(value):
                samples[metric].append(value)
    point = classifier_metrics(
        frame.y_suff_final.to_numpy(int), frame.probability.to_numpy(float), 0.5,
    )
    return [{
        "metric": metric, "point_estimate": point[metric], "confidence_level": 0.95,
        "ci_low": float(np.percentile(values, 2.5)) if values else None,
        "ci_high": float(np.percentile(values, 97.5)) if values else None,
        "requested_replicates": replicates, "valid_replicates": len(values),
        "seed": seed, "resampling_unit": "question_id",
    } for metric, values in samples.items()]


def _frozen_threshold_rows(frame: pd.DataFrame, thresholds: dict[str, float]) -> list[dict[str, Any]]:
    output = []
    y = frame.y_suff_final.to_numpy(int)
    p = frame.probability.to_numpy(float)
    for name, threshold in thresholds.items():
        answered = p >= float(threshold)
        count = int(answered.sum())
        unsafe = int(((y == 0) & answered).sum())
        output.append({
            "validation_risk_constraint": float(name), "frozen_validation_threshold": float(threshold),
            "test_answered_count": count, "test_total_count": len(frame),
            "test_coverage": count / len(frame), "test_unsafe_answer_count": unsafe,
            "test_selective_risk": unsafe / count if count else None,
            "threshold_selected_on_test": False,
        })
    return output


def _trajectory_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows); answered = [row for row in rows if row["answered"]]
    unsafe = sum(row["unsafe_answer"] for row in rows)
    safe = sum(row["safe_answer"] is True for row in rows)
    expanded = sum(row["expansion_triggered"] for row in rows)
    sufficient = sum(int(row["y_suff_final_context"]) == 1 for row in rows)
    false = sum(row["false_abstention"] for row in rows)
    mean_depth = float(np.mean([row["retrieved_depth"] for row in rows]))
    return {
        "eligible_questions": n, "answer_count": len(answered),
        "answer_coverage": len(answered) / n, "abstention_count": n - len(answered),
        "retrieval_expansion_count": expanded, "retrieval_expansion_rate": expanded / n,
        "selective_risk": unsafe / len(answered) if answered else None,
        "selective_risk_numerator": unsafe, "selective_risk_denominator": len(answered),
        "false_abstention_rate": false / sufficient if sufficient else None,
        "false_abstention_count": false, "false_abstention_denominator": sufficient,
        "safe_answer_count": safe, "unsafe_answer_count": unsafe,
        "mean_retrieval_depth": mean_depth, "retrieval_cost_proxy": mean_depth / 5.0,
    }


def _simulate_policy(
    trajectories: pd.DataFrame, policy_id: str, risk_constraint: float, t_low: float, t_high: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for row in trajectories.sort_values("question_id").itertuples(index=False):
        if float(row.p5) >= t_high:
            initial, expanded, post, action, final_k, final_y = "answer", False, None, "ANSWER_AT_K5", 5, int(row.y5)
        elif float(row.p5) < t_low:
            initial, expanded, post, action, final_k, final_y = "abstain", False, None, "ABSTAIN", 5, int(row.y5)
        elif float(row.p10) >= t_high:
            initial, expanded, post, action, final_k, final_y = "request_more_evidence", True, "answer", "ANSWER_AT_K10", 10, int(row.y10)
        else:
            initial, expanded, post, action, final_k, final_y = "request_more_evidence", True, "abstain", "ABSTAIN", 10, int(row.y10)
        answered = action.startswith("ANSWER")
        rows.append({
            "schema_version": "phase07-test-policy-trajectory-v1", "policy_id": policy_id,
            "risk_constraint": risk_constraint, "question_id": row.question_id,
            "split_group_id": row.split_group_id, "p5": float(row.p5), "p10": float(row.p10),
            "y5": int(row.y5), "y10": int(row.y10), "t_low": t_low, "t_high": t_high,
            "initial_action": initial, "expansion_triggered": expanded,
            "post_expansion_action": post, "final_action": action, "final_k": final_k,
            "y_suff_final_context": final_y, "answered": answered,
            "safe_answer": bool(final_y) if answered else None,
            "unsafe_answer": bool(answered and not final_y),
            "false_abstention": bool((not answered) and final_y),
            "retrieved_depth": final_k, "retrieval_cost_proxy": final_k / 5.0,
        })
    return {"policy_id": policy_id, "risk_constraint": risk_constraint, "t_low": t_low, "t_high": t_high,
            **_trajectory_metrics(rows)}, rows


def _policy_bootstrap(policy_id: str, rows: list[dict[str, Any]], replicates: int = 5000) -> list[dict[str, Any]]:
    rng = np.random.default_rng(42); n = len(rows)
    metrics = ("answer_coverage", "selective_risk", "false_abstention_rate", "retrieval_expansion_rate")
    samples: dict[str, list[float]] = {metric: [] for metric in metrics}
    for _ in range(replicates):
        sample = [rows[int(index)] for index in rng.integers(0, n, size=n)]
        values = _trajectory_metrics(sample)
        for metric in metrics:
            value = values[metric]
            if value is not None and np.isfinite(float(value)):
                samples[metric].append(float(value))
    point = _trajectory_metrics(rows)
    return [{
        "policy_id": policy_id, "metric": metric, "point_estimate": point[metric],
        "ci_low": float(np.percentile(values, 2.5)) if values else None,
        "ci_high": float(np.percentile(values, 97.5)) if values else None,
        "confidence_level": 0.95, "requested_replicates": replicates,
        "valid_replicates": len(values), "seed": 42, "resampling_unit": "question_id",
    } for metric, values in samples.items()]


def run_test_inference(root: Path, config_path: Path) -> dict[str, Any]:
    config = Phase07Config.load(config_path); require_unsealed(root, config)
    target_path = root / RESULTS / "phase07_test_final_target.parquet"
    target_manifest = json.loads(
        (root / RESULTS / "phase07_test_target_manifest.json").read_text(encoding="utf-8")
    )
    if not target_manifest["immutable"]:
        raise ValueError("TEST target is not closed and immutable")
    targets = pq.read_table(target_path).to_pandas()
    registry = load_feature_registry(root / "configs/phase04_feature_registry.json")
    selected = json.loads((root / "configs/phase04_selected_model.json").read_text(encoding="utf-8"))
    if tuple(selected["features"]) != registry.model_features or len(registry.model_features) != 39:
        raise ValueError("frozen 39-feature registry differs from selected model")
    resources = TestFeatureResources.load(root, registry, set(targets.question_id))
    feature_rows = resources.make_rows(targets)
    feature_fields = list(FEATURE_PREFIX) + [
        feature for feature in registry.model_features if feature not in {"retrieval_strategy", "k"}
    ]
    feature_path = root / "data/derived/phase07/phase07_test_inference_features.parquet"
    feature_artifact = write_canonical_parquet(
        feature_path, feature_rows, feature_fields, ("question_id", "retrieval_strategy", "k")
    )
    features = pq.read_table(feature_path).to_pandas()
    write_json(root / RESULTS / "phase07_test_feature_manifest.json", {
        "schema_version": "phase07-test-feature-manifest-v1",
        "feature_artifact": feature_artifact, "feature_count": 39,
        "feature_names": list(registry.model_features),
        "feature_registry_path": "configs/phase04_feature_registry.json",
        "feature_registry_sha256": selected["feature_registry_sha256"],
        "target_artifact_sha256": sha256_file(target_path),
        "question_count": int(features.question_id.nunique()), "condition_count": len(features),
        "missing_value_counts": {name: int(features[name].isna().sum()) for name in registry.model_features},
        "test_fit_calls": 0, "test_preprocessing_fit": False,
        "leakage_guard_passed": True,
    })
    X = features.loc[:, registry.model_features]
    registry.validate_model_columns(X.columns)
    bundle = joblib.load(root / selected["model_artifact_path"])
    if tuple(bundle["feature_names"]) != registry.model_features:
        raise ValueError("serialized model feature order differs")
    if bundle["modeling_config_sha256"] != selected["phase04_modeling_config_sha256"]:
        raise ValueError("serialized model configuration differs")
    # Phase 7 calls prediction only. No fit/fit_transform/calibrator fit is invoked on TEST.
    raw = bundle["pipeline"].predict_proba(X)[:, 1]
    probability = bundle["calibrator"].transform(raw)
    if selected["calibration_method"] != "uncalibrated" or not np.array_equal(raw, probability):
        raise ValueError("frozen uncalibrated Phase 4 inference differs")
    prediction_rows = []
    for row, p in zip(features.itertuples(index=False), probability, strict=True):
        predicted = int(float(p) >= 0.5)
        prediction_rows.append({
            "schema_version": "phase07-test-classifier-prediction-v1",
            "retrieval_condition_id": row.retrieval_condition_id,
            "question_id": row.question_id, "split": "test",
            "retrieval_strategy": row.retrieval_strategy, "k": int(row.k),
            "y_suff_final": int(row.y_suff_final), "probability": float(p),
            "decision_threshold": 0.5, "predicted_label": predicted,
            "correct": predicted == int(row.y_suff_final),
            "feature_registry_sha256": selected["feature_registry_sha256"],
            "selected_model_config_sha256": selected["selected_model_config_sha256"],
            "model_artifact_sha256": selected["model_artifact_sha256"],
            "phase07_config_sha256": config.canonical_sha256,
        })
    prediction_path = root / RESULTS / "phase07_test_classifier_predictions.parquet"
    prediction_artifact = write_canonical_parquet(
        prediction_path, prediction_rows, PREDICTION_FIELDS,
        ("question_id", "retrieval_strategy", "k"),
    )
    frame = pq.read_table(prediction_path).to_pandas()
    metrics = classifier_metrics(frame.y_suff_final.to_numpy(int), frame.probability.to_numpy(float), 0.5)
    metric_artifact = {
        "schema_version": "phase07-test-classifier-metrics-v1", "split": "test",
        "question_count": int(frame.question_id.nunique()), "condition_count": len(frame),
        "decision_threshold": 0.5, **metrics,
    }
    write_json(root / RESULTS / "phase07_test_classifier_metrics.json", metric_artifact)
    bootstrap = _classifier_bootstrap(frame)
    boot_fields = (
        "metric", "point_estimate", "confidence_level", "ci_low", "ci_high",
        "requested_replicates", "valid_replicates", "seed", "resampling_unit",
    )
    write_csv(root / RESULTS / "phase07_test_classifier_bootstrap_intervals.csv", bootstrap, boot_fields)
    _, bins = reliability_bins(frame.y_suff_final.to_numpy(int), frame.probability.to_numpy(float), 10)
    for row in bins:
        row["absolute_gap"] = (
            None if row["count"] == 0 else abs(float(row["positive_fraction"]) - float(row["mean_probability"]))
        )
    bin_fields = (
        "bin", "lower", "upper", "upper_inclusive", "count", "mean_probability",
        "positive_fraction", "absolute_gap", "ece_contribution",
    )
    write_csv(root / RESULTS / "phase07_test_reliability_bins.csv", bins, bin_fields)
    curve = risk_coverage_curve(frame.y_suff_final.to_numpy(int), frame.probability.to_numpy(float))
    area = aurc(curve)
    for row in curve:
        row["aurc"] = area
    curve_fields = (
        "threshold", "answered_count", "total_count", "coverage", "unsafe_answered_count",
        "selective_risk", "aurc",
    )
    write_csv(root / RESULTS / "phase07_test_risk_coverage_curve.csv", curve, curve_fields)
    write_json(root / RESULTS / "phase07_test_aurc.json", {
        "schema_version": "phase07-test-aurc-v1", "aurc": area,
        "integration": "right_endpoint_rectangle_over_increasing_observed_coverage",
    })
    thresholds = config.values["risk_coverage"]["frozen_validation_thresholds"]
    threshold_rows = _frozen_threshold_rows(frame, thresholds)
    threshold_fields = (
        "validation_risk_constraint", "frozen_validation_threshold", "test_answered_count",
        "test_total_count", "test_coverage", "test_unsafe_answer_count",
        "test_selective_risk", "threshold_selected_on_test",
    )
    write_csv(root / RESULTS / "phase07_test_frozen_risk_operating_points.csv", threshold_rows, threshold_fields)
    hybrid = frame.loc[(frame.retrieval_strategy == "hybrid") & frame.k.isin([5, 10])].copy()
    group_map = targets[["question_id", "split_group_id"]].drop_duplicates().set_index("question_id")["split_group_id"].to_dict()
    pivot_p = hybrid.pivot(index="question_id", columns="k", values="probability")
    pivot_y = hybrid.pivot(index="question_id", columns="k", values="y_suff_final")
    trajectories = pd.DataFrame({
        "question_id": pivot_p.index,
        "split_group_id": [group_map[qid] for qid in pivot_p.index],
        "p5": pivot_p[5].to_numpy(float), "p10": pivot_p[10].to_numpy(float),
        "y5": pivot_y[5].to_numpy(int), "y10": pivot_y[10].to_numpy(int),
    })
    if ((trajectories.y5 == 1) & (trajectories.y10 == 0)).any():
        raise ValueError("TEST target violates k5-to-k10 monotonicity")
    policy_specs = (
        ("G2", 0.10, 0.78, 0.82), ("G3", 0.20, 0.56, 0.72),
    )
    policy_metrics: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    policy_bootstrap: list[dict[str, Any]] = []
    for policy_id, constraint, low, high in policy_specs:
        summary, rows = _simulate_policy(trajectories, policy_id, constraint, low, high)
        policy_metrics.append(summary); policy_rows.extend(rows)
        policy_bootstrap.extend(_policy_bootstrap(policy_id, rows))
    policy_artifact = write_canonical_parquet(
        root / RESULTS / "phase07_test_policy_trajectories.parquet", policy_rows,
        POLICY_TRAJECTORY_FIELDS, ("policy_id", "question_id"),
    )
    policy_metric_fields = (
        "policy_id", "risk_constraint", "t_low", "t_high", "eligible_questions",
        "answer_count", "answer_coverage", "abstention_count", "retrieval_expansion_count",
        "retrieval_expansion_rate", "selective_risk", "selective_risk_numerator",
        "selective_risk_denominator", "false_abstention_rate", "false_abstention_count",
        "false_abstention_denominator", "safe_answer_count", "unsafe_answer_count",
        "mean_retrieval_depth", "retrieval_cost_proxy",
    )
    write_csv(root / RESULTS / "phase07_test_policy_metrics.csv", policy_metrics, policy_metric_fields)
    policy_boot_fields = (
        "policy_id", "metric", "point_estimate", "ci_low", "ci_high", "confidence_level",
        "requested_replicates", "valid_replicates", "seed", "resampling_unit",
    )
    write_csv(root / RESULTS / "phase07_test_policy_bootstrap_intervals.csv", policy_bootstrap, policy_boot_fields)
    manifest = {
        "schema_version": "phase07-test-inference-manifest-v1",
        "phase07_config_sha256": config.canonical_sha256,
        "feature_artifact": feature_artifact, "prediction_artifact": prediction_artifact,
        "policy_artifact": policy_artifact, "feature_count": 39,
        "feature_names": list(registry.model_features), "test_fit_calls": 0,
        "model_retrained": False, "calibrator_refit": False,
        "test_threshold_selected": False, "classifier_metrics": metrics,
        "aurc": area, "policy_metrics": policy_metrics,
    }
    write_json(root / RESULTS / "phase07_test_inference_manifest.json", manifest)
    return manifest


def run_benchmark_impossible_sensitivity(root: Path, config_path: Path) -> dict[str, Any]:
    """Apply the selected frozen model to preliminary-negative TEST cases, separately."""
    config = Phase07Config.load(config_path); require_unsealed(root, config)
    source = pq.read_table(
        root / "artifacts/data/context_sufficiency_labels.parquet",
        filters=[("split", "=", "test")],
        columns=["example_id", "question_id", "split", "retrieval_strategy", "k", "benchmark_is_impossible", "y_suff"],
    ).to_pandas()
    impossible = source.loc[source.benchmark_is_impossible == True].copy()  # noqa: E712
    if impossible.question_id.nunique() != 45 or len(impossible) != 540 or not (impossible.y_suff == 0).all():
        raise ValueError("frozen benchmark-impossible TEST census differs")
    impossible["y_suff_final"] = 0
    registry = load_feature_registry(root / "configs/phase04_feature_registry.json")
    selected = json.loads((root / "configs/phase04_selected_model.json").read_text(encoding="utf-8"))
    fields = list(FEATURE_PREFIX) + [feature for feature in registry.model_features if feature not in {"retrieval_strategy", "k"}]
    feature_path = root / "data/derived/phase07/phase07_test_benchmark_impossible_features.parquet"
    if feature_path.exists():
        features = pq.read_table(feature_path).to_pandas()
        if len(features) != 540 or set(features.question_id.astype(str)) != set(impossible.question_id.astype(str)):
            raise ValueError("existing benchmark-impossible TEST feature identity differs")
        feature_artifact = {
            "path": feature_path.relative_to(root).as_posix(), "bytes": feature_path.stat().st_size,
            "physical_sha256": sha256_file(feature_path), "rows": len(features),
            "columns": list(features.columns),
        }
    else:
        resources = TestFeatureResources.load(root, registry, set(impossible.question_id.astype(str)))
        rows = resources.make_rows(impossible)
        feature_artifact = write_canonical_parquet(feature_path, rows, fields, ("question_id", "retrieval_strategy", "k"))
        features = pq.read_table(feature_path).to_pandas()
    primary_train = pq.read_table(root / "data/derived/phase04/phase04_inference_features.parquet").to_pandas()
    impossible_upstream = pq.read_table(root / "data/derived/phase04/phase04_benchmark_impossible_features.parquet").to_pandas()
    augmented_train = pd.concat([
        primary_train.loc[primary_train.split == "train"],
        impossible_upstream.loc[impossible_upstream.split == "train"],
    ], ignore_index=True)
    if len(augmented_train) != 7608 or augmented_train.question_id.nunique() != 634:
        raise ValueError("frozen Phase 4 augmented sensitivity TRAIN population differs")
    primary_test = pq.read_table(root / "data/derived/phase07/phase07_test_inference_features.parquet").to_pandas()
    combined = pd.concat([primary_test, features], ignore_index=True)
    fitted, calibrator, combined_probability = _fit_with_calibration(
        selected["model_family"], selected["hyperparameters"], selected["calibration_method"],
        registry, registry.model_features,
        augmented_train.loc[:, registry.model_features], augmented_train.y_suff_final.to_numpy(int),
        augmented_train.question_id.to_numpy(str), combined.loc[:, registry.model_features],
    )
    del fitted, calibrator
    probability = combined_probability[len(primary_test):]
    prediction_rows = [{
        "schema_version": "phase07-test-benchmark-impossible-prediction-v1",
        "retrieval_condition_id": row.retrieval_condition_id, "question_id": row.question_id,
        "split": "test", "retrieval_strategy": row.retrieval_strategy, "k": int(row.k),
        "preliminary_target": 0, "probability": float(p), "predicted_label": int(p >= 0.5),
    } for row, p in zip(features.itertuples(index=False), probability, strict=True)]
    prediction_fields = (
        "schema_version", "retrieval_condition_id", "question_id", "split", "retrieval_strategy",
        "k", "preliminary_target", "probability", "predicted_label",
    )
    prediction_artifact = write_canonical_parquet(
        root / RESULTS / "phase07_test_benchmark_impossible_predictions.parquet",
        prediction_rows, prediction_fields, ("question_id", "retrieval_strategy", "k"),
    )
    combined_y = combined.y_suff_final.to_numpy(int)
    combined_p = np.asarray(combined_probability, dtype=float)
    result = {
        "schema_version": "phase07-benchmark-impossible-test-sensitivity-v1",
        "population": "separate_benchmark_impossible_test_sensitivity",
        "question_count": 45, "condition_count": 540, "preliminary_target": 0,
        "predicted_sufficient_count_at_0_5": int((probability >= 0.5).sum()),
        "predicted_sufficient_rate_at_0_5": float((probability >= 0.5).mean()),
        "mean_predicted_sufficiency": float(np.mean(probability)),
        "preliminary_negative_accuracy": float((probability < 0.5).mean()),
        "preliminary_negative_brier": float(np.mean(probability ** 2)),
        "combined_primary_plus_sensitivity_metrics": classifier_metrics(combined_y, combined_p, 0.5),
        "auroc_impossible_only": None,
        "feature_artifact": feature_artifact, "prediction_artifact": prediction_artifact,
        "sensitivity_model_fit_calls": 1, "sensitivity_model_fit_population": "frozen TRAIN only",
        "test_fit_calls": 0, "augmented_train_questions": 634, "augmented_train_conditions": 7608,
        "added_impossible_train_questions": 210, "added_impossible_train_conditions": 2520,
        "selected_family": selected["model_family"], "selected_hyperparameters": selected["hyperparameters"],
        "calibration_method": selected["calibration_method"], "hyperparameters_retuned": False,
        "family_reselected": False, "primary_model_changed": False,
        "primary_metrics_influenced": False,
        "limitation": "Benchmark-impossible is benchmark-relative and known to be contaminated; its preliminary zero target is not perfect context-sufficiency ground truth. The fixed Phase 4 sensitivity model is fitted only on the frozen augmented TRAIN population and cannot influence PRIMARY results.",
    }
    write_json(root / RESULTS / "phase07_benchmark_impossible_test_sensitivity.json", result)
    return result


def write_test_feature_manifest(root: Path, config_path: Path) -> dict[str, Any]:
    """Close feature provenance without repeating model inference."""
    config = Phase07Config.load(config_path); require_unsealed(root, config)
    feature_path = root / "data/derived/phase07/phase07_test_inference_features.parquet"
    target_path = root / RESULTS / "phase07_test_final_target.parquet"
    features = pq.read_table(feature_path).to_pandas()
    registry = load_feature_registry(root / "configs/phase04_feature_registry.json")
    selected = json.loads((root / "configs/phase04_selected_model.json").read_text(encoding="utf-8"))
    if len(features) != 1056 or features.question_id.nunique() != 88:
        raise ValueError("closed TEST feature population differs")
    registry.validate_model_columns(features.loc[:, registry.model_features].columns)
    artifact = {
        "path": feature_path.relative_to(root).as_posix(), "bytes": feature_path.stat().st_size,
        "physical_sha256": sha256_file(feature_path), "rows": len(features),
        "columns": list(features.columns),
    }
    manifest = {
        "schema_version": "phase07-test-feature-manifest-v1",
        "feature_artifact": artifact, "feature_count": 39,
        "feature_names": list(registry.model_features),
        "feature_registry_path": "configs/phase04_feature_registry.json",
        "feature_registry_sha256": selected["feature_registry_sha256"],
        "target_artifact_sha256": sha256_file(target_path),
        "question_count": int(features.question_id.nunique()), "condition_count": len(features),
        "missing_value_counts": {name: int(features[name].isna().sum()) for name in registry.model_features},
        "test_fit_calls": 0, "test_preprocessing_fit": False, "leakage_guard_passed": True,
    }
    write_json(root / RESULTS / "phase07_test_feature_manifest.json", manifest)
    return manifest
