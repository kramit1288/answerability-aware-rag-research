"""End-to-end Phase 4 artifact generation with an enforced TEST seal."""

from __future__ import annotations

import importlib.metadata
import json
import math
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, precision_score

from answerability_rag.data.techqa import load_techqa_rows
from answerability_rag.hashing import canonical_json_sha256, sha256_file
from answerability_rag.retrieval.artifacts import write_canonical_parquet

from .features import condition_features
from .modeling import (
    ProbabilityCalibrator, classifier_metrics, fold_manifest, grouped_bootstrap_intervals,
    grouped_oof_probabilities, grouped_splits, logistic_candidates, make_pipeline,
    random_forest_candidates, reliability_bins, search_b1_threshold, search_candidates,
)
from .registry import FeatureRegistry, assert_test_sealed, load_feature_registry
from .selective import (
    always_answer_k5, aurc, policy_grid, risk_coverage_curve, select_policy,
    select_risk_operating_point, select_two_way, simulate_policy,
)


PHASE04_CONFIG_SHA = "f0a0377e4001e6401e4cce77ad15e0904ec13e87719d56660a40171341a6801e"
UPSTREAM = {
    "phase03_final_manifest": ("artifacts/results/phase03_final_artifact_manifest.json",
                               "b02a0ca0e3352798c8d38ef4942fdca59e23d8d24c662493a1249d2864ca9879"),
    "phase03_primary_target": ("artifacts/data/phase03_final_primary_target.parquet",
                               "b775276ace9113286a8e35fd29acaee5623d83be60400537e6902027b8cc07c1"),
}
FEATURE_FIELDS = (
    "schema_version", "retrieval_condition_id", "question_id", "split",
    "retrieval_strategy", "k", "y_suff_final",
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return None if np.isnan(value) else float(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (list, tuple)): return [_jsonable(item) for item in value]
    if isinstance(value, dict): return {key: _jsonable(item) for key, item in value.items()}
    return value


def _write_csv(path: Path, records: Sequence[dict[str, Any]], columns: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = []
    for record in records:
        row = {}
        for key, value in record.items():
            if isinstance(value, (list, dict, tuple)):
                row[key] = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))
            else:
                row[key] = _jsonable(value)
        normalized.append(row)
    pd.DataFrame(normalized, columns=columns).to_csv(
        path, index=False, encoding="utf-8", lineterminator="\n", float_format="%.17g",
    )


def phase04_manifest_paths(selected_family: str) -> list[str]:
    """Return the independently hashed Phase 4 governance, code, test, and result boundary."""
    paths = [
        ".gitignore",
        "configs/phase04_feature_registry.json", "configs/phase04_modeling.json",
        "configs/phase04_selected_model.json", "configs/phase04_selected_policy.json",
        "docs/PHASE_04_RESEARCH_DECISIONS.md", "docs/PHASE_04_EXECUTION.md",
        "docs/PHASE_04_ARTIFACT_SCHEMAS.md", "docs/PHASE_04_RESULTS_INTERPRETATION.md",
        "artifacts/results/phase04_modeling_config_freeze.json",
        "src/answerability_rag/answerability/__init__.py",
        "src/answerability_rag/answerability/registry.py",
        "src/answerability_rag/answerability/features.py",
        "src/answerability_rag/answerability/modeling.py",
        "src/answerability_rag/answerability/selective.py",
        "src/answerability_rag/answerability/pipeline.py",
        "scripts/run_phase04_modeling.py", "scripts/check_phase04.py",
        "scripts/finalize_phase04_artifacts.py",
        "tests/test_phase04_features.py", "tests/test_phase04_modeling.py",
        "tests/test_phase04_selective_policy.py",
        "data/derived/phase04/phase04_inference_features.parquet",
        "data/derived/phase04/phase04_benchmark_impossible_features.parquet",
        "artifacts/models/phase04_selected_model.joblib",
        "artifacts/figures/phase04_reliability_diagram.svg",
        "artifacts/results/phase04_feature_manifest.json",
        "artifacts/results/phase04_grouped_cv_folds.csv",
        "artifacts/results/phase04_grouped_cv_candidates.csv",
        "artifacts/results/phase04_b1_threshold_cv.csv",
        "artifacts/results/phase04_logistic_selected_results.json",
        "artifacts/results/phase04_random_forest_selected_results.json",
        "artifacts/results/phase04_model_validation_metrics.csv",
        "artifacts/results/phase04_rf_calibration_comparison.csv",
        "artifacts/results/phase04_oof_calibration_provenance.parquet",
        "artifacts/results/phase04_reliability_bins.csv",
        "artifacts/results/phase04_bootstrap_confidence_intervals.csv",
        "artifacts/results/phase04_feature_ablation.csv",
        "artifacts/results/phase04_risk_coverage_curve.csv",
        "artifacts/results/phase04_aurc.json", "artifacts/results/phase04_risk_operating_points.csv",
        "artifacts/results/phase04_policy_threshold_grid.csv",
        "artifacts/results/phase04_policy_operating_points.csv",
        "artifacts/results/phase04_policy_trajectories.csv",
        "artifacts/results/phase04_policy_baselines.csv",
        "artifacts/results/phase04_benchmark_impossible_sensitivity.json",
        "artifacts/results/phase04_integrity_report.json",
    ]
    paths.append(
        "artifacts/results/phase04_logistic_coefficients.csv"
        if selected_family == "logistic_regression"
        else "artifacts/results/phase04_rf_permutation_importance.csv"
    )
    return paths


def _read_config(root: Path) -> tuple[dict[str, Any], FeatureRegistry]:
    config_path = root / "configs/phase04_modeling.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if canonical_json_sha256(config) != PHASE04_CONFIG_SHA:
        raise ValueError("Phase 4 modeling config differs from its pre-results freeze")
    freeze = json.loads((root / "artifacts/results/phase04_modeling_config_freeze.json").read_text(encoding="utf-8"))
    if freeze["phase04_modeling_canonical_sha256"] != PHASE04_CONFIG_SHA:
        raise ValueError("Phase 4 configuration-freeze artifact differs")
    for item in freeze["governance_files"]:
        if sha256_file(root / item["path"]) != item["physical_sha256"]:
            raise ValueError(f"Phase 4 frozen governance file differs: {item['path']}")
    registry = load_feature_registry(root / config["feature_registry"]["path"])
    return config, registry


def verify_upstream(root: Path) -> None:
    for name, (relative, expected) in UPSTREAM.items():
        observed = sha256_file(root / relative)
        if observed != expected:
            raise ValueError(f"immutable upstream {name} differs: {observed} != {expected}")
    manifest = json.loads((root / UPSTREAM["phase03_final_manifest"][0]).read_text(encoding="utf-8"))
    failures = [item["path"] for item in manifest["artifacts"].values()
                if sha256_file(root / item["path"]) != item["physical_sha256"]]
    if failures:
        raise ValueError(f"Phase 3 manifest dependency differs: {failures}")


@dataclass
class FeatureResources:
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
    def load(cls, root: Path, registry: FeatureRegistry) -> "FeatureResources":
        chunks = pq.read_table(
            root / "artifacts/data/techqa_chunk_manifest.parquet",
            columns=["chunk_id", "doc_id", "token_length", "text"],
        ).to_pandas()
        chunk_index = {chunk_id: index for index, chunk_id in enumerate(chunks["chunk_id"])}
        dense_metadata_path = next((root / "data/derived/phase02/cache").glob("dense-*.json"))
        dense_metadata = json.loads(dense_metadata_path.read_text(encoding="utf-8"))
        dense_path = dense_metadata_path.with_suffix(".f32")
        shape = tuple(dense_metadata["shape"])
        if shape != (len(chunks), 384) or dense_path.stat().st_size != int(np.prod(shape)) * 4:
            raise ValueError("frozen dense embedding cache shape/bytes differ")
        embeddings = np.memmap(dense_path, dtype=np.float32, mode="r", shape=shape)

        # The filter prevents Phase 4 from reading any TEST retrieval row.
        conditions = pq.read_table(
            root / "artifacts/results/retrieval_query_level.parquet",
            filters=[("split", "in", ["train", "validation"])],
            columns=["question_id", "split", "retrieval_strategy", "k",
                     "ordered_chunk_ids_json", "ordered_doc_ids_json", "ordered_raw_scores_json"],
        ).to_pandas()
        if set(conditions["split"]) != {"train", "validation"}:
            raise ValueError("Phase 4 retrieval input split filter differs")

        phase01 = json.loads((root / "configs/phase01_data.json").read_text(encoding="utf-8"))
        raw_path = root / "data/raw/techqa" / phase01["techqa"]["revision"] / "train.json"
        allowed_ids = set(conditions["question_id"])
        questions = {row.question_id: row.question for row in load_techqa_rows(raw_path)
                     if row.question_id in allowed_ids}
        if set(questions) != allowed_ids:
            raise ValueError("TRAIN/VALIDATION questions do not resolve exactly")

        snapshot = (root / "data/derived/phase02/model_cache/models--sentence-transformers--all-MiniLM-L6-v2"
                    / "snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
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
        del bm25
        return cls(root, registry, chunks, chunk_index, embeddings, conditions,
                   questions, query_embeddings, idf, oov_idf)

    def make_rows(self, targets: pd.DataFrame) -> list[dict[str, Any]]:
        if "test" in set(targets["split"].str.casefold()):
            assert_test_sealed("test", "feature generation")
        target_keys = set(zip(targets.question_id, targets.retrieval_strategy, targets.k))
        available = self.conditions[
            self.conditions.apply(
                lambda row: (row.question_id, row.retrieval_strategy, row.k) in target_keys, axis=1
            )
        ]
        condition_map = {
            (row.question_id, row.retrieval_strategy, int(row.k)): row
            for row in available.itertuples(index=False)
        }
        agreement: dict[tuple[str, int, str], list[str]] = {}
        for row in self.conditions.itertuples(index=False):
            if row.retrieval_strategy in ("bm25", "dense"):
                agreement[(row.question_id, int(row.k), row.retrieval_strategy)] = json.loads(
                    row.ordered_chunk_ids_json
                )
        rows: list[dict[str, Any]] = []
        for target in targets.sort_values(["question_id", "retrieval_strategy", "k"]).itertuples(index=False):
            key = (target.question_id, target.retrieval_strategy, int(target.k))
            condition = condition_map.get(key)
            if condition is None:
                raise ValueError(f"retrieval condition unavailable: {key}")
            chunk_ids = json.loads(condition.ordered_chunk_ids_json)
            scores = json.loads(condition.ordered_raw_scores_json)
            indexes = [self.chunk_index[chunk_id] for chunk_id in chunk_ids]
            chunks = self.chunks.iloc[indexes]
            features = condition_features(
                question=self.questions[target.question_id], scores=scores,
                chunk_texts=chunks["text"].tolist(), token_lengths=chunks["token_length"].tolist(),
                doc_ids=chunks["doc_id"].tolist(), query_embedding=self.query_embeddings[target.question_id],
                chunk_embeddings=np.asarray(self.embeddings[indexes]),
                bm25_chunk_ids=agreement[(target.question_id, int(target.k), "bm25")],
                dense_chunk_ids=agreement[(target.question_id, int(target.k), "dense")],
                idf=self.idf, oov_idf=self.oov_idf,
                identifier_regex=self.registry.identifier_regex,
                retrieval_strategy=target.retrieval_strategy, k=int(target.k),
            )
            self.registry.validate_model_columns(features.keys())
            rows.append({
                "schema_version": "phase04-features-v1",
                "retrieval_condition_id": target.example_id,
                "question_id": target.question_id, "split": target.split,
                "retrieval_strategy": target.retrieval_strategy, "k": int(target.k),
                "y_suff_final": int(target.y_suff_final), **features,
            })
        if len(rows) != len(targets):
            raise AssertionError("feature rows and target rows differ")
        return rows


def _primary_targets(root: Path) -> pd.DataFrame:
    targets = pq.read_table(root / "artifacts/data/phase03_final_primary_target.parquet").to_pandas()
    expected = {("train", 424, 5088), ("validation", 89, 1068)}
    observed = {(split, frame.question_id.nunique(), len(frame)) for split, frame in targets.groupby("split")}
    if observed != expected or len(targets) != 6156 or int(targets.y_suff_final.sum()) != 2802:
        raise ValueError(f"frozen PRIMARY population differs: {observed}")
    return targets


def _feature_manifest(root: Path, frame: pd.DataFrame, registry: FeatureRegistry) -> dict[str, Any]:
    summaries = {}
    for split, subset in frame.groupby("split"):
        summaries[split] = {
            "questions": int(subset.question_id.nunique()), "conditions": len(subset),
            "positive": int(subset.y_suff_final.sum()),
            "negative": int(len(subset) - subset.y_suff_final.sum()),
        }
    return {
        "schema_version": "phase04-feature-manifest-v1",
        "phase04_modeling_config_sha256": PHASE04_CONFIG_SHA,
        "feature_registry_sha256": sha256_file(root / "configs/phase04_feature_registry.json"),
        "feature_artifact": {
            "path": "data/derived/phase04/phase04_inference_features.parquet",
            "physical_sha256": sha256_file(root / "data/derived/phase04/phase04_inference_features.parquet"),
            "rows": len(frame), "columns": list(frame.columns),
        },
        "splits": summaries, "test_rows": int((frame.split == "test").sum()),
        "missing_counts": {feature: int(frame[feature].isna().sum()) for feature in registry.numeric},
        "semantic_model": "sentence-transformers/all-MiniLM-L6-v2",
        "semantic_model_revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    }


def _probability_rows(
    name: str, method: str, y: np.ndarray, probability: np.ndarray, *, threshold: float = 0.5,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics = {"model": name, "probability_method": method, "decision_threshold": threshold,
               **classifier_metrics(y, probability, threshold)}
    _, bins = reliability_bins(y, probability)
    for row in bins:
        row.update({"model": name, "probability_method": method})
    return metrics, bins


def _selected_config(
    root: Path, registry: FeatureRegistry, family: str, params: dict[str, Any],
    calibration: str, model_path: Path,
) -> dict[str, Any]:
    values = {
        "schema_version": "phase04-selected-model-v1", "model_family": family,
        "hyperparameters": params, "calibration_method": calibration,
        "features": list(registry.model_features), "feature_count": len(registry.model_features),
        "preprocessing": {
            "score_normalization": "TRAIN-fitted within retrieval_strategy",
            "numeric_imputation": "TRAIN median with missing indicators",
            "categorical": "fixed one-hot bm25,dense,hybrid",
            "standard_scaler": family == "logistic_regression",
        },
        "random_seed": 42, "model_artifact_path": model_path.relative_to(root).as_posix(),
        "model_artifact_sha256": sha256_file(model_path),
        "phase04_modeling_config_sha256": PHASE04_CONFIG_SHA,
        "feature_registry_sha256": sha256_file(root / "configs/phase04_feature_registry.json"),
        "phase03_final_target_config_sha256": "5b9a394e5e97776844054f3282af815462148adc62b528c77245d61707406977",
    }
    values["selected_model_config_sha256"] = canonical_json_sha256(values)
    return values


def _fit_with_calibration(
    family: str, params: dict[str, Any], calibration: str, registry: FeatureRegistry,
    features: Sequence[str], X_train: pd.DataFrame, y_train: np.ndarray,
    groups: np.ndarray, X_validation: pd.DataFrame,
) -> tuple[Any, ProbabilityCalibrator, np.ndarray]:
    pipeline = make_pipeline(family, registry, features, params, 42)
    calibrator = ProbabilityCalibrator(calibration)
    if calibration == "uncalibrated":
        calibrator.fit(np.array([0.0, 1.0]), np.array([0, 1]))
    else:
        oof, _, _ = grouped_oof_probabilities(pipeline, X_train, y_train, groups)
        calibrator.fit(oof, y_train)
    fitted = pipeline.fit(X_train, y_train)
    probability = calibrator.transform(fitted.predict_proba(X_validation)[:, 1])
    return fitted, calibrator, probability


def _benchmark_impossible_targets(root: Path) -> pd.DataFrame:
    columns = ["example_id", "question_id", "split", "retrieval_strategy", "k",
               "benchmark_is_impossible", "label_status", "y_suff"]
    frame = pq.read_table(root / "artifacts/data/context_sufficiency_labels.parquet",
                          columns=columns).to_pandas()
    frame = frame[(frame.benchmark_is_impossible == True) & frame.split.isin(["train", "validation"])].copy()  # noqa: E712
    if not (frame.y_suff == 0).all():
        raise ValueError("benchmark-impossible sensitivity targets are not preliminary negatives")
    frame["y_suff_final"] = 0
    return frame[["example_id", "question_id", "split", "retrieval_strategy", "k", "y_suff_final"]]


def run_phase04(root: Path) -> dict[str, Any]:
    verify_upstream(root)
    config, registry = _read_config(root)
    seed = int(config["random_seed"])
    results_dir = root / "artifacts/results"
    data_dir = root / "data/derived/phase04"
    model_dir = root / "artifacts/models"
    data_dir.mkdir(parents=True, exist_ok=True); model_dir.mkdir(parents=True, exist_ok=True)

    targets = _primary_targets(root)
    resources = FeatureResources.load(root, registry)
    rows = resources.make_rows(targets)
    fields = list(FEATURE_FIELDS) + [feature for feature in registry.model_features
                                    if feature not in {"retrieval_strategy", "k"}]
    feature_artifact = write_canonical_parquet(
        data_dir / "phase04_inference_features.parquet", rows, fields,
        ("split", "question_id", "retrieval_strategy", "k"),
    )
    frame = pq.read_table(data_dir / "phase04_inference_features.parquet").to_pandas()
    _write_json(results_dir / "phase04_feature_manifest.json", _feature_manifest(root, frame, registry))

    train = frame[frame.split == "train"].reset_index(drop=True)
    validation = frame[frame.split == "validation"].reset_index(drop=True)
    X_train = train.loc[:, registry.model_features]
    X_validation = validation.loc[:, registry.model_features]
    registry.validate_model_columns(X_train.columns)
    registry.validate_model_columns(X_validation.columns)
    y_train = train.y_suff_final.to_numpy(int); y_validation = validation.y_suff_final.to_numpy(int)
    groups_train = train.question_id.to_numpy(str); groups_validation = validation.question_id.to_numpy(str)

    folds = grouped_splits(groups_train)
    fold_rows = fold_manifest(y_train, groups_train, folds, purpose="hyperparameter_selection")
    b1_threshold, b1_rows = search_b1_threshold(
        train.idf_weighted_query_token_context_coverage, y_train, groups_train,
        np.round(np.arange(0.0, 1.0001, 0.01), 2),
    )
    _write_csv(results_dir / "phase04_b1_threshold_cv.csv", b1_rows)

    lr_params, lr_candidates = search_candidates(
        "logistic_regression", X_train, y_train, groups_train, registry,
        registry.model_features, logistic_candidates(), seed,
    )
    rf_params, rf_candidates = search_candidates(
        "random_forest", X_train, y_train, groups_train, registry,
        registry.model_features, random_forest_candidates(), seed,
    )
    _write_csv(results_dir / "phase04_grouped_cv_candidates.csv", lr_candidates + rf_candidates)

    lr = make_pipeline("logistic_regression", registry, registry.model_features, lr_params, seed).fit(X_train, y_train)
    lr_probability = lr.predict_proba(X_validation)[:, 1]
    rf_template = make_pipeline("random_forest", registry, registry.model_features, rf_params, seed)
    rf_oof, rf_fold_ids, calibration_folds = grouped_oof_probabilities(
        rf_template, X_train, y_train, groups_train,
    )
    fold_rows.extend(calibration_folds)
    _write_csv(results_dir / "phase04_grouped_cv_folds.csv", fold_rows)
    oof_records = [{
        "retrieval_condition_id": train.iloc[index].retrieval_condition_id,
        "question_id": train.iloc[index].question_id, "fold": int(rf_fold_ids[index]),
        "raw_oof_probability": float(rf_oof[index]), "y_suff_final": int(y_train[index]),
        "calibration_fit_partition": "train_oof", "question_overlap_count": 0,
    } for index in range(len(train))]
    write_canonical_parquet(
        results_dir / "phase04_oof_calibration_provenance.parquet", oof_records,
        tuple(oof_records[0]), ("fold", "question_id", "retrieval_condition_id"),
    )

    rf = rf_template.fit(X_train, y_train)
    rf_raw = rf.predict_proba(X_validation)[:, 1]
    calibrators = {
        method: ProbabilityCalibrator(method).fit(rf_oof, y_train)
        for method in ("uncalibrated", "sigmoid", "isotonic")
    }
    rf_probabilities = {method: calibrator.transform(rf_raw)
                        for method, calibrator in calibrators.items()}

    validation_metrics: list[dict[str, Any]] = []
    reliability: list[dict[str, Any]] = []
    b0_prediction = np.ones(len(y_validation), dtype=int)
    validation_metrics.append({
        "model": "B0_always_sufficient", "probability_method": "constant",
        "decision_threshold": None, "accuracy": float((b0_prediction == y_validation).mean()),
        "precision": float(precision_score(y_validation, b0_prediction)), "recall": 1.0,
        "f1": float(f1_score(y_validation, b0_prediction)), "auroc": None, "auprc": None,
        "brier": None, "ece": None,
    })
    b1_metrics, b1_bins = _probability_rows(
        "B1_idf_coverage_threshold", "raw_feature", y_validation,
        validation.idf_weighted_query_token_context_coverage.to_numpy(float),
        threshold=b1_threshold,
    )
    validation_metrics.append(b1_metrics); reliability.extend(b1_bins)
    lr_metrics, lr_bins = _probability_rows("B2_logistic_regression", "uncalibrated", y_validation, lr_probability)
    validation_metrics.append(lr_metrics); reliability.extend(lr_bins)
    calibration_rows = []
    for method, probability in rf_probabilities.items():
        name = "B3_random_forest" if method == "uncalibrated" else "B4_calibrated_random_forest"
        metrics, bins = _probability_rows(name, method, y_validation, probability)
        validation_metrics.append(metrics); reliability.extend(bins)
        calibration_rows.append(metrics.copy())
    selected_calibration_row = min(calibration_rows, key=lambda row: (row["brier"], row["ece"],
                                                                       row["probability_method"] != "uncalibrated"))
    selected_calibration = selected_calibration_row["probability_method"]
    for row in calibration_rows:
        row["selected_calibration"] = row["probability_method"] == selected_calibration
    _write_csv(results_dir / "phase04_rf_calibration_comparison.csv", calibration_rows)
    _write_csv(results_dir / "phase04_reliability_bins.csv", reliability)

    family_rows = [
        {"family": "logistic_regression", "calibration": "uncalibrated", **classifier_metrics(y_validation, lr_probability)},
        {"family": "random_forest", "calibration": selected_calibration,
         **classifier_metrics(y_validation, rf_probabilities[selected_calibration])},
    ]
    family_rows.sort(key=lambda row: (row["auprc"], row["auroc"], -row["brier"],
                                      row["family"] == "logistic_regression"), reverse=True)
    selected_family = family_rows[0]["family"]
    selected_params = lr_params if selected_family == "logistic_regression" else rf_params
    selected_calibration = "uncalibrated" if selected_family == "logistic_regression" else selected_calibration
    selected_pipeline = lr if selected_family == "logistic_regression" else rf
    selected_calibrator = (ProbabilityCalibrator("uncalibrated").fit(np.array([0., 1.]), np.array([0, 1]))
                           if selected_family == "logistic_regression" else calibrators[selected_calibration])
    selected_probability = (lr_probability if selected_family == "logistic_regression"
                            else rf_probabilities[selected_calibration])
    for row in family_rows:
        row["selected_family"] = row["family"] == selected_family
    for row in validation_metrics:
        row["selected_model"] = (
            (selected_family == "logistic_regression" and row["model"] == "B2_logistic_regression") or
            (selected_family == "random_forest" and row["model"] in ("B3_random_forest", "B4_calibrated_random_forest")
             and row["probability_method"] == selected_calibration)
        )
    _write_csv(results_dir / "phase04_model_validation_metrics.csv", validation_metrics)
    _write_json(results_dir / "phase04_logistic_selected_results.json", {
        "schema_version": "phase04-family-selected-result-v1",
        "model_family": "logistic_regression", "selected_hyperparameters": lr_params,
        "grouped_cv_selection": next(row for row in lr_candidates if row["selected"]),
        "validation_metrics": next(row for row in validation_metrics
                                   if row["model"] == "B2_logistic_regression"),
        "selected_primary_family": selected_family == "logistic_regression",
    })
    _write_json(results_dir / "phase04_random_forest_selected_results.json", {
        "schema_version": "phase04-family-selected-result-v1",
        "model_family": "random_forest", "selected_hyperparameters": rf_params,
        "grouped_cv_selection": next(row for row in rf_candidates if row["selected"]),
        "selected_calibration": selected_calibration,
        "validation_metrics": next(row for row in validation_metrics
                                   if row["model"] in ("B3_random_forest", "B4_calibrated_random_forest")
                                   and row["probability_method"] == selected_calibration),
        "selected_primary_family": selected_family == "random_forest",
    })

    model_path = model_dir / "phase04_selected_model.joblib"
    joblib.dump({
        "pipeline": selected_pipeline, "calibrator": selected_calibrator,
        "feature_names": list(registry.model_features), "modeling_config_sha256": PHASE04_CONFIG_SHA,
        "test_inference_permitted": False,
    }, model_path, compress=3)
    selected_config = _selected_config(
        root, registry, selected_family, selected_params, selected_calibration, model_path,
    )
    _write_json(root / "configs/phase04_selected_model.json", selected_config)
    selected_model_sha = selected_config["selected_model_config_sha256"]

    bootstrap = grouped_bootstrap_intervals(
        y_validation, selected_probability, groups_validation, replicates=1000, seed=seed,
    )
    _write_csv(results_dir / "phase04_bootstrap_confidence_intervals.csv", bootstrap)

    if selected_family == "logistic_regression":
        names = selected_pipeline.named_steps["columns"].get_feature_names_out()
        coefficients = selected_pipeline.named_steps["model"].coef_[0]
        importance_rows = sorted([
            {"feature": name, "standardized_coefficient": float(value),
             "absolute_coefficient": abs(float(value))}
            for name, value in zip(names, coefficients, strict=True)
        ], key=lambda row: (-row["absolute_coefficient"], row["feature"]))
        for rank, row in enumerate(importance_rows, 1): row["rank"] = rank
        _write_csv(results_dir / "phase04_logistic_coefficients.csv", importance_rows)
    else:
        importance = permutation_importance(
            selected_pipeline, X_validation, y_validation, scoring="average_precision",
            n_repeats=30, random_state=seed, n_jobs=-1,
        )
        importance_rows = sorted([
            {"feature": feature, "permutation_auprc_decrease_mean": float(mean),
             "permutation_auprc_decrease_std": float(std), "repeats": 30}
            for feature, mean, std in zip(registry.model_features,
                                          importance.importances_mean,
                                          importance.importances_std, strict=True)
        ], key=lambda row: (-row["permutation_auprc_decrease_mean"], row["feature"]))
        for rank, row in enumerate(importance_rows, 1): row["rank"] = rank
        _write_csv(results_dir / "phase04_rf_permutation_importance.csv", importance_rows)

    ablation_definitions = {
        "A": ("retrieval_score",),
        "B": ("query_context_lexical", "query_context_semantic"),
        "C": ("retrieval_score", "query_context_lexical", "query_context_semantic"),
        "D": tuple(registry.families),
    }
    ablation_rows = []
    for name, families in ablation_definitions.items():
        features = registry.features_for_families(families)
        fitted, calibrator, probability = _fit_with_calibration(
            selected_family, selected_params, selected_calibration, registry, features,
            X_train, y_train, groups_train, X_validation,
        )
        metrics = classifier_metrics(y_validation, probability)
        ablation_rows.append({
            "ablation": name, "families": list(families), "features": list(features),
            "feature_count": len(features), "metadata_included": "retrieval_condition_metadata" in families,
            "retuned": False, "auroc": metrics["auroc"], "auprc": metrics["auprc"], "f1": metrics["f1"],
        })
    _write_csv(results_dir / "phase04_feature_ablation.csv", ablation_rows)

    curve = risk_coverage_curve(y_validation, selected_probability)
    _write_csv(results_dir / "phase04_risk_coverage_curve.csv", curve)
    aurc_value = aurc(curve)
    _write_json(results_dir / "phase04_aurc.json", {
        "schema_version": "phase04-aurc-v1", "aurc": aurc_value,
        "integration": "right-endpoint rectangular over ascending observed coverage",
        "unit": "eligible validation retrieval condition", "model_config_sha256": selected_model_sha,
    })
    constraints = (0.05, 0.10, 0.20)
    operating_points = [select_risk_operating_point(curve, constraint) for constraint in constraints]
    _write_csv(results_dir / "phase04_risk_operating_points.csv", operating_points)

    validation_probabilities = validation[["question_id", "retrieval_strategy", "k", "y_suff_final"]].copy()
    validation_probabilities["probability"] = selected_probability
    hybrid = validation_probabilities[validation_probabilities.retrieval_strategy == "hybrid"]
    p5 = hybrid[hybrid.k == 5].set_index("question_id")
    p10 = hybrid[hybrid.k == 10].set_index("question_id")
    if set(p5.index) != set(p10.index) or len(p5) != 89:
        raise ValueError("hybrid validation trajectories are incomplete")
    trajectories = pd.DataFrame({
        "question_id": sorted(p5.index),
        "p5": [p5.loc[qid, "probability"] for qid in sorted(p5.index)],
        "p10": [p10.loc[qid, "probability"] for qid in sorted(p5.index)],
        "y5": [p5.loc[qid, "y_suff_final"] for qid in sorted(p5.index)],
        "y10": [p10.loc[qid, "y_suff_final"] for qid in sorted(p5.index)],
    })
    thresholds = np.round(np.arange(0.0, 1.0001, 0.02), 2)
    grid = policy_grid(trajectories, thresholds)
    _write_csv(results_dir / "phase04_policy_threshold_grid.csv", grid)
    policy_points = [select_policy(grid, constraint) for constraint in constraints]
    _write_csv(results_dir / "phase04_policy_operating_points.csv", policy_points)
    primary_policy = next(row for row in policy_points if row["risk_constraint"] == 0.10)
    if not primary_policy["feasible"]:
        raise RuntimeError("no feasible 10% three-way policy on the frozen threshold grid")
    trajectory_records = []
    for point in policy_points:
        if point["feasible"]:
            _, selected_trajectories = simulate_policy(
                trajectories, point["t_low"], point["t_high"]
            )
            for record in selected_trajectories:
                record["risk_constraint"] = point["risk_constraint"]
            trajectory_records.extend(selected_trajectories)
    _write_csv(results_dir / "phase04_policy_trajectories.csv", trajectory_records)

    baselines = []
    p0 = always_answer_k5(trajectories)
    for constraint in constraints:
        baselines.append({"policy": "P0_always_answer_hybrid_k5", "risk_constraint": constraint,
                          "feasible": p0["final_selective_risk"] <= constraint, **p0})
        baselines.append({"policy": "P1_two_way_hybrid_k5",
                          **select_two_way(trajectories, at_k=5, thresholds=thresholds,
                                           risk_constraint=constraint)})
        baselines.append({"policy": "P2_always_retrieve_hybrid_k10_then_two_way",
                          **select_two_way(trajectories, at_k=10, thresholds=thresholds,
                                           risk_constraint=constraint)})
    _write_csv(results_dir / "phase04_policy_baselines.csv", baselines)

    policy_config = {
        "schema_version": "phase04-selected-policy-v1",
        "selected_model_config_sha256": selected_model_sha,
        "phase04_modeling_config_sha256": PHASE04_CONFIG_SHA,
        "feature_registry_sha256": sha256_file(root / "configs/phase04_feature_registry.json"),
        "calibration_method": selected_calibration, "risk_constraint": 0.10,
        "t_low": primary_policy["t_low"], "t_high": primary_policy["t_high"],
        "retrieval_strategy": "hybrid", "initial_k": 5, "expanded_k": 10,
        "maximum_expansions": 1,
        "rule": "answer if p5>=t_high; abstain if p5<t_low; otherwise expand once to k10 and answer iff p10>=t_high",
        "test_adjustment_permitted": False,
    }
    policy_config["selected_policy_config_sha256"] = canonical_json_sha256(policy_config)
    _write_json(root / "configs/phase04_selected_policy.json", policy_config)

    # Separate post-freeze sensitivity: features and analysis cannot alter PRIMARY selection.
    impossible_targets = _benchmark_impossible_targets(root)
    impossible_rows = resources.make_rows(impossible_targets)
    write_canonical_parquet(
        data_dir / "phase04_benchmark_impossible_features.parquet", impossible_rows, fields,
        ("split", "question_id", "retrieval_strategy", "k"),
    )
    impossible = pq.read_table(data_dir / "phase04_benchmark_impossible_features.parquet").to_pandas()
    augmented_train = pd.concat([train, impossible[impossible.split == "train"]], ignore_index=True)
    sensitivity_validation = pd.concat([validation, impossible[impossible.split == "validation"]], ignore_index=True)
    X_aug = augmented_train.loc[:, registry.model_features]
    y_aug = augmented_train.y_suff_final.to_numpy(int)
    groups_aug = augmented_train.question_id.to_numpy(str)
    X_sens = sensitivity_validation.loc[:, registry.model_features]
    fitted_sens, calibrator_sens, sensitivity_probability = _fit_with_calibration(
        selected_family, selected_params, selected_calibration, registry, registry.model_features,
        X_aug, y_aug, groups_aug, X_sens,
    )
    overall_sensitivity = classifier_metrics(
        sensitivity_validation.y_suff_final.to_numpy(int), sensitivity_probability,
    )
    impossible_mask = sensitivity_validation.question_id.isin(
        set(impossible[impossible.split == "validation"].question_id)
    ).to_numpy()
    impossible_predicted_positive_rate = float((sensitivity_probability[impossible_mask] >= 0.5).mean())
    sensitivity_result = {
        "schema_version": "phase04-benchmark-impossible-sensitivity-v1",
        "timing": "after PRIMARY selected model and policy configurations were frozen",
        "architecture_changed": False, "hyperparameters_retuned": False,
        "family_reselected": False, "primary_thresholds_changed": False,
        "selected_family": selected_family, "selected_hyperparameters": selected_params,
        "calibration_method": selected_calibration,
        "augmented_train_questions": int(augmented_train.question_id.nunique()),
        "augmented_train_conditions": len(augmented_train),
        "added_impossible_train_questions": int(impossible[impossible.split == "train"].question_id.nunique()),
        "added_impossible_train_conditions": int((impossible.split == "train").sum()),
        "evaluation_questions": int(sensitivity_validation.question_id.nunique()),
        "evaluation_conditions": len(sensitivity_validation),
        "added_impossible_validation_questions": int(impossible[impossible.split == "validation"].question_id.nunique()),
        "added_impossible_validation_conditions": int((impossible.split == "validation").sum()),
        "combined_validation_metrics": overall_sensitivity,
        "impossible_validation_predicted_sufficient_rate_at_0_5": impossible_predicted_positive_rate,
        "influences_primary_conclusions": False,
    }
    _write_json(results_dir / "phase04_benchmark_impossible_sensitivity.json", sensitivity_result)

    versions = {name: importlib.metadata.version(name) for name in
                ("numpy", "pandas", "pyarrow", "scikit-learn", "sentence-transformers", "joblib")}
    integrity = {
        "schema_version": "phase04-integrity-report-v1", "status": "pass",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "upstream_verified": True, "phase03_manifest_immutable": True,
        "primary_population": {"questions": 513, "conditions": 6156, "positive": 2802, "negative": 3354},
        "train_population": {"questions": 424, "conditions": 5088},
        "validation_population": {"questions": 89, "conditions": 1068},
        "feature_count": len(registry.model_features), "feature_leakage_check": "pass",
        "grouped_cv_fold_count": 5,
        "grouped_cv_zero_question_overlap": all(row["zero_question_overlap"] for row in fold_rows),
        "policy_grid_pairs": len(grid), "test_rows": 0, "test_inference_run": False,
        "test_aggregate_results": False, "test_sealed": True,
        "phase5_started": False, "library_versions": versions,
        "selected_model_config_sha256": selected_model_sha,
        "selected_policy_config_sha256": policy_config["selected_policy_config_sha256"],
    }
    _write_json(results_dir / "phase04_integrity_report.json", integrity)

    manifest_paths = phase04_manifest_paths(selected_family)
    manifest = {
        "schema_version": "phase04-artifact-manifest-v1",
        "phase03_upstream_manifest_sha256": UPSTREAM["phase03_final_manifest"][1],
        "phase04_modeling_config_sha256": PHASE04_CONFIG_SHA,
        "artifacts": [{"path": relative, "physical_sha256": sha256_file(root / relative),
                       "bytes": (root / relative).stat().st_size} for relative in manifest_paths],
        "manifest_includes_itself": False, "test_sealed": True, "phase5_started": False,
    }
    _write_json(results_dir / "phase04_artifact_manifest.json", manifest)
    return {
        "status": "complete", "selected_family": selected_family,
        "selected_calibration": selected_calibration, "selected_model_config_sha256": selected_model_sha,
        "selected_policy_config_sha256": policy_config["selected_policy_config_sha256"],
        "manifest_sha256": sha256_file(results_dir / "phase04_artifact_manifest.json"),
        "feature_artifact": feature_artifact,
    }
