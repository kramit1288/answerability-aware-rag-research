"""Phase 7 final TEST governance, leakage, alignment, and immutability checks."""

from __future__ import annotations

import inspect
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from answerability_rag.final_test import evaluation, generation, grounding
from answerability_rag.final_test.common import PHASE07_CONFIG_SHA256, Phase07Config
from answerability_rag.final_test.integrity import verify_post_test_integrity
from answerability_rag.hashing import sha256_file
from answerability_rag.statistics.core import cluster_bootstrap_samples


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase07_final_test.json"


def test_governance_precedes_single_unseal() -> None:
    freeze = json.loads((ROOT / "artifacts/results/phase07_pre_test_governance_freeze.json").read_text(encoding="utf-8"))
    integrity = json.loads((ROOT / "artifacts/results/phase07_upstream_integrity_pre_unseal.json").read_text(encoding="utf-8"))
    unseal = json.loads((ROOT / "artifacts/results/phase07_test_unseal_record.json").read_text(encoding="utf-8"))
    assert freeze["no_post_test_scientific_choice_statement"] == "No post-TEST scientific choice is permitted."
    assert datetime.fromisoformat(freeze["freeze_timestamp"]) < datetime.fromisoformat(integrity["checked_at"]) < datetime.fromisoformat(unseal["unseal_timestamp"])
    assert not freeze["techqa_test_scientific_content_accessed_before_freeze"]
    assert unseal["declarations"]["test_unsealed_exactly_once"]


def test_upstream_hashes_remain_immutable() -> None:
    config = Phase07Config.load(CONFIG)
    assert config.canonical_sha256 == PHASE07_CONFIG_SHA256
    result = verify_post_test_integrity(ROOT, CONFIG, require_complete=False)
    assert result["status"] == "pass"
    assert result["upstream_hashes_match_unseal_record"]
    assert not result["post_test_tuning_detected"]


def test_no_test_fit_or_threshold_selection_code_path() -> None:
    source = inspect.getsource(evaluation.run_test_inference)
    assert ".fit(" not in source and ".fit_transform(" not in source
    assert "predict_proba" in source
    assert "_frozen_threshold_rows" in source
    assert "test_threshold_selected" in source


def test_fixed_features_model_and_thresholds() -> None:
    config = Phase07Config.load(CONFIG).values
    selected = json.loads((ROOT / "configs/phase04_selected_model.json").read_text(encoding="utf-8"))
    policy = json.loads((ROOT / "configs/phase04_selected_policy.json").read_text(encoding="utf-8"))
    assert len(selected["features"]) == config["features_and_model"]["feature_count"] == 39
    assert sha256_file(ROOT / selected["model_artifact_path"]) == config["upstream_freeze"]["phase04_model_binary_physical_sha256"]
    assert (policy["t_low"], policy["t_high"]) == (0.78, 0.82)
    assert config["policies"]["sensitivity_20_percent"] == {"policy_id": "G3", "t_low": 0.56, "t_high": 0.72}
    assert config["risk_coverage"]["test_threshold_selection_permitted"] is False


def test_frozen_generation_and_grounding() -> None:
    config = Phase07Config.load(CONFIG).values
    assert config["generation"]["prompt_sha256"] == sha256_file(ROOT / config["generation"]["prompt_path"])
    assert config["generation"]["model_revision"] == "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    assert config["generation"]["do_sample"] is False and config["generation"]["max_new_tokens"] == 128
    assert config["grounding"]["candidate_count"] == 3
    assert config["grounding"]["support_threshold"] == 0.16
    assert "select_support_threshold" not in inspect.getsource(grounding.evaluate_test_grounding)


def test_test_eligibility_and_primary_sensitivity_separation() -> None:
    census = json.loads((ROOT / "artifacts/results/phase07_test_population_census.json").read_text(encoding="utf-8"))
    exclusions = pd.read_csv(ROOT / "artifacts/results/phase07_test_exclusion_manifest.csv")
    target = pq.read_table(ROOT / "artifacts/results/phase07_test_final_target.parquet").to_pandas()
    assert census["total_test_questions"] == 136
    assert census["benchmark_answerable_test_questions"] == 91
    assert census["benchmark_impossible_test_questions"] == 45
    assert len(target) == census["primary_eligible_retrieval_conditions"]
    assert target.question_id.nunique() == census["primary_eligible_test_questions"]
    impossible_ids = set(exclusions.loc[exclusions.population_role == "benchmark_impossible_sensitivity", "question_id"])
    assert impossible_ids and impossible_ids.isdisjoint(set(target.question_id))


def test_policy_trajectories_use_frozen_mechanics() -> None:
    rows = pq.read_table(ROOT / "artifacts/results/phase07_test_policy_trajectories.parquet").to_pandas()
    for row in rows.itertuples(index=False):
        if row.p5 >= row.t_high:
            assert row.final_action == "ANSWER_AT_K5" and not row.expansion_triggered
        elif row.p5 < row.t_low:
            assert row.final_action == "ABSTAIN" and not row.expansion_triggered
        else:
            assert row.expansion_triggered and row.final_action in {"ANSWER_AT_K10", "ABSTAIN"}


def test_abstention_quality_is_na() -> None:
    for policy in ("G2", "G3"):
        frame = pq.read_table(ROOT / f"artifacts/results/phase07_test_policy_{policy}.parquet").to_pandas()
        abstained = frame.loc[~frame.answered.astype(bool)]
        for column in ("rouge_l_f1", "bertscore_f1", "unsupported_claim_rate", "fully_supported_response"):
            assert abstained[column].isna().all()


def test_grouped_bootstrap_preserves_cluster_rows_and_occurrences() -> None:
    frame = pd.DataFrame({"question_id": ["a", "a", "b", "b"], "value": [1, 2, 3, 4]})
    sample = next(cluster_bootstrap_samples(frame, "question_id", replicates=1, seed=42))
    assert len(sample) == 4
    assert "_bootstrap_occurrence" in sample.columns
    assert sample.groupby("_bootstrap_occurrence").size().eq(2).all()


def test_paired_alignment_and_holm_families() -> None:
    paired = pd.read_csv(ROOT / "artifacts/results/phase07_test_paired_continuous_statistics.csv")
    tests = pd.read_csv(ROOT / "artifacts/results/phase07_test_statistical_tests.csv")
    assert (paired.n_pairs + paired.excluded_pairs).nunique() == 1
    assert tests.groupby("family_id").size().to_dict() == {
        "A_paired_k5_k10_continuous": 5,
        "B_sufficiency_association": 8,
        "C_paired_k5_k10_binary": 2,
    }
    assert tests.p_holm.between(0, 1).all()


def test_final_no_tuning_and_no_phase8_implementation() -> None:
    integrity = json.loads((ROOT / "artifacts/results/phase07_post_test_integrity.json").read_text(encoding="utf-8"))
    assert integrity["status"] == "pass"
    assert integrity["post_test_tuning_detected"] is False
    assert not list((ROOT / "configs").glob("phase08*"))
    assert not list((ROOT / "docs").glob("PHASE_08*"))
