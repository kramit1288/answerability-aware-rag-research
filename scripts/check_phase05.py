"""Independent Phase 5 integrity, leakage, grouping, and TEST-seal checker."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from answerability_rag.generation.config import Phase05Config, verify_upstream  # noqa: E402
from answerability_rag.hashing import sha256_file  # noqa: E402


def require(condition: bool, message: str, checks: list[str]) -> None:
    if not condition:
        raise AssertionError(message)
    checks.append(message)


def main() -> None:
    checks: list[str] = []
    config = Phase05Config.load(ROOT / "configs/phase05_generation_grounding.json", ROOT)
    verify_upstream(ROOT, config)
    require(True, "all frozen Phase 3/4 physical and canonical hashes reproduce", checks)
    freeze = json.loads((ROOT / "artifacts/results/phase05_pre_results_governance_freeze.json").read_text(encoding="utf-8"))
    require(not freeze["techqa_generation_started_before_freeze"]
            and not freeze["techqa_generation_results_observed_before_freeze"]
            and not freeze["techqa_metric_results_observed_before_freeze"],
            "Phase 5 governance freeze predates every TechQA result", checks)

    contexts = pq.read_table(ROOT / "artifacts/results/phase05_techqa_context_manifest.parquet").to_pandas()
    require(len(contexts) == 178 and contexts.question_id.nunique() == 89
            and set(contexts.k) == {5, 10} and set(contexts.retrieval_strategy) == {"hybrid"}
            and set(contexts.split) == {"validation"},
            "exactly 178 VALIDATION-only hybrid k5/k10 contexts are frozen", checks)
    require(contexts.groupby("question_id").k.nunique().eq(2).all(),
            "every eligible question has paired k5/k10 states", checks)
    require(contexts[contexts.k == 10].exposes_context_beyond_k5.astype(bool).all(),
            "every k10 state exposes additional prompt-visible context", checks)
    require(contexts.final_truncated_chunk_id.isna().all()
            and (contexts.retrieved_chunks_not_included == 0).all(),
            "all frozen k5/k10 chunks fit without prompt truncation", checks)
    require(not any(token in text for text in contexts.rendered_prompt.astype(str) for token in (
        "y_suff_final", "model_probability", "gold_evidence", "benchmark_answer",
    )), "persisted generation prompts contain no forbidden field names", checks)

    generations = pq.read_table(ROOT / "artifacts/results/phase05_generation_cache.parquet").to_pandas()
    require(len(generations) == 178 and generations.question_id.nunique() == 89
            and generations.cache_key.nunique() == 178,
            "all 178 generation states have unique frozen cache identities", checks)
    require(set(generations.generation_status).issubset({"generated", "empty_output", "generation_failed"}),
            "generation statuses use only frozen explicit values", checks)
    require(generations.loc[generations.generation_status == "generation_failed", "raw_generated_text"].isna().all(),
            "generation failures are not converted to empty answers", checks)
    require((generations.attempt_count <= 2).all(), "generation retries obey the frozen bound", checks)

    quality = pq.read_table(ROOT / "artifacts/results/phase05_answer_quality.parquet").to_pandas()
    require(len(quality) == 178 and set(quality.response_id) == set(generations.response_id),
            "quality records cover exactly the generation-state population", checks)
    undefined = quality.metric_status != "evaluated"
    require(quality.loc[undefined, ["rouge_l_f1", "bertscore_f1"]].isna().all().all(),
            "undefined answer-quality metrics remain NA", checks)

    audit = json.loads((ROOT / "artifacts/results/phase05_ragtruth_schema_alignment_audit.json").read_text(encoding="utf-8"))
    require(audit["qa_train_responses"] == 5034 and audit["qa_test_responses"] == 900
            and audit["qa_train_sources"] == 839 and audit["qa_test_sources"] == 150
            and not audit["source_id_crossings"],
            "RAGTruth QA census and official source isolation reproduce", checks)
    require(audit["claim_level_alignment_supported"] and audit["offset_mismatch_count"] == 0,
            "all supplied RAGTruth QA span offsets align", checks)
    selected = json.loads((ROOT / "artifacts/results/phase05_selected_grounding_threshold.json").read_text(encoding="utf-8"))
    threshold_grid = pd.read_csv(ROOT / "artifacts/results/phase05_ragtruth_threshold_search.csv")
    require(len(threshold_grid) == 51 and threshold_grid.selected.sum() == 1
            and np.isclose(threshold_grid.loc[threshold_grid.selected, "t_support"].iloc[0], selected["t_support"]),
            "complete RAGTruth TRAIN threshold grid has one frozen selection", checks)
    require(selected["selection_split"] == "ragtruth_official_train"
            and selected["ragtruth_test_labels_accessed_for_selection"] is False,
            "RAGTruth TEST labels did not influence threshold selection", checks)
    bootstrap = pd.read_csv(ROOT / "artifacts/results/phase05_ragtruth_bootstrap_intervals.csv")
    require((bootstrap.requested_replicates == 1000).all()
            and (bootstrap.resampling_unit == "source_id").all(),
            "RAGTruth intervals use 1,000 source-cluster bootstrap replicates", checks)

    claims = pq.read_table(ROOT / "artifacts/results/phase05_techqa_claim_grounding.parquet").to_pandas()
    responses = pq.read_table(ROOT / "artifacts/results/phase05_techqa_response_grounding.parquet").to_pandas()
    require(len(responses) == 178 and responses.question_id.nunique() == 89,
            "TechQA grounding covers every generation state", checks)
    require(responses.loc[responses.evaluable_claim_count == 0, "unsupported_claim_rate"].isna().all(),
            "zero-evaluable-claim unsupported rates remain NA", checks)
    require((claims.candidate_chunk_ids_json.apply(lambda value: len(json.loads(value))) <= 3).all(),
            "grounding evaluates at most the frozen three candidate chunks", checks)

    policy_summary = pd.read_csv(ROOT / "artifacts/results/phase05_policy_generation_comparison.csv")
    require(set(policy_summary.policy_id) == {"G0", "G1", "G2", "G3"},
            "G0/G1/G2/G3 policy comparisons are complete", checks)
    g2 = policy_summary[policy_summary.policy_id == "G2"].iloc[0]
    g3 = policy_summary[policy_summary.policy_id == "G3"].iloc[0]
    require(g2.policy_answer_count == 2 and g2.abstention_count == 87
            and np.isclose(g2.policy_answer_coverage, 2 / 89),
            "frozen primary 10% generation view reproduces 2 answers and 87 abstentions", checks)
    require(g3.policy_answer_count == 16 and g3.abstention_count == 73
            and np.isclose(g3.policy_answer_coverage, 16 / 89),
            "frozen 20% sensitivity view reproduces 16 answers and 73 abstentions", checks)
    g2_rows = pq.read_table(ROOT / "artifacts/results/phase05_policy_G2.parquet").to_pandas()
    require(g2_rows.loc[~g2_rows.answered.astype(bool), ["rouge_l_f1", "bertscore_f1", "unsupported_claim_rate"]].isna().all().all(),
            "policy abstentions retain NA quality and grounding", checks)
    paired = pq.read_table(ROOT / "artifacts/results/phase05_paired_k5_k10.parquet").to_pandas()
    require(len(paired) == 89 and paired.question_id.nunique() == 89,
            "paired k5/k10 Phase 6 input has exactly one row per question", checks)

    manifest_path = ROOT / "artifacts/results/phase05_artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(all(sha256_file(ROOT / item["path"]) == item["physical_sha256"]
                for item in manifest["artifacts"]),
            "every Phase 5 manifest entry reproduces byte-for-byte", checks)
    require(manifest["techqa_test_sealed"] and not manifest["phase06_started"],
            "TechQA TEST remains sealed and Phase 6 remains unstarted", checks)
    require(not any("test" in str(value).casefold() and value != "validation"
                    for value in responses.get("split", pd.Series(dtype=str)).tolist()),
            "no TechQA TEST result row is present", checks)
    print(json.dumps({"status": "pass", "check_count": len(checks), "checks": checks,
                      "phase05_config_canonical_sha256": config.canonical_sha256,
                      "phase05_manifest_sha256": sha256_file(manifest_path),
                      "techqa_test_sealed": True, "phase06_started": False}, indent=2))


if __name__ == "__main__":
    main()
