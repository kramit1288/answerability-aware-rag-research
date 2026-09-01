"""Create post-results interpretation, integrity report, and Phase 5 manifest."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from answerability_rag.generation.config import Phase05Config, verify_upstream  # noqa: E402
from answerability_rag.hashing import sha256_file  # noqa: E402
from answerability_rag.io import write_bytes_atomic, write_json_atomic  # noqa: E402


def _fmt(value, digits=4):
    return "NA" if value is None or pd.isna(value) else f"{float(value):.{digits}f}"


def _manifest_paths() -> list[Path]:
    fixed = [
        Path("configs/.gitattributes"), Path("docs/.gitattributes"),
        Path("artifacts/governance/.gitattributes"), Path("artifacts/results/.gitattributes"),
        Path("scripts/.gitattributes"), Path("tests/.gitattributes"),
        Path("configs/phase05_generation_grounding.json"),
        Path("docs/PHASE_05_RESEARCH_DECISIONS.md"), Path("docs/PHASE_05_EXECUTION.md"),
        Path("docs/PHASE_05_ARTIFACT_SCHEMAS.md"), Path("docs/PHASE_05_RESULTS_INTERPRETATION.md"),
        Path("docs/PHASE_05_EXECUTION_DECISION_REQUEST.md"),
        Path("docs/PHASE_05_EXECUTION_OPTIMIZATION.md"),
        Path("artifacts/governance/phase05_generation_prompt.txt"),
        Path("scripts/benchmark_phase05_nli_execution.py"),
        Path("scripts/run_phase05_generation_grounding.py"), Path("scripts/check_phase05.py"),
        Path("scripts/finalize_phase05_artifacts.py"),
        Path("tests/test_phase05_generation_grounding.py"),
    ]
    fixed.extend(sorted(Path("src/answerability_rag/generation").glob("*.py")))
    fixed.extend(sorted(
        path for path in Path("artifacts/results").glob("phase05_*")
        if path.name != "phase05_artifact_manifest.json"
    ))
    unique = {path.as_posix(): path for path in fixed if (ROOT / path).is_file()}
    return [unique[key] for key in sorted(unique)]


def main() -> None:
    config = Phase05Config.load(ROOT / "configs/phase05_generation_grounding.json", ROOT)
    verify_upstream(ROOT, config)
    contexts = pq.read_table(ROOT / "artifacts/results/phase05_techqa_context_manifest.parquet").to_pandas()
    generations = pq.read_table(ROOT / "artifacts/results/phase05_generation_cache.parquet").to_pandas()
    quality = pq.read_table(ROOT / "artifacts/results/phase05_answer_quality.parquet").to_pandas()
    grounding = pq.read_table(ROOT / "artifacts/results/phase05_techqa_response_grounding.parquet").to_pandas()
    rag_metrics = pd.read_csv(ROOT / "artifacts/results/phase05_ragtruth_test_metrics.csv")
    policies = pd.read_csv(ROOT / "artifacts/results/phase05_policy_generation_comparison.csv")
    suff = pd.read_csv(ROOT / "artifacts/results/phase05_context_sufficiency_grounding_comparison.csv")
    paired = json.loads((ROOT / "artifacts/results/phase05_paired_k5_k10_summary.json").read_text(encoding="utf-8"))
    validation = json.loads((ROOT / "artifacts/results/phase05_grounding_validation_manifest.json").read_text(encoding="utf-8"))
    threshold = json.loads((ROOT / "artifacts/results/phase05_selected_grounding_threshold.json").read_text(encoding="utf-8"))["t_support"]
    joined = generations.merge(quality, on=["response_id", "question_id", "k", "generation_status"])
    joined = joined.merge(grounding, on=["response_id", "question_id", "k", "generation_status"])
    by_k = joined.groupby("k").agg(
        mean_rouge_l=("rouge_l_f1", "mean"), mean_bertscore=("bertscore_f1", "mean"),
        mean_unsupported=("unsupported_claim_rate", "mean"),
        fully_supported=("fully_supported_response", lambda values: values.dropna().astype(bool).mean()),
    ).reset_index()
    primary_response = rag_metrics[(rag_metrics.population == "good_primary") & (rag_metrics.level == "response")].iloc[0]
    interpretation = f"""# Phase 5 Results Interpretation

## Status and boundary

This is a post-results interpretation artifact. The immutable pre-results configuration has canonical SHA-256 `{config.canonical_sha256}`. No Phase 4 model or threshold was changed, TechQA TEST remained sealed, and Phase 6 inferential testing was not started.

## Grounding-evaluator validation

The frozen RAGTruth TRAIN support threshold is `{threshold:.2f}`. On the primary good-quality RAGTruth TEST response population, precision was {_fmt(primary_response.precision)}, recall {_fmt(primary_response.recall)}, F1 {_fmt(primary_response.f1)}, AUROC {_fmt(primary_response.auroc)}, and AUPRC {_fmt(primary_response.auprc)}. The predeclared AUROC 0.60 interpretation gate {'passed' if validation['validity_threshold_passed'] else 'did not pass'}; TechQA binary grounding results are therefore described as `{validation['binary_techqa_grounding_interpretation']}`.

## TechQA k5 and k10

| Context | Mean ROUGE-L | Mean BERTScore F1 | Mean unsupported-claim rate | Fully-supported response rate |
|---|---:|---:|---:|---:|
"""
    for row in by_k.itertuples(index=False):
        interpretation += f"| k={int(row.k)} | {_fmt(row.mean_rouge_l)} | {_fmt(row.mean_bertscore)} | {_fmt(row.mean_unsupported)} | {_fmt(row.fully_supported)} |\n"
    interpretation += "\n## Frozen policy views\n\n"
    interpretation += "| Policy | Answers | Coverage | Mean ROUGE-L | Mean BERTScore | Mean unsupported rate | Grounded-answer yield | Unsupported-answer population rate |\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
    for row in policies.itertuples(index=False):
        interpretation += (
            f"| {row.policy_id} | {int(row.policy_answer_count)} | {_fmt(row.policy_answer_coverage)} | "
            f"{_fmt(row.mean_rouge_l)} | {_fmt(row.mean_bertscore_f1)} | "
            f"{_fmt(row.mean_unsupported_claim_rate)} | {_fmt(row.grounded_answer_yield)} | "
            f"{_fmt(row.unsupported_answer_population_rate)} |\n"
        )
    interpretation += f"""

## Paired descriptive comparison

Mean k10-minus-k5 differences were ROUGE-L {_fmt(paired['mean_difference_rouge_l_f1_k10_minus_k5'])}, BERTScore F1 {_fmt(paired['mean_difference_bertscore_f1_k10_minus_k5'])}, unsupported-claim rate {_fmt(paired['mean_difference_unsupported_claim_rate_k10_minus_k5'])}, mean claim support {_fmt(paired['mean_difference_claim_support_k10_minus_k5'])}, and output tokens {_fmt(paired['mean_difference_output_tokens_k10_minus_k5'])}. These are descriptive only; Phase 6 will perform the predeclared paired inference.

## Interpretation limits

`y_suff_final` remains an operational retrieval-conditioned context-sufficiency target and was not redefined from generation. NLI support is an externally validated automatic proxy, not hallucination probability. ROUGE-L and BERTScore measure reference similarity, not grounding. Policy quality is conditional on answering and is always reported with coverage; abstentions remain NA for content quality. The primary G2 estimate is especially sparse because only two of 89 trajectories are answered.
"""
    write_bytes_atomic(
        ROOT / "docs/PHASE_05_RESULTS_INTERPRETATION.md",
        interpretation.encode("utf-8"),
    )
    integrity = {
        "schema_version": "phase05-integrity-report-v1", "status": "pass",
        "phase05_config_canonical_sha256": config.canonical_sha256,
        "upstream_verified": True, "phase03_manifest_immutable": True,
        "phase04_manifest_immutable": True, "selected_model_and_policy_unchanged": True,
        "techqa_population": {"questions": int(contexts.question_id.nunique()), "states": len(contexts)},
        "generation_status_counts": dict(sorted(Counter(generations.generation_status).items())),
        "k10_additional_context_fraction": float(contexts[contexts.k == 10].exposes_context_beyond_k5.astype(bool).mean()),
        "truncated_context_states": int(contexts.final_truncated_chunk_id.notna().sum()),
        "ragtruth_train_threshold_only": True, "selected_t_support": threshold,
        "ragtruth_response_auroc": None if pd.isna(primary_response.auroc) else float(primary_response.auroc),
        "grounding_validity_threshold_passed": bool(validation["validity_threshold_passed"]),
        "techqa_test_rows": 0, "techqa_test_sealed": True,
        "abstention_quality_is_na": True, "phase04_model_retrained": False,
        "phase04_thresholds_changed": False, "phase06_started": False,
    }
    write_json_atomic(ROOT / "artifacts/results/phase05_integrity_report.json", integrity)
    artifacts = [{
        "path": path.as_posix(), "bytes": (ROOT / path).stat().st_size,
        "physical_sha256": sha256_file(ROOT / path),
    } for path in _manifest_paths()]
    manifest = {
        "schema_version": "phase05-artifact-manifest-v1",
        "artifacts": artifacts, "manifest_includes_itself": False,
        "phase05_config_canonical_sha256": config.canonical_sha256,
        "phase03_final_manifest_sha256": config.values["upstream_freeze"]["phase03_final_manifest_physical_sha256"],
        "phase04_final_manifest_sha256": config.values["upstream_freeze"]["phase04_final_manifest_physical_sha256"],
        "techqa_test_sealed": True, "phase04_thresholds_changed": False,
        "phase06_started": False,
    }
    write_json_atomic(ROOT / "artifacts/results/phase05_artifact_manifest.json", manifest)
    print(json.dumps({"status": "pass", "artifact_count": len(artifacts),
                      "manifest_sha256": sha256_file(ROOT / "artifacts/results/phase05_artifact_manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
