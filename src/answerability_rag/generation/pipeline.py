"""Resumable Phase 5 execution stages over frozen VALIDATION and RAGTruth inputs."""

from __future__ import annotations

import gc
import json
import math
import os
import platform
import random
import statistics
import time
import unicodedata
from collections import Counter, defaultdict
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from answerability_rag.data.ragtruth import (
    load_ragtruth_responses,
    load_ragtruth_sources,
    validate_ragtruth,
)
from answerability_rag.hashing import canonical_json_sha256, sha256_file, sha256_text
from answerability_rag.io import write_csv_atomic, write_json_atomic
from answerability_rag.retrieval.artifacts import read_parquet_records, write_canonical_parquet

from .claims import segment_claims
from .config import Phase05Config, assert_techqa_split_allowed, verify_upstream
from .context import assemble_ranked_context, context_identity
from .grounding import (
    GroundingModels,
    aggregate_candidate_nli,
    candidate_json,
    discrimination_metrics,
    parse_ragtruth_passages,
    response_grounding_metrics,
    select_candidate_chunks,
    select_support_threshold,
)
from .policy import build_policy_view
from .quality import FrozenBERTScorer, rouge_l_f1


RESULTS = Path("artifacts/results")
DERIVED = Path("data/derived/phase05")
EXECUTION_AMENDMENT = RESULTS / "phase05_execution_optimization.json"
CONTEXT_FIELDS = (
    "schema_version", "response_id", "question_id", "split", "retrieval_strategy", "k",
    "question", "ordered_retrieved_chunk_ids_json", "prompt_visible_chunk_ids_json",
    "prompt_visible_chunks_json", "assembled_context", "assembled_context_sha256",
    "rendered_prompt", "rendered_prompt_sha256", "input_token_count",
    "fully_included_chunk_count", "final_truncated_chunk_id",
    "final_truncated_chunk_original_tokens", "final_truncated_chunk_included_tokens",
    "retrieved_chunks_not_included", "k5_context_prefix_sha256",
    "exposes_context_beyond_k5", "context_identity_sha256", "prompt_sha256",
    "generation_config_sha256", "y_suff_final",
)
GENERATION_FIELDS = (
    "schema_version", "cache_key", "response_id", "question_id", "retrieval_strategy", "k",
    "model_revision", "tokenizer_revision", "generation_config_sha256", "prompt_sha256",
    "assembled_context_sha256", "raw_generated_text", "normalized_generated_text",
    "input_token_count", "output_token_count", "generation_status", "attempt_count",
    "runtime_seconds", "runtime_metadata_json", "error_class", "error_message_sha256",
    "output_sha256",
)
QUALITY_FIELDS = (
    "schema_version", "response_id", "question_id", "k", "generation_status",
    "reference_status", "rouge_l_f1", "bertscore_f1", "metric_status",
    "generation_cache_sha256",
)
CLAIM_FIELDS = (
    "schema_version", "dataset", "response_id", "source_id", "question_id",
    "official_split", "quality", "claim_id", "claim_index", "claim_text", "claim_start",
    "claim_end", "human_unsupported", "candidate_chunk_ids_json",
    "candidate_similarities_json", "candidate_nli_json", "claim_token_length",
    "pair_special_token_count", "evaluation_status", "claim_support_score",
    "claim_unsupportedness_score", "maximum_claim_contradiction", "supporting_chunk_id",
    "grounding_config_sha256",
)


def _package_versions() -> dict[str, str]:
    names = (
        "torch", "transformers", "tokenizers", "huggingface-hub", "safetensors",
        "sentence-transformers", "bert-score", "rouge-score", "numpy", "pandas",
        "pyarrow", "scikit-learn",
    )
    return {name: metadata.version(name) for name in names}


def _generation_config_sha(config: Phase05Config) -> str:
    return canonical_json_sha256({
        "generator": config.values["generator"],
        "prompt": config.values["prompt"],
        "context_assembly": config.values["context_assembly"],
        "generation": config.values["generation"],
    })


def _grounding_config_sha(config: Phase05Config) -> str:
    return canonical_json_sha256({
        "claim_segmentation": config.values["claim_segmentation"],
        "grounding_evaluator": config.values["grounding_evaluator"],
        "ragtruth_validation": config.values["ragtruth_validation"],
    })


def _apply_execution_amendment(root: Path, config: Phase05Config) -> dict[str, Any]:
    """Apply the approved CPU-only execution amendment before NLI inference."""
    path = root / EXECUTION_AMENDMENT
    if not path.exists():
        raise FileNotFoundError(f"approved Phase 5 execution amendment is missing: {path}")
    amendment = json.loads(path.read_text(encoding="utf-8"))
    if amendment.get("scientific_config_canonical_sha256") != config.canonical_sha256:
        raise ValueError("execution amendment does not match the frozen Phase 5 configuration")
    selected = amendment.get("selected_execution", {})
    if selected.get("device") != "cpu" or selected.get("dtype") != "float32":
        raise ValueError("only CPU float32 execution is permitted by the Phase 5 amendment")
    if int(selected.get("nli_batch_size", -1)) not in {16, 32, 64}:
        raise ValueError("execution amendment selected an unbenchmarked NLI batch size")
    torch.set_num_threads(int(selected["torch_num_threads"]))
    torch.set_num_interop_threads(int(selected["torch_num_interop_threads"]))
    return amendment


def _artifact(path: Path, row_count: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "path": path.as_posix(), "bytes": path.stat().st_size,
        "physical_sha256": sha256_file(path),
    }
    if row_count is not None:
        value["rows"] = row_count
    return value


def _mean(values: Iterable[Any]) -> float | None:
    clean = [float(value) for value in values if value is not None and not pd.isna(value)]
    return sum(clean) / len(clean) if clean else None


def _median(values: Iterable[Any]) -> float | None:
    clean = [float(value) for value in values if value is not None and not pd.isna(value)]
    return statistics.median(clean) if clean else None


def _normalize_generated_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    return " ".join(normalized.replace("\r\n", "\n").replace("\r", "\n").split()).strip()


def _load_raw_techqa(root: Path, config: Phase05Config, question_ids: set[str]) -> dict[str, dict[str, Any]]:
    assert_techqa_split_allowed(config.values["techqa_population"]["split"], "load TechQA questions")
    source = root / config.values["techqa_population"]["source_questions_and_references"]
    records = json.loads(source.read_text(encoding="utf-8"))
    selected = {str(row["id"]): row for row in records if str(row["id"]) in question_ids}
    if set(selected) != question_ids:
        raise ValueError(f"TechQA source is missing allowed IDs: {sorted(question_ids - set(selected))}")
    return selected


def _ragtruth_audit(root: Path, config: Phase05Config) -> dict[str, Any]:
    cfg = config.values["ragtruth_validation"]
    sources = load_ragtruth_sources(root / cfg["sources_path"])
    responses = load_ragtruth_responses(root / cfg["responses_path"])
    validation = validate_ragtruth(sources, responses)
    if not validation.passed:
        raise ValueError("RAGTruth pinned validation failed")
    qa_ids = {source.source_id for source in sources if source.task_type == "QA"}
    qa = [response for response in responses if response.source_id in qa_ids]
    split_sources: dict[str, set[str]] = defaultdict(set)
    for response in qa:
        split_sources[response.official_split].add(response.source_id)
    offset_mismatches = []
    for response in qa:
        for index, label in enumerate(response.labels):
            if response.response[label["start"]:label["end"]] != label["text"]:
                offset_mismatches.append({"response_id": response.response_id, "label_index": index})
    parsed_counts = Counter(len(parse_ragtruth_passages(
        next(source for source in sources if source.source_id == source_id).source_info["passages"]
    )) for source_id in qa_ids)
    source_crossing = split_sources["train"] & split_sources["test"]
    result = {
        "schema_version": "phase05-ragtruth-schema-alignment-audit-v1",
        "dataset_revision": cfg["dataset_revision"],
        "source_fields": ["source_id", "task_type", "source", "source_info", "prompt"],
        "response_fields": ["id", "source_id", "model", "temperature", "labels", "split", "quality", "response"],
        "label_fields": ["start", "end", "text", "meta", "label_type", "implicit_true", "due_to_null"],
        "qa_responses": len(qa), "qa_sources": len(qa_ids),
        "qa_train_responses": sum(row.official_split == "train" for row in qa),
        "qa_test_responses": sum(row.official_split == "test" for row in qa),
        "qa_train_sources": len(split_sources["train"]),
        "qa_test_sources": len(split_sources["test"]),
        "source_id_crossings": sorted(source_crossing),
        "quality_counts": dict(sorted(Counter(str(row.quality) for row in qa).items())),
        "responses_with_labels": sum(bool(row.labels) for row in qa),
        "human_label_span_count": sum(len(row.labels) for row in qa),
        "offset_mismatch_count": len(offset_mismatches),
        "offset_mismatch_examples": offset_mismatches[:20],
        "claim_level_alignment_supported": not offset_mismatches,
        "parsed_passages_per_source": {str(key): value for key, value in sorted(parsed_counts.items())},
        "validation_status": "pass" if not source_crossing and not offset_mismatches else "fail",
    }
    if result["validation_status"] != "pass":
        raise ValueError("RAGTruth source isolation or span alignment failed")
    return result


def run_preflight(root: Path, config: Phase05Config) -> dict[str, Any]:
    upstream = verify_upstream(root, config)
    audit = _ragtruth_audit(root, config)
    write_json_atomic(root / RESULTS / "phase05_ragtruth_schema_alignment_audit.json", audit)
    generator = config.values["generator"]
    write_json_atomic(root / RESULTS / "phase05_generation_model_manifest.json", {
        "schema_version": "phase05-generation-model-manifest-v1",
        **generator,
        "packages": _package_versions(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "test_content_used_in_smoke": False,
    })
    write_json_atomic(root / RESULTS / "phase05_generation_config_manifest.json", {
        "schema_version": "phase05-generation-config-manifest-v1",
        "phase05_config_canonical_sha256": config.canonical_sha256,
        "generation_config_sha256": _generation_config_sha(config),
        "prompt": config.values["prompt"],
        "context_assembly": config.values["context_assembly"],
        "generation": config.values["generation"],
    })
    write_json_atomic(root / RESULTS / "phase05_grounding_evaluator_config.json", {
        "schema_version": "phase05-grounding-evaluator-config-v1",
        "grounding_config_sha256": _grounding_config_sha(config),
        "claim_segmentation": config.values["claim_segmentation"],
        "grounding_evaluator": config.values["grounding_evaluator"],
        "ragtruth_validation": config.values["ragtruth_validation"],
    })
    return {"upstream": upstream, "ragtruth_audit": audit}


def run_context_assembly(root: Path, config: Phase05Config) -> dict[str, Any]:
    from transformers import AutoTokenizer

    population = config.values["techqa_population"]
    assert_techqa_split_allowed(population["split"], "assemble generation contexts")
    targets = read_parquet_records(root / population["source_targets"])
    selected_targets = [
        row for row in targets
        if row["split"] == "validation" and row["retrieval_strategy"] == "hybrid"
        and int(row["k"]) in {5, 10}
    ]
    if len(selected_targets) != 178 or len({row["question_id"] for row in selected_targets}) != 89:
        raise ValueError("frozen TechQA VALIDATION hybrid k5/k10 population does not reproduce")
    question_ids = {str(row["question_id"]) for row in selected_targets}
    raw = _load_raw_techqa(root, config, question_ids)
    ranked = read_parquet_records(root / population["source_rankings"])
    ranked_by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ranked:
        if row["split"] == "validation" and row["retrieval_strategy"] == "hybrid" and row["question_id"] in question_ids:
            ranked_by_question[str(row["question_id"])].append(row)
    if sum(len(rows) for rows in ranked_by_question.values()) != 890:
        raise ValueError("expected exactly ten frozen hybrid hits per VALIDATION question")
    chunk_table = pq.read_table(
        root / population["source_chunks"], columns=["chunk_id", "doc_id", "filename", "text", "text_sha256"]
    ).to_pylist()
    needed_ids = {str(row["chunk_id"]) for rows in ranked_by_question.values() for row in rows}
    chunks = {str(row["chunk_id"]): row for row in chunk_table if str(row["chunk_id"]) in needed_ids}
    if set(chunks) != needed_ids:
        raise ValueError("retrieval ranking references missing chunks")
    generator = config.values["generator"]
    tokenizer = AutoTokenizer.from_pretrained(
        generator["model_id"], revision=generator["tokenizer_revision"],
        cache_dir=str(root / generator["cache_directory"]), use_fast=True,
    )
    prompt_path = root / config.values["prompt"]["path"]
    if sha256_file(prompt_path) != config.values["prompt"]["physical_sha256"]:
        raise ValueError("frozen generation prompt hash changed")
    prompt_template = prompt_path.read_text(encoding="utf-8")
    target_by_key = {(str(row["question_id"]), int(row["k"])): row for row in selected_targets}
    output: list[dict[str, Any]] = []
    generation_sha = _generation_config_sha(config)
    for question_id in sorted(question_ids):
        hits = sorted(ranked_by_question[question_id], key=lambda row: int(row["rank"]))
        if [int(row["rank"]) for row in hits] != list(range(1, 11)):
            raise ValueError(f"non-contiguous hybrid ranks for {question_id}")
        states: dict[int, dict[str, Any]] = {}
        for k in (5, 10):
            source_chunks = []
            for hit in hits[:k]:
                chunk = chunks[str(hit["chunk_id"])]
                source_chunks.append({
                    "chunk_id": str(hit["chunk_id"]), "doc_id": str(hit["doc_id"]),
                    "filename": str(hit["filename"]), "rank": int(hit["rank"]),
                    "text": str(chunk["text"]), "text_sha256": str(chunk["text_sha256"]),
                })
            assembled = assemble_ranked_context(
                tokenizer=tokenizer, prompt_template=prompt_template,
                question=str(raw[question_id]["question"]), chunks=source_chunks,
                maximum_input_tokens=int(generator["maximum_input_tokens"]),
            )
            visible_chunks = []
            visible_ids = set(assembled.prompt_visible_chunk_ids)
            for chunk in source_chunks:
                if chunk["chunk_id"] not in visible_ids:
                    continue
                text = chunk["text"]
                if chunk["chunk_id"] == assembled.final_truncated_chunk_id:
                    token_ids = tokenizer.encode(text, add_special_tokens=False)
                    text = tokenizer.decode(
                        token_ids[:int(assembled.final_truncated_chunk_included_tokens)],
                        skip_special_tokens=True,
                    )
                visible_chunks.append({**chunk, "text": text})
            row = {
                "schema_version": "phase05-techqa-context-manifest-v1",
                "response_id": f"techqa-validation-{question_id}-hybrid-k{k}",
                "question_id": question_id, "split": "validation",
                "retrieval_strategy": "hybrid", "k": k,
                "question": str(raw[question_id]["question"]),
                "ordered_retrieved_chunk_ids_json": json.dumps(
                    [row["chunk_id"] for row in source_chunks], separators=(",", ":")
                ),
                "prompt_visible_chunk_ids_json": json.dumps(
                    list(assembled.prompt_visible_chunk_ids), separators=(",", ":")
                ),
                "prompt_visible_chunks_json": json.dumps(
                    visible_chunks, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
                "assembled_context": assembled.context,
                "assembled_context_sha256": assembled.assembled_context_sha256,
                "rendered_prompt": assembled.rendered_prompt,
                "rendered_prompt_sha256": sha256_text(assembled.rendered_prompt),
                "input_token_count": assembled.input_token_count,
                "fully_included_chunk_count": assembled.fully_included_chunk_count,
                "final_truncated_chunk_id": assembled.final_truncated_chunk_id,
                "final_truncated_chunk_original_tokens": assembled.final_truncated_chunk_original_tokens,
                "final_truncated_chunk_included_tokens": assembled.final_truncated_chunk_included_tokens,
                "retrieved_chunks_not_included": assembled.retrieved_chunks_not_included,
                "k5_context_prefix_sha256": None,
                "exposes_context_beyond_k5": None,
                "context_identity_sha256": None,
                "prompt_sha256": config.values["prompt"]["physical_sha256"],
                "generation_config_sha256": generation_sha,
                "y_suff_final": int(target_by_key[(question_id, k)]["y_suff_final"]),
            }
            states[k] = row
        states[5]["k5_context_prefix_sha256"] = states[5]["assembled_context_sha256"]
        states[5]["exposes_context_beyond_k5"] = False
        states[10]["k5_context_prefix_sha256"] = states[5]["assembled_context_sha256"]
        states[10]["exposes_context_beyond_k5"] = (
            states[10]["assembled_context"].startswith(states[5]["assembled_context"])
            and states[10]["assembled_context"] != states[5]["assembled_context"]
        )
        for row in states.values():
            row["context_identity_sha256"] = context_identity(row)
            output.append(row)
    additional_count = sum(bool(row["exposes_context_beyond_k5"]) for row in output if int(row["k"]) == 10)
    if additional_count / 89 < 0.80:
        raise RuntimeError(
            f"k10 prompt-visible additional-context rule failed: {additional_count}/89"
        )
    path = root / RESULTS / "phase05_techqa_context_manifest.parquet"
    metadata = write_canonical_parquet(
        path, output, CONTEXT_FIELDS, ("question_id", "k")
    )
    utilization = {
        "schema_version": "phase05-context-utilization-v1",
        "state_count": len(output), "question_count": 89,
        "k5_input_tokens": _distribution([int(row["input_token_count"]) for row in output if row["k"] == 5]),
        "k10_input_tokens": _distribution([int(row["input_token_count"]) for row in output if row["k"] == 10]),
        "k10_exposes_additional_context_count": additional_count,
        "k10_exposes_additional_context_fraction": additional_count / 89,
        "truncated_state_count": sum(row["final_truncated_chunk_id"] is not None for row in output),
        "context_manifest": metadata,
    }
    write_json_atomic(root / RESULTS / "phase05_techqa_context_utilization.json", utilization)
    return utilization


def _distribution(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "count": len(values), "minimum": min(values), "maximum": max(values),
        "mean": sum(values) / len(values), "median": statistics.median(values),
        "p25": float(np.percentile(ordered, 25)), "p75": float(np.percentile(ordered, 75)),
    }


def _generation_cache_key(row: dict[str, Any], config: Phase05Config) -> str:
    generator = config.values["generator"]
    return canonical_json_sha256({
        "question_id": row["question_id"], "retrieval_strategy": row["retrieval_strategy"],
        "k": int(row["k"]), "model_revision": generator["model_revision"],
        "tokenizer_revision": generator["tokenizer_revision"],
        "generation_config_sha256": row["generation_config_sha256"],
        "prompt_sha256": row["prompt_sha256"],
        "assembled_context_sha256": row["assembled_context_sha256"],
    })


def run_generation(root: Path, config: Phase05Config) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert_techqa_split_allowed("validation", "generate answers")
    context_path = root / RESULTS / "phase05_techqa_context_manifest.parquet"
    contexts = read_parquet_records(context_path)
    if len(contexts) != 178 or {row["split"] for row in contexts} != {"validation"}:
        raise ValueError("TechQA generation context manifest is incomplete or unsealed")
    checkpoint = root / DERIVED / "phase05_generation_checkpoint.json"
    cached_rows = json.loads(checkpoint.read_text(encoding="utf-8"))["rows"] if checkpoint.exists() else []
    cached = {str(row["cache_key"]): row for row in cached_rows}
    generator = config.values["generator"]
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(
        generator["model_id"], revision=generator["tokenizer_revision"],
        cache_dir=str(root / generator["cache_directory"]), use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        generator["model_id"], revision=generator["model_revision"],
        cache_dir=str(root / generator["cache_directory"]), dtype=torch.float32,
        low_cpu_mem_usage=True, use_safetensors=True,
    ).to("cpu")
    model.eval()
    parameters = {parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()}
    if parameters != {torch.float32}:
        raise ValueError(f"generator did not load entirely in float32: {parameters}")
    for index, context in enumerate(sorted(contexts, key=lambda row: (row["question_id"], int(row["k"]))), 1):
        key = _generation_cache_key(context, config)
        prior = cached.get(key)
        if prior and prior["generation_status"] in {"generated", "empty_output"}:
            continue
        attempts = int(prior["attempt_count"]) if prior else 0
        if attempts >= 2:
            continue
        started = time.perf_counter()
        row = {
            "schema_version": "phase05-generation-cache-v1", "cache_key": key,
            "response_id": context["response_id"], "question_id": context["question_id"],
            "retrieval_strategy": "hybrid", "k": int(context["k"]),
            "model_revision": generator["model_revision"],
            "tokenizer_revision": generator["tokenizer_revision"],
            "generation_config_sha256": context["generation_config_sha256"],
            "prompt_sha256": context["prompt_sha256"],
            "assembled_context_sha256": context["assembled_context_sha256"],
            "raw_generated_text": None, "normalized_generated_text": None,
            "input_token_count": int(context["input_token_count"]), "output_token_count": None,
            "generation_status": "generation_failed", "attempt_count": attempts + 1,
            "runtime_seconds": None,
            "runtime_metadata_json": json.dumps({
                "device": "cpu", "dtype": "float32", "torch": metadata.version("torch"),
                "transformers": metadata.version("transformers"),
            }, sort_keys=True, separators=(",", ":")),
            "error_class": None, "error_message_sha256": None, "output_sha256": None,
        }
        try:
            encoded = tokenizer(
                str(context["rendered_prompt"]), return_tensors="pt", add_special_tokens=False,
            )
            if int(encoded["input_ids"].shape[1]) != int(context["input_token_count"]):
                raise ValueError("generation input token count differs from context manifest")
            torch.manual_seed(42)
            with torch.no_grad():
                output = model.generate(
                    **encoded, do_sample=False, num_beams=1, max_new_tokens=128,
                    repetition_penalty=1.0, pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            output_ids = output[0, encoded["input_ids"].shape[1]:]
            raw_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            normalized = _normalize_generated_text(raw_text)
            row.update({
                "raw_generated_text": raw_text, "normalized_generated_text": normalized,
                "output_token_count": int(output_ids.shape[0]),
                "generation_status": "generated" if normalized else "empty_output",
                "output_sha256": sha256_text(raw_text),
            })
        except Exception as error:  # persisted infrastructure failure; identical retry only
            row["error_class"] = type(error).__name__
            row["error_message_sha256"] = sha256_text(str(error))
        row["runtime_seconds"] = time.perf_counter() - started
        cached[key] = row
        write_json_atomic(checkpoint, {
            "schema_version": "phase05-generation-checkpoint-v1",
            "generation_config_sha256": _generation_config_sha(config),
            "rows": sorted(cached.values(), key=lambda item: (item["question_id"], int(item["k"]))),
        })
        print(
            f"Phase 5 generation {index}/178 {context['question_id']} k={context['k']} "
            f"status={row['generation_status']} output_tokens={row['output_token_count']}",
            flush=True,
        )
    del model, tokenizer
    gc.collect()
    rows = sorted(cached.values(), key=lambda row: (row["question_id"], int(row["k"])))
    expected_keys = {_generation_cache_key(row, config) for row in contexts}
    if set(cached) != expected_keys:
        raise RuntimeError(f"generation checkpoint has {len(set(cached) & expected_keys)}/178 required states")
    path = root / RESULTS / "phase05_generation_cache.parquet"
    artifact = write_canonical_parquet(path, rows, GENERATION_FIELDS, ("question_id", "k"))
    provenance = {
        "schema_version": "phase05-generation-provenance-v1",
        "generation_closed_before_reference_evaluation": True,
        "state_count": len(rows), "question_count": len({row["question_id"] for row in rows}),
        "status_counts": dict(sorted(Counter(row["generation_status"] for row in rows).items())),
        "k5_status_counts": dict(sorted(Counter(row["generation_status"] for row in rows if row["k"] == 5).items())),
        "k10_status_counts": dict(sorted(Counter(row["generation_status"] for row in rows if row["k"] == 10).items())),
        "attempt_count_distribution": dict(sorted(Counter(int(row["attempt_count"]) for row in rows).items())),
        "generation_cache": artifact,
        "context_manifest_sha256": sha256_file(context_path),
        "techqa_test_rows": 0,
    }
    write_json_atomic(root / RESULTS / "phase05_generation_provenance.json", provenance)
    return provenance


def run_quality(root: Path, config: Phase05Config) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    assert_techqa_split_allowed("validation", "calculate answer-quality metrics")
    generation_path = root / RESULTS / "phase05_generation_cache.parquet"
    provenance = json.loads((root / RESULTS / "phase05_generation_provenance.json").read_text(encoding="utf-8"))
    if not provenance["generation_closed_before_reference_evaluation"]:
        raise ValueError("benchmark answers cannot be loaded before generation closes")
    generations = read_parquet_records(generation_path)
    if len(generations) != 178:
        raise ValueError("generation cache is incomplete")
    question_ids = {str(row["question_id"]) for row in generations}
    raw = _load_raw_techqa(root, config, question_ids)
    eligible = [row for row in generations if row["generation_status"] == "generated"]
    references = {qid: str(raw[qid]["answer"]) for qid in question_ids}
    score_cfg = config.values["answer_quality"]["bertscore"]
    snapshot = Path(snapshot_download(
        repo_id=score_cfg["model_type"], revision=score_cfg["model_revision"],
        cache_dir=str(root / config.values["generator"]["cache_directory"]),
        allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"],
    ))
    bert = FrozenBERTScorer(snapshot, int(score_cfg["num_layers"]), int(score_cfg["batch_size"]))
    bert_values = bert.score(
        [str(row["normalized_generated_text"]) for row in eligible],
        [references[str(row["question_id"])] for row in eligible],
    )
    bert_by_response = {row["response_id"]: value for row, value in zip(eligible, bert_values)}
    cache_hash = sha256_file(generation_path)
    rows: list[dict[str, Any]] = []
    for row in generations:
        reference = references[str(row["question_id"])]
        reference_ok = bool(reference.strip()) and reference.strip() != "-"
        metric_ok = row["generation_status"] == "generated" and reference_ok
        rows.append({
            "schema_version": "phase05-answer-quality-v1", "response_id": row["response_id"],
            "question_id": row["question_id"], "k": int(row["k"]),
            "generation_status": row["generation_status"],
            "reference_status": "usable" if reference_ok else "unusable",
            "rouge_l_f1": rouge_l_f1(str(row["normalized_generated_text"]), reference) if metric_ok else None,
            "bertscore_f1": bert_by_response.get(row["response_id"]) if metric_ok else None,
            "metric_status": "evaluated" if metric_ok else "undefined",
            "generation_cache_sha256": cache_hash,
        })
    del bert
    gc.collect()
    path = root / RESULTS / "phase05_answer_quality.parquet"
    artifact = write_canonical_parquet(path, rows, QUALITY_FIELDS, ("question_id", "k"))
    manifest = {
        "schema_version": "phase05-quality-manifest-v1", "quality_artifact": artifact,
        "generation_cache_sha256": cache_hash,
        "evaluated_responses": sum(row["metric_status"] == "evaluated" for row in rows),
        "undefined_responses": sum(row["metric_status"] != "evaluated" for row in rows),
        "rouge_l": config.values["answer_quality"]["rouge_l"],
        "bertscore": config.values["answer_quality"]["bertscore"],
        "benchmark_answers_entered_generation": False, "techqa_test_rows": 0,
    }
    write_json_atomic(root / RESULTS / "phase05_quality_manifest.json", manifest)
    return manifest


def _load_grounding_checkpoint(path: Path, grounding_sha: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row["grounding_config_sha256"] != grounding_sha:
                raise ValueError(f"stale grounding checkpoint at {path}:{line_number}")
            claim_id = str(row["claim_id"])
            if claim_id in rows and rows[claim_id] != row:
                raise ValueError(f"conflicting duplicate grounding checkpoint row: {claim_id}")
            rows[claim_id] = row
    return rows


def _append_grounding_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _score_grounding_records(
    root: Path, config: Phase05Config, models: GroundingModels,
    records: list[dict[str, Any]], checkpoint_name: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Score claim records in resumable batches; contexts never include references."""
    grounding_sha = _grounding_config_sha(config)
    checkpoint = root / DERIVED / checkpoint_name
    cached = _load_grounding_checkpoint(checkpoint, grounding_sha)
    claims: list[dict[str, Any]] = []
    contexts: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        response_id = str(record["response_id"])
        contexts[str(record["context_id"])] = record["chunks"]
        for claim in segment_claims(str(record["response_text"])):
            claim_id = sha256_text(f"{record['dataset']}\n{response_id}\n{claim.claim_index}\n{claim.start}\n{claim.end}\n{claim.text}")
            human = None
            if record.get("human_labels") is not None:
                human = int(any(
                    max(claim.start, int(label["start"])) < min(claim.end, int(label["end"]))
                    for label in record["human_labels"]
                ))
            claims.append({
                "schema_version": "phase05-claim-grounding-v1", "dataset": record["dataset"],
                "response_id": response_id, "source_id": record.get("source_id"),
                "question_id": record.get("question_id"),
                "official_split": record.get("official_split"), "quality": record.get("quality"),
                "context_id": str(record["context_id"]), "claim_id": claim_id,
                "claim_index": claim.claim_index, "claim_text": claim.text,
                "claim_start": claim.start, "claim_end": claim.end,
                "human_unsupported": human,
            })
    remaining = [row for row in claims if row["claim_id"] not in cached]
    if remaining:
        unique_chunk_rows: dict[tuple[str, str], dict[str, Any]] = {}
        for row in remaining:
            for chunk in contexts[row["context_id"]]:
                unique_chunk_rows[(row["context_id"], str(chunk["chunk_id"]))] = chunk
        chunk_keys = sorted(unique_chunk_rows)
        chunk_embeddings_array = models.encode(
            [str(unique_chunk_rows[key]["text"]) for key in chunk_keys]
        )
        chunk_embeddings = {key: value for key, value in zip(chunk_keys, chunk_embeddings_array)}
        claim_embeddings_array = models.encode([str(row["claim_text"]) for row in remaining])
        for start in range(0, len(remaining), 128):
            batch = remaining[start:start + 128]
            batch_embeddings = claim_embeddings_array[start:start + len(batch)]
            prepared: list[tuple[dict[str, Any], list[dict[str, Any]], int, int]] = []
            nli_pairs: list[dict[str, Any]] = []
            for row, claim_embedding in zip(batch, batch_embeddings):
                chunks = contexts[row["context_id"]]
                matrix = np.stack([
                    chunk_embeddings[(row["context_id"], str(chunk["chunk_id"]))]
                    for chunk in chunks
                ])
                similarities = matrix @ claim_embedding
                candidates = select_candidate_chunks(similarities.tolist(), chunks, 3)
                fits, claim_length, special = models.claim_fits(str(row["claim_text"]))
                prepared.append((row, candidates, claim_length, special))
                if fits:
                    for candidate_index, candidate in enumerate(candidates, 1):
                        unit = {
                            "unit_id": canonical_json_sha256({
                                "claim_id": row["claim_id"], "chunk_id": candidate["chunk_id"],
                                "rank": int(candidate["rank"]), "text_sha256": sha256_text(str(candidate["text"])),
                            }),
                            "unit_type": "independent_candidate_chunk",
                            "chunk_ids": [str(candidate["chunk_id"])],
                            "ranks": [int(candidate["rank"])],
                            "constituents": [str(candidate["text"])],
                        }
                        nli_pairs.append({
                            "claim_id": row["claim_id"], "claim_text": row["claim_text"],
                            "candidate_index": candidate_index, "chunk_id": str(candidate["chunk_id"]),
                            "rank": int(candidate["rank"]), "similarity": float(candidate["similarity"]),
                            "unit": unit,
                        })
            scored_pairs = models.score_pairs(nli_pairs) if nli_pairs else []
            by_claim: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for pair in scored_pairs:
                by_claim[str(pair["claim_id"])].append({
                    "evaluation_status": "evaluable", "candidate_index": int(pair["candidate_index"]),
                    "chunk_id": str(pair["chunk_id"]), "rank": int(pair["rank"]),
                    "similarity": float(pair["similarity"]), "entailment": float(pair["entailment"]),
                    "neutral": float(pair["neutral"]), "contradiction": float(pair["contradiction"]),
                })
            completed: list[dict[str, Any]] = []
            for row, candidates, claim_length, special in prepared:
                nli = sorted(by_claim.get(row["claim_id"], []), key=lambda item: int(item["candidate_index"]))
                aggregate = aggregate_candidate_nli(nli)
                completed.append({
                    **{key: row.get(key) for key in (
                        "schema_version", "dataset", "response_id", "source_id", "question_id",
                        "official_split", "quality", "claim_id", "claim_index", "claim_text",
                        "claim_start", "claim_end", "human_unsupported",
                    )},
                    "candidate_chunk_ids_json": candidate_json(candidates, ["chunk_id", "rank"]),
                    "candidate_similarities_json": candidate_json(candidates, ["chunk_id", "rank", "similarity"]),
                    "candidate_nli_json": candidate_json(
                        nli, ["candidate_index", "chunk_id", "rank", "similarity", "entailment", "neutral", "contradiction"]
                    ),
                    "claim_token_length": claim_length, "pair_special_token_count": special,
                    **aggregate, "grounding_config_sha256": grounding_sha,
                })
            _append_grounding_rows(checkpoint, completed)
            cached.update({row["claim_id"]: row for row in completed})
            print(
                f"Grounding {records[0]['dataset'] if records else 'dataset'} claims "
                f"{min(start + len(batch), len(remaining))}/{len(remaining)} new "
                f"({len(cached)}/{len(claims)} total)", flush=True,
            )
    if set(cached) != {row["claim_id"] for row in claims}:
        extra = set(cached) - {row["claim_id"] for row in claims}
        missing = {row["claim_id"] for row in claims} - set(cached)
        raise ValueError(f"grounding checkpoint identity mismatch: extra={len(extra)} missing={len(missing)}")
    rows = sorted(cached.values(), key=lambda row: (
        str(row.get("source_id") or row.get("question_id")), str(row["response_id"]), int(row["claim_index"])
    ))
    counts = {
        "response_count": len(records), "claim_count": len(rows),
        "evaluable_claim_count": sum(row["evaluation_status"] == "evaluable" for row in rows),
        "unevaluable_claim_count": sum(row["evaluation_status"] != "evaluable" for row in rows),
        "zero_claim_response_count": len(records) - len({row["response_id"] for row in rows}),
    }
    return rows, counts


def _ragtruth_records(root: Path, config: Phase05Config) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = config.values["ragtruth_validation"]
    sources = load_ragtruth_sources(root / cfg["sources_path"])
    responses = load_ragtruth_responses(root / cfg["responses_path"])
    source_by_id = {source.source_id: source for source in sources if source.task_type == "QA"}
    records = []
    response_metadata: dict[str, Any] = {}
    for response in responses:
        if response.source_id not in source_by_id:
            continue
        source = source_by_id[response.source_id]
        chunks = parse_ragtruth_passages(str(source.source_info["passages"]))
        record = {
            "dataset": "ragtruth", "response_id": response.response_id,
            "source_id": response.source_id, "question_id": None,
            "official_split": response.official_split, "quality": response.quality,
            "context_id": response.source_id, "response_text": response.response,
            "chunks": chunks, "human_labels": list(response.labels),
        }
        records.append(record)
        response_metadata[response.response_id] = {
            "source_id": response.source_id, "official_split": response.official_split,
            "quality": response.quality, "human_response_unsupported": int(bool(response.labels)),
        }
    return records, response_metadata


def _response_validation_rows(
    response_metadata: dict[str, Any], claim_rows: list[dict[str, Any]], threshold: float,
) -> list[dict[str, Any]]:
    by_response: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in claim_rows:
        by_response[str(row["response_id"])].append(row)
    output = []
    for response_id, metadata_row in response_metadata.items():
        evaluable = [row for row in by_response.get(response_id, []) if row["evaluation_status"] == "evaluable"]
        output.append({
            "response_id": response_id, **metadata_row,
            "evaluation_status": "evaluable" if evaluable else "no_evaluable_claim",
            "prediction": int(any(float(row["claim_support_score"]) < threshold for row in evaluable)) if evaluable else None,
            "unsupportedness": max(float(row["claim_unsupportedness_score"]) for row in evaluable) if evaluable else None,
            "claim_count": len(by_response.get(response_id, [])), "evaluable_claim_count": len(evaluable),
        })
    return output


def _metric_row(rows: list[dict[str, Any]], *, population: str, level: str) -> dict[str, Any]:
    eligible = [row for row in rows if row["prediction"] is not None]
    values = discrimination_metrics(
        [int(row["truth"]) for row in eligible],
        [int(row["prediction"]) for row in eligible],
        [float(row["unsupportedness"]) for row in eligible],
    )
    return {"population": population, "level": level, "eligible_count": len(eligible), **values}


def _cluster_bootstrap(
    rows: list[dict[str, Any]], *, population: str, level: str,
    replicates: int, seed: int,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["prediction"] is not None:
            groups[str(row["source_id"])].append(row)
    ids = sorted(groups)
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {name: [] for name in ("precision", "recall", "f1", "auroc", "auprc")}
    for _ in range(replicates):
        sampled = rng.choice(ids, size=len(ids), replace=True)
        sample = [row for source_id in sampled for row in groups[str(source_id)]]
        metric = discrimination_metrics(
            [int(row["truth"]) for row in sample], [int(row["prediction"]) for row in sample],
            [float(row["unsupportedness"]) for row in sample],
        )
        for name in values:
            if metric[name] is not None and math.isfinite(float(metric[name])):
                values[name].append(float(metric[name]))
    point = _metric_row(rows, population=population, level=level)
    output = []
    for name, samples in values.items():
        output.append({
            "population": population, "level": level, "metric": name,
            "point_estimate": point[name], "confidence_level": 0.95,
            "ci_low": float(np.percentile(samples, 2.5)) if samples else None,
            "ci_high": float(np.percentile(samples, 97.5)) if samples else None,
            "method": "source_id_cluster_percentile_bootstrap", "resampling_unit": "source_id",
            "requested_replicates": replicates, "valid_replicates": len(samples), "seed": seed,
        })
    return output


def run_ragtruth_grounding(root: Path, config: Phase05Config) -> dict[str, Any]:
    amendment = _apply_execution_amendment(root, config)
    records, response_metadata = _ragtruth_records(root, config)
    evaluator = config.values["grounding_evaluator"]
    models = GroundingModels(
        evaluator["candidate_embedding_model"], evaluator["candidate_embedding_revision"],
        evaluator["nli_model_id"], evaluator["nli_model_revision"],
        root / config.values["generator"]["cache_directory"],
        int(evaluator["nli_max_pair_tokens"]), int(amendment["selected_execution"]["nli_batch_size"]),
        bool(amendment["selected_execution"].get("length_bucketing", False)),
    )
    claims, counts = _score_grounding_records(
        root, config, models, records, "phase05_ragtruth_grounding_checkpoint.jsonl"
    )
    models.close(); del models; gc.collect()
    claim_path = root / RESULTS / "phase05_ragtruth_claim_scores.parquet"
    claim_artifact = write_canonical_parquet(
        claim_path, claims, CLAIM_FIELDS, ("source_id", "response_id", "claim_index")
    )
    train_primary = [
        row for row in claims if row["official_split"] == "train" and row["quality"] == "good"
    ]
    threshold, search = select_support_threshold(train_primary)
    search_fields = (
        "t_support", "eligible_claim_count", "tn", "fp", "fn", "tp",
        "precision", "recall", "f1", "selected",
    )
    write_csv_atomic(root / RESULTS / "phase05_ragtruth_threshold_search.csv", search, search_fields)
    train_identity = canonical_json_sha256([
        {"claim_id": row["claim_id"], "human_unsupported": row["human_unsupported"],
         "claim_support_score": row["claim_support_score"]}
        for row in train_primary if row["evaluation_status"] == "evaluable"
    ])
    selected_threshold = {
        "schema_version": "phase05-selected-grounding-threshold-v1",
        "t_support": threshold, "selection_split": "ragtruth_official_train",
        "selection_population": "QA quality=good evaluable claims",
        "selection_rule": config.values["ragtruth_validation"]["threshold_selection"],
        "threshold_grid": config.values["ragtruth_validation"]["threshold_grid"],
        "train_claim_identity_sha256": train_identity,
        "ragtruth_test_labels_accessed_for_selection": False,
        "grounding_config_sha256": _grounding_config_sha(config),
    }
    write_json_atomic(root / RESULTS / "phase05_selected_grounding_threshold.json", selected_threshold)
    response_rows = _response_validation_rows(response_metadata, claims, threshold)
    metric_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for population, quality_filter in (("good_primary", "good"), ("all_quality_sensitivity", None)):
        test_claims = [
            row for row in claims if row["official_split"] == "test"
            and (quality_filter is None or row["quality"] == quality_filter)
            and row["evaluation_status"] == "evaluable"
        ]
        claim_validation = [{
            "source_id": row["source_id"], "truth": int(row["human_unsupported"]),
            "prediction": int(float(row["claim_support_score"]) < threshold),
            "unsupportedness": float(row["claim_unsupportedness_score"]),
        } for row in test_claims]
        test_responses = [
            row for row in response_rows if row["official_split"] == "test"
            and (quality_filter is None or row["quality"] == quality_filter)
        ]
        response_validation = [{
            "source_id": row["source_id"], "truth": int(row["human_response_unsupported"]),
            "prediction": row["prediction"], "unsupportedness": row["unsupportedness"],
        } for row in test_responses]
        for level, validation_rows in (("claim", claim_validation), ("response", response_validation)):
            metric_rows.append(_metric_row(validation_rows, population=population, level=level))
            bootstrap_rows.extend(_cluster_bootstrap(
                validation_rows, population=population, level=level,
                replicates=int(config.values["ragtruth_validation"]["bootstrap"]["replicates"]),
                seed=int(config.values["ragtruth_validation"]["bootstrap"]["seed"]),
            ))
    metric_fields = (
        "population", "level", "eligible_count", "n", "tn", "fp", "fn", "tp",
        "precision", "recall", "f1", "auroc", "auprc",
    )
    write_csv_atomic(root / RESULTS / "phase05_ragtruth_test_metrics.csv", metric_rows, metric_fields)
    bootstrap_fields = (
        "population", "level", "metric", "point_estimate", "confidence_level", "ci_low",
        "ci_high", "method", "resampling_unit", "requested_replicates", "valid_replicates", "seed",
    )
    write_csv_atomic(root / RESULTS / "phase05_ragtruth_bootstrap_intervals.csv", bootstrap_rows, bootstrap_fields)
    primary_response = next(
        row for row in metric_rows if row["population"] == "good_primary" and row["level"] == "response"
    )
    passed = primary_response["auroc"] is not None and float(primary_response["auroc"]) >= 0.60
    manifest = {
        "schema_version": "phase05-grounding-validation-manifest-v1",
        "ragtruth_census": {
            "qa_train_responses": 5034, "qa_test_responses": 900,
            "qa_train_sources": 839, "qa_test_sources": 150, "source_crossings": 0,
        },
        "claim_counts": counts, "claim_artifact": claim_artifact,
        "selected_t_support": threshold, "threshold_selection_split": "train",
        "test_influence_on_threshold": False, "primary_test_response_auroc": primary_response["auroc"],
        "validity_threshold": 0.60, "validity_threshold_passed": passed,
        "binary_techqa_grounding_interpretation": "validated_proxy" if passed else "exploratory",
        "evaluator_retreated_or_replaced": False,
        "metrics_sha256": sha256_file(root / RESULTS / "phase05_ragtruth_test_metrics.csv"),
        "bootstrap_sha256": sha256_file(root / RESULTS / "phase05_ragtruth_bootstrap_intervals.csv"),
    }
    write_json_atomic(root / RESULTS / "phase05_grounding_validation_manifest.json", manifest)
    return manifest


def run_techqa_grounding(root: Path, config: Phase05Config) -> dict[str, Any]:
    assert_techqa_split_allowed("validation", "calculate grounding metrics")
    amendment = _apply_execution_amendment(root, config)
    contexts = read_parquet_records(root / RESULTS / "phase05_techqa_context_manifest.parquet")
    generations = read_parquet_records(root / RESULTS / "phase05_generation_cache.parquet")
    if len(contexts) != 178 or len(generations) != 178:
        raise ValueError("TechQA state artifacts are incomplete")
    context_by_response = {str(row["response_id"]): row for row in contexts}
    records = []
    for generation in generations:
        if generation["generation_status"] != "generated":
            continue
        context = context_by_response[str(generation["response_id"])]
        records.append({
            "dataset": "techqa", "response_id": generation["response_id"],
            "source_id": None, "question_id": generation["question_id"],
            "official_split": "validation", "quality": None,
            "context_id": generation["response_id"],
            "response_text": generation["normalized_generated_text"],
            "chunks": json.loads(context["prompt_visible_chunks_json"]),
            "human_labels": None,
        })
    evaluator = config.values["grounding_evaluator"]
    models = GroundingModels(
        evaluator["candidate_embedding_model"], evaluator["candidate_embedding_revision"],
        evaluator["nli_model_id"], evaluator["nli_model_revision"],
        root / config.values["generator"]["cache_directory"],
        int(evaluator["nli_max_pair_tokens"]), int(amendment["selected_execution"]["nli_batch_size"]),
        bool(amendment["selected_execution"].get("length_bucketing", False)),
    )
    claims, counts = _score_grounding_records(
        root, config, models, records, "phase05_techqa_grounding_checkpoint.jsonl"
    )
    models.close(); del models; gc.collect()
    claim_path = root / RESULTS / "phase05_techqa_claim_grounding.parquet"
    claim_artifact = write_canonical_parquet(
        claim_path, claims, CLAIM_FIELDS, ("question_id", "response_id", "claim_index")
    )
    generated_claim_fields = (
        "schema_version", "response_id", "question_id", "claim_id", "claim_index",
        "claim_text", "claim_start", "claim_end",
    )
    generated_claims = [{
        **{field: row.get(field) for field in generated_claim_fields},
        "schema_version": "phase05-techqa-generated-claims-v1",
    } for row in claims]
    generated_path = root / RESULTS / "phase05_techqa_generated_claims.parquet"
    generated_artifact = write_canonical_parquet(
        generated_path, generated_claims, generated_claim_fields, ("question_id", "response_id", "claim_index")
    )
    threshold = json.loads((root / RESULTS / "phase05_selected_grounding_threshold.json").read_text(encoding="utf-8"))["t_support"]
    claims_by_response: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in claims:
        claims_by_response[str(row["response_id"])].append(row)
    context_y = {str(row["response_id"]): int(row["y_suff_final"]) for row in contexts}
    response_rows = []
    unevaluable = []
    for generation in sorted(generations, key=lambda row: (row["question_id"], int(row["k"]))):
        response_id = str(generation["response_id"])
        response_claims = claims_by_response.get(response_id, [])
        if generation["generation_status"] == "generated":
            metric = response_grounding_metrics(response_claims, float(threshold))
        else:
            metric = {
                "claim_count": 0, "evaluable_claim_count": 0, "unevaluable_claim_count": 0,
                "mean_claim_support_score": None, "minimum_claim_support_score": None,
                "unsupported_claim_count": 0, "unsupported_claim_rate": None,
                "maximum_claim_contradiction": None, "fully_supported_response": None,
                "response_with_any_unsupported_claim": None,
                "response_grounding_status": generation["generation_status"],
            }
        response_rows.append({
            "schema_version": "phase05-techqa-response-grounding-v1",
            "response_id": response_id, "question_id": generation["question_id"],
            "k": int(generation["k"]), "generation_status": generation["generation_status"],
            **metric, "y_suff_final": context_y[response_id], "t_support": float(threshold),
        })
        for claim in response_claims:
            if claim["evaluation_status"] != "evaluable":
                unevaluable.append({
                    "response_id": response_id, "question_id": generation["question_id"],
                    "k": int(generation["k"]), "claim_id": claim["claim_id"],
                    "claim_text": claim["claim_text"], "claim_token_length": claim["claim_token_length"],
                    "reason": "claim_exceeds_frozen_nli_pair_budget",
                })
    response_fields = (
        "schema_version", "response_id", "question_id", "k", "generation_status",
        "claim_count", "evaluable_claim_count", "unevaluable_claim_count",
        "mean_claim_support_score", "minimum_claim_support_score", "unsupported_claim_count",
        "unsupported_claim_rate", "maximum_claim_contradiction", "fully_supported_response",
        "response_with_any_unsupported_claim", "response_grounding_status", "y_suff_final", "t_support",
    )
    response_path = root / RESULTS / "phase05_techqa_response_grounding.parquet"
    response_artifact = write_canonical_parquet(
        response_path, response_rows, response_fields, ("question_id", "k")
    )
    unevaluable_fields = (
        "response_id", "question_id", "k", "claim_id", "claim_text", "claim_token_length", "reason",
    )
    write_csv_atomic(root / RESULTS / "phase05_evaluator_unevaluable_claims.csv", unevaluable, unevaluable_fields)
    manifest = {
        "schema_version": "phase05-techqa-grounding-manifest-v1",
        "claim_artifact": claim_artifact, "generated_claim_artifact": generated_artifact,
        "response_artifact": response_artifact, "counts": counts,
        "unevaluable_claim_count": len(unevaluable),
        "unevaluable_claim_rate": len(unevaluable) / len(claims) if claims else None,
        "selected_t_support": threshold,
        "ragtruth_validation_manifest_sha256": sha256_file(root / RESULTS / "phase05_grounding_validation_manifest.json"),
        "techqa_test_rows": 0,
    }
    write_json_atomic(root / RESULTS / "phase05_techqa_grounding_manifest.json", manifest)
    return manifest


def _policy_summary(policy_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    answered = [row for row in rows if bool(row["answered"])]
    evaluable_grounding = [row for row in answered if row.get("unsupported_claim_rate") is not None]
    fully_defined = [row for row in answered if row.get("fully_supported_response") is not None]
    unsupported_defined = [
        row for row in answered if row.get("response_with_any_unsupported_claim") is not None
    ]
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
        "mean_unsupported_claim_rate": _mean(row.get("unsupported_claim_rate") for row in evaluable_grounding),
        "unsupported_claim_rate_denominator": len(evaluable_grounding),
        "fully_supported_response_rate": fully_count / len(fully_defined) if fully_defined else None,
        "fully_supported_response_rate_denominator": len(fully_defined),
        "response_with_any_unsupported_claim_rate": unsupported_count / len(unsupported_defined) if unsupported_defined else None,
        "response_with_any_unsupported_claim_rate_denominator": len(unsupported_defined),
        "mean_claim_support_score": _mean(row.get("mean_claim_support_score") for row in answered),
        "mean_output_tokens": _mean(row.get("output_token_count") for row in answered),
        "grounded_answer_yield": fully_count / len(rows),
        "unsupported_answer_population_rate": unsupported_count / len(rows),
    }


def run_analysis(root: Path, config: Phase05Config) -> dict[str, Any]:
    assert_techqa_split_allowed("validation", "construct policy and paired analysis")
    generations = read_parquet_records(root / RESULTS / "phase05_generation_cache.parquet")
    quality = read_parquet_records(root / RESULTS / "phase05_answer_quality.parquet")
    grounding = read_parquet_records(root / RESULTS / "phase05_techqa_response_grounding.parquet")
    if not (len(generations) == len(quality) == len(grounding) == 178):
        raise ValueError("Phase 5 TechQA per-response artifacts are incomplete")
    quality_by_response = {str(row["response_id"]): row for row in quality}
    grounding_by_response = {str(row["response_id"]): row for row in grounding}
    states: dict[tuple[str, int], dict[str, Any]] = {}
    for generation in generations:
        response_id = str(generation["response_id"])
        q = quality_by_response[response_id]
        g = grounding_by_response[response_id]
        states[(str(generation["question_id"]), int(generation["k"]))] = {
            "response_id": response_id, "generation_status": generation["generation_status"],
            "rouge_l_f1": q["rouge_l_f1"], "bertscore_f1": q["bertscore_f1"],
            "unsupported_claim_rate": g["unsupported_claim_rate"],
            "fully_supported_response": g["fully_supported_response"],
            "response_with_any_unsupported_claim": g["response_with_any_unsupported_claim"],
            "mean_claim_support_score": g["mean_claim_support_score"],
            "minimum_claim_support_score": g["minimum_claim_support_score"],
            "output_token_count": generation["output_token_count"],
            "y_suff_final": int(g["y_suff_final"]),
        }
    question_ids = sorted({key[0] for key in states})
    if len(question_ids) != 89 or any((qid, k) not in states for qid in question_ids for k in (5, 10)):
        raise ValueError("paired k5/k10 state identity is incomplete")
    trajectories = pd.read_csv(root / "artifacts/results/phase04_policy_trajectories.csv")
    actions: dict[float, dict[str, str]] = {}
    for risk in (0.1, 0.2):
        selected = trajectories[np.isclose(trajectories["risk_constraint"], risk)]
        if len(selected) != 89 or selected["question_id"].nunique() != 89:
            raise ValueError(f"frozen Phase 4 trajectory population failed for risk={risk}")
        actions[risk] = dict(zip(selected["question_id"].astype(str), selected["final_action"].astype(str)))
    if Counter(actions[0.1].values()) != Counter({"ABSTAIN": 87, "ANSWER_AT_K10": 2}):
        raise ValueError("frozen primary 10% policy behavior did not reproduce")
    if Counter(actions[0.2].values()) != Counter({"ABSTAIN": 73, "ANSWER_AT_K5": 9, "ANSWER_AT_K10": 7}):
        raise ValueError("frozen 20% policy behavior did not reproduce")
    policies = {
        "G0": build_policy_view(question_ids, states, policy_id="G0", fixed_k=5),
        "G1": build_policy_view(question_ids, states, policy_id="G1", fixed_k=10),
        "G2": build_policy_view(question_ids, states, policy_id="G2", actions=actions[0.1]),
        "G3": build_policy_view(question_ids, states, policy_id="G3", actions=actions[0.2]),
    }
    policy_fields = (
        "policy_id", "question_id", "final_action", "answered", "selected_k", "response_id",
        "generation_status", "rouge_l_f1", "bertscore_f1", "unsupported_claim_rate",
        "fully_supported_response", "response_with_any_unsupported_claim",
        "mean_claim_support_score", "minimum_claim_support_score", "output_token_count", "y_suff_final",
    )
    policy_artifacts = {}
    for policy_id, rows in policies.items():
        path = root / RESULTS / f"phase05_policy_{policy_id}.parquet"
        policy_artifacts[policy_id] = write_canonical_parquet(
            path, rows, policy_fields, ("question_id",)
        )
    summaries = [_policy_summary(policy_id, policies[policy_id]) for policy_id in ("G0", "G1", "G2", "G3")]
    summary_fields = tuple(summaries[0])
    write_csv_atomic(root / RESULTS / "phase05_policy_generation_comparison.csv", summaries, summary_fields)
    sufficiency_rows = []
    for k in (5, 10):
        for y in (0, 1):
            selected = [row for (qid, depth), row in states.items() if depth == k and row["y_suff_final"] == y]
            evaluable = [row for row in selected if row["unsupported_claim_rate"] is not None]
            fully = [row for row in selected if row["fully_supported_response"] is not None]
            sufficiency_rows.append({
                "k": k, "y_suff_final": y, "response_count": len(selected),
                "evaluable_grounding_count": len(evaluable),
                "mean_unsupported_claim_rate": _mean(row["unsupported_claim_rate"] for row in evaluable),
                "fully_supported_response_rate": (
                    sum(row["fully_supported_response"] is True for row in fully) / len(fully) if fully else None
                ),
                "mean_rouge_l": _mean(row["rouge_l_f1"] for row in selected),
                "mean_bertscore_f1": _mean(row["bertscore_f1"] for row in selected),
                "mean_claim_support_score": _mean(row["mean_claim_support_score"] for row in selected),
            })
    sufficiency_fields = tuple(sufficiency_rows[0])
    write_csv_atomic(
        root / RESULTS / "phase05_context_sufficiency_grounding_comparison.csv",
        sufficiency_rows, sufficiency_fields,
    )
    paired = []
    for question_id in question_ids:
        k5, k10 = states[(question_id, 5)], states[(question_id, 10)]
        def difference(name: str) -> float | None:
            a, b = k5.get(name), k10.get(name)
            return float(b) - float(a) if a is not None and b is not None else None
        paired.append({
            "schema_version": "phase05-paired-k5-k10-v1", "question_id": question_id,
            "k5_response_id": k5["response_id"], "k10_response_id": k10["response_id"],
            "k5_rouge_l_f1": k5["rouge_l_f1"], "k10_rouge_l_f1": k10["rouge_l_f1"],
            "difference_rouge_l_f1_k10_minus_k5": difference("rouge_l_f1"),
            "k5_bertscore_f1": k5["bertscore_f1"], "k10_bertscore_f1": k10["bertscore_f1"],
            "difference_bertscore_f1_k10_minus_k5": difference("bertscore_f1"),
            "k5_unsupported_claim_rate": k5["unsupported_claim_rate"],
            "k10_unsupported_claim_rate": k10["unsupported_claim_rate"],
            "difference_unsupported_claim_rate_k10_minus_k5": difference("unsupported_claim_rate"),
            "k5_mean_claim_support_score": k5["mean_claim_support_score"],
            "k10_mean_claim_support_score": k10["mean_claim_support_score"],
            "difference_mean_claim_support_k10_minus_k5": difference("mean_claim_support_score"),
            "k5_output_tokens": k5["output_token_count"], "k10_output_tokens": k10["output_token_count"],
            "difference_output_tokens_k10_minus_k5": difference("output_token_count"),
            "k5_fully_supported_response": k5["fully_supported_response"],
            "k10_fully_supported_response": k10["fully_supported_response"],
        })
    paired_fields = tuple(paired[0])
    paired_path = root / RESULTS / "phase05_paired_k5_k10.parquet"
    paired_artifact = write_canonical_parquet(paired_path, paired, paired_fields, ("question_id",))
    descriptive = {
        "schema_version": "phase05-paired-descriptive-summary-v1",
        "question_count": 89,
        "mean_difference_rouge_l_f1_k10_minus_k5": _mean(row["difference_rouge_l_f1_k10_minus_k5"] for row in paired),
        "mean_difference_bertscore_f1_k10_minus_k5": _mean(row["difference_bertscore_f1_k10_minus_k5"] for row in paired),
        "mean_difference_unsupported_claim_rate_k10_minus_k5": _mean(row["difference_unsupported_claim_rate_k10_minus_k5"] for row in paired),
        "mean_difference_claim_support_k10_minus_k5": _mean(row["difference_mean_claim_support_k10_minus_k5"] for row in paired),
        "mean_difference_output_tokens_k10_minus_k5": _mean(row["difference_output_tokens_k10_minus_k5"] for row in paired),
        "inferential_tests_performed": False,
        "paired_artifact": paired_artifact,
    }
    write_json_atomic(root / RESULTS / "phase05_paired_k5_k10_summary.json", descriptive)
    manifest = {
        "schema_version": "phase05-policy-analysis-manifest-v1",
        "policy_artifacts": policy_artifacts,
        "policy_summary_sha256": sha256_file(root / RESULTS / "phase05_policy_generation_comparison.csv"),
        "context_sufficiency_comparison_sha256": sha256_file(root / RESULTS / "phase05_context_sufficiency_grounding_comparison.csv"),
        "paired_artifact": paired_artifact, "paired_summary": descriptive,
        "primary_phase04_thresholds_unchanged": True,
        "sensitivity_phase04_thresholds_unchanged": True,
        "techqa_test_rows": 0, "phase06_statistics_performed": False,
    }
    write_json_atomic(root / RESULTS / "phase05_policy_analysis_manifest.json", manifest)
    return manifest


def run_stage(root: Path, config_path: Path, stage: str) -> dict[str, Any]:
    config = Phase05Config.load(config_path, root)
    stages: dict[str, Callable[[Path, Phase05Config], dict[str, Any]]] = {
        "preflight": run_preflight,
        "contexts": run_context_assembly,
        "generate": run_generation,
        "quality": run_quality,
        "ragtruth": run_ragtruth_grounding,
        "techqa_grounding": run_techqa_grounding,
        "analysis": run_analysis,
    }
    if stage not in stages:
        raise ValueError(f"unknown Phase 5 stage: {stage}")
    return stages[stage](root, config)
