"""Frozen Phase 5 context assembly, generation, and quality scoring on TEST."""

from __future__ import annotations

import gc
import json
import random
import time
from collections import Counter
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

from answerability_rag.data.techqa import load_techqa_rows
from answerability_rag.generation.config import Phase05Config
from answerability_rag.generation.context import assemble_ranked_context, context_identity
from answerability_rag.generation.pipeline import (
    CONTEXT_FIELDS,
    GENERATION_FIELDS,
    QUALITY_FIELDS,
    _generation_cache_key,
    _generation_config_sha,
    _normalize_generated_text,
)
from answerability_rag.generation.quality import FrozenBERTScorer, rouge_l_f1
from answerability_rag.hashing import canonical_json_sha256, sha256_file, sha256_text
from answerability_rag.retrieval.artifacts import write_canonical_parquet

from .common import RESULTS, Phase07Config, require_unsealed, write_json


DERIVED = Path("data/derived/phase07")


def _phase05(root: Path) -> Phase05Config:
    return Phase05Config.load(root / "configs/phase05_generation_grounding.json", root)


def _raw_path(root: Path) -> Path:
    config = json.loads((root / "configs/phase01_data.json").read_text(encoding="utf-8"))
    return root / "data/raw/techqa" / config["techqa"]["revision"] / "train.json"


def _question_texts(root: Path, ids: set[str]) -> dict[str, str]:
    # This stage exposes only question strings to context assembly and generation.
    result = {row.question_id: row.question for row in load_techqa_rows(_raw_path(root)) if row.question_id in ids}
    if set(result) != ids:
        raise ValueError("raw TechQA source is missing PRIMARY TEST questions")
    return result


def assemble_test_contexts(root: Path, config_path: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer

    config = Phase07Config.load(config_path)
    require_unsealed(root, config)
    phase05 = _phase05(root)
    targets = pq.read_table(root / RESULTS / "phase07_test_final_target.parquet").to_pylist()
    selected = [row for row in targets if row["retrieval_strategy"] == "hybrid" and int(row["k"]) in {5, 10}]
    ids = {str(row["question_id"]) for row in selected}
    if len(selected) != 2 * len(ids):
        raise ValueError("PRIMARY TEST target is not paired at hybrid k5/k10")
    questions = _question_texts(root, ids)
    rankings = pq.read_table(
        root / "artifacts/results/retrieval_ranked_hits.parquet", filters=[("split", "=", "test")]
    ).to_pylist()
    by_question: dict[str, list[dict[str, Any]]] = {qid: [] for qid in ids}
    for row in rankings:
        if row["retrieval_strategy"] == "hybrid" and str(row["question_id"]) in ids:
            by_question[str(row["question_id"])].append(row)
    if any(len(rows) != 10 for rows in by_question.values()):
        raise ValueError("each PRIMARY TEST question must have ten frozen hybrid ranked hits")
    chunk_rows = pq.read_table(
        root / "artifacts/data/techqa_chunk_manifest.parquet",
        columns=["chunk_id", "doc_id", "filename", "text", "text_sha256"],
    ).to_pylist()
    needed = {str(row["chunk_id"]) for rows in by_question.values() for row in rows}
    chunks = {str(row["chunk_id"]): row for row in chunk_rows if str(row["chunk_id"]) in needed}
    if set(chunks) != needed:
        raise ValueError("TEST ranking references missing frozen chunks")
    generator = phase05.values["generator"]
    tokenizer = AutoTokenizer.from_pretrained(
        generator["model_id"], revision=generator["tokenizer_revision"],
        cache_dir=str(root / generator["cache_directory"]), use_fast=True,
    )
    prompt_path = root / phase05.values["prompt"]["path"]
    if sha256_file(prompt_path) != config.values["generation"]["prompt_sha256"]:
        raise ValueError("frozen generation prompt changed")
    prompt = prompt_path.read_text(encoding="utf-8")
    generation_sha = _generation_config_sha(phase05)
    if generation_sha != config.values["generation"]["generation_config_sha256"]:
        raise ValueError("frozen Phase 5 generation configuration changed")
    target_by_key = {(str(row["question_id"]), int(row["k"])): row for row in selected}
    output: list[dict[str, Any]] = []
    for question_id in sorted(ids):
        hits = sorted(by_question[question_id], key=lambda row: int(row["rank"]))
        if [int(row["rank"]) for row in hits] != list(range(1, 11)):
            raise ValueError(f"non-contiguous frozen ranks for {question_id}")
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
                tokenizer=tokenizer, prompt_template=prompt, question=questions[question_id],
                chunks=source_chunks, maximum_input_tokens=int(generator["maximum_input_tokens"]),
            )
            visible_ids = set(assembled.prompt_visible_chunk_ids)
            visible_chunks = []
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
                "schema_version": "phase07-test-context-manifest-v1",
                "response_id": f"techqa-test-{question_id}-hybrid-k{k}",
                "question_id": question_id, "split": "test", "retrieval_strategy": "hybrid", "k": k,
                "question": questions[question_id],
                "ordered_retrieved_chunk_ids_json": json.dumps([x["chunk_id"] for x in source_chunks], separators=(",", ":")),
                "prompt_visible_chunk_ids_json": json.dumps(list(assembled.prompt_visible_chunk_ids), separators=(",", ":")),
                "prompt_visible_chunks_json": json.dumps(visible_chunks, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
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
                "k5_context_prefix_sha256": None, "exposes_context_beyond_k5": None,
                "context_identity_sha256": None,
                "prompt_sha256": config.values["generation"]["prompt_sha256"],
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
    del tokenizer
    gc.collect()
    path = root / RESULTS / "phase07_test_context_manifest.parquet"
    artifact = write_canonical_parquet(path, output, CONTEXT_FIELDS, ("question_id", "k"))
    manifest = {
        "schema_version": "phase07-test-context-manifest-summary-v1",
        "question_count": len(ids), "state_count": len(output),
        "k10_exposes_additional_context_count": sum(bool(row["exposes_context_beyond_k5"]) for row in output if row["k"] == 10),
        "context_artifact": artifact, "benchmark_answers_entered_context_or_prompt": False,
        "prompt_sha256": config.values["generation"]["prompt_sha256"],
        "generation_config_sha256": generation_sha,
    }
    write_json(root / RESULTS / "phase07_test_context_manifest.json", manifest)
    return manifest


def run_test_generation(root: Path, config_path: Path) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = Phase07Config.load(config_path)
    require_unsealed(root, config)
    phase05 = _phase05(root)
    contexts = pq.read_table(root / RESULTS / "phase07_test_context_manifest.parquet").to_pylist()
    if not contexts or {row["split"] for row in contexts} != {"test"}:
        raise ValueError("Phase 7 TEST context manifest is incomplete")
    checkpoint = root / DERIVED / "phase07_test_generation_checkpoint.json"
    cached_rows = json.loads(checkpoint.read_text(encoding="utf-8"))["rows"] if checkpoint.exists() else []
    cached = {str(row["cache_key"]): row for row in cached_rows}
    generator = phase05.values["generator"]
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
    if {p.dtype for p in model.parameters() if p.is_floating_point()} != {torch.float32}:
        raise ValueError("frozen generator did not load entirely in float32")
    ordered = sorted(contexts, key=lambda row: (row["question_id"], int(row["k"])))
    for index, context in enumerate(ordered, 1):
        key = _generation_cache_key(context, phase05)
        prior = cached.get(key)
        if prior and prior["generation_status"] in {"generated", "empty_output"}:
            continue
        attempts = int(prior["attempt_count"]) if prior else 0
        if attempts >= 2:
            continue
        started = time.perf_counter()
        row = {
            "schema_version": "phase07-test-generation-cache-v1", "cache_key": key,
            "response_id": context["response_id"], "question_id": context["question_id"],
            "retrieval_strategy": "hybrid", "k": int(context["k"]),
            "model_revision": generator["model_revision"], "tokenizer_revision": generator["tokenizer_revision"],
            "generation_config_sha256": context["generation_config_sha256"], "prompt_sha256": context["prompt_sha256"],
            "assembled_context_sha256": context["assembled_context_sha256"],
            "raw_generated_text": None, "normalized_generated_text": None,
            "input_token_count": int(context["input_token_count"]), "output_token_count": None,
            "generation_status": "generation_failed", "attempt_count": attempts + 1,
            "runtime_seconds": None,
            "runtime_metadata_json": json.dumps({"device": "cpu", "dtype": "float32", "torch": metadata.version("torch"), "transformers": metadata.version("transformers")}, sort_keys=True, separators=(",", ":")),
            "error_class": None, "error_message_sha256": None, "output_sha256": None,
        }
        try:
            encoded = tokenizer(str(context["rendered_prompt"]), return_tensors="pt", add_special_tokens=False)
            if int(encoded["input_ids"].shape[1]) != int(context["input_token_count"]):
                raise ValueError("generation input differs from immutable context manifest")
            torch.manual_seed(42)
            with torch.no_grad():
                generated = model.generate(
                    **encoded, do_sample=False, num_beams=1, max_new_tokens=128,
                    repetition_penalty=1.0, pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            output_ids = generated[0, encoded["input_ids"].shape[1]:]
            raw = tokenizer.decode(output_ids, skip_special_tokens=True)
            normalized = _normalize_generated_text(raw)
            row.update({
                "raw_generated_text": raw, "normalized_generated_text": normalized,
                "output_token_count": int(output_ids.shape[0]),
                "generation_status": "generated" if normalized else "empty_output",
                "output_sha256": sha256_text(raw),
            })
        except Exception as error:
            row["error_class"] = type(error).__name__
            row["error_message_sha256"] = sha256_text(str(error))
        row["runtime_seconds"] = time.perf_counter() - started
        cached[key] = row
        write_json(checkpoint, {
            "schema_version": "phase07-test-generation-checkpoint-v1",
            "generation_config_sha256": _generation_config_sha(phase05),
            "rows": sorted(cached.values(), key=lambda item: (item["question_id"], int(item["k"]))),
        })
        print(f"Phase 7 TEST generation {index}/{len(ordered)} {context['question_id']} k={context['k']} status={row['generation_status']}", flush=True)
    del model, tokenizer
    gc.collect()
    expected = {_generation_cache_key(row, phase05) for row in contexts}
    if set(cached) != expected:
        raise RuntimeError(f"generation checkpoint has {len(set(cached) & expected)}/{len(expected)} required states")
    rows = sorted(cached.values(), key=lambda row: (row["question_id"], int(row["k"])))
    path = root / RESULTS / "phase07_test_generation_cache.parquet"
    artifact = write_canonical_parquet(path, rows, GENERATION_FIELDS, ("question_id", "k"))
    provenance = {
        "schema_version": "phase07-test-generation-provenance-v1",
        "generation_closed_before_reference_evaluation": True,
        "state_count": len(rows), "question_count": len({row["question_id"] for row in rows}),
        "status_counts": dict(sorted(Counter(row["generation_status"] for row in rows).items())),
        "attempt_count_distribution": dict(sorted(Counter(int(row["attempt_count"]) for row in rows).items())),
        "generation_cache": artifact,
        "context_manifest_sha256": sha256_file(root / RESULTS / "phase07_test_context_manifest.parquet"),
        "benchmark_answers_entered_generation": False,
        "generation_config_sha256": config.values["generation"]["generation_config_sha256"],
    }
    write_json(root / RESULTS / "phase07_test_generation_provenance.json", provenance)
    return provenance


def evaluate_test_quality(root: Path, config_path: Path) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    config = Phase07Config.load(config_path)
    require_unsealed(root, config)
    phase05 = _phase05(root)
    provenance = json.loads((root / RESULTS / "phase07_test_generation_provenance.json").read_text(encoding="utf-8"))
    if not provenance["generation_closed_before_reference_evaluation"]:
        raise ValueError("reference boundary violated: generation is not closed")
    generations = pq.read_table(root / RESULTS / "phase07_test_generation_cache.parquet").to_pylist()
    ids = {str(row["question_id"]) for row in generations}
    references = {row.question_id: row.answer for row in load_techqa_rows(_raw_path(root)) if row.question_id in ids}
    if set(references) != ids:
        raise ValueError("raw TechQA references are incomplete")
    eligible = [row for row in generations if row["generation_status"] == "generated" and references[row["question_id"]].strip() not in {"", "-"}]
    score_cfg = phase05.values["answer_quality"]["bertscore"]
    snapshot = Path(snapshot_download(
        repo_id=score_cfg["model_type"], revision=score_cfg["model_revision"],
        cache_dir=str(root / phase05.values["generator"]["cache_directory"]),
        allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"],
    ))
    bert = FrozenBERTScorer(snapshot, int(score_cfg["num_layers"]), int(score_cfg["batch_size"]))
    values = bert.score(
        [str(row["normalized_generated_text"]) for row in eligible],
        [references[str(row["question_id"])] for row in eligible],
    )
    bert_by_response = {row["response_id"]: value for row, value in zip(eligible, values)}
    cache_hash = sha256_file(root / RESULTS / "phase07_test_generation_cache.parquet")
    rows = []
    for row in generations:
        reference = references[str(row["question_id"])]
        usable = bool(reference.strip()) and reference.strip() != "-"
        okay = row["generation_status"] == "generated" and usable
        rows.append({
            "schema_version": "phase07-test-answer-quality-v1", "response_id": row["response_id"],
            "question_id": row["question_id"], "k": int(row["k"]),
            "generation_status": row["generation_status"], "reference_status": "usable" if usable else "unusable",
            "rouge_l_f1": rouge_l_f1(str(row["normalized_generated_text"]), reference) if okay else None,
            "bertscore_f1": bert_by_response.get(row["response_id"]) if okay else None,
            "metric_status": "evaluated" if okay else "undefined", "generation_cache_sha256": cache_hash,
        })
    del bert
    gc.collect()
    path = root / RESULTS / "phase07_test_answer_quality.parquet"
    artifact = write_canonical_parquet(path, rows, QUALITY_FIELDS, ("question_id", "k"))
    manifest = {
        "schema_version": "phase07-test-quality-manifest-v1", "quality_artifact": artifact,
        "generation_cache_sha256": cache_hash,
        "evaluated_responses": sum(row["metric_status"] == "evaluated" for row in rows),
        "undefined_responses": sum(row["metric_status"] != "evaluated" for row in rows),
        "benchmark_answers_entered_generation": False,
        "references_first_accessed_after_generation_closed": True,
        "rouge_l": phase05.values["answer_quality"]["rouge_l"],
        "bertscore": phase05.values["answer_quality"]["bertscore"],
    }
    write_json(root / RESULTS / "phase07_test_quality_manifest.json", manifest)
    return manifest
