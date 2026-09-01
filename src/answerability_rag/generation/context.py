"""Leakage-safe prompt construction and deterministic ranked context assembly."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

from answerability_rag.hashing import sha256_text


FORBIDDEN_GENERATION_FIELDS = frozenset({
    "answer", "benchmark_answer", "reference_answer", "gold_answer",
    "y_suff_final", "y_suff", "model_probability", "probability", "p_suff",
    "retrieval_score", "nli_score", "nli_scores", "entailment", "contradiction",
    "span_coverage", "human_annotation", "human_annotations", "gold_evidence",
    "policy_action", "final_action",
})


def reject_generation_metadata(metadata: dict[str, Any] | None) -> None:
    keys = {str(key).strip().casefold() for key in (metadata or {})}
    forbidden = sorted(keys & FORBIDDEN_GENERATION_FIELDS)
    if forbidden:
        raise ValueError(f"generation input contains forbidden evaluation fields: {forbidden}")


def build_user_prompt(
    template: str, *, question: str, context: str, metadata: dict[str, Any] | None = None,
) -> str:
    reject_generation_metadata(metadata)
    if not isinstance(question, str) or not question.strip():
        raise ValueError("generation question must be a non-empty string")
    if not isinstance(context, str) or not context.strip():
        raise ValueError("generation context must be a non-empty string")
    if set(part[1] for part in __import__("string").Formatter().parse(template) if part[1]) != {"question", "context"}:
        raise ValueError("frozen generation template must contain only question/context fields")
    return template.format(question=question, context=context)


@dataclass(frozen=True)
class AssembledContext:
    context: str
    rendered_prompt: str
    input_token_count: int
    fully_included_chunk_count: int
    final_truncated_chunk_id: str | None
    final_truncated_chunk_original_tokens: int | None
    final_truncated_chunk_included_tokens: int | None
    retrieved_chunks_not_included: int
    prompt_visible_chunk_ids: tuple[str, ...]
    assembled_context_sha256: str


def _render_context(blocks: Iterable[str]) -> str:
    return "\n\n".join(blocks)


def _chat_prompt(tokenizer: Any, user_prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _input_length(tokenizer: Any, rendered: str) -> int:
    return len(tokenizer.encode(rendered, add_special_tokens=False))


def assemble_ranked_context(
    *, tokenizer: Any, prompt_template: str, question: str,
    chunks: list[dict[str, Any]], maximum_input_tokens: int,
) -> AssembledContext:
    if not chunks:
        raise ValueError("retrieved context contains no chunks")
    ordered = sorted(chunks, key=lambda row: (int(row["rank"]), str(row["chunk_id"])))
    if [int(row["rank"]) for row in ordered] != list(range(1, len(ordered) + 1)):
        raise ValueError("retrieved chunk ranks must be contiguous from one")
    if any(not str(row["text"]).strip() for row in ordered):
        raise ValueError("frozen retrieval contains an empty chunk")
    blocks: list[str] = []
    visible: list[str] = []
    fully_included = 0
    truncated_id: str | None = None
    original_tokens: int | None = None
    included_tokens: int | None = None
    for row in ordered:
        header = f"[CHUNK {int(row['rank'])}]\n"
        full_block = header + str(row["text"])
        candidate_context = _render_context([*blocks, full_block])
        candidate_user = build_user_prompt(
            prompt_template, question=question, context=candidate_context,
        )
        candidate_rendered = _chat_prompt(tokenizer, candidate_user)
        if _input_length(tokenizer, candidate_rendered) <= maximum_input_tokens:
            blocks.append(full_block)
            visible.append(str(row["chunk_id"]))
            fully_included += 1
            continue
        token_ids = tokenizer.encode(str(row["text"]), add_special_tokens=False)
        original_tokens = len(token_ids)
        low, high, best = 0, len(token_ids), 0
        while low <= high:
            middle = (low + high) // 2
            prefix = tokenizer.decode(token_ids[:middle], skip_special_tokens=True)
            block = header + prefix
            context = _render_context([*blocks, block])
            user = build_user_prompt(prompt_template, question=question, context=context)
            rendered = _chat_prompt(tokenizer, user)
            if prefix.strip() and _input_length(tokenizer, rendered) <= maximum_input_tokens:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        if best:
            prefix = tokenizer.decode(token_ids[:best], skip_special_tokens=True)
            blocks.append(header + prefix)
            visible.append(str(row["chunk_id"]))
            truncated_id = str(row["chunk_id"])
            included_tokens = best
        break
    context = _render_context(blocks)
    if not context:
        raise ValueError("input budget cannot fit any substantive retrieved context")
    user_prompt = build_user_prompt(prompt_template, question=question, context=context)
    rendered_prompt = _chat_prompt(tokenizer, user_prompt)
    input_count = _input_length(tokenizer, rendered_prompt)
    if input_count > maximum_input_tokens:
        raise AssertionError("assembled prompt exceeds frozen input budget")
    return AssembledContext(
        context=context,
        rendered_prompt=rendered_prompt,
        input_token_count=input_count,
        fully_included_chunk_count=fully_included,
        final_truncated_chunk_id=truncated_id,
        final_truncated_chunk_original_tokens=original_tokens if truncated_id else None,
        final_truncated_chunk_included_tokens=included_tokens,
        retrieved_chunks_not_included=len(ordered) - len(visible),
        prompt_visible_chunk_ids=tuple(visible),
        assembled_context_sha256=sha256_text(context),
    )


def context_identity(row: dict[str, Any]) -> str:
    """Canonical identity for exact prompt-visible context provenance."""
    from answerability_rag.hashing import canonical_json_sha256

    return canonical_json_sha256({
        "question_id": row["question_id"],
        "retrieval_strategy": row["retrieval_strategy"],
        "k": int(row["k"]),
        "ordered_chunk_ids": json.loads(row["ordered_retrieved_chunk_ids_json"]),
        "prompt_visible_chunk_ids": json.loads(row["prompt_visible_chunk_ids_json"]),
        "assembled_context_sha256": row["assembled_context_sha256"],
        "input_token_count": int(row["input_token_count"]),
    })
