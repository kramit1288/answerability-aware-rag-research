"""Pinned local NLI inference with balanced deterministic multi-chunk premise handling."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class NLIScorer:
    model_id: str
    revision: str
    cache_dir: Path
    max_length: int
    batch_size: int

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, revision=self.revision, cache_dir=self.cache_dir, use_fast=True,
        )
        config = AutoConfig.from_pretrained(
            self.model_id, revision=self.revision, cache_dir=self.cache_dir,
        )
        labels = {int(index): str(label).casefold() for index, label in config.id2label.items()}
        expected = {"entailment", "neutral", "contradiction"}
        if set(labels.values()) != expected:
            raise ValueError(f"unsupported NLI label mapping for {self.model_id}: {labels}")
        self.label_indices = {label: index for index, label in labels.items()}
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, revision=self.revision, cache_dir=self.cache_dir,
            use_safetensors=True, low_cpu_mem_usage=True, dtype=torch.float32,
        ).to("cpu")
        floating_dtypes = {
            parameter.dtype for parameter in self.model.parameters()
            if parameter.is_floating_point()
        }
        if floating_dtypes != {torch.float32}:
            raise ValueError(
                f"frozen float32 NLI dtype was not enforced for {self.model_id}: "
                f"{sorted(str(dtype) for dtype in floating_dtypes)}"
            )
        self.model.eval()
        self.compatibility = {
            "model_id": self.model_id,
            "revision": self.revision,
            "model_class": type(self.model).__name__,
            "tokenizer_class": type(self.tokenizer).__name__,
            "id2label": {str(key): value for key, value in labels.items()},
            "max_sequence_length": self.max_length,
            "device": "cpu",
            "dtype": str(next(self.model.parameters()).dtype).removeprefix("torch."),
            "status": "compatible",
        }

    def close(self) -> None:
        del self.model
        del self.tokenizer
        gc.collect()

    def _head_tail(self, token_ids: list[int], budget: int) -> list[int]:
        if len(token_ids) <= budget:
            return token_ids
        if budget <= 0:
            raise ValueError("NLI premise token budget is non-positive")
        head = (budget + 1) // 2
        return token_ids[:head] + token_ids[-(budget - head):]

    def render_premise(self, claim: str, unit: dict[str, Any]) -> str:
        claim_ids = self.tokenizer.encode(claim, add_special_tokens=False)
        special = int(self.tokenizer.num_special_tokens_to_add(pair=True))
        if len(claim_ids) + special >= self.max_length:
            raise ValueError(
                f"claim exceeds frozen NLI pair budget for {self.model_id}: "
                f"{len(claim_ids)} tokens"
            )
        markers = [f"[Retrieved chunk rank {rank}]" for rank in unit["ranks"]]
        marker_tokens = sum(len(self.tokenizer.encode(marker, add_special_tokens=False)) for marker in markers)
        available = self.max_length - len(claim_ids) - special - marker_tokens - 2 * len(markers)
        constituents = unit["constituents"]
        if not constituents:
            raise ValueError("NLI context unit contains no constituents")
        constituent_ids = [
            self.tokenizer.encode(text, add_special_tokens=False) for text in constituents
        ]
        base, remainder = divmod(available, len(constituents))
        budgets = [base + (1 if index < remainder else 0) for index in range(len(constituents))]

        def render() -> str:
            rendered = []
            for marker, ids, budget in zip(markers, constituent_ids, budgets):
                retained = self._head_tail(ids, budget)
                rendered.append(
                    marker + "\n" + self.tokenizer.decode(retained, skip_special_tokens=True)
                )
            return "\n\n".join(rendered)

        premise = render()
        encoded = self.tokenizer(premise, claim, add_special_tokens=True, truncation=False)
        # Some tokenizers (notably BART byte-level BPE) can re-tokenize decoded
        # head/tail text to a few more tokens than the arithmetic estimate. Trim
        # only that measured excess. Removing from the last largest budget keeps
        # constituent budgets equal to within one token and is deterministic.
        while len(encoded["input_ids"]) > self.max_length:
            positive = [index for index, budget in enumerate(budgets) if budget > 0]
            if not positive:
                raise ValueError("balanced premise markers exceed frozen max_sequence_length")
            largest = max(budgets[index] for index in positive)
            trim_index = max(index for index in positive if budgets[index] == largest)
            budgets[trim_index] -= 1
            premise = render()
            encoded = self.tokenizer(premise, claim, add_special_tokens=True, truncation=False)
        return premise

    @torch.inference_mode()
    def score(self, pairs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = list(pairs)
        output: list[dict[str, Any]] = []
        for start in range(0, len(rows), self.batch_size):
            batch = rows[start:start + self.batch_size]
            premises = [self.render_premise(row["claim_text"], row["unit"]) for row in batch]
            claims = [row["claim_text"] for row in batch]
            encoded = self.tokenizer(
                premises, claims, padding=True, truncation=False,
                max_length=self.max_length, return_tensors="pt",
            )
            logits = self.model(**encoded).logits
            probabilities = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            for row, values in zip(batch, probabilities):
                output.append({
                    **{key: value for key, value in row.items() if key != "unit"},
                    "unit_id": row["unit"]["unit_id"],
                    "unit_type": row["unit"]["unit_type"],
                    "chunk_ids": row["unit"]["chunk_ids"],
                    "entailment": float(values[self.label_indices["entailment"]]),
                    "neutral": float(values[self.label_indices["neutral"]]),
                    "contradiction": float(values[self.label_indices["contradiction"]]),
                })
        return output
