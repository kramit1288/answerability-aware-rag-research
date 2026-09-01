"""Frozen deterministic reference-answer quality metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rouge_score.rouge_scorer import RougeScorer


def rouge_l_f1(prediction: str, reference: str) -> float:
    scorer = RougeScorer(["rougeL"], use_stemmer=False)
    return float(scorer.score(str(reference), str(prediction))["rougeL"].fmeasure)


@dataclass
class FrozenBERTScorer:
    snapshot_path: Path
    num_layers: int = 5
    batch_size: int = 8

    def __post_init__(self) -> None:
        from bert_score import BERTScorer

        self.scorer = BERTScorer(
            model_type=str(self.snapshot_path.resolve()),
            num_layers=self.num_layers,
            idf=False,
            rescale_with_baseline=False,
            device="cpu",
            batch_size=self.batch_size,
            use_fast_tokenizer=True,
        )

    def score(self, predictions: list[str], references: list[str]) -> list[float]:
        if len(predictions) != len(references):
            raise ValueError("BERTScore prediction/reference lengths differ")
        if not predictions:
            return []
        _, _, f1 = self.scorer.score(predictions, references)
        return [float(value) for value in np.asarray(f1.cpu(), dtype=np.float64)]
