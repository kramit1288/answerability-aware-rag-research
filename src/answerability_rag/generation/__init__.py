"""Phase 5 deterministic generation, answer-quality, and grounding evaluation."""

from .claims import ClaimSegment, segment_claims
from .config import PHASE05_CONFIG_SHA256, Phase05Config, assert_techqa_split_allowed

__all__ = [
    "ClaimSegment",
    "PHASE05_CONFIG_SHA256",
    "Phase05Config",
    "assert_techqa_split_allowed",
    "segment_claims",
]
