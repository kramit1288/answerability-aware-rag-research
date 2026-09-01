"""Final one-time TechQA TEST evaluation under the frozen Phase 7 protocol."""

from .evaluation import run_test_inference
from .target import construct_test_target

__all__ = ["construct_test_target", "run_test_inference"]
