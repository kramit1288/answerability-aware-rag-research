from __future__ import annotations

import hashlib

from answerability_rag.hashing import normalize_question_for_grouping, normalized_question_sha256


def test_frozen_question_normalization_handles_unicode_case_punctuation_and_space() -> None:
    source = "  ＣＡＦÉ—Test!!\tNext\nline…  "
    assert normalize_question_for_grouping(source) == "café test next line"


def test_normalized_hash_is_deterministic_across_equivalent_forms() -> None:
    variants = ["Café—TEST?", "Cafe\u0301 test", "  CAFÉ...test  "]
    hashes = {normalized_question_sha256(value) for value in variants}
    expected = hashlib.sha256("café test".encode("utf-8")).hexdigest()
    assert hashes == {expected}


def test_symbols_are_retained_while_all_unicode_punctuation_is_removed() -> None:
    # '+' is Unicode Symbol/Math and remains; '.', '#', and '?' are punctuation and become spaces.
    assert normalize_question_for_grouping("C++ vs. C#?") == "c++ vs c"
