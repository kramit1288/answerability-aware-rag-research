"""Frozen lexical preprocessing used by both BM25 indexing and querying."""

from __future__ import annotations

import re
import unicodedata


# A token consists of one or more Unicode word characters (letters, numbers, or
# underscore), optionally joined by technical separators. Separators are only
# retained internally, never at token boundaries.
BM25_TOKEN_PATTERN = r"(?u)\w+(?:[./:-]+\w+)*"
_BM25_TOKEN_RE = re.compile(BM25_TOKEN_PATTERN)


def normalize_bm25_text(text: str) -> str:
    """Apply the frozen Unicode normalization and case handling."""
    if not isinstance(text, str):
        raise TypeError("BM25 preprocessing requires a string")
    return unicodedata.normalize("NFKC", text).casefold()


def tokenize_bm25(text: str) -> list[str]:
    """Tokenize text without stop-word removal, stemming, or lemmatization."""
    return _BM25_TOKEN_RE.findall(normalize_bm25_text(text))
