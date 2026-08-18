"""Deterministic evidence normalization with reversible character mappings."""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass

from answerability_rag.data.techqa import _DATA_URI, normalize_corpus_text


_ENTITY = re.compile(r"&(?:#[0-9]+|#[xX][0-9A-Fa-f]+|[A-Za-z][A-Za-z0-9]+);")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class MappedText:
    text: str
    source_starts: tuple[int, ...]
    source_ends: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.text) != len(self.source_starts) or len(self.text) != len(self.source_ends):
            raise ValueError("mapped text and offset arrays differ in length")

    def source_span(self, start: int, end: int) -> tuple[int, int]:
        if not 0 <= start < end <= len(self.text):
            raise ValueError(f"invalid mapped interval [{start},{end})")
        return min(self.source_starts[start:end]), max(self.source_ends[start:end])


def _from_text(text: str) -> MappedText:
    return MappedText(text, tuple(range(len(text))), tuple(range(1, len(text) + 1)))


def _replace_line_endings(value: MappedText) -> MappedText:
    chars: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    index = 0
    while index < len(value.text):
        if value.text[index:index + 2] == "\r\n":
            chars.append("\n")
            starts.append(value.source_starts[index])
            ends.append(value.source_ends[index + 1])
            index += 2
        elif value.text[index] == "\r":
            chars.append("\n")
            starts.append(value.source_starts[index])
            ends.append(value.source_ends[index])
            index += 1
        else:
            chars.append(value.text[index])
            starts.append(value.source_starts[index])
            ends.append(value.source_ends[index])
            index += 1
    return MappedText("".join(chars), tuple(starts), tuple(ends))


def _unicode_transform(value: MappedText, *, casefold: bool) -> MappedText:
    chars: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    index = 0
    while index < len(value.text):
        end = index + 1
        while end < len(value.text) and unicodedata.combining(value.text[end]):
            end += 1
        transformed = unicodedata.normalize("NFKC", value.text[index:end])
        if casefold:
            transformed = transformed.casefold()
        source_start = min(value.source_starts[index:end])
        source_end = max(value.source_ends[index:end])
        chars.extend(transformed)
        starts.extend([source_start] * len(transformed))
        ends.extend([source_end] * len(transformed))
        index = end
    expected = unicodedata.normalize("NFKC", value.text)
    if casefold:
        expected = expected.casefold()
    observed = "".join(chars)
    if observed != expected:
        raise ValueError("Unicode normalization crossed an unsupported cluster boundary")
    return MappedText(observed, tuple(starts), tuple(ends))


def _regex_replace_with_space(value: MappedText, pattern: re.Pattern[str]) -> MappedText:
    chars: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    cursor = 0
    for match in pattern.finditer(value.text):
        for index in range(cursor, match.start()):
            chars.append(value.text[index])
            starts.append(value.source_starts[index])
            ends.append(value.source_ends[index])
        chars.append(" ")
        starts.append(min(value.source_starts[match.start():match.end()]))
        ends.append(max(value.source_ends[match.start():match.end()]))
        cursor = match.end()
    for index in range(cursor, len(value.text)):
        chars.append(value.text[index])
        starts.append(value.source_starts[index])
        ends.append(value.source_ends[index])
    return MappedText("".join(chars), tuple(starts), tuple(ends))


def _collapse_whitespace(value: MappedText) -> MappedText:
    chars: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    cursor = 0
    for match in _WHITESPACE.finditer(value.text):
        for index in range(cursor, match.start()):
            chars.append(value.text[index])
            starts.append(value.source_starts[index])
            ends.append(value.source_ends[index])
        chars.append(" ")
        starts.append(min(value.source_starts[match.start():match.end()]))
        ends.append(max(value.source_ends[match.start():match.end()]))
        cursor = match.end()
    for index in range(cursor, len(value.text)):
        chars.append(value.text[index])
        starts.append(value.source_starts[index])
        ends.append(value.source_ends[index])
    while chars and chars[0] == " ":
        chars.pop(0); starts.pop(0); ends.pop(0)
    while chars and chars[-1] == " ":
        chars.pop(); starts.pop(); ends.pop()
    return MappedText("".join(chars), tuple(starts), tuple(ends))


def normalize_retrieval_source_with_mapping(raw_text: str) -> MappedText:
    """Reproduce the Phase 2 source transform and map every output char to raw offsets."""
    value = _replace_line_endings(_from_text(raw_text))
    value = _unicode_transform(value, casefold=False)
    value = _regex_replace_with_space(value, _DATA_URI)
    value = _collapse_whitespace(value)
    expected, _ = normalize_corpus_text(raw_text)
    if value.text != expected:
        raise ValueError("mapped retrieval normalization differs from the frozen Phase 2 transform")
    return value


def _html_unescape(value: MappedText) -> MappedText:
    chars: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    cursor = 0
    for match in _ENTITY.finditer(value.text):
        for index in range(cursor, match.start()):
            chars.append(value.text[index])
            starts.append(value.source_starts[index])
            ends.append(value.source_ends[index])
        replacement = html.unescape(match.group())
        if replacement == match.group():
            for index in range(match.start(), match.end()):
                chars.append(value.text[index])
                starts.append(value.source_starts[index])
                ends.append(value.source_ends[index])
        else:
            source_start = min(value.source_starts[match.start():match.end()])
            source_end = max(value.source_ends[match.start():match.end()])
            chars.extend(replacement)
            starts.extend([source_start] * len(replacement))
            ends.extend([source_end] * len(replacement))
        cursor = match.end()
    for index in range(cursor, len(value.text)):
        chars.append(value.text[index])
        starts.append(value.source_starts[index])
        ends.append(value.source_ends[index])
    return MappedText("".join(chars), tuple(starts), tuple(ends))


def normalize_alignment_text_with_mapping(source_text: str) -> MappedText:
    """Apply the frozen exact/normalized alignment rule with reversible source offsets."""
    value = _html_unescape(_from_text(source_text))
    value = _unicode_transform(value, casefold=True)
    punctuation = MappedText(
        "".join(" " if unicodedata.category(char).startswith("P") else char for char in value.text),
        value.source_starts,
        value.source_ends,
    )
    return _collapse_whitespace(punctuation)


def normalize_alignment_text(text: str) -> str:
    return normalize_alignment_text_with_mapping(text).text


def find_all(text: str, needle: str) -> list[tuple[int, int]]:
    if not needle:
        return []
    matches: list[tuple[int, int]] = []
    start = 0
    while True:
        position = text.find(needle, start)
        if position < 0:
            return matches
        matches.append((position, position + len(needle)))
        start = position + 1
