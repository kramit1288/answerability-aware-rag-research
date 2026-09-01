"""Deterministic generated-response claim segmentation with source offsets."""

from __future__ import annotations

import re
from dataclasses import dataclass


_BULLET = re.compile(r"^\s*(?:[-*\u2022]+|\(?\d+[.)]|[A-Za-z][.)])\s+")
_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(<\[])")
_ALNUM = re.compile(r"[^\W_]", re.UNICODE)


@dataclass(frozen=True)
class ClaimSegment:
    claim_index: int
    text: str
    start: int
    end: int


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def segment_claims(response: str) -> list[ClaimSegment]:
    """Split on line/bullet and conservative punctuation boundaries without rewriting."""
    if not isinstance(response, str) or not response.strip():
        return []
    spans: list[tuple[int, int]] = []
    for line_match in re.finditer(r"[^\r\n]+", response):
        line_start, line_end = line_match.span()
        start, end = _trim_span(response, line_start, line_end)
        if start >= end:
            continue
        marker = _BULLET.match(response[start:end])
        if marker:
            start += marker.end()
            start, end = _trim_span(response, start, end)
        if start >= end:
            continue
        cursor = start
        for boundary in _BOUNDARY.finditer(response, start, end):
            part_start, part_end = _trim_span(response, cursor, boundary.start())
            if part_start < part_end:
                spans.append((part_start, part_end))
            cursor = boundary.end()
        part_start, part_end = _trim_span(response, cursor, end)
        if part_start < part_end:
            spans.append((part_start, part_end))
    output: list[ClaimSegment] = []
    for start, end in spans:
        claim = response[start:end]
        if _ALNUM.search(claim):
            output.append(ClaimSegment(len(output) + 1, claim, start, end))
    return output
