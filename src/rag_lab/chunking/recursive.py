"""Step 3c: recursive chunking.

Try the biggest natural boundary first (paragraphs). Any piece that is still too long is
split again with the next smaller boundary (lines, then sentences, then words, and as a last
resort raw characters). Small neighbouring pieces are then packed back together up to
``max_chars``. Like the other chunkers it only decides WHERE to cut, using (start, end) spans.
"""

from __future__ import annotations

import re

from rag_lab.chunking.base import build_chunks
from rag_lab.chunking.fixed import fixed_size_spans
from rag_lab.chunking.sentences import _PARAGRAPH_BREAK, _content_span, sentence_spans
from rag_lab.types import Chunk, Document

STRATEGY = "recursive"

_LINE_BREAK = re.compile(r"\n")
_WORD_BREAK = re.compile(r"\s+")
# From the biggest boundary to the smallest. The last level (characters) always succeeds.
LEVELS = ("paragraph", "line", "sentence", "word", "character")

Span = tuple[int, int]


def _split(text: str, start: int, end: int, level: int, max_chars: int) -> list[Span]:
    """Cut ``text[start:end]`` at the boundary type for ``level``.

    The separator stays attached to the piece before it, so the pieces are consecutive and
    cover every character.
    """
    name = LEVELS[level]
    if name == "sentence":
        return [(start + a, start + b) for a, b in sentence_spans(text[start:end])]
    if name == "character":
        return [(start + a, start + b) for a, b in fixed_size_spans(end - start, max_chars)]
    pattern = {"paragraph": _PARAGRAPH_BREAK, "line": _LINE_BREAK, "word": _WORD_BREAK}[name]
    cuts = sorted({m.end() for m in pattern.finditer(text, start, end)} | {end})
    spans, piece_start = [], start
    for cut in cuts:
        if cut > piece_start:
            spans.append((piece_start, cut))
            piece_start = cut
    return spans


def _pack(spans: list[Span], max_chars: int) -> list[Span]:
    """Greedily merge consecutive spans while the merged span still fits."""
    packed: list[Span] = []
    for start, end in spans:
        if packed and end - packed[-1][0] <= max_chars:
            packed[-1] = (packed[-1][0], end)
        else:
            packed.append((start, end))
    return packed


def recursive_spans(text: str, max_chars: int, start: int = 0, end: int | None = None, level: int = 0) -> list[Span]:
    """Spans of at most ``max_chars`` characters, cut at the biggest boundary that works."""
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1")
    end = len(text) if end is None else end
    if end <= start:
        return []
    if end - start <= max_chars:
        return [(start, end)]
    pieces = _split(text, start, end, level, max_chars)
    if len(pieces) == 1 and level < len(LEVELS) - 1:         # this boundary does not occur here
        return recursive_spans(text, max_chars, start, end, level + 1)
    small: list[Span] = []
    for a, b in pieces:
        small.extend(recursive_spans(text, max_chars, a, b, level + 1) if b - a > max_chars else [(a, b)])
    return _pack(small, max_chars)


def chunk_recursive(document: Document, max_chars: int = 500) -> list[Chunk]:
    """Split ``document.text`` into chunks of at most ``max_chars`` characters.

    Paragraphs are kept whole when they fit; only too-long pieces are split further.
    Leading and trailing whitespace is trimmed from each chunk (offsets stay exact).
    """
    text = document.text
    trimmed = [s for s in (_content_span(text, a, b) for a, b in recursive_spans(text, max_chars)) if s]
    return build_chunks(document, trimmed, STRATEGY)
