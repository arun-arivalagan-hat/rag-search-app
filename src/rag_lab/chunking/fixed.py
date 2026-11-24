"""Step 3a: fixed-size chunking.

Cut the text every ``chunk_size`` characters, ignoring words, sentences and paragraphs.
It is the simplest possible strategy and the baseline the smarter chunkers are compared to.
"""

from __future__ import annotations

from rag_lab.chunking.base import build_chunks
from rag_lab.types import Chunk, Document

STRATEGY = "fixed"


def fixed_size_spans(length: int, chunk_size: int) -> list[tuple[int, int]]:
    """Spans (start, end) covering ``range(length)`` in steps of ``chunk_size``."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    return [(start, min(start + chunk_size, length)) for start in range(0, length, chunk_size)]


def chunk_fixed_size(document: Document, chunk_size: int = 500) -> list[Chunk]:
    """Split ``document.text`` into consecutive pieces of ``chunk_size`` characters.

    The last piece may be shorter. Pieces do not overlap (overlap is step 3d).
    """
    spans = fixed_size_spans(len(document.text), chunk_size)
    return build_chunks(document, spans, STRATEGY)
