"""Step 3d: overlap between neighbouring chunks.

Without overlap, a fact that sits across a cut is split in two halves and neither chunk
contains all of it. Overlap repeats the tail of the previous chunk at the start of the next
one. It works on the output of ANY chunker, so each strategy stays simple.
"""

from __future__ import annotations

from rag_lab.chunking.base import make_chunk
from rag_lab.types import Chunk, Document, verify_chunk_integrity


def _overlap_start(text: str, wanted: int, floor: int, start: int) -> int:
    """Move ``wanted`` forward to the nearest word start, never past ``start``.

    ``floor`` is the earliest allowed position (the start of the previous chunk).
    Returns ``start`` itself when no clean overlap fits.
    """
    pos = max(wanted, floor)
    if pos >= start:
        return start
    if pos > 0 and not text[pos - 1].isspace() and not text[pos].isspace():
        while pos < start and not text[pos].isspace():      # we are inside a word: skip its end
            pos += 1
    while pos < start and text[pos].isspace():              # skip the gap to the next word
        pos += 1
    return min(pos, start)


def add_overlap(document: Document, chunks: list[Chunk], overlap: int) -> list[Chunk]:
    """Return new chunks where each one (except the first) starts up to ``overlap`` characters
    earlier, on a word boundary, so it repeats the end of the chunk before it.

    A chunk can therefore grow to ``max_chars + overlap``. The end offsets, ids, order and
    strategy do not change; ``overlap_chars`` in the metadata records the real overlap.
    """
    if overlap < 0:
        raise ValueError("overlap must be 0 or more")
    result: list[Chunk] = []
    for i, chunk in enumerate(chunks):
        start = chunk.start_char
        if i > 0 and overlap:
            start = _overlap_start(document.text, chunk.start_char - overlap, chunks[i - 1].start_char, chunk.start_char)
        extra = {"overlap_chars": chunk.start_char - start}
        result.append(make_chunk(document, i, start, chunk.end_char, chunk.metadata["chunk_strategy"], extra))
    verify_chunk_integrity(document, result)
    return result
