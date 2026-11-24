"""Shared pieces for every chunker (Module 3).

A chunker only decides WHERE to cut. It never rewrites text. ``make_chunk`` is the one
place that turns a (start, end) span into a ``Chunk``, so every strategy produces the
same shape, the same ids and the same inherited metadata.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rag_lab.types import Chunk, Document, verify_chunk_integrity


def make_chunk(
    document: Document, index: int, start: int, end: int, strategy: str, extra: Mapping[str, Any] | None = None
) -> Chunk:
    """Build the chunk for ``document.text[start:end]`` (offsets are into ``document.text``)."""
    if not 0 <= start < end <= len(document.text):
        raise ValueError(f"span [{start}:{end}] is outside the document (length {len(document.text)})")
    metadata = dict(document.metadata)
    metadata.update(chunk_index=index, chunk_strategy=strategy)
    metadata.update(extra or {})
    return Chunk(
        chunk_id=Chunk.make_id(document.doc_id, index),
        doc_id=document.doc_id,
        text=document.text[start:end],
        start_char=start,
        end_char=end,
        metadata=metadata,
    )


def build_chunks(document: Document, spans: list[tuple[int, int]], strategy: str) -> list[Chunk]:
    """Turn spans into numbered chunks, skipping spans that are only whitespace.

    Whitespace-only spans carry no meaning (and would embed as noise), so they are
    dropped; the remaining chunks keep their true offsets. Indexes stay consecutive.
    """
    chunks: list[Chunk] = []
    for start, end in spans:
        if document.text[start:end].strip():
            chunks.append(make_chunk(document, len(chunks), start, end, strategy))
    verify_chunk_integrity(document, chunks)
    return chunks
