"""Core data types shared by every module.

Three small, immutable records flow through the whole system:

    Document         one ingested file (full cleaned text + metadata)
    Chunk            a slice of a Document, with exact character offsets
    RetrievalResult  a Chunk plus the score/rank a retriever gave it

Offsets are the key design decision: every Chunk records ``start_char`` and
``end_char`` into ``Document.text`` so that

    document.text[chunk.start_char:chunk.end_char] == chunk.text

always holds. That invariant is checked by :func:`verify_chunk_integrity`, which
the chunking tests call for every strategy. It guarantees citations can point
back to the exact place in the source.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

Metadata = Mapping[str, Any]


def _freeze(meta: Metadata | None) -> Metadata:
    return MappingProxyType(dict(meta or {}))


@dataclass(frozen=True)
class Document:
    """A whole source document after loading (and, later, cleaning)."""

    doc_id: str
    text: str
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.doc_id:
            raise ValueError("doc_id must be non-empty")
        object.__setattr__(self, "metadata", _freeze(self.metadata))


@dataclass(frozen=True)
class Chunk:
    """A contiguous slice of a document.

    ``chunk_id`` is ``"{doc_id}::{index}"`` by convention (see :meth:`make_id`).
    Metadata is inherited from the parent document so retrievers can filter on it.
    """

    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("chunk text must be non-empty")
        if self.start_char < 0 or self.end_char <= self.start_char:
            raise ValueError(
                f"invalid offsets: start={self.start_char}, end={self.end_char}"
            )
        object.__setattr__(self, "metadata", _freeze(self.metadata))

    @staticmethod
    def make_id(doc_id: str, index: int) -> str:
        """Build a chunk id from its document and position, for example ``mars::0``."""
        return f"{doc_id}::{index}"

    def __len__(self) -> int:
        return len(self.text)


@dataclass(frozen=True)
class RetrievalResult:
    """One ranked hit. ``rank`` is 1-based (rank 1 = best).

    ``score`` is only comparable within a single retriever: cosine similarity,
    BM25 and RRF scores live on different scales. That is exactly why fusion
    (Module 10) is needed.
    """

    chunk: Chunk
    score: float
    rank: int
    retriever: str

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("rank is 1-based")


def verify_chunk_integrity(document: Document, chunks: list[Chunk]) -> None:
    """Raise ``AssertionError`` if any chunk disagrees with its source text.

    Checks, per chunk:
      * it belongs to ``document``
      * its offsets lie inside the document
      * ``document.text[start:end] == chunk.text``  (the text-integrity invariant)
    """
    for c in chunks:
        assert c.doc_id == document.doc_id, f"{c.chunk_id}: wrong doc_id {c.doc_id!r}"
        assert c.end_char <= len(document.text), f"{c.chunk_id}: end beyond document"
        actual = document.text[c.start_char : c.end_char]
        assert actual == c.text, (
            f"{c.chunk_id}: text mismatch at [{c.start_char}:{c.end_char}]\n"
            f"  expected: {actual[:60]!r}\n  got:      {c.text[:60]!r}"
        )
