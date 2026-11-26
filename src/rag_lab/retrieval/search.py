"""Step 5d: keyword search. Turn BM25 scores into a ranked top-k list of chunks.

``top_k`` sorts scores, best first. ``KeywordIndex`` ties the whole of Module 5 together:
chunks -> tokenize (5a) -> BM25 (5b, 5c) -> ranked results.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rag_lab.retrieval.bm25 import DEFAULT_B, DEFAULT_K1, BM25
from rag_lab.retrieval.tokenize import tokenize
from rag_lab.types import Chunk


def top_k(scores: Sequence[float], k: int = 5) -> list[tuple[int, float]]:
    """The ``k`` best ``(position, score)`` pairs, highest score first.

    Chunks scoring 0 matched nothing, so they are never returned (a query that matches
    nothing gives an empty list rather than ``k`` random chunks). Ties keep their original
    order, so results are repeatable.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])      # sorted() is stable
    return [(i, scores[i]) for i in ranked[:k] if scores[i] > 0]


@dataclass(frozen=True)
class KeywordHit:
    """One search result: the chunk, its BM25 score and its rank (1 = best)."""

    rank: int
    score: float
    chunk: Chunk


class KeywordIndex:
    """Keyword (BM25) search over a fixed list of chunks."""

    def __init__(self, chunks: Sequence[Chunk], k1: float = DEFAULT_K1, b: float = DEFAULT_B) -> None:
        if len(chunks) == 0:
            raise ValueError("need at least one chunk to search")
        self.chunks = list(chunks)
        self._bm25 = BM25([tokenize(c.text) for c in self.chunks], k1=k1, b=b)

    def search(self, query: str, k: int = 5) -> list[KeywordHit]:
        """The ``k`` chunks that best match ``query``, best first. Empty if nothing matches."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        scores = self._bm25.scores(tokenize(query))
        return [KeywordHit(rank=r, score=s, chunk=self.chunks[i]) for r, (i, s) in enumerate(top_k(scores, k), start=1)]
