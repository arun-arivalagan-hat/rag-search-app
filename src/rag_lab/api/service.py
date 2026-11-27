"""Step 9a: one search function for the three methods the frontend offers.

The page has a Method switch (Vector, BM25, Hybrid). ``SearchService.search`` takes the
method name and always returns the same shape, so the HTTP layer does not care which search
ran. All three searches share the same chunks and the same embedding matrix, which are built
once when the service is created (embedding is the slow part).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from rag_lab.chunking import add_overlap, chunk_recursive
from rag_lab.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, Settings
from rag_lab.embeddings.embedder import Embedder
from rag_lab.embeddings.index import embed_chunks, nearest
from rag_lab.ingestion import clean_corpus, load_corpus
from rag_lab.retrieval import HybridIndex, KeywordIndex
from rag_lab.types import Chunk

METHODS = ("vector", "bm25", "hybrid")


@dataclass(frozen=True)
class SearchResult:
    """One ranked hit. ``relative`` is score / best score (0 to 1), used to draw the score bar."""
    rank: int                # 1 = best
    score: float             # raw score; its scale depends on the method (BM25 ~ 10, cosine ~ 0.5, RRF ~ 0.03)
    relative: float          # score divided by the best score, 0..1, for drawing the score bar
    chunk: Chunk


class SearchService:
    """Vector, BM25 and hybrid search over one fixed list of chunks."""

    def __init__(self, chunks: Sequence[Chunk], embedder: Any) -> None:
        self.chunks = list(chunks)
        self.embedder = embedder
        self._matrix = embed_chunks(embedder, self.chunks)
        self._keyword = KeywordIndex(self.chunks)                       # raises if there are no chunks
        self._hybrid = HybridIndex(self.chunks, embedder, matrix=self._matrix)

    def search(self, question: str, method: str = "hybrid", k: int = 5) -> list[SearchResult]:
        """The ``k`` best chunks for ``question``, best first. BM25 returns fewer (or none) when
        few chunks share a word with the question; vector search always returns ``k``."""
        if method not in METHODS:
            raise ValueError(f"method must be one of {list(METHODS)}, got {method!r}")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if k < 1:
            raise ValueError("k must be at least 1")

        if method == "vector":
            vector = self.embedder.embed_text(question, normalize=True)
            pairs = [(self.chunks[i], score) for i, score in nearest(vector, self._matrix, k=k)]
        elif method == "bm25":
            pairs = [(h.chunk, h.score) for h in self._keyword.search(question, k=k)]
        else:
            pairs = [(h.chunk, h.score) for h in self._hybrid.search(question, k=k)]

        best = pairs[0][1] if pairs else 0.0
        return [
            SearchResult(rank=r, score=score, relative=(max(score, 0.0) / best if best > 0 else 0.0), chunk=chunk)
            for r, (chunk, score) in enumerate(pairs, start=1)
        ]


def build_service(settings: Settings) -> SearchService:
    """Load the sample corpus, chunk it with the project defaults and index it."""
    docs = [c.retrieval_document for c in clean_corpus(load_corpus(settings.sample_docs_dir))]
    chunks = [c for d in docs for c in add_overlap(d, chunk_recursive(d, DEFAULT_CHUNK_SIZE), DEFAULT_CHUNK_OVERLAP)]
    return SearchService(chunks, Embedder(settings.embedding_model))
