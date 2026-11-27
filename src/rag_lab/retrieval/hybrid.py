"""Step 6b: hybrid search = keyword (BM25) + semantic (embeddings), merged with RRF.

Keyword search is exact (finds rare words and codes but misses synonyms). Semantic search
understands meaning (finds paraphrases but can blur exact terms). Each makes its own ranking
of the chunks; ``reciprocal_rank_fusion`` (step 6a) merges the two rankings into one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from rag_lab.embeddings.embedder import Embedder
from rag_lab.embeddings.index import embed_chunks, nearest
from rag_lab.retrieval.bm25 import DEFAULT_B, DEFAULT_K1
from rag_lab.retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from rag_lab.retrieval.search import KeywordIndex
from rag_lab.types import Chunk


@dataclass(frozen=True)
class HybridHit:
    """One merged result. ``keyword_rank`` / ``semantic_rank`` are the positions (1 = best)
    each search gave this chunk, or ``None`` if that search did not return it."""

    rank: int
    score: float
    chunk: Chunk
    keyword_rank: int | None
    semantic_rank: int | None


class HybridIndex:
    """Hybrid search over a fixed list of chunks (chunk ids must be unique)."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        embedder: Embedder,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
        rrf_k: int = DEFAULT_RRF_K,
        matrix: np.ndarray | None = None,
    ) -> None:
        """``matrix`` is optional: the chunk embeddings if you already have them (row ``i``
        for ``chunks[i]``). Passing it skips the slow embedding step, which is what lets
        the tuner build hundreds of indexes cheaply."""
        if len({c.chunk_id for c in chunks}) != len(chunks):
            raise ValueError("chunk ids must be unique")
        self.chunks = list(chunks)
        self.embedder = embedder
        self.rrf_k = rrf_k
        self._by_id = {c.chunk_id: c for c in self.chunks}
        self._keyword = KeywordIndex(self.chunks, k1=k1, b=b)          # raises if there are no chunks
        if matrix is None:
            matrix = embed_chunks(embedder, self.chunks)
        elif matrix.ndim != 2 or matrix.shape[0] != len(self.chunks):
            raise ValueError("matrix must have one row per chunk")
        self._matrix = matrix

    def search(self, query: str, k: int = 5, candidates: int = 20) -> list[HybridHit]:
        """The ``k`` best chunks for ``query``, best first.

        Each search first proposes its own top ``candidates`` chunks (never fewer than ``k``);
        those two lists are then merged. Keyword search proposes nothing when no query word
        occurs anywhere, while semantic search always proposes its closest chunks.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if k < 1:
            raise ValueError("k must be at least 1")
        candidates = max(candidates, k)

        keyword_ids = [h.chunk.chunk_id for h in self._keyword.search(query, k=candidates)]
        query_vector = self.embedder.embed_text(query, normalize=True)
        semantic_ids = [self.chunks[i].chunk_id for i, _ in nearest(query_vector, self._matrix, k=candidates)]

        keyword_rank = {cid: r for r, cid in enumerate(keyword_ids, start=1)}
        semantic_rank = {cid: r for r, cid in enumerate(semantic_ids, start=1)}
        fused = reciprocal_rank_fusion([keyword_ids, semantic_ids], k=self.rrf_k, limit=k)
        return [
            HybridHit(
                rank=r,
                score=score,
                chunk=self._by_id[cid],
                keyword_rank=keyword_rank.get(cid),
                semantic_rank=semantic_rank.get(cid),
            )
            for r, (cid, score) in enumerate(fused, start=1)
        ]
