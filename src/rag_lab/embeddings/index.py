"""Step 4d: embed every chunk, then find the chunks nearest to a query.

``embed_chunks`` returns a matrix with one row per chunk, in the same order as the chunks.
``nearest`` scores a query against every row with a dot product. Because the embedder
normalises its vectors to length 1, that dot product IS the cosine similarity (step 4b).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from rag_lab.embeddings.embedder import Embedder
from rag_lab.types import Chunk


def embed_chunks(embedder: Embedder, chunks: Sequence[Chunk]) -> np.ndarray:
    """Embed ``chunk.text`` for every chunk. Row ``i`` belongs to ``chunks[i]``."""
    return embedder.embed_texts([c.text for c in chunks], normalize=True)


def nearest(query_vector: np.ndarray, matrix: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
    """The ``k`` rows of ``matrix`` most similar to ``query_vector``, best first.

    Returns ``(row_index, score)`` pairs. Both inputs must be normalised (length 1).
    Ties keep their original order. If ``k`` is larger than the number of rows, all rows
    are returned.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if matrix.ndim != 2 or query_vector.ndim != 1:
        raise ValueError("expected a 2-D matrix and a 1-D query vector")
    if matrix.shape[1] != query_vector.shape[0]:
        raise ValueError(f"dimension mismatch: matrix has {matrix.shape[1]}, query has {query_vector.shape[0]}")
    scores = matrix @ query_vector                     # one dot product per chunk
    order = np.argsort(-scores, kind="stable")[:k]     # highest score first
    return [(int(i), float(scores[i])) for i in order]
