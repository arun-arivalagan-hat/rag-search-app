"""Step 4b: cosine similarity, written by hand from the step 4a helpers.

Cosine similarity compares the DIRECTION of two vectors and ignores their length:
    cosine(a, b) = (a . b) / (|a| * |b|)
1 means the same direction (very similar meaning), 0 means unrelated, -1 means opposite.
"""

from __future__ import annotations

from collections.abc import Sequence

from rag_lab.embeddings.vectors import dot_product, vector_norm


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine of the angle between ``a`` and ``b``, always in [-1, 1]."""
    norm_a, norm_b = vector_norm(a), vector_norm(b)
    if norm_a == 0 or norm_b == 0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    score = dot_product(a, b) / (norm_a * norm_b)
    return max(-1.0, min(1.0, score))      # rounding can give 1.0000000000000002
