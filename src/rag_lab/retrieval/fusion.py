"""Step 6a: Reciprocal Rank Fusion (RRF), merging several rankings into one.

Keyword scores (BM25, e.g. 16.9) and semantic scores (cosine, e.g. 0.53) live on different
scales, so they cannot be added. RRF ignores the scores and uses only the RANK (1st, 2nd,
3rd ...). Each ranking gives an item ``1 / (k + rank)`` points and the points are added up.
An item that is high in several rankings beats an item that is first in only one.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence

DEFAULT_RRF_K = 60       # the value from the original RRF paper; larger = flatter, smaller = top-heavy


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Hashable]], k: int = DEFAULT_RRF_K, limit: int | None = None
) -> list[tuple[Hashable, float]]:
    """Merge rankings (each a list of ids, best first) into ``(id, rrf_score)`` pairs, best first.

    An id missing from a ranking simply earns nothing from it. Ties keep the order in which
    the ids were first seen, so results are repeatable. ``limit`` keeps only the top results.
    """
    if k < 0:
        raise ValueError("k must be 0 or more")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    scores: dict[Hashable, float] = {}               # dicts keep insertion order = first-seen order
    for ranking in rankings:
        if len(set(ranking)) != len(ranking):
            raise ValueError("a ranking must not list the same id twice")
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1 / (k + rank)
    fused = sorted(scores.items(), key=lambda pair: -pair[1])      # stable sort keeps first-seen on ties
    return fused[:limit]
