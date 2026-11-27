"""Step 7a: retrieval metrics, written by hand.

Every function takes ``ranked`` (document ids, best first, as a search returned them) and
``relevant`` (the set of document ids a human marked as correct for that query).

A search returns CHUNKS, and several chunks can come from one document. The labels are per
document, so repeats are removed first (``unique_in_order``): the best-placed chunk of each
document decides that document's position.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence


def unique_in_order(items: Iterable[str]) -> list[str]:
    """Drop repeats, keeping the first occurrence: ``[a, b, a, c]`` -> ``[a, b, c]``."""
    return list(dict.fromkeys(items))


def _check(relevant: Collection[str], k: int | None = None) -> None:
    if len(relevant) == 0:
        raise ValueError("relevant must contain at least one document id")
    if k is not None and k < 1:
        raise ValueError("k must be at least 1")


def hits_in_top_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> int:
    """How many distinct relevant documents appear in the first ``k`` documents."""
    _check(relevant, k)
    return sum(1 for doc in unique_in_order(ranked)[:k] if doc in relevant)


def precision_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Of the ``k`` slots, what fraction hold a relevant document? Always divides by ``k``,
    so returning fewer than ``k`` documents is penalised."""
    return hits_in_top_k(ranked, relevant, k) / k


def recall_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Of all the relevant documents, what fraction appears in the first ``k``?"""
    return hits_in_top_k(ranked, relevant, k) / len(relevant)


def hit_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """1.0 if at least one relevant document is in the first ``k``, else 0.0."""
    return 1.0 if hits_in_top_k(ranked, relevant, k) > 0 else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: Collection[str]) -> float:
    """``1 / position`` of the first relevant document (1st -> 1.0, 2nd -> 0.5, ...); 0.0 if none."""
    _check(relevant)
    for position, doc in enumerate(unique_in_order(ranked), start=1):
        if doc in relevant:
            return 1 / position
    return 0.0


def mean(values: Sequence[float]) -> float:
    """Plain average. Averaging a metric over all queries gives e.g. MRR or mean Recall@k."""
    if len(values) == 0:
        raise ValueError("cannot average an empty list")
    return sum(values) / len(values)
