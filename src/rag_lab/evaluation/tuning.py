"""Step 7d: helpers for tuning search settings.

Tuning = try many combinations of settings, score each one with the evaluation runner, and
compare. Three small tools live here: ``grid`` (all combinations), ``CachingEmbedder``
(so the same query is never embedded twice) and ``marginal_means`` (how much does each
single setting matter on average?).
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def grid(**ranges: Sequence[Any]) -> list[dict[str, Any]]:
    """Every combination of the given values: ``grid(a=[1, 2], b=[10])`` -> two dicts."""
    if not ranges:
        raise ValueError("give at least one setting to vary")
    if any(len(values) == 0 for values in ranges.values()):
        raise ValueError("every setting needs at least one value")
    names = list(ranges)
    return [dict(zip(names, combo)) for combo in itertools.product(*ranges.values())]


class CachingEmbedder:
    """Wraps an embedder and remembers query vectors. Tuning asks for the same query
    hundreds of times (once per setting combination); the answer never changes."""

    def __init__(self, embedder: Any) -> None:
        self._embedder = embedder
        self._cache: dict[tuple[str, bool], np.ndarray] = {}
        self.model_calls = 0

    def embed_text(self, text: str, normalize: bool = True) -> np.ndarray:
        """Embed one query, reusing the saved vector if this exact query was seen before."""
        key = (text, normalize)
        if key not in self._cache:
            self.model_calls += 1
            self._cache[key] = self._embedder.embed_text(text, normalize=normalize)
        return self._cache[key]

    def embed_texts(self, texts: Sequence[str], normalize: bool = True) -> np.ndarray:
        """Embed many texts (the chunks); these are done once, so they are not cached."""
        return self._embedder.embed_texts(texts, normalize=normalize)


def marginal_means(rows: Sequence[Mapping[str, Any]], setting: str, metric: str) -> dict[Any, float]:
    """Average ``metric`` over all rows that share each value of ``setting``.

    ``rows`` are dicts holding the settings and the scores, e.g.
    ``{"k1": 1.5, "b": 0.75, "mrr": 0.97}``. The result answers "if I only look at ``b``,
    which value is best on average?". A flat result means the setting barely matters.
    """
    buckets: dict[Any, list[float]] = defaultdict(list)
    for row in rows:
        buckets[row[setting]].append(row[metric])
    if not buckets:
        raise ValueError("no rows to average")
    return {value: sum(scores) / len(scores) for value, scores in buckets.items()}
