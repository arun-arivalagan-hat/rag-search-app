"""Step 4a: vector basics, written by hand.

A vector is just a list of numbers. These three helpers are the building blocks of cosine
similarity (step 4b). Plain Python loops are used on purpose so every step is visible;
the tests check them against numpy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def dot_product(a: Sequence[float], b: Sequence[float]) -> float:
    """Multiply matching positions and add them up: a[0]*b[0] + a[1]*b[1] + ..."""
    if len(a) != len(b):
        raise ValueError(f"vectors must have the same length, got {len(a)} and {len(b)}")
    return sum(x * y for x, y in zip(a, b))


def vector_norm(a: Sequence[float]) -> float:
    """The length of the vector: the square root of (a dot a)."""
    return math.sqrt(dot_product(a, a))


def normalize(a: Sequence[float]) -> list[float]:
    """Scale the vector to length 1 while keeping its direction.

    A vector of all zeros has no direction, so it cannot be normalised.
    """
    length = vector_norm(a)
    if length == 0:
        raise ValueError("cannot normalize a zero vector")
    return [x / length for x in a]
