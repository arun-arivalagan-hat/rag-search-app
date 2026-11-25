import math

import numpy as np
import pytest

from rag_lab.embeddings.vectors import dot_product, normalize, vector_norm


def test_dot_product_simple():
    assert dot_product([1, 2, 3], [4, 5, 6]) == 32          # 4 + 10 + 18


def test_dot_product_of_perpendicular_vectors_is_zero():
    assert dot_product([1, 0], [0, 1]) == 0


def test_dot_product_length_mismatch_rejected():
    with pytest.raises(ValueError, match="same length"):
        dot_product([1, 2], [1, 2, 3])


def test_norm_of_3_4_is_5():
    assert vector_norm([3, 4]) == 5


def test_norm_of_zero_vector_is_zero():
    assert vector_norm([0, 0, 0]) == 0


def test_normalize_gives_length_one_and_same_direction():
    unit = normalize([3, 4])
    assert unit == [0.6, 0.8]
    assert math.isclose(vector_norm(unit), 1.0)


def test_normalize_zero_vector_rejected():
    with pytest.raises(ValueError, match="zero vector"):
        normalize([0, 0])


def test_matches_numpy_on_random_vectors():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b = rng.normal(size=384), rng.normal(size=384)
        assert math.isclose(dot_product(a, b), float(np.dot(a, b)), rel_tol=1e-9)
        assert math.isclose(vector_norm(a), float(np.linalg.norm(a)), rel_tol=1e-9)
        assert np.allclose(normalize(a), a / np.linalg.norm(a))
