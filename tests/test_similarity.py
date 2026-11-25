import math

import numpy as np
import pytest
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

from rag_lab.embeddings.similarity import cosine_similarity


def test_same_direction_is_one():
    assert math.isclose(cosine_similarity([1, 2, 3], [2, 4, 6]), 1.0)     # longer, same direction


def test_perpendicular_is_zero():
    assert cosine_similarity([1, 0], [0, 1]) == 0


def test_opposite_is_minus_one():
    assert math.isclose(cosine_similarity([1, 2], [-1, -2]), -1.0)


def test_length_does_not_matter():
    assert math.isclose(cosine_similarity([1, 1], [5, 5]), cosine_similarity([1, 1], [100, 100]))


def test_known_45_degree_angle():
    assert math.isclose(cosine_similarity([1, 0], [1, 1]), math.sqrt(2) / 2)


def test_symmetric():
    a, b = [1, 2, 3], [3, 1, 2]
    assert math.isclose(cosine_similarity(a, b), cosine_similarity(b, a))


@pytest.mark.parametrize("zero_first", [True, False])
def test_zero_vector_rejected(zero_first):
    pair = ([0, 0], [1, 1]) if zero_first else ([1, 1], [0, 0])
    with pytest.raises(ValueError, match="zero vector"):
        cosine_similarity(*pair)


def test_length_mismatch_rejected():
    with pytest.raises(ValueError, match="same length"):
        cosine_similarity([1, 2], [1, 2, 3])


def test_result_never_leaves_minus_one_to_one():
    v = [0.1] * 384
    assert -1.0 <= cosine_similarity(v, v) <= 1.0


def test_matches_scikit_learn_on_random_vectors():
    rng = np.random.default_rng(1)
    for _ in range(50):
        a, b = rng.normal(size=384), rng.normal(size=384)
        expected = float(sk_cosine(a.reshape(1, -1), b.reshape(1, -1))[0, 0])
        assert math.isclose(cosine_similarity(a, b), expected, abs_tol=1e-9)
