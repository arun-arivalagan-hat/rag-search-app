import math

import numpy as np
import pytest

from rag_lab.evaluation.tuning import CachingEmbedder, grid, marginal_means
from rag_lab.retrieval.hybrid import HybridIndex
from rag_lab.types import Chunk


# ------------------------------------------------ grid
def test_grid_makes_every_combination():
    combos = grid(a=[1, 2], b=["x", "y", "z"])
    assert len(combos) == 6
    assert {"a": 2, "b": "y"} in combos


def test_grid_with_one_setting():
    assert grid(k=[1, 2, 3]) == [{"k": 1}, {"k": 2}, {"k": 3}]


def test_grid_rejects_empty_input():
    with pytest.raises(ValueError):
        grid()
    with pytest.raises(ValueError):
        grid(a=[])


# ------------------------------------------------ CachingEmbedder
class CountingEmbedder:
    def __init__(self):
        self.calls = 0

    def embed_text(self, text, normalize=True):
        self.calls += 1
        return np.array([float(len(text)), 1.0], dtype=np.float32)

    def embed_texts(self, texts, normalize=True):
        return np.array([self.embed_text(t) for t in texts])


def test_same_query_is_embedded_only_once():
    inner = CountingEmbedder()
    cached = CachingEmbedder(inner)
    first = cached.embed_text("moon")
    second = cached.embed_text("moon")
    assert inner.calls == 1 and cached.model_calls == 1
    assert np.array_equal(first, second)


def test_different_queries_are_each_embedded():
    inner = CountingEmbedder()
    cached = CachingEmbedder(inner)
    cached.embed_text("moon")
    cached.embed_text("sun")
    assert inner.calls == 2


def test_embed_texts_passes_through():
    assert CachingEmbedder(CountingEmbedder()).embed_texts(["a", "bb"]).shape == (2, 2)


# ------------------------------------------------ marginal_means
def test_marginal_means_average_per_value():
    rows = [
        {"b": 0.0, "mrr": 0.5},
        {"b": 0.0, "mrr": 1.0},
        {"b": 1.0, "mrr": 0.25},
    ]
    result = marginal_means(rows, "b", "mrr")
    assert math.isclose(result[0.0], 0.75)
    assert math.isclose(result[1.0], 0.25)


def test_flat_setting_gives_equal_means():
    rows = [{"k": 1, "mrr": 0.9}, {"k": 2, "mrr": 0.9}]
    assert len(set(marginal_means(rows, "k", "mrr").values())) == 1


def test_marginal_means_of_nothing_rejected():
    with pytest.raises(ValueError):
        marginal_means([], "k", "mrr")


# ------------------------------------------------ HybridIndex accepts a precomputed matrix
class NeverCalledEmbedder:
    def embed_texts(self, texts, normalize=True):
        raise AssertionError("the chunks must not be embedded when a matrix is supplied")

    def embed_text(self, text, normalize=True):
        return np.array([1.0, 0.0], dtype=np.float32)


def make_chunks(n):
    return [Chunk(chunk_id=f"c{i}", doc_id="d", text=f"word{i}", start_char=0, end_char=5) for i in range(n)]


def test_supplied_matrix_skips_embedding_the_chunks():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index = HybridIndex(make_chunks(2), NeverCalledEmbedder(), matrix=matrix)
    assert index.search("anything", k=1)[0].chunk.chunk_id == "c0"      # row 0 is closest to [1, 0]


def test_matrix_with_wrong_row_count_rejected():
    with pytest.raises(ValueError, match="one row per chunk"):
        HybridIndex(make_chunks(3), NeverCalledEmbedder(), matrix=np.zeros((2, 2), dtype=np.float32))
