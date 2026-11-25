import math

import numpy as np
import pytest

from rag_lab.embeddings.embedder import Embedder
from rag_lab.embeddings.index import embed_chunks, nearest
from rag_lab.embeddings.similarity import cosine_similarity
from rag_lab.types import Chunk


class FakeModel:
    """Maps a text to a 3-number vector from its first letter, so results are predictable."""

    def get_embedding_dimension(self):
        return 3

    def encode(self, texts, batch_size, normalize_embeddings, convert_to_numpy, show_progress_bar):
        table = {"a": [1.0, 0.0, 0.0], "b": [0.0, 1.0, 0.0], "c": [0.0, 0.0, 1.0]}
        out = np.array([table[t[0]] for t in texts])
        return out / np.linalg.norm(out, axis=1, keepdims=True) if normalize_embeddings else out


def chunk(i: int, text: str) -> Chunk:
    return Chunk(chunk_id=f"d::{i}", doc_id="d", text=text, start_char=0, end_char=len(text), metadata={})


def embedder() -> Embedder:
    return Embedder("fake", loader=lambda name: FakeModel())


# ------------------------------------------------ embed_chunks
def test_one_row_per_chunk_in_the_same_order():
    chunks = [chunk(0, "alpha"), chunk(1, "beta"), chunk(2, "charlie")]
    matrix = embed_chunks(embedder(), chunks)
    assert matrix.shape == (3, 3)
    assert np.allclose(matrix, np.eye(3))                # row i is chunk i


def test_no_chunks_gives_empty_matrix():
    assert embed_chunks(embedder(), []).shape == (0, 3)


# ------------------------------------------------ nearest
def test_nearest_returns_best_first():
    matrix = np.eye(3, dtype=np.float32)
    query = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert [i for i, _ in nearest(query, matrix, k=3)][0] == 1


def test_scores_are_descending():
    rng = np.random.default_rng(2)
    matrix = rng.normal(size=(20, 8))
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    query = matrix[5]
    scores = [s for _, s in nearest(query, matrix, k=20)]
    assert scores == sorted(scores, reverse=True)
    assert nearest(query, matrix, k=1)[0][0] == 5        # a vector is nearest to itself


def test_k_larger_than_rows_returns_all_rows():
    matrix = np.eye(3, dtype=np.float32)
    assert len(nearest(matrix[0], matrix, k=10)) == 3


def test_scores_equal_the_hand_written_cosine_similarity():
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(10, 384))
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    query = rng.normal(size=384)
    query /= np.linalg.norm(query)
    for i, score in nearest(query, matrix, k=10):
        assert math.isclose(score, cosine_similarity(query, matrix[i]), abs_tol=1e-9)


def test_ties_keep_original_order():
    matrix = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert [i for i, _ in nearest(np.array([1.0, 0.0]), matrix, k=3)][:2] == [0, 1]


@pytest.mark.parametrize("bad_k", [0, -1])
def test_bad_k_rejected(bad_k):
    with pytest.raises(ValueError, match="k must"):
        nearest(np.array([1.0, 0.0]), np.eye(2), k=bad_k)


def test_dimension_mismatch_rejected():
    with pytest.raises(ValueError, match="dimension mismatch"):
        nearest(np.array([1.0, 0.0, 0.0]), np.eye(2), k=1)


def test_wrong_shapes_rejected():
    with pytest.raises(ValueError, match="2-D matrix"):
        nearest(np.eye(2), np.eye(2), k=1)


# ------------------------------------------------ real model
@pytest.mark.integration
def test_real_model_finds_the_moon_chunk_first():
    chunks = [chunk(0, "Vitamin C is found in citrus fruit."),
              chunk(1, "Apollo 11 landed on the Moon in 1969."),
              chunk(2, "A B-tree index speeds up database lookups.")]
    e = Embedder()
    matrix = embed_chunks(e, chunks)
    best, _ = nearest(e.embed_text("When did humans first walk on the Moon?"), matrix, k=1)[0]
    assert best == 1
