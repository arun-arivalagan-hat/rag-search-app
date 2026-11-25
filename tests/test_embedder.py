import numpy as np
import pytest

from rag_lab.embeddings.embedder import Embedder
from rag_lab.embeddings.similarity import cosine_similarity


class FakeModel:
    """Stands in for SentenceTransformer: 4 numbers per text, no download."""

    def __init__(self):
        self.encode_calls = []

    def get_embedding_dimension(self):
        return 4

    def encode(self, texts, batch_size, normalize_embeddings, convert_to_numpy, show_progress_bar):
        self.encode_calls.append((list(texts), normalize_embeddings))
        out = np.array([[len(t), 1.0, 2.0, 3.0] for t in texts], dtype=np.float64)
        if normalize_embeddings:
            out = out / np.linalg.norm(out, axis=1, keepdims=True)
        return out


def make_embedder():
    loads = []
    model = FakeModel()

    def loader(name):
        loads.append(name)
        return model

    return Embedder("fake-model", loader=loader), loads, model


def test_model_is_not_loaded_until_first_use():
    _, loads, _ = make_embedder()
    assert loads == []


def test_model_is_loaded_once_and_with_the_configured_name():
    e, loads, _ = make_embedder()
    e.embed_texts(["a"])
    e.embed_texts(["b"])
    assert loads == ["fake-model"]


def test_shape_and_dtype():
    e, _, _ = make_embedder()
    out = e.embed_texts(["a", "bb", "ccc"])
    assert out.shape == (3, 4)
    assert out.dtype == np.float32


def test_normalized_rows_have_length_one():
    e, _, _ = make_embedder()
    assert np.allclose(np.linalg.norm(e.embed_texts(["a", "bb"]), axis=1), 1.0, atol=1e-6)


def test_normalize_false_is_passed_to_the_model():
    e, _, model = make_embedder()
    e.embed_texts(["abc"], normalize=False)
    assert model.encode_calls[-1][1] is False


def test_empty_list_gives_empty_array_with_right_width():
    e, _, model = make_embedder()
    out = e.embed_texts([])
    assert out.shape == (0, 4)
    assert model.encode_calls == []


def test_single_string_rejected():
    e, _, _ = make_embedder()
    with pytest.raises(TypeError, match="list of strings"):
        e.embed_texts("hello")


def test_non_string_item_rejected():
    e, _, _ = make_embedder()
    with pytest.raises(TypeError, match="must be a string"):
        e.embed_texts(["ok", 5])


def test_embed_text_returns_one_vector():
    e, _, _ = make_embedder()
    assert e.embed_text("hello").shape == (4,)


def test_dimension_works_with_the_old_method_name_too():
    class OldModel:
        def get_sentence_embedding_dimension(self):
            return 7

    assert Embedder("old", loader=lambda name: OldModel()).dimension == 7


def test_bad_batch_size_rejected():
    with pytest.raises(ValueError, match="batch_size"):
        Embedder(batch_size=0)


@pytest.mark.integration
def test_real_model_meaning_and_shape():
    e = Embedder()
    v = e.embed_texts(["Apollo 11 landed on the Moon", "Humans first reached the Moon", "Vitamin C sources"])
    assert v.shape == (3, 384)
    assert cosine_similarity(v[0], v[1]) > cosine_similarity(v[0], v[2])
