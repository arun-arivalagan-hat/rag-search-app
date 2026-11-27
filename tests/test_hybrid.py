import numpy as np
import pytest

from rag_lab.retrieval.hybrid import HybridIndex
from rag_lab.types import Chunk

MOON_WORDS = {"moon", "lunar", "selene"}          # the fake model treats these as one idea


class FakeEmbedder:
    """Offline stand-in: 3 topic counts (moon-ish, vitamin, sun) plus a tiny constant, normalised.
    "selene" and "lunar" count as moon, so a query using them is a SYNONYM of the chunk text."""

    def _vector(self, text):
        words = text.casefold().split()
        v = np.array(
            [sum(w in MOON_WORDS for w in words), words.count("vitamin"), words.count("sun"), 0.1],
            dtype=np.float32,
        )
        return v / np.linalg.norm(v)

    def embed_texts(self, texts, normalize=True):
        return np.array([self._vector(t) for t in texts], dtype=np.float32)

    def embed_text(self, text, normalize=True):
        return self._vector(text)


TEXTS = [
    "the moon landing in 1969",
    "vitamin c and health",
    "sun and moon and stars",
    "apollo astronauts walked on the lunar surface",
]


def make_chunks(texts=TEXTS):
    return [Chunk(chunk_id=f"c{i}", doc_id="d", text=t, start_char=0, end_char=len(t)) for i, t in enumerate(texts)]


@pytest.fixture
def index():
    return HybridIndex(make_chunks(), FakeEmbedder())


def test_exact_word_query_ranks_the_exact_match_first(index):
    hits = index.search("vitamin")
    assert hits[0].chunk.chunk_id == "c1"
    assert hits[0].keyword_rank == 1 and hits[0].semantic_rank == 1


def test_synonym_query_is_rescued_by_semantic_search(index):
    # "selene" appears in no chunk, so keyword search returns nothing, yet meaning matches the moon chunks
    hits = index.search("selene", k=3)
    assert {h.chunk.chunk_id for h in hits} == {"c0", "c2", "c3"}
    assert all(h.keyword_rank is None for h in hits)
    assert all(h.semantic_rank is not None for h in hits)


def test_chunk_found_by_both_searches_beats_chunk_found_by_one():
    chunks = make_chunks(["moon moon moon", "moon", "sun sun sun"])
    hits = HybridIndex(chunks, FakeEmbedder()).search("moon", k=3)
    assert hits[0].keyword_rank is not None and hits[0].semantic_rank is not None
    assert hits[0].score > hits[-1].score


def test_results_are_ranked_from_one_and_sorted(index):
    hits = index.search("moon", k=4)
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_k_limits_the_number_of_results(index):
    assert len(index.search("moon", k=2)) == 2


def test_k_larger_than_corpus_returns_at_most_the_corpus(index):
    assert len(index.search("moon", k=50)) <= len(TEXTS)


def test_candidates_smaller_than_k_is_raised_to_k(index):
    assert len(index.search("moon", k=3, candidates=1)) == 3


def test_search_is_repeatable(index):
    first = [(h.chunk.chunk_id, h.score) for h in index.search("moon")]
    assert first == [(h.chunk.chunk_id, h.score) for h in index.search("moon")]


def test_duplicate_chunk_ids_rejected():
    chunks = make_chunks()
    chunks[1] = Chunk(chunk_id="c0", doc_id="d", text="x", start_char=0, end_char=1)
    with pytest.raises(ValueError, match="unique"):
        HybridIndex(chunks, FakeEmbedder())


def test_empty_chunk_list_rejected():
    with pytest.raises(ValueError, match="at least one chunk"):
        HybridIndex([], FakeEmbedder())


def test_bad_query_and_k_rejected(index):
    with pytest.raises(TypeError):
        index.search(None)
    with pytest.raises(ValueError):
        index.search("moon", k=0)
