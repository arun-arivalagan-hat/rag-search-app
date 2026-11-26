import pytest

from rag_lab.chunking import add_overlap, chunk_recursive
from rag_lab.config import Settings
from rag_lab.ingestion import clean_corpus, load_corpus
from rag_lab.retrieval.search import KeywordIndex, top_k
from rag_lab.types import Chunk

SETTINGS = Settings.from_env({})


# ------------------------------------------------ top_k
def test_top_k_orders_best_first():
    assert top_k([0.5, 2.0, 1.0], k=3) == [(1, 2.0), (2, 1.0), (0, 0.5)]


def test_top_k_limits_to_k():
    assert top_k([3, 2, 1], k=2) == [(0, 3), (1, 2)]


def test_top_k_larger_than_corpus_returns_everything_that_matched():
    assert len(top_k([1, 2], k=10)) == 2


def test_top_k_drops_zero_scores():
    assert top_k([0, 2.0, 0, 1.0], k=4) == [(1, 2.0), (3, 1.0)]


def test_top_k_nothing_matches_gives_empty_list():
    assert top_k([0, 0, 0]) == []


def test_top_k_ties_keep_original_order():
    assert top_k([1.0, 1.0, 1.0], k=3) == [(0, 1.0), (1, 1.0), (2, 1.0)]


def test_top_k_empty_scores():
    assert top_k([]) == []


@pytest.mark.parametrize("k", [0, -1])
def test_top_k_rejects_bad_k(k):
    with pytest.raises(ValueError):
        top_k([1], k=k)


# ------------------------------------------------ KeywordIndex on a tiny corpus
def make_chunks(texts):
    return [Chunk(chunk_id=f"c{i}", doc_id="d", text=t, start_char=0, end_char=len(t)) for i, t in enumerate(texts)]


TEXTS = ["the moon landing in 1969", "vitamin c and health", "sun and moon and stars"]


def test_search_returns_best_chunk_first_with_ranks():
    hits = KeywordIndex(make_chunks(TEXTS)).search("moon landing")
    assert [h.rank for h in hits] == [1, 2]
    assert hits[0].chunk.text == TEXTS[0]
    assert hits[0].score > hits[1].score


def test_search_only_returns_chunks_that_matched():
    hits = KeywordIndex(make_chunks(TEXTS)).search("vitamin")
    assert [h.chunk.text for h in hits] == [TEXTS[1]]


def test_search_with_no_match_is_empty():
    assert KeywordIndex(make_chunks(TEXTS)).search("zebra") == []


def test_search_with_empty_query_is_empty():
    assert KeywordIndex(make_chunks(TEXTS)).search("") == []


def test_search_respects_k():
    assert len(KeywordIndex(make_chunks(TEXTS)).search("moon", k=1)) == 1


def test_search_is_case_insensitive():
    index = KeywordIndex(make_chunks(TEXTS))
    assert [h.chunk.chunk_id for h in index.search("MOON")] == [h.chunk.chunk_id for h in index.search("moon")]


def test_search_rejects_non_string_query():
    with pytest.raises(TypeError):
        KeywordIndex(make_chunks(TEXTS)).search(None)


def test_empty_chunk_list_rejected():
    with pytest.raises(ValueError, match="at least one chunk"):
        KeywordIndex([])


# ------------------------------------------------ real corpus
@pytest.fixture(scope="module")
def real_index():
    docs = [c.retrieval_document for c in clean_corpus(load_corpus(SETTINGS.sample_docs_dir))]
    chunks = [c for d in docs for c in add_overlap(d, chunk_recursive(d, 500), 50)]
    return KeywordIndex(chunks)


def test_real_queries_find_the_right_document(real_index):
    cases = {
        "suprachiasmatic nucleus melatonin": "health_sleep_science",
        "three-way handshake SYN ACK": "computing_tcp_ip_networking",
        "Perseverance Jezero Crater river delta": "space_mars_exploration",
        "Neil Armstrong Sea of Tranquility": "space_apollo_program",
    }
    for query, doc_id in cases.items():
        assert real_index.search(query)[0].chunk.doc_id == doc_id, query


def test_real_results_are_sorted_and_ranked(real_index):
    hits = real_index.search("moon landing", k=5)
    assert len(hits) == 5
    assert [h.rank for h in hits] == [1, 2, 3, 4, 5]
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
