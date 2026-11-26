import math

import pytest

from rag_lab.chunking import add_overlap, chunk_recursive
from rag_lab.config import Settings
from rag_lab.ingestion import clean_corpus, load_corpus
from rag_lab.retrieval.stats import document_frequencies, idf, idf_table, term_frequencies
from rag_lab.retrieval.tokenize import tokenize

SETTINGS = Settings.from_env({})


# ------------------------------------------------ term frequency
def test_term_frequencies_counts_repeats():
    assert term_frequencies(["a", "b", "a", "a"]) == {"a": 3, "b": 1}


def test_term_frequencies_of_nothing_is_empty():
    assert term_frequencies([]) == {}


# ------------------------------------------------ document frequency
def test_document_frequency_counts_chunks_not_occurrences():
    corpus = [["moon", "moon", "moon"], ["moon", "sun"], ["sun"]]
    assert document_frequencies(corpus) == {"moon": 2, "sun": 2}


def test_document_frequency_of_empty_corpus_is_empty():
    assert document_frequencies([]) == {}


# ------------------------------------------------ idf
def test_idf_known_values():
    assert math.isclose(idf(43, 1), math.log(1 + 42.5 / 1.5))       # about 3.379
    assert math.isclose(idf(43, 43), math.log(1 + 0.5 / 43.5))      # about 0.0114


def test_idf_falls_as_the_word_gets_more_common():
    values = [idf(100, n) for n in range(0, 101)]
    assert values == sorted(values, reverse=True)
    assert len(set(values)) == len(values)                          # strictly decreasing


def test_idf_is_never_negative_even_for_a_word_in_every_chunk():
    assert all(idf(10, n) > 0 for n in range(0, 11))


def test_unseen_word_is_allowed_and_scores_highest():
    assert idf(10, 0) > idf(10, 1)


@pytest.mark.parametrize("n_docs,doc_freq", [(0, 0), (-1, 0), (5, -1), (5, 6)])
def test_idf_rejects_impossible_counts(n_docs, doc_freq):
    with pytest.raises(ValueError):
        idf(n_docs, doc_freq)


def test_idf_table_matches_idf():
    corpus = [["a", "b"], ["a"], ["a", "c"]]
    table = idf_table(corpus)
    assert table["a"] == idf(3, 3)
    assert table["b"] == idf(3, 1)
    assert table["b"] > table["a"]


# ------------------------------------------------ real corpus
def real_chunk_tokens():
    docs = [c.retrieval_document for c in clean_corpus(load_corpus(SETTINGS.sample_docs_dir))]
    chunks = [c for d in docs for c in add_overlap(d, chunk_recursive(d, 500), 50)]
    return [tokenize(c.text) for c in chunks]


def test_rare_words_score_higher_than_common_words_on_the_real_corpus():
    tokens = real_chunk_tokens()
    table = idf_table(tokens)
    df = document_frequencies(tokens)
    assert df["the"] > 0.7 * len(tokens)                            # "the" is in most chunks
    assert table["melatonin"] > 5 * table["the"]
    assert all(v > 0 for v in table.values())
