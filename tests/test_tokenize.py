import pytest

from rag_lab.config import Settings
from rag_lab.evaluation.qrels import load_qrels
from rag_lab.ingestion import clean_corpus, load_corpus
from rag_lab.retrieval.tokenize import tokenize

SETTINGS = Settings.from_env({})


def test_lowercases_and_splits_on_spaces():
    assert tokenize("The Moon Landing") == ["the", "moon", "landing"]


def test_punctuation_is_dropped():
    assert tokenize("Hello, world! (Really?)") == ["hello", "world", "really"]


def test_repeats_are_kept_and_order_is_preserved():
    assert tokenize("to be or not to be") == ["to", "be", "or", "not", "to", "be"]


def test_hyphenated_words_are_split():
    assert tokenize("three-way handshake") == ["three", "way", "handshake"]


def test_underscore_separates_tokens():
    assert tokenize("snake_case") == ["snake", "case"]


def test_decimal_numbers_stay_whole():
    assert tokenize("It is 7.6 million km.") == ["it", "is", "7.6", "million", "km"]


def test_trailing_dot_is_not_part_of_the_token():
    assert tokenize("It ended. Then") == ["it", "ended", "then"]


@pytest.mark.parametrize("text", ["don't stop", "don" + chr(0x2019) + "t stop"])
def test_apostrophe_inside_a_word_stays(text):
    assert tokenize(text) == [text.split()[0].casefold(), "stop"]


def test_leading_and_trailing_apostrophes_are_dropped():
    assert tokenize("'quoted'") == ["quoted"]


def test_abbreviation_keeps_inner_dots_only():
    assert tokenize("the U.S. economy") == ["the", "u.s", "economy"]


def test_digits_and_letters_mix():
    assert tokenize("Apollo 11 launched in 1969") == ["apollo", "11", "launched", "in", "1969"]


def test_accented_letters_are_kept():
    assert tokenize("Caf" + chr(0xE9) + " au lait") == ["caf" + chr(0xE9), "au", "lait"]


@pytest.mark.parametrize("text", ["", "   ", "!!! ... ---", "\n\n"])
def test_nothing_to_tokenize_gives_empty_list(text):
    assert tokenize(text) == []


def test_query_from_the_eval_set():
    assert tokenize("three-way handshake SYN ACK") == ["three", "way", "handshake", "syn", "ack"]


def test_real_documents_and_queries_tokenize_cleanly():
    docs = [c.retrieval_document for c in clean_corpus(load_corpus(SETTINGS.sample_docs_dir))]
    for doc in docs:
        tokens = tokenize(doc.text)
        assert tokens, doc.doc_id
        assert all(t == t.casefold() and t.strip() == t and t for t in tokens)
    for q in load_qrels(SETTINGS.eval_dir / "qrels.json"):
        assert tokenize(q.query), q.query_id
