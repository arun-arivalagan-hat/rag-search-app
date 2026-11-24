import pytest

from rag_lab.chunking import chunk_recursive, recursive_spans
from rag_lab.config import Settings
from rag_lab.ingestion import clean_corpus, load_corpus
from rag_lab.types import Document, verify_chunk_integrity

CLEANED = {c.doc_id: c.retrieval_document for c in clean_corpus(load_corpus(Settings.from_env({}).sample_docs_dir))}


def doc(text: str) -> Document:
    return Document("d", text)


def test_short_text_is_one_span():
    assert recursive_spans("hello world", 50) == [(0, 11)]


def test_empty_text_has_no_spans():
    assert recursive_spans("", 50) == []


@pytest.mark.parametrize("bad", [0, -3])
def test_bad_max_chars_rejected(bad):
    with pytest.raises(ValueError, match="max_chars"):
        recursive_spans("abc", bad)


def test_paragraphs_kept_whole_when_they_fit():
    text = "First paragraph here.\n\nSecond paragraph here."
    chunks = chunk_recursive(doc(text), max_chars=25)
    assert [c.text for c in chunks] == ["First paragraph here.", "Second paragraph here."]


def test_small_paragraphs_are_packed_together():
    text = "aaa\n\nbbb\n\nccc"
    assert len(chunk_recursive(doc(text), max_chars=100)) == 1
    assert len(chunk_recursive(doc(text), max_chars=8)) == 2


def test_long_paragraph_falls_back_to_sentences():
    text = "One two three. Four five six. Seven eight nine."
    chunks = chunk_recursive(doc(text), max_chars=30)
    assert [c.text for c in chunks] == ["One two three. Four five six.", "Seven eight nine."]


def test_long_sentence_falls_back_to_words():
    chunks = chunk_recursive(doc("alpha beta gamma delta epsilon"), max_chars=12)
    assert all(len(c.text) <= 12 for c in chunks)
    assert " ".join(c.text for c in chunks) == "alpha beta gamma delta epsilon"


def test_word_longer_than_limit_falls_back_to_characters():
    chunks = chunk_recursive(doc("x" * 25), max_chars=10)
    assert [len(c.text) for c in chunks] == [10, 10, 5]


def test_strategy_and_ids():
    chunks = chunk_recursive(doc("aaa\n\nbbb"), max_chars=4)
    assert len(chunks) == 2
    assert [c.metadata["chunk_index"] for c in chunks] == [0, 1]
    assert all(c.metadata["chunk_strategy"] == "recursive" for c in chunks)


@pytest.mark.parametrize("max_chars", [40, 120, 500])
@pytest.mark.parametrize("doc_id", sorted(CLEANED))
def test_real_documents_respect_limit_and_offsets(doc_id, max_chars):
    document = CLEANED[doc_id]
    chunks = chunk_recursive(document, max_chars)
    assert chunks
    assert all(len(c.text) <= max_chars for c in chunks)
    verify_chunk_integrity(document, chunks)
