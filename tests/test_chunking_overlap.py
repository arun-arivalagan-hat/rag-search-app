import pytest

from rag_lab.chunking import add_overlap, chunk_fixed_size, chunk_recursive, chunk_sentences
from rag_lab.config import Settings
from rag_lab.ingestion import clean_corpus, load_corpus
from rag_lab.types import Document, verify_chunk_integrity

CLEANED = {c.doc_id: c.retrieval_document for c in clean_corpus(load_corpus(Settings.from_env({}).sample_docs_dir))}


def doc(text: str) -> Document:
    return Document("d", text)


def test_zero_overlap_changes_nothing():
    d = doc("one two three four five six seven")
    base = chunk_recursive(d, 14)
    assert [(c.start_char, c.end_char) for c in add_overlap(d, base, 0)] == [(c.start_char, c.end_char) for c in base]


def test_negative_overlap_rejected():
    d = doc("abc")
    with pytest.raises(ValueError, match="overlap"):
        add_overlap(d, chunk_recursive(d, 5), -1)


def test_next_chunk_repeats_end_of_previous_on_a_word_boundary():
    d = doc("alpha beta gamma delta epsilon zeta")
    chunks = add_overlap(d, chunk_recursive(d, 17), 8)
    assert [c.text for c in chunks] == ["alpha beta gamma", "gamma delta epsilon", "epsilon zeta"]
    assert [c.metadata["overlap_chars"] for c in chunks] == [0, 6, 8]


def test_first_chunk_never_overlaps_and_ends_are_unchanged():
    d = doc("alpha beta gamma delta epsilon zeta")
    base = chunk_recursive(d, 17)
    chunks = add_overlap(d, base, 8)
    assert chunks[0].metadata["overlap_chars"] == 0
    assert [c.end_char for c in chunks] == [c.end_char for c in base]


def test_overlap_never_cuts_a_word_in_half():
    d = doc("alpha beta gamma delta epsilon zeta")
    for c in add_overlap(d, chunk_recursive(d, 17), 9):
        assert c.start_char == 0 or d.text[c.start_char - 1].isspace()


def test_overlap_larger_than_previous_chunk_stays_inside_it():
    d = doc("aaaa bbbb cccc dddd")
    base = chunk_recursive(d, 5)
    for prev, cur in zip(base, add_overlap(d, base, 1000)[1:]):
        assert cur.start_char >= prev.start_char


@pytest.mark.parametrize("chunker", [chunk_fixed_size, chunk_sentences, chunk_recursive])
@pytest.mark.parametrize("doc_id", sorted(CLEANED))
def test_works_on_every_chunker_with_exact_offsets(doc_id, chunker):
    document = CLEANED[doc_id]
    base = chunker(document, 200)
    chunks = add_overlap(document, base, 40)
    verify_chunk_integrity(document, chunks)
    assert len(chunks) == len(base)
    assert all(c.metadata["overlap_chars"] <= 40 for c in chunks)
    assert all(c.metadata["chunk_strategy"] == b.metadata["chunk_strategy"] for c, b in zip(chunks, base))
