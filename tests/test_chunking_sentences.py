import re

import pytest

from rag_lab.chunking import chunk_fixed_size, chunk_sentences, sentence_spans
from rag_lab.config import Settings
from rag_lab.ingestion import clean_corpus, load_corpus
from rag_lab.types import Document, verify_chunk_integrity

CLEANED = {c.doc_id: c.retrieval_document for c in clean_corpus(load_corpus(Settings.from_env({}).sample_docs_dir))}


def pieces(text: str) -> list[str]:
    return [text[a:b].strip() for a, b in sentence_spans(text)]


def doc(text: str) -> Document:
    return Document("d", text)


# ------------------------------------------------ boundary rules (small made-up strings)
@pytest.mark.parametrize("text,expected", [
    ("Hi there. Bye now.", ["Hi there.", "Bye now."]),
    ("Really? Yes! Fine.", ["Really?", "Yes!", "Fine."]),
    ("It is 7.6 million km. Next.", ["It is 7.6 million km.", "Next."]),       # decimal, no space after "."
    ("Mail a@example.com now. Then wait.", ["Mail a@example.com now.", "Then wait."]),
    ("Ask Dr. Smith. He knows.", ["Ask Dr. Smith.", "He knows."]),             # abbreviation list
    ("Use e.g. this one. Done.", ["Use e.g. this one.", "Done."]),             # lower-case after the stop
    ("Salt, pepper, etc. and oil. Mix.", ["Salt, pepper, etc. and oil.", "Mix."]),
    ("J. R. Tolkien wrote it. Yes.", ["J. R. Tolkien wrote it.", "Yes."]),    # initials
    ("Wait... what? Yes.", ["Wait... what?", "Yes."]),                          # ellipsis then lower case
    ('He said "stop." Then left.', ['He said "stop."', "Then left."]),          # closing quote after the stop
    ("It grew (a lot). Then fell.", ["It grew (a lot).", "Then fell."]),
    ("Chapter 1. 2 men left.", ["Chapter 1.", "2 men left."]),                  # digit may start a sentence
])
def test_sentence_boundaries(text, expected):
    assert pieces(text) == expected


def test_paragraph_break_ends_a_sentence_even_without_a_full_stop():
    assert pieces("A heading\n\nBody text here.") == ["A heading", "Body text here."]


def test_single_newline_alone_does_not_end_a_sentence():
    assert pieces("one line\ncontinues here.") == ["one line\ncontinues here."]


def test_text_without_any_ending_is_one_sentence():
    assert pieces("no full stop at all") == ["no full stop at all"]


def test_empty_text_has_no_spans():
    assert sentence_spans("") == []


@pytest.mark.parametrize("text", [
    "Hi there. Bye now.", "  leading and trailing  ", "A\n\nB\n\n\nC.", "Dr. Who? Yes.   Fine.  ",
])
def test_spans_cover_every_character_without_gaps(text):
    spans = sentence_spans(text)
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))


# ------------------------------------------------ grouping into chunks
def test_sentences_are_packed_up_to_the_limit():
    text = "Aaaa aaaa. Bbbb bbbb. Cccc cccc. Dddd dddd."          # each sentence = 10 chars
    chunks = chunk_sentences(doc(text), max_chars=22)
    assert [c.text for c in chunks] == ["Aaaa aaaa. Bbbb bbbb.", "Cccc cccc. Dddd dddd."]
    assert [c.metadata["sentence_count"] for c in chunks] == [2, 2]


def test_chunk_text_has_no_leading_or_trailing_whitespace():
    for c in chunk_sentences(doc("One.\n\nTwo is here.\n\nThree."), max_chars=12):
        assert c.text == c.text.strip()


def test_a_sentence_longer_than_the_limit_stays_whole_and_is_flagged():
    chunks = chunk_sentences(doc("Short one. " + "Word " * 20 + "end. Last."), max_chars=30)
    assert [c.metadata["oversized"] for c in chunks] == [False, True, False]
    assert len(chunks[1]) > 30 and chunks[1].text.endswith("end.")


def test_offsets_match_the_source():
    d = doc("  First one. Second one.\n\nThird one.  ")
    chunks = chunk_sentences(d, max_chars=15)
    verify_chunk_integrity(d, chunks)
    assert [c.text for c in chunks] == ["First one.", "Second one.", "Third one."]
    assert [(c.start_char, c.end_char) for c in chunks] == [(2, 12), (13, 24), (26, 36)]


def test_blank_document_gives_no_chunks():
    assert chunk_sentences(doc("  \n\n  ")) == []


def test_bad_limit_rejected():
    with pytest.raises(ValueError, match="max_chars"):
        chunk_sentences(doc("x"), max_chars=0)


def test_metadata_inherited():
    d = Document("d", "One. Two.", {"topic": "space"})
    assert chunk_sentences(d, 100)[0].metadata["topic"] == "space"
    assert chunk_sentences(d, 100)[0].metadata["chunk_strategy"] == "sentence"


# ------------------------------------------------ real documents
@pytest.mark.parametrize("doc_id", sorted(CLEANED))
def test_real_documents_integrity_and_nothing_lost(doc_id):
    d = CLEANED[doc_id]
    chunks = chunk_sentences(d, 500)
    verify_chunk_integrity(d, chunks)
    rebuilt = "".join(c.text for c in chunks)
    assert re.sub(r"\s+", "", rebuilt) == re.sub(r"\s+", "", d.text)     # same characters, only gaps differ
    assert all(len(c) <= 500 or c.metadata["oversized"] for c in chunks)


@pytest.mark.parametrize("doc_id", sorted(CLEANED))
def test_real_documents_every_chunk_starts_on_a_sentence_and_ends_on_punctuation(doc_id):
    d = CLEANED[doc_id]
    sentence_starts = set()
    for a, b in sentence_spans(d.text):
        sentence_starts.add(a + len(d.text[a:b]) - len(d.text[a:b].lstrip()))
    for c in chunk_sentences(d, 500):
        assert c.start_char in sentence_starts
        assert c.text[-1] in ".!?\"')]", c.text[-40:]          # never stops after a heading or mid-sentence


def test_a_heading_is_not_left_at_the_end_of_a_chunk():
    text = "Intro one. Intro two.\n\nTrade-offs\n\nBody is here."
    chunks = chunk_sentences(doc(text), max_chars=32)
    assert [c.text for c in chunks] == ["Intro one. Intro two.", "Trade-offs\n\nBody is here."]


def test_a_heading_alone_in_its_chunk_is_allowed():
    chunks = chunk_sentences(doc("Heading\n\nBody text."), max_chars=7)
    assert chunks[0].text == "Heading"            # nothing to move it away from


def test_real_decimals_are_not_split():
    d = CLEANED["space_james_webb_telescope"]
    assert any("6.5 metres" in c.text for c in chunk_sentences(d, 200))
    assert not any(c.text.endswith("6.") for c in chunk_sentences(d, 200))


def test_real_documents_no_chunk_ends_mid_word_unlike_fixed_size():
    mid_word = lambda chunks: sum(1 for a, b in zip(chunks, chunks[1:]) if a.end_char == b.start_char and a.text[-1].isalnum() and b.text[0].isalnum())
    sentence_cuts = sum(mid_word(chunk_sentences(d, 500)) for d in CLEANED.values())
    fixed_cuts = sum(mid_word(chunk_fixed_size(d, 500)) for d in CLEANED.values())
    assert sentence_cuts == 0 and fixed_cuts == 14          # 14 mid-word cuts out of 21 boundaries (step 3a)


def test_real_documents_are_deterministic():
    d = CLEANED["space_black_holes"]
    assert chunk_sentences(d, 300) == chunk_sentences(d, 300)
