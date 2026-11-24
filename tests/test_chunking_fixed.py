import pytest

from rag_lab.chunking import build_chunks, chunk_fixed_size, fixed_size_spans, make_chunk
from rag_lab.config import Settings
from rag_lab.ingestion import clean_corpus, load_corpus
from rag_lab.types import Document, verify_chunk_integrity

CLEANED = {c.doc_id: c.retrieval_document for c in clean_corpus(load_corpus(Settings.from_env({}).sample_docs_dir))}


def doc(text: str, **meta) -> Document:
    return Document("d", text, meta)


# ------------------------------------------------ spans
def test_spans_cover_everything_without_gaps_or_overlap():
    assert fixed_size_spans(10, 4) == [(0, 4), (4, 8), (8, 10)]


def test_exact_multiple_has_no_empty_last_span():
    assert fixed_size_spans(8, 4) == [(0, 4), (4, 8)]


def test_text_shorter_than_chunk_size_is_one_chunk():
    assert fixed_size_spans(3, 100) == [(0, 3)]


def test_empty_text_has_no_spans():
    assert fixed_size_spans(0, 5) == []


@pytest.mark.parametrize("bad", [0, -1])
def test_bad_chunk_size_rejected(bad):
    with pytest.raises(ValueError, match="chunk_size"):
        fixed_size_spans(10, bad)


# ------------------------------------------------ chunks
def test_chunk_text_ids_and_offsets():
    chunks = chunk_fixed_size(doc("abcdefghij"), chunk_size=4)
    assert [c.text for c in chunks] == ["abcd", "efgh", "ij"]
    assert [(c.start_char, c.end_char) for c in chunks] == [(0, 4), (4, 8), (8, 10)]
    assert [c.chunk_id for c in chunks] == ["d::0", "d::1", "d::2"]


def test_it_really_cuts_words_in_half():
    # This is the known weakness that motivates the other strategies.
    chunks = chunk_fixed_size(doc("Jezero Crater is big"), chunk_size=10)
    assert chunks[0].text == "Jezero Cra" and chunks[1].text == "ter is big"


def test_metadata_is_inherited_and_extended():
    c = chunk_fixed_size(doc("abcdef", topic="space", year=2023), chunk_size=3)[1]
    assert c.metadata["topic"] == "space" and c.metadata["year"] == 2023
    assert c.metadata["chunk_index"] == 1 and c.metadata["chunk_strategy"] == "fixed"


def test_whitespace_only_chunks_are_skipped_and_indexes_stay_consecutive():
    chunks = chunk_fixed_size(doc("ab      cd"), chunk_size=2)
    assert [c.text for c in chunks] == ["ab", "cd"]
    assert [c.chunk_id for c in chunks] == ["d::0", "d::1"]
    assert (chunks[1].start_char, chunks[1].end_char) == (8, 10)      # true offsets kept


def test_chunking_is_deterministic():
    d = CLEANED["space_mars_exploration"]
    assert chunk_fixed_size(d, 300) == chunk_fixed_size(d, 300)


# ------------------------------------------------ make_chunk / build_chunks
@pytest.mark.parametrize("start,end", [(-1, 3), (3, 3), (4, 3), (0, 99)])
def test_make_chunk_rejects_bad_spans(start, end):
    with pytest.raises(ValueError, match="outside"):
        make_chunk(doc("abcdef"), 0, start, end, "x")


def test_integrity_check_catches_a_chunk_that_does_not_match_its_document():
    d = doc("abcdef")
    chunk = chunk_fixed_size(d, 3)[0]
    with pytest.raises(AssertionError, match="text mismatch"):
        verify_chunk_integrity(Document("d", "XYZdef"), [chunk])


def test_build_chunks_runs_the_integrity_check():
    assert build_chunks(doc("abcdef"), [(0, 3), (3, 6)], "x")[1].text == "def"


# ------------------------------------------------ real documents
@pytest.mark.parametrize("doc_id", sorted(CLEANED))
@pytest.mark.parametrize("size", [200, 500, 1000])
def test_real_documents_chunks_match_source_and_rebuild_it(doc_id, size):
    d = CLEANED[doc_id]
    chunks = chunk_fixed_size(d, size)
    verify_chunk_integrity(d, chunks)
    assert all(len(c) <= size for c in chunks)
    # No overlap and (cleaned text has no whitespace-only stretches) no gaps:
    assert "".join(c.text for c in chunks) == d.text
    assert all(a.end_char == b.start_char for a, b in zip(chunks, chunks[1:]))
