import dataclasses

import pytest

from rag_lab.types import Chunk, Document, RetrievalResult, verify_chunk_integrity

TEXT = "The quick brown fox jumps over the lazy dog."


def make_doc():
    return Document("d1", TEXT, {"topic": "demo"})


def test_chunk_make_id():
    assert Chunk.make_id("d1", 3) == "d1::3"


def test_valid_chunk_passes_integrity():
    doc = make_doc()
    chunk = Chunk("d1::0", "d1", TEXT[4:9], 4, 9, doc.metadata)
    verify_chunk_integrity(doc, [chunk])  # no exception


def test_integrity_detects_wrong_text():
    doc = make_doc()
    bad = Chunk("d1::0", "d1", "XXXXX", 4, 9)
    with pytest.raises(AssertionError, match="text mismatch"):
        verify_chunk_integrity(doc, [bad])


def test_integrity_detects_offset_beyond_document():
    doc = make_doc()
    bad = Chunk("d1::0", "d1", "dog.", len(TEXT) - 2, len(TEXT) + 2)
    with pytest.raises(AssertionError, match="beyond document"):
        verify_chunk_integrity(doc, [bad])


def test_integrity_detects_wrong_document():
    doc = make_doc()
    other = Chunk("x::0", "other", TEXT[:3], 0, 3)
    with pytest.raises(AssertionError, match="wrong doc_id"):
        verify_chunk_integrity(doc, [other])


@pytest.mark.parametrize("start,end", [(-1, 3), (5, 5), (7, 3)])
def test_chunk_rejects_bad_offsets(start, end):
    with pytest.raises(ValueError, match="offsets"):
        Chunk("c", "d", "abc", start, end)


def test_chunk_rejects_empty_text():
    with pytest.raises(ValueError, match="non-empty"):
        Chunk("c", "d", "", 0, 1)


def test_types_are_immutable():
    chunk = Chunk("c", "d", "abc", 0, 3, {"k": "v"})
    with pytest.raises(dataclasses.FrozenInstanceError):
        chunk.text = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        chunk.metadata["k"] = "changed"  # type: ignore[index]


def test_metadata_is_copied_not_aliased():
    meta = {"k": "v"}
    doc = Document("d", "text", meta)
    meta["k"] = "mutated"
    assert doc.metadata["k"] == "v"


def test_retrieval_result_rank_is_one_based():
    chunk = Chunk("c", "d", "abc", 0, 3)
    assert RetrievalResult(chunk, 0.9, 1, "demo").rank == 1
    with pytest.raises(ValueError, match="1-based"):
        RetrievalResult(chunk, 0.9, 0, "demo")
