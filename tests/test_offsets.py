import pytest

from rag_lab.config import Settings
from rag_lab.ingestion import load_corpus
from rag_lab.ingestion.offsets import NormalizedText

DOCS = {d.doc_id: d for d in load_corpus(Settings.from_env({}).sample_docs_dir)}


def collapse_blank_lines(original: str) -> NormalizedText:
    """TEST-ONLY helper: turn every run of newlines into a single newline,
    building the origin list by hand. (The real cleaners come in steps 2b-2e.)"""
    kept_chars: list[str] = []
    origin: list[int] = []
    for i, ch in enumerate(original):
        if ch == "\n" and kept_chars and kept_chars[-1] == "\n":
            continue  # drop: the previous kept character was already a newline
        kept_chars.append(ch)
        origin.append(i)
    return NormalizedText(original, "".join(kept_chars), tuple(origin))


# ---------------------------------------------------------------- toy example
def test_toy_example_from_the_lesson():
    n = NormalizedText(original="a  b", text="a b", origin=(0, 1, 3))
    assert n.to_original_span(2, 3) == (3, 4)      # "b"   -> original[3:4]
    assert n.original_text(2, 3) == "b"
    assert n.to_original_span(0, 3) == (0, 4)      # "a b" -> covers both spaces
    assert n.original_text(0, 3) == "a  b"
    assert n.original_text(0, 1) == "a"            # deleted char outside the span is excluded


# ---------------------------------------------------------------- identity
def test_identity_changes_nothing():
    n = NormalizedText.identity("hello")
    assert n.text == "hello" and n.origin == (0, 1, 2, 3, 4)
    assert n.to_original_span(1, 4) == (1, 4)


@pytest.mark.parametrize("doc_id", sorted(DOCS))
def test_identity_round_trip_on_every_real_document(doc_id):
    original = DOCS[doc_id].text
    n = NormalizedText.identity(original)
    for start, end in [(0, 1), (0, len(original)), (10, 60), (len(original) - 5, len(original))]:
        assert n.original_text(start, end) == original[start:end]


# ---------------------------------------------------------------- real cleaning example
def test_real_document_citation_points_to_original_after_cleaning():
    original = DOCS["space_mars_exploration"].text
    n = collapse_blank_lines(original)
    assert len(n.text) < len(original)                       # cleaning did shrink the text

    needle = "Jezero Crater"
    cs = n.text.index(needle)
    ce = cs + len(needle)
    assert (cs, ce) == (836, 849)                            # position in the CLEANED text
    assert n.to_original_span(cs, ce) == (840, 853)          # position in the ORIGINAL text
    assert n.original_text(cs, ce) == needle                 # citation quotes the right words
    assert original[cs:ce] != needle                         # the uncorrected span is wrong


def test_cleaned_characters_come_from_matching_original_characters():
    # For a cleaner that only DELETES characters, every kept char equals its source.
    n = collapse_blank_lines(DOCS["health_sleep_science"].text)
    assert all(n.text[i] == n.original[o] for i, o in enumerate(n.origin))


# ---------------------------------------------------------------- one char -> many
def test_one_original_char_may_expand_to_several():
    n = NormalizedText(original="\ufb01sh", text="fish", origin=(0, 0, 1, 2))  # "U+FB01" ligature
    assert n.original_text(0, 2) == "\ufb01"                  # "fi" maps to the single ligature
    assert n.to_original_span(0, 4) == (0, 3)


# ---------------------------------------------------------------- validation
def test_length_mismatch_rejected():
    with pytest.raises(ValueError, match="entries"):
        NormalizedText("abc", "ab", (0,))


def test_out_of_range_origin_rejected():
    with pytest.raises(ValueError, match="outside"):
        NormalizedText("abc", "ab", (0, 3))


def test_backwards_origin_rejected():
    with pytest.raises(ValueError, match="backwards"):
        NormalizedText("abc", "ab", (2, 1))


@pytest.mark.parametrize("start,end", [(-1, 2), (2, 2), (3, 1), (0, 4)])
def test_bad_spans_rejected(start, end):
    n = NormalizedText.identity("abc")
    with pytest.raises(ValueError, match="invalid span"):
        n.to_original_span(start, end)
