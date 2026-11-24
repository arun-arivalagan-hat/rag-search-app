import re

import pytest

from rag_lab.config import Settings
from rag_lab.ingestion import load_corpus
from rag_lab.ingestion.offsets import NormalizedText

DOCS = {d.doc_id: d for d in load_corpus(Settings.from_env({}).sample_docs_dir)}


def collapse_blank_lines_edits(text: str):
    """Keep the first newline of every run, delete the rest."""
    return [(m.start() + 1, m.end(), "") for m in re.finditer(r"\n{2,}", text)]


# ---------------------------------------------------------------- origin_end
def test_origin_end_defaults_to_origin_plus_one():
    n = NormalizedText("abc", "abc", (0, 1, 2))
    assert n.origin_end == (1, 2, 3)


def test_many_to_one_span_covers_every_source_character():
    # "e" + combining acute accent (2 characters) -> "é" (1 character)
    original = "cafe\u0301!"
    n = NormalizedText.identity(original).apply_edits([(3, 5, "\u00e9")])
    assert n.text == "caf\u00e9!"
    assert n.to_original_span(3, 4) == (3, 5)          # the accent mark is included
    assert n.original_text(3, 4) == "e\u0301"


def test_bad_origin_end_rejected():
    with pytest.raises(ValueError, match="origin_end"):
        NormalizedText("abc", "ab", (0, 1), (0, 2))     # end must be > start


# ---------------------------------------------------------------- apply_edits
def test_deleting_with_edits_matches_the_hand_built_map():
    original = "a  b"
    n = NormalizedText.identity(original).apply_edits([(2, 3, "")])
    assert n.text == "a b"
    assert n.origin == (0, 1, 3)
    assert n.original_text(0, 3) == "a  b"


def test_one_to_many_ligature_expansion():
    n = NormalizedText.identity("\ufb01sh").apply_edits([(0, 1, "fi")])
    assert n.text == "fish"
    assert n.original_text(0, 2) == "\ufb01"
    assert n.original_text(1, 2) == "\ufb01"           # half of "fi" still points at the ligature
    assert n.original_text(2, 4) == "sh"


def test_no_edits_changes_nothing():
    n = NormalizedText.identity("hello").apply_edits([])
    assert n.text == "hello" and n.origin == (0, 1, 2, 3, 4)


def test_real_document_blank_line_collapse():
    original = DOCS["space_mars_exploration"].text
    n = NormalizedText.identity(original).apply_edits(collapse_blank_lines_edits(original))
    assert len(n.text) == len(original) - 6
    cs = n.text.index("Jezero Crater")
    assert (cs, cs + 13) == (836, 849)
    assert n.to_original_span(cs, cs + 13) == (840, 853)
    assert n.original_text(cs, cs + 13) == "Jezero Crater"


def test_edits_can_be_chained_and_still_point_at_the_original():
    original = DOCS["space_mars_exploration"].text
    step1 = NormalizedText.identity(original).apply_edits(collapse_blank_lines_edits(original))
    i = step1.text.index("Jezero")
    step2 = step1.apply_edits([(i, i + 6, "JEZERO")])          # second step works on step 1's output
    assert step2.original is original                           # same untouched source
    j = step2.text.index("JEZERO Crater")
    assert step2.original_text(j, j + 13) == "Jezero Crater"    # composed map is still correct
    assert step2.to_original_span(j, j + 13) == (840, 853)


@pytest.mark.parametrize("doc_id", sorted(DOCS))
def test_every_kept_character_maps_to_an_identical_original_character(doc_id):
    original = DOCS[doc_id].text
    n = NormalizedText.identity(original).apply_edits(collapse_blank_lines_edits(original))
    assert all(n.text[i] == original[o] for i, o in enumerate(n.origin))
    assert n.text.count("\n\n") == 0


def test_mapped_span_of_whole_cleaned_text_covers_first_to_last_source():
    original = "\n\nhello  world\n\n"
    # positions: "\n\n"=0-1, hello=2-6, spaces=7-8, world=9-13, "\n\n"=14-15
    n = NormalizedText.identity(original).apply_edits([(0, 2, ""), (7, 8, ""), (14, 16, "")])
    assert n.text == "hello world"
    assert n.to_original_span(0, len(n.text)) == (2, 14)       # excludes deleted edges


# ---------------------------------------------------------------- bad edits
@pytest.mark.parametrize("edits", [
    [(3, 5, ""), (4, 6, "")],      # overlap
    [(4, 6, ""), (0, 2, "")],      # unsorted
    [(2, 2, "x")],                 # empty source range
    [(0, 99, "")],                 # beyond the text
    [(-1, 2, "")],                 # negative start
])
def test_invalid_edits_rejected(edits):
    with pytest.raises(ValueError, match="edit"):
        NormalizedText.identity("abcdefgh").apply_edits(edits)
