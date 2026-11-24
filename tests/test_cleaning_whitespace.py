import re

import pytest

from rag_lab.config import Settings
from rag_lab.ingestion import load_corpus
from rag_lab.ingestion.cleaning import normalize_unicode, normalize_whitespace
from rag_lab.ingestion.offsets import NormalizedText

DOCS = {d.doc_id: d for d in load_corpus(Settings.from_env({}).sample_docs_dir)}


def clean(s: str) -> NormalizedText:
    return normalize_whitespace(NormalizedText.identity(s))


# ------------------------------------------------ each pass
@pytest.mark.parametrize("dirty,expected", [
    ("one   two", "one two"),                    # A: repeated spaces
    ("one\ttwo", "one two"),                     # A: lone tab between words
    ("one \t \t two", "one two"),                # A: mixed spaces and tabs
    ("one two", "one two"),                      # A: already fine
    ("end   \nnext", "end\nnext"),               # B: trailing spaces
    ("end\t\t\nnext", "end\nnext"),              # B: trailing tabs
    ("a\n   \nb", "a\n\nb"),                     # B: whitespace-only line becomes a blank line
    ("a\n\n\n\n\nb", "a\n\nb"),                  # C: 5 newlines -> 2
    ("a\n\n\nb", "a\n\nb"),                      # C: 3 -> 2
    ("a\n \n \n \nb", "a\n\nb"),                 # B+C together
    ("\n\n  hello\n\n", "hello"),                # D: edges
])
def test_each_technique(dirty, expected):
    assert clean(dirty).text == expected


# ------------------------------------------------ what must NOT change
def test_paragraph_break_is_kept():
    assert clean("para one.\n\npara two.").text == "para one.\n\npara two."


def test_single_newline_is_never_merged():
    # Hard-wrapped lines are the job of step 2d, not of whitespace rules.
    assert clean("can\nescape once").text == "can\nescape once"


def test_markdown_indentation_is_preserved():
    md = "- item\n    - nested\n\n    code  block"
    assert clean(md).text == "- item\n    - nested\n\n    code block"   # only the INNER run collapsed


def test_tabs_used_as_leading_indentation_are_preserved():
    assert clean("a\n\tb").text == "a\n\tb"


def test_all_whitespace_becomes_empty():
    n = clean(" \n\t \n ")
    assert n.text == "" and n.origin == ()


def test_idempotent():
    dirty = "  a   b \t\n\n\n\n  c\t\td  \n\n"
    once = clean(dirty)
    assert normalize_whitespace(once).text == once.text


# ------------------------------------------------ offsets
def test_citation_includes_the_original_extra_whitespace():
    n = clean("one   two   three")
    assert n.text == "one two three"
    assert n.original_text(0, 7) == "one   two"                      # "one two" -> 9 original chars
    assert n.to_original_span(4, 7) == (6, 9)                         # "two"


def test_chained_with_unicode_step_still_maps_to_the_original():
    raw = "a\u00a0 \u00a0b"                                          # no-break space, space, no-break space
    n = normalize_whitespace(normalize_unicode(NormalizedText.identity(raw)))
    assert n.text == "a b"
    assert n.to_original_span(0, 3) == (0, 5)
    assert n.original is raw


def test_leading_strip_shifts_offsets_correctly():
    raw = "\n\n   Moon landing  \n"
    n = clean(raw)
    assert n.text == "Moon landing"
    assert n.to_original_span(0, 4) == (5, 9)
    assert n.original_text(0, 4) == "Moon"


# ------------------------------------------------ real documents
def expected_after_whitespace_step(original: str) -> str:
    """What step 2c should produce from an already-clean-looking document:
    edges trimmed, and any run of 3+ newlines reduced to a paragraph break."""
    return re.sub(r"\n{3,}", "\n\n", original.strip())


@pytest.mark.parametrize("doc_id", sorted(DOCS))
def test_real_documents_only_lose_edge_whitespace_and_excess_blank_lines(doc_id):
    original = DOCS[doc_id].text
    assert clean(original).text == expected_after_whitespace_step(original)


def test_real_pdf_page_boundary_has_a_triple_newline_that_is_reduced():
    # pypdf ends each page with "\n" and our loader adds the "\n\n" page separator.
    from rag_lab.ingestion import page_for_offset

    pdf = DOCS["space_black_holes"]
    assert "motion.\n\n\nIn 2019" in pdf.text
    n = clean(pdf.text)
    assert "motion.\n\nIn 2019" in n.text
    i = n.text.index("In 2019")
    original_start, _ = n.to_original_span(i, i + 7)
    assert pdf.text[original_start:original_start + 7] == "In 2019"
    assert page_for_offset(pdf, original_start) == 2     # citation still lands on page 2


def test_real_markdown_keeps_all_paragraph_breaks():
    original = DOCS["space_mars_exploration"].text
    n = clean(original)
    assert n.text.count("\n\n") == original.strip().count("\n\n") == 6


def test_real_pdf_hard_wrapped_lines_are_untouched_by_this_step():
    original = DOCS["space_black_holes"].text
    n = clean(original)
    single = lambda t: len(re.findall(r"(?<!\n)\n(?!\n)", t))
    # 22 in the raw text; 21 after, because the very last "\n" of the document is trimmed.
    # Nothing inside the paragraphs was merged.
    assert single(original) == 22
    assert single(n.text) == single(original.strip()) == 21


def dirty_version(text: str) -> str:
    """Realistically damage a REAL document's whitespace."""
    dirty = (text.replace(". ", ".   ").replace(" the ", " \t the  ")
                 .replace("\n\n", "\n\n\n \n\n\t\n"))
    dirty = dirty.replace("\n", " \n")
    return "\n \n" + dirty + "  \n\n\n"


@pytest.mark.parametrize("doc_id", sorted(DOCS))
def test_cleaning_a_dirtied_real_document_recovers_it_exactly(doc_id):
    original = DOCS[doc_id].text
    dirty = dirty_version(original)
    assert dirty != original and len(dirty) > len(original)
    assert clean(dirty).text == expected_after_whitespace_step(original)
