import re
import textwrap

import pytest

from rag_lab.config import Settings
from rag_lab.ingestion import load_corpus, page_for_offset
from rag_lab.ingestion.cleaning import (
    FULL_WIDTH_RATIO,
    _ends_sentence,
    _is_structural,
    estimate_wrap_width,
    normalize_unicode,
    normalize_whitespace,
    reflow_decisions,
    reflow_hard_wrapped_lines,
)
from rag_lab.ingestion.offsets import NormalizedText

DOCS = {d.doc_id: d for d in load_corpus(Settings.from_env({}).sample_docs_dir)}
SINGLE_NEWLINE = re.compile(r"(?<!\n)\n(?!\n)")


def through_2b_2c(text: str) -> NormalizedText:
    return normalize_whitespace(normalize_unicode(NormalizedText.identity(text)))


def reflow(text: str) -> NormalizedText:
    return reflow_hard_wrapped_lines(NormalizedText.identity(text))


# A small hand-made hard-wrapped passage (width ~40) with a title, 2 paragraphs, a list.
WRAPPED = (
    "A Title\n"
    "The telescope looks at infrared light and\n"
    "can see through cosmic dust. It sits far\n"
    "from Earth near a special point in space\n"
    "where the Sun stays behind the shield.\n"
    "\n"
    "Second paragraph is built from lines that\n"
    "also stop in the middle of a sentence so\n"
    "they must be joined together as one text.\n"
    "\n"
    "- first bullet\n"
    "- second bullet\n"
)


# ------------------------------------------------ the rules
def test_mid_sentence_full_width_lines_are_joined():
    out = reflow(WRAPPED).text
    assert "infrared light and can see through cosmic dust." in out
    assert "from Earth near a special point in space where the Sun" in out


def test_title_is_kept_on_its_own_line():
    assert reflow(WRAPPED).text.startswith("A Title\nThe telescope")


def test_paragraph_breaks_are_preserved():
    assert reflow(WRAPPED).text.count("\n\n") == WRAPPED.count("\n\n") == 2


def test_lists_are_not_merged():
    assert reflow(WRAPPED).text.endswith("- first bullet\n- second bullet\n")


def test_sentence_end_is_kept_even_when_the_line_is_full():
    text = ("x" * 40 + " word one two\n") * 3 + "y" * 40 + " ends here.\nNext sentence starts here and goes\non\n"
    decisions = {d.line_before[-10:]: d for d in reflow_decisions(text)}
    assert decisions["ends here."].join is False
    assert "ends a sentence" in decisions["ends here."].reason


@pytest.mark.parametrize("line", ["# Heading", "- item", "* item", "+ item", "1. item", "2) item",
                                  "> quote", "```python", "~~~", "| a | b |", "    indented", "\tcode"])
def test_structural_lines_detected(line):
    assert _is_structural(line)


@pytest.mark.parametrize("line", ["Plain words", "7.6 million", "**bold** text", "-5 degrees", "2019 was"])
def test_ordinary_lines_are_not_structural(line):
    assert not _is_structural(line)


@pytest.mark.parametrize("line,expected", [
    ("It ends.", True), ('He said "stop."', True), ("(see below.)", True), ("Really?", True),
    ("Note:", True), ("in the middle", False), ("a comma,", False), ("", False), ('"', False),
])
def test_sentence_end_detection(line, expected):
    assert _ends_sentence(line) is expected


def test_hyphen_at_line_end_is_left_alone_for_step_2e():
    text = ("w" * 38 + " inter-\nnational ") * 1 + "\n".join(["w" * 38 + " filler line goes on and on"] * 6)
    decisions = reflow_decisions(text)
    hyphen = next(d for d in decisions if d.line_before.endswith("inter-"))
    assert hyphen.join is False and "hyphen" in hyphen.reason


def test_not_enough_evidence_means_nothing_changes():
    text = "short\nlines\nonly three\nhere\n"          # 3 single newlines < MIN_EVIDENCE
    assert estimate_wrap_width(text) is None
    assert reflow(text).text == text


def test_empty_and_newline_free_text():
    assert reflow("one long line with no newline").text == "one long line with no newline"
    assert reflow("").text == ""


def test_idempotent():
    once = reflow(WRAPPED)
    assert reflow_hard_wrapped_lines(once).text == once.text


# ------------------------------------------------ offsets
def test_citation_across_a_joined_line_quotes_the_original_newline():
    n = reflow(WRAPPED)
    i = n.text.index("light and can")
    assert n.original_text(i, i + len("light and can")) == "light and\ncan"


# ------------------------------------------------ real documents
def test_real_pdf_phrase_split_across_lines_is_found_after_reflow():
    # This is "Experiment 3" from the lesson on why cleaning is needed.
    pdf = DOCS["space_black_holes"]
    assert "in direct proportion" not in pdf.text            # raw: broken by "\n"
    cleaned = reflow_hard_wrapped_lines(through_2b_2c(pdf.text))
    assert "in direct proportion" in cleaned.text
    i = cleaned.text.index("in direct proportion")
    assert cleaned.original_text(i, i + 20) == "in direct\nproportion"


def test_real_pdf_title_and_paragraph_boundaries_survive():
    cleaned = reflow_hard_wrapped_lines(through_2b_2c(DOCS["space_black_holes"].text))
    assert cleaned.text.startswith("Black Holes\nA black hole is")      # title kept apart
    assert "supernova explosion.\nThe size of the event horizon" in cleaned.text   # sentence end kept


def test_real_pdf_fewer_lines_and_no_words_lost():
    base = through_2b_2c(DOCS["space_black_holes"].text)
    cleaned = reflow_hard_wrapped_lines(base)
    assert cleaned.text.split() == base.text.split()                    # every word still there
    assert len(cleaned.text.split("\n")) < len(base.text.split("\n"))
    assert cleaned.text.count("\n\n") == base.text.count("\n\n")


def test_real_pdf_page_lookup_still_works_through_the_map():
    pdf = DOCS["space_black_holes"]
    cleaned = reflow_hard_wrapped_lines(through_2b_2c(pdf.text))
    i = cleaned.text.index("LIGO")
    start, _ = cleaned.to_original_span(i, i + 4)
    assert pdf.text[start:start + 4] == "LIGO"
    assert page_for_offset(pdf, start) == 2


@pytest.mark.parametrize("doc_id", sorted(set(DOCS) - {"space_black_holes"}))
def test_unwrapped_txt_and_markdown_documents_are_left_alone(doc_id):
    base = through_2b_2c(DOCS[doc_id].text)
    assert reflow_hard_wrapped_lines(base).text == base.text


# ------------------------------------------------ ground truth: wrap real text ourselves
def wrap_paragraphs(text: str, width: int) -> str:
    """Hard-wrap each single-line paragraph of a REAL document, like a PDF would."""
    blocks = []
    for para in text.split("\n\n"):
        if "\n" in para or para.startswith("#"):
            blocks.append(para)                                # list blocks / headings stay as they are
        else:
            blocks.append(textwrap.fill(para, width=width, break_long_words=False, break_on_hyphens=False))
    return "\n\n".join(blocks)


@pytest.mark.parametrize("width", [60, 80])
@pytest.mark.parametrize("doc_id", sorted(set(DOCS) - {"space_black_holes"}))
def test_wrapping_real_documents_then_reflowing_loses_nothing_and_joins_mid_sentence_wraps(doc_id, width):
    base = through_2b_2c(DOCS[doc_id].text).text
    wrapped = wrap_paragraphs(base, width)
    assert wrapped != base

    result = reflow(wrapped).text
    assert result.split() == base.split()                                      # no word lost or glued
    assert result.count("\n\n") == base.count("\n\n")                          # paragraph breaks intact

    lengths_width = estimate_wrap_width(wrapped)
    assert lengths_width is not None
    for m in SINGLE_NEWLINE.finditer(result):                                  # every newline still left ...
        before = result[result.rfind("\n", 0, m.start()) + 1 : m.start()]
        after = result[m.end() : (result.find("\n", m.end()) if result.find("\n", m.end()) != -1 else len(result))]
        assert (_is_structural(before) or _is_structural(after) or _ends_sentence(before)
                or len(before) < FULL_WIDTH_RATIO * lengths_width), (before, after)  # ... has a reason
