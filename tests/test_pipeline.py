import re

import pytest

from rag_lab.config import Settings
from rag_lab.ingestion import (
    CleanedDocument,
    clean_corpus,
    clean_document,
    load_corpus,
)
from rag_lab.ingestion.cleaning import strip_markdown_markup
from rag_lab.ingestion.offsets import NormalizedText
from rag_lab.types import Document

DOCS = {d.doc_id: d for d in load_corpus(Settings.from_env({}).sample_docs_dir)}
CLEANED = {c.doc_id: c for c in clean_corpus(DOCS.values())}
LETTER_TOKENS = re.compile(r"[^\W\d_]+")


def tokens(s: str) -> list[str]:
    return LETTER_TOKENS.findall(s.lower())


def md(text: str) -> str:
    return strip_markdown_markup(NormalizedText.identity(text)).text


def doc(text: str, file_type: str = "markdown", doc_id: str = "d") -> Document:
    return Document(doc_id, text, {"file_type": file_type, "source": f"{doc_id}.{file_type}"})


# ------------------------------------------------ markdown markup removal
@pytest.mark.parametrize("raw,expected", [
    ("# Title", "Title"),
    ("### Deep heading", "Deep heading"),
    ("**bold** and *italic* and _also_ and __strong__", "bold and italic and also and strong"),
    ("use `WHERE x = 1` here", "use WHERE x = 1 here"),
    ("see [the docs](http://x.org/a_b) now", "see the docs now"),
    ("![a diagram](pic.png) below", "a diagram below"),
    ("![](pic.png)text", "text"),
    ("~~old~~ new", "old new"),
])
def test_markup_removed_words_kept(raw, expected):
    assert md(raw) == expected


@pytest.mark.parametrize("keep", [
    "snake_case_name and my_var_2",       # underscores inside words
    "2 * 3 * 4 = 24",                     # lone asterisks
    "- bullet item",                      # list markers are structure: kept
    "1. numbered item",
    "> a quote",
    "| a | b |",
    "plain text, nothing to strip.",
    "#hashtag is not a heading",
])
def test_things_that_are_not_markup_are_untouched(keep):
    assert md(keep) == keep


def test_only_the_markers_are_deleted_so_citations_inside_links_are_exact():
    n = strip_markdown_markup(NormalizedText.identity("see [the docs](http://x.org) now"))
    assert n.text == "see the docs now"
    assert n.original_text(4, 12) == "the docs"                       # not the url, not the brackets
    assert n.original_text(0, len(n.text)) == "see [the docs](http://x.org) now"


# ------------------------------------------------ the pipeline on real documents
@pytest.mark.parametrize("doc_id", sorted(DOCS))
def test_the_original_is_never_modified(doc_id):
    c = CLEANED[doc_id]
    assert c.original is DOCS[doc_id]
    assert c.normalized.original == DOCS[doc_id].text


@pytest.mark.parametrize("doc_id", sorted(DOCS))
def test_cleaned_text_is_tidy(doc_id):
    t = CLEANED[doc_id].text
    assert t == t.strip()
    assert "\r" not in t and "\t" not in t
    assert not re.search(r"[ \t]{2,}", t) and "\n\n\n" not in t
    assert not re.search(r"[ \t]\n", t)


@pytest.mark.parametrize("doc_id", sorted(DOCS))
def test_no_words_are_lost_or_invented(doc_id):
    assert tokens(CLEANED[doc_id].text) == tokens(DOCS[doc_id].text)


@pytest.mark.parametrize("doc_id", sorted(DOCS))
def test_every_cleaned_window_quotes_the_matching_original_words(doc_id):
    c = CLEANED[doc_id]
    for start in range(0, len(c.text) - 30, 97):
        quote = c.quote(start, start + 30)
        assert tokens(quote) == tokens(c.text[start:start + 30])
        a, b = c.source_span(start, start + 30)
        assert c.original.text[a:b] == quote


@pytest.mark.parametrize("doc_id", sorted(DOCS))
def test_cleaning_the_retrieval_document_again_changes_nothing(doc_id):
    once = CLEANED[doc_id].retrieval_document
    assert clean_document(once).text == once.text


# ------------------------------------------------ steps recorded per file type
def test_steps_for_each_kind_of_document():
    assert CLEANED["space_black_holes"].steps == ("unicode", "whitespace", "reflow", "hyphenation")
    assert CLEANED["computing_tcp_ip_networking"].steps == ("unicode", "whitespace", "reflow", "hyphenation")
    assert CLEANED["space_mars_exploration"].steps == (
        "unicode", "whitespace", "reflow", "hyphenation", "markdown", "whitespace")


# ------------------------------------------------ real markdown and pdf
def test_real_markdown_headings_and_bold_are_stripped_but_text_and_bullets_stay():
    mars = CLEANED["space_mars_exploration"].text
    assert mars.startswith("Exploring Mars\n\nMars is the fourth planet")
    assert "## " not in mars and "# " not in mars
    vit = CLEANED["health_vitamins_nutrition"].text
    assert "- Vitamin C supports collagen synthesis" in vit            # bullet kept, bold marker gone
    assert "**" not in vit
    db = CLEANED["computing_database_indexing"].text
    assert "WHERE email = 'a@example.com'" in db and "`" not in db     # inline code ticks removed


def test_real_original_markdown_is_still_available_untouched():
    c = CLEANED["space_mars_exploration"]
    assert c.original.text.startswith("# Exploring Mars")
    # The "# " marker was deleted BEFORE the first visible letter, so it lies outside the span:
    assert c.source_span(0, len("Exploring Mars")) == (2, 16)
    assert c.quote(0, len("Exploring Mars")) == "Exploring Mars"
    assert c.original.text[:16] == "# Exploring Mars"                 # ...but the original still has it
    # A span that continues past the heading includes markers that lie INSIDE it:
    i = c.text.index("Rovers")
    assert "## Rovers" in c.original.text
    assert c.quote(i - 3, i + 6).endswith("\n\n## Rovers")             # marker INSIDE the span is quoted


def test_real_pdf_is_reflowed_and_phrase_search_now_works():
    c = CLEANED["space_black_holes"]
    assert "in direct proportion" in c.text
    assert "in direct proportion" not in c.original.text
    assert c.text.startswith("Black Holes\nA black hole is")
    i = c.text.index("in direct proportion")
    assert c.quote(i, i + 20) == "in direct\nproportion"


def test_pdf_page_lookup_through_the_map():
    c = CLEANED["space_black_holes"]
    i = c.text.index("LIGO")
    assert c.page_for_span(i, i + 4) == 2
    assert c.page_for_span(0, 11) == 1
    assert CLEANED["space_apollo_program"].page_for_span(0, 5) is None


# ------------------------------------------------ retrieval_document
def test_retrieval_document_has_cleaned_text_and_safe_metadata():
    pdf = CLEANED["space_black_holes"]
    r = pdf.retrieval_document
    assert r.doc_id == "space_black_holes" and r.text == pdf.text
    assert "page_ranges" not in r.metadata and "headings" not in r.metadata    # offsets would be wrong
    assert r.metadata["topic"] == "space" and r.metadata["cleaned"] is True
    assert r.metadata["char_count"] == len(r.text) != r.metadata["original_char_count"]
    assert r.metadata["cleaning_steps"] == pdf.steps
    assert "page_ranges" in pdf.original.metadata                              # original keeps its own


def test_markdown_headings_metadata_not_copied_to_retrieval_document():
    r = CLEANED["space_mars_exploration"].retrieval_document
    assert "headings" not in r.metadata
    assert "headings" in DOCS["space_mars_exploration"].metadata


# ------------------------------------------------ fenced code is protected
CODE_MD = "# Demo\n\nUse **this**:\n\n```python\nx  =  a * b * c   # keep  as is\nname = 'a_b_c'\n```\n\nDone.\n"


def test_markdown_with_code_fence_only_gets_unicode_and_whitespace_cleaning():
    c = clean_document(doc(CODE_MD))
    assert c.steps == ("unicode", "whitespace")
    assert "```python" in c.text and "**this**" in c.text                 # markup left alone
    assert "name = 'a_b_c'" in c.text and "a * b * c" in c.text           # code content intact


# ------------------------------------------------ corpus-wide hyphenation evidence
def test_corpus_vocabulary_lets_one_document_help_another():
    a = doc("The international team met. Many international groups came.", "txt", "a")
    b = doc("A big inter-\nnational group joined later on Tuesday.", "txt", "b")
    alone = clean_document(b)
    together = {c.doc_id: c for c in clean_corpus([a, b])}["b"]
    assert "inter-national" in alone.text                                 # no evidence alone: hyphen kept
    assert "international group" in together.text                         # evidence from document a
    assert together.quote(together.text.index("international"), together.text.index("international") + 13) \
        == "inter-\nnational"


# ------------------------------------------------ validation
def test_cleaned_document_rejects_a_mismatched_map():
    d1, d2 = doc("hello world", "txt", "x"), doc("something else", "txt", "y")
    with pytest.raises(ValueError, match="not derived"):
        CleanedDocument(d1, NormalizedText.identity(d2.text), ())
