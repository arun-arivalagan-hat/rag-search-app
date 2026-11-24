import unicodedata

import pytest

from rag_lab.config import Settings
from rag_lab.ingestion import load_corpus
from rag_lab.ingestion.cleaning import normalize_unicode
from rag_lab.ingestion.offsets import NormalizedText

DOCS = {d.doc_id: d for d in load_corpus(Settings.from_env({}).sample_docs_dir)}


def clean(s: str) -> NormalizedText:
    return normalize_unicode(NormalizedText.identity(s))


# ------------------------------------------------ the individual techniques
@pytest.mark.parametrize("dirty,expected", [
    ("\ufb01rst \ufb02ow o\ufb03ce", "first flow office"),         # ligatures
    ("don\u2019t \u2018x\u2019", "don't 'x'"),                      # curly single quotes
    ("\u201cquoted\u201d \u201ehi\u201f", '"quoted" "hi"'),         # curly double quotes
    ("Moon\u00a0landing", "Moon landing"),                           # no-break space
    ("a\u2009b\u202fc\u3000d", "a b c d"),                           # thin / narrow / ideographic
    ("Perse\u00adverance", "Perseverance"),                          # soft hyphen
    ("zero\u200bwidth", "zerowidth"),                                # zero-width space
    ("\ufeffstart", "start"),                                        # stray BOM
    ("well\u2010known non\u2011breaking", "well-known non-breaking"),# real hyphens
    ("page1\x0cpage2", "page1\npage2"),                              # form feed -> newline
    ("a\x00b\x07c", "abc"),                                          # control characters
])
def test_each_technique(dirty, expected):
    assert clean(dirty).text == expected


def test_tabs_and_newlines_are_kept():
    assert clean("a\tb\nc").text == "a\tb\nc"


# ------------------------------------------------ NFC
def test_nfc_joins_letter_and_accent():
    n = clean("cafe\u0301!")
    assert n.text == "caf\u00e9!"
    assert n.to_original_span(3, 4) == (3, 5)                        # both source characters
    assert n.original_text(0, len(n.text)) == "cafe\u0301!"


def test_nfc_edit_is_minimal_not_the_whole_word():
    n = clean("cafe\u0301 au lait")
    assert n.to_original_span(0, 3) == (0, 3)                        # "caf" untouched


def test_composed_text_is_unchanged():
    assert clean("caf\u00e9").text == "caf\u00e9"


def test_tamil_decomposed_vowel_sign_is_composed():
    decomposed = "\u0b95\u0bc6\u0bbe"      # KA + vowel sign E + AA
    n = clean(decomposed)
    assert n.text == "\u0b95\u0bca"         # KA + vowel sign O
    assert n.original_text(0, 2) == decomposed


# ------------------------------------------------ meaning must be preserved
@pytest.mark.parametrize("keep", [
    "m\u00b2", "H\u2082O", "Intel\u2122", "\u00bd cup",              # NFKC would damage these
    "10\u201320", "wait\u2014what",                                  # en dash / em dash
    "Sun-Earth", "stellar-mass",                                     # ordinary hyphens
    "na\u00efve caf\u00e9", "\u65e5\u672c\u8a9e", "\u0431\u0443\u043a\u0432\u0430",  # accents, CJK, Cyrillic
    "\U0001F642", "\U0001F468\u200d\U0001F469\u200d\U0001F467",     # emoji, family emoji (ZWJ)
    "\u0645\u06cc\u200c\u062e\u0648\u0627\u0647\u0645",             # Persian word with ZWNJ
    "\u0ba4\u0bae\u0bbf\u0bb4\u0bcd",                                # Tamil
])
def test_legitimate_characters_survive(keep):
    assert clean(keep).text == keep


def test_idempotent():
    dirty = "\ufb01rst \u201cx\u201d\u00a0cafe\u0301\u200b"
    once = clean(dirty)
    twice = normalize_unicode(once)
    assert twice.text == once.text


# ------------------------------------------------ real documents
@pytest.mark.parametrize("doc_id", sorted(DOCS))
def test_real_documents_are_already_clean_so_nothing_changes(doc_id):
    original = DOCS[doc_id].text
    n = clean(original)
    assert n.text == original
    assert n.origin == tuple(range(len(original)))


def dirty_version(text: str) -> str:
    """Realistically damage a REAL document: ligatures, curly quotes, no-break spaces,
    invisible characters, soft hyphens, decomposed accents."""
    return (text.replace("fi", "\ufb01").replace("fl", "\ufb02")
                .replace("'", "\u2019").replace('"', "\u201c")
                .replace(" of ", "\u00a0of\u00a0")
                .replace(",", ",\u200b")
                .replace("tion", "ti\u00adon"))


@pytest.mark.parametrize("doc_id", sorted(DOCS))
def test_cleaning_a_dirtied_real_document_recovers_it_exactly(doc_id):
    original = DOCS[doc_id].text
    dirty = dirty_version(original)
    assert dirty != original
    assert clean(dirty).text == original


def test_citations_point_into_the_dirty_original():
    dirty = dirty_version(DOCS["space_mars_exploration"].text)
    n = clean(dirty)
    # "was designed" sits after several dirtied characters ("U+FB01", ",<ZWSP>", ...)
    i = n.text.index("Opportunity was designed")
    span = n.original_text(i, i + len("Opportunity was designed"))
    assert span == "Opportunity\u00a0was designed" or span == "Opportunity was designed"
    # a span that contains a ligature must quote the ORIGINAL ligature glyph
    j = n.text.index("first wheeled")
    quoted = n.original_text(j, j + len("first wheeled"))
    assert quoted == "\ufb01rst wheeled"
    # the map is doing real work: it is not the do-nothing 0,1,2,... map
    # (ligatures shrink the text and zero-width spaces grow it, so shifts can cancel locally)
    assert n.origin != tuple(range(len(n.origin)))
    assert len(n.original) != len(n.text) or n.original != n.text


def test_every_edit_keeps_the_map_valid_on_a_dirtied_corpus():
    for d in DOCS.values():
        n = clean(dirty_version(d.text))
        for start, end in [(0, 1), (0, len(n.text)), (len(n.text) // 2, len(n.text) // 2 + 20)]:
            a, b = n.to_original_span(start, end)
            assert 0 <= a < b <= len(n.original)
