import re

import pytest

from rag_lab.config import Settings
from rag_lab.ingestion import load_corpus
from rag_lab.ingestion.cleaning import (
    _is_structural,
    build_vocabulary,
    hyphenation_decisions,
    normalize_unicode,
    normalize_whitespace,
    reflow_hard_wrapped_lines,
    repair_hyphenation,
)
from rag_lab.ingestion.offsets import NormalizedText

DOCS = {d.doc_id: d for d in load_corpus(Settings.from_env({}).sample_docs_dir)}


def repair(text: str, known=()) -> NormalizedText:
    return repair_hyphenation(NormalizedText.identity(text), known)


def action_for(text: str, known=()) -> str:
    (decision,) = hyphenation_decisions(text, known)
    return decision.action


# ------------------------------------------------ joining needs evidence
def test_joins_when_the_joined_word_exists_elsewhere_in_the_document():
    text = "The international team met on Monday. A big inter-\nnational group joined later."
    assert repair(text).text == "The international team met on Monday. A big international group joined later."


def test_without_evidence_the_hyphen_is_kept_never_inventing_a_word():
    out = repair("A big inter-\nnational group joined later.").text
    assert out == "A big inter-national group joined later."


def test_outside_word_list_can_provide_the_evidence():
    text = "A big inter-\nnational group joined later."
    assert repair(text, known={"International"}).text == "A big international group joined later."


# ------------------------------------------------ legitimate hyphens survive
def test_compound_with_a_capitalised_second_part_keeps_its_hyphen():
    assert repair("from the Sun-\nEarth line").text == "from the Sun-Earth line"


def test_compound_whose_hyphenated_form_exists_elsewhere_keeps_its_hyphen():
    text = "A stellar-mass hole and a stellar-\nmass star"
    assert repair(text).text == "A stellar-mass hole and a stellar-mass star"


def test_compound_is_not_joined_even_if_the_joined_form_were_known():
    text = "a stellar-mass hole and a stellar-\nmass star"
    assert repair(text, known={"stellarmass"}).text == "a stellar-mass hole and a stellar-mass star"


def test_when_both_spellings_exist_the_hyphen_wins():
    text = "send an email now, an e-mail later, and an e-\nmail again"
    assert action_for(text) == "keep_hyphen"
    assert repair(text).text.endswith("and an e-mail again")


@pytest.mark.parametrize("text,expected", [
    ("COVID-\n19 cases", "COVID-19 cases"),
    ("in the 1990-\n2000 era", "in the 1990-2000 era"),
])
def test_numbers_next_to_the_hyphen_keep_it(text, expected):
    assert repair(text).text == expected


# ------------------------------------------------ things that must not be touched
@pytest.mark.parametrize("text", [
    "a stellar-mass black hole",                  # hyphen inside a line, no newline
    "word-\n\nnext paragraph",                    # hyphen then a BLANK line (paragraph end)
    "- item one\n- item two",                     # bullets
    "wait --\nnext",                              # double hyphen / dash
    "a dash -\nnext",                             # dash after a space
    "end-\n(see below)",                          # next line does not start with a word
    "Heading-\n# Next heading",                   # structural line after
    "word-\n    indented code",                   # indented line after
    "last word-",                                 # hyphen at the very end of the text
])
def test_untouched_cases(text):
    assert repair(text).text == text


def test_idempotent():
    text = "The international team. A big inter-\nnational group and Sun-\nEarth."
    once = repair(text)
    assert repair_hyphenation(once).text == once.text


def test_multiple_breaks_in_one_text():
    text = "gravitational waves and gravita-\ntional waves, Sun-\nEarth, gravita-\ntional."
    assert repair(text).text == "gravitational waves and gravitational waves, Sun-Earth, gravitational."


# ------------------------------------------------ decisions explain themselves
def test_decisions_carry_pieces_and_reasons():
    text = "international and inter-\nnational"
    (d,) = hyphenation_decisions(text)
    assert (d.left, d.right, d.action) == ("inter", "national", "join")
    assert "joined word exists" in d.reason
    assert text[d.position:d.position + 2] == "-\n"


def test_vocabulary_is_lowercase_words_and_hyphenated_words():
    v = build_vocabulary("Sun-Earth and the Stellar-mass hole, re-run!")
    assert {"sun-earth", "and", "stellar-mass", "hole", "re-run"} <= v


# ------------------------------------------------ offsets
def test_citation_over_a_joined_word_quotes_the_original_hyphen_and_newline():
    text = "gravitational waves and gravita-\ntional waves"
    n = repair(text)
    i = n.text.rindex("gravitational")
    assert n.original_text(i, i + 13) == "gravita-\ntional"


def test_citation_over_a_kept_hyphen_quotes_the_original_newline():
    n = repair("from the Sun-\nEarth line")
    i = n.text.index("Sun-Earth")
    assert n.original_text(i, i + 9) == "Sun-\nEarth"


# ------------------------------------------------ real documents
def test_real_corpus_has_no_line_end_hyphens_so_nothing_changes():
    for doc_id, d in DOCS.items():
        assert "-\n" not in d.text, doc_id                   # a fact about OUR data, not a guarantee
        base = normalize_whitespace(normalize_unicode(NormalizedText.identity(d.text)))
        assert repair_hyphenation(base).text == base.text


def inject_syllable_breaks(text: str, every: int = 3):
    """TEST-ONLY: break every ``every``-th long lowercase word in the middle with '-\\n',
    like a typesetter. Skips headings/lists/indented lines."""
    counter = 0
    injected = 0
    out = []

    def split(m: re.Match) -> str:
        nonlocal counter, injected
        counter += 1
        if counter % every:
            return m.group()
        injected += 1
        w = m.group()
        mid = len(w) // 2
        return w[:mid] + "-\n" + w[mid:]

    for line in text.split("\n"):
        out.append(line if _is_structural(line) else re.sub(r"(?<![\w-])[a-z]{9,}(?![\w-])", split, line))
    return "\n".join(out), injected


def inject_compound_breaks(text: str):
    """TEST-ONLY: put '-\\n' after every hyphen that sits between two letters."""
    injected = 0
    out = []
    for line in text.split("\n"):
        if _is_structural(line):
            out.append(line)
            continue
        new, k = re.subn(r"(?<=[A-Za-z])-(?=[A-Za-z])", "-\n", line)
        injected += k
        out.append(new)
    return "\n".join(out), injected


def clean_text(doc_id: str) -> str:
    return normalize_whitespace(normalize_unicode(NormalizedText.identity(DOCS[doc_id].text))).text


@pytest.mark.parametrize("doc_id", sorted(DOCS))
def test_syllable_breaks_injected_into_a_real_document_are_repaired_exactly(doc_id):
    clean = clean_text(doc_id)
    broken, n = inject_syllable_breaks(clean)
    assert broken != clean or n == 0
    # Evidence: the words of the clean document (as other occurrences / other documents would give).
    assert repair_hyphenation(NormalizedText.identity(broken), known_words=build_vocabulary(clean)).text == clean


@pytest.mark.parametrize("doc_id", sorted(DOCS))
def test_real_compounds_broken_at_their_hyphen_keep_the_hyphen(doc_id):
    clean = clean_text(doc_id)
    broken, n = inject_compound_breaks(clean)
    assert repair_hyphenation(NormalizedText.identity(broken)).text == clean       # no evidence needed


def test_the_injection_tests_are_not_vacuous():
    syllables = sum(inject_syllable_breaks(clean_text(d))[1] for d in DOCS)
    compounds = sum(inject_compound_breaks(clean_text(d))[1] for d in DOCS)
    assert syllables >= 15 and compounds >= 8


def test_pipeline_order_2d_then_2e_on_a_real_pdf_with_injected_hyphenation():
    pdf = DOCS["space_black_holes"]
    assert "in direct\nproportion" in pdf.text                                     # a normal PDF wrap
    raw = pdf.text.replace("the first direct", "the first di-\nrect", 1)           # inject a syllable break
    assert "di-\nrect" in raw
    n = normalize_whitespace(normalize_unicode(NormalizedText.identity(raw)))
    n = reflow_hard_wrapped_lines(n)                                               # 2d skips the hyphen line
    assert "di-\nrect" in n.text
    n = repair_hyphenation(n)                                                      # 2e
    assert "first direct" in n.text and "di-" not in n.text
    assert "in direct proportion" in n.text                                        # 2d still worked
