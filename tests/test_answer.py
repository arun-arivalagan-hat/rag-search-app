from rag_lab.llm.answer import (
    NO_SOURCES_TEXT,
    REFUSAL_TEXT,
    build_answer,
    extract_citations,
    no_sources_answer,
)
from rag_lab.prompts import SourceRef, build_prompt
from rag_lab.types import Chunk

CHUNKS = [
    Chunk(chunk_id="mars::0", doc_id="mars", text="Jezero was chosen for its river delta.", start_char=0, end_char=38),
    Chunk(chunk_id="apollo::2", doc_id="apollo", text="Apollo 11 landed in 1969.", start_char=0, end_char=25),
]
PROMPT = build_prompt("Why Jezero?", CHUNKS)


def build(text, **overrides):
    flags = dict(refused=False, truncated=False, model="m", input_tokens=5, output_tokens=7)
    flags.update(overrides)
    return build_answer(PROMPT, text, **flags)


# ------------------------------------------------ build_answer
def test_text_citations_usage_and_model_are_carried_over():
    answer = build("Because of its river delta [1].")
    assert answer.text == "Because of its river delta [1]."
    assert [s.chunk_id for s in answer.cited] == ["mars::0"]
    assert answer.invalid_citations == () and answer.called_model
    assert (answer.model, answer.input_tokens, answer.output_tokens) == ("m", 5, 7)
    assert [s.number for s in answer.sources] == [1, 2]
    assert not (answer.unknown or answer.refused or answer.truncated)


def test_surrounding_whitespace_is_trimmed():
    assert build("  \n answer [1] \n").text == "answer [1]"


def test_a_citation_to_a_missing_source_is_reported_not_trusted():
    answer = build("Claim [1] and another [9].")
    assert [s.number for s in answer.cited] == [1] and answer.invalid_citations == (9,)


def test_dont_know_is_flagged_whatever_its_capitalisation():
    assert build("I don't know based on the provided sources.").unknown
    assert build("i DON'T know based on the provided sources").unknown
    assert not build("It was 1969 [2].").unknown


def test_refusal_with_no_text_gets_a_standard_message():
    answer = build("", refused=True)
    assert answer.refused and answer.text == REFUSAL_TEXT


def test_refusal_keeps_the_text_the_model_gave():
    assert build("I can't help with that.", refused=True).text == "I can't help with that."


def test_empty_reply_without_a_refusal_stays_empty():
    answer = build("")
    assert answer.text == "" and not answer.refused


def test_truncated_flag_is_carried_over():
    assert build("Because of its", truncated=True).truncated


# ------------------------------------------------ no_sources_answer
def test_no_sources_answer_says_so_and_marks_that_no_model_was_called():
    answer = no_sources_answer()
    assert answer.text == NO_SOURCES_TEXT
    assert answer.unknown and not answer.called_model
    assert answer.cited == () and answer.sources == ()
    assert (answer.input_tokens, answer.output_tokens, answer.model) == (0, 0, None)


# ------------------------------------------------ citation parsing
REFS = (SourceRef(1, "c1", "d1"), SourceRef(2, "c2", "d2"), SourceRef(3, "c3", "d3"))


def test_extract_citations_handles_adjacent_and_repeated_markers():
    cited, invalid = extract_citations("A [2][3]. B [2]. C [1].", REFS)
    assert [s.number for s in cited] == [1, 2, 3] and invalid == ()


def test_extract_citations_reports_numbers_with_no_source():
    cited, invalid = extract_citations("Claim [1] and another [9] and [0].", REFS)
    assert [s.number for s in cited] == [1] and invalid == (0, 9)


def test_extract_citations_ignores_other_brackets():
    cited, invalid = extract_citations("See [a], [1a], [] and (2).", REFS)
    assert cited == () and invalid == ()


def test_extract_citations_with_no_markers():
    assert extract_citations("No citations here.", REFS) == ((), ())
