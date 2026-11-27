"""Step 8b: the answer type and the helpers every model call shares.

A model call (see ``openrouter.py``) ends by turning the reply into an ``Answer``: the text,
which sources it cited, and flags for the cases a caller must handle (no sources found, the
model said it does not know, the model refused, the reply was cut off). Keeping that step in one
place means citation parsing and the flags behave the same whichever model wrote the reply.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from rag_lab.prompts import Prompt, SourceRef

NO_SOURCES_TEXT = "I don't know based on the provided sources."
REFUSAL_TEXT = "The model declined to answer this request."
UNKNOWN_PREFIX = "i don't know based on the provided sources"
_CITATION = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class Answer:
    text: str
    cited: tuple[SourceRef, ...]            # the sources the answer cites, in citation-number order
    invalid_citations: tuple[int, ...]      # numbers cited that match no source (a model slip)
    sources: tuple[SourceRef, ...]          # every source that was shown to the model
    called_model: bool                      # False when no chunks were found (no API call was made)
    unknown: bool                           # the answer says the sources do not contain it
    refused: bool                           # the model declined to answer
    truncated: bool                         # the reply hit the token limit and was cut off
    model: str | None
    input_tokens: int
    output_tokens: int


def extract_citations(text: str, sources: Sequence[SourceRef]) -> tuple[tuple[SourceRef, ...], tuple[int, ...]]:
    """Find ``[n]`` markers. Returns (sources cited, numbers that matched no source), each once, ascending."""
    by_number = {s.number: s for s in sources}
    numbers = sorted({int(m) for m in _CITATION.findall(text)})
    cited = tuple(by_number[n] for n in numbers if n in by_number)
    invalid = tuple(n for n in numbers if n not in by_number)
    return cited, invalid


def no_sources_answer() -> Answer:
    """The reply when retrieval found nothing: no model call is made, and it says so."""
    return Answer(
        text=NO_SOURCES_TEXT, cited=(), invalid_citations=(), sources=(), called_model=False,
        unknown=True, refused=False, truncated=False, model=None, input_tokens=0, output_tokens=0,
    )


def build_answer(
    prompt: Prompt, text: str, *, refused: bool, truncated: bool, model: str | None, input_tokens: int, output_tokens: int
) -> Answer:
    """Turn a model reply (already reduced to plain text and flags) into an ``Answer``."""
    text = text.strip()
    if refused:
        text = text or REFUSAL_TEXT
    cited, invalid = extract_citations(text, prompt.sources)
    return Answer(
        text=text,
        cited=cited,
        invalid_citations=invalid,
        sources=prompt.sources,
        called_model=True,
        unknown=text.casefold().startswith(UNKNOWN_PREFIX),
        refused=refused,
        truncated=truncated,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
