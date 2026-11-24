"""Step 3b: sentence splitting and sentence-based chunking.

Part 1, ``sentence_spans``: find sentence boundaries with simple, visible rules.
Part 2, ``chunk_sentences``: pack whole sentences into chunks up to a size limit.

Both work with (start, end) spans into the text, so offsets stay exact.
"""

from __future__ import annotations

import re

from rag_lab.chunking.base import make_chunk
from rag_lab.types import Chunk, Document, verify_chunk_integrity

STRATEGY = "sentence"

# A full stop after one of these words does NOT end a sentence (compared in lower case).
ABBREVIATIONS = frozenset({
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.", "etc.", "approx.",
    "e.g.", "i.e.", "fig.", "no.", "eq.", "inc.", "ltd.", "co.", "cf.", "al.",
})

# ".", "!" or "?" (repeated for "..." or "?!"), optional closing quote/bracket, then
# whitespace or the end of the text.
_TERMINATOR = re.compile(r"[.!?]+[\"')\]]*(\s+|$)")
# Two newlines (with optional blanks between) always separate paragraphs.
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n\s*")
_OPENERS = "\"'(["


def _word_before(text: str, end: int) -> str:
    """The whitespace-delimited word that ends just before ``end`` (a punctuation mark)."""
    start = end
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    return text[start:end].lstrip(_OPENERS)


def _is_sentence_end(text: str, match: re.Match[str]) -> bool:
    if match.end() >= len(text):
        return True                                   # the text ends here
    if "\n" in match.group(1) and _PARAGRAPH_BREAK.fullmatch(match.group(1)):
        return True                                   # paragraph break after the mark
    next_char = text[match.end()]
    if not (next_char.isupper() or next_char.isdigit() or next_char in _OPENERS):
        return False                                  # "e.g. this", "etc. and"
    mark = match.group(0).rstrip(" \t\n")
    if mark.startswith(".") and not mark.startswith(".."):   # a lone "." can be an abbreviation
        word = _word_before(text, match.start() + 1)
        if word.lower() in ABBREVIATIONS:
            return False                              # "Dr. Smith"
        if len(word) == 2 and word[0].isupper():
            return False                              # initial: "J. Smith"
    return True


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Split ``text`` into consecutive sentence spans that cover every character.

    Whitespace between sentences belongs to the sentence before it; whitespace at the very
    start belongs to the first sentence. A paragraph break always ends a sentence.
    """
    if not text:
        return []
    ends = {m.end() for m in _PARAGRAPH_BREAK.finditer(text)}
    ends |= {m.end() for m in _TERMINATOR.finditer(text) if _is_sentence_end(text, m)}
    ends.add(len(text))
    spans, start = [], 0
    for end in sorted(ends):
        if end > start:
            spans.append((start, end))
            start = end
    return spans


def _content_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    """Shrink a span so it has no leading or trailing whitespace (``None`` if all blank)."""
    piece = text[start:end]
    stripped = piece.strip()
    if not stripped:
        return None
    lead = len(piece) - len(piece.lstrip())
    return start + lead, start + lead + len(stripped)


def _is_heading(text: str, span: tuple[int, int]) -> bool:
    """A "sentence" with no ending punctuation, followed by a paragraph break: a heading."""
    start, end = span
    return text[end - 1] not in ".!?\"')]" and _PARAGRAPH_BREAK.match(text, end) is not None


def chunk_sentences(document: Document, max_chars: int = 500) -> list[Chunk]:
    """Pack whole sentences into chunks of at most ``max_chars`` characters.

    Sentences are added to the current chunk while it still fits. A single sentence longer
    than ``max_chars`` is kept whole as its own chunk and marked ``oversized=True``
    (the recursive chunker, step 3c, is the one that splits such text). A heading (a line with
    no ending punctuation, followed by a blank line) is never left at the end of a chunk: it
    moves to the start of the next one, so it stays with the text it introduces.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1")
    text = document.text
    sentences = [s for s in (_content_span(text, a, b) for a, b in sentence_spans(text)) if s]
    groups: list[list[tuple[int, int]]] = []
    for sentence in sentences:
        if groups and sentence[1] - groups[-1][0][0] <= max_chars:
            groups[-1].append(sentence)
        elif groups and len(groups[-1]) > 1 and _is_heading(text, groups[-1][-1]):
            groups.append([groups[-1].pop(), sentence])      # keep a heading with the text under it
        else:
            groups.append([sentence])
    chunks = []
    for index, group in enumerate(groups):
        start, end = group[0][0], group[-1][1]
        extra = {"sentence_count": len(group), "oversized": end - start > max_chars}
        chunks.append(make_chunk(document, index, start, end, STRATEGY, extra))
    verify_chunk_integrity(document, chunks)
    return chunks
