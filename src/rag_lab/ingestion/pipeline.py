"""Step 2f: the full cleaning pipeline.

    Document (original, never changed)
        -> unicode -> whitespace -> reflow -> hyphenation -> [markdown markup] -> whitespace
        -> CleanedDocument (original + cleaned text + offset map + which steps ran)

The original ``Document`` stays the source of truth: every span of cleaned text can be
translated back to the original wording (for citations) and to a PDF page.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable
from dataclasses import dataclass

from rag_lab.ingestion.cleaning import (
    build_vocabulary,
    normalize_unicode,
    normalize_whitespace,
    reflow_hard_wrapped_lines,
    repair_hyphenation,
    strip_markdown_markup,
)
from rag_lab.ingestion.loaders import page_for_offset
from rag_lab.ingestion.offsets import NormalizedText
from rag_lab.types import Document

_FENCED_CODE = re.compile(r"^[ \t]*(```|~~~)", re.MULTILINE)
# Metadata whose character offsets point into the ORIGINAL text: wrong for cleaned text.
_ORIGINAL_OFFSET_METADATA = ("page_ranges", "headings")


@dataclass(frozen=True)
class CleanedDocument:
    """A document plus its cleaned retrieval text, linked character by character."""

    original: Document
    normalized: NormalizedText
    steps: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.normalized.original != self.original.text:
            raise ValueError("normalized text was not derived from this document's text")

    @property
    def doc_id(self) -> str:
        """The id of the document this cleaned text came from."""
        return self.original.doc_id

    @property
    def text(self) -> str:
        """The cleaned text used for chunking, embedding and keyword search."""
        return self.normalized.text

    def source_span(self, start: int, end: int) -> tuple[int, int]:
        """Cleaned span -> span in the original text (``start`` inclusive, ``end`` exclusive)."""
        return self.normalized.to_original_span(start, end)

    def quote(self, start: int, end: int) -> str:
        """The original wording for a cleaned span: what a citation should show."""
        return self.normalized.original_text(start, end)

    def page_for_span(self, start: int, end: int) -> int | None:
        """1-based PDF page where the span starts; ``None`` for non-paginated files."""
        if "page_ranges" not in self.original.metadata:
            return None
        original_start, _ = self.source_span(start, end)
        return page_for_offset(self.original, original_start)

    @property
    def retrieval_document(self) -> Document:
        """A ``Document`` whose text is the cleaned text, ready for chunking (Module 3).

        Metadata that holds offsets into the ORIGINAL text (page ranges, heading
        offsets) is dropped, because it would be silently wrong for this text.
        Use :meth:`page_for_span` and :meth:`source_span` for that information.
        """
        metadata = {k: v for k, v in self.original.metadata.items() if k not in _ORIGINAL_OFFSET_METADATA}
        metadata.update(
            cleaned=True,
            char_count=len(self.text),
            original_char_count=len(self.original.text),
            cleaning_steps=self.steps,
        )
        return Document(self.original.doc_id, self.text, metadata)


def clean_document(document: Document, known_words: Collection[str] = ()) -> CleanedDocument:
    """Run the cleaning steps on one document.

    * Every document: Unicode (2b) and whitespace (2c).
    * Reflow (2d) and hyphenation (2e): everything EXCEPT Markdown that contains
      fenced code blocks (code must not be rewritten).
    * Markdown only: markup removal, then a final whitespace pass. Skipped when the
      file has fenced code blocks.
    ``known_words`` is extra spelling evidence for hyphenation (see :func:`clean_corpus`).
    """
    is_markdown = document.metadata.get("file_type") == "markdown"
    current = NormalizedText.identity(document.text)
    steps: list[str] = []

    current = normalize_unicode(current)
    steps.append("unicode")
    current = normalize_whitespace(current)
    steps.append("whitespace")

    protect_code = is_markdown and _FENCED_CODE.search(current.text) is not None
    if not protect_code:
        current = reflow_hard_wrapped_lines(current)
        steps.append("reflow")
        current = repair_hyphenation(current, known_words)
        steps.append("hyphenation")
        if is_markdown:
            current = strip_markdown_markup(current)
            steps.append("markdown")
            current = normalize_whitespace(current)   # markers can leave doubled spaces
            steps.append("whitespace")

    return CleanedDocument(document, current, tuple(steps))


def clean_corpus(documents: Iterable[Document]) -> list[CleanedDocument]:
    """Clean many documents, letting each one's hyphenation use the vocabulary of ALL of them."""
    documents = list(documents)
    vocabulary: set[str] = set()
    for d in documents:
        vocabulary |= build_vocabulary(normalize_unicode(NormalizedText.identity(d.text)).text)
    return [clean_document(d, known_words=vocabulary) for d in documents]
