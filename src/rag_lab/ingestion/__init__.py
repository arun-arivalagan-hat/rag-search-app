"""Module 1 and 2: load documents and clean their text.

    loaders.py   read TXT / Markdown / PDF and the manifest.json metadata
    offsets.py   an offset map, so cleaned text can still point back to the original
    cleaning.py  Unicode, whitespace, PDF line reflow, hyphenation, Markdown markup
    pipeline.py  load_corpus then clean_corpus: the two calls the rest of the app uses
"""

from rag_lab.ingestion.loaders import (
    SUPPORTED_EXTENSIONS,
    IngestionError,
    load_corpus,
    load_document,
    load_manifest,
    page_for_offset,
)

from rag_lab.ingestion.pipeline import CleanedDocument, clean_corpus, clean_document

__all__ = [
    "CleanedDocument",
    "clean_corpus",
    "clean_document",
    "SUPPORTED_EXTENSIONS",
    "IngestionError",
    "load_corpus",
    "load_document",
    "load_manifest",
    "page_for_offset",
]
