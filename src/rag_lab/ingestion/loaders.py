"""Document loaders: files on disk -> :class:`~rag_lab.types.Document`.

Pipeline for one file::

    path ─► read bytes ─► loader for the extension ─► (text, format metadata)
                 │                                          │
                 └─► sha256, size ─► base metadata ─────────┴─► + manifest metadata ─► Document

Design rules
------------
* **Fail loudly.** Undecodable bytes, unsupported types and PDFs without a text
  layer raise :class:`IngestionError`; nothing is silently replaced or skipped.
* **Minimal transformation.** Loaders only decode, normalise newlines and strip a
  BOM. Whitespace clean-up, de-hyphenation, markup stripping etc. belong to
  Module 2 so that each step is separately testable.
* **Canonical text.** ``Document.text`` is the single string all later character
  offsets refer to. Loading the same file twice yields identical text and offsets.
* **Metadata is flat and immutable-friendly**: scalars and tuples only.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from rag_lab.types import Document

PAGE_SEPARATOR = "\n\n"


class IngestionError(Exception):
    """A file could not be turned into a faithful Document."""


@dataclass(frozen=True)
class _Parsed:
    """What a format-specific loader returns: text plus metadata it discovered."""

    text: str
    metadata: dict[str, Any]


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def decode_text(raw: bytes, source: str = "<bytes>") -> str:
    """Decode UTF-8 (BOM tolerated) and normalise newlines to ``\\n``.

    ``utf-8-sig`` silently drops a leading BOM and otherwise behaves like UTF-8.
    Invalid bytes raise instead of being replaced: guessing would corrupt text
    without anyone noticing.
    """
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise IngestionError(
            f"{source}: not valid UTF-8 (byte {exc.start}). Re-save the file as UTF-8."
        ) from exc
    # Order matters: CRLF first, otherwise it would become two newlines.
    return text.replace("\r\n", "\n").replace("\r", "\n")


# --------------------------------------------------------------------------- #
# TXT
# --------------------------------------------------------------------------- #
def parse_txt(raw: bytes, source: str) -> _Parsed:
    return _Parsed(decode_text(raw, source), {})


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---[ \t]*(?:\n|\Z)", re.DOTALL)
_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE = re.compile(r"^[ \t]*(```|~~~)")


def _parse_front_matter(block: str) -> dict[str, str]:
    """Parse flat ``key: value`` lines. Nested YAML is intentionally unsupported."""
    meta: dict[str, str] = {}
    for line in block.split("\n"):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, sep, value = line.partition(":")
        if not sep or not key.strip():
            raise IngestionError(f"unsupported front-matter line: {line!r}")
        meta[key.strip()] = value.strip().strip("\"'")
    return meta


def extract_headings(text: str) -> tuple[tuple[int, str, int], ...]:
    """Return ``(level, title, start_char)`` for every ATX heading.

    Lines inside fenced code blocks are skipped: ``# install deps`` in a bash
    snippet is a comment, not a section.
    """
    headings: list[tuple[int, str, int]] = []
    in_fence = False
    offset = 0
    for line in text.split("\n"):
        if _FENCE.match(line):
            in_fence = not in_fence
        elif not in_fence:
            m = _HEADING.match(line)
            if m:
                headings.append((len(m.group(1)), m.group(2), offset))
        offset += len(line) + 1  # +1 for the "\n" removed by split
    return tuple(headings)


def parse_markdown(raw: bytes, source: str) -> _Parsed:
    """Markdown is kept *as markup*; structure is lifted into metadata.

    Front matter is removed from the text (it is metadata, not content), so all
    offsets, including heading offsets, are relative to the remaining body.
    """
    text = decode_text(raw, source)
    meta: dict[str, Any] = {}

    m = _FRONT_MATTER.match(text)
    if m:
        meta.update({f"fm_{k}": v for k, v in _parse_front_matter(m.group(1)).items()})
        text = text[m.end():]

    headings = extract_headings(text)
    meta["headings"] = headings
    first_h1 = next((h for h in headings if h[0] == 1), None)
    if first_h1:
        meta["title"] = first_h1[1]
    return _Parsed(text, meta)


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #
def parse_pdf(raw: bytes, source: str) -> _Parsed:
    """Extract text page by page and remember where each page sits.

    ``page_ranges[i] == (start, end)`` is the span of page ``i+1`` inside
    ``Document.text``; pages are joined with :data:`PAGE_SEPARATOR`.
    """
    try:
        reader = PdfReader(io.BytesIO(raw))
        if reader.is_encrypted:
            raise IngestionError(f"{source}: PDF is encrypted")
        page_texts = [page.extract_text() or "" for page in reader.pages]
        info = reader.metadata
    except PyPdfError as exc:
        raise IngestionError(f"{source}: cannot parse PDF ({exc})") from exc

    if not any(t.strip() for t in page_texts):
        raise IngestionError(
            f"{source}: PDF has no extractable text (scanned images? it needs OCR)"
        )

    ranges: list[tuple[int, int]] = []
    cursor = 0
    for t in page_texts:
        ranges.append((cursor, cursor + len(t)))
        cursor += len(t) + len(PAGE_SEPARATOR)
    text = PAGE_SEPARATOR.join(page_texts)

    meta: dict[str, Any] = {"page_count": len(page_texts), "page_ranges": tuple(ranges)}
    if info is not None:
        if info.title:
            meta["title"] = str(info.title)
        if info.author:
            meta["author"] = str(info.author)
    return _Parsed(text, meta)


def page_for_offset(document: Document, offset: int) -> int:
    """1-based page number containing ``offset`` (PDF documents only).

    An offset that falls in the blank-line gap between pages is attributed to the
    page before the gap, so every valid offset maps to exactly one page.
    """
    ranges = document.metadata.get("page_ranges")
    if ranges is None:
        raise ValueError(f"{document.doc_id} is not a paginated document")
    if not 0 <= offset < len(document.text):
        raise ValueError(f"offset {offset} outside document of length {len(document.text)}")
    starts = [start for start, _ in ranges]
    return bisect_right(starts, offset)  # number of pages starting at/before offset


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
_PARSERS: dict[str, tuple[str, Callable[[bytes, str], _Parsed]]] = {
    ".txt": ("txt", parse_txt),
    ".md": ("markdown", parse_markdown),
    ".markdown": ("markdown", parse_markdown),
    ".pdf": ("pdf", parse_pdf),
}
SUPPORTED_EXTENSIONS = frozenset(_PARSERS)


def load_document(
    path: str | Path,
    extra_metadata: dict[str, Any] | None = None,
    doc_id: str | None = None,
) -> Document:
    """Load one file. ``extra_metadata`` (e.g. from the manifest) wins over
    anything derived from the file itself."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in _PARSERS:
        raise IngestionError(
            f"{path.name}: unsupported type {ext!r}; supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise IngestionError(f"{path.name}: cannot read file ({exc})") from exc

    file_type, parser = _PARSERS[ext]
    parsed = parser(raw, path.name)
    if not parsed.text.strip():
        raise IngestionError(f"{path.name}: file is empty")

    metadata: dict[str, Any] = {
        "source": path.name,
        "file_type": file_type,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "file_bytes": len(raw),
        "char_count": len(parsed.text),
        **parsed.metadata,
        **(extra_metadata or {}),
    }
    return Document(doc_id or path.stem, parsed.text, metadata)


def load_manifest(directory: str | Path) -> dict[str, dict[str, Any]]:
    """Read ``manifest.json`` -> ``{filename: metadata}``. Missing file -> ``{}``."""
    path = Path(directory) / "manifest.json"
    if not path.exists():
        return {}
    entries = json.loads(path.read_text(encoding="utf-8"))["documents"]
    out: dict[str, dict[str, Any]] = {}
    for entry in entries:
        entry = dict(entry)
        name = entry.pop("file")
        if name in out:
            raise IngestionError(f"manifest lists {name!r} twice")
        out[name] = entry
    return out


def load_corpus(directory: str | Path) -> list[Document]:
    """Load every supported file in ``directory`` (non-recursive), sorted by name.

    * Files listed in the manifest get its metadata.
    * A manifest entry whose file is missing is an error (stale manifest).
    * Files not in the manifest still load, with file-derived metadata only.
    """
    directory = Path(directory)
    manifest = load_manifest(directory)
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)

    missing = set(manifest) - {p.name for p in files}
    if missing:
        raise IngestionError(f"manifest lists files that do not exist: {sorted(missing)}")

    docs: list[Document] = []
    seen: set[str] = set()
    for p in files:
        doc = load_document(p, extra_metadata=manifest.get(p.name))
        if doc.doc_id in seen:
            raise IngestionError(f"duplicate doc_id {doc.doc_id!r} (same stem, different extension)")
        seen.add(doc.doc_id)
        docs.append(doc)
    return docs
