"""Offset tracking between original text and cleaned ("normalized") text.

Why this exists
---------------
Cleaning changes the text, so character positions in the cleaned text no longer
match positions in the original. Citations must point into the ORIGINAL, so every
cleaned character remembers which original characters it came from:

    original   : "a  b"                (two spaces)
    cleaned    : "a b"                 (second space dropped)
    origin     : (0, 1, 3)             origin[i]     = where cleaned char i's source STARTS
    origin_end : (1, 2, 4)             origin_end[i] = where it ENDS (exclusive)

``origin_end`` matters when several original characters become one cleaned
character (``e`` + accent mark -> ``é``): the cleaned character's source is a
RANGE, not a single position.

The original string is never modified; it stays the source of truth.

Cleaning steps do not build these tuples by hand. They describe *edits* ("replace
characters 5..8 with this text") and call :meth:`NormalizedText.apply_edits`,
which rebuilds the text and composes the maps, so steps can be chained.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

#: (start, end, replacement) over the CURRENT text; start inclusive, end exclusive.
Edit = tuple[int, int, str]


@dataclass(frozen=True)
class NormalizedText:
    """Cleaned text plus a per-character link back to the original.

    Attributes
    ----------
    original   : the untouched source string.
    text       : the cleaned string (what chunking / search will use).
    origin     : ``origin[i]`` = index in ``original`` where the source of
                 ``text[i]`` starts.
    origin_end : ``origin_end[i]`` = index just past where that source ends.
                 Optional: if omitted, every character is assumed to come from
                 exactly one original character (``origin[i] + 1``).

    Both tuples have exactly ``len(text)`` entries, never decrease, and point
    inside ``original``. They may repeat values (one original character that
    becomes several cleaned ones, e.g. the ligature "U+FB01" -> "fi").
    """

    original: str
    text: str
    origin: tuple[int, ...]
    origin_end: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        # Validate immediately: a wrong map would silently give wrong citations.
        if len(self.origin) != len(self.text):
            raise ValueError(
                f"origin has {len(self.origin)} entries but text has {len(self.text)} characters"
            )
        if self.origin_end is None:  # frozen dataclass: must bypass normal assignment
            object.__setattr__(self, "origin_end", tuple(o + 1 for o in self.origin))
        if len(self.origin_end) != len(self.text):
            raise ValueError(
                f"origin_end has {len(self.origin_end)} entries but text has {len(self.text)} characters"
            )

        previous_start = previous_end = -1
        for i, (o, e) in enumerate(zip(self.origin, self.origin_end)):
            if not 0 <= o < len(self.original):
                raise ValueError(f"origin[{i}]={o} is outside the original text")
            if not o < e <= len(self.original):
                raise ValueError(f"origin_end[{i}]={e} must be > origin[{i}]={o} and inside the original")
            if o < previous_start or e < previous_end:
                raise ValueError(f"origin goes backwards at index {i}")
            previous_start, previous_end = o, e

    @classmethod
    def identity(cls, original: str) -> NormalizedText:
        """The "nothing cleaned yet" starting point: every character maps to itself."""
        return cls(original=original, text=original, origin=tuple(range(len(original))))

    # ------------------------------------------------------------------ lookup
    def to_original_span(self, start: int, end: int) -> tuple[int, int]:
        """Translate a span of the cleaned text into a span of the original.

        ``start`` is inclusive and ``end`` exclusive, like Python slicing.

        * original start = where the FIRST cleaned character's source starts
        * original end   = where the LAST cleaned character's source ends

        Characters the cleaner deleted *inside* the span are naturally included in
        the original span; deleted characters *outside* it are not.
        """
        if not 0 <= start < end <= len(self.text):
            raise ValueError(
                f"invalid span ({start}, {end}) for cleaned text of length {len(self.text)}"
            )
        return self.origin[start], self.origin_end[end - 1]

    def original_text(self, start: int, end: int) -> str:
        """The original wording for a cleaned span (what a citation should quote)."""
        original_start, original_end = self.to_original_span(start, end)
        return self.original[original_start:original_end]

    # ------------------------------------------------------------------ editing
    def apply_edits(self, edits: Sequence[Edit]) -> NormalizedText:
        """Return a new NormalizedText with ``edits`` applied to the current text.

        Each edit replaces ``text[start:end]`` with ``replacement`` (``""`` deletes).
        Rules: edits sorted by position, non-overlapping, ``start < end``.
        Text between edits is kept one-to-one.

        Every character of a replacement maps to the WHOLE original range that the
        replaced characters came from. Because we compose with the existing maps,
        the result still points at the real original, however many steps came before.
        """
        parts: list[str] = []
        starts: list[int] = []
        ends: list[int] = []
        cursor = 0

        def keep(upto: int) -> None:
            """Copy the untouched text up to ``upto`` and remember where each character came from."""
            parts.append(self.text[cursor:upto])
            starts.extend(self.origin[cursor:upto])
            ends.extend(self.origin_end[cursor:upto])

        for start, end, replacement in edits:
            if not cursor <= start < end <= len(self.text):
                raise ValueError(
                    f"invalid or overlapping edit ({start}, {end}); edits must be sorted, "
                    f"non-overlapping, start < end, and inside the text (cursor at {cursor})"
                )
            keep(start)
            if replacement:
                span_start, span_end = self.origin[start], self.origin_end[end - 1]
                parts.append(replacement)
                starts.extend([span_start] * len(replacement))
                ends.extend([span_end] * len(replacement))
            cursor = end
        keep(len(self.text))

        return NormalizedText(self.original, "".join(parts), tuple(starts), tuple(ends))
