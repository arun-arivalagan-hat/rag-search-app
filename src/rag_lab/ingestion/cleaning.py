"""Text cleaning steps. Each step takes a NormalizedText and returns a new one,
so the steps can be chained while every character stays linked to the original
(see offsets.py).

Step 2b (this file, so far): Unicode normalization.

Policy
------
We do NOT blindly apply NFKC: it rewrites "m²" to "m2", "H₂O" to "H2O" and
"™" to "TM", changing meaning. Instead we use an explicit, reviewable table of safe
changes plus NFC (which only joins base letters with their accent marks).

Deliberately left alone: en/em dashes, superscripts and symbols, and the joiners
U+200C / U+200D (needed to spell Persian, Indic scripts and emoji correctly).
"""

from __future__ import annotations

import os
import re
import unicodedata
from collections.abc import Collection
from dataclasses import dataclass

from rag_lab.ingestion.offsets import Edit, NormalizedText

# Invisible characters with no meaning in running text -> deleted.
_DELETE = frozenset({
    "\u00ad",  # SOFT HYPHEN
    "\u200b",  # ZERO WIDTH SPACE
    "\u2060",  # WORD JOINER
    "\ufeff",  # BOM / ZERO WIDTH NO-BREAK SPACE found inside text
})

# One character -> its standard replacement.
_REPLACE: dict[str, str] = {
    # Ligatures that PDFs store as a single glyph
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi",
    "\ufb04": "ffl", "\ufb05": "st", "\ufb06": "st",
    # Typographic quotes -> plain quotes
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
    # Characters that really are hyphens
    "\u2010": "-", "\u2011": "-",
    # Exotic spaces -> plain space
    "\u00a0": " ", "\u1680": " ", "\u202f": " ", "\u205f": " ", "\u3000": " ",
    # Line-like control codes -> newline
    "\x0b": "\n", "\x0c": "\n", "\x85": "\n", "\u2028": "\n", "\u2029": "\n\n",
}
_REPLACE.update({chr(cp): " " for cp in range(0x2000, 0x200B)})  # en/em/thin/hair spaces ...

_KEEP_CONTROLS = frozenset({"\n", "\t"})
_WORD = re.compile(r"\S+")


def _character_edits(text: str) -> list[Edit]:
    """Pass 1: single-character deletions and replacements."""
    edits: list[Edit] = []
    for i, ch in enumerate(text):
        if ch in _REPLACE:
            edits.append((i, i + 1, _REPLACE[ch]))
        elif ch in _DELETE or (unicodedata.category(ch) == "Cc" and ch not in _KEEP_CONTROLS):
            edits.append((i, i + 1, ""))
    return edits


def _minimal_edit(offset: int, old: str, new: str) -> Edit:
    """Smallest single edit turning ``old`` into ``new`` (they share a start and end).

    ``"cafe\\u0301!"`` -> ``"café!"`` becomes an edit of just the ``e`` + accent, so
    a citation that ends in this word does not grow more than necessary.
    """
    prefix = len(os.path.commonprefix([old, new]))
    suffix = len(os.path.commonprefix([old[::-1], new[::-1]]))
    suffix = min(suffix, len(old) - prefix, len(new) - prefix)
    if len(old) - prefix - suffix == 0:  # pure insertion: cannot be expressed, replace whole word
        return (offset, offset + len(old), new)
    return (offset + prefix, offset + len(old) - suffix, new[prefix : len(new) - suffix])


def _nfc_edits(text: str) -> list[Edit]:
    """Pass 2: one edit per word that is not already in NFC form."""
    edits: list[Edit] = []
    for m in _WORD.finditer(text):
        word = m.group()
        if unicodedata.is_normalized("NFC", word):
            continue
        edits.append(_minimal_edit(m.start(), word, unicodedata.normalize("NFC", word)))
    return edits


def normalize_unicode(source: NormalizedText) -> NormalizedText:
    """Step 2b. Invisible/odd characters first, then NFC (order matters: a zero-width
    space between ``e`` and its accent would otherwise stop them being joined)."""
    step1 = source.apply_edits(_character_edits(source.text))
    return step1.apply_edits(_nfc_edits(step1.text))


# --------------------------------------------------------------------------- #
# Step 2c: whitespace normalization
# --------------------------------------------------------------------------- #
# Run of spaces/tabs sitting BETWEEN two visible characters. Leading indentation
# (after a newline) and trailing spaces (before one) are not matched on purpose.
_INNER_BLANKS = re.compile(r"(?<=\S)[ \t]+(?=\S)")
_TRAILING_BLANKS = re.compile(r"[ \t]+(?=\n|\Z)")
_EXCESS_NEWLINES = re.compile(r"\n{3,}")


def _inner_blank_edits(text: str) -> list[Edit]:
    """Pass A: 'one   two' -> 'one two'; a lone tab between words becomes a space."""
    return [
        (m.start(), m.end(), " ")
        for m in _INNER_BLANKS.finditer(text)
        if m.group() != " "  # already a single space: nothing to do
    ]


def _trailing_blank_edits(text: str) -> list[Edit]:
    """Pass B: delete spaces/tabs at the end of lines (and of the text)."""
    return [(m.start(), m.end(), "") for m in _TRAILING_BLANKS.finditer(text)]


def _excess_newline_edits(text: str) -> list[Edit]:
    """Pass C: 3+ newlines -> 2 (one blank line). Exactly 2 is a paragraph break: kept."""
    return [(m.start() + 2, m.end(), "") for m in _EXCESS_NEWLINES.finditer(text)]


def _edge_edits(text: str) -> list[Edit]:
    """Pass D: strip leading and trailing whitespace of the whole text."""
    lead = re.match(r"\s+", text)
    tail = re.search(r"\s+\Z", text)
    if lead and tail and lead.end() >= tail.start():  # the text is nothing but whitespace
        return [(0, len(text), "")]
    edits: list[Edit] = []
    if lead:
        edits.append((0, lead.end(), ""))
    if tail:
        edits.append((tail.start(), len(text), ""))
    return edits


def normalize_whitespace(source: NormalizedText) -> NormalizedText:
    """Step 2c. Four passes, each chained through ``apply_edits``.

    Single newlines are never touched here: whether one is a hard-wrapped line or a
    real line break is decided by the reflow step (2d), not by whitespace rules.
    Assumes step 2b ran first (so no-break spaces / form feeds are already plain).
    """
    result = source
    for find_edits in (
        _inner_blank_edits,      # A
        _trailing_blank_edits,   # B  (must precede C: whitespace-only lines become empty)
        _excess_newline_edits,   # C
        _edge_edits,             # D
    ):
        result = result.apply_edits(find_edits(result.text))
    return result


# --------------------------------------------------------------------------- #
# Step 2d: reflow of hard-wrapped lines (mainly PDF)
# --------------------------------------------------------------------------- #
# A single newline = a "\n" with no other newline right before or after it.
_SINGLE_NEWLINE = re.compile(r"(?<!\n)\n(?!\n)")
# Lines that begin a heading, list item, quote, code fence or table row.
_STRUCTURAL_START = re.compile(r"^(?:#{1,6}[ \t]|[-*+][ \t]|\d{1,3}[.)][ \t]|>|```|~~~|\|)")
_SENTENCE_END = ".!?:"
_CLOSING_MARKS = "\"')]}"

MIN_EVIDENCE = 5          # fewer single newlines than this -> no proof of hard wrapping
FULL_WIDTH_RATIO = 0.7    # a line must reach 70% of the typical wrap width to be joined


@dataclass(frozen=True)
class ReflowDecision:
    """What we decided for one single newline, and why (kept for teaching/debugging)."""

    position: int       # index of the "\n" in the text
    line_before: str
    line_after: str
    join: bool
    reason: str


def _is_structural(line: str) -> bool:
    """Heading, list item, quote, fence, table row, or an indented line (code / nesting)."""
    return line[:1] in (" ", "\t") or bool(_STRUCTURAL_START.match(line))


def _ends_sentence(line: str) -> bool:
    return line.rstrip(_CLOSING_MARKS)[-1:] in _SENTENCE_END and bool(line.rstrip(_CLOSING_MARKS))


def _newline_spots(text: str) -> list[tuple[int, str, str]]:
    """For every single newline: (its index, the line before it, the line after it)."""
    spots = []
    for m in _SINGLE_NEWLINE.finditer(text):
        i = m.start()
        line_before = text[text.rfind("\n", 0, i) + 1 : i]
        next_newline = text.find("\n", i + 1)
        line_after = text[i + 1 : len(text) if next_newline == -1 else next_newline]
        spots.append((i, line_before, line_after))
    return spots


def estimate_wrap_width(text: str) -> int | None:
    """Typical hard-wrap width: the 90th percentile of the lengths of lines that are
    followed by a single newline. ``None`` if there is too little evidence."""
    spots = _newline_spots(text)
    if len(spots) < MIN_EVIDENCE:
        return None
    lengths = sorted(len(before) for _, before, _ in spots)
    return lengths[int(0.9 * (len(lengths) - 1))]


def reflow_decisions(text: str) -> list[ReflowDecision]:
    """Decide, for every single newline, whether it is a wrap (join) or real (keep)."""
    spots = _newline_spots(text)
    width = estimate_wrap_width(text)
    decisions: list[ReflowDecision] = []
    for i, before, after in spots:
        if width is None:
            join, reason = False, "kept: not enough evidence of hard wrapping"
        elif not before or not after:
            join, reason = False, "kept: empty line next to it"
        elif _is_structural(before) or _is_structural(after):
            join, reason = False, "kept: heading / list / quote / code / table / indented"
        elif before.endswith("-"):
            join, reason = False, "kept: line ends with a hyphen (step 2e decides)"
        elif _ends_sentence(before):
            join, reason = False, "kept: line ends a sentence (maybe a paragraph end)"
        elif len(before) < FULL_WIDTH_RATIO * width:
            join, reason = False, "kept: short line (title, heading or paragraph end)"
        else:
            join, reason = True, "JOIN: full-width line that stops mid-sentence"
        decisions.append(ReflowDecision(i, before, after, join, reason))
    return decisions


def reflow_hard_wrapped_lines(source: NormalizedText) -> NormalizedText:
    """Step 2d. Replace each newline judged to be a layout wrap by one space.

    Conservative by design: when unsure the newline is kept. Lines ending in a hyphen
    are left for step 2e. Assumes steps 2b and 2c ran first (so lines carry no
    trailing spaces and blank lines are exactly ``\\n\\n``).
    """
    edits = [(d.position, d.position + 1, " ") for d in reflow_decisions(source.text) if d.join]
    return source.apply_edits(edits)


# --------------------------------------------------------------------------- #
# Step 2e: hyphenation repair
# --------------------------------------------------------------------------- #
# "Letters" in any script: \w minus digits and underscore.
_LETTERS = r"[^\W\d_]+"
_HYPHENATED_WORD = rf"{_LETTERS}(?:-{_LETTERS})*"      # 'stellar', 'stellar-mass', 'well-known-fact'
_WORD_TOKEN = re.compile(_HYPHENATED_WORD)
_FRAGMENT_BEFORE = re.compile(rf"({_HYPHENATED_WORD})-$")   # last word of a line that ends in '-'
_FRAGMENT_AFTER = re.compile(rf"^{_HYPHENATED_WORD}")       # first word of the next line


def build_vocabulary(text: str) -> frozenset[str]:
    """Lower-cased words (and hyphenated words) found in ``text``: our evidence of
    which spellings really exist."""
    return frozenset(_WORD_TOKEN.findall(text.lower()))


@dataclass(frozen=True)
class HyphenDecision:
    """What we decided for one line that ends in ``letter-``, and why."""

    position: int    # index of the "-" (the "\n" is at position + 1)
    left: str        # word piece before the hyphen, e.g. "gravita"
    right: str       # word piece after the line break, e.g. "tional"
    action: str      # "join" = delete "-\n";  "keep_hyphen" = delete only "\n";  "leave" = untouched
    reason: str


def hyphenation_decisions(text: str, known_words: Collection[str] = ()) -> list[HyphenDecision]:
    """Decide, for every line ending in ``<letter or digit>-``, what to do.

    ``known_words`` adds outside evidence (e.g. words from other documents) to the
    document's own vocabulary. Matching is case-insensitive.
    """
    vocabulary = build_vocabulary(text) | {w.lower() for w in known_words}
    decisions: list[HyphenDecision] = []
    for newline_index, before, after in _newline_spots(text):
        if not before.endswith("-") or len(before) < 2 or not before[-2].isalnum():
            continue  # not word-hyphen-newline (e.g. a dash " -", "--", or a bullet "-")
        position = newline_index - 1
        match = _FRAGMENT_BEFORE.search(before)
        left = match.group(1) if match else ""

        def decide(action: str, reason: str, right: str = "") -> None:
            decisions.append(HyphenDecision(position, left, right, action, reason))

        if not after or _is_structural(before) or _is_structural(after):
            decide("leave", "heading / list / indented line next to it")
        elif after[0].isdigit() or before[-2].isdigit():
            decide("keep_hyphen", "number next to the hyphen (e.g. COVID-19, 1990-2000)", after.split()[0])
        elif not (fragment := _FRAGMENT_AFTER.match(after)):
            decide("leave", "next line does not start with a word")
        else:
            right = fragment.group()
            joined, hyphenated = (left + right).lower(), f"{left}-{right}".lower()
            if after[0].isupper():
                decide("keep_hyphen", "next word is capitalised (compound with a name)", right)
            elif hyphenated in vocabulary:
                decide("keep_hyphen", "the hyphenated form exists elsewhere", right)
            elif joined in vocabulary:
                decide("join", "the joined word exists elsewhere", right)
            else:
                decide("keep_hyphen", "no evidence: keeping the hyphen (never invent a word)", right)
    return decisions


def repair_hyphenation(source: NormalizedText, known_words: Collection[str] = ()) -> NormalizedText:
    """Step 2e. Remove the line break after a line-end hyphen, and the hyphen too
    only when evidence says the joined word exists. Run AFTER step 2d."""
    edits: list[Edit] = []
    for d in hyphenation_decisions(source.text, known_words):
        if d.action == "join":
            edits.append((d.position, d.position + 2, ""))        # delete "-\n"
        elif d.action == "keep_hyphen":
            edits.append((d.position + 1, d.position + 2, ""))    # delete "\n" only
    return source.apply_edits(edits)


# --------------------------------------------------------------------------- #
# Step 2f (part 1): Markdown markup removal, for the RETRIEVAL text only
# --------------------------------------------------------------------------- #
# Each pattern captures the visible text in one group; only the markers around it are
# deleted, so offsets inside the text stay exact. Conservative on purpose: `_` inside
# a word (snake_case) and lone `*` (bullets, multiplication) are never treated as emphasis.
_MD_IMAGE = re.compile(r"!\[([^\]\n]*)\]\([^)\n]*\)")                       # ![alt](url)  -> alt
_MD_LINK = re.compile(r"\[([^\]\n]+)\]\([^)\n]*\)")                         # [text](url)  -> text
_MD_CODE_SPAN = re.compile(r"`([^`\n]+)`")                                  # `code`       -> code
_MD_BOLD = re.compile(r"(?<!\w)(\*\*|__)(?=\S)(.+?)(?<=\S)\1(?!\w)")        # **x** / __x__ -> x
_MD_ITALIC = re.compile(r"(?<![\w*])([*_])(?=[^\s*_])([^*_\n]+?)(?<=[^\s*_])\1(?![\w*])")  # *x* / _x_
_MD_STRIKE = re.compile(r"(~~)(?=\S)(.+?)(?<=\S)\1")                        # ~~x~~        -> x
_MD_HEADING_MARKER = re.compile(r"^#{1,6}[ \t]+", re.MULTILINE)            # "## " at line start


def _marker_edits(pattern: re.Pattern[str], text: str, inner_group: int) -> list[Edit]:
    """Delete everything a match contains EXCEPT its inner (visible) group."""
    edits: list[Edit] = []
    for m in pattern.finditer(text):
        inner_start, inner_end = m.span(inner_group)
        if inner_start == inner_end:                       # nothing visible inside: drop it all
            edits.append((m.start(), m.end(), ""))
            continue
        if m.start() < inner_start:
            edits.append((m.start(), inner_start, ""))     # opening marker
        if inner_end < m.end():
            edits.append((inner_end, m.end(), ""))         # closing marker (and url)
    return edits


def strip_markdown_markup(source: NormalizedText) -> NormalizedText:
    """Remove inline Markdown markers and heading hashes; keep the words.

    Lists (``-``, ``1.``), quotes (``>``) and tables (``|``) are kept: they carry
    structure that chunking may use. NOT safe for text containing fenced code blocks
    (code would be rewritten): the pipeline skips this step for such documents.
    """
    result = source
    for pattern, group in (
        (_MD_IMAGE, 1), (_MD_LINK, 1), (_MD_CODE_SPAN, 1),
        (_MD_BOLD, 2), (_MD_ITALIC, 2), (_MD_STRIKE, 2),
    ):
        result = result.apply_edits(_marker_edits(pattern, result.text, group))
    heading_edits = [(m.start(), m.end(), "") for m in _MD_HEADING_MARKER.finditer(result.text)]
    return result.apply_edits(heading_edits)
