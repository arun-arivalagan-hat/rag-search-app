"""Source files must not contain invisible or look-alike characters.

A literal soft hyphen, zero-width space, no-break space, curly quote or 'fi' ligature
is invisible or indistinguishable from a plain character, so a reader cannot tell what a
cleaning rule or a test actually says. Write such characters as visible escapes
(backslash-u followed by four hex digits) instead.

Gotcha: this project's file-writing tool silently turns those escapes into the real
characters when they appear in written content. Create the escape text another way
(for example `chr(92) + "u00ad"` in a script) and re-run this test afterwards.
"""

import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = sorted(
    p for folder in ("src", "tests", "scripts") for p in (ROOT / folder).rglob("*.py")
)


def is_invisible_or_lookalike(ch: str) -> bool:
    cp = ord(ch)
    return cp > 127 and (
        unicodedata.category(ch) in {"Cf", "Zs", "Zl", "Zp", "Cc", "Mn"}
        or 0x2010 <= cp <= 0x201F          # hyphens, dashes, curly quotes
        or 0xFB00 <= cp <= 0xFB06          # f-ligatures
    )


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_invisible_or_lookalike_characters_in_source(path):
    offenders = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        for ch in line:
            if is_invisible_or_lookalike(ch):
                offenders.append(f"line {line_number}: U+{ord(ch):04X} {unicodedata.name(ch, '?')}")
    assert not offenders, "\n".join(offenders[:10])


def test_the_guard_itself_detects_the_problem_characters():
    # Built with chr() so this test file stays clean.
    for cp in (0x00AD, 0x200B, 0x00A0, 0xFB01, 0x2019, 0x0301, 0xFEFF):
        assert is_invisible_or_lookalike(chr(cp))
    for ch in ("a", " ", "e", "-", "'", chr(0xE9), chr(0xB2)):
        assert not is_invisible_or_lookalike(ch)
