"""Step 5a: turn text into tokens (lower-case words) for keyword search.

BM25 (step 5c) counts words, so before it can count anything both the chunks and the query
must be cut into the same kind of tokens. The rules are deliberately small and visible:

* lower-case everything (``Moon`` and ``moon`` are the same word);
* a token is a run of letters and digits;
* an apostrophe or a dot BETWEEN letters/digits stays inside the token
  (``don't``, ``7.6``, ``u.s``), but one at the end does not (``end.`` -> ``end``);
* everything else (spaces, hyphens, underscores, other punctuation) separates tokens,
  so ``three-way`` becomes ``three`` and ``way``.
"""

from __future__ import annotations

import re

# A plain apostrophe and the curly one, built from code points so the source stays plain ASCII.
_APOSTROPHES = "'" + chr(0x2019)
_JOINERS = "." + _APOSTROPHES

# [^\W_] = a letter or digit in any language (\w minus the underscore).
_TOKEN = re.compile(rf"[^\W_]+(?:[{re.escape(_JOINERS)}][^\W_]+)*")


def tokenize(text: str) -> list[str]:
    """Split ``text`` into lower-case tokens, in order, keeping repeats."""
    return _TOKEN.findall(text.casefold())
