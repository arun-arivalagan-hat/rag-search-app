"""Step 5b: word statistics for keyword search (term frequency and inverse document frequency).

BM25 (step 5c) is built from two ingredients, both made here:

* term frequency (TF): how many times a word appears in ONE chunk;
* document frequency (DF) and inverse document frequency (IDF): in how many chunks a word
  appears, and from that how RARE (so how informative) the word is.

Here a "document" means one chunk, because chunks are what we search.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence


def term_frequencies(tokens: Iterable[str]) -> dict[str, int]:
    """Count how many times each word occurs: ``["a", "b", "a"]`` -> ``{"a": 2, "b": 1}``."""
    return dict(Counter(tokens))


def document_frequencies(corpus_tokens: Iterable[Sequence[str]]) -> dict[str, int]:
    """For each word, in how many chunks it appears AT LEAST once.

    A word that appears 10 times in one chunk still counts as 1 chunk.
    """
    counts: Counter[str] = Counter()
    for tokens in corpus_tokens:
        counts.update(set(tokens))          # set(): each chunk counts a word only once
    return dict(counts)


def idf(n_docs: int, doc_freq: int) -> float:
    """How rare a word is: ``ln(1 + (N - n + 0.5) / (n + 0.5))``.

    ``N`` is the number of chunks and ``n`` the number of chunks containing the word.
    The result is always positive; it is largest for a word found in one chunk and close to
    zero for a word found in every chunk. A word never seen (``n = 0``) is allowed and
    scores highest. The ``+ 0.5`` terms avoid dividing by zero, and the ``1 +`` keeps the
    result from going negative (this is the form used by Lucene and Elasticsearch).
    """
    if n_docs < 1:
        raise ValueError("n_docs must be at least 1")
    if not 0 <= doc_freq <= n_docs:
        raise ValueError(f"doc_freq must be between 0 and n_docs ({n_docs}), got {doc_freq}")
    return math.log(1 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))


def idf_table(corpus_tokens: Sequence[Sequence[str]]) -> dict[str, float]:
    """The IDF of every word that appears in the corpus."""
    n_docs = len(corpus_tokens)
    return {word: idf(n_docs, df) for word, df in document_frequencies(corpus_tokens).items()}
