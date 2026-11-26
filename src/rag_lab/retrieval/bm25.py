"""Step 5c: the BM25 score, written by hand.

BM25 rates how well one chunk matches a query. For every query word it adds up

    idf(word) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / average_length))

* ``idf``  : rarity of the word (step 5b), so rare words count more;
* ``tf``   : how many times the word occurs in this chunk (step 5b);
* ``k1``   : saturation. Repeating a word helps less and less. With ``k1 = 0`` the number of
             repeats is ignored; a large ``k1`` makes repeats count almost linearly;
* ``b``    : length fairness. ``b = 1`` fully penalises chunks longer than average, ``b = 0``
             ignores length;
* ``length / average_length`` : how long this chunk is compared with the typical chunk.
"""

from __future__ import annotations

from collections.abc import Sequence

from rag_lab.retrieval.stats import document_frequencies, idf, term_frequencies

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


class BM25:
    """BM25 over a fixed list of tokenised chunks. Chunk ``i`` is ``corpus_tokens[i]``."""

    def __init__(self, corpus_tokens: Sequence[Sequence[str]], k1: float = DEFAULT_K1, b: float = DEFAULT_B) -> None:
        if len(corpus_tokens) == 0:
            raise ValueError("corpus must contain at least one chunk")
        if k1 < 0:
            raise ValueError("k1 must be 0 or more")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        self.k1 = k1
        self.b = b
        self.n_docs = len(corpus_tokens)
        self._term_counts = [term_frequencies(tokens) for tokens in corpus_tokens]
        self._lengths = [len(tokens) for tokens in corpus_tokens]
        self.avg_length = sum(self._lengths) / self.n_docs
        self._idf = {word: idf(self.n_docs, df) for word, df in document_frequencies(corpus_tokens).items()}

    def idf_of(self, word: str) -> float:
        """Rarity of ``word`` in this corpus (a word never seen gets the highest value)."""
        return self._idf.get(word, idf(self.n_docs, 0))

    def _length_factor(self, index: int) -> float:
        """``1 - b + b * length / average_length``: above 1 for long chunks, below 1 for short."""
        if self.avg_length == 0:                 # every chunk is empty; nothing to compare
            return 1.0
        return 1 - self.b + self.b * self._lengths[index] / self.avg_length

    def score(self, query_tokens: Sequence[str], index: int) -> float:
        """BM25 score of chunk ``index`` for the query. A word repeated in the query counts each time."""
        if not 0 <= index < self.n_docs:
            raise IndexError(f"chunk index {index} is out of range (0..{self.n_docs - 1})")
        counts = self._term_counts[index]
        length_factor = self._length_factor(index)
        total = 0.0
        for word in query_tokens:
            tf = counts.get(word, 0)
            if tf == 0:                          # the chunk lacks this word, so it adds nothing
                continue
            total += self.idf_of(word) * tf * (self.k1 + 1) / (tf + self.k1 * length_factor)
        return total

    def scores(self, query_tokens: Sequence[str]) -> list[float]:
        """The BM25 score of every chunk, in chunk order."""
        return [self.score(query_tokens, i) for i in range(self.n_docs)]
