"""Step 4c: a thin wrapper around the real embedding model.

``Embedder.embed_texts`` turns a list of strings into a 2-D numpy array, one row per text
(384 numbers per row for all-MiniLM-L6-v2). The model is loaded lazily, on the first call,
because loading takes seconds and importing this module should stay instant.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from rag_lab.config import DEFAULT_EMBEDDING_MODEL


def _load_sentence_transformer(name: str) -> Any:
    from sentence_transformers import SentenceTransformer   # heavy import, only when needed

    return SentenceTransformer(name)


class Embedder:
    """Embeds text with a sentence-transformers model.

    ``loader`` builds the model from its name. It exists so tests can pass in a fake model
    instead of downloading the real one.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        loader: Callable[[str], Any] = _load_sentence_transformer,
        batch_size: int = 32,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.model_name = model_name
        self.batch_size = batch_size
        self._loader = loader
        self._model: Any = None

    @property
    def model(self) -> Any:
        """The embedding model, loaded on first use so importing this module stays fast."""
        if self._model is None:
            self._model = self._loader(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        """How many numbers each vector has (384 for all-MiniLM-L6-v2)."""
        # The method was renamed in newer sentence-transformers; support both.
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        return int(get_dim())

    def embed_texts(self, texts: Sequence[str], normalize: bool = True) -> np.ndarray:
        """Return an array of shape ``(len(texts), dimension)``, dtype float32.

        With ``normalize=True`` every row has length 1, so cosine similarity between two
        rows is just their dot product.
        """
        if isinstance(texts, str):
            raise TypeError("embed_texts expects a list of strings, not a single string")
        if not all(isinstance(t, str) for t in texts):
            raise TypeError("every item passed to embed_texts must be a string")
        if len(texts) == 0:
            return np.empty((0, self.dimension), dtype=np.float32)
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_text(self, text: str, normalize: bool = True) -> np.ndarray:
        """Embed one string; returns a 1-D array of ``dimension`` numbers."""
        return self.embed_texts([text], normalize=normalize)[0]
