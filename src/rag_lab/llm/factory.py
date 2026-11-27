"""Step 8e: build the one ``generate(question, chunks)`` function the rest of the app uses.

Callers (the demo, the API) never touch the HTTP details:

    label, generate = make_generator(get_settings())
    answer = generate("Why Jezero?", chunks)

A missing key fails here with a message that says what to set, before any request is made.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from rag_lab.config import Settings
from rag_lab.llm.answer import Answer
from rag_lab.llm.openrouter import OPENROUTER_BASE_URL, generate_answer_openrouter
from rag_lab.types import Chunk

Generator = Callable[[str, Sequence[Chunk]], Answer]


def make_generator(settings: Settings, max_tokens: int = 4096) -> tuple[str, Generator]:
    """Return ``(label, generate)`` for the OpenRouter model in ``settings``."""
    key = settings.require_openrouter_key()
    import httpx

    http = httpx.Client(
        base_url=OPENROUTER_BASE_URL,
        headers={"Authorization": f"Bearer {key}", "X-Title": "RAG Search Lab"},
        timeout=httpx.Timeout(20.0, connect=10.0),
    )
    model = settings.openrouter_model

    def generate(question: str, chunks: Sequence[Chunk]) -> Answer:
        return generate_answer_openrouter(http, question, chunks, model=model, max_tokens=max_tokens)

    return f"openrouter:{model}", generate
