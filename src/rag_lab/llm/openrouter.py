"""Step 8f: the same grounded answer, written through OpenRouter.

OpenRouter is one web address (and one key) in front of many models. It speaks the common
"chat completions" format: send a list of messages, get ``choices[0].message.content`` back. The
prompt (step 8a), citation parsing and the ``Answer`` type are shared with the other providers.

The HTTP client is passed in (``httpx.Client`` in real use, a fake in tests). Temporary errors
(429, 500, 502, 503, 504) are retried with growing pauses. Real mistakes (a bad key, no credit,
an unknown model) are raised at once. The error text keeps the status and OpenRouter's message
but never the request, so the key cannot leak into a log. Temperature is deliberately not sent:
some current models reject it, and the model default is fine for answering from sources.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

from rag_lab.llm.answer import Answer, build_answer, no_sources_answer
from rag_lab.prompts import build_prompt
from rag_lab.types import Chunk

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
RETRY_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
DEFAULT_ATTEMPTS = 3


class OpenRouterError(RuntimeError):
    """OpenRouter answered with an error (``status`` is its HTTP status code)."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"OpenRouter error {status}: {message}")
        self.status = status


def _error_from(status: int, body: Any) -> OpenRouterError:
    """Build an error from a failed reply, keeping only OpenRouter's own message."""
    message = ""
    if isinstance(body, dict):
        err = body.get("error")
        message = str(err.get("message", "")) if isinstance(err, dict) else str(err or "")
    return OpenRouterError(status, message[:300] or "no details")


def post_chat(client: Any, payload: dict, *, attempts: int = DEFAULT_ATTEMPTS, sleep: Callable[[float], None] = time.sleep) -> dict:
    """POST one chat request, retrying temporary failures. Returns the parsed reply."""
    for attempt in range(1, attempts + 1):
        response = client.post("/chat/completions", json=payload)
        try:
            body = response.json()
        except ValueError:
            body = None
        status = response.status_code
        # OpenRouter can also report an upstream failure inside a 200 reply, as {"error": {...}}.
        if status == 200 and isinstance(body, dict) and body.get("error") and not body.get("choices"):
            code = body["error"].get("code") if isinstance(body["error"], dict) else None
            status = code if isinstance(code, int) else 502
        if status == 200:
            if not isinstance(body, dict) or not body.get("choices"):
                raise OpenRouterError(502, "reply had no choices")
            return body
        if status in RETRY_STATUS_CODES and attempt < attempts:
            sleep(2 ** (attempt - 1))                      # 1 s, 2 s, 4 s ...
            continue
        raise _error_from(status, body)
    raise OpenRouterError(502, "no attempts were made")     # only if attempts < 1


def generate_answer_openrouter(
    client: Any,
    question: str,
    chunks: Sequence[Chunk],
    *,
    model: str,
    max_tokens: int = 4096,
    max_context_chars: int | None = None,
    attempts: int = DEFAULT_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
) -> Answer:
    """Answer ``question`` from ``chunks`` (best first) using ``model`` through OpenRouter.

    ``max_tokens`` also pays for any thinking the model does, so keep it generous.
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    if len(chunks) == 0:
        return no_sources_answer()

    prompt = build_prompt(question, chunks, max_context_chars=max_context_chars)
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ],
    }
    body = post_chat(client, payload, attempts=attempts, sleep=sleep)

    choice = body["choices"][0]
    message = choice.get("message") or {}
    finish = str(choice.get("finish_reason") or "")
    usage = body.get("usage") or {}
    return build_answer(
        prompt,
        message.get("content") or "",
        refused=finish == "content_filter" or bool(message.get("refusal")),
        truncated=finish == "length",
        model=body.get("model") or model,
        input_tokens=int(usage.get("prompt_tokens") or 0),
        output_tokens=int(usage.get("completion_tokens") or 0),
    )
