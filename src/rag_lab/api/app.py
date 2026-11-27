"""Step 9b: the HTTP API the frontend talks to.

    GET  /api/health    is the server up, how many chunks are indexed, is answering configured
    POST /api/search    ranked passages for a question (free, fast, no model call)
    POST /api/answer    the same passages plus a written answer with citations (calls the LLM)

Search and answer are separate on purpose: search costs nothing, answering costs money, so the
page can show results first and the developer can test it with no API key.

Safety choices: the question is length-limited, ``k`` is capped, errors from the LLM provider are
logged on the server and replaced by a generic message (they can contain request details),
API keys never leave the server, and CORS is off unless origins are listed explicitly.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from rag_lab.api.service import SearchResult, SearchService, build_service
from rag_lab.config import get_settings
from rag_lab.llm import Answer, make_generator
from rag_lab.types import Chunk

log = logging.getLogger("rag_lab.api")

MAX_QUESTION_CHARS = 500
MAX_K = 20
Generator = Callable[[str, Sequence[Chunk]], Answer]


# ------------------------------------------------ request and response shapes
class SearchRequest(BaseModel):
    """What the browser sends: the question, which search method, and how many results (k)."""
    question: str = Field(max_length=MAX_QUESTION_CHARS)
    method: Literal["vector", "bm25", "hybrid"] = "hybrid"
    k: int = Field(default=5, ge=1, le=MAX_K)

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class ResultOut(BaseModel):
    """One search result as the page needs it: rank, score, passage text and where it came from."""
    rank: int
    score: float
    relative_score: float            # 0..1, ready for the score bar
    chunk_id: str
    doc_id: str
    title: str
    source: str                      # file name
    file_type: str | None
    page: int | None                 # not tracked per chunk yet, so null for now
    text: str


class SearchResponse(BaseModel):
    """Reply of /api/search: the results and how long the search took."""
    question: str
    method: str
    k: int
    took_ms: float
    results: list[ResultOut]


class CitationOut(BaseModel):
    """Links an [n] marker in the answer text to the passage it points at."""
    number: int                      # the [n] in the answer text
    chunk_id: str
    doc_id: str


class AnswerOut(BaseModel):
    """The written answer, its flags (unknown, refused, truncated) and token usage."""
    text: str
    citations: list[CitationOut]
    unknown: bool                    # the sources did not contain the answer
    refused: bool
    truncated: bool
    called_model: bool
    model: str | None
    input_tokens: int
    output_tokens: int


class AnswerResponse(SearchResponse):
    """Reply of /api/answer: the search results plus the written answer."""
    answer: AnswerOut


def _result_out(result: SearchResult) -> ResultOut:
    meta = result.chunk.metadata
    return ResultOut(
        rank=result.rank,
        score=result.score,
        relative_score=result.relative,
        chunk_id=result.chunk.chunk_id,
        doc_id=result.chunk.doc_id,
        title=str(meta.get("title") or result.chunk.doc_id),
        source=str(meta.get("source") or result.chunk.doc_id),
        file_type=meta.get("file_type"),
        page=meta.get("page"),
        text=result.chunk.text,
    )


# ------------------------------------------------ the app
def create_app(
    service: SearchService | None = None,
    generator: Generator | None = None,
    generator_label: str | None = None,
    static_dir: Path | None = None,
    cors_origins: Sequence[str] = (),
) -> FastAPI:
    """Build the API. In tests, pass a ready ``service`` and ``generator``. With no ``service``,
    the real corpus is indexed once at startup and the generator is chosen from the settings."""

    state: dict = {"service": service, "generator": generator, "label": generator_label}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if state["service"] is None:
            settings = get_settings()
            state["service"] = build_service(settings)
            try:
                state["label"], state["generator"] = make_generator(settings)
            except Exception as exc:                       # missing key or SDK problem: search still works
                log.warning("answer generation is not available: %s", exc)
        yield

    app = FastAPI(title="RAG Search Lab", lifespan=lifespan)
    if cors_origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=list(cors_origins), allow_methods=["GET", "POST"], allow_headers=["Content-Type"]
        )

    def run_search(body: SearchRequest) -> tuple[list[SearchResult], float]:
        """Run one search and measure how long it took, in milliseconds."""
        started = time.perf_counter()
        results = state["service"].search(body.question, body.method, body.k)
        return results, (time.perf_counter() - started) * 1000

    @app.get("/api/health")
    def health() -> dict:
        """Is the server up, how many chunks are indexed, and is answering configured?"""
        return {
            "status": "ok",
            "chunks": len(state["service"].chunks),
            "answers_available": state["generator"] is not None,
            "model": state["label"],
        }

    @app.post("/api/search", response_model=SearchResponse)
    def search(body: SearchRequest) -> SearchResponse:
        """Free and fast: return the best passages. No model call is made."""
        results, took = run_search(body)
        return SearchResponse(
            question=body.question, method=body.method, k=body.k, took_ms=round(took, 1),
            results=[_result_out(r) for r in results],
        )

    @app.post("/api/answer", response_model=AnswerResponse)
    def answer(body: SearchRequest) -> AnswerResponse:
        """Search, then ask the model to write an answer from the passages. This costs tokens."""
        if state["generator"] is None:
            raise HTTPException(503, "Answers are not configured. Set OPENROUTER_API_KEY in .env.")
        results, took = run_search(body)
        try:
            reply = state["generator"](body.question, [r.chunk for r in results])
        except Exception:
            log.exception("answer generation failed")          # details stay in the server log
            raise HTTPException(502, "The answer service failed. Please try again.") from None
        return AnswerResponse(
            question=body.question, method=body.method, k=body.k, took_ms=round(took, 1),
            results=[_result_out(r) for r in results],
            answer=AnswerOut(
                text=reply.text,
                citations=[CitationOut(number=s.number, chunk_id=s.chunk_id, doc_id=s.doc_id) for s in reply.cited],
                unknown=reply.unknown, refused=reply.refused, truncated=reply.truncated,
                called_model=reply.called_model, model=reply.model,
                input_tokens=reply.input_tokens, output_tokens=reply.output_tokens,
            ),
        )

    if static_dir is not None and static_dir.is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")   # last: API routes win
    return app
