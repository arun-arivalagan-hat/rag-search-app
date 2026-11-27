import logging

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rag_lab.api.app import MAX_K, MAX_QUESTION_CHARS, create_app
from rag_lab.api.service import METHODS, SearchService
from rag_lab.llm import build_answer
from rag_lab.prompts import build_prompt
from rag_lab.types import Chunk

MOON_WORDS = {"moon", "lunar", "selene"}


class FakeEmbedder:
    """Offline stand-in: topic counts (moon-ish, vitamin, sun) plus a tiny constant, normalised."""

    def _vector(self, text):
        words = text.casefold().split()
        v = np.array([sum(w in MOON_WORDS for w in words), words.count("vitamin"), words.count("sun"), 0.1], dtype=np.float32)
        return v / np.linalg.norm(v)

    def embed_texts(self, texts, normalize=True):
        return np.array([self._vector(t) for t in texts], dtype=np.float32)

    def embed_text(self, text, normalize=True):
        return self._vector(text)


TEXTS = ["the moon landing in 1969", "vitamin c and health", "sun and moon and stars", "apollo astronauts walked on the lunar surface"]


def make_chunks():
    meta = {"title": "Space Notes", "source": "space.md", "file_type": "markdown"}
    return [
        Chunk(chunk_id=f"d::{i}", doc_id="d", text=t, start_char=0, end_char=len(t), metadata=meta)
        for i, t in enumerate(TEXTS)
    ]


class RecordingGenerator:
    """A fake answer writer that records what it was given and cites source [1]."""

    def __init__(self, error=None):
        self.received = []
        self.error = error

    def __call__(self, question, chunks):
        if self.error:
            raise self.error
        self.received = list(chunks)
        return build_answer(
            build_prompt(question, chunks), "The moon landing was in 1969 [1].",
            refused=False, truncated=False, model="fake-model", input_tokens=50, output_tokens=9,
        )


@pytest.fixture
def service():
    return SearchService(make_chunks(), FakeEmbedder())


@pytest.fixture
def generator():
    return RecordingGenerator()


@pytest.fixture
def client(service, generator):
    return TestClient(create_app(service, generator, "fake:model"))


def post(client, path, **body):
    body.setdefault("question", "moon landing")
    return client.post(path, json=body)


# ------------------------------------------------ SearchService
@pytest.mark.parametrize("method", METHODS)
def test_service_returns_ranked_results_with_a_top_relative_score_of_one(service, method):
    results = service.search("moon landing", method, 3)
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    assert results[0].relative == 1.0
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)
    assert all(0.0 <= r.relative <= 1.0 for r in results)


def test_service_bm25_with_no_matching_word_returns_nothing(service):
    assert service.search("zebra", "bm25", 3) == []


def test_service_vector_always_returns_k(service):
    assert len(service.search("zebra", "vector", 3)) == 3


@pytest.mark.parametrize("args", [("q", "magic", 3), ("", "hybrid", 3), ("  ", "hybrid", 3), ("q", "hybrid", 0)])
def test_service_rejects_bad_input(service, args):
    with pytest.raises(ValueError):
        service.search(*args)


# ------------------------------------------------ /api/health
def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok", "chunks": 4, "answers_available": True, "model": "fake:model"}


def test_health_says_when_answers_are_not_configured(service):
    body = TestClient(create_app(service)).get("/api/health").json()
    assert body["answers_available"] is False and body["model"] is None


# ------------------------------------------------ /api/search
def test_search_returns_everything_a_result_card_needs(client):
    body = post(client, "/api/search", method="bm25", k=2).json()
    assert body["method"] == "bm25" and body["k"] == 2 and body["took_ms"] >= 0
    first = body["results"][0]
    assert set(first) == {"rank", "score", "relative_score", "chunk_id", "doc_id", "title", "source", "file_type", "page", "text"}
    assert first["rank"] == 1 and first["relative_score"] == 1.0
    assert first["title"] == "Space Notes" and first["source"] == "space.md" and first["page"] is None


def test_search_defaults_to_hybrid_with_five_results(client):
    body = client.post("/api/search", json={"question": "moon"}).json()
    assert body["method"] == "hybrid" and body["k"] == 5 and len(body["results"]) <= 5


def test_search_with_no_match_is_an_empty_list_not_an_error(client):
    response = post(client, "/api/search", question="zebra", method="bm25")
    assert response.status_code == 200 and response.json()["results"] == []


def test_question_is_trimmed(client):
    assert post(client, "/api/search", question="  moon  ").json()["question"] == "moon"


@pytest.mark.parametrize(
    "body",
    [
        {"question": ""},
        {"question": "   "},
        {"question": "x" * (MAX_QUESTION_CHARS + 1)},
        {"question": "moon", "method": "magic"},
        {"question": "moon", "k": 0},
        {"question": "moon", "k": MAX_K + 1},
        {"method": "bm25"},
        {},
    ],
)
def test_bad_requests_are_rejected_with_422(client, body):
    assert client.post("/api/search", json=body).status_code == 422


def test_search_never_calls_the_answer_generator(client, generator):
    post(client, "/api/search")
    assert generator.received == []


# ------------------------------------------------ /api/answer
def test_answer_returns_results_and_an_answer_with_citations(client):
    body = post(client, "/api/answer", k=3).json()
    assert len(body["results"]) == 3
    answer = body["answer"]
    assert answer["text"] == "The moon landing was in 1969 [1]."
    assert [c["number"] for c in answer["citations"]] == [1]
    assert answer["citations"][0]["chunk_id"] == body["results"][0]["chunk_id"]       # [1] is the top result
    assert (answer["called_model"], answer["unknown"], answer["refused"], answer["truncated"]) == (True, False, False, False)
    assert (answer["model"], answer["input_tokens"], answer["output_tokens"]) == ("fake-model", 50, 9)


def test_answer_is_written_from_exactly_the_results_that_are_shown(client, generator):
    body = post(client, "/api/answer", k=2).json()
    assert [c.chunk_id for c in generator.received] == [r["chunk_id"] for r in body["results"]]


def test_answer_without_a_configured_generator_is_503(service):
    response = TestClient(create_app(service)).post("/api/answer", json={"question": "moon"})
    assert response.status_code == 503 and "not configured" in response.json()["detail"]


def test_provider_failure_is_502_and_does_not_leak_the_error(service, caplog):
    secret = "request failed for key sk-SECRET-123"
    client = TestClient(create_app(service, RecordingGenerator(error=RuntimeError(secret)), "fake:model"))
    with caplog.at_level(logging.ERROR, logger="rag_lab.api"):
        response = client.post("/api/answer", json={"question": "moon"})
    assert response.status_code == 502
    assert "SECRET" not in response.text and "failed" in response.json()["detail"]
    assert "SECRET" in caplog.text                                  # the detail is kept in the server log


def test_answer_with_no_matching_passages_is_still_valid(service):
    class Unknown:
        def __call__(self, question, chunks):
            from rag_lab.llm import no_sources_answer
            return no_sources_answer()

    client = TestClient(create_app(service, Unknown(), "fake:model"))
    body = client.post("/api/answer", json={"question": "zebra", "method": "bm25"}).json()
    assert body["results"] == [] and body["answer"]["unknown"] and not body["answer"]["called_model"]


# ------------------------------------------------ static frontend and CORS
def test_frontend_is_served_but_api_routes_win(service, tmp_path):
    (tmp_path / "index.html").write_text("<h1>page</h1>")
    client = TestClient(create_app(service, static_dir=tmp_path))
    assert "<h1>page</h1>" in client.get("/").text
    assert client.get("/api/health").json()["status"] == "ok"


def test_unknown_api_path_is_404(client):
    assert client.get("/api/nothing").status_code == 404


def test_cors_is_off_by_default(client):
    response = client.get("/api/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in response.headers


def test_cors_allows_only_listed_origins(service):
    client = TestClient(create_app(service, cors_origins=["http://127.0.0.1:5500"]))
    ok = client.get("/api/health", headers={"Origin": "http://127.0.0.1:5500"})
    other = client.get("/api/health", headers={"Origin": "http://evil.example"})
    assert ok.headers["access-control-allow-origin"] == "http://127.0.0.1:5500"
    assert "access-control-allow-origin" not in other.headers
