import pytest

from rag_lab.config import DEFAULT_OPENROUTER_MODEL, Settings
from rag_lab.llm.answer import NO_SOURCES_TEXT, REFUSAL_TEXT
from rag_lab.llm.factory import make_generator
from rag_lab.llm.openrouter import OpenRouterError, generate_answer_openrouter
from rag_lab.types import Chunk

CHUNKS = [
    Chunk(chunk_id="mars::0", doc_id="mars", text="Jezero was chosen for its river delta.", start_char=0, end_char=38),
    Chunk(chunk_id="apollo::2", doc_id="apollo", text="Apollo 11 landed in 1969.", start_char=0, end_char=25),
]
SECRET = "sk-or-SECRET-KEY-123"


def good(text="Because of its river delta [1].", finish="stop", **extra):
    message = {"role": "assistant", "content": text, **extra}
    return 200, {
        "model": "google/fake-model",
        "choices": [{"message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 260, "completion_tokens": 11},
    }


class FakeResponse:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        if self._body is None:
            raise ValueError("not json")
        return self._body


class FakeHttp:
    """Stands in for httpx.Client: replays a list of (status, body) replies, records requests."""

    def __init__(self, *replies):
        self.replies = list(replies) or [good()]
        self.calls = []

    def post(self, path, json):
        self.calls.append((path, json))
        status, body = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        return FakeResponse(status, body)


def ask(http, chunks=CHUNKS, sleeps=None, **kw):
    sleeps = sleeps if sleeps is not None else []
    return generate_answer_openrouter(http, "Why Jezero?", chunks, model="google/fake-model", sleep=sleeps.append, **kw)


# ------------------------------------------------ the request
def test_request_has_model_system_and_user_messages_and_no_temperature():
    http = FakeHttp()
    ask(http, max_tokens=777)
    (path, payload), = http.calls
    assert path == "/chat/completions"
    assert payload["model"] == "google/fake-model" and payload["max_tokens"] == 777
    assert [m["role"] for m in payload["messages"]] == ["system", "user"]
    assert "numbered sources" in payload["messages"][0]["content"] and "Why Jezero?" in payload["messages"][1]["content"]
    assert not {"temperature", "top_p", "top_k"} & payload.keys()


# ------------------------------------------------ the reply
def test_text_citations_usage_and_model_are_read():
    answer = ask(FakeHttp(good()))
    assert answer.text == "Because of its river delta [1]."
    assert [s.chunk_id for s in answer.cited] == ["mars::0"]
    assert (answer.input_tokens, answer.output_tokens, answer.model) == (260, 11, "google/fake-model")
    assert answer.called_model and not (answer.unknown or answer.refused or answer.truncated)


def test_dont_know_is_flagged():
    assert ask(FakeHttp(good("I don't know based on the provided sources."))).unknown


def test_finish_reason_length_means_cut_off():
    assert ask(FakeHttp(good("Because of its", finish="length"))).truncated


def test_content_filter_and_refusal_field_mean_refused():
    assert ask(FakeHttp(good("", finish="content_filter"))).refused
    answer = ask(FakeHttp(good(None, refusal="I can't help with that")))
    assert answer.refused and answer.text == REFUSAL_TEXT


def test_missing_content_gives_empty_text_not_a_crash():
    answer = ask(FakeHttp(good(None)))
    assert answer.text == "" and not answer.refused


# ------------------------------------------------ errors and retries
def test_bad_key_is_raised_at_once_without_retrying_and_without_the_key():
    http = FakeHttp((401, {"error": {"code": 401, "message": "No auth credentials found"}}))
    sleeps = []
    with pytest.raises(OpenRouterError) as info:
        ask(http, sleeps=sleeps)
    assert info.value.status == 401 and "No auth credentials" in str(info.value)
    assert len(http.calls) == 1 and sleeps == []
    assert SECRET not in str(info.value)


@pytest.mark.parametrize("status", [402, 404, 400])
def test_other_client_errors_are_not_retried(status):
    http = FakeHttp((status, {"error": {"message": "nope"}}))
    with pytest.raises(OpenRouterError) as info:
        ask(http)
    assert info.value.status == status and len(http.calls) == 1


@pytest.mark.parametrize("status", [429, 502, 503])
def test_temporary_errors_are_retried_with_growing_pauses_then_succeed(status):
    http = FakeHttp((status, {"error": {"message": "busy"}}), (status, None), good())
    sleeps = []
    answer = ask(http, sleeps=sleeps)
    assert answer.text.startswith("Because") and len(http.calls) == 3
    assert sleeps == [1, 2]


def test_gives_up_after_the_last_attempt():
    http = FakeHttp((503, {"error": {"message": "busy"}}))
    sleeps = []
    with pytest.raises(OpenRouterError) as info:
        ask(http, sleeps=sleeps, attempts=3)
    assert info.value.status == 503 and len(http.calls) == 3 and sleeps == [1, 2]


def test_error_inside_a_200_reply_is_treated_like_that_status():
    http = FakeHttp((200, {"error": {"code": 503, "message": "upstream overloaded"}}), good())
    assert ask(http).text.startswith("Because") and len(http.calls) == 2


def test_a_200_reply_without_choices_is_an_error():
    with pytest.raises(OpenRouterError):
        ask(FakeHttp((200, {"choices": []})))


# ------------------------------------------------ edge cases
def test_no_chunks_means_no_request():
    http = FakeHttp()
    answer = ask(http, chunks=[])
    assert http.calls == [] and answer.text == NO_SOURCES_TEXT and not answer.called_model


def test_bad_max_tokens_rejected():
    with pytest.raises(ValueError):
        ask(FakeHttp(), max_tokens=0)


# ------------------------------------------------ factory
def env(**kw):
    return Settings.from_env(kw)


def test_factory_reports_a_missing_openrouter_key_clearly():
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        make_generator(env())


def test_factory_builds_a_generator_without_any_network_call():
    label, generate = make_generator(env(OPENROUTER_API_KEY=SECRET, OPENROUTER_MODEL="google/x"))
    assert label == "openrouter:google/x" and callable(generate)
    assert SECRET not in label


def test_factory_uses_the_default_model_when_none_is_set():
    label, _ = make_generator(env(OPENROUTER_API_KEY="k"))
    assert label == f"openrouter:{DEFAULT_OPENROUTER_MODEL}"
