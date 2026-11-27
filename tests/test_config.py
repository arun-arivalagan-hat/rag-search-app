from pathlib import Path

import pytest

from rag_lab.config import DEFAULT_OPENROUTER_MODEL, Settings


def test_defaults_when_env_empty():
    s = Settings.from_env({})
    assert s.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert s.openrouter_model == DEFAULT_OPENROUTER_MODEL
    assert s.openrouter_api_key is None


def test_models_are_env_configurable():
    s = Settings.from_env({"EMBEDDING_MODEL": "my/embedder", "OPENROUTER_MODEL": "anthropic/claude-haiku-5.5"})
    assert s.embedding_model == "my/embedder"
    assert s.openrouter_model == "anthropic/claude-haiku-5.5"


def test_blank_value_falls_back_to_default():
    # `.env.example` ships with blank values; blank must mean "use default".
    s = Settings.from_env({"OPENROUTER_MODEL": "   ", "OPENROUTER_API_KEY": "", "EMBEDDING_MODEL": ""})
    assert s.openrouter_model == DEFAULT_OPENROUTER_MODEL
    assert s.openrouter_api_key is None
    assert s.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"


def test_openrouter_key_is_accepted_under_either_spelling():
    assert Settings.from_env({"OPENROUTER_API_KEY": " k1 "}).openrouter_api_key == "k1"
    assert Settings.from_env({"OPEN_ROUTER_API_KEY": "k2"}).openrouter_api_key == "k2"
    both = {"OPENROUTER_API_KEY": "k1", "OPEN_ROUTER_API_KEY": "k2"}
    assert Settings.from_env(both).openrouter_api_key == "k1"


def test_require_openrouter_key():
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        Settings.from_env({}).require_openrouter_key()
    assert Settings.from_env({"OPENROUTER_API_KEY": "sk-test"}).require_openrouter_key() == "sk-test"


def test_old_provider_settings_are_gone():
    s = Settings.from_env({"ANTHROPIC_API_KEY": "a", "GEMINI_API_KEY": "g", "LLM_PROVIDER": "claude"})
    for name in ("anthropic_api_key", "claude_model", "gemini_api_key", "gemini_model", "llm_provider"):
        assert not hasattr(s, name), name


def test_data_dir_override_and_derived_paths(tmp_path):
    s = Settings.from_env({"RAG_DATA_DIR": str(tmp_path)})
    assert s.data_dir == tmp_path
    assert s.sample_docs_dir == tmp_path / "sample_docs"
    assert s.index_dir == tmp_path / "indexes"


def test_default_data_dir_points_at_repo_data():
    s = Settings.from_env({})
    assert s.data_dir.name == "data"
    assert isinstance(s.data_dir, Path) and s.data_dir.is_absolute()
