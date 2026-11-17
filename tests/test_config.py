from pathlib import Path

import pytest

from rag_lab.config import DEFAULT_CLAUDE_MODEL, Settings


def test_defaults_when_env_empty():
    s = Settings.from_env({})
    assert s.claude_model == DEFAULT_CLAUDE_MODEL
    assert s.claude_max_tokens == 1024
    assert s.claude_temperature == 0.0
    assert s.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert s.anthropic_api_key is None


def test_claude_model_is_env_configurable():
    s = Settings.from_env({"CLAUDE_MODEL": "claude-opus-5-5", "CLAUDE_MAX_TOKENS": "2048",
                           "CLAUDE_TEMPERATURE": "0.3"})
    assert s.claude_model == "claude-opus-5-5"
    assert s.claude_max_tokens == 2048
    assert s.claude_temperature == pytest.approx(0.3)


def test_blank_value_falls_back_to_default():
    # `.env.example` ships with blank values; blank must mean "use default".
    s = Settings.from_env({"CLAUDE_MODEL": "   ", "ANTHROPIC_API_KEY": ""})
    assert s.claude_model == DEFAULT_CLAUDE_MODEL
    assert s.anthropic_api_key is None


def test_bad_number_raises_with_variable_name():
    with pytest.raises(ValueError, match="CLAUDE_MAX_TOKENS"):
        Settings.from_env({"CLAUDE_MAX_TOKENS": "lots"})


def test_require_api_key():
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        Settings.from_env({}).require_api_key()
    assert Settings.from_env({"ANTHROPIC_API_KEY": "sk-test"}).require_api_key() == "sk-test"


def test_data_dir_override_and_derived_paths(tmp_path):
    s = Settings.from_env({"RAG_DATA_DIR": str(tmp_path)})
    assert s.data_dir == tmp_path
    assert s.sample_docs_dir == tmp_path / "sample_docs"
    assert s.index_dir == tmp_path / "indexes"


def test_default_data_dir_points_at_repo_data():
    s = Settings.from_env({})
    assert s.data_dir.name == "data"
    assert isinstance(s.data_dir, Path) and s.data_dir.is_absolute()
