"""Environment-based configuration.

Every tunable (model names, paths, Claude parameters) comes from an environment
variable with a sensible default. Precedence, highest first:

    1. real environment variables
    2. a ``.env`` file in the working directory (never overrides real env vars)
    3. defaults below

Nothing is read at import time: call :func:`get_settings` (or
:meth:`Settings.from_env`) so tests can inject their own environment.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# src/rag_lab/config.py -> repo root is three parents up.
_REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CLAUDE_MODEL = "claude-sonnet-5-5"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass(frozen=True)
class Settings:
    """Immutable bag of settings. Frozen so it can be shared safely."""

    # Claude
    anthropic_api_key: str | None
    claude_model: str
    claude_max_tokens: int
    claude_temperature: float
    # Local models
    embedding_model: str
    rerank_model: str
    # Paths
    data_dir: Path

    @property
    def sample_docs_dir(self) -> Path:
        return self.data_dir / "sample_docs"

    @property
    def eval_dir(self) -> Path:
        return self.data_dir / "eval"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "indexes"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Build settings from ``env`` (defaults to ``os.environ``).

        Passing an explicit mapping skips ``.env`` loading, which keeps tests
        hermetic.
        """
        if env is None:
            load_dotenv(override=False)
            env = os.environ

        def get(name: str, default: str) -> str:
            value = env.get(name, "").strip()
            return value or default  # empty string counts as "unset"

        data_dir = Path(get("RAG_DATA_DIR", str(_REPO_ROOT / "data"))).expanduser()
        if not data_dir.is_absolute():
            data_dir = (Path.cwd() / data_dir).resolve()

        return cls(
            anthropic_api_key=env.get("ANTHROPIC_API_KEY", "").strip() or None,
            claude_model=get("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL),
            claude_max_tokens=_parse(int, "CLAUDE_MAX_TOKENS", get("CLAUDE_MAX_TOKENS", "1024")),
            claude_temperature=_parse(
                float, "CLAUDE_TEMPERATURE", get("CLAUDE_TEMPERATURE", "0.0")
            ),
            embedding_model=get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            rerank_model=get("RERANK_MODEL", DEFAULT_RERANK_MODEL),
            data_dir=data_dir,
        )

    def require_api_key(self) -> str:
        """Return the API key or fail with an actionable message."""
        if not self.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        return self.anthropic_api_key


def _parse(cast, name: str, raw: str):
    try:
        return cast(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a {cast.__name__}, got {raw!r}") from exc


def get_settings() -> Settings:
    """Convenience wrapper: settings from the real environment."""
    return Settings.from_env()
