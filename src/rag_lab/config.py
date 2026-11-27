"""Environment-based configuration.

Every tunable (model names, paths, the OpenRouter key) comes from an environment
variable with a sensible default. Precedence, highest first:

    1. real environment variables
    2. a ``.env`` file (never overrides real env vars)
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

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Answers are written through OpenRouter: one key that reaches many models. Model ids look like
# "google/gemini-3.5-flash-lite"; the public list is https://openrouter.ai/api/v1/models
# (a listed model can still be refused for your account, so a real answer is the only proof).
DEFAULT_OPENROUTER_MODEL = "google/gemini-3.5-flash-lite"

# Chunking defaults, chosen by comparing sizes on the labelled queries (step 7f).
DEFAULT_CHUNK_SIZE = 300      # characters per chunk (recursive chunking)
DEFAULT_CHUNK_OVERLAP = 50    # characters repeated from the previous chunk


@dataclass(frozen=True)
class Settings:
    """Immutable bag of settings. Frozen so it can be shared safely."""

    # Local models
    embedding_model: str
    rerank_model: str
    # Paths
    data_dir: Path
    # Answer generation (OpenRouter)
    openrouter_api_key: str | None = None
    openrouter_model: str = DEFAULT_OPENROUTER_MODEL

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
            embedding_model=get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            rerank_model=get("RERANK_MODEL", DEFAULT_RERANK_MODEL),
            data_dir=data_dir,
            openrouter_api_key=_openrouter_key(env),
            openrouter_model=get("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL),
        )

    def require_openrouter_key(self) -> str:
        """Return the OpenRouter key or fail with an actionable message."""
        if not self.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set. Put it in .env (see .env.example).")
        return self.openrouter_api_key


def _openrouter_key(env: Mapping[str, str]) -> str | None:
    """OPENROUTER_API_KEY, or the spelling OPEN_ROUTER_API_KEY (both are accepted)."""
    for name in ("OPENROUTER_API_KEY", "OPEN_ROUTER_API_KEY"):
        value = env.get(name, "").strip()
        if value:
            return value
    return None


def get_settings() -> Settings:
    """Convenience wrapper: settings from the real environment."""
    return Settings.from_env()
