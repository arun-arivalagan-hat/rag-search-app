"""Entry point for the web server.

    uvicorn rag_lab.api.main:app --host 127.0.0.1 --port 8000

then open http://127.0.0.1:8000 : the server also serves the frontend, so page and API share one
address and the browser needs no CORS. If you serve the frontend from somewhere else (for example
the VS Code Live Server on port 5500), list its address in ALLOWED_ORIGINS, comma separated:

    ALLOWED_ORIGINS=http://127.0.0.1:5500,http://localhost:5500

Keep ``--host 127.0.0.1``. /api/answer spends money and there is no login or rate limit yet, so the
server must not be reachable from other machines.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from rag_lab.api.app import create_app
from rag_lab.utils import configure_logging

load_dotenv(override=False)          # so ALLOWED_ORIGINS in .env is visible below (real env vars still win)

ROOT_DIR = Path(__file__).resolve().parents[3]
FRONTEND_DIR = ROOT_DIR / "frontend" / "src"      # repo root / frontend / src
configure_logging(ROOT_DIR / "logs")             # console + logs/rag_lab.log


def _origins() -> list[str]:
    return [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]


app = create_app(static_dir=FRONTEND_DIR, cors_origins=_origins())
