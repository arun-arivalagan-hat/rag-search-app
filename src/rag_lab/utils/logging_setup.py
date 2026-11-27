"""Send log messages to the console and to a file in ``logs/``.

The API logs the real details of a failed model call here (the browser only gets a generic
message), so this file is where to look when ``/api/answer`` returns 502.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
MAX_LOG_BYTES = 1_000_000        # roll over at ~1 MB so the file cannot grow forever
KEPT_LOG_FILES = 3


def configure_logging(log_dir: Path, level: int = logging.INFO) -> Path:
    """Log to the console and to ``log_dir/rag_lab.log``. Safe to call more than once."""
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "rag_lab.log"
    logger = logging.getLogger("rag_lab")
    logger.setLevel(level)
    if not any(getattr(h, "_rag_lab_file", False) for h in logger.handlers):
        file_handler = RotatingFileHandler(path, maxBytes=MAX_LOG_BYTES, backupCount=KEPT_LOG_FILES, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        file_handler._rag_lab_file = True                      # marks it so a second call adds nothing
        logger.addHandler(file_handler)
    if not any(getattr(h, "_rag_lab_console", False) for h in logger.handlers):
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(LOG_FORMAT))
        console._rag_lab_console = True
        logger.addHandler(console)
    return path
