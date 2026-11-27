import logging

from rag_lab.utils import configure_logging


def test_messages_reach_the_log_file(tmp_path):
    path = configure_logging(tmp_path / "logs")
    logging.getLogger("rag_lab.api").error("boom details")
    for handler in logging.getLogger("rag_lab").handlers:
        handler.flush()
    assert path.exists() and "boom details" in path.read_text(encoding="utf-8")


def test_calling_twice_does_not_duplicate_handlers(tmp_path):
    configure_logging(tmp_path / "a")
    before = len(logging.getLogger("rag_lab").handlers)
    configure_logging(tmp_path / "a")
    assert len(logging.getLogger("rag_lab").handlers) == before
