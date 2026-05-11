"""Central logging configuration for the Numobel RAG chatbot."""

import json
import logging
import os
from pathlib import Path

_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_FILE = _LOG_DIR / "rag_chatbot.log"

_TEXT_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({
            "time":    self.formatTime(record, self._fmt),
            "level":   record.levelname,
            "message": record.getMessage(),
        })


def setup_logging() -> logging.Logger:
    """Configure the root 'rag_chatbot' logger. Idempotent — safe to call multiple times."""
    logger = logging.getLogger("rag_chatbot")
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(_TEXT_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    if os.getenv("LOG_FORMAT", "").lower() == "json":
        ch.setFormatter(_JsonFormatter())
    else:
        ch.setFormatter(logging.Formatter(_TEXT_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(ch)

    return logger


def get_logger() -> logging.Logger:
    """Return the configured 'rag_chatbot' logger, setting it up if not already done."""
    logger = logging.getLogger("rag_chatbot")
    if not logger.handlers:
        setup_logging()
    return logger
