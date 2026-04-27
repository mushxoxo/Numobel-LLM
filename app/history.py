import json
import logging
import os
from datetime import datetime, timedelta

from rag_chatbot import MEMORY_LIMIT

log = logging.getLogger('rag_chatbot')

_SESSIONS_DIR     = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sessions')
_SESSION_TIMEOUT  = timedelta(minutes=5)


def _session_path(phone: str) -> str:
    os.makedirs(_SESSIONS_DIR, exist_ok=True)
    return os.path.join(_SESSIONS_DIR, f"{phone}.json")


def load_history(phone: str) -> list[dict]:
    """Load conversation history for phone. Returns [] if expired or missing."""
    path = _session_path(phone)
    if not os.path.exists(path):
        return []

    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)

        last_active = datetime.fromisoformat(data['last_active'])
        if datetime.now() - last_active > _SESSION_TIMEOUT:
            os.remove(path)
            log.debug("HISTORY | expired session deleted for %s", phone)
            return []

        log.debug("HISTORY | loaded %d messages for %s", len(data['messages']), phone)
        return data['messages']

    except (json.JSONDecodeError, KeyError, ValueError):
        os.remove(path)
        log.warning("HISTORY | corrupt session file deleted for %s", phone)
        return []


def save_history(phone: str, history: list[dict]) -> None:
    """Persist conversation history, trimmed to last MEMORY_LIMIT * 2 messages."""
    trimmed = history[-(MEMORY_LIMIT * 2):]
    path    = _session_path(phone)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({
            'last_active': datetime.now().isoformat(timespec='seconds'),
            'messages':    trimmed,
        }, f, ensure_ascii=False, indent=2)
    log.debug("HISTORY | saved %d messages for %s", len(trimmed), phone)
