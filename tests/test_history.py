import json
import os
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch


PHONE = "919999999999"
MESSAGES = [
    {"role": "user",      "content": "hi"},
    {"role": "assistant", "content": "Hello!"},
]


@pytest.fixture(autouse=True)
def tmp_sessions(tmp_path, monkeypatch):
    """Redirect sessions dir to a temp directory for all tests."""
    monkeypatch.setattr("app.history._SESSIONS_DIR", str(tmp_path))
    return tmp_path


# ─── load_history ─────────────────────────────────────────────────────────────

def test_load_history_returns_empty_when_no_file():
    from app.history import load_history
    assert load_history(PHONE) == []


def test_load_history_returns_messages_when_recent(tmp_path):
    from app.history import load_history
    path = tmp_path / f"{PHONE}.json"
    path.write_text(json.dumps({
        "last_active": datetime.now().isoformat(timespec='seconds'),
        "messages": MESSAGES,
    }))
    assert load_history(PHONE) == MESSAGES


def test_load_history_returns_empty_and_deletes_when_expired(tmp_path):
    from app.history import load_history
    path = tmp_path / f"{PHONE}.json"
    expired_time = (datetime.now() - timedelta(minutes=10)).isoformat(timespec='seconds')
    path.write_text(json.dumps({"last_active": expired_time, "messages": MESSAGES}))

    result = load_history(PHONE)
    assert result == []
    assert not path.exists()


def test_load_history_returns_empty_and_deletes_corrupt_file(tmp_path):
    from app.history import load_history
    path = tmp_path / f"{PHONE}.json"
    path.write_text("not valid json {{{{")

    result = load_history(PHONE)
    assert result == []
    assert not path.exists()


def test_load_history_returns_empty_and_deletes_missing_key(tmp_path):
    from app.history import load_history
    path = tmp_path / f"{PHONE}.json"
    path.write_text(json.dumps({"messages": MESSAGES}))  # missing last_active

    result = load_history(PHONE)
    assert result == []
    assert not path.exists()


# ─── save_history ─────────────────────────────────────────────────────────────

def test_save_history_creates_file(tmp_path):
    from app.history import save_history
    save_history(PHONE, MESSAGES)
    assert (tmp_path / f"{PHONE}.json").exists()


def test_save_history_writes_correct_structure(tmp_path):
    from app.history import save_history
    save_history(PHONE, MESSAGES)
    data = json.loads((tmp_path / f"{PHONE}.json").read_text())
    assert "last_active" in data
    assert data["messages"] == MESSAGES


def test_save_history_updates_last_active(tmp_path):
    from app.history import save_history
    before = datetime.now()
    save_history(PHONE, MESSAGES)
    data = json.loads((tmp_path / f"{PHONE}.json").read_text())
    saved_time = datetime.fromisoformat(data["last_active"])
    assert saved_time >= before.replace(microsecond=0)


def test_save_history_trims_to_memory_limit(tmp_path):
    from app.history import save_history
    from rag_chatbot import MEMORY_LIMIT
    long_history = [{"role": "user", "content": str(i)} for i in range(MEMORY_LIMIT * 3)]
    save_history(PHONE, long_history)
    data = json.loads((tmp_path / f"{PHONE}.json").read_text())
    assert len(data["messages"]) == MEMORY_LIMIT * 2


def test_save_history_keeps_most_recent_messages(tmp_path):
    from app.history import save_history
    from rag_chatbot import MEMORY_LIMIT
    messages = [{"role": "user", "content": str(i)} for i in range(MEMORY_LIMIT * 3)]
    save_history(PHONE, messages)
    data = json.loads((tmp_path / f"{PHONE}.json").read_text())
    # Most recent messages should be kept
    assert data["messages"][-1]["content"] == str(MEMORY_LIMIT * 3 - 1)


# ─── round-trip ───────────────────────────────────────────────────────────────

def test_save_then_load_returns_same_messages(tmp_path):
    from app.history import save_history, load_history
    save_history(PHONE, MESSAGES)
    assert load_history(PHONE) == MESSAGES
