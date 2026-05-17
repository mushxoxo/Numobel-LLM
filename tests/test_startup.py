"""Tests for app/startup.py."""

import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def reset_ready_and_env(monkeypatch, tmp_path):
    import app.config as cfg
    import app.db as db_module
    import app.startup as startup_module

    monkeypatch.setenv("WA2MATION_API_KEY", "test-key")
    monkeypatch.setenv("WA2MATION_VENDOR_UID", "test-uid")
    monkeypatch.setattr(cfg, "SQLITE_PATH", tmp_path / "startup-test.db")
    monkeypatch.setattr(cfg, "APPROVED_PATH", tmp_path / "approved.jsonl")
    db_module._conn = None
    startup_module._ready = False
    yield
    startup_module._ready = False
    db_module._conn = None


def test_validate_env_raises_on_missing_key(monkeypatch):
    import app.startup as startup_module

    monkeypatch.delenv("WA2MATION_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as exc_info:
        startup_module.validate_env()
    assert "WA2MATION_API_KEY" in str(exc_info.value)
    assert "Check your .env file" in str(exc_info.value)


def test_validate_env_passes_when_all_keys_set():
    import app.startup as startup_module

    assert startup_module.validate_env() is None


def test_qna_migration_runs_once():
    import app.config as cfg
    import app.startup as startup_module

    pairs = [
        {"question": "What is X?", "answer": "X is Y.", "message_type": "text", "approved": False},
        {"question": "Hello", "answer": "Hi", "message_type": "text", "approved": True},
    ]
    cfg.APPROVED_PATH.write_text("\n".join(json.dumps(pair) for pair in pairs) + "\n")

    ingested = []
    fake_collection = MagicMock()
    with patch("app.rag.get_collection", return_value=fake_collection), \
         patch("app.rag.ingest_qna_pair", side_effect=lambda pair, collection: ingested.append(pair)):
        startup_module._migrate_qna_if_needed()

    assert len(ingested) == 2
    from app.db import get_meta

    assert get_meta("qna_migration_done") == "1"


def test_qna_migration_skipped_when_flag_set():
    """The app_meta qna_migration_done flag prevents re-migration."""
    import app.startup as startup_module
    from app.db import set_meta

    set_meta("qna_migration_done", "1")
    with patch("app.refinement.storage.load_jsonl") as mock_load:
        startup_module._migrate_qna_if_needed()
    assert not mock_load.called


def test_qna_migration_missing_file_marks_done():
    import app.config as cfg
    import app.startup as startup_module

    assert not cfg.APPROVED_PATH.exists()
    startup_module._migrate_qna_if_needed()
    from app.db import get_meta

    assert get_meta("qna_migration_done") == "1"


def test_startup_sets_ready_after_all_steps(tmp_path, monkeypatch):
    import app.config as cfg
    import app.startup as startup_module

    data_file = tmp_path / "cp.json"
    data_file.write_text("[]")
    monkeypatch.setattr(cfg, "DATA_FILE", data_file)
    cfg.APPROVED_PATH.write_text("")

    fake_collection = MagicMock()
    fake_collection.count.return_value = 1
    with patch("app.rag.get_collection", return_value=fake_collection), \
         patch("app.rag.ingest_data") as mock_ingest, \
         patch("app.rag.ingest_qna_pair"):
        assert startup_module._ready is False
        startup_module._startup_orchestrator()

    assert startup_module._ready is True
    assert not mock_ingest.called


def test_ready_stays_false_when_step_fails():
    import app.startup as startup_module

    with patch("app.db.sync_products", side_effect=RuntimeError("sync exploded")):
        startup_module._startup_orchestrator()
    assert startup_module._ready is False


def test_run_startup_validates_env_before_thread(monkeypatch):
    import app.startup as startup_module

    monkeypatch.delenv("WA2MATION_API_KEY", raising=False)
    with patch("threading.Thread") as mock_thread:
        with pytest.raises(RuntimeError):
            startup_module.run_startup()
        assert not mock_thread.called
