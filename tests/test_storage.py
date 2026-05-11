import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch

import app.refinement.storage as storage_module


@pytest.fixture(autouse=True)
def tmp_dirs(tmp_path, monkeypatch):
    """Redirect all storage paths to a temp directory."""
    monkeypatch.setattr(storage_module, '_REFINE_STATE_DIR', tmp_path / 'refine_state')
    monkeypatch.setattr(storage_module, '_PREFS_DIR',        tmp_path / 'prefs')
    monkeypatch.setattr(storage_module, '_PENDING_PATH',     tmp_path / 'pending.jsonl')
    yield tmp_path


PHONE = '919999999999'


# ─── Refine state ─────────────────────────────────────────────────────────────

def test_save_and_load_refine_state():
    state = {'original_pair': {'question': 'Q?'}, 'current_pair': {'question': 'Q?'}, 'chat_history': []}
    storage_module.save_refine_state(PHONE, state)
    loaded = storage_module.load_refine_state(PHONE)
    assert loaded['original_pair']['question'] == 'Q?'
    assert 'last_active' in loaded


def test_load_missing_refine_state():
    assert storage_module.load_refine_state(PHONE) == {}


def test_clear_refine_state():
    storage_module.save_refine_state(PHONE, {'x': 1})
    storage_module.clear_refine_state(PHONE)
    assert storage_module.load_refine_state(PHONE) == {}


def test_atomic_write_no_partial_file(tmp_path, monkeypatch):
    """Simulate crash mid-write: tmp file created but os.replace not called."""
    state_path = storage_module._REFINE_STATE_DIR / f"{storage_module._safe_phone(PHONE)}.json"

    original_replace = os.replace
    calls = []

    def crashing_replace(src, dst):
        calls.append((src, dst))
        # Don't actually replace — simulate crash after .tmp write
        if str(dst).endswith('.json') and 'refine_state' in str(dst):
            raise OSError("simulated crash")
        original_replace(src, dst)

    monkeypatch.setattr(os, 'replace', crashing_replace)

    with pytest.raises(OSError):
        storage_module.save_refine_state(PHONE, {'x': 1})

    # The actual .json file must not exist (tmp file is there but not swapped in)
    assert not state_path.exists()


# ─── Admin preferences ────────────────────────────────────────────────────────

def test_append_and_load_prefs():
    storage_module.append_admin_prefs(PHONE, [{'text': 'Use ₹ for prices', 'evidence': 'always'}])
    prefs = storage_module.load_admin_prefs(PHONE)
    assert len(prefs) == 1
    assert prefs[0]['text'] == 'Use ₹ for prices'


def test_dedup_prefs():
    storage_module.append_admin_prefs(PHONE, [{'text': 'Use ₹ for prices', 'evidence': ''}])
    storage_module.append_admin_prefs(PHONE, [{'text': 'Use ₹ for prices', 'evidence': ''}])
    prefs = storage_module.load_admin_prefs(PHONE)
    assert len(prefs) == 1


def test_dedup_case_insensitive():
    storage_module.append_admin_prefs(PHONE, [{'text': 'use ₹ for prices', 'evidence': ''}])
    storage_module.append_admin_prefs(PHONE, [{'text': 'USE ₹ FOR PRICES', 'evidence': ''}])
    prefs = storage_module.load_admin_prefs(PHONE)
    assert len(prefs) == 1


def test_fifo_overflow_at_cap(monkeypatch):
    monkeypatch.setattr(storage_module, '_MAX_PREFS', 3)
    for i in range(5):
        storage_module.append_admin_prefs(PHONE, [{'text': f'Rule {i}', 'evidence': ''}])
    prefs = storage_module.load_admin_prefs(PHONE)
    assert len(prefs) == 3
    # Oldest entries dropped
    texts = [p['text'] for p in prefs]
    assert 'Rule 0' not in texts
    assert 'Rule 4' in texts


def test_load_prefs_missing():
    assert storage_module.load_admin_prefs(PHONE) == []


# ─── Pair locking ─────────────────────────────────────────────────────────────

def _write_pending(tmp_path, pairs):
    path = storage_module._PENDING_PATH
    with open(path, 'w') as f:
        for p in pairs:
            f.write(json.dumps(p) + '\n')


def test_lock_pair_success(tmp_path):
    _write_pending(tmp_path, [{'question': 'Q?', 'answer': 'A.'}])
    result = storage_module.lock_pair('Q?', PHONE)
    assert result is True
    assert storage_module.is_locked('Q?') == PHONE


def test_lock_pair_already_locked_by_other(tmp_path):
    _write_pending(tmp_path, [{'question': 'Q?', 'answer': 'A.', 'in_review': '911111111111'}])
    result = storage_module.lock_pair('Q?', PHONE)
    assert result is False


def test_lock_pair_reacquire_own_lock(tmp_path):
    _write_pending(tmp_path, [{'question': 'Q?', 'answer': 'A.', 'in_review': PHONE}])
    result = storage_module.lock_pair('Q?', PHONE)
    assert result is True


def test_unlock_pair(tmp_path):
    _write_pending(tmp_path, [{'question': 'Q?', 'answer': 'A.', 'in_review': PHONE}])
    storage_module.unlock_pair('Q?')
    assert storage_module.is_locked('Q?') is None


def test_is_locked_missing_file():
    assert storage_module.is_locked('anything') is None
