"""Per-admin refinement state and preferences I/O. All writes are atomic."""

import json
import os
import re
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_REFINE_STATE_DIR = _PROJECT_ROOT / 'training' / 'admin_refine_state'
_PREFS_DIR        = _PROJECT_ROOT / 'training' / 'admin_preferences'
_PENDING_PATH     = _PROJECT_ROOT / 'training' / 'qna_pairs' / 'pending.jsonl'

_MAX_PREFS = 20


def _atomic_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def _safe_phone(phone: str) -> str:
    return phone.lstrip('+').replace(' ', '_')


# ─── Refine state ─────────────────────────────────────────────────────────────

def load_refine_state(phone: str) -> dict:
    p = _REFINE_STATE_DIR / f"{_safe_phone(phone)}.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return {}


def save_refine_state(phone: str, state: dict) -> None:
    """Persist state to disk. Normalises last_active to ISO string."""
    to_save = {k: v for k, v in state.items() if k != 'last_active'}
    to_save['last_active'] = datetime.utcnow().isoformat()
    _atomic_write(_REFINE_STATE_DIR / f"{_safe_phone(phone)}.json", to_save)


def clear_refine_state(phone: str) -> None:
    p = _REFINE_STATE_DIR / f"{_safe_phone(phone)}.json"
    if p.exists():
        p.unlink()


# ─── Admin preferences ────────────────────────────────────────────────────────

def load_admin_prefs(phone: str) -> list[dict]:
    p = _PREFS_DIR / f"{_safe_phone(phone)}.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding='utf-8')).get('prefs', [])
    except Exception:
        return []


def _dedup_key(text: str) -> str:
    return re.sub(r'[\W_]+', '', text).lower()


def append_admin_prefs(phone: str, new_patterns: list[dict]) -> None:
    """Append new patterns, deduplicating and capping at _MAX_PREFS (FIFO)."""
    existing = load_admin_prefs(phone)
    existing_keys = {_dedup_key(p['text']) for p in existing}

    for pattern in new_patterns:
        key = _dedup_key(pattern['text'])
        if key in existing_keys:
            continue
        existing.append({
            'id':                len(existing) + 1,
            'text':              pattern['text'],
            'learned_at':        datetime.utcnow().isoformat(),
            'evidence_question': pattern.get('evidence', ''),
        })
        existing_keys.add(key)

    if len(existing) > _MAX_PREFS:
        existing = existing[-_MAX_PREFS:]

    _atomic_write(_PREFS_DIR / f"{_safe_phone(phone)}.json", {'prefs': existing})


# ─── Pair locking ─────────────────────────────────────────────────────────────

def _load_pending_lines() -> list[str]:
    if not _PENDING_PATH.exists():
        return []
    with open(_PENDING_PATH) as f:
        return [line.rstrip('\n') for line in f]


def _write_pending_lines(lines: list[str]) -> None:
    tmp = _PENDING_PATH.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        for line in lines:
            f.write(line + '\n')
    os.replace(tmp, _PENDING_PATH)


def lock_pair(question: str, phone: str) -> bool:
    """
    Lock the pending pair for the given admin phone.
    Returns True if lock acquired, False if already locked by another admin.
    """
    lines = _load_pending_lines()
    updated = []
    acquired = False
    for line in lines:
        if line.strip():
            try:
                p = json.loads(line)
                if p.get('question') == question:
                    existing_lock = p.get('in_review')
                    if existing_lock and existing_lock != phone:
                        return False
                    p['in_review'] = phone
                    acquired = True
                    line = json.dumps(p)
            except json.JSONDecodeError:
                pass
        updated.append(line)
    if acquired:
        _write_pending_lines(updated)
    return acquired


def unlock_pair(question: str) -> None:
    lines = _load_pending_lines()
    updated = []
    for line in lines:
        if line.strip():
            try:
                p = json.loads(line)
                if p.get('question') == question:
                    p.pop('in_review', None)
                    line = json.dumps(p)
            except json.JSONDecodeError:
                pass
        updated.append(line)
    _write_pending_lines(updated)


def is_locked(question: str) -> str | None:
    """Return the phone holding the lock, or None if unlocked."""
    if not _PENDING_PATH.exists():
        return None
    with open(_PENDING_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                p = json.loads(line)
                if p.get('question') == question:
                    return p.get('in_review') or None
            except json.JSONDecodeError:
                pass
    return None
