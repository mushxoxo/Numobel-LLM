"""Per-admin refinement state and preferences I/O. All writes are atomic."""

import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path

from app.config import PENDING_PATH as _PENDING_PATH, APPROVED_PATH as _APPROVED_PATH

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_REFINE_STATE_DIR = _PROJECT_ROOT / 'training' / 'admin_refine_state'
_PREFS_DIR        = _PROJECT_ROOT / 'training' / 'admin_preferences'

_MAX_PREFS = 20


def _atomic_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def _safe_phone(phone: str) -> str:
    return phone.lstrip('+').replace(' ', '_')


def _match_pair(p: dict, identifier: str) -> bool:
    """Match a pair by pair_id first, fall back to question text for legacy pairs."""
    return p.get('pair_id') == identifier or p.get('question') == identifier


# ─── Pair identity ────────────────────────────────────────────────────────────

def ensure_pair_id(pair: dict) -> dict:
    """Add a stable pair_id if absent. Mutates in-place and returns the pair."""
    if not pair.get('pair_id'):
        pair['pair_id'] = str(uuid.uuid4())
    return pair


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


# ─── JSONL helpers ────────────────────────────────────────────────────────────

def _load_pending_lines() -> list[str]:
    if not _PENDING_PATH.exists():
        return []
    with open(_PENDING_PATH) as f:
        return [line.rstrip('\n') for line in f]


def _write_pending_lines(lines: list[str]) -> None:
    _PENDING_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _PENDING_PATH.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        for line in lines:
            f.write(line + '\n')
    os.replace(tmp, _PENDING_PATH)


def load_next_pending(skip_locked: bool = False) -> dict | None:
    """Return the first unapproved pending pair. With skip_locked=True, skip in-review pairs."""
    if not _PENDING_PATH.exists():
        return None
    with open(_PENDING_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                pair = json.loads(line)
                if not pair.get('approved'):
                    if skip_locked and pair.get('in_review'):
                        continue
                    return pair
            except json.JSONDecodeError:
                continue
    return None


def mark_approved(identifier: str) -> None:
    """Mark the pair matching identifier as approved in pending.jsonl."""
    lines = _load_pending_lines()
    updated = []
    for line in lines:
        if line.strip():
            try:
                p = json.loads(line)
                if _match_pair(p, identifier):
                    p['approved'] = True
                    p.pop('in_review', None)
                    line = json.dumps(p)
            except json.JSONDecodeError:
                pass
        updated.append(line)
    _write_pending_lines(updated)


def append_approved(pair: dict) -> None:
    """Append a pair to approved.jsonl."""
    _APPROVED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_APPROVED_PATH, 'a') as f:
        f.write(json.dumps(pair, ensure_ascii=False) + '\n')


def load_style_examples(limit: int = 3) -> list[dict]:
    """Load up to limit diverse approved pairs as style examples."""
    if not _APPROVED_PATH.exists():
        return []
    examples = []
    seen_types: set[str] = set()
    with open(_APPROVED_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                p = json.loads(line)
                mt = p.get('message_type', 'text')
                if mt not in seen_types:
                    examples.append(p)
                    seen_types.add(mt)
                    if len(examples) >= limit:
                        break
            except json.JSONDecodeError:
                continue
    return examples


# ─── Pair locking ─────────────────────────────────────────────────────────────

def lock_pair(identifier: str, phone: str) -> bool:
    """
    Lock the pending pair for the given admin phone.
    identifier is pair_id (preferred) or question text (legacy fallback).
    Returns True if lock acquired, False if already locked by another admin.
    """
    lines = _load_pending_lines()
    updated = []
    acquired = False
    for line in lines:
        if line.strip():
            try:
                p = json.loads(line)
                if _match_pair(p, identifier):
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


def unlock_pair(identifier: str) -> None:
    """Unlock the pending pair matching identifier (pair_id or question)."""
    lines = _load_pending_lines()
    updated = []
    for line in lines:
        if line.strip():
            try:
                p = json.loads(line)
                if _match_pair(p, identifier):
                    p.pop('in_review', None)
                    line = json.dumps(p)
            except json.JSONDecodeError:
                pass
        updated.append(line)
    _write_pending_lines(updated)


def is_locked(identifier: str) -> str | None:
    """Return the phone holding the lock on this pair, or None if unlocked."""
    if not _PENDING_PATH.exists():
        return None
    with open(_PENDING_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                p = json.loads(line)
                if _match_pair(p, identifier):
                    return p.get('in_review') or None
            except json.JSONDecodeError:
                pass
    return None
