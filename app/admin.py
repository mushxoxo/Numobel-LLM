import json
import os
import re
from datetime import datetime, timedelta
from enum import Enum

import app.rag as rag
from app.config import BASE_DIR as _BASE_DIR
from app.log import get_logger
from app.messaging import send_text, send_interactive, send_media
from app.refinement.constraints import BODY_MAX_CHARS
from app.refinement import (
    chat_turn, extract_patterns, validate_pair,
    load_refine_state, save_refine_state, clear_refine_state,
    load_admin_prefs, append_admin_prefs,
    load_next_pending, mark_approved, append_approved, load_style_examples,
    lock_pair, unlock_pair, is_locked,
)

log = get_logger()

_CONFIG_PATH = _BASE_DIR / 'config.json'

# In-memory admin sessions: {phone: {state, current_pair, last_active, ...}}
_sessions: dict = {}


class AdminState(str, Enum):
    IDLE                   = "idle"
    MENU                   = "menu"
    TRAINING_REVIEW        = "training_review"
    TRAINING_CHAT          = "training_chat"
    TRAINING_CONFIRM_SAVE  = "training_confirm_save"
    TRAINING_CONFIRM_PREFS = "training_confirm_prefs"


_RESUMABLE_STATES = {
    AdminState.TRAINING_CHAT,
    AdminState.TRAINING_CONFIRM_SAVE,
    AdminState.TRAINING_CONFIRM_PREFS,
}


def _load_config() -> dict:
    try:
        with open(_CONFIG_PATH) as f:
            return json.load(f)
    except Exception:
        return {"admin_phones": [], "admin_timeout_minutes": 15}


def is_admin(phone: str) -> bool:
    config = _load_config()
    normalized = phone.lstrip('+')
    return normalized in [str(p).lstrip('+') for p in config.get("admin_phones", [])]


def needs_admin_handling(phone: str, message: str) -> bool:
    """True if this message should be routed to handle_admin() instead of the RAG pipeline."""
    if not is_admin(phone):
        return False
    msg = message.strip().lower()
    if msg.startswith(':admin') or msg == ':train':
        return True
    return _sessions.get(phone, {}).get("state", AdminState.IDLE) != AdminState.IDLE


# ─── Session helpers ──────────────────────────────────────────────────────────

def _set_session(phone: str, state: AdminState, current_pair=None, **extra):
    _sessions[phone] = {
        "state": state,
        "current_pair": current_pair,
        "last_active": datetime.utcnow(),
        **extra,
    }


def _clear_session(phone: str):
    _sessions.pop(phone, None)


def _touch_session(phone: str):
    if phone in _sessions:
        _sessions[phone]["last_active"] = datetime.utcnow()


def _is_timed_out(phone: str) -> bool:
    session = _sessions.get(phone)
    if not session or not session.get("last_active"):
        return False
    timeout = timedelta(minutes=_load_config().get("admin_timeout_minutes", 15))
    last = session["last_active"]
    if isinstance(last, str):
        try:
            last = datetime.fromisoformat(last)
        except ValueError:
            return False
    return datetime.utcnow() - last > timeout


def _get_model_and_key() -> tuple[str, str | None]:
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if api_key:
        return 'claude-sonnet-4-6', api_key
    return 'qwen2.5:14b', None


def _pair_id(pair: dict | None) -> str:
    """Return the canonical identifier for locking/matching this pair."""
    if not pair:
        return ''
    return pair.get('pair_id') or pair.get('question', '')


# ─── WhatsApp send helpers ────────────────────────────────────────────────────

def _send_admin_menu(phone: str):
    send_interactive(
        phone=phone,
        body="Choose an action:",
        buttons=["QnA Review", "Auto-generate", "Test Bot"],
        header="Numobel Admin",
        footer=":admin off to exit",
    )


def _send_pair_for_review(phone: str, pair: dict):
    """Send Q&A pair as text + interactive [Approve]/[Refine]/[Cancel] buttons."""
    msg_type = pair.get("message_type", "text")
    btns = pair.get("buttons")
    buttons_line = f"\nButtons: {', '.join(btns)}" if btns else ""
    body = (
        f"*Q:* {pair['question']}\n\n"
        f"*A:* {pair['answer']}\n\n"
        f"Type: {msg_type}{buttons_line}"
    )
    send_text(phone, body)
    send_interactive(
        phone=phone,
        body="Approve this pair?",
        buttons=["Approve", "Refine", "Cancel"],
        header="Training Review",
    )


def _send_chat_prompt(phone: str, reply: str):
    """Send a chat reply with [Approve]/[Cancel] buttons appended."""
    _BODY_LIMIT = BODY_MAX_CHARS - 4  # 4-char safety margin for WhatsApp rendering
    send_interactive(
        phone=phone,
        body=reply[:_BODY_LIMIT] if len(reply) > _BODY_LIMIT else reply,
        buttons=["Approve", "Cancel"],
        header="Refine Chat",
    )


def _send_preview(phone: str, pair: dict):
    """Send the actual WhatsApp message the end-user would receive (preview)."""
    mt = pair.get('message_type', 'text')
    if mt == 'interactive':
        buttons = pair.get('buttons') or []
        if buttons:
            send_interactive(phone, pair.get('answer', ''), buttons, header="Preview")
        else:
            send_text(phone, pair.get('answer', ''))
    elif mt == 'media':
        img = pair.get('image_url', '')
        if img:
            send_media(phone, img, pair.get('answer', ''), 'image')
        else:
            send_text(phone, pair.get('answer', ''))
    elif mt == 'carousel':
        send_text(
            phone,
            "(Carousel preview not supported in admin chat — "
            "saved pair will use the nutoy_stacker template at runtime.)"
        )
    else:
        send_text(phone, pair.get('answer', ''))


# ─── Main dispatcher ──────────────────────────────────────────────────────────

def handle_admin(phone: str, message: str, collection) -> None:
    """Handle an admin message. Sends WhatsApp replies directly; returns nothing."""
    msg = message.strip()
    msg_lower = msg.lower()

    if _is_timed_out(phone):
        session = _sessions.get(phone, {})
        if session.get('current_pair'):
            unlock_pair(_pair_id(session['current_pair']))
        clear_refine_state(phone)
        _clear_session(phone)
        send_text(phone, "Admin session timed out. Send :admin on to re-enter.")
        return

    # ── Explicit entry / exit commands ───────────────────────────────────────
    if msg_lower.startswith(':admin on'):
        disk = load_refine_state(phone)
        if disk and disk.get('state') in _RESUMABLE_STATES:
            last_str = disk.get('last_active', '')
            try:
                last_dt = datetime.fromisoformat(last_str)
                timeout = timedelta(minutes=_load_config().get("admin_timeout_minutes", 15))
                if datetime.utcnow() - last_dt <= timeout:
                    disk['last_active'] = last_dt
                    _sessions[phone] = disk
                    send_text(phone, "Resumed your previous refinement session.")
                    _send_chat_prompt(phone, "I've restored the pair — what would you like to change?")
                    return
            except (ValueError, TypeError):
                pass
            clear_refine_state(phone)

        _set_session(phone, AdminState.MENU)
        _send_admin_menu(phone)
        return

    if msg_lower.startswith(':admin off'):
        session = _sessions.get(phone, {})
        if session.get('current_pair'):
            unlock_pair(_pair_id(session['current_pair']))
        clear_refine_state(phone)
        _clear_session(phone)
        send_text(phone, "Admin mode off.")
        return

    session  = _sessions.get(phone, {})
    state    = session.get("state", AdminState.IDLE)
    cur_pair = session.get("current_pair")

    # ── Menu-level commands (also reachable via :train shortcut) ─────────────
    if msg_lower in (':train', 'qna review'):
        pair = load_next_pending(skip_locked=True)
        if not pair:
            send_text(phone, "No pending Q&A pairs. Run `python training/generate_qna.py` first.")
            _set_session(phone, AdminState.MENU)
            _send_admin_menu(phone)
            return
        _set_session(phone, AdminState.TRAINING_REVIEW, current_pair=pair)
        _send_pair_for_review(phone, pair)
        return

    if msg_lower == 'auto-generate':
        send_text(phone, "Auto-generate is coming soon. Use `python training/generate_qna.py` from the CLI for now.")
        _touch_session(phone)
        return

    if msg_lower == 'test bot':
        send_text(phone, "Send :admin off to chat normally, then :admin on to return to admin mode.")
        _touch_session(phone)
        return

    # ── Training review: waiting for Approve / Refine / Cancel ───────────────
    if state == AdminState.TRAINING_REVIEW:
        if msg_lower == 'approve':
            if cur_pair:
                mark_approved(_pair_id(cur_pair))
                append_approved(cur_pair)
                rag.ingest_qna_pair(cur_pair, collection)
                send_text(phone, "Approved and ingested into ChromaDB.")
            next_pair = load_next_pending(skip_locked=True)
            if next_pair:
                _set_session(phone, AdminState.TRAINING_REVIEW, current_pair=next_pair)
                _send_pair_for_review(phone, next_pair)
            else:
                send_text(phone, "All pairs reviewed! No more pending pairs.")
                _set_session(phone, AdminState.MENU)
                _send_admin_menu(phone)
            return

        if msg_lower == 'refine':
            if not cur_pair:
                _set_session(phone, AdminState.MENU)
                _send_admin_menu(phone)
                return
            identifier = _pair_id(cur_pair)
            locked_by = is_locked(identifier)
            if locked_by and locked_by != phone:
                send_text(phone, "This pair is being reviewed by another admin. Loading the next one.")
                next_pair = load_next_pending(skip_locked=True)
                if next_pair:
                    _set_session(phone, AdminState.TRAINING_REVIEW, current_pair=next_pair)
                    _send_pair_for_review(phone, next_pair)
                else:
                    send_text(phone, "No more unlocked pairs available.")
                    _set_session(phone, AdminState.MENU)
                    _send_admin_menu(phone)
                return
            lock_pair(identifier, phone)
            style_examples = load_style_examples()
            opener = "I've loaded the pair — what would you like to change?"
            _set_session(
                phone, AdminState.TRAINING_CHAT, current_pair=dict(cur_pair),
                original_pair=dict(cur_pair),
                chat_history=[{'role': 'assistant', 'content': opener}],
                style_examples=style_examples,
            )
            save_refine_state(phone, _sessions[phone])
            _send_chat_prompt(phone, opener)
            return

        if msg_lower == 'cancel':
            _set_session(phone, AdminState.MENU)
            send_text(phone, "Training cancelled.")
            _send_admin_menu(phone)
            return

    # ── Training chat: multi-turn refinement ──────────────────────────────────
    if state == AdminState.TRAINING_CHAT:
        if msg_lower == 'cancel':
            original = session.get('original_pair') or cur_pair
            if cur_pair:
                unlock_pair(_pair_id(cur_pair))
            clear_refine_state(phone)
            _set_session(phone, AdminState.TRAINING_REVIEW, current_pair=original)
            send_text(phone, "Refinement cancelled. Showing original pair.")
            if original:
                _send_pair_for_review(phone, original)
            return

        if msg_lower == 'approve':
            violations = validate_pair(cur_pair or {})
            if violations:
                viol_text = "\n".join(f"• {v}" for v in violations)
                send_text(phone, f"Cannot approve — fix these issues first:\n{viol_text}")
                _send_chat_prompt(phone, "Let's fix it before saving.")
                return
            _send_preview(phone, cur_pair)
            send_interactive(
                phone=phone,
                body="Preview above. Save this pair?",
                buttons=["Confirm Save", "Back to refine"],
                header="Approve",
            )
            _sessions[phone]['state'] = AdminState.TRAINING_CONFIRM_SAVE
            save_refine_state(phone, _sessions[phone])
            return

        model, api_key = _get_model_and_key()
        reply, updated_pair = chat_turn(_sessions[phone], msg, model, api_key, phone)
        if updated_pair:
            _sessions[phone]['current_pair'] = updated_pair
        _touch_session(phone)
        _send_chat_prompt(phone, reply)
        return

    # ── Training confirm save ─────────────────────────────────────────────────
    if state == AdminState.TRAINING_CONFIRM_SAVE:
        if msg_lower in ('confirm save', 'confirm', 'yes'):
            model, api_key = _get_model_and_key()
            chat_history = session.get('chat_history', [])
            patterns = extract_patterns(chat_history, model, api_key)
            if patterns:
                _sessions[phone]['patterns_found'] = patterns
                _sessions[phone]['state'] = AdminState.TRAINING_CONFIRM_PREFS
                save_refine_state(phone, _sessions[phone])
                lines = "\n".join(f"{i+1}. {p['text']}" for i, p in enumerate(patterns))
                send_interactive(
                    phone=phone,
                    body=f"Save these style preferences?\n\n{lines}",
                    buttons=["Save All", "Skip"],
                    header="Style Preferences",
                )
            else:
                _do_ingest_and_advance(phone, cur_pair, collection)
            return

        if msg_lower in ('back to refine', 'back', 'no'):
            _sessions[phone]['state'] = AdminState.TRAINING_CHAT
            save_refine_state(phone, _sessions[phone])
            _send_chat_prompt(phone, "OK, continuing the refinement. What would you like to change?")
            return

        send_interactive(
            phone=phone,
            body="Please choose:",
            buttons=["Confirm Save", "Back to refine"],
            header="Approve",
        )
        return

    # ── Training confirm prefs ────────────────────────────────────────────────
    if state == AdminState.TRAINING_CONFIRM_PREFS:
        patterns = session.get('patterns_found', [])
        if msg_lower in ('save all', 'save'):
            accepted = patterns
        elif msg_lower in ('skip', 'none'):
            accepted = []
        else:
            nums_str = re.sub(r'[^0-9,\s]', '', msg_lower)
            nums = [int(x.strip()) for x in nums_str.split(',') if x.strip().isdigit()]
            valid_nums = [n for n in nums if 1 <= n <= len(patterns)]
            if not valid_nums and nums_str.strip():
                send_interactive(
                    phone=phone,
                    body="Reply with [Save All], [Skip], or comma-separated numbers like `1,3`.",
                    buttons=["Save All", "Skip"],
                    header="Style Preferences",
                )
                return
            accepted = [patterns[n - 1] for n in valid_nums]

        if accepted:
            append_admin_prefs(phone, accepted)
            send_text(phone, f"Saved {len(accepted)} style preference(s).")

        _do_ingest_and_advance(phone, cur_pair, collection)
        return

    # ── Fallback ──────────────────────────────────────────────────────────────
    _touch_session(phone)
    send_text(phone, "Unknown command.")
    _send_admin_menu(phone)


def _do_ingest_and_advance(phone: str, pair: dict, collection) -> None:
    """Ingest the approved pair, clear refinement state, advance to next pair."""
    if pair:
        unlock_pair(_pair_id(pair))
        mark_approved(_pair_id(pair))
        append_approved(pair)
        rag.ingest_qna_pair(pair, collection)
    clear_refine_state(phone)
    send_text(phone, "Saved + ingested. Loading next pair...")
    next_pair = load_next_pending(skip_locked=True)
    if next_pair:
        _set_session(phone, AdminState.TRAINING_REVIEW, current_pair=next_pair)
        _send_pair_for_review(phone, next_pair)
    else:
        send_text(phone, "All pairs reviewed!")
        _set_session(phone, AdminState.MENU)
        _send_admin_menu(phone)
