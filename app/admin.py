import json
import logging
import os
from datetime import datetime, timedelta

import ollama

import rag_chatbot as rag
from app.messaging import send_text, send_interactive

log = logging.getLogger('rag_chatbot')

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH = os.path.join(_PROJECT_ROOT, 'config.json')
_PENDING_PATH = os.path.join(_PROJECT_ROOT, 'training', 'qna_pairs', 'pending.jsonl')
_APPROVED_PATH = os.path.join(_PROJECT_ROOT, 'training', 'qna_pairs', 'approved.jsonl')

# In-memory admin sessions: {phone: {state, current_pair, last_active}}
# Lost on server restart — admin resends :admin on to re-enter.
_sessions: dict = {}

_IDLE             = "idle"
_MENU             = "menu"
_TRAINING_REVIEW  = "training_review"
_TRAINING_SUGGEST = "training_suggest"


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
    return _sessions.get(phone, {}).get("state", _IDLE) != _IDLE


# ─── Session helpers ──────────────────────────────────────────────────────────

def _set_session(phone: str, state: str, current_pair=None):
    _sessions[phone] = {
        "state": state,
        "current_pair": current_pair,
        "last_active": datetime.utcnow(),
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
    return datetime.utcnow() - session["last_active"] > timeout


# ─── JSONL helpers ────────────────────────────────────────────────────────────

def _load_next_pending() -> dict | None:
    """Return the first pair in pending.jsonl that hasn't been approved yet."""
    if not os.path.exists(_PENDING_PATH):
        return None
    with open(_PENDING_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                pair = json.loads(line)
                if not pair.get("approved"):
                    return pair
            except json.JSONDecodeError:
                continue
    return None


def _mark_approved_in_pending(pair: dict):
    """Flip approved=True for the matching pair in pending.jsonl (matched by question)."""
    if not os.path.exists(_PENDING_PATH):
        return
    with open(_PENDING_PATH) as f:
        lines = [l.rstrip('\n') for l in f]
    updated = []
    for line in lines:
        if not line.strip():
            updated.append(line)
            continue
        try:
            p = json.loads(line)
            if p.get("question") == pair.get("question"):
                p["approved"] = True
                line = json.dumps(p)
        except json.JSONDecodeError:
            pass
        updated.append(line)
    with open(_PENDING_PATH, 'w') as f:
        f.write('\n'.join(updated))
        if updated:
            f.write('\n')


def _append_approved(pair: dict):
    os.makedirs(os.path.dirname(_APPROVED_PATH), exist_ok=True)
    with open(_APPROVED_PATH, 'a') as f:
        f.write(json.dumps(pair) + '\n')


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
    """Send Q&A pair as text + interactive [Approve]/[Suggest] buttons."""
    msg_type = pair.get("message_type", "text")
    btns = pair.get("buttons")
    buttons_line = f"\nButtons: {', '.join(btns)}" if btns else ""
    body = (
        f"*Q:* {pair['question']}\n\n"
        f"*A:* {pair['answer']}\n\n"
        f"Type: {msg_type}{buttons_line}"
    )
    send_text(phone, body)
    # wa2mation button taps arrive as message.body = button label text.
    # If Approve/Suggest aren't recognised, check whatsapp_webhook_payload for button_id.
    send_interactive(
        phone=phone,
        body="Approve this pair?",
        buttons=["Approve", "Suggest"],
        header="Training Review",
    )


# ─── LLM refinement ──────────────────────────────────────────────────────────

def _refine_with_llm(pair: dict, suggestion: str) -> dict:
    prompt = (
        "Refine this WhatsApp chatbot Q&A pair based on the admin's suggestion.\n"
        f"Question: {pair['question']}\n"
        f"Current answer: {pair['answer']}\n"
        f"Admin suggestion: {suggestion}\n\n"
        "Reply with ONLY the improved answer text, nothing else."
    )
    response = ollama.chat(
        model='llama3.2',
        messages=[{"role": "user", "content": prompt}],
    )
    return {**pair, "answer": response['message']['content'].strip()}


# ─── Main dispatcher ──────────────────────────────────────────────────────────

def handle_admin(phone: str, message: str, collection) -> None:
    """Handle an admin message. Sends WhatsApp replies directly; returns nothing."""
    msg = message.strip()
    msg_lower = msg.lower()

    # Auto-timeout: clear stale session and inform admin
    if _is_timed_out(phone):
        _clear_session(phone)
        send_text(phone, "Admin session timed out. Send :admin on to re-enter.")
        return

    # ── Explicit entry / exit commands ───────────────────────────────────────
    if msg_lower.startswith(':admin on'):
        _set_session(phone, _MENU)
        _send_admin_menu(phone)
        return

    if msg_lower.startswith(':admin off'):
        _clear_session(phone)
        send_text(phone, "Admin mode off.")
        return

    session  = _sessions.get(phone, {})
    state    = session.get("state", _IDLE)
    cur_pair = session.get("current_pair")

    # ── Menu-level commands (also reachable via :train shortcut) ─────────────
    if msg_lower in (':train', 'qna review'):
        pair = _load_next_pending()
        if not pair:
            send_text(phone, "No pending Q&A pairs. Run `python training/generate_qna.py` first.")
            _set_session(phone, _MENU)
            _send_admin_menu(phone)
            return
        _set_session(phone, _TRAINING_REVIEW, current_pair=pair)
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

    # ── Training review: waiting for Approve or Suggest ──────────────────────
    if state == _TRAINING_REVIEW:
        if msg_lower == 'approve':
            if cur_pair:
                _mark_approved_in_pending(cur_pair)
                _append_approved(cur_pair)
                rag.ingest_qna_pair(cur_pair, collection)
                send_text(phone, "Approved and ingested into ChromaDB.")
            next_pair = _load_next_pending()
            if next_pair:
                _set_session(phone, _TRAINING_REVIEW, current_pair=next_pair)
                _send_pair_for_review(phone, next_pair)
            else:
                send_text(phone, "All pairs reviewed! No more pending pairs.")
                _set_session(phone, _MENU)
                _send_admin_menu(phone)
            return

        if msg_lower == 'suggest':
            _set_session(phone, _TRAINING_SUGGEST, current_pair=cur_pair)
            send_text(phone, "Type your suggestion for improving this pair:")
            return

    # ── Training suggest: waiting for free-text feedback ─────────────────────
    if state == _TRAINING_SUGGEST:
        if cur_pair:
            try:
                refined = _refine_with_llm(cur_pair, msg)
            except Exception as e:
                log.error("ADMIN | LLM refinement failed: %s", e)
                send_text(phone, "Refinement failed. Showing original pair.")
                refined = cur_pair
            _set_session(phone, _TRAINING_REVIEW, current_pair=refined)
            _send_pair_for_review(phone, refined)
        else:
            _set_session(phone, _MENU)
            _send_admin_menu(phone)
        return

    # ── Fallback: unknown input in active session ─────────────────────────────
    _touch_session(phone)
    send_text(phone, "Unknown command.")
    _send_admin_menu(phone)
