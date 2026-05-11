"""
Shared refinement engine: chat_turn, extract_patterns, _normalize_refined.
Used by both app/admin.py (WhatsApp) and training/approve_qna_cli.py (CLI).
"""

import json
import logging

import ollama

from app.refinement.constraints import validate_pair
from app.refinement.prompts import build_system_prompt, build_pattern_extraction_prompt
from app.refinement.storage import save_refine_state, load_admin_prefs

log = logging.getLogger('rag_chatbot')

_FALLBACK_REPLY = (
    "⚠️ I didn't understand my own response. Could you rephrase your suggestion?"
)


def _normalize_refined(updated: dict, original: dict) -> dict:
    """Enforce consistency between message_type, buttons, and image_url."""
    mt        = updated.get('message_type', original.get('message_type', 'text'))
    buttons   = updated.get('buttons',   original.get('buttons'))
    image_url = updated.get('image_url', original.get('image_url'))

    if mt == 'text' and buttons:
        mt = 'interactive'
    if mt == 'text' and image_url:
        mt = 'media'
    if mt == 'interactive':
        image_url = None
    if mt in ('media', 'text', 'carousel'):
        buttons = None

    return {
        **original,
        'answer':       updated.get('answer', original['answer']),
        'message_type': mt,
        'buttons':      buttons,
        'image_url':    image_url,
    }


def _call_llm(messages: list[dict], model: str, api_key: str | None) -> str:
    """Call LLM and return raw text response."""
    if model.startswith('claude') and api_key:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        system_content = next(
            (m['content'] for m in messages if m['role'] == 'system'), ''
        )
        user_messages = [m for m in messages if m['role'] != 'system']
        resp = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_content,
            messages=user_messages,
        )
        return resp.content[0].text.strip()
    else:
        response = ollama.chat(model=model, messages=messages, format='json')
        return response['message']['content'].strip()


def chat_turn(
    state: dict,
    admin_msg: str,
    model: str,
    api_key: str | None,
    phone: str,
) -> tuple[str, dict | None]:
    """
    Process one admin message in the refinement chat.

    state keys used/updated: original_pair, current_pair, chat_history, style_examples.
    Modifies state in-place and persists to disk.
    Returns (reply_text, updated_pair_or_none).
    """
    original_pair = state.get('original_pair', {})
    current_pair  = state.get('current_pair', original_pair)
    chat_history  = state.setdefault('chat_history', [])
    style_examples = state.get('style_examples', [])

    chat_history.append({'role': 'user', 'content': admin_msg})

    prefs = load_admin_prefs(phone)
    system_content = build_system_prompt(current_pair, prefs, style_examples)

    llm_messages = [{'role': 'system', 'content': system_content}] + list(chat_history)

    raw = None
    try:
        raw = _call_llm(llm_messages, model, api_key)
        parsed = json.loads(raw)
        reply_text = parsed.get('reply', '')
        pair_update = parsed.get('pair')
        if not isinstance(reply_text, str) or 'pair' not in parsed:
            raise ValueError("missing required keys")
    except Exception:
        log.warning("ENGINE | chat_turn parse failure. Raw: %.120s", raw or '(no response)')
        chat_history.append({'role': 'assistant', 'content': _FALLBACK_REPLY})
        save_refine_state(phone, state)
        return _FALLBACK_REPLY, None

    updated_pair = None
    if pair_update and isinstance(pair_update, dict):
        updated_pair = _normalize_refined(pair_update, current_pair)
        violations = validate_pair(updated_pair)
        if violations:
            warning_lines = "\n".join(f"⚠️ {v}" for v in violations)
            reply_text = f"{warning_lines}\n\n{reply_text}"
        state['current_pair'] = updated_pair

    chat_history.append({'role': 'assistant', 'content': reply_text})
    save_refine_state(phone, state)

    return reply_text, updated_pair


def extract_patterns(
    chat_history: list[dict],
    model: str,
    api_key: str | None,
) -> list[dict]:
    """
    Extract generalizable admin style preferences from chat history.
    Returns list of {text, evidence} dicts. Skips if < 4 messages.
    """
    if len(chat_history) < 4:
        return []

    prompt = build_pattern_extraction_prompt(chat_history)
    raw = None
    try:
        raw = _call_llm([{'role': 'user', 'content': prompt}], model, api_key)
        patterns = json.loads(raw)
        if isinstance(patterns, list):
            return [p for p in patterns if isinstance(p, dict) and 'text' in p]
    except Exception as e:
        log.warning("ENGINE | extract_patterns failed: %s (raw=%.80s)", e, raw or '')
    return []
