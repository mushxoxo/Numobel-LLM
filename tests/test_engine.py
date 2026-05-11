import json
import pytest
from unittest.mock import patch, MagicMock

import app.refinement.engine as engine_module
from app.refinement.engine import chat_turn, extract_patterns, _normalize_refined, _FALLBACK_REPLY


PHONE = '919999999999'

_ORIG_PAIR = {
    'question':     'What is Nutoy Stacker?',
    'answer':       'A wooden stacking toy.',
    'message_type': 'text',
    'buttons':      None,
    'image_url':    None,
}


def _make_state(pair=None):
    p = pair or dict(_ORIG_PAIR)
    return {
        'original_pair':  dict(p),
        'current_pair':   dict(p),
        'chat_history':   [],
        'style_examples': [],
    }


# ─── _normalize_refined ───────────────────────────────────────────────────────

def test_normalize_text_with_buttons_becomes_interactive():
    result = _normalize_refined({'message_type': 'text', 'buttons': ['A', 'B']}, _ORIG_PAIR)
    assert result['message_type'] == 'interactive'


def test_normalize_text_with_image_becomes_media():
    result = _normalize_refined(
        {'message_type': 'text', 'image_url': 'https://x.com/img.jpg'},
        _ORIG_PAIR,
    )
    assert result['message_type'] == 'media'


def test_normalize_interactive_strips_image():
    result = _normalize_refined(
        {'message_type': 'interactive', 'buttons': ['OK'], 'image_url': 'https://x.com/img.jpg'},
        _ORIG_PAIR,
    )
    assert result['image_url'] is None


def test_normalize_media_strips_buttons():
    result = _normalize_refined(
        {'message_type': 'media', 'image_url': 'https://x.com/img.jpg', 'buttons': ['A']},
        _ORIG_PAIR,
    )
    assert result['buttons'] is None


# ─── chat_turn ────────────────────────────────────────────────────────────────

def _mock_llm_response(reply: str, pair: dict | None):
    payload = json.dumps({'reply': reply, 'pair': pair})
    with patch.object(engine_module, '_call_llm', return_value=payload) as mock_llm, \
         patch('app.refinement.engine.save_refine_state'), \
         patch('app.refinement.engine.load_admin_prefs', return_value=[]):
        yield mock_llm


@pytest.fixture
def mock_save():
    with patch('app.refinement.engine.save_refine_state'), \
         patch('app.refinement.engine.load_admin_prefs', return_value=[]):
        yield


def test_chat_turn_valid_json_no_pair_change(mock_save):
    payload = json.dumps({'reply': 'Sounds good!', 'pair': None})
    state = _make_state()
    with patch.object(engine_module, '_call_llm', return_value=payload):
        reply, updated = chat_turn(state, 'looks fine', 'qwen2.5:14b', None, PHONE)
    assert reply == 'Sounds good!'
    assert updated is None
    # chat_history now has the user + assistant message
    assert len(state['chat_history']) == 2


def test_chat_turn_with_pair_update(mock_save):
    new_pair = {**_ORIG_PAIR, 'answer': 'Updated answer.'}
    payload = json.dumps({'reply': 'Updated!', 'pair': new_pair})
    state = _make_state()
    with patch.object(engine_module, '_call_llm', return_value=payload):
        reply, updated = chat_turn(state, 'shorten it', 'qwen2.5:14b', None, PHONE)
    assert updated is not None
    assert state['current_pair']['answer'] == 'Updated answer.'


def test_chat_turn_malformed_json_fallback(mock_save):
    state = _make_state()
    with patch.object(engine_module, '_call_llm', return_value='NOT JSON AT ALL'):
        reply, updated = chat_turn(state, 'whatever', 'qwen2.5:14b', None, PHONE)
    assert reply == _FALLBACK_REPLY
    assert updated is None
    # Pair unchanged
    assert state['current_pair']['answer'] == _ORIG_PAIR['answer']


def test_chat_turn_missing_keys_fallback(mock_save):
    state = _make_state()
    # Missing "pair" key
    with patch.object(engine_module, '_call_llm', return_value=json.dumps({'reply': 'hi'})):
        reply, updated = chat_turn(state, 'whatever', 'qwen2.5:14b', None, PHONE)
    assert reply == _FALLBACK_REPLY


def test_chat_turn_violation_prepended(mock_save):
    """When the LLM emits a pair that violates constraints, warnings are prepended."""
    bad_pair = {
        **_ORIG_PAIR,
        'message_type': 'interactive',
        'buttons': ['Order now (product page link)'],  # 30 chars — too long
    }
    payload = json.dumps({'reply': 'How about this?', 'pair': bad_pair})
    state = _make_state()
    with patch.object(engine_module, '_call_llm', return_value=payload):
        reply, updated = chat_turn(state, 'add a button', 'qwen2.5:14b', None, PHONE)
    assert '⚠️' in reply


def test_chat_turn_persists_state(mock_save):
    payload = json.dumps({'reply': 'OK', 'pair': None})
    state = _make_state()
    with patch.object(engine_module, '_call_llm', return_value=payload) as _, \
         patch('app.refinement.engine.save_refine_state') as mock_save_fn, \
         patch('app.refinement.engine.load_admin_prefs', return_value=[]):
        chat_turn(state, 'test', 'qwen2.5:14b', None, PHONE)
    mock_save_fn.assert_called_once_with(PHONE, state)


# ─── extract_patterns ────────────────────────────────────────────────────────

def test_extract_patterns_too_short():
    history = [
        {'role': 'user', 'content': 'change it'},
        {'role': 'assistant', 'content': 'done'},
    ]
    result = extract_patterns(history, 'qwen2.5:14b', None)
    assert result == []


def test_extract_patterns_valid(mock_save):
    history = [
        {'role': 'user', 'content': 'use ₹ for prices'},
        {'role': 'assistant', 'content': 'OK'},
        {'role': 'user', 'content': 'keep it concise'},
        {'role': 'assistant', 'content': 'Updated'},
    ]
    patterns = [{'text': 'Use ₹ for prices', 'evidence': 'user said so'}]
    with patch.object(engine_module, '_call_llm', return_value=json.dumps(patterns)), \
         patch('app.refinement.engine.save_refine_state'), \
         patch('app.refinement.engine.load_admin_prefs', return_value=[]):
        result = extract_patterns(history, 'qwen2.5:14b', None)
    assert len(result) == 1
    assert result[0]['text'] == 'Use ₹ for prices'


def test_extract_patterns_malformed_returns_empty(mock_save):
    history = [{'role': 'user', 'content': 'x'}] * 4
    with patch.object(engine_module, '_call_llm', return_value='NOT JSON'), \
         patch('app.refinement.engine.save_refine_state'), \
         patch('app.refinement.engine.load_admin_prefs', return_value=[]):
        result = extract_patterns(history, 'qwen2.5:14b', None)
    assert result == []
