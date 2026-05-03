import json
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Make training/ importable without requiring __init__.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
import generate_qna as gqna


# ── helpers ─────────────────────────────────────────────────────────────────

def _make_pair(question="Q?", answer="A.", message_type="text", product="Prod"):
    return {
        "question": question, "answer": answer,
        "message_type": message_type, "buttons": None,
        "image_url": None, "product": product, "approved": False,
    }

def _make_product(name="Test Product"):
    return {
        "name": name, "brand": "TestBrand", "product_line": None,
        "description": "A product.", "price": {"original": 999.0, "discounted": 999.0},
        "attributes": {"colors": [], "size": ["1L"], "specifications": "Type: test"},
        "media": {"images": ["https://example.com/img.jpg"]},
        "metadata": {"product_link": "https://www.numobel.in/product-page/test"},
    }


# ── test 1: already_done_products returns correct set ───────────────────────

def test_already_done_products_returns_correct_set(tmp_path, monkeypatch):
    pending = tmp_path / 'pending.jsonl'
    pending.write_text(
        json.dumps({"product": "Product A", "approved": False}) + '\n' +
        json.dumps({"product": "Product B", "approved": True}) + '\n'
    )
    monkeypatch.setattr(gqna, '_PENDING_PATH', pending)
    result = gqna.already_done_products()
    assert result == {'Product A', 'Product B'}


# ── test 2: already_done_products returns empty set when file missing ────────

def test_already_done_products_empty_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(gqna, '_PENDING_PATH', tmp_path / 'nonexistent.jsonl')
    assert gqna.already_done_products() == set()


# ── test 3: select_few_shot_examples returns diverse examples ───────────────

def test_select_few_shot_examples_returns_diverse():
    approved = [
        _make_pair(message_type="text", question="Q1"),
        _make_pair(message_type="interactive", question="Q2"),
        _make_pair(message_type="media", question="Q3"),
        _make_pair(message_type="text", question="Q4"),
    ]
    result = gqna.select_few_shot_examples(approved, n=3)
    assert len(result) == 3
    types = [p['message_type'] for p in result]
    assert 'text' in types
    assert 'interactive' in types
    assert 'media' in types


# ── test 4: select_few_shot_examples returns [] on cold start ───────────────

def test_select_few_shot_examples_cold_start():
    assert gqna.select_few_shot_examples([], n=3) == []


# ── test 5: parse_pairs strips fences and injects product fields ─────────────

def test_parse_pairs_strips_fences_and_injects_fields():
    product = _make_product("My Product")
    raw = '```json\n[{"question":"Q?","answer":"A.","message_type":"text","buttons":null,"image_url":null}]\n```'
    result = gqna.parse_pairs(raw, product)
    assert len(result) == 1
    assert result[0]['product'] == 'My Product'
    assert result[0]['approved'] is False
    assert result[0]['question'] == 'Q?'


# ── test 6: parse_pairs returns [] on malformed JSON ────────────────────────

def test_parse_pairs_returns_empty_on_bad_json():
    product = _make_product()
    result = gqna.parse_pairs("not json at all }{", product)
    assert result == []


# ── test 7: parse_pairs drops pairs with invalid message_type ───────────────

def test_parse_pairs_drops_invalid_message_type():
    product = _make_product()
    raw = json.dumps([
        {"question": "Q1?", "answer": "A1.", "message_type": "notification"},
        {"question": "Q2?", "answer": "A2.", "message_type": "text"},
    ])
    result = gqna.parse_pairs(raw, product)
    assert len(result) == 1
    assert result[0]['question'] == 'Q2?'


# ── test 8: call_llm routes to ollama.chat for llama3.2 ─────────────────────

def test_call_llm_routes_to_ollama():
    mock_response = {'message': {'content': '[{"question":"Q?","answer":"A.","message_type":"text"}]'}}
    with patch.object(gqna.ollama, 'chat', return_value=mock_response) as mock_chat:
        result = gqna.call_llm("prompt", 'llama3.2', None)
    mock_chat.assert_called_once()
    call_args = mock_chat.call_args
    assert call_args[1]['model'] == 'llama3.2' or call_args[0][0] == 'llama3.2' or call_args[1].get('model') == 'llama3.2'
    assert '[' in result


# ── test 9: call_llm routes to anthropic for claude ─────────────────────────

def test_call_llm_routes_to_claude():
    mock_text = '[{"question":"Q?","answer":"A.","message_type":"text"}]'
    mock_content = MagicMock()
    mock_content.text = mock_text
    mock_message = MagicMock()
    mock_message.content = [mock_content]

    with patch.dict('sys.modules', {'anthropic': MagicMock()}):
        import anthropic as mock_anthropic
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message
        mock_anthropic.Anthropic.return_value = mock_client

        result = gqna.call_llm("prompt", 'claude-sonnet-4-6', 'fake-key')

    assert result == mock_text
