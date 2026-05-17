import json
import pytest
from unittest.mock import patch, MagicMock


def _mock_ollama_response(content: str):
    return {
        'message': {'content': content},
        'prompt_eval_count': 10,
        'eval_count': 20,
    }


# ─── _keyword_fallback ────────────────────────────────────────────────────────

def test_fallback_greeting():
    from app.rag import _keyword_fallback
    assert _keyword_fallback("hi there")['message_type'] == 'interactive'

def test_fallback_hello():
    from app.rag import _keyword_fallback
    assert _keyword_fallback("hello")['message_type'] == 'interactive'

def test_fallback_show_me():
    from app.rag import _keyword_fallback
    assert _keyword_fallback("show me what you have")['message_type'] == 'interactive'

def test_fallback_carousel_stacker():
    from app.rag import _keyword_fallback
    assert _keyword_fallback("tell me about stackers")['message_type'] == 'carousel'

def test_fallback_media():
    from app.rag import _keyword_fallback
    assert _keyword_fallback("can I see a photo")['message_type'] == 'media'

def test_fallback_text_default():
    from app.rag import _keyword_fallback
    assert _keyword_fallback("what is the price of the wood finish")['message_type'] == 'text'


# ─── generate_answer — valid JSON ────────────────────────────────────────────

@pytest.fixture
def dummy_chunks():
    return [{'text': 'Nutoy Stacker Mountain costs ₹1499', 'metadata': {'product_name': 'Nutoy-Stacker-Mountain', 'brand': 'Nutoy'}}]


def test_generate_answer_parses_valid_json(dummy_chunks):
    payload = json.dumps({
        'message_type': 'interactive',
        'content': 'We have stackers!',
        'buttons': ['Stackers', 'On Wheels'],
        'image_url': None,
    })
    with patch('app.rag.ollama.chat', return_value=_mock_ollama_response(payload)):
        from app.rag import generate_answer
        result = generate_answer("what toys do you have", dummy_chunks)
    assert result['message_type'] == 'interactive'
    assert result['content'] == 'We have stackers!'
    assert result['buttons'] == ['Stackers', 'On Wheels']
    assert result['image_url'] is None


def test_generate_answer_strips_markdown_fences(dummy_chunks):
    payload = '```json\n' + json.dumps({
        'message_type': 'text', 'content': 'Sure!', 'buttons': None, 'image_url': None
    }) + '\n```'
    with patch('app.rag.ollama.chat', return_value=_mock_ollama_response(payload)):
        from app.rag import generate_answer
        result = generate_answer("tell me about rubio", dummy_chunks)
    assert result['message_type'] == 'text'
    assert result['content'] == 'Sure!'


def test_generate_answer_fallback_on_invalid_json(dummy_chunks):
    with patch('app.rag.ollama.chat', return_value=_mock_ollama_response("This is plain text")):
        from app.rag import generate_answer
        result = generate_answer("hi", dummy_chunks)
    assert result['message_type'] == 'interactive'
    # Raw LLM garbage is no longer sent to users — static fallback message returned instead
    assert "try again" in result['content'].lower()


def test_generate_answer_fallback_extracts_partial_content(dummy_chunks):
    partial = '{"message_type": "text", "content": "Great product!", "buttons":'
    with patch('app.rag.ollama.chat', return_value=_mock_ollama_response(partial)):
        from app.rag import generate_answer
        result = generate_answer("tell me about rubio", dummy_chunks)
    assert result['content'] == 'Great product!'


def test_generate_answer_includes_tokens(dummy_chunks):
    payload = json.dumps({'message_type': 'text', 'content': 'ok', 'buttons': None, 'image_url': None})
    with patch('app.rag.ollama.chat', return_value=_mock_ollama_response(payload)):
        from app.rag import generate_answer
        result = generate_answer("test", dummy_chunks)
    assert result['prompt_tokens'] == 10
    assert result['completion_tokens'] == 20


# ─── ingest_qna_pair ─────────────────────────────────────────────────────────

def test_ingest_qna_pair_upserts_with_correct_metadata():
    mock_collection = MagicMock()
    with patch('app.rag.get_embedding', return_value=[0.1] * 10):
        from app.rag import ingest_qna_pair
        pair = {
            'question': 'What is a stacker?',
            'answer': 'A wooden stacking toy.',
            'message_type': 'carousel',
            'buttons': ['Stackers'],
            'product': 'Nutoy-Stacker-Mountain',
        }
        doc_id = ingest_qna_pair(pair, mock_collection)

    mock_collection.upsert.assert_called_once()
    call_kwargs = mock_collection.upsert.call_args[1]
    assert call_kwargs['metadatas'][0]['source'] == 'approved_training'
    assert call_kwargs['metadatas'][0]['message_type'] == 'carousel'
    assert doc_id.startswith('chunk_')


def test_ingest_qna_pair_document_contains_qa():
    mock_collection = MagicMock()
    with patch('app.rag.get_embedding', return_value=[0.1] * 10):
        from app.rag import ingest_qna_pair
        ingest_qna_pair({'question': 'Q?', 'answer': 'A.'}, mock_collection)
    doc = mock_collection.upsert.call_args[1]['documents'][0]
    assert 'Q?' in doc
    assert 'A.' in doc


# ─── build_system_prompt + generate_answer (Phase 3 stubs) ───────────────────

import app.rag as rag


def test_build_system_prompt():
    fn = getattr(rag, "build_system_prompt", None)
    if fn is None:
        pytest.skip("build_system_prompt not yet implemented")
    result = fn(["Nuacoustics", "Nupanel", "Nutoy", "Nuwork", "Rubio Monocoat"])
    assert "Nuacoustics" in result
    assert "Nupanel" in result
    assert "Nutoy" in result
    assert "Nuwork" in result
    assert "Rubio Monocoat" in result
    assert "Never mention any other brand name" in result
    assert "Authorized brands" in result


def test_build_system_prompt_uncertainty():
    fn = getattr(rag, "build_system_prompt", None)
    if fn is None:
        pytest.skip("build_system_prompt not yet implemented")
    result = fn(["Nuacoustics", "Nupanel", "Nutoy", "Nuwork", "Rubio Monocoat"])
    assert "I don't have information about that" in result


def test_generate_answer_respects_message_type(mocker):
    mocker.patch(
        "app.rag.ollama.chat",
        return_value={"message": {"content": '{"content":"hello","buttons":null,"image_url":null}'}},
    )
    result = rag.generate_answer(query="x", context_chunks=[], history=None, message_type="media")
    assert result["message_type"] == "media"


def test_product_name_in_context(mocker):
    captured = {}

    def fake_chat(**kwargs):
        captured["messages"] = kwargs.get("messages", [])
        return {"message": {"content": '{"content":"test","buttons":null,"image_url":null}'}}

    mocker.patch("app.rag.ollama.chat", side_effect=fake_chat)
    rag.generate_answer(
        query="tell me about rainbow stacker",
        context_chunks=[{"text": "Nutoy Rainbow Stacker is a wooden toy", "metadata": {"name": "Rainbow Stacker", "brand": "Nutoy"}}],
        history=None,
    )
    all_content = " ".join(m.get("content", "") for m in captured.get("messages", []))
    assert "Rainbow Stacker" in all_content
    system_msgs = [m.get("content", "") for m in captured.get("messages", []) if m.get("role") == "system"]
    assert any("Use the exact product name from the context verbatim" in s for s in system_msgs)
