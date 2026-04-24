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
    from rag_chatbot import _keyword_fallback
    assert _keyword_fallback("hi there")['message_type'] == 'interactive'

def test_fallback_hello():
    from rag_chatbot import _keyword_fallback
    assert _keyword_fallback("hello")['message_type'] == 'interactive'

def test_fallback_show_me():
    from rag_chatbot import _keyword_fallback
    assert _keyword_fallback("show me what you have")['message_type'] == 'interactive'

def test_fallback_carousel_stacker():
    from rag_chatbot import _keyword_fallback
    assert _keyword_fallback("tell me about stackers")['message_type'] == 'carousel'

def test_fallback_media():
    from rag_chatbot import _keyword_fallback
    assert _keyword_fallback("can I see a photo")['message_type'] == 'media'

def test_fallback_text_default():
    from rag_chatbot import _keyword_fallback
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
    with patch('rag_chatbot.ollama.chat', return_value=_mock_ollama_response(payload)):
        from rag_chatbot import generate_answer
        result = generate_answer("what toys do you have", dummy_chunks)
    assert result['message_type'] == 'interactive'
    assert result['content'] == 'We have stackers!'
    assert result['buttons'] == ['Stackers', 'On Wheels']
    assert result['image_url'] is None


def test_generate_answer_strips_markdown_fences(dummy_chunks):
    payload = '```json\n' + json.dumps({
        'message_type': 'text', 'content': 'Sure!', 'buttons': None, 'image_url': None
    }) + '\n```'
    with patch('rag_chatbot.ollama.chat', return_value=_mock_ollama_response(payload)):
        from rag_chatbot import generate_answer
        result = generate_answer("tell me about rubio", dummy_chunks)
    assert result['message_type'] == 'text'
    assert result['content'] == 'Sure!'


def test_generate_answer_fallback_on_invalid_json(dummy_chunks):
    with patch('rag_chatbot.ollama.chat', return_value=_mock_ollama_response("This is plain text")):
        from rag_chatbot import generate_answer
        result = generate_answer("hi", dummy_chunks)
    assert result['message_type'] == 'interactive'
    assert result['content'] == 'This is plain text'


def test_generate_answer_includes_tokens(dummy_chunks):
    payload = json.dumps({'message_type': 'text', 'content': 'ok', 'buttons': None, 'image_url': None})
    with patch('rag_chatbot.ollama.chat', return_value=_mock_ollama_response(payload)):
        from rag_chatbot import generate_answer
        result = generate_answer("test", dummy_chunks)
    assert result['prompt_tokens'] == 10
    assert result['completion_tokens'] == 20


# ─── ingest_qna_pair ─────────────────────────────────────────────────────────

def test_ingest_qna_pair_upserts_with_correct_metadata():
    mock_collection = MagicMock()
    with patch('rag_chatbot.get_embedding', return_value=[0.1] * 10):
        from rag_chatbot import ingest_qna_pair
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
    with patch('rag_chatbot.get_embedding', return_value=[0.1] * 10):
        from rag_chatbot import ingest_qna_pair
        ingest_qna_pair({'question': 'Q?', 'answer': 'A.'}, mock_collection)
    doc = mock_collection.upsert.call_args[1]['documents'][0]
    assert 'Q?' in doc
    assert 'A.' in doc
