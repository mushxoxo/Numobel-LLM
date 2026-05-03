import json
import pytest
from unittest.mock import patch, MagicMock


INCOMING = {
    "contact": {"phone_number": "919999999999"},
    "message": {"body": "hi there"},
}

RAG_RESULT = {
    "message_type": "interactive",
    "content": "Welcome to Numobel!",
    "buttons": ["Wood Finishes", "Wooden Toys", "Acoustic Panels"],
    "image_url": None,
    "prompt_tokens": 10,
    "completion_tokens": 20,
}


@pytest.fixture
def client():
    with patch("app.webhook.rag.get_collection"), \
         patch("app.webhook.collection") as mock_col:
        mock_col.count.return_value = 1  # skip ingest
        from app.webhook import app
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c


def test_webhook_returns_success(client):
    with patch("app.webhook.load_history", return_value=[]), \
         patch("app.webhook.rag.rewrite_query", return_value="hi"), \
         patch("app.webhook.rag.retrieve", return_value=[]), \
         patch("app.webhook.rag.generate_answer", return_value=RAG_RESULT), \
         patch("app.webhook.dispatch") as mock_dispatch, \
         patch("app.webhook.save_history"):
        resp = client.post("/webhook", json=INCOMING)
    assert resp.status_code == 200
    assert resp.json["status"] == "success"


def test_webhook_calls_dispatch_with_result_and_hits(client):
    hits = [{"text": "chunk", "metadata": {"images": "https://example.com/img.jpg"}}]
    with patch("app.webhook.load_history", return_value=[]), \
         patch("app.webhook.rag.rewrite_query", return_value="hi"), \
         patch("app.webhook.rag.retrieve", return_value=hits), \
         patch("app.webhook.rag.generate_answer", return_value=RAG_RESULT), \
         patch("app.webhook.dispatch") as mock_dispatch, \
         patch("app.webhook.save_history"):
        client.post("/webhook", json=INCOMING)
    mock_dispatch.assert_called_once_with("919999999999", RAG_RESULT, hits)


def test_webhook_saves_history_after_response(client):
    with patch("app.webhook.load_history", return_value=[]), \
         patch("app.webhook.rag.rewrite_query", return_value="hi"), \
         patch("app.webhook.rag.retrieve", return_value=[]), \
         patch("app.webhook.rag.generate_answer", return_value=RAG_RESULT), \
         patch("app.webhook.dispatch"), \
         patch("app.webhook.save_history") as mock_save:
        client.post("/webhook", json=INCOMING)
    mock_save.assert_called_once()
    phone, history = mock_save.call_args[0]
    assert phone == "919999999999"
    assert history[-2]["role"] == "user"
    assert history[-1]["role"] == "assistant"


def test_webhook_ignores_empty_body(client):
    with patch("app.webhook.dispatch") as mock_dispatch:
        resp = client.post("/webhook", json={
            "contact": {"phone_number": "919999999999"},
            "message": {"body": ""},
        })
    assert resp.json["status"] == "ignored"
    mock_dispatch.assert_not_called()


def test_webhook_ignores_null_body_delivery_receipt(client):
    """wa2mation sends delivery/read receipts with body=null — must not 500."""
    with patch("app.webhook.dispatch") as mock_dispatch:
        resp = client.post("/webhook", json={
            "contact": {"phone_number": "919999999999"},
            "message": {"body": None, "is_new_message": False, "status": "delivered"},
        })
    assert resp.json["status"] == "ignored"
    mock_dispatch.assert_not_called()


def test_webhook_ignores_missing_phone(client):
    with patch("app.webhook.dispatch") as mock_dispatch:
        resp = client.post("/webhook", json={
            "contact": {},
            "message": {"body": "hello"},
        })
    assert resp.json["status"] == "ignored"
    mock_dispatch.assert_not_called()


def test_webhook_returns_500_on_exception(client):
    with patch("app.webhook.load_history", side_effect=Exception("DB error")):
        resp = client.post("/webhook", json=INCOMING)
    assert resp.status_code == 500
    assert resp.json["status"] == "error"


def test_webhook_routes_admin_command_to_admin_handler(client):
    with patch("app.webhook.needs_admin_handling", return_value=True), \
         patch("app.webhook.handle_admin") as mock_admin, \
         patch("app.webhook.dispatch") as mock_dispatch:
        resp = client.post("/webhook", json={
            "contact": {"phone_number": "919999999999"},
            "message": {"body": ":admin on"},
        })
    assert resp.json["status"] == "success"
    mock_admin.assert_called_once()
    mock_dispatch.assert_not_called()  # RAG pipeline bypassed


def test_webhook_skips_admin_for_regular_users(client):
    with patch("app.webhook.needs_admin_handling", return_value=False), \
         patch("app.webhook.load_history", return_value=[]), \
         patch("app.webhook.rag.rewrite_query", return_value="hi"), \
         patch("app.webhook.rag.retrieve", return_value=[]), \
         patch("app.webhook.rag.generate_answer", return_value=RAG_RESULT), \
         patch("app.webhook.dispatch") as mock_dispatch, \
         patch("app.webhook.save_history"), \
         patch("app.webhook.handle_admin") as mock_admin:
        client.post("/webhook", json=INCOMING)
    mock_dispatch.assert_called_once()
    mock_admin.assert_not_called()
