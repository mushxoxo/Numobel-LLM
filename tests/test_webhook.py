import json
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

from app.intent import IntentEnum
from app.validators.response import ValidationResult


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
    os.environ.setdefault("WA2MATION_API_KEY", "test")
    os.environ.setdefault("WA2MATION_VENDOR_UID", "test")
    with patch("app.startup.run_startup"):
        sys.modules.pop("app.webhook", None)
        import app.webhook as webhook_module

        webhook_module._startup._ready = True
        with patch("app.webhook.rag.get_collection", return_value=MagicMock()), \
             patch("app.webhook.plan_response", return_value="text"), \
             patch("app.webhook.validate_whatsapp_response", side_effect=lambda x: x), \
             patch("app.webhook.validate_response", return_value=ValidationResult(valid=True)):
            app = webhook_module.app
            app.config["TESTING"] = True
            with app.test_client() as c:
                yield c


def test_webhook_returns_success(client):
    emb = [0.1] * 1024
    with patch("app.webhook.load_history", return_value=[]), \
         patch("app.webhook.rag.get_embedding", return_value=emb), \
         patch("app.webhook.classify_intent", return_value={"intent": IntentEnum.GENERAL_QNA, "confidence": 0.8, "layer": "embedding"}), \
         patch("app.webhook.rag.get_collection") as mock_col, \
         patch("app.webhook.rag.rewrite_query", return_value="hi"), \
         patch("app.webhook.rag.retrieve", return_value=[]), \
         patch("app.webhook.rag.generate_answer", return_value=RAG_RESULT), \
         patch("app.webhook.dispatch") as mock_dispatch, \
         patch("app.webhook.save_history"):
        mock_col.return_value.query.return_value = {"distances": [[1.0]], "documents": [[]], "metadatas": [[]]}
        resp = client.post("/webhook", json=INCOMING)
    assert resp.status_code == 200
    assert resp.json["status"] == "success"


def test_webhook_calls_dispatch_with_result_and_hits(client):
    emb = [0.1] * 1024
    hits = [{"text": "chunk", "metadata": {"images": "https://example.com/img.jpg"}}]
    with patch("app.webhook.load_history", return_value=[]), \
         patch("app.webhook.rag.get_embedding", return_value=emb), \
         patch("app.webhook.classify_intent", return_value={"intent": IntentEnum.GENERAL_QNA, "confidence": 0.8, "layer": "embedding"}), \
         patch("app.webhook.rag.get_collection") as mock_col, \
         patch("app.webhook.rag.rewrite_query", return_value="hi"), \
         patch("app.webhook.rag.retrieve", return_value=hits), \
         patch("app.webhook.rag.generate_answer", return_value=RAG_RESULT), \
         patch("app.webhook.dispatch") as mock_dispatch, \
         patch("app.webhook.save_history"):
        mock_col.return_value.query.return_value = {"distances": [[1.0]], "documents": [[]], "metadatas": [[]]}
        client.post("/webhook", json=INCOMING)
    mock_dispatch.assert_called_once_with("919999999999", RAG_RESULT, hits)


def test_webhook_saves_history_after_response(client):
    emb = [0.1] * 1024
    with patch("app.webhook.load_history", return_value=[]), \
         patch("app.webhook.rag.get_embedding", return_value=emb), \
         patch("app.webhook.classify_intent", return_value={"intent": IntentEnum.GENERAL_QNA, "confidence": 0.8, "layer": "embedding"}), \
         patch("app.webhook.rag.get_collection") as mock_col, \
         patch("app.webhook.rag.rewrite_query", return_value="hi"), \
         patch("app.webhook.rag.retrieve", return_value=[]), \
         patch("app.webhook.rag.generate_answer", return_value=RAG_RESULT), \
         patch("app.webhook.dispatch"), \
         patch("app.webhook.save_history") as mock_save:
        mock_col.return_value.query.return_value = {"distances": [[1.0]], "documents": [[]], "metadatas": [[]]}
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


def test_webhook_returns_200_on_exception(client):
    """Unhandled exceptions return HTTP 200 to prevent wa2mation retry storms."""
    with patch("app.webhook.load_history", side_effect=Exception("secret /tmp/private.db")):
        resp = client.post("/webhook", json=INCOMING)
    assert resp.status_code == 200
    assert resp.json["status"] == "error"
    assert "secret" not in resp.get_data(as_text=True)
    assert "/tmp/private.db" not in resp.get_data(as_text=True)


def test_webhook_ignores_invalid_phone_format(client):
    with patch("app.webhook.dispatch") as mock_dispatch:
        resp = client.post("/webhook", json={
            "contact": {"phone_number": "not-a-phone"},
            "message": {"body": "hello"},
        })
    assert resp.json["status"] == "ignored"
    mock_dispatch.assert_not_called()


def test_webhook_ignores_non_json_payload(client):
    resp = client.post("/webhook", data="not json", content_type="text/plain")
    assert resp.status_code == 200
    assert resp.is_json
    assert resp.json["status"] == "ignored"


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
    emb = [0.1] * 1024
    with patch("app.webhook.needs_admin_handling", return_value=False), \
         patch("app.webhook.load_history", return_value=[]), \
         patch("app.webhook.rag.get_embedding", return_value=emb), \
         patch("app.webhook.classify_intent", return_value={"intent": IntentEnum.GENERAL_QNA, "confidence": 0.8, "layer": "embedding"}), \
         patch("app.webhook.rag.get_collection") as mock_col, \
         patch("app.webhook.rag.rewrite_query", return_value="hi"), \
         patch("app.webhook.rag.retrieve", return_value=[]), \
         patch("app.webhook.rag.generate_answer", return_value=RAG_RESULT), \
         patch("app.webhook.dispatch") as mock_dispatch, \
         patch("app.webhook.save_history"), \
         patch("app.webhook.handle_admin") as mock_admin:
        mock_col.return_value.query.return_value = {"distances": [[1.0]], "documents": [[]], "metadatas": [[]]}
        client.post("/webhook", json=INCOMING)
    mock_dispatch.assert_called_once()
    mock_admin.assert_not_called()


# ─── /health and readiness gate (INFRA-06, INFRA-08) ─────────────────────

@pytest.fixture
def health_client():
    os.environ.setdefault("WA2MATION_API_KEY", "test")
    os.environ.setdefault("WA2MATION_VENDOR_UID", "test")
    with patch("app.startup.run_startup"):
        sys.modules.pop("app.webhook", None)
        import app.webhook as webhook_module
        import app.startup as startup_module

        webhook_module.app.config["TESTING"] = True
        with webhook_module.app.test_client() as test_client:
            yield {
                "client": test_client,
                "webhook": webhook_module,
                "startup": startup_module,
            }


def test_health_endpoint_returns_200_when_ready(health_client):
    health_client["startup"]._ready = True
    with patch("app.db.get_db") as mock_db, \
         patch("ollama.list", return_value=MagicMock()):
        mock_db.return_value.execute.return_value = MagicMock()
        resp = health_client["client"].get("/health")

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ready"] is True
    assert body["sqlite"] is True
    assert body["ollama"] is True
    assert body["chromadb"] is True
    for forbidden in ("WA2MATION_API_KEY", "VENDOR_UID", "ANTHROPIC_API_KEY", "/numobel.db", "test-key"):
        assert forbidden not in resp.get_data(as_text=True)


def test_health_endpoint_returns_503_when_sqlite_down(health_client):
    health_client["startup"]._ready = True
    with patch("app.db.get_db", side_effect=Exception("db unreachable")), \
         patch("ollama.list", return_value=MagicMock()):
        resp = health_client["client"].get("/health")

    assert resp.status_code == 503
    assert resp.get_json()["sqlite"] is False


def test_health_endpoint_returns_503_when_not_ready(health_client):
    health_client["startup"]._ready = False
    with patch("app.db.get_db") as mock_db, \
         patch("ollama.list", return_value=MagicMock()):
        mock_db.return_value.execute.return_value = MagicMock()
        resp = health_client["client"].get("/health")

    assert resp.status_code == 503
    assert resp.get_json()["ready"] is False


def test_webhook_returns_503_before_ready(health_client):
    health_client["startup"]._ready = False
    resp = health_client["client"].post("/webhook", json=INCOMING)

    assert resp.status_code == 503
    assert resp.get_json()["status"] == "starting"


# ─── Intent-conditional routing tests (02-03) ────────────────────────────────

def test_greeting_bypasses_retrieval(client):
    """GREETING intent: no rag.rewrite_query, retrieve, or generate_answer calls."""
    with patch("app.webhook._startup._ready", True), \
         patch("app.webhook.load_history", return_value=[]), \
         patch("app.webhook.classify_intent", return_value={"intent": IntentEnum.GREETING, "confidence": 1.0, "layer": "rule"}), \
         patch("app.webhook.rag.rewrite_query") as mock_rewrite, \
         patch("app.webhook.rag.retrieve") as mock_retrieve, \
         patch("app.webhook.rag.generate_answer") as mock_gen, \
         patch("app.webhook.dispatch") as mock_dispatch, \
         patch("app.webhook.save_history"):
        resp = client.post("/webhook", json=INCOMING)
        assert resp.status_code == 200
        mock_rewrite.assert_not_called()
        mock_retrieve.assert_not_called()
        mock_gen.assert_not_called()
        mock_dispatch.assert_called_once()
        call_args = mock_dispatch.call_args[0]
        assert call_args[1]["message_type"] == "text"


def test_oos_bypasses_retrieval(client):
    """OUT_OF_SCOPE intent: generate_answer not called."""
    with patch("app.webhook._startup._ready", True), \
         patch("app.webhook.load_history", return_value=[]), \
         patch("app.webhook.classify_intent", return_value={"intent": IntentEnum.OUT_OF_SCOPE, "confidence": 1.0, "layer": "rule"}), \
         patch("app.webhook.rag.generate_answer") as mock_gen, \
         patch("app.webhook.dispatch"), \
         patch("app.webhook.save_history"):
        resp = client.post("/webhook", json=INCOMING)
        assert resp.status_code == 200
        mock_gen.assert_not_called()


def test_general_qna_runs_full_pipeline(client):
    """GENERAL_QNA: rewrite_query, retrieve, and generate_answer all called."""
    emb = [0.1] * 1024
    with patch("app.webhook._startup._ready", True), \
         patch("app.webhook.load_history", return_value=[]), \
         patch("app.webhook.rag.get_embedding", return_value=emb), \
         patch("app.webhook.classify_intent", return_value={"intent": IntentEnum.GENERAL_QNA, "confidence": 0.8, "layer": "embedding"}), \
         patch("app.webhook.rag.get_collection") as mock_col, \
         patch("app.webhook.rag.rewrite_query", return_value="query") as mock_rewrite, \
         patch("app.webhook.rag.retrieve", return_value=[]) as mock_retrieve, \
         patch("app.webhook.rag.generate_answer", return_value=RAG_RESULT) as mock_gen, \
         patch("app.webhook.dispatch"), \
         patch("app.webhook.save_history"):
        # QnA override returns distance > threshold → falls through to full RAG
        mock_col.return_value.query.return_value = {"distances": [[1.0]], "documents": [[]], "metadatas": [[]]}
        resp = client.post("/webhook", json=INCOMING)
        assert resp.status_code == 200
        mock_rewrite.assert_called_once()
        mock_retrieve.assert_called_once()
        mock_gen.assert_called_once()


def test_qna_override_skips_llm(client):
    """QnA override (distance < threshold): generate_answer NOT called."""
    emb = [0.1] * 1024
    stored_meta = {"message_type": "text", "buttons": "[]", "image_url": None}
    with patch("app.webhook._startup._ready", True), \
         patch("app.webhook.load_history", return_value=[]), \
         patch("app.webhook.rag.get_embedding", return_value=emb), \
         patch("app.webhook.classify_intent", return_value={"intent": IntentEnum.GENERAL_QNA, "confidence": 0.8, "layer": "embedding"}), \
         patch("app.webhook.rag.get_collection") as mock_col, \
         patch("app.webhook.rag.generate_answer") as mock_gen, \
         patch("app.webhook.dispatch"), \
         patch("app.webhook.save_history"):
        mock_col.return_value.query.return_value = {
            "distances": [[0.10]],
            "documents": [["Here is info about rubio monocoat"]],
            "metadatas": [[stored_meta]],
        }
        resp = client.post("/webhook", json=INCOMING)
        assert resp.status_code == 200
        mock_gen.assert_not_called()


def test_history_includes_intent_field(client):
    """History user entry must include 'intent' field after a successful request."""
    emb = [0.1] * 1024
    with patch("app.webhook._startup._ready", True), \
         patch("app.webhook.load_history", return_value=[]), \
         patch("app.webhook.rag.get_embedding", return_value=emb), \
         patch("app.webhook.classify_intent", return_value={"intent": IntentEnum.GENERAL_QNA, "confidence": 0.8, "layer": "embedding"}), \
         patch("app.webhook.rag.get_collection") as mock_col, \
         patch("app.webhook.rag.rewrite_query", return_value="query"), \
         patch("app.webhook.rag.retrieve", return_value=[]), \
         patch("app.webhook.rag.generate_answer", return_value=RAG_RESULT), \
         patch("app.webhook.dispatch"), \
         patch("app.webhook.save_history") as mock_save:
        mock_col.return_value.query.return_value = {"distances": [[1.0]], "documents": [[]], "metadatas": [[]]}
        resp = client.post("/webhook", json=INCOMING)
        assert resp.status_code == 200
        saved_history = mock_save.call_args[0][1]
        user_entries = [e for e in saved_history if e["role"] == "user"]
        assert len(user_entries) >= 1
        assert "intent" in user_entries[-1]
        assert user_entries[-1]["intent"] == "general_qna"
