import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("WA2MATION_API_KEY", "test-key")
    monkeypatch.setenv("WA2MATION_VENDOR_UID", "test-uid")


@pytest.fixture
def mock_post():
    with patch("app.messaging.text.requests.post") as m:
        m.return_value = MagicMock(status_code=200, json=lambda: {"result": "success"})
        yield m


def test_send_text_posts_to_correct_url(mock_env, mock_post):
    from app.messaging.text import send_text
    send_text("919999999999", "Hello!")
    url = mock_post.call_args[0][0]
    assert "contact/send-message" in url


def test_send_text_payload(mock_env, mock_post):
    from app.messaging.text import send_text
    send_text("919999999999", "Hello!")
    payload = mock_post.call_args[1]["json"]
    assert payload["phone_number"] == "919999999999"
    assert payload["message_body"] == "Hello!"


def test_send_text_auth_header(mock_env, mock_post):
    from app.messaging.text import send_text
    send_text("919999999999", "Hello!")
    headers = mock_post.call_args[1]["headers"]
    assert headers["Authorization"] == "Bearer test-key"
    assert headers["Content-Type"] == "application/json"


def test_send_text_returns_response(mock_env, mock_post):
    from app.messaging.text import send_text
    resp = send_text("919999999999", "Hello!")
    assert resp.status_code == 200
