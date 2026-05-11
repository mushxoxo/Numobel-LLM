import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def reset_client(monkeypatch):
    import app.messaging.client as client_mod
    monkeypatch.setattr(client_mod, "_session", None)


@pytest.fixture
def mock_post():
    with patch("app.messaging.text.wa2mation_post") as m:
        m.return_value = MagicMock(status_code=200)
        yield m


def test_send_text_posts_to_correct_endpoint(mock_post):
    from app.messaging.text import send_text
    send_text("919999999999", "Hello!")
    assert mock_post.call_args[0][0] == "send-message"


def test_send_text_payload(mock_post):
    from app.messaging.text import send_text
    send_text("919999999999", "Hello!")
    payload = mock_post.call_args[0][1]
    assert payload["phone_number"] == "919999999999"
    assert payload["message_body"] == "Hello!"


def test_send_text_returns_response(mock_post):
    from app.messaging.text import send_text
    resp = send_text("919999999999", "Hello!")
    assert resp.status_code == 200
