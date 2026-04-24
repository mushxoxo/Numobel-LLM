import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("WA2MATION_API_KEY", "test-key")
    monkeypatch.setenv("WA2MATION_VENDOR_UID", "test-uid")


@pytest.fixture
def mock_post():
    with patch("app.messaging.interactive.requests.post") as m:
        m.return_value = MagicMock(status_code=200)
        yield m


def test_send_interactive_correct_url(mock_env, mock_post):
    from app.messaging.interactive import send_interactive
    send_interactive("919999999999", "Choose:", ["A", "B"])
    assert "contact/send-interactive-message" in mock_post.call_args[0][0]


def test_send_interactive_button_type(mock_env, mock_post):
    from app.messaging.interactive import send_interactive
    send_interactive("919999999999", "Choose:", ["A", "B"])
    assert mock_post.call_args[1]["json"]["interactive_type"] == "button"


def test_send_interactive_buttons_mapped_to_numbered_keys(mock_env, mock_post):
    from app.messaging.interactive import send_interactive
    send_interactive("919999999999", "Choose:", ["Wood", "Toys", "Acoustic"])
    buttons = mock_post.call_args[1]["json"]["buttons"]
    assert buttons == {"1": "Wood", "2": "Toys", "3": "Acoustic"}


def test_send_interactive_buttons_capped_at_3(mock_env, mock_post):
    from app.messaging.interactive import send_interactive
    send_interactive("919999999999", "Choose:", ["A", "B", "C", "D", "E"])
    buttons = mock_post.call_args[1]["json"]["buttons"]
    assert len(buttons) == 3


def test_send_interactive_header_omitted_when_empty(mock_env, mock_post):
    from app.messaging.interactive import send_interactive
    send_interactive("919999999999", "Choose:", ["A"])
    payload = mock_post.call_args[1]["json"]
    assert "header_type" not in payload
    assert "header_text" not in payload


def test_send_interactive_header_included_when_provided(mock_env, mock_post):
    from app.messaging.interactive import send_interactive
    send_interactive("919999999999", "Choose:", ["A"], header="Welcome")
    payload = mock_post.call_args[1]["json"]
    assert payload["header_type"] == "text"
    assert payload["header_text"] == "Welcome"


def test_send_interactive_footer_included_when_provided(mock_env, mock_post):
    from app.messaging.interactive import send_interactive
    send_interactive("919999999999", "Choose:", ["A"], footer="numobel.in")
    assert mock_post.call_args[1]["json"]["footer_text"] == "numobel.in"


def test_send_interactive_footer_omitted_when_empty(mock_env, mock_post):
    from app.messaging.interactive import send_interactive
    send_interactive("919999999999", "Choose:", ["A"])
    assert "footer_text" not in mock_post.call_args[1]["json"]
