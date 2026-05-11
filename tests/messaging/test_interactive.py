import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def reset_client(monkeypatch):
    import app.messaging.client as client_mod
    monkeypatch.setattr(client_mod, "_session", None)


@pytest.fixture
def mock_post():
    with patch("app.messaging.interactive.wa2mation_post") as m:
        m.return_value = MagicMock(status_code=200)
        yield m


def test_send_interactive_correct_endpoint(mock_post):
    from app.messaging.interactive import send_interactive
    send_interactive("919999999999", "Choose:", ["A", "B"])
    assert mock_post.call_args[0][0] == "send-interactive-message"


def test_send_interactive_button_type(mock_post):
    from app.messaging.interactive import send_interactive
    send_interactive("919999999999", "Choose:", ["A", "B"])
    assert mock_post.call_args[0][1]["interactive_type"] == "button"


def test_send_interactive_buttons_mapped_to_numbered_keys(mock_post):
    from app.messaging.interactive import send_interactive
    send_interactive("919999999999", "Choose:", ["Wood", "Toys", "Acoustic"])
    buttons = mock_post.call_args[0][1]["buttons"]
    assert buttons == {"1": "Wood", "2": "Toys", "3": "Acoustic"}


def test_send_interactive_buttons_capped_at_3(mock_post):
    from app.messaging.interactive import send_interactive
    send_interactive("919999999999", "Choose:", ["A", "B", "C", "D", "E"])
    buttons = mock_post.call_args[0][1]["buttons"]
    assert len(buttons) == 3


def test_send_interactive_header_omitted_when_empty(mock_post):
    from app.messaging.interactive import send_interactive
    send_interactive("919999999999", "Choose:", ["A"])
    payload = mock_post.call_args[0][1]
    assert "header_type" not in payload
    assert "header_text" not in payload


def test_send_interactive_header_included_when_provided(mock_post):
    from app.messaging.interactive import send_interactive
    send_interactive("919999999999", "Choose:", ["A"], header="Welcome")
    payload = mock_post.call_args[0][1]
    assert payload["header_type"] == "text"
    assert payload["header_text"] == "Welcome"


def test_send_interactive_footer_included_when_provided(mock_post):
    from app.messaging.interactive import send_interactive
    send_interactive("919999999999", "Choose:", ["A"], footer="numobel.in")
    assert mock_post.call_args[0][1]["footer_text"] == "numobel.in"


def test_send_interactive_footer_omitted_when_empty(mock_post):
    from app.messaging.interactive import send_interactive
    send_interactive("919999999999", "Choose:", ["A"])
    assert "footer_text" not in mock_post.call_args[0][1]
